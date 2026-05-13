# Session 5 plan — close the gap with llama.cpp on a3b prefill

Entry-point: [`qwen_batched_prefill_session4_landed.md`].

**End-state goal:** prefill rate on Qwen3.6-A3B at or above llama.cpp.
Session 4 hit 4× on the bench.py essay+512 workload (post-Phase-G-revert).
Session 5 closes most of the remaining gap by:

1. Zero-copy expert blobs via `mmap` + `newBufferWithBytesNoCopy`
   (eliminates both the pread copy AND the host→GPU memcpy that
   `MtlBuffer::with_data` does today).
2. Real Phase G: add the prefetch state machine to
   `step_internal_batched_gqa` so the batched path can route
   `eval_token` through itself without losing the 0.396 decode hit
   rate.
3. Per-chunk scratch buffer hoisting (allocator churn cleanup).
4. Dead-code cleanup of the old per-token GPU SDPA fast path.

Order matters: (1) is the headline win; (2) unblocks the eval_token
routing change; (3) is a tactical perf cleanup once (1)+(2) land;
(4) is the closing dead-code commit.

## Phase 1 — Zero-copy expert blobs (the headline)

**Today:** `crates/moeflux/src/riir/expert_io.rs:180`'s `read_expert`
does a synchronous `file.read_at(out, off)` into a caller-provided
buffer. Then `batched_full_attn_layer_forward` wraps the host buffer
in a Metal buffer via `MtlBuffer::with_data` — another full memcpy
(host slice → newly-allocated Metal-shared buffer).

So a single expert read pays:
- One pread (kernel page cache → user-space `blob_scratch` Vec).
- One `device.new_buffer_with_data` memcpy (Vec → Metal-shared buffer).

For N=8192 chunked prefill at ~200 unique experts per layer × 10
full-attn layers × ~3 chunks = ~6000 expert reads per request, each
~1.7 MB. ~10 GB of double-memcpy bandwidth that should be zero.

**Insight from the original C `flash-moe` (Opus 4.6, per Mike):**
let the page cache handle expert blobs. Don't double-buffer them
through user-space. The C path mmaps each layer file and binds slices
directly to Metal buffers via `newBufferWithBytesNoCopy:`. We need to
mirror that in Rust.

**Implementation.** All machinery is already in moeflux:

- `memmap2 = "0.9"` is a dep (Cargo.toml:22).
- `Mmap::map(&file)` is how `weight_file.rs:101` does it for the
  main weights file.
- `device.new_buffer_with_bytes_no_copy(...)` with a `None`
  deallocator is the pattern in `mtl_weight_buf.rs:107`. The wrapper
  holds the mmap; Metal references its pages without owning the
  lifetime.

### Concrete changes

**`crates/moeflux/src/riir/expert_io.rs`:**

```rust
pub struct ExpertFiles {
    /// One per-layer slot. Each `Some(layer)` holds the layer file's
    /// mmap + the Metal buffer wrapping the entire mapping at
    /// page-aligned length. Per-expert binding is a (buffer, offset)
    /// pair, mirroring how `MtlWeightBuf` exposes per-tensor offsets.
    layers: Vec<Option<LayerMmap>>,
    expert_size: usize,
    experts_dir: PathBuf,
}

pub struct LayerMmap {
    /// Backing mmap. Holds the kernel page-cache reference until drop.
    /// The Metal buffer's pages live in this mapping.
    mmap: Mmap,
    /// Metal buffer wrapping the WHOLE mapping at page-aligned length.
    /// Bind with `set_buffer(slot, Some(&buf), expert_idx * expert_size)`.
    /// Page-alignment of the trailing tail is by mmap construction;
    /// kernels bounds-check on expert_size so they never read past the
    /// last expert.
    buf: metal::Buffer,
}

impl ExpertFiles {
    pub fn open(experts_dir: &Path, device: &metal::Device, expert_size: usize)
        -> Result<Self, ExpertIoError>;

    /// Returns `(metal_buffer, byte_offset, len)` for the slice of
    /// the layer's mmap covering one expert blob. Caller binds with
    /// `set_buffer(slot, Some(buf), byte_offset)`.
    pub fn expert_slice(&self, layer_idx: usize, expert_idx: usize)
        -> Result<(&metal::Buffer, u64), ExpertIoError>;

    /// Backwards-compat: copy an expert into a host buffer. Kept for
    /// the per-token oracle path (`moe_dispatch_per_token` uses
    /// `moe_buffers.data_synced[slot]` which expects a separate
    /// buffer per slot). Backed by a memcpy from the mmap into `out`,
    /// so still cheaper than the old pread but not zero-copy.
    pub fn read_expert(&self, layer_idx: usize, expert_idx: usize, out: &mut [u8])
        -> Result<(), ExpertIoError>;
}
```

**`crates/moeflux/src/riir/full_attn_forward.rs::batched_full_attn_layer_forward`:**

Replace the current pre-load loop:

```rust
// BEFORE — serial pread + memcpy
for &expert_id in &buckets.expert_ids {
    expert_files.read_expert(layer_idx, expert_id as usize, &mut blob_scratch)?;
    expert_blobs.push(MtlBuffer::<u8>::with_data(&device, &blob_scratch));
}
```

with:

```rust
// AFTER — zero-copy; ExpertBlobRef is (buf: &Buffer, off: u64)
let expert_blob_refs: Vec<(&Buffer, u64)> = buckets.expert_ids.iter()
    .map(|&eid| expert_files.expert_slice(layer_idx, eid as usize))
    .collect::<Result<_, _>>()?;
```

Then `encode_moe_batched_permute_fuse` needs to bind from a
`(buf, off)` pair instead of expecting one `&MtlBuffer<u8>` per
expert. That signature change is small — internally the kernel
already reads at offsets (gate/up/down weights all live inside the
single expert blob at different offsets) so it's just plumbing the
caller-side offset through.

**`crates/moeflux/src/riir/expert_forward.rs::encode_moe_batched_permute_fuse`:**

```rust
pub fn encode_moe_batched_permute_fuse(
    cmdbuf: &CommandBufferRef,
    matvec: &MatvecPipelines,
    swiglu: &ComputePipelineState,
    bucket_accumulate: &ComputePipelineState,
    // CHANGED: was `&[MtlBuffer<u8>]`, now `&[(&Buffer, u64)]` — the
    // u64 is the byte offset into the (shared) layer mmap buffer.
    expert_blobs: &[(&Buffer, u64)],
    bucket_input: &Buffer,
    /* ... rest unchanged ... */
);
```

Inside, each `encode_matvec_n_tokens` call adds the per-expert
base offset to the weight/scale/bias offsets it computes from the
4-bit layout. Today it passes `blob.buffer()` + computed offsets;
after the change it passes the layer buffer + (base_off +
computed offset).

### Risks

- **Page alignment**: mmap returns page-aligned base, but Apple
  Silicon's GPU requires the wrapped buffer's length to be
  page-aligned too. `MtlWeightBuf` rounds up to 16384 bytes already
  (`mtl_weight_buf.rs:97`). The per-layer expert file is N experts ×
  `expert_size`. `expert_size` for A3B is `~1.7 MB` and naturally
  page-aligned (4-bit layout's natural alignment is well above 16K);
  worth a `debug_assert!(expert_size % 16384 == 0)` at open time so
  per-expert offsets stay aligned.
- **Mmap lifetime vs Metal buffer**: the buffer holds a raw pointer
  into the mmap. The `None` deallocator means Metal doesn't free
  anything on buffer-drop. The mmap must outlive every buffer
  binding. Solution: keep both inside `LayerMmap`; drop order is
  buffer-then-mmap (Rust struct drop order is field-declaration
  order, top-to-bottom — declare `buf` before `mmap`).
- **F_RDAHEAD**: per `qwen_prefetch_set_based_landed.md`, F_RDAHEAD=0
  is cogito-only currently. With mmap, the readahead behavior changes
  shape — kernel demand-pages on first access. The first chunk's
  cold reads are paid here; warm chunks page-fault into the same
  cached pages. Likely net win vs explicit pread (which also fills
  the page cache as a side effect, but does the extra copy).
- **Snapshot v2 wire format**: doesn't touch expert blobs (they're
  read-only weights, not state). No wire format impact.

### Verification

- All session-4 cosine canaries: unchanged. The math doesn't depend
  on how blobs reach the GPU.
- `batched_diff_oracle::moe_permute_fuse_n_tokens_matches_tokenwise`:
  the same diff, post the encoder signature change.
- New: `expert_io::mmap_expert_slice_round_trip` unit test that
  validates `expert_slice(l, e)` returns bytes byte-identical to a
  control `read_expert` call.
- Bench: re-run `./bench.py --model a3b --prompt-file
  prefill_prompt.txt --max-tokens 1 -n 3` and compare to today's
  ~21 prefill tok/s baseline. Expected: meaningful step-up (the
  ~10 GB of double-memcpy per request goes away).

### Estimated effort

90–120 minutes including the encoder signature plumb-through.

## Phase 2 — Prefetch state machine in `step_internal_batched_gqa` (real Phase G)

**Today:** session-4 routed `eval_token` through `step_internal(&[t], pos, ...)`
because a 32-token directional bench showed +17.6%. The protocol-
compliant bench.py revealed -27% on essay+512-decode — the batched
orchestrator never calls `prefetch.dispatch`, so `decode_hit` dropped
from 0.396 to 0.000. Session 4 reverted (commit `e10ab65`).

**Fix:** add the predict/dispatch pattern from `step_internal_per_token_oracle`
(mod.rs:2087-2106) to `step_internal_batched_gqa`. For each full-attn
layer in the chunk loop:

1. `prefetch.predict_for(layer_idx)` — predict K active experts for
   this layer from the previous chunk's actuals at this layer.
2. `prefetch.dispatch(layer_idx, predicted, k_active, data_prefetch,
   io_pool, experts)` — kick async pread of those K blobs into
   `moe_buffers.data_prefetch[set]`.
3. Run `batched_full_attn_layer_forward` — which inside must:
   - `prefetch.wait_for(layer_idx)` (after Phase 1's mmap landing,
     this is checking whether the prefetched (buf, off) pairs cover
     the layer's bucket set).
   - For each non-empty bucket: if the prefetched set covers
     `expert_id`, use the prefetched slice. Otherwise, sync-load
     (zero-copy mmap fallback from Phase 1).
4. `prefetch.record_actual(layer_idx, actuals)` — record the layer's
   actual bucket-set for the next chunk's prediction.

### Wrinkle: set-based, not slot-based

The per-token oracle's prefetch is K-slot keyed (K=8 prefetched
blobs, K=8 active experts per token, set-based match). For batched,
buckets contain up to all-256 experts. We need a different cardinality:

- **Decode chunk (N=1):** at most K=8 experts. Same as oracle. Prefetch
  predicts K, hit rate is high.
- **Large prefill chunk (N=8192):** potentially all 256 experts. No
  prediction wins here — every expert is hit anyway. Skip prefetch
  for N > some threshold (e.g., `N >= 32` matches the per-token
  oracle's existing GPU SDPA gate value).

Implementation: `step_internal_batched_gqa` checks
`if n_tokens < PREFETCH_BATCH_THRESHOLD` before firing
prefetch.dispatch. Threshold chosen by bench (probably 16 or 32).

### Then route eval_token

Once Phase 2 is green, change `eval_token` to
`self.step_internal(&[token], pos as i32, Some(logits))` (the same
one-line change session 4 tried and reverted). Verify with bench.py
that mean tok/s on essay+512-decode matches or exceeds the oracle path.

### Risks

- The set-based hit lookup is new. Per-bucket: scan
  `prefetched_indices[..K]` for the bucket's expert_id. K=8 → trivial.
- `moe_buffers.data_prefetch[set]` is sized for K=8 slots, which is
  exactly what we want for decode prefetch. No allocation change.

### Estimated effort

60–90 minutes.

## Phase 3 — Per-chunk scratch buffer hoisting

**Today:** `batched_full_attn_layer_forward` allocates fresh GPU
buffers per call:

- `q_buf`, `k_gpu`, `v_gpu`, `attn_out_buf`, `running_max`,
  `running_denom`, `v_partial` (Phase 2, batched SDPA).
- `normed_buf`, `q_proj_buf`, `k_proj_buf`, `v_proj_buf` (Phase 1b,
  batched QKV).
- `attn_with_gate_buf`, `o_proj_buf` (Phase 3b, batched O proj).
- `h_post_buf`, `shared_gate_buf`, `shared_up_buf`, `shared_act_buf`,
  `shared_down_buf` (Phase 3d, batched shared FFN).
- `bucket_input`, `bucket_gate/up/act/out`, `bucket_token_idx`,
  `bucket_weights`, `out_sum` (Phase 4, batched MoE permute-fuse).

At CHUNK_SIZE=8192 these total ~3 GB per layer call × 40 layers =
~120 GB of allocator churn per chunk. The chip has 96 GB unified
memory; under page-cache contention this could measurably slow
prefill at large N.

**Fix:** introduce `BatchedScratch` in `step_internal_batched_gqa`:

```rust
struct BatchedScratch {
    // All sized for max_chunk_size = BATCHED_CHUNK_SIZE = 8192.
    normed: MtlBuffer<f32>,        // [max_n, hidden_dim]
    q_proj: MtlBuffer<f32>,        // [max_n, q_proj_dim]
    k_proj: MtlBuffer<f32>,        // [max_n, kv_dim]
    v_proj: MtlBuffer<f32>,        // [max_n, kv_dim]
    q: MtlBuffer<f32>,             // [max_n, q_dim]
    attn_out: MtlBuffer<f32>,      // [max_n, q_dim]
    running_max: MtlBuffer<f32>,   // [max_n, num_attn_heads]
    running_denom: MtlBuffer<f32>, // [max_n, num_attn_heads]
    v_partial: MtlBuffer<f32>,     // [max_n, num_attn_heads, head_dim]
    attn_with_gate: MtlBuffer<f32>,
    o_proj: MtlBuffer<f32>,
    h_post: MtlBuffer<f32>,
    shared_gate: MtlBuffer<f32>,   // [max_n, shared_intermediate]
    shared_up: MtlBuffer<f32>,
    shared_act: MtlBuffer<f32>,
    shared_down: MtlBuffer<f32>,
    bucket_input: MtlBuffer<f32>,  // [max_n * k_active, hidden_dim]
    bucket_gate: MtlBuffer<f32>,   // [max_n * k_active, moe_intermediate]
    bucket_up: MtlBuffer<f32>,
    bucket_act: MtlBuffer<f32>,
    bucket_out: MtlBuffer<f32>,
    bucket_token_idx: MtlBuffer<i32>,
    bucket_weights: MtlBuffer<f32>,
    out_sum: MtlBuffer<f32>,
}
```

Hosting site: either `RsCtx` as a lazily-allocated `Option<BatchedScratch>`
populated on first batched call (analogous to how `linear_buffers` is
lazily built), or owned by `step_internal_batched_gqa` and rebuilt
when `n_tokens` exceeds the previous max (one-time growth, then
amortized).

Pass `&mut scratch` into `batched_full_attn_layer_forward`. Replace
every `MtlBuffer::<f32>::with_len/with_data` allocation site with the
corresponding scratch field. The `with_data` sites that load from
host need a memcpy into the scratch buffer's shared-storage contents
(`unsafe { ptr::copy_nonoverlapping(...) }`).

Total memory at CHUNK_SIZE=8192 + a3b dims: ~3 GB once per Ctx
instead of ~3 GB per layer call.

### Risks

- Lazy allocation must happen before any cmdbuf dispatch reads from
  the scratch. Allocate at the start of `step_internal_batched_gqa`
  if `scratch.is_none()`.
- `k_active` is variant-locked at compile time (8 for A3B), so
  bucket scratch sizing is straightforward.
- Aliasing: each scratch buffer is used in one phase per layer call.
  No two phases overlap in their use of the same field. Safe.

### Estimated effort

60–90 minutes.

## Phase 4 — Dead-code cleanup (deferred Phase G)

**Today** (post-session-4 Phase G revert): `eval_token` is back on
the per-token oracle path which uses the GPU SDPA fast path
(attn_scores_batched / attn_softmax_batched / attn_values_batched
kernels + the per-layer `gpu_kv_k` / `gpu_kv_v` mirrors). Those are
still live for decode.

**After Phase 2 (real Phase G):** `eval_token` routes through
`step_internal_batched_gqa` which uses tiled SDPA. The per-token GPU
SDPA fast path becomes dead code:

- Three kernels: `attn_scores_batched`, `attn_softmax_batched`,
  `attn_values_batched`. Defined in `shaders.metal`; encoded in
  `linear_attn_forward.rs:967-1001` (inside `post_attention_pre_moe`,
  `gpu_attn_args=Some` branch).
- Persistent `gpu_attn_scores` scratch buffer (≈67 MB at
  GPU_KV_SEQ=8192 × num_attn_heads × f32).
- Per-full-attn-layer `gpu_kv_k[fa_idx]` / `gpu_kv_v[fa_idx]` mirrors
  (15 × 16.8 MB each = ≈500 MB).
- The `kv_len >= 32 && kv_len < GPU_KV_SEQ` gate at
  `full_attn_forward.rs:331-333`.
- The host-side GPU mirror KV append in `full_attn_forward.rs:300-322`.
- The `pub use ... gpu_attn_scores_batched, gpu_sigmoid_gate` exports
  in mod.rs:72-73 (no in-crate callers after the deletion).
- `post_attention_post_o_proj_to_intermediates`'s
  `#[allow(dead_code)]` — was kept as a forward-looking building
  block; if Phase 2's batched eval_token doesn't use it either, delete.

**Snapshot v2 wire format:** `LayerForwardBuffers` exposes these
fields publicly; removing them changes the struct layout that
state_snapshot.rs reads/writes. Bump the snapshot wire-format version
(or simply update v2 — the only existing v1 callers are the test
suite, per the session-2 memo, and we have a clean cutover window
before Council prompt caching lands).

Net: ~150 LOC removed, ~570 MB of persistent GPU buffer reclaimed
per Ctx.

### Risks

- Snapshot v2 wire format change. Verify with `snapshot_v2_roundtrip`
  test post-deletion.
- MLA variants (Cogito-V2) don't use the Gqa GPU SDPA fast path —
  they go through `step_internal_mla_*`. So deletion is Gqa-only
  surface; MLA path unaffected.

### Estimated effort

45–60 minutes.

## Order of operations

1. **Phase 1 (zero-copy mmap)** — biggest win, no other phases
   depend on it. Land first.
2. **Phase 2 (batched prefetch)** — composes on top of Phase 1's
   `expert_slice(layer, expert)` API. Land second.
3. **Phase 3 (scratch hoisting)** — orthogonal to 1 and 2; can land
   any time after both are in, but cleaner after them.
4. **Phase 4 (dead-code cleanup)** — gated on Phase 2's eval_token
   re-routing being green. Land last.

Each phase has its own cosine gate (the session-4 canary battery)
and bench.py number to validate.

## Outside-scope-but-still-on-the-list

- **Protocol-compliant headline bench.** Reboot, n≥3, high-perf
  power, against llama.cpp. Standalone bench-only session.
- **Prefill progress callback** for drama_llama. Per
  `future_work_prefill_progress_callback.md`. Independent session.
- **blallama emits prefill_ms separately.** `log_stats` in
  `bin/blallama/blallama.rs:178-196` only emits `elapsed_ms` total.
  Adding a `prefill_ms` field would let `bench.py`'s prefill_tok/s
  metric stop being an approximation. ~30 min change.

## Expected end-state

After session 5: prefill on a3b should be at or above llama.cpp on
the same hardware. Decode stays within ±5% of current (the prefetch
state machine in batched should match the oracle's hit rate). Total
LOC delta: net-negative (Phase 4 alone removes ~150 lines).
