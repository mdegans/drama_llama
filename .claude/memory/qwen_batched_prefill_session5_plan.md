# Session 5 plan — close the gap with llama.cpp on a3b prefill

Entry-point: [`qwen_batched_prefill_session4_landed.md`].

**End-state goal:** prefill rate on Qwen3.6-A3B at or above llama.cpp.
Session 4 hit 4× on bench.py essay+512 (post-Phase-G-revert) and
~21 prefill tok/s on the 992-token prefill_prompt.txt with max_tokens=1.
Session 5 closes most of the remaining gap.

## What the post-session-4 profile actually says

Samply profile of `bench.py --model a3b --prompt-file prefill_prompt.txt
--max-tokens 1`, 992-token prefill, ~50s window:

```
=== SELF ===
91.2%  __psynch_cvwait        ← main thread sleeping on commit_and_wait
 1.2%  _platform_memmove      ← host memcpy (MtlBuffer::with_data etc.)
 0.7%  kevent_id              ← async dispatch
 0.5%  pread                  ← expert blob disk I/O
 0.2%  rms_norm_per_head_cpu
 0.2%  moe_router_cpu

=== INCLUSIVE ===
55.0%  linear_attn_layer_forward       ← per-token linear-attn fallback
38.4%  commit_and_wait_labeled
34.2%  complete_deferred_experts_into
31.6%  post_attention_pre_moe          (called from linear-attn forward)
22.9%  moe_dispatch_per_token          (called from linear-attn forward)
 9.9%  batched_full_attn_layer_forward ← session-4 batched path
```

**The session-4 batched full-attn work is 9.9% of the time.** The
real bottleneck is the per-token linear-attn fallback in
`step_internal_batched_gqa` — A3B has 31 linear-attn layers ×
N=992 tokens = 30,752 per-token GPU-dispatch cycles per chunk, each
fenced by `commit_and_wait`. That's the dominant CPU↔GPU toggle
Mike observed in activity monitor.

Optimizations targeting the batched full-attn path (mmap, scratch
hoist, etc.) cap out at +9.9% absolute. Linear-attn batching is
where the next big leverage lives.

## C-side reference — what flash-moe actually does

`metal_infer/infer.m` evolved past the comment in the Rust port's
`expert_io.rs:144`. Key findings from auditing the C source:

- **`InferPreadTask`** has a field `const void *mmap_base; // if
  non-NULL, memcpy from mmap instead of pread`. Both reads land in
  the same destination Metal buffer. **Runtime switch, not a static
  choice.**
- C **pre-allocates** K aligned (2 MB aligned for DMA throughput
  per the comment "3.6x faster than 16KB") Metal-shared buffers via
  `posix_memalign` + `newBufferWithBytesNoCopy:length:options:
  deallocator:nil`.
- C **pread**s the expert blob directly into `[buf contents]` —
  no intermediate user-space Vec. Single memcpy from file to Metal
  buffer backing memory.
- Async via `dispatch_group_async(g_io_gcd_queue, ...)` —
  parallel preads overlap with GPU compute.
- Has an LRU `g_expert_cache` (`expert_cache_lookup` /
  `expert_cache_insert`) that holds recently-used expert blobs as
  pre-loaded Metal buffers across layers/tokens.

The Rust port's `MoeBuffers::data_synced[k]` mirrors C's
`buf_multi_expert_data[k]` for the per-token oracle, and the io_pool
parallel pread in `moe_dispatch_per_token` mirrors C's
`dispatch_group_async`. **Per-token oracle is structurally aligned
with C.** The session-4 batched path is where the extra memcpy
sneaks in (via `MtlBuffer::with_data(&blob_scratch)`).

## Priority order

1. **Chunkwise / batch-pipelined linear-attn** — biggest single lever.
2. **Expert IO refactor (A/B mmap vs pread-direct)** — measurable,
   moderate effort.
3. **Real Phase G (prefetch in batched orchestrator)** — required
   for decode parity if we re-route eval_token.
4. **Per-chunk scratch hoist** — small win, low effort.
5. **Dead-code cleanup of per-token GPU SDPA fast path** — closing
   commit.

Per Mike: aim to land all of these in one session.

## Phase 1 — Batch-pipelined linear-attn (headline)

**Today:** every linear-attn layer's forward is called per-token in a
loop inside `step_internal_batched_gqa::step_internal_batched_gqa`
(mod.rs ~1380-1420). Each call:

- Sets `buffers.input` from `batched_hidden[t]`.
- Runs `linear_attn_layer_forward(... gpu_combine=true, chain_next=None ...)`
  which issues CMD1 (input rms_norm + 4 batched projections) + 5
  linear-attn fused kernels + `post_attention_tail` (= pre-MoE +
  K-expert dispatch). Each layer call commits multiple cmdbufs.
- Drains the deferred K-expert dispatch via
  `complete_deferred_experts_into` into the next slot of
  `hidden_out_stack`.

At N tokens × 31 layers, that's ~31N×(few cmdbuf-cycles + parallel
pread + deferred drain). The CPU↔GPU toggle is per-token-per-layer.

**Fix shape:** the linear-attn forward has THREE kinds of work:

- **Matvecs**: input rms_norm (1) + projections (4: qkv, z, beta,
  alpha) + post-attn rms_norm (1) + gate logits (1, 8-bit) + shared
  gate scalar (1, 8-bit) + shared FFN (3) + o_proj (1) = ~11
  matvecs per token per layer. All independent across tokens —
  **batchable over N via `encode_matvec_n_tokens` / bf16 variant**.
- **Linear-attn recurrent kernels**: conv1d_step, rms_norm_qk,
  compute_decay_beta, gated_delta_net_step, gated_rms_norm. The
  recurrence (delta_state) MUST advance one step per token —
  truly sequential. But each kernel is small.
- **MoE K-expert dispatch + combine**: this is the heaviest GPU work
  per layer per token. Same shape as full-attn's MoE — can be
  batched via the existing `encode_moe_batched_permute_fuse` once we
  have the joint per-token routing CSR.

**Implementation outline:**

1. New `batched_linear_attn_layer_forward(...)` in
   `crates/moeflux/src/riir/linear_attn_forward.rs`, sibling to
   the existing `linear_attn_layer_forward`. Signature mirrors
   `batched_full_attn_layer_forward`: takes hidden_in / hidden_out
   stacks and the layer's recurrent state.

   Body:
   - Phase 1a: per-token input rms_norm → `normed_stack` (or batched
     via the new `rms_norm_apply_bf16_n_tokens` kernel from Phase 1b
     below).
   - Phase 1b: batched 4 projections (qkv, z, beta, alpha) via
     `encode_matvec_n_tokens` × 4. All weights 4-bit.
   - Phase 1c: per-token loop running conv1d_step + rms_norm_qk +
     compute_decay_beta + gated_delta_net_step + gated_rms_norm.
     Recurrent state advances sequentially. **All 5 dispatches stay
     in ONE shared cmdbuf across the whole N-loop** — no
     commit_and_wait inside the per-token loop. The recurrent state
     buffers (conv_state, delta_state) are read+written per kernel
     and serialize on Metal encoder ordering inside the cmdbuf.
     Final commit_and_wait at the end of the N-loop.
   - Phase 1d: batched o_proj.
   - Phase 1e: per-token post_attention pre-MoE via the new
     `post_attention_residual_norm_route` helper (already exists
     from B4) → captures per-token routing + h_post + shared_gate.
   - Phase 1f: batched shared FFN (mirroring B4's structure).
   - Phase 1g: batched MoE permute-fuse (mirroring B1's structure).
   - Phase 1h: CPU per-token combine.

2. Update `step_internal_batched_gqa` to dispatch linear-attn layers
   to the new batched function instead of the per-token loop.

**Risk: cmdbuf-internal serialization for the recurrent kernels.**
Today the per-token oracle commits between every linear-attn kernel
because each token's compute is one cmdbuf. Batching N tokens into
one cmdbuf means N×5 kernel dispatches in one buffer. Need to
verify that Metal's encoder ordering preserves the recurrence
(state[t] visible to state[t+1]'s dispatch). It should — same
encoder, write-then-read in dispatch order is the documented
contract. But test cosine=1.0 at N=4 against the per-token oracle
before scaling up.

**Risk: cmdbuf scratch size.** N × per-token recurrence buffers
might exceed Metal's per-cmdbuf binding limits. Mitigate by
unbinding/rebinding inside the cmdbuf (Metal allows that).

**Win estimate:** the profile says linear-attn is 55% of total
time. If batching matvecs + pipelining the cmdbuf gets us to ≤15%,
that's a ~2× total speedup (we drop from 100% to ~60% inclusive
linear-attn → ~30% inclusive after batching → total work shrinks
from 100% to ~75%, giving 100/75 ≈ 1.33×). Plus the full-attn path
gains a bit from the deferred-ring-free fast cmdbuf scheduling. Net:
expect 1.5–2× over today's 21 prefill tok/s.

**Files touched:**
- `crates/moeflux/src/riir/linear_attn_forward.rs` —
  `batched_linear_attn_layer_forward`.
- `crates/moeflux/src/riir/mod.rs::step_internal_batched_gqa` —
  dispatch wiring.
- Maybe a new `rms_norm_apply_bf16_n_tokens` shader if Phase 2 below
  doesn't already add it.

**Estimated effort:** 90–150 minutes.

## Phase 2 — Expert IO refactor (mmap-vs-pread A/B)

**Today:** `batched_full_attn_layer_forward` does:
```rust
for &expert_id in &buckets.expert_ids {
    expert_files.read_expert(layer_idx, expert_id as usize, &mut blob_scratch)?;
    expert_blobs.push(MtlBuffer::<u8>::with_data(&device, &blob_scratch));
}
```
Two memcpys per expert (pread into Vec, then Vec → fresh Metal buffer).

**Per-token oracle path does it right already** (matches C's
`buf_multi_expert_data` shape): pre-allocated Metal-shared buffers
in `MoeBuffers::data_synced[k]`, pread directly into
`as_mut_slice()`. Single memcpy from file to Metal buffer.

**Approach A — pread-direct (matches C production):**

- Pre-allocate per-layer Metal-shared aligned buffers on Ctx setup.
  One buffer per layer sized for `num_experts * expert_size` (~435 MB
  per layer × 40 = 17 GB — too much). Better: a pool of K_buf
  (~64) buffers reused across layers in a small LRU.
- OR: per-call (per-chunk per-layer): allocate ONE big aligned
  Metal-shared buffer sized for `num_unique_experts_in_chunk *
  expert_size`. pread each unique expert into its slot. Bind
  `(layer_buf, expert_idx * expert_size)` to the permute-fuse
  encoder. Buffer freed at end of chunk.
- Parallel pread via `io_pool.par_iter_mut` (already used by the
  per-token oracle).
- 2 MB alignment via `MTLDevice::heapBufferDescriptor` or a custom
  aligned-vec wrapper + `newBufferWithBytesNoCopy`. C claims 3.6×
  DMA throughput at 2 MB alignment.

**Approach B — mmap + newBufferWithBytesNoCopy:**

- `Mmap::map(layer_file)` once per layer at Ctx setup. Hold in
  `ExpertFiles::layers[i].mmap`.
- `device.new_buffer_with_bytes_no_copy(mmap.as_ptr(), aligned_len,
  StorageModeShared, None)` once per layer. Hold in
  `ExpertFiles::layers[i].buf`.
- Per-expert binding: `(layer_buf, expert_idx * expert_size)`.
- Zero copy. Kernel demand-pages on first access.

**Test plan:** implement both behind a thread-local switch (similar
to `set_batched_chunk_size_for_test`), bench both vs current
implementation. Keep the winner; document trade-offs.

Hypotheses:
- pread-direct likely wins for large N (most experts get touched
  → page-cache miss latency adds up); 2 MB alignment buys real DMA
  throughput.
- mmap likely wins for small N (decode K=8 → only those pages
  faulted; mmap's lazy paging avoids reading unused experts).
- Hybrid: chunk-size-conditional path. At N > threshold, pread
  whole-layer ahead of time. At N ≤ threshold, mmap.

**Files touched:**
- `crates/moeflux/src/riir/expert_io.rs` — add the
  `expert_slice(layer, expert)` API, both backends. Keep
  `read_expert` for backwards compat.
- `crates/moeflux/src/riir/expert_forward.rs::encode_moe_batched_permute_fuse`
  — signature changes from `&[MtlBuffer<u8>]` to `&[(&Buffer, u64)]`.
- `crates/moeflux/src/riir/full_attn_forward.rs::batched_full_attn_layer_forward`
  — drop the `MtlBuffer::with_data` loop, use `expert_slice`.
- `crates/moeflux/src/riir/linear_attn_forward.rs::batched_linear_attn_layer_forward`
  (from Phase 1) — same.

**Estimated effort:** 90–120 minutes including the A/B harness.

## Phase 3 — Real Phase G (prefetch in batched orchestrator)

**Today:** `step_internal_batched_gqa` never calls
`prefetch.dispatch(...)`. Decode through this path got 0.000
decode_hit vs the per-token oracle's 0.396 (session-4 bench).
Session 4 reverted Phase G's `eval_token` routing for this reason.

**Fix:** mirror the per-token oracle's pattern (mod.rs:2087-2106)
inside `step_internal_batched_gqa`'s per-layer chunked loop:

1. Before each full-attn / linear-attn layer's batched forward:
   `prefetch.predict_for(layer_idx)` → if Some, `prefetch.dispatch(
   layer_idx, predicted, k_active, data_prefetch_set, io_pool,
   experts)`.
2. In the batched layer forward (after Phase 2's `expert_slice`
   integration): when resolving the bucket's `expert_ids` to GPU
   buffer refs, check if `data_prefetch[set][buf_idx]` holds any of
   them. Use those if so; sync-load misses via `expert_slice` or
   pread-direct.
3. `prefetch.record_actual(layer_idx, actuals)` at the end of the
   layer.

Threshold gating: at large N (say N ≥ 32 — same gate value the
per-token GPU SDPA fast path uses), every expert is touched anyway
and prefetch is wasted work. Skip the dispatch above the threshold.

After Phase 3 is green, re-route `eval_token` through
`self.step_internal(&[token], pos as i32, Some(logits))`. Bench.py
n=3 on essay+512: should match or exceed the oracle's ~10.5 tok/s.

**Files touched:**
- `crates/moeflux/src/riir/mod.rs::step_internal_batched_gqa` —
  add the per-layer dispatch.
- `crates/moeflux/src/riir/full_attn_forward.rs::batched_full_attn_layer_forward`
  and the new `batched_linear_attn_layer_forward` — accept the
  prefetched set, do set-based hit lookup.
- `crates/moeflux/src/riir/mod.rs::eval_token` — re-route to
  `step_internal`.

**Estimated effort:** 60–90 minutes.

## Phase 4 — Per-chunk scratch buffer hoist

**Today:** `batched_full_attn_layer_forward` allocates ~3 GB of GPU
scratch per call. 40 layers × ~3 GB = ~120 GB allocator churn per
chunk.

**Fix:** introduce `BatchedScratch` on `RsCtx`, lazily allocated on
first batched call, sized for `BATCHED_CHUNK_SIZE = 8192`. Fields:
all the `MtlBuffer<f32>`s currently created inside the layer call
(normed, q_proj, k_proj, v_proj, q, attn_out, running_max,
running_denom, v_partial, attn_with_gate, o_proj, h_post,
shared_gate, shared_up, shared_act, shared_down, bucket_input,
bucket_gate, bucket_up, bucket_act, bucket_out, bucket_token_idx,
bucket_weights, out_sum).

Total ~3 GB once per Ctx instead of per call.

Same shape applies to the new `batched_linear_attn_layer_forward`
from Phase 1.

**Files touched:**
- `crates/moeflux/src/riir/mod.rs::RsCtx` — new
  `batched_scratch: Option<BatchedScratch>` field.
- `crates/moeflux/src/riir/full_attn_forward.rs` and
  `crates/moeflux/src/riir/linear_attn_forward.rs` — take
  `&mut BatchedScratch`, replace `MtlBuffer::with_len/with_data`
  call sites.

**Estimated effort:** 60–90 minutes.

## Phase 5 — Dead-code cleanup of per-token GPU SDPA fast path

After Phase 3's `eval_token` routing change, the per-token GPU SDPA
fast path is dead for Gqa production. Delete:

- Kernels `attn_scores_batched`, `attn_softmax_batched`,
  `attn_values_batched` (and `gpu_sigmoid_gate` if unused elsewhere).
- The `gpu_attn_args=Some` branch in `post_attention_pre_moe`
  (linear_attn_forward.rs:867-924).
- Persistent buffers `gpu_attn_q`, `gpu_attn_scores`, `gpu_attn_out`,
  `gpu_attn_gate` in `LayerForwardBuffers`.
- Per-full-attn-layer mirrors `gpu_kv_k[fa_idx]` / `gpu_kv_v[fa_idx]`.
- The `kv_len >= 32 && kv_len < GPU_KV_SEQ` gate in
  `full_attn_forward.rs:331-333`.
- GPU mirror KV append at `full_attn_forward.rs:300-322`.
- `post_attention_post_o_proj_to_intermediates`'s
  `#[allow(dead_code)]` — no callers expected post-Phase-3.

Net: ~150 LOC removed, ~570 MB of persistent GPU buffer reclaimed
per Ctx.

**Snapshot v2 wire format:** `LayerForwardBuffers` exposes
`gpu_kv_k/v` as public fields. Their removal changes the struct
layout that `state_snapshot.rs` references. Bump snapshot wire
format (or update v2 in place — test suite is the only existing
v1 caller per session-2 memo).

**Files touched:**
- `crates/moeflux/shaders/shaders.metal` — delete kernel definitions.
- `crates/moeflux/src/riir/gpu_attn.rs` — delete pub funcs and
  pipeline struct.
- `crates/moeflux/src/riir/linear_attn_forward.rs::LayerForwardBuffers`
  and `post_attention_pre_moe`.
- `crates/moeflux/src/riir/full_attn_forward.rs::full_attn_layer_forward`
  / `full_attn_pre_moe_layer_forward`.
- `crates/moeflux/src/riir/state_snapshot.rs` — wire format.
- `crates/moeflux/src/riir/metal.rs` — ALL_KERNELS list trimming.
- `crates/moeflux/src/riir/mod.rs` — `pub use` exports.

**Estimated effort:** 45–60 minutes.

## Order of operations (in one session)

1. Phase 1 (batched linear-attn). Lands the biggest perf win.
2. Phase 2 (expert IO). Both A/B variants behind a switch, bench,
   keep winner.
3. Phase 3 (real Phase G). Composes on top of Phase 1's batched
   linear-attn and Phase 2's `expert_slice` API.
4. Phase 4 (scratch hoist). Lower-priority perf cleanup; can land
   any time after Phase 1 and 2.
5. Phase 5 (dead-code cleanup). Closing commit.

Each phase has its own cosine gate (session-4 canary battery) and a
bench.py / profile.py number to validate.

Estimated total: 4–7 hours of focused work, doable in one session
per session-4's precedent.

## Verification protocol

```bash
cd ~/Projects/moeflux

# Cosine canaries (real artifacts):
cargo test --release -p moeflux --features model-qwen3-6-35b-a3b \
  --test diff_oracle -- --ignored --nocapture --test-threads=1 \
  state_round_trip_rust eval_token_matches_c_single_step \
  eval_prompt_matches_per_token_oracle slot_reuse_race_regression_rust \
  state_load_rust_from_c_save eval_prompt_matches_c_multi_token \
  eval_prompt_chunked_matches_eval_prompt_whole_prompt \
  prompt_cache_start_pos_nonzero_matches diag_b2_eval_prompt_chunk_1

# Per-phase profile (samply) to confirm where time moved:
cd ~/Projects/drama_llama
./profile.py --model a3b --prompt-file prefill_prompt.txt \
  --max-tokens 1 --duration 60 --top 30

# Bench progression (single-iter, directional; protocol-compliant
# bench is its own session):
./bench.py --model a3b --prompt-file prefill_prompt.txt \
  --max-tokens 1 -n 3
./bench.py --model a3b -n 3                        # decode regression check
```

After Phase 1: linear_attn_layer_forward inclusive should drop from
55% to <20%. After Phase 2: pread + with_data self-time should
collapse. After Phase 3: bench.py decode tok/s should match the
oracle path (~10.5 tok/s).

## Expected end-state

Prefill on a3b at or above llama.cpp on the same hardware. Decode
within ±5% of the per-token oracle (Phase 3's prefetch in batched
matches the hit rate). LOC delta: net-negative once Phase 5 runs.
Memory delta: -570 MB persistent GPU buffer per Ctx.

## Notes on the C reference (saved here for the implementer)

- C's hot path: pread directly into pre-allocated Metal-shared
  buffers (`buf_multi_expert_data[k]`). 2 MB aligned. Async via
  Grand Central Dispatch's `dispatch_group_async` on
  `g_io_gcd_queue`.
- C also keeps an LRU `g_expert_cache` (Metal buffers keyed by
  (layer, expert)) for cross-token reuse. **Not in Rust port today.**
  Worth considering once Phase 2 lands — if a token's bucket
  overlaps with a prior chunk's bucket, the cached buffer wins.
- C's `mmap_base` field on `InferPreadTask` is the runtime A/B
  switch between pread and `memcpy(dst, mmap_base + off, sz)`. Both
  paths use the same destination buffer.
- `fcntl(fd, F_RDAHEAD, 0)` is set on cogito-v2 (the model where
  the working set doesn't fit in RAM) and unset elsewhere. The
  Rust port already gates F_RDAHEAD=0 to cogito per
  `qwen_prefetch_set_based_landed.md`.
