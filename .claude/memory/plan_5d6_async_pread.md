---
name: Plan — slice 5d-6 async / parallel expert pread
description: Pre-implementation plan for the async/parallel pread slice. Bridges yesterday's 5d-5 (pread direct into shared-storage) and the next session's plan-mode implementation. Captures C-side reference, riir-side surfaces, proposed two-step shape, diff strategy, and open decisions.
type: project
---

# Slice 5d-6: async / parallel expert pread — implementation plan

**Status**: planning, not started.
**Authored**: 2026-04-28 morning.
**Branch when implementing**: `riir` in `~/Projects/moeflux`.
**Diff baseline**: 5.58 cold / 5.38 warm tok/s (after 5d-5).
**Target**: parity with C path (~10.13 cold / 8.70 warm), or as close as the
no-prediction variant gets.

---

## 1. Why this slice

After 5d-5, blallama A3B essay perf is 5.4 warm / 5.6 cold tok/s vs C 8.7 / 10.1.
Activity Monitor reports CPU 60% / GPU 20% during steady-state generation; C
runs ~40% GPU on A3B (~60% on A17B). We're CPU-bound, the GPU is starved.

The 5d-4 re-profile showed `pread` at 32.7% of CPU samples (~70 ms / token at
our current rate). Eliminating it from the critical path is the single biggest
remaining lever.

The architectural mechanism: C runs K-expert disk reads in **4 parallel
pthreads** on a **double-buffered** set, with optional **speculative
prefetch** for the next layer's experts. Reference: detailed C-side scout
report below (§3).

---

## 2. Goal and scope

### In scope
- Parallel K-expert preads (4 worker threads or equivalent) so within-layer
  pread becomes ~K× faster.
- Double-buffered `MoeBuffers.data` (set A vs set B) with explicit swap.
- Speculative prefetch using **last-token-same-layer** expert indices as the
  prediction (the simplest scheme that captures token-to-token expert
  locality — same as what C does).
- Hit/miss path: if prediction matched actual routing → reuse set B (save a
  pread); if missed → sync pread fallback into set A.
- Diff oracle stays green at the existing tight floors.

### Out of scope
- LZ4-compressed expert path (slice 9f / Phase 7 deferable).
- 2-bit quantization path (still FIXME from slice 9a).
- Prediction schemes more sophisticated than "same indices as last token".
- `gpu_combine = false` CPU-finalize path — keep that on the synchronous
  shape (it's diff-oracle-only and doesn't need to be fast).

### Explicit non-goal: do not block on prediction-accuracy correctness
The prefetch is purely a hint — every miss falls back to a sync pread, so
incorrect predictions cost only the wasted prefetch I/O. We do NOT need to
prove prediction quality before landing.

---

## 3. C-side reference (from the morning scout)

### Algorithm
Three I/O strategies coexist:
1. **Sync pthread pool** (4 workers, `g_io_pool`) — used by the LZ4
   compressed path. Persistent threads sleeping on a condvar; main thread
   broadcasts work, workers stride through tasks, signal completion.
   `infer.m:3149-3234`.
2. **Async GCD dispatch_group** (`g_async_pread`) — the non-LZ4 hot path.
   `dispatch_group_async` per K expert returns immediately;
   `dispatch_group_wait` blocks before GPU dispatch. `infer.m:3240-3282`.
3. **Speculative prefetch** — after CMD1 wait, predicted experts for layer
   N+1 are loaded into `buf_multi_expert_data_B` via async pread. On layer
   N+1, predictions are matched against actual routing; hits skip pread,
   misses sync-pread into `buf_multi_expert_data`. `infer.m:4460-4470`,
   `5510-5562`, `5583-5601`.

### Buffer layout (`infer.m:991-1000`)
```c
id<MTLBuffer> buf_multi_expert_data[MAX_K];    // set A
id<MTLBuffer> buf_multi_expert_data_B[MAX_K];  // set B (prefetch target)
id<MTLBuffer> buf_multi_expert_gate[MAX_K];    // single — consumed inside one dispatch
id<MTLBuffer> buf_multi_expert_up[MAX_K];      // single
id<MTLBuffer> buf_multi_expert_act[MAX_K];     // single
id<MTLBuffer> buf_multi_expert_out[MAX_K];     // single
```
`MAX_K = 16`. EXPERT_SIZE on A3B is ~1.77 MB; on A17B ~6.75 MB.

Allocation rounds to 2 MB boundary for DMA alignment (the comment at
`infer.m:1196` says "The pread DMA controller transfers 3.6x faster with 2MB
alignment vs 16KB"). **Worth replicating in the Rust port.**

### Per-layer timeline (paraphrased from the scout)
```
LAYER N START
├─ input_norm + attn (CPU/GPU CMD1)
├─ MoE router (CPU): gate_logits → expert_indices for layer N
├─ Decision: were these indices predicted (= last token's layer-N indices)?
│   ├─ HIT for slot k → expert_bufs[k] = data_B[p]  (no I/O needed)
│   └─ MISS for slot k → submit sync pread into data_A[k]
├─ CMD2 encode (o_proj + residual + post-attn norm + shared FFN)
│   [pread runs concurrently here]
├─ async_pread_wait()  — block until any sync preads complete
├─ CMD3 encode (K-expert FFN + combine), commit DEFERRED (no wait)
├─ async_pread_start() for predicted layer N+1 indices into data_B
│   [these run concurrently with next layer's CMD1+CMD2]
LAYER N END
```

Critical detail: the **prediction for layer N+1** is just "the indices the
router selected for layer N+1 in the **previous token**". Tokens have high
expert-locality so this hit-rate is high in practice.

### Synchronization primitives
- Sync pool: `pthread_mutex_t` + `pthread_cond_t work_ready/work_done`,
  generation counter so workers detect new batches.
- Async path: GCD `dispatch_group_t`, equivalent to a counted semaphore.

### Edge cases
- First layer of first token: no predictions exist; falls back to sync pread.
- Last layer: prediction is computed but no prefetch (no layer N+1 exists).
- `memory_clear`: predictions invalidated (`g_pred_valid = 0`).
- `state_save`/`state_load`: no interaction (predictions are per-token state,
  not part of the snapshot).

### What we're NOT copying
- LZ4 path (out of scope).
- The dual sync/async coexistence (we use one mechanism for all loads).
- GCD specifically (Rust ecosystem has equivalents).

---

## 4. Riir-side current state (inventory, 2026-04-28)

### `MoeBuffers` (`crates/moeflux/src/riir/expert_forward.rs:274-339`)
Single set of per-slot buffers:
```rust
pub struct MoeBuffers {
    data: [MtlBuffer<u8>; MAX_K],     // EXPERT_SIZE bytes each
    gate: [MtlBuffer<f32>; MAX_K],    // MOE_INTERMEDIATE floats
    up: [MtlBuffer<f32>; MAX_K],
    act: [MtlBuffer<f32>; MAX_K],
    out: [MtlBuffer<f32>; MAX_K],     // HIDDEN_DIM floats
    input: MtlBuffer<f32>,
    h_mid: MtlBuffer<f32>,
    shared_out: MtlBuffer<f32>,
    moe_hidden: MtlBuffer<f32>,
    combine_params: MtlBuffer<f32>,
}
```
Public accessor (added 5d-5): `data_slot_mut(slot) -> &mut [u8]`.
`MAX_K = 16` (`expert_forward.rs:57`).

### `ExpertFiles` (`crates/moeflux/src/riir/expert_io.rs:75-117`)
```rust
pub struct ExpertFiles {
    layers: Vec<Option<File>>,   // per-layer fd, opened once at RsCtx::open
    expert_size: usize,
    experts_dir: PathBuf,
}

impl ExpertFiles {
    pub fn read_expert(&self, layer_idx: usize, expert_idx: usize, out: &mut [u8])
        -> Result<(), ExpertIoError>;
}
```
`read_expert` uses `std::os::unix::fs::FileExt::read_at` (= `pread64`).
`&self` and `&mut [u8] out` ⇒ **safe to call concurrently from multiple
threads** (different `out` slices, no shared per-fd offset). `ExpertFiles`
should already implement `Sync`.

### Production call site (`linear_attn_forward.rs:836+`, post-5d-5)
Inside `post_attention_tail`, after MoE router resolves indices:
```rust
let k = k_active;
if gpu_combine {
    for slot in 0..k {
        let expert_idx = indices[slot] as usize;
        let dst = moe.data_slot_mut(slot);
        expert_files.read_expert(layer_idx, expert_idx, dst)?;  // SERIAL
    }
    gpu_batched_experts_begin_pre_staged(metal, moe, deferred, k as i32, ...)?;
}
```
This is the serial bottleneck. Sequential K reads, on the critical path
between MoE router (CPU) and CMD3b dispatch.

### Lazy-init flow (`mod.rs::ensure_linear_resources`)
`MoeBuffers` is allocated lazily on first call; lives on `RsCtx.moe_buffers`.
A new "set B" allocation slots in here.

### Threading deps in workspace
None. Cargo.toml has no `rayon`, `crossbeam`, or `tokio` (other than what
drama_llama brings in). Need to add a dependency or use `std::thread::scope`.

### Diff oracle touchpoints
- `gpu_batched_experts_forward` (synchronous test path) — uses host-slice API
  through `gpu_batched_experts_encode`, which stages into `MoeBuffers.data`.
  Stays synchronous; no async needed for correctness tests.
- `RsCtx::begin_deferred_experts` (the public API behind diff oracle hooks)
  — keeps host-slice signature. Internally stages, then calls the same
  encode path the production fast path uses.

The diff oracle does NOT exercise the prefetch / double-buffer scheme. We
need a NEW test that does.

---

## 5. Proposed design — two-step

The full async + prefetch slice is large. Split into two commits so we
land the smaller win first and can measure each step.

### 5d-6a: parallel sync preads
**Hypothesis**: K serial preads → K parallel preads using a thread pool, but
still synchronous on the critical path. Saves ~(K−1)/K × pread time per layer
when reads are I/O-bound; less when in OS page cache.

**Changes**:
- Add a dependency: `rayon` (preferred, well-known, used by candle/mistral.rs)
  OR use `std::thread::scope` (no dep, slightly more code). Mike's call.
- New pool struct (lives on `RsCtx` or as a process-global). On `RsCtx::open`,
  init with 4 worker threads (matches C's `NUM_IO_THREADS = 4`).
- Replace the sequential `for slot in 0..k` loop in `post_attention_tail`
  with a parallel dispatch. Each worker calls `expert_files.read_expert`
  into `moe.data_slot_mut(slot)` for its assigned slots.
- Borrow safety: `&mut [u8]` slices to disjoint buffers — straightforward
  with `rayon::scope` or `std::thread::scope`.

**Expected perf**: 30-50% of the slice 5d-6 win (depends on whether disk
or CPU is the per-pread bottleneck — page-cache hits are mostly CPU
syscall + memcpy time).

**Risks**: low. Determinism preserved (waits before dispatch). Diff tests
unchanged.

### 5d-6b: double-buffer + speculative prefetch + async overlap
**Hypothesis**: hide the entire pread cost under GPU compute by prefetching
the next layer's experts during this layer's CMD2 + CMD3.

**Changes**:
1. **Buffer layout**: extend `MoeBuffers.data` to two sets:
   ```rust
   data_a: [MtlBuffer<u8>; MAX_K],
   data_b: [MtlBuffer<u8>; MAX_K],
   ```
   ~28 MB more memory on A3B (acceptable; no other resource is constrained).

2. **Per-layer prediction state**: on `RsCtx`:
   ```rust
   /// Last token's expert indices per layer, or None if no prediction yet.
   /// Indexed by layer_idx; inner Vec is the K indices used by that layer
   /// in the previous token.
   last_token_indices: Vec<Option<Vec<i32>>>,
   ```
   Length = `VARIANT.num_layers`. Reset in `memory_clear`.

3. **Prefetch state machine** on `RsCtx` (or `MoeBuffers`):
   ```rust
   pub struct PrefetchState {
       /// Layer that data_b is being / has been prefetched FOR.
       /// None if no prefetch in flight or buf is empty.
       target_layer: Option<usize>,
       /// Indices that data_b was loaded with (so layer N+1 can match).
       loaded_indices: Vec<i32>,
       /// Async handle / channel / JoinHandle — pool implementation
       /// detail.
       pool_handle: Option<...>,
   }
   ```

4. **Per-layer flow**, inside `post_attention_tail`:
   ```rust
   // Before MoE router: ensure any in-flight prefetch for THIS layer
   // has completed.
   prefetch_state.wait_for(layer_idx);

   // MoE router → actual indices for this layer.
   moe_router_cpu(...);
   let actual_indices: Vec<i32> = ...;

   // Decide source per slot.
   let mut expert_bufs: [&BufferRef; MAX_K] = ...;
   for slot in 0..k {
       if prefetch_state.hit(layer_idx, slot, actual_indices[slot]) {
           expert_bufs[slot] = moe.data_b_slot(slot);  // hit, no pread
       } else {
           // miss: sync pread into data_a[slot] (still parallel via the pool).
           expert_files.read_expert(layer_idx, actual_indices[slot] as usize,
                                    moe.data_a_slot_mut(slot))?;
           expert_bufs[slot] = moe.data_a_slot(slot);
       }
   }

   // Encode K-expert dispatch with the chosen buffers, commit deferred.
   gpu_batched_experts_begin_pre_staged_with_bufs(...);

   // Update prediction for next token's same layer.
   last_token_indices[layer_idx] = Some(actual_indices.clone());

   // Kick off prefetch for layer N+1, if not the last layer.
   if layer_idx + 1 < num_layers {
       let predicted = last_token_indices[layer_idx + 1].clone();
       if let Some(predicted_indices) = predicted {
           prefetch_state.start_async_prefetch(
               layer_idx + 1,
               predicted_indices,
               moe.data_b_slots_mut(),
           );
       }
   }
   ```

5. **Encoder change**: `gpu_batched_experts_encode_pre_staged` currently
   binds `bufs.data[slot]` for each expert. Slice 5d-6b needs to bind
   either `data_a[slot]` or `data_b[slot]` per expert based on which set
   holds that slot's data. Two options:

   - **(a) Per-slot buffer ref array**: pass `expert_bufs: &[&BufferRef; MAX_K]`
     to the encoder; bindings come from there. Cleanest API.
   - **(b) Per-slot index flag**: pass `data_set_per_slot: &[u8; MAX_K]`
     where 0 = set A, 1 = set B; encoder does the lookup. Simpler types but
     more state.

   I lean (a) — let the caller resolve which buffer; encoder just binds
   what it's told.

6. **Pool API**: needs to support
   - Submit a "prefetch this set of (layer_idx, expert_idx, dst_buffer)
     tasks" job that returns immediately.
   - Wait on a previously-submitted job.
   - Cancel an in-flight job (for `memory_clear`).

   Simplest implementation: `crossbeam::channel` MPMC + 4 worker threads
   spawned at `RsCtx::open`, joined on `Drop`. Or `rayon::ThreadPool::scope`
   per call. Or `tokio::task::spawn_blocking` × N (probably overkill).

   **Decision needed**: see §8 below.

7. **Edge cases**:
   - First token's first layer: no prediction (`last_token_indices[0] ==
     None`). Sync pread into data_a, no prefetch issued for layer 1 either
     (because layer 1's prediction is also None on token 0).
   - Subsequent tokens: predictions exist, prefetch can run.
   - `memory_clear` (`mod.rs:1043+`): drain any in-flight prefetch (to keep
     buffers GPU-quiescent), then null out `last_token_indices`.
   - `state_save` / `state_load`: ignore prefetch state. Drain on save (the
     deferred-experts state is already drained per `state_snapshot.rs`); no
     prefetch state to serialize.

**Expected perf**: with prediction hit-rate >50% and overlap working, the
pread cost (~70 ms/token at current rates) collapses to near-zero on the
critical path. Estimated 5.4 → 7-8 tok/s warm, more if GPU saturates.

**Risks**:
- Race: GPU dispatch reads `data_a[slot]` while pool writes `data_b[slot]`
  for next layer. Different sets ⇒ no race. (This is the load-bearing
  design choice.)
- Race within `data_b`: prefetch for layer N+1 writes `data_b[slot]`; if
  layer N+1's HIT path then reads `data_b[slot]` from GPU dispatch, the
  prefetch must have completed. Wait at top of layer N+1 enforces this.
- Race within `data_a`: layer N's miss path writes `data_a[slot]` while
  layer N-1's GPU dispatch (reading `data_a[slot]` from N-1's pread) is
  in flight. Solved by the existing wait-at-top-of-layer drain
  (`complete_deferred_experts_into`) which waits for layer N-1's dispatch
  before layer N starts touching `data_a`.

---

## 6. Diff test strategy

### Existing tests to keep green
- `eval_token_matches_c_single_step` — end-to-end against C. Must stay green
  at `cosine ≥ 0.9999` (probably bit-exact if dispatch order is preserved).
- `eval_prompt_matches_c_multi_token` — multi-token, ditto.
- `layer_forward_dump_close_c_vs_rust*` (3 variants) — per-layer comparison.
  These bracket forward calls with discard/complete; prefetch needs to be
  drained at the brackets.
- `state_round_trip_rust`, `deferred_experts_begin_complete_close_c_vs_rust`
  — should be unaffected if prefetch is opt-in or correctly drained.

### New tests
- **Slot-reuse race regression**: two consecutive `step_internal` calls,
  forcing 100% prediction misses (e.g. by clearing `last_token_indices`
  between them) and verifying outputs match a fresh-Ctx baseline. Catches
  the case where stale `data_a` from layer N's serial read gets reused
  for layer N+1's dispatch incorrectly.
- **Prefetch hit/miss equivalence**: same prompt, two runs. Run A: clear
  `last_token_indices` between layers (forces all misses). Run B: normal
  flow (prefetch hits where applicable). Outputs must agree at the existing
  cosine floor.
- **Memory-clear cancels prefetch**: `step` → `memory_clear` → `step`
  produces identical output to fresh-Ctx `step` → `step`. Catches any
  leaked prefetch state across a clear.

### Determinism note
Async prefetch doesn't introduce nondeterminism if we **drain** before
GPU dispatch — the dispatch order is the same. The only non-determinism
window is "did we get the prefetch done in time to be a hit?" which only
affects PERF, not correctness (miss path is still correct, just slower).

For diff testing against C, set the same prompt + same seed; both backends
will produce the same expert indices per layer per token; both will use
their own prefetch logic; outputs match because the dispatched arithmetic
is identical regardless of which buffer the bytes came from.

---

## 7. Sequencing / commit plan

Two commits to land 5d-6:

**Commit A — slice 5d-6a (parallel sync pread)**:
- Add threading dep (rayon OR std::thread::scope helper).
- Build 4-worker pool; lazy-init on `RsCtx::open` (or process-global).
- Replace serial pread loop in `post_attention_tail` with parallel dispatch.
- Run diff tests, run blallama perf bench, commit.

**Commit B — slice 5d-6b (double-buffer + prefetch)**:
- Add `data_b: [MtlBuffer<u8>; MAX_K]` to `MoeBuffers`.
- Add `last_token_indices: Vec<Option<Vec<i32>>>` to `RsCtx`.
- Add `PrefetchState` and async-job interface.
- Update `post_attention_tail` to use prefetch logic.
- Update `memory_clear` to invalidate prefetch.
- Add new diff tests for prefetch correctness.
- Run full diff oracle suite, run blallama perf bench, commit.

If 5d-6a's perf is strong enough that we don't need 5d-6b for the use case,
we can stop after A and revisit B later. (Unlikely — the bigger win is the
overlap, not the parallelism, but worth measuring.)

---

## 8. Open decisions for next-session plan-mode

These are the design choices I'd surface for sign-off when we enter plan
mode:

1. **Threading primitive**: `rayon` is the lean. The K-expert pread is
   exactly a `par_iter().for_each()` shape, and rayon's scheduler avoids
   the SPMC channel question entirely (`std::sync::mpsc` is
   multi-producer-SINGLE-consumer; for a true worker pool you'd need
   `Arc<Mutex<Receiver>>` — kills throughput — or `crossbeam-channel` for
   real MPMC). `std::thread::scope` works too if we keep the scope
   tight, but rayon reads cleaner for the map shape. Mike sign-off
   confirmed (2026-04-28).
   - For 5d-6a: `rayon::scope` or `par_iter` over the K disjoint slots.
   - For 5d-6b: same pool reused for async prefetch; rayon doesn't have
     a fire-and-forget API directly but `rayon::spawn` does, with the
     return-value via a oneshot channel or atomic flag.

2. **2 MB pread alignment**: C says it's 3.6× faster. Worth implementing
   in 5d-6a or defer? My lean: do it in 5d-6a since allocation alignment
   is cheap to add and the comment claims it's a big factor.

3. **Prefetch state location**: on `RsCtx` directly, on `MoeBuffers`, or
   a new `PrefetchState` struct that owns the data_b half of MoeBuffers?
   My lean: new struct, owns data_b + last_token_indices + pool_handle —
   keeps `MoeBuffers` focused on per-call buffers.

4. **Encoder API**: pass `[&BufferRef; MAX_K]` per slot vs a `[bool; MAX_K]`
   selecting set A/B. My lean: pass refs (caller resolves; encoder is
   dumb).

5. **First-layer-of-first-token handling**: separate code path or just let
   it fall through the miss path naturally? My lean: let it fall through.
   The cost is one extra serial pread on token 0 only.

6. **Cancellation on memory_clear**: drain (wait for in-flight prefetch to
   complete, discard results) vs cancel (try to abort the pread). My lean:
   drain — pread is fast, and cancellation across pthread+pread is fiddly.

7. **Should we skip 5d-6a** and go straight to 5d-6b? Argument for skip:
   the parallelism win is small if reads hit page cache; the overlap win
   is the big one. Argument against: 5d-6a is ~1 hour of work and gives a
   measurable number that helps us evaluate 5d-6b's increment cleanly.

   My lean: do 5d-6a — the data point is worth the hour, and 5d-6a is a
   clean prerequisite for 5d-6b's pool reuse.

---

## 9. References

- C scout report: this file's §3, generated by the morning Explore agent
  reading `metal_infer/infer.m`.
- Riir code as of commit `de47fa3` (slice 5d-5).
- Yesterday's perf log: see `riir_moeflux_strategy.md` § Phase 5 progress
  + blallama perf log.
- `metal_infer/infer.m` line refs:
  - Pool: 3149-3234
  - Async dispatch: 3240-3282
  - Buffer layout: 991-1000
  - Allocation alignment: 1196-1211
  - Per-layer flow: 5304-5605
  - Prediction prefetch: 4460-4470, 5510-5562, 5583-5601

---

## 10. What success looks like

After 5d-6 lands:
- Activity Monitor shows GPU utilization closer to C path's ~40%.
- Profile shows `pread` self-time drops from ~33% to <10%.
- blallama A3B essay perf in the 7-9 tok/s range warm (target: parity with
  C's 8.7).
- 7/7+ critical diff tests still green at the existing tight floors.
- New prefetch-correctness diff tests pass.
