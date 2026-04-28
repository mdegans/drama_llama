---
name: Plan — slice 5d-9 overlap layer N+1 CMD1 with layer N K-expert
description: Pre-implementation plan for the next-session slice. Eliminates the inter-layer wait that the post-5d-8 profile identified as the dominant time sink (76% of main-thread self-time). Two-set prefetch buffer + drop the hard sync at the layer boundary.
type: project
---

# Slice 5d-9: overlap layer N+1 CMD1 with layer N K-expert

**Status**: planning, not started.
**Authored**: 2026-04-28, end of 5d-8 session, immediately after the
post-5d-8 samply profile (`profile_post_5d8.md`).
**Branch when implementing**: `riir` in `~/Projects/moeflux`, base
commit `0b20e20` (5d-8 lands).
**Diff baseline**: cold 7.38 / warm 7.25 tok/s (post-5d-8).
**Target**: 9–10 warm tok/s if GPU-idle gap closes; matches or
exceeds C's 8.70 warm.

---

## 1. Why this slice

The post-5d-8 profile (see `profile_post_5d8.md`) is unambiguous:
**76% of main-thread self-time is in `__psynch_cvwait`** (the GPU
completion syscall), 99.7% inclusive in
`complete_deferred_experts_chained → wait_until_completed`. The
chain (5d-8) eliminated the host roundtrip but the serial wait
itself is still on the critical path.

The wait is load-bearing because of one specific dependency: layer
N+1's prefetch dispatch reuses `bufs.data_prefetch[slot]`, and the
previous layer's GPU read of that same slot must complete before
the prefetch overwrites it.

If we two-set the prefetch buffer so layer N+1's prefetch goes to
a DIFFERENT slot than layer N is reading from, the wait becomes
unnecessary. The CPU can submit layer N+1's CMD1 while the GPU is
still executing layer N's K-expert cmdbuf. Metal serializes
cmdbufs on the same queue in commit order, so correctness holds —
we just stop blocking the CPU thread.

GPU at ~40-50% util means there's roughly equal headroom; full
overlap could close most of the gap to C (8.70) and beyond.

**Bias check**: if 5d-9 lands flat on warm, the GPU-wait pattern is
not what the syscall implies (e.g. there's a smaller per-cmdbuf
floor we can't drop below), and the only remaining lever is sampling
(16%, drama_llama-side, separate slice).

---

## 2. Goal and scope

### In scope

- Add a second set of `data_prefetch` buffers to `MoeBuffers`
  (call them `data_prefetch_a` and `data_prefetch_b`, ping-ponged
  by layer parity).
- Modify the prefetch dispatch in `step_internal` to write into the
  alternate set than the one this layer is reading from.
- Drop the hard wait at the layer boundary: instead of calling
  `complete_deferred_experts_chained` synchronously, defer the
  wait until the deferred slot is needed (which won't be until the
  same layer's K-expert dispatch fires next token, by which point
  it'll have completed naturally).
- Verify Metal queue serialization handles the inter-cmdbuf
  dependency: layer N+1's CMD1 reads `linear_buffers.normed`, which
  is written by layer N's appended chain (5d-8). They're on the same
  command queue → ordered by commit time → safe.

### Out of scope

- The CPU-combine path (`gpu_combine = false`): chain doesn't apply,
  no overlap to attempt.
- Layer 0: no previous layer, no overlap. Still uses initial wait.
- Sampling speedup: separate slice (5d-10 candidate, drama_llama-
  side).
- Per-head Q/K rms_norm + RoPE GPU port: skip — profile says it
  won't move the needle.

### Explicit non-goal

Don't break the diff suite. The chain itself is unchanged; only the
schedule of CPU-side cmdbuf commits changes. Bit-exact correctness
must hold (eval_token / eval_prompt cosine 1.0000000).

---

## 3. Riir-side current state (post-5d-8)

### Where the wait happens

`step_internal` in `mod.rs:~1057..1156` (post-5d-8 line numbers):

```rust
for layer_idx in 0..v.num_layers {
    if layer_idx > 0 {
        if prev_layer_chained {
            deferred::complete_deferred_experts_chained(deferred)?;
            // ↑ THIS IS THE WAIT
        } else {
            deferred::complete_deferred_experts_into(deferred, ...)?;
        }
    }
    // ... prefetch dispatch ...
    // ... layer forward ...
}
```

`complete_deferred_experts_chained` (in `deferred.rs:288..298`) is:

```rust
pub(crate) fn complete_deferred_experts_chained(
    slot: &mut Option<DeferredState>,
) -> Result<(), DeferredError> {
    let Some(state) = slot.take() else { return Ok(()); };
    state.cmd_buffer.wait_until_completed();
    Ok(())
}
```

The wait is the only thing this function does (other than clearing
the slot). Removing it means the in-flight cmdbuf stays "live" in
`*slot` across layer iterations.

### Why we wait today

Three reasons, in order of severity:

1. **Prefetch slot reuse** (load-bearing): `bufs.data_prefetch[slot]`
   is read by layer N's K-expert dispatch and overwritten by layer
   N+1's prefetch. Without the wait, the prefetch can race the GPU
   read. **5d-9 fixes this via two-set ping-pong.**

2. **Single deferred slot**: `RsCtx::deferred: Option<DeferredState>`
   is a single slot; calling `gpu_batched_experts_begin_pre_staged`
   when it's `Some` errors with `AlreadyActive`. **5d-9 fixes by
   either (a) widening to a 2-deep ring, or (b) waiting at the
   "begin" call site only when actually needed for buffer reuse.**

3. **`linear_buffers` contents at CPU read points**: in particular
   the `discard_deferred_experts_in` call at start of `step_internal`,
   and various `to_vec()` calls inside `post_attention_tail`. We
   need to verify nothing on the CPU reads from a buffer that the
   GPU is concurrently writing.

### Buffers the chain touches

- `linear_buffers.input` (combine output target, chained)
- `linear_buffers.sum_sq` (chain scratch)
- `linear_buffers.normed` (chain output, next layer's CMD1 input)
- `bufs.data_prefetch[slot]` (read by K-expert dispatch)

`linear_buffers.input/sum_sq/normed` are read ONLY by GPU
dispatches in subsequent cmdbufs on the same queue → in-order →
safe without explicit wait. **`bufs.data_prefetch[slot]` is the
only buffer whose reuse would race the GPU.**

---

## 4. Proposed design

### A. Two-set `data_prefetch` on `MoeBuffers`

In `expert_forward.rs`:

```rust
pub struct MoeBuffers {
    // existing:
    pub data_synced: [MtlBuffer<u8>; MAX_K],
    // change:
    pub data_prefetch_a: [MtlBuffer<u8>; MAX_K],  // was data_prefetch
    pub data_prefetch_b: [MtlBuffer<u8>; MAX_K],  // new
    // ...
}
```

Memory cost: another K × EXPERT_SIZE = ~7 MB on a3b, ~15 MB on
a17b. Negligible.

Add a `prefetch_set: PrefetchSet` field somewhere (could live on
`PrefetchState` or `MoeBuffers`) that flips A↔B per layer.

### B. SlotSource enum extension

`SlotSource` (in `prefetch.rs`) gains a third variant:

```rust
pub enum SlotSource {
    Synced,        // bufs.data_synced[slot] — sync-pread
    PrefetchedA,   // bufs.data_prefetch_a[slot] — prefetch set A
    PrefetchedB,   // bufs.data_prefetch_b[slot] — prefetch set B
}
```

The encoder's `data_set_per_slot: &[SlotSource; MAX_K]` parameter
plumbs through unchanged in shape. The encoder's `pick(slot)`
helper grows a third branch.

### C. PrefetchState ping-pong

`PrefetchState::dispatch(layer_idx, ...)` writes to set A on even
layers, set B on odd layers (or whichever convention). The
corresponding GPU read in layer N's K-expert dispatch reads from
the SAME set the prefetch wrote to (which is the set assigned to
layer N's parity — already in flight from when layer N-2 fired).

Wait, that's wrong. Let me re-think.

Each layer's K-expert dispatch reads from a fixed prefetch slot.
That slot was written by THIS layer's earlier prefetch dispatch
(fired at the top of `step_internal`'s iteration for this layer).

Today (single-set):
- Layer N iteration top: prefetch fires for layer N (writes
  `data_prefetch[slot]`).
- Layer N's K-expert dispatch reads `data_prefetch[slot]`.
- Layer N+1 iteration top: prefetch fires for layer N+1 (writes
  same `data_prefetch[slot]`). RACE if layer N's K-expert isn't
  finished.
- Mitigation today: wait at top-of-N+1 before the prefetch fires.

Two-set:
- Layer N iteration top: prefetch fires for layer N → writes set
  ((N % 2)==0 ? A : B)`[slot]`.
- Layer N's K-expert reads set ((N % 2)==0 ? A : B)`[slot]`.
- Layer N+1 iteration top: prefetch fires for layer N+1 → writes
  set ((N+1) % 2 == 0 ? A : B)`[slot]` = the OTHER set. NO RACE
  with layer N because different physical buffers.
- Layer N+2 iteration top: prefetch fires → writes set A again.
  Race with layer N's READ? Only if layer N's K-expert hasn't
  finished by the time layer N+2's prefetch starts WRITING. That's
  ~2 layer durations — should be enough.

But for safety we should still wait on layer N's deferred BEFORE
layer N+2's prefetch dispatch (i.e. one-layer pipeline depth, not
two). That brings us back to a wait, just delayed by one layer.

Cleaner: keep two-set, but wait on layer N's deferred at top of
layer N+1's iteration (not before kicking N+1's prefetch — AFTER).
Order:
1. Layer N+1 top: kick layer N+1's prefetch (writes the OTHER set,
   no race with layer N's READ because different buffer).
2. Layer N+1: drain layer N's deferred (cheap if it finished while
   we were prefetching; otherwise short wait).
3. Layer N+1 forward.

The drain at step 2 is still a wait, but now it overlaps with the
prefetch at step 1. The TIMING win is bounded by min(prefetch
time, GPU layer-N residual time). prefetch is fast (KB-MB pread on
warm pages); layer-N residual is the back half of the K-expert
cmdbuf ≈ a few ms. Overlap saves a few ms/layer × 60 = a few hundred
ms/token if it works.

**Better still**: skip the explicit wait entirely. Submit layer
N+1's CMD1 + CMD2 + chain N+1's K-expert WITHOUT waiting for
layer N. Metal queue serialization guarantees they run in order.
The wait collapses to "deferred-state cleanup" = no-op against
correctness; just need to retain the cmdbuf reference somewhere
until it completes (so we don't drop it).

Design: change `DeferredState` to be a small ring buffer (depth 2)
rather than a single Option. Layer N+1's begin doesn't error; it
pushes onto the ring. The drain happens lazily — when the ring is
full, the oldest is dropped (after `wait_until_completed`).

### D. `step_internal` orchestration

```rust
let mut deferred_ring: VecDeque<DeferredState> = VecDeque::with_capacity(2);
for layer_idx in 0..v.num_layers {
    // Drain any deferred state that's >= 1 layer old (i.e. layer
    // N-2's). At depth 2, we always have at most one in-flight
    // dispatch when entering a layer.
    while deferred_ring.len() >= 2 {
        let oldest = deferred_ring.pop_front().unwrap();
        oldest.cmd_buffer.wait_until_completed();
        // (no readback for chained mode; in unchained mode would
        // need a host slice — but the unchained path fires on
        // last layer only, post-loop)
    }

    // Prefetch THIS layer's expected K experts into the alternate
    // set (the one layer N-1 didn't write to).
    let prefetch_set = if layer_idx % 2 == 0 { SetA } else { SetB };
    if let Some(predicted) = prefetch.predict_for(layer_idx) {
        prefetch.dispatch(layer_idx, predicted, k_active,
            moe_buffers.prefetch_slots_mut_for_set(prefetch_set),
            io_pool, experts);
    }

    // Layer forward — chain enabled for non-last; reads from
    // prefetch set assigned to THIS layer.
    let chain_next = layer_idx + 1 < v.num_layers;
    let chain_next_norm_off = chain_next.then(|| ...);
    let data_set_per_slot = build_data_set_per_slot(layer_idx, &actuals);

    layer_forward(..., data_set_per_slot, prev_layer_chained,
                  chain_next_norm_off);
    // The K-expert begin_pre_staged inside layer_forward pushes
    // its DeferredState onto the ring instead of writing the
    // single Option.

    prev_layer_chained = chain_next;
}

// Post-loop: drain anything left.
while let Some(state) = deferred_ring.pop_front() {
    state.cmd_buffer.wait_until_completed();
}

// Final layer's hidden state needs a real readback for the LM head
// path — handle separately (last layer doesn't chain; combine wrote
// to bufs.moe_hidden, readback from there as today).
```

### E. Buffer-set wiring through encoder

Encoder's `data_set_per_slot: &[SlotSource; MAX_K]` is per-slot
(supports the existing prefetch hit/miss mix). 5d-9 just adds a new
`SlotSource::PrefetchedB` variant. The pick() function:

```rust
let pick = |slot: usize| -> &MtlBuffer<u8> {
    match data_set_per_slot[slot] {
        SlotSource::Synced => &bufs.data_synced[slot],
        SlotSource::PrefetchedA => &bufs.data_prefetch_a[slot],
        SlotSource::PrefetchedB => &bufs.data_prefetch_b[slot],
    }
};
```

---

## 5. Diff test strategy

### Existing tests must stay green

- `eval_token_matches_c_single_step` — exercises every layer's
  chain, validates pipelined execution doesn't break ordering.
- `eval_prompt_matches_c_multi_token` — multi-token; would catch
  state pollution across tokens if the ring isn't cleaned between.
- `slot_reuse_race_regression_rust` — directly tests prefetch slot
  reuse correctness; will exercise the new ping-pong path.
- `state_round_trip_rust` / `state_load_*` — chain doesn't touch
  wire format; should be unaffected.
- All 4 attn-kernel diff tests — unaffected.
- `cpu_combine_path_matches_c` — chain disabled; ring depth still
  needs to handle the unchained path.

### New regression risk

If the ring depth-2 invariant holds, no new race is introduced.
Risk concentrates on:
- Last-layer post-loop drain (which buffer is canonical?)
- `memory_clear` / `state_save` / `state_load` — these need to
  drain the ring fully (they currently call
  `discard_deferred_experts_in` on the single Option).
- `layer_forward_dump_inner` — single-layer test path; ring should
  hold at most one state, must drain on exit.

Add at minimum: `pipelined_layer_dispatch_no_race` test that runs
50 tokens back-to-back, comparing per-token logits to a
single-layer-at-a-time reference.

---

## 6. Sequencing / commit plan

Single commit. Phasing introduces dead state.

```
riir: Phase 5d-9 — overlap layer N+1 CMD1 with layer N K-expert

Drop the inter-layer wait. Two-set the data_prefetch buffer
(ping-pong by layer parity) so layer N+1's prefetch can fire
without racing layer N's GPU read. DeferredState upgrades to a
depth-2 ring; layer N's begin_pre_staged pushes onto the ring
without erroring on AlreadyActive. The wait collapses into
ring-cleanup: drain the oldest only when the ring is full.

Metal queue serialization guarantees layer N+1's CMD1 reads
linear_buffers.normed AFTER layer N's chain wrote it (same queue,
commit-order ordering). No explicit synchronization needed.

post-5d-8 profile: 76% of main-thread self-time was in cvwait
inside complete_deferred_experts_chained → wait_until_completed.
This slice attacks that directly.

Diff suite green. blallama A3B essay perf: cold X.XX / warm Y.YY
tok/s (was 7.38 / 7.25; target ~9-10 warm).
```

If perf comes in flat: the GPU-wait was a measurement artifact
(samply samples inside the syscall during NORMAL pipelining waits
that aren't on the critical path). The next move is sampling
speedup (drama_llama-side, slice 5d-10).

---

## 7. Open decisions for next-session plan-mode

1. **Ring depth**: 2 (layer N-1 in flight while N forwards) or
   higher? Higher = more pipeline = more memory + more risk.
   **Lean: 2.** Matches the natural prefetch ping-pong depth.

2. **Where the ring lives**: `RsCtx.deferred: VecDeque<DeferredState>`
   (replacing the Option) or a dedicated `DeferredRing` type?
   **Lean: rename + replace.** Type the depth invariant.

3. **`AlreadyActive` error**: keep as runtime guard or remove?
   **Lean: keep, but only fires if ring is at capacity.** Same
   semantic as today, just different threshold.

4. **post-loop final drain**: explicit `while let Some` loop or a
   `Drop` impl? **Lean: explicit in `step_internal`.** Drop runs
   on `RsCtx` shutdown, not inter-token; we want the wait every
   token to return clean state.

5. **Memory budget**: 7-15 MB per extra set is fine on a3b/a17b.
   What about the 397B / 1.6T variants? K=8 × ~7 MB × 2 sets ≈
   112 MB extra. Still fine.

6. **`data_synced` set**: is that affected? No — `data_synced` is
   the sync-pread fallback path, never racing with GPU. Single
   set retained.

---

## 8. References

- post-5d-8 profile: `profile_post_5d8.md` (this directory).
- 5d-6b prefetch landing (single-set): commit `f9d8ff3`.
- 5d-8 chain landing: commit `0b20e20`.
- C-side analogue (worth scouting before implementing): does C use
  cmdbuf pipelining? `infer.m:5747..5776` is the deferred-state
  shape on C; check if it serializes or pipelines across layers.

---

## 9. What success looks like

After 5d-9 lands:
- Diff suite (40+ tests) green at the existing tight floors.
- blallama A3B essay perf in 9-10 warm tok/s range — closes the
  gap to C (8.70) AND beats it.
- Activity Monitor: GPU utilization climbs from ~50% toward 80-90%
  (the proof that overlap is working).
- Re-profile shows main-thread cvwait drops substantially; the
  remaining hot spot becomes either sampling or new compute.

If warm lands flat (~7.25): the cvwait was structural (Metal
per-cmdbuf submission overhead, not waitable-away), and the
sampling slice is the only remaining lever before structural
rework. Commit anyway, update perf log, move to sampling.
