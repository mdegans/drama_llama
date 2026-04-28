---
name: Post-5d-8 samply profile findings (a3b warm)
description: Fresh self-time + inclusive profile of a3b warm decode after 5d-8 lands. Replaces the stale post-5d-1 profile that drove 5d-7 / 5d-8 (both flat). Identifies the actual remaining levers.
type: project
---

# Post-5d-8 samply profile (a3b warm, 2026-04-28)

**Captured**: 2026-04-28, after 5d-8 commit `0b20e20`. Apollo essay
× 2 sequential @ 512 max_tokens, --seed 42, M2 Max. Warm run gave
7.09 tok/s (within noise of the 7.25 logged in the strategy doc).
samply spawn-mode (not attach), 1000 Hz, full debug + dSYM, 138k
samples on the main worker thread.

## Main thread (tokio-rt-worker, 138,294 samples)

### Self-time by library

| % | Library | Notes |
|---|---|---|
| 77.4% | `libsystem_kernel.dylib` | syscall-bound: GPU completion wait + pread + mach_msg |
| 18.9% | `blallama` | application code |
| 1.1% | `libsystem_m.dylib` | math (exp / log) |
| 1.0% | `AGXMetalG14X` | GPU driver dispatch |
| 0.7% | other system | malloc, objc, dispatch |

### Self-time hot spots (resolved)

| % | Symbol | File:Line | Notes |
|---|---|---|---|
| ~76% | kernel syscall (likely `__psynch_cvwait`) | — | GPU cmdbuf wait |
| **16.0% inclusive** | `partial_sort::partial_sort` | partial_sort/src/lib.rs:60–69 | **drama_llama sampling chain** |
| 0.6% | DYLD-STUB$$log | — | log() — softmax / rep penalty |

`partial_sort` is reached via the drama_llama sampling chain (called
multiple times per token across the min-p / top-k / top-p / mirostat
filters). At ~16% of wall time and ~7 tok/s = ~22 ms/token of sampling
on a 248,320-vocab model.

### Inclusive call stack (top of main thread)

```
99.7%  moeflux::riir::RsCtx::step_internal              mod.rs:1159
99.7%  moeflux::riir::deferred::complete_deferred_experts_chained
                                                        deferred.rs:303
35.6%  moeflux::riir::linear_attn_forward::post_attention_tail
                                                        linear_attn_forward.rs:1020
21.7%  (full_attn or other layer-forward branches)
```

**Reading**: 99.7% of main-thread time is spent inside `step_internal`,
of which the bulk is the `wait_until_completed` inside
`complete_deferred_experts_chained` — i.e. waiting for the K-expert
cmdbuf to finish so the next layer can start. The 5d-8 chain
removed the host roundtrip but did NOT remove this wait — it's
load-bearing because the next layer's prefetch slot reuses
`bufs.data_prefetch[slot]` which the previous layer's GPU dispatch
is still reading.

## moeflux-io workers (8 threads, ~110k samples each)

| % | Activity | Notes |
|---|---|---|
| 75.9% | `rayon_core::sleep::Sleep::sleep` | workers idle |
| 17.5% | `File::read_at` (pread) | actual disk I/O |
| 4.3% | `WorkerThread::wait_until_cold` | work-steal idle |
| ~2% | other | locking, wakeups |

**Reading**: io workers are mostly idle (75% asleep). The speculative
prefetch (5d-6b) is keeping up — pread is NOT the bottleneck on a3b
post-5d-6. The post-5d-1 profile's 30.6% pread cost has been
substantially absorbed by the prefetch overlap.

## Helper threads

- **Thread 77677354** (98k samples, 94.5% kernel): Metal command-
  buffer completion thread. Mostly waiting in `mach_msg_trap` for
  GPU events. Diagnostic only.
- **Thread 77682522** (44k samples, late-starter): likely the GPU
  LM-head dispatcher.
- **Thread 77677353** (62k samples): tokio worker; mostly idle.

## What actually limits a3b warm tok/s

Three signals, in order:

1. **Inter-layer GPU-wait dominates** (76% of main-thread self-time):
   the CPU is gated on `wait_until_completed` after each K-expert
   dispatch. The chain (5d-8) eliminated the host roundtrip but the
   serial wait remains — load-bearing because the prefetch slot is
   reused.

2. **Sampling is bigger than expected** (16% of wall time): the
   `partial_sort` chain runs 4–6 times per token over 248k logits.
   Independent of moeflux; this is drama_llama-side.

3. **pread is no longer the bottleneck**: io workers 75% idle.
   Sequential bf16 / quant cost is also negligible on the main
   thread (libsystem_m at 1.1%).

Per-head Q/K rms_norm + RoPE doesn't appear as a hot blallama
self-time address. It's CPU work but small enough to be drowned by
the GPU wait. Porting it to GPU would shave a fraction of the
non-wait time but won't move the floor — the wait is the floor.

## Implication for next slice

The plan-doc-suggested "GPU per-head Q/K rms_norm + RoPE fusion"
was wrong-headed. Two real candidates:

### A. Eliminate the inter-layer wait

Two-set the prefetch buffer so the next layer's prefetch can fire
WITHOUT waiting for the previous layer's GPU read of the same slot.
Then `complete_deferred_experts_chained` doesn't need to be a hard
sync point — it can drop to a deferred-mark, and layer N+1's CMD1
commits while N's K-expert cmdbuf is still running. Metal serializes
cmdbufs on the queue, so correctness holds.

Memory cost: ~7 MB extra per slot × 16 slots × 2 sets = ~112 MB
on a3b (or just ping-pong the WHOLE bufs.data_prefetch — ~115 MB).
Cheap relative to the ~16 GB of total moeflux memory.

Estimated upper bound: if GPU is 40-50% util, full overlap could
hit ~10-12 tok/s warm on a3b — closing most of the gap to C and
then some.

### B. Shrink the sampling cost

The drama_llama sampling chain runs partial_sort multiple times
over 248k floats. Options:
- Cap the chain at one canonical sort (caller-provided top-K),
  caching the sorted view across filters.
- SIMD partial_sort (NEON) for the inner loop.
- Use a heap-based selection rather than partial_sort for the
  top-K-of-248k case.

Estimated: cutting sampling in half = ~8% wall-time gain.

### Verdict

Slice **A** (overlap layer N+1 CMD1 with layer N K-expert) is the
big lever — it attacks the largest contributor (76% of self-time)
and has a clean path that doesn't require GPU kernel work. Slice
**B** is a smaller follow-up but lives in drama_llama, not moeflux,
so it's easy to schedule independently.

Skip the GPU per-head Q/K rms_norm + RoPE port for now. The data
says it won't move the needle.

## Methodology note

samply was run in spawn mode (not attach) because attach requires
`samply setup` to grant macOS task_for_pid entitlements. Spawn mode
records the entire process lifetime; warmup + profiled run both
land in one trace. Future profiles: attach mode after `samply
setup` is once-and-done.

dSYM bundle was needed for symbolication. CARGO_PROFILE_RELEASE_DEBUG
defaults to nothing on cargo's release profile; explicit
`CARGO_PROFILE_RELEASE_DEBUG=true` env var on the build, then
`dsymutil target/release/blallama` to package, then samply picks up
the dSYM automatically. `line-tables-only` does NOT produce enough
DWARF for atos symbolication.

To re-symbolicate raw addresses from a samply trace:
`xcrun atos -o target/release/blallama -arch arm64 -l 0x100000000 -- <hex addr>`.
The `-l 0x100000000` is the standard ARM64 Mach-O image base.
