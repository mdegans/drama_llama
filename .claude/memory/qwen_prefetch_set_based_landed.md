# Qwen prefetch — set-based matching landed (2026-05-01)

Outcome memo for the prefetch / hit-rate session.

## What landed

### Instrumentation (commits 8a76eb1 moeflux + 602fe04 drama_llama)
- `PrefetchState` got `AtomicU64` hits/misses + `record_outcome` /
  `stats` / `reset_stats` methods
  (`moeflux/.../prefetch.rs`).
- `RsCtx::prefetch_stats()` accessor; `MoefluxDecoder` forwards;
  `Session::<MoefluxBackend>::prefetch_stats()` exposes to blallama.
- `bin/blallama/blallama.rs` resets at request start, reads after
  `spawn_blocking` returns, emits a per-request `moeflux_prefetch`
  tracing event alongside the existing `stats` event.

### Per-phase split (commit d59aa93 drama_llama)
- `MoefluxDecoder` snapshots the underlying counter before/after
  each `prefill` / `step` call, folds delta into separate
  prefill_/decode_ counters.
- `Session::prefetch_stats()` returns
  `PrefetchStats { prefill_hits, prefill_misses, decode_hits,
  decode_misses }`. blallama logs both.

### Set-based matching (commit f86fd93 moeflux)
- `SlotSource::Prefetched` → `SlotSource::Prefetched(usize)`
  carrying the prefetch buffer index where the actual expert
  landed.
- `linear_attn_forward.rs::post_attention_tail` now scans the
  prefetched indices for each actual expert (set-based) instead of
  comparing slot[i] to slot[i] (position-locked).
- `expert_forward.rs::emit_batched_experts::pick(slot)` reads
  `data_prefetch[set][buf_idx]` instead of `[slot]`. Buffer pool
  size unchanged.

## The headline number

| Metric | Pre-session | Post-session |
|---|---|---|
| Prefill hit rate | 5.2% | 34.1% |
| Decode hit rate | 5.2% | 36.1% |
| Tok/s on Hello/26-tok | 0.90 (samply) | 1.77 (samply) ≈ 1.78 (clean) |

The 5.2% headline was almost entirely a position-locking artifact;
the **set** hit rate was always ~36%, but slot-position drift across
tokens hid most matches. Removing the lock unlocked the prefetch
machinery that was already there.

## Profile shift (samply)

The set-based matching shifted the warm-decode profile from
IO-bound to compute-bound:

- IO worker threads dropped from ~44K samples each to ~10K (−77%).
- Main-thread `rayon::LockLatch::wait` (the prefetch-miss fallback
  pole) dropped from 32.7% inclusive to 21.3%.
- Main-thread `MTLCommandBuffer::waitUntilCompleted` rose from
  57.5% inclusive to 60.2% — now the dominant pole.
- GPU utilization went from 40-45% to 60% (Mike's observation).

## On the perceived "regression vs 1.96"

The 5d-9 baseline of 1.96 tok/s on Qwen3.5-A17B (max_tokens=512,
measured 2026-04-28 17:41 at moeflux 914db09 + drama_llama
f97d238) was *not* reproducible on this machine on 2026-05-01.

Diagnostic ladder:
1. Fresh post-set-based-matching benchmark at max_tokens=512:
   **1.78 tok/s.**
2. Suspected F_RDAHEAD=0 (added unconditionally in cogito-v2 work
   at moeflux 43f6535) as the regression vector. Gated to
   cogito-only (commit 612ef3c). A/B re-bench: **1.78 → 1.78**.
   Not the cause.
3. Worktree bisect: built old code (moeflux 914db09 + drama_llama
   f97d238) on the *current machine state*. Bench: **1.7535
   tok/s.** Slightly *slower* than current code.

Conclusion: **the 1.96 → 1.78 gap is machine state, not code.**
Long uptime (Mike noted "haven't rebooted in a very long time")
plausibly degrades page cache / mmap throughput. Reboot before any
A/B comparison going forward.

Anecdotally Mike has seen tok/s swing as high as 2.5 in the past
under similar conditions, consistent with cache state being a
significant factor on this hardware.

## New tools

- `~/Projects/drama_llama/bench.py` (gitignored) — reproducible
  tok/s benchmark with pinned prompt/seed/max_tokens. Supports
  `--binary path/to/blallama` for benching out-of-tree builds (e.g.
  worktrees), `-n N` for mean/stdev across N runs, prints uptime
  alongside results so the bench log carries machine-state
  context. **Reboot before important benches.**
- `~/Projects/drama_llama/profile.py` (already existed; this session
  extended its samply aggregator to resolve symbols against
  *all* libs, not just the bin — so `<?>0x450c` style stubs now
  resolve to e.g. `libsystem_kernel.dylib:pread`).

## What's deferred

- **Overprovision (last-2-token union prediction + 2× prefetch
  buffer pool)**: was the original lever #1 plan. Skipped because
  set-based matching alone moved us from IO-bound to compute-bound;
  IO slack exists but won't translate to tok/s gains until the GPU
  pole shrinks. Revisit if cmdbuf consolidation lands and IO
  becomes the pole again.

## Calibration note for next session

- Use **max_tokens=512** with the deterministic essay prompt for
  tok/s comparisons. That matches the historical baseline and
  bench.py's defaults.
- **Reboot first.** Page-cache state is a real source of variance.
- F_RDAHEAD=0 is now cogito-gated. If we run cogito and want to
  verify the gate matters there too, A/B by toggling the cfg —
  the C path applied it unconditionally but the rationale was
  speculative even there.
