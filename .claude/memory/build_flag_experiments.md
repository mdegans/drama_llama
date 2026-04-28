---
name: Build-flag experiments (LTO / codegen-units / PGO)
description: Findings on aggressive release-profile flags for drama_llama. Negative results are durable — re-trying without new motivation is wasted time.
type: project
---

# LTO / codegen-units / PGO experiments

## 2026-04-28: `lto = "fat"` + `codegen-units = 1` — neutral-to-slight-regression

**Context**: end of 5d-9 session. drama_llama had no `[profile.release]`
override; moeflux already had `lto = "thin"`. We promoted drama_llama
to `lto = "fat"` + `codegen-units = 1` to see if it would push the
A3B warm number past 8.15 tok/s (5d-9 result).

**Setup**: A3B Apollo essay × 3 sequential @ 512 max_tokens, M2 Max,
--seed 42. Same prompt + seed as the 5d-9 baseline run.

**Result**:

| Build | Cold | Warm avg | Build time |
|---|---|---|---|
| Default release (no override) | 7.69 | 8.15 (8.11 / 8.19) | ~10s |
| `lto="fat"` + `codegen-units=1` | 7.47 | 8.01 (8.06 / 7.96) | 90s |

Net: a hair slower or within noise. Build time 9× longer.

**Why it didn't pay**:

- The hot CPU loop is `partial_sort` over 248k logits (drama_llama
  sampling chain) — already vectorized; LTO can't make scalar ops
  faster.
- A meaningful chunk of CPU is in the moeflux-side rayon io_pool
  doing parallel `pread`. LTO can't unblock IO.
- Inference loops cross FFI boundaries (`metal-rs` → libsystem).
  LTO can't optimize across an FFI boundary.
- moeflux already had `lto = "thin"`, so the moeflux side was
  already getting cross-crate inlining. The drama_llama side
  doesn't have a hot enough monomorphic-generic boundary for fat
  LTO to find new wins.

**Mike's CPU/GPU obs (from his Activity Monitor)**:
- Default release: CPU ~200% / GPU ~60%
- LTO=fat:        CPU ~225% / GPU ~55-60%

LTO inlining produced slightly hotter CPU loops without converting
the cycles into more tokens — wider unrolling probably ate more
cache or branch-predictor budget per token.

**Decision**: reverted the change. drama_llama's release profile
is back to cargo defaults (opt-level=3, no LTO, codegen-units=16).

## When to re-try

LTO might help if **all** of these become true:

- A new monomorphic-generic boundary becomes a hot path. (Today's
  hot path is partial_sort + Metal FFI, neither of which benefits.)
- We add CPU-side compute that's amenable to cross-crate inlining
  (e.g. a SIMD'd top-K heap that replaces `partial_sort`, with
  generic numeric trait bounds).
- moeflux Phase 6 cutover lands and the Metal kernel side becomes
  callable from drama_llama directly without FFI.

Otherwise, default release stays. Build-time-cost-vs-runtime-gain
isn't there.

## PGO

Untried as of 2026-04-28. PGO (profile-guided optimization) requires
a profiling pass + a build pass; for this workload the benefit would
be on hot CPU branches in `partial_sort` / candidates filtering, but
those are deterministic enough that branch prediction is probably
already good. Lower priority than the candidates.rs "multiple
sorted views" idea; do that first if sampling speedup becomes
worth pursuing.
