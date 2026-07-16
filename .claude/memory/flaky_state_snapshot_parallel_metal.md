# Flaky: state_snapshot tests under same-binary parallelism (pre-existing)

**Observed 2026-07-16** during Phase 2 sampler work, and **confirmed
pre-existing** at commit `3b61cfa` (pre-hoist, changes stashed): running
`cargo test --test state_snapshot -- --include-ignored` intermittently
fails ~1 in 3 runs, and the failing test MOVES between runs:

- `forget_pos_and_memory_clear_free_snapshots` — panicked at a bare
  `prefill_chunk(...).expect("prefill")` (tests/state_snapshot.rs:186).
- `state_seq_roundtrip_resumes_generation_identically` — restore
  divergence assert.

Each test constructs its own `LlamaCppEngine`; the test binary runs them
on parallel threads ⇒ multiple simultaneous Metal contexts. Every test
passes reliably when run alone (`--test state_snapshot <name>`), and the
whole binary often passes. Not sampler-related; not caused by the
config/state split (predates it).

Suspects, in blame-Apple-first order (see
`feedback_reboot_on_gpu_weirdness`, `feedback_blame_apple_before_reverting`):
1. Metal multi-context contention / allocation failure surfacing as a
   failed decode in `prefill_chunk`.
2. Something in the global backend-init `ENGINE_COUNT` lifecycle under
   concurrent engine construction/drop.

**Workaround for CI/local certainty:** `--test-threads=1` for this
binary, or run tests individually.

**Next steps when someone picks this up:** repro after a cold boot; if
it survives, capture the actual `prefill_chunk` error value (the
`.expect` hides it) and check whether failures correlate with engine
construction overlap (add a test-local mutex as a bisect probe).
