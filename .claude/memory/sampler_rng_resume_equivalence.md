# Cache-hit resume reproduces the identical RNG stream as a miss

**Invariant.** A prefix-cache *hit* resumes sampling from the exact same
RNG position it would have reached on a *miss*. `SamplerState` carries the
generator itself — `rng: rand_pcg::Pcg64Mcg` (`src/sample.rs:987`) — and
`SamplerState::resumed_from` preserves it across a reconcile. Guarded by
explicit tests: `assert_eq!(resumed.rng, cached.rng)`
(`src/sample.rs:1800, 1814, 1974`). So cache hits are output-equivalent to
misses, not merely position-equivalent: the stochastic stream is byte-for-
byte the same. This is *why* the prefix cache is safe to turn on under a
seed — it does not perturb generation.

**The trichotomy that consumes `Session::seed`** (see
`build_initial_state`, `src/session/mod.rs`):

- `Some(seed)` ⇒ **fork**: fresh deterministic state from `seed`, the
  cached (KV-paired) sampler snapshot is *ignored*, cold prose fold from
  the top. Reproducible across runs, but it does **not** exercise resume.
- `None` + a cached state at the matched breakpoint ⇒ **resume**:
  `resumed_from`, fold covers only the suffix past the matched cursor.
- `None` + no cached state ⇒ **fresh**: fresh entropy, cold fold.

The sampler snapshots are **paired with the KV slot**:
`kv_setup_and_chunk_prefill` returns `Option<(SamplerState, SeedCursor)>`
alongside the restored KV, and `build_initial_state` snapshots one per
breakpoint (`bp_states`). KV *restore* itself is seed-independent
(`select_slot` keys on hash / LCP), but forking discards the paired
snapshot.

**Consequence for a `--seed` default** (issue #46): defaulting examples to
`Some(seed)` forces fork for *every* call, so the `None + cached ⇒ resume`
branch becomes unreachable and the cache examples (council / swarm /
prompt_caching) stop exercising the very resume path this invariant
protects — plus they eat a per-call ngram re-fold. Hence `--seed` defaults
to **unset (`None`)**; determinism is opt-in. See [[feedback_in_repo_memory]].
