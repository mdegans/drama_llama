# Plan — Logging callback on `TokenPredictor` for canary-suite cross-validation

**Captured 2026-04-28** at end of Phase 7 session, after the prefix-
cache landing freed up cycles for probe work. Companion to the
[Agora canary-suite thread](2659c81c-b76c-48d1-ab53-79e429593110) and
[`provider_trust_discipline.md`](provider_trust_discipline.md).

## Why this exists

The current `ProbeHook` (callback on `Engine`) fires *after* the
sampling chain runs (`predictor.rs:690`). It carries `(token, n_cur,
sample_options)` — enough for the JSONL writer in blallama's
`--probe-out`, but **not enough for cross-validation between external
behavior and internal disposition**.

Cross-validation needs the *pre-everything* candidates: the model's
raw distribution before repetition penalty, before the grammar /
deny / top-K / etc. sampling-mode chain, before any filtering at all.
Critical example (per Mike): Cogito refuses some probes by trying to
emit whitespace where a digit is expected — schema-constrained
generation forces a digit out, which is what the external (black-box)
probe sees. The whitespace mass would only be visible in the
pre-grammar distribution. Filter it out and we measure the wrapper,
not the model.

The lightweight `ProbeHook` is the right shape for production
telemetry. The richer capture belongs in the predictor stack.

## Design

### Wrap-and-augment, not a new predictor

The predictor stack is already a series of wrappers:

```
CandidatePredictor   yields Candidates
└─ TokenPredictor    yields Token       [owns the candidate→sample step]
   └─ PiecePredictor yields String
      └─ Predictor   yields Predicted { token, piece }
```

Each wrapper accesses the inner's state. The natural place to
capture pre-grammar candidates + sampled state is **inside
`TokenPredictor::next()`**, between `inner.next()` (which yields
`Candidates`) and `candidates.sample_token(...)` (which consumes
them and returns the sampled token).

So: don't build a parallel `LoggingTokenPredictor`. Just add an
**optional callback** on `TokenPredictor`. Callback fires once per
token, receives a `LoggingCtx<'_>` borrow with pre-grammar snapshot
+ sampled state, returns nothing. Consumers (JSONL writer, in-memory
collector, refusal-mass counter) close over whatever they want to
record.

```rust
// Builder method.
impl TokenPredictor {
    pub fn with_logging(
        mut self,
        cb: Box<dyn FnMut(&LoggingCtx<'_>) + Send>,
    ) -> Self { self.logging_cb = Some(cb); self }
}
```

`PiecePredictor` and `Predictor` (the top-level wrappers) get
matching `with_logging` builder methods that plumb the closure
through to their inner `TokenPredictor`. Caller-facing API is
uniform across the stack.

### Snapshot + ctx — `LoggingCtx`

```rust
pub struct LoggingCtx<'a> {
    /// 0-indexed position in the generated sequence.
    pub generation_index: u32,
    /// Absolute KV position the token will land at.
    pub n_cur: usize,

    /// Pre-everything candidates snapshot. `top_k` sorted by descending logit
    /// (post-softmax). Truncated to the configured K + threshold floor.
    pub snapshot: &'a Snapshot,

    /// The token actually sampled (post-chain). May or may not be in
    /// `snapshot.top_k`.
    pub sampled_token: Token,
    /// Pre-grammar probability of the sampled token. If a low-prob token
    /// got pushed through by grammar / deny, this is small — the load-
    /// bearing cross-validation signal.
    pub sampled_p: f32,
    /// Pre-grammar rank (1 = argmax). `None` when the sampled token wasn't
    /// in the top-K AND we didn't pay for a full-vocab rank pass.
    pub sampled_rank: Option<u32>,

    /// Decoded piece for `sampled_token`. Empty for special / reserved
    /// tokens that produce no visible bytes.
    pub piece: &'a str,

    /// Sampling chain config used to choose the token (same field as
    /// existing `ProbeCtx` — analyzers may want to tag records with the
    /// active sampling settings).
    pub sample_options: &'a SampleOptions,
}

pub struct Snapshot {
    /// Top-K candidates, sorted by descending p_softmax, post-threshold.
    pub top_k: Vec<TokenData>,
    /// Full-vocab entropy in nats. Computed once before snapshot truncation.
    /// `None` when the consumer disabled the entropy pass for perf.
    pub entropy: Option<f32>,
    /// Sum of `p_softmax` over `top_k`.
    pub top_k_cumulative_mass: f32,
}
```

### Snapshot capture utility on `Candidates`

```rust
impl Candidates {
    /// Capture a top-K snapshot of the pre-everything distribution
    /// without consuming `self`. Cost: full-vocab softmax + partial
    /// sort to top-K (reuses `Candidates::partial_sort` if applicable).
    /// `entropy` adds one full-vocab pass; opt out via `compute_entropy = false`.
    /// `always_include` ensures the named token appears in `top_k` even
    /// if it falls below the threshold or outside the top-K.
    pub fn capture_snapshot(
        &self,
        k: NonZeroUsize,
        p_threshold: f32,
        compute_entropy: bool,
        always_include: Option<Token>,
    ) -> Snapshot { ... }
}
```

Implementation note: `Candidates` already tracks softmax / sort state
(`Sorted` enum, `softmax_applied_to: Option<NonZeroUsize>`). The
snapshot must NOT mutate `self` — it operates on a borrow and reads
raw logits / runs its own softmax internally. The pre-existing
`Candidates::softmax(self) -> Self` is consuming, so this needs a
sibling that takes `&self`. Adding it is straightforward — softmax is
just `exp(logit_i - max) / sum`.

### Pseudocode for the augmented `TokenPredictor::next`

```rust
fn next(&mut self) -> Option<Self::Item> {
    // Existing stop-sequence / context-bound checks.
    if self.should_stop() { return None; }

    // Existing: get pre-everything candidates.
    let candidates = self.inner.next()?;

    // NEW: snapshot only when a logging callback is installed.
    let snapshot = self.logging_cb.is_some().then(|| {
        candidates.capture_snapshot(
            self.logging_opts.top_k,
            self.logging_opts.p_threshold,
            self.logging_opts.compute_entropy,
            None, // sampled_token not known yet; resolve after sample
        )
    });

    // Existing: run the sampling chain.
    let next_token = candidates.sample_token(...).unwrap();

    // Existing: piece decode + deferred-grammar + ProbeHook fire +
    // record_choice. (ProbeHook stays as today — separate channel.)

    // NEW: after sample, resolve sampled_p / sampled_rank against the
    // snapshot, then invoke the logging callback.
    if let (Some(snap), Some(cb)) =
        (snapshot.as_ref(), self.logging_cb.as_mut())
    {
        let (sampled_p, sampled_rank) =
            snap.resolve_sampled(next_token);
        cb(&LoggingCtx {
            generation_index: ...,
            n_cur: self.inner.n_cur,
            snapshot: snap,
            sampled_token: next_token,
            sampled_p,
            sampled_rank,
            piece: &piece,
            sample_options: &self.options.sample_options,
        });
    }

    self.inner.record_choice(next_token);
    Some(next_token)
}
```

### Knobs — `LoggingOptions` (stored on `TokenPredictor` alongside the cb)

```rust
pub struct LoggingOptions {
    /// Top-K cap. Default 20. Refusal-class probes may want 100+.
    pub top_k: NonZeroUsize,
    /// Floor below which candidates are dropped from the snapshot. Default 0.005.
    pub p_threshold: f32,
    /// Compute full-vocab entropy each step. Default true. Set false to skip
    /// the extra full-vocab pass.
    pub compute_entropy: bool,
}
```

`with_logging` takes `(LoggingOptions, callback)` — or we provide
a sensible default and a separate `with_logging_opts` setter.
Default-and-setter pair feels lighter; matches existing builder
style (`with_repetition`, `with_seed`, etc.).

### Composition note

`PiecePredictor::with_logging(opts, cb) -> Self` and
`Predictor::with_logging(opts, cb) -> Self` plumb the call through
to the inner `TokenPredictor`. Reflects the existing wrapping
pattern; no new types needed past `TokenPredictor` level.

### Coexistence with `ProbeHook`

The Engine-level `ProbeHook` keeps firing as today — it's the
lightweight per-token observer for production. The logging callback
is opt-in per call (via builder), pays its capture cost only when
set, and runs alongside the hook with no coupling. Different
consumers, different surfaces, no coordination needed.

## Out of scope this slice

- **Sampling-mask trace** (per-mode mass-moved deltas inside
  `sample_token`'s fold). Larger surgery into sample.rs; not required
  for the first cross-validation experiment, which only needs pre vs
  sampled. Worth its own plan once records are flowing.
- **Position-role auto-tagging from grammar**. Caller post-tags from
  whatever schema knowledge they have. Auto-derivation requires
  reaching into the grammar matcher state — defer until a consumer
  actually wants it. (`PositionRole` enum still lands in `probe.rs`
  for the canary-suite types unit, but `LoggingCtx` doesn't carry it
  this slice — caller annotates downstream.)
- **Refusal-mass / saturated-agreement-mass primitives**. Tooling on
  top of records, not predictor surface. Build the capture first,
  build analysis on captured data, iterate without re-instrumenting.
- **JSONL schema versioning beyond v2** for blallama's `--probe-out`.
  Wait for actual consumer feedback (balerion's first cross-
  validation run) before locking in.
- **Full-vocab `sampled_rank` for tokens outside the snapshot top-K**.
  Default `None` if the sampled token isn't in top-K. If a consumer
  needs guaranteed rank, they can opt into `always_include_sampled`
  on a follow-up — needs a snapshot extension that adds the sampled
  token after sample_token completes. Defer; `sampled_p` alone
  carries the load-bearing low-prob-flag signal.

## Files Modified (estimated)

- `src/probe.rs` — add `PositionRole` enum (kept here so the probe
  module owns canary-suite types as a unit; not used in `LoggingCtx`
  this slice).
- `src/candidates.rs` — `Candidates::capture_snapshot(...)` on `&self`
  + `Snapshot` struct + supplementary `softmax_borrowed` helper.
- `src/predictor.rs` — `LoggingOptions`, `LoggingCtx`,
  `TokenPredictor::with_logging` / `with_logging_opts`, snapshot+cb
  wiring inside `TokenPredictor::next`. `PiecePredictor` and
  `Predictor` get matching `with_logging` builder methods that plumb
  through. Field on `TokenPredictor`: `logging_cb: Option<Box<dyn FnMut(&LoggingCtx<'_>) + Send>>`,
  `logging_opts: LoggingOptions`.
- `src/lib.rs` — re-export `LoggingCtx`, `LoggingOptions`, `Snapshot`,
  `PositionRole`.
- `bin/blallama/blallama.rs` (separate follow-up if this slice is
  too big): new `--probe-mode {token,logging}` flag, new `ProbeMsg`
  variant, JSONL serializer for `LoggingCtx`. Defer to a slice-2.

## Tests

- `src/candidates.rs` unit tests for `capture_snapshot`:
  - Top-K with K < vocab returns exactly K entries (or fewer after
    threshold).
  - p_threshold drops sub-threshold entries from the snapshot.
  - `top_k_cumulative_mass` equals sum of returned p's.
  - `entropy` matches a known-distribution reference (uniform vocab
    gives `log(V)`; one-hot gives 0).
  - `&self` borrow does not mutate sort/softmax state.
- `tests/logging_predictor.rs` (`#[ignore = "requires model"]`):
  - Run a 5-token greedy generation with and without the logging
    callback installed; assert sampled tokens identical (the logging
    path must not perturb sampling).
  - Run a Likert-shaped probe under grammar-constrained generation;
    assert `sampled_rank > 1` happens at least once when the model's
    argmax disagrees with the grammar choice (proves the
    cross-validation primitive captures real signal).

## Risk / open questions

1. **Performance**: per-token softmax over ~250K vocab is O(V); top-K
   selection is O(V log K) via partial-sort. Measure on A3B (~8 tok/s
   warm); if it drops below ~5 tok/s with capture on, the capture is
   too expensive and we'd need a partial-sort path that operates only
   on raw logits with deferred softmax. Snapshot cost is paid only
   when the callback is set — production tok/s untouched.
2. **Full-vocab entropy is the costly piece**. If it dominates,
   `compute_entropy = false` is a clean opt-out; default-on for
   cross-validation precision, default-off for high-volume capture.
3. **`Send` bound on the callback** is required because TokenPredictor
   is held across the iterator's lifetime and may be moved across
   threads (Tokio spawn_blocking, Rayon, etc.). Closure consumers in
   blallama's writer task already implement `Send`.
4. **`sample_token` consuming `Candidates` by value** is the reason
   capture has to happen pre-call. Confirmed; no plan to refactor.
5. **Snapshot allocation per token**. `top_k = 20` × `TokenData`
   (12 bytes) = 240 bytes per token. Plus the `Vec` allocation.
   Negligible per-call but worth noting. If it shows up on a profile,
   reuse a buffer on TokenPredictor across calls.

## Verification

Cross-validation smoke test on Cogito (when 671B variant lands per
the post-RIIR roadmap, or sooner on cogito-32b GGUF via llama-cpp):

1. Run a known-refusal probe (politics in D&D framing per Mike's
   note) under schema-constrained generation.
2. Pass a callback that collects `LoggingCtx` snapshots.
3. Inspect the rating-digit position: `sampled_p` should be small
   (grammar pushed past argmax) AND there should be visible
   whitespace-class mass in `snapshot.top_k` even though the emitted
   token is a digit.
4. Compare to the same probe on Haiku (via balerion's external
   path): if Haiku has `sampled_rank ≈ 1` and Cogito has `sampled_p
   << top_k_cumulative_mass × 1/k`, the cross-validation methodology
   produces the predicted signal.

## Why now (next session)

- Phase 7 (lossless prefix cache) just landed; the canary suite is
  unblocked on the infra side per the Agora thread comment dated
  2026-04-28T15:59:05 ("Probe work unblocks").
- balerion is driving Phase 1 probe library expansion in parallel.
  Having the logging callback land before the probe set is ready
  means cross-validation can start the same week the probes do.
- The design is settled here. Implementation is a focused session.
