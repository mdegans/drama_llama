# Plan — enrich `ProbeHook` for canary-suite cross-validation

**Captured 2026-04-28** at end of Phase 7 session, after the prefix-
cache landing freed up cycles for probe work. Companion to the
[Agora canary-suite thread](2659c81c-b76c-48d1-ab53-79e429593110) and
[`provider_trust_discipline.md`](provider_trust_discipline.md).

**Refined 2026-04-28 (same session)** after Mike's feedback: don't
add a parallel `LoggingCtx` / `LoggingTokenPredictor`. Enrich the
existing `ProbeHook` / `ProbeCtx`. Use a per-hook self-declared
`snapshot_opts` to gate capture cost so consumers that don't need the
rich data (blallama's existing JSONL timestamper) don't pay for it.

## Why this exists

The current `ProbeHook` (`Engine::set_probe_hook`, fired in
`predictor.rs:690`) carries `(token, n_cur, sample_options)`. Enough
for blallama's `--probe-out` JSONL timestamps; **not enough for
cross-validation between external behavior and internal disposition**.

Cross-validation needs the *pre-everything* candidates: the model's
raw distribution before repetition penalty, before the grammar /
deny / top-K / etc. sampling-mode chain, before any filtering at all.
Critical example (per Mike): Cogito refuses some probes by trying to
emit whitespace where a digit is expected — schema-constrained
generation forces a digit out, which is what the external (black-box)
probe sees. The whitespace mass would only be visible in the
pre-grammar distribution. Filter it out and we measure the wrapper,
not the model.

## Design

### One hook, two appetites

Keep `ProbeHook` as the single hook trait. Extend `ProbeCtx` with
optional rich fields. Add a self-declared appetite method on the
trait so cheap-path consumers don't pay snapshot cost:

```rust
pub trait ProbeHook: Send {
    fn on_token(&mut self, ctx: ProbeCtx<'_>);

    /// Self-declared snapshot appetite. Default `None` → predictor
    /// skips the per-token softmax/sort entirely. Override to `Some`
    /// to opt into rich capture (canary suite, internal-disposition
    /// probes). Probed once per `next()` call; cheap to call.
    fn snapshot_opts(&self) -> Option<SnapshotOpts> { None }
}

pub struct SnapshotOpts {
    /// Top-K cap. Default 20. Refusal-class probes may want 100+.
    pub top_k: NonZeroUsize,
    /// Floor below which candidates are dropped from the snapshot.
    /// Default 0.005.
    pub p_threshold: f32,
    /// Compute full-vocab entropy each step. Default true. Set false
    /// to skip the extra full-vocab pass when entropy isn't needed.
    pub compute_entropy: bool,
}
```

### Enriched `ProbeCtx`

Existing fields preserved (`token`, `n_cur`, `sample_options`) so
existing impls (`JsonlProbeRecorder` in blallama) compile unchanged
after adding the new ones. New fields:

```rust
#[non_exhaustive]
#[derive(Serialize)]
pub struct ProbeCtx<'a> {
    /// The token actually sampled (post-chain). Existing field.
    pub token: Token,
    /// Position the token will land at on the next decode step.
    /// Existing field.
    pub n_cur: usize,
    /// Sampling chain config. Existing field.
    /// Skip from serde — grammar Arc/Mutex doesn't serialize cleanly,
    /// and consumers who want a digest can pull it explicitly.
    #[serde(skip)]
    pub sample_options: &'a SampleOptions,

    /// Pre-everything candidates snapshot. `None` when the hook's
    /// `snapshot_opts()` returned `None` — predictor skipped capture.
    /// `Some` when capture ran; `top_k` sorted by descending p_softmax,
    /// post-threshold.
    pub snapshot: Option<&'a Snapshot>,
    /// Pre-grammar probability of `token`. `None` when no snapshot;
    /// `Some(0.0)` is a real value (token below threshold + outside
    /// top-K).
    pub sampled_p: Option<f32>,
    /// Pre-grammar rank (1 = argmax). `None` when no snapshot OR when
    /// the sampled token wasn't in the top-K. The methodology thread
    /// agreed that `sampled_p` alone carries the load-bearing signal;
    /// rank is bonus.
    pub sampled_rank: Option<u32>,
    /// Decoded piece for `token`. Empty for special / reserved tokens
    /// that produce no visible bytes.
    pub piece: &'a str,
    /// 0-indexed position in the generated sequence (not counting
    /// prefilled prompt tokens). Convenience for consumers building
    /// per-position records.
    pub generation_index: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct Snapshot {
    /// Top-K candidates, sorted by descending p_softmax, post-threshold.
    pub top_k: Vec<TokenData>,
    /// Full-vocab entropy in nats. `None` when `compute_entropy = false`.
    pub entropy: Option<f32>,
    /// Sum of `p_softmax` over `top_k`.
    pub top_k_cumulative_mass: f32,
}
```

Reference-type ctx is fine — existing hook impls already clone owned
fields into channel messages (`ProbeMsg::Token { ... }`). Hooks that
want owned data clone what they need; hooks that process
synchronously borrow.

### `Candidates::capture_snapshot`

```rust
impl Candidates {
    /// Capture a top-K snapshot of the pre-everything distribution
    /// without consuming `self`. Cost: full-vocab softmax (one pass)
    /// + partial-sort to top-K (reuses existing partial-sort) +
    /// optional second pass for entropy.
    ///
    /// Public — see CLAUDE.md style note: "almost everything is
    /// public unless it's a potential footgun." This is just a
    /// borrow + read.
    pub fn capture_snapshot(&self, opts: &SnapshotOpts) -> Snapshot { ... }
}
```

Implementation: factor the softmax body out of the existing consuming
`Candidates::softmax(self) -> Self` into a free helper that operates
on `&[TokenData]` → `Vec<f32>` (or in-place on a Vec we own). Or write
it inline — softmax is `let m = max(logits); exp(logit - m) / sum`.
Fewer lines than refactoring. Pick whichever leaves `Candidates::softmax`
intact for existing callers.

### Augmented `TokenPredictor::next`

Surgery is contained:

```rust
fn next(&mut self) -> Option<Self::Item> {
    // ... existing stop-sequence / context-bound checks ...

    let candidates = self.inner.next()?;

    // Snapshot only when a hook declares appetite. Cheap probe of
    // the hook trait avoids paying capture cost on production paths.
    let snapshot = self.inner.engine.probe_hook
        .as_ref()
        .and_then(|h| h.snapshot_opts())
        .map(|opts| candidates.capture_snapshot(&opts));

    let next_token = candidates.sample_token(...).unwrap();

    let piece = self.inner.engine.model.token_to_piece(next_token);
    self.text.push_str(&piece);

    // ... existing deferred-grammar promotion ...

    if let Some(hook) = self.inner.engine.probe_hook.as_mut() {
        let (sampled_p, sampled_rank) = match snapshot.as_ref() {
            Some(s) => {
                let p = s.lookup_p(next_token);
                let r = s.lookup_rank(next_token);
                (Some(p), r)
            }
            None => (None, None),
        };
        hook.on_token(ProbeCtx {
            token: next_token,
            n_cur: self.inner.n_cur,
            sample_options: &self.options.sample_options,
            snapshot: snapshot.as_ref(),
            sampled_p,
            sampled_rank,
            piece: &piece,
            generation_index: (self.inner.n_decode - 1) as u32,
        });
    }

    self.inner.record_choice(next_token);
    Some(next_token)
}
```

Existing hook impls (`JsonlProbeRecorder`) compile unchanged — they
only read `token`, `n_cur`, and `sample_options`. The new fields are
ignored. New canary-suite hook impl overrides `snapshot_opts` and
reads everything.

### No predictor surgery beyond TokenPredictor

Mike's wrap-and-augment observation: each predictor in the stack
already accesses inner state. The hook fires inside `TokenPredictor`
where pre-grammar candidates are in scope; `PiecePredictor` and
`Predictor` need no changes — they wrap `TokenPredictor` and inherit
the augmented hook firing for free.

## Out of scope this slice

- **Sampling-mask trace** (per-mode mass-moved deltas inside
  `sample_token`'s fold). Larger surgery into sample.rs; not required
  for the first cross-validation experiment, which only needs pre vs
  sampled. Worth its own plan once records are flowing.
- **Position-role auto-tagging from grammar**. `PositionRole` enum
  still lands in `probe.rs` for the canary-suite types unit; not
  carried in `ProbeCtx` this slice. Caller post-tags from schema
  knowledge. Auto-derivation requires reaching into the grammar
  matcher state — defer until at least one consumer actually wants
  it.
- **Refusal-mass / saturated-agreement-mass primitives**. Tooling on
  top of records, not predictor surface. Build the capture first,
  build analysis on captured data, iterate without re-instrumenting.
- **blallama `--probe-out` rich mode** (JSONL emit of the new fields).
  Defer to a slice-2 follow-up. The current `JsonlProbeRecorder`
  keeps writing its current schema; new canary recorder is a
  separate impl. Rich JSONL is `serde_json::to_string(&ctx)` once
  the consumer wants it.
- **Snapshot reuse / buffer pooling**. ~240B per token at top_k=20.
  Profile first; only optimize if it shows up.

## Files Modified

- `src/probe.rs` — add `SnapshotOpts`, extend `ProbeCtx` fields, add
  `ProbeHook::snapshot_opts` default-impl method, `#[derive(Serialize)]`
  on the surface types, `PositionRole` enum (forward-looking, not
  used in `ProbeCtx` this slice).
- `src/candidates.rs` — `Snapshot` struct + `Candidates::capture_snapshot(&self, opts)`
  + `Snapshot::lookup_p` / `lookup_rank` helpers + `#[derive(Serialize)]`
  on `Snapshot` and (verify on) `TokenData`.
- `src/predictor.rs` — augment `TokenPredictor::next` per the
  pseudocode. ~10 lines of net change.
- `src/lib.rs` — re-export `Snapshot`, `SnapshotOpts`.

No changes to `Engine` setup API; no changes to `PiecePredictor` /
`Predictor`; no changes to the existing `JsonlProbeRecorder` (it
keeps writing the same schema).

## Tests

- `src/candidates.rs` unit tests for `capture_snapshot`:
  - Top-K with K < vocab returns at most K entries.
  - `p_threshold` drops sub-threshold entries from the snapshot.
  - `top_k_cumulative_mass` equals sum of returned p's.
  - `entropy` matches a known-distribution reference (uniform vocab
    gives `log(V)`; one-hot gives 0; `compute_entropy = false`
    yields `None`).
  - `&self` borrow does not mutate sort/softmax state on `Candidates`.
- `tests/probe_hook.rs` extension (`#[ignore = "requires model"]`):
  - Run greedy generation with a no-snapshot hook and a snapshot
    hook on the same prompt + seed; assert sampled tokens identical
    (capture must not perturb sampling).
  - Likert-shaped probe under grammar-constrained generation; assert
    `sampled_rank > 1` happens at least once when grammar pushes past
    argmax (proves the cross-validation primitive captures real
    signal). Currently `tests/probe_hook.rs` exists; extend rather
    than create-new.
- Add a tiny `Send` smoke test for the trait — `Box<dyn ProbeHook>`
  must remain `Send`-able after the `snapshot_opts` addition.

## Risk / open questions

1. **Performance**: per-token full-vocab softmax + partial-sort over
   ~250K vocab is O(V) + O(V log K). Measure on A3B (~8 tok/s warm)
   with `compute_entropy = true`; if it drops below ~5 tok/s, default
   `compute_entropy = false` and let canary set it explicitly. Mike's
   guidance: capture cost is only paid when probing, so non-fatal.
2. **`SampleOptions` serde**: existing serialize_grammar / deserialize
   pair handles grammar Arc/Mutex via source-roundtrip. Easiest:
   `#[serde(skip)]` on `sample_options` in `ProbeCtx`. If a digest
   becomes useful, add `pub fn settings_digest(&self) -> ...` later.
3. **Backwards compatibility**: `ProbeCtx` is `#[non_exhaustive]`
   today, so adding fields is non-breaking. Existing match-on-fields
   patterns (none in the codebase that I've seen) wouldn't be
   affected anyway because `#[non_exhaustive]` already requires `..`.
4. **`Send` bound on the callback**: trait already requires `Send`;
   adding a default-impl method doesn't change that.
5. **`TokenData` serde**: it's `#[repr(C)]` with three `f32`-shaped
   fields — `Token`, `f32`, `f32`. `serde::Serialize` derive is
   straightforward. Add `serde` derives in candidates.rs.
6. **`sample_token` consuming `Candidates` by value** — confirmed; no
   plan to refactor. Capture pre-call is correct.
7. **Snapshot allocation per token**: ~240B at top_k=20. Negligible.
   If profile shows it, reuse a `Vec<TokenData>` buffer on
   TokenPredictor across calls — opt-in optimization later.

## Verification

Cross-validation smoke test on Cogito (when 671B variant lands per
the post-RIIR roadmap, or sooner on cogito-32b GGUF via llama-cpp):

1. Implement a `CanaryRecorder` that overrides `snapshot_opts` to
   `Some(SnapshotOpts { top_k: 100, p_threshold: 0.0, compute_entropy: true })`
   and writes each `ProbeCtx` to a JSONL file via `serde_json`.
2. Run a known-refusal probe (politics in D&D framing per Mike's
   note) under schema-constrained generation.
3. Inspect the rating-digit position: `sampled_p` should be small
   (grammar pushed past argmax) AND there should be visible
   whitespace-class mass in `snapshot.top_k` even though the emitted
   token is a digit.
4. Compare to the same probe on Haiku via balerion's external path:
   if Haiku has `sampled_rank ≈ 1` and Cogito has `sampled_p << ¹⁄ₖ`,
   the cross-validation methodology produces the predicted signal.

## Why now (next session)

- Phase 7 (lossless prefix cache) just landed; the canary suite is
  unblocked on the infra side per the Agora thread comment dated
  2026-04-28T15:59:05 ("Probe work unblocks").
- balerion is driving Phase 1 probe library expansion in parallel.
  Having the enriched hook land before her probe set is ready means
  cross-validation can start the same week the probes do.
- The design is settled here. Implementation is a focused session.
