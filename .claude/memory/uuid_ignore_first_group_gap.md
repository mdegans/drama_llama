# `IgnoreCategory::Uuids` — the first-group gap and its follow-ups

**Landed 2026-09-22.** Read before widening the UUID suspension, before
"fixing" the seeding fold to match the live pass, or before proposing a
token-set answer to a shape problem.

## What shipped

A byte-level `UuidTracker` (`src/sample/uuid.rs`) rides `SamplerState`,
is fed each accepted token's piece by `SamplerState::advance`, and
`sample_token` skips the **whole** repetition pass — record, apply, step,
in either regime — while the generated tail is an unfinished UUID
(`9 <= run < 36`: from the first dash to the 36th character). Exposed as
`ignored_categories = [..., "Uuids"]`; on by default in
`RepetitionOptions::default()`, but the serde shadow defaults the set to
*empty*, so every sidecar that lists categories names it explicitly (all
nine local sidecars do as of landing).

## The gap, deliberately left

The **first group** (eight hex chars) is recorded and penalized like
prose: nothing distinguishes it from a hex word until the dash arrives.
In surgical mode this bites only from the second sighting (`effective >
penalty_max_count`), so a UUID read once from a feed and emitted once is
clean; the model's own repeats accrue. The multiplicative term is the
real risk on the first group — under the Qwen3.8 sidecar (`1.06`, sizes
2–5) a fully-stacked token takes `1.06^14 ≈ 2.3×` on a positive logit.

Watch for: the "I can't trust my own memory" tell persisting after
several emissions, with the drift *in the first eight characters*.

## Follow-ups, cheapest first

1. **Fold-side lookahead mask.** The seeding fold
   (`session/mod.rs::seed_prose_tokens`) has whole block text, so it can
   regex complete UUIDs and skip n-grams whose trailing token lies inside
   one — first group included. Protects prompt-seeded occurrences (tool
   results, prior tool args). No invariant is touched: **live and fold
   corpora already differ** (template-markup n-grams, BOS, `windows(max)`
   shape, step counting) and nothing asserts them equal — the
   cold≡incremental oracle is fold-vs-fold at a prompt breakpoint, and
   `tests/sampler_state_cache.rs` is resume-vs-resume. I got this wrong
   in the plan draft; the reviewer caught it.
2. **Live retraction.** `NGramStats::retract_since(step)` is trivial on
   the `VecDeque` positions, but mapping the eight bytes back to steps
   needs a token-byte-length history. Not trivial; only if (1) isn't
   enough.

## Rejected: a vocab-scan ignore set

"Every token whose piece is `^[0-9a-fA-F-]+$`" is stateless and closes
the first-group gap, but it is not shape-gated: digits, hashes and dates
all become unpenalizable, and hex-letter BPE pieces (`ed`, `ce`, `ad`,
`face`, `bad`) lose prose pressure. The codebase already treats digit
tokens as a case to *protect* selectively (`session/mod.rs`, the echoed
`"3"` note), not to exempt wholesale.

## Known false positive

A compact timestamp (`20260922-1430`) suspends until its first
non-fitting byte (`T`, space, `:`, or a fifth char in a group) — bounded
at 27 bytes. ISO dates never match (dash at position 4).
