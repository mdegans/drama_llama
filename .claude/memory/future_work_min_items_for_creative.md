# Future work: opt-in `minItems >= 2` enforcement (creative contexts)

**Date:** 2026-07-20. **Status:** new information logged against
[`schema_constraint_keywords_decision.md`](schema_constraint_keywords_decision.md),
which permits revisiting only on exactly that.

## The new information

Mike, on being reminded we cap `minItems` enforcement at
non-emptiness:

> I *would* like to but you're right in some contexts it could force
> hallucinated entries. For creative things, however, it's great.

That reframes the decision. The existing memo rejected count
enforcement because forcing N items makes the model *manufacture
filler* — "right shape, wrong content". That reasoning is sound for
**extractive / factual** schemas (the whodunit case: don't invent a
suspect the scenario never named). It is exactly backwards for
**generative** schemas: "give me five plot twists", "three alternate
endings", "eight NPC names" — there, inventing the fifth entry *is
the task*, and a model that returns three has under-delivered.

Same mechanism, opposite valence. So the blanket "don't enforce" is
too coarse; the missing thing is a way for the caller to say which
kind of schema this is.

## Why it isn't a one-line flip

Enforcement is refused in **two independent places**, and both have
to move:

1. **misanthropic strips it before we see it.**
   `sanitize_for_anthropic` (`src/prompt/output.rs:342`, and the doc
   at `:333`) removes `minItems` when outside `{0, 1}` — correct for
   the wire, since Anthropic only enforces non-emptiness server-side.
   The existing memo notes min=1 "survives the sanitizer and compiles
   into the grammar", i.e. drama_llama compiles the **post**-sanitizer
   schema. So a `min = 2` never reaches our compiler at all. Fixing
   this is upstream work in misanthropic, not here.
2. **`schema_to_gbnf` ignores counts beyond 1 deliberately.**
   `src/grammar_compile.rs:246` tests `>= 1` and nothing else, with
   the filler-entry rationale inline.

## Design constraint: it must be opt-in, and the axis is task type

The knob cannot be inferred from the schema — `Vec<Suspect>` and
`Vec<PlotTwist>` are structurally identical. It has to be declared.
Rough options, unranked:

- A flag on `OutputConfigOptions` (`enforce_item_counts: bool`),
  simplest, but whole-schema granularity — a mixed schema with one
  extractive and one generative array gets one answer for both.
- Per-field, via a schemars extension keyword the sanitizer is taught
  to preserve for local use. Finer-grained and honest, but needs a
  keyword that survives both sanitizers and means nothing to the API.
- Local-only bypass: keep the wire schema sanitized, compile the
  grammar from the **pre**-sanitized schema. Attractive because the
  wire and the local grammar legitimately want different strictness,
  but it makes local and cloud behavior diverge for the same
  `Prompt`, which cuts against the "no conversion layer" property
  `src/prompt.rs` is built on.

Whichever wins, the default must stay **off** — the extractive
failure mode is the deceptive one (passes validation, means nothing),
and silent filler is worse than a short array.

## Test note

`tests/output_config.rs::whodunit_verdict` asserted
`suspects_considered.len() >= 2` while the schema only carried
`length(min = 1)`; it broke when sampling got hotter and was relaxed
to non-emptiness (`0234162`). If count enforcement lands, that
assertion is a natural regression probe — but for the *extractive*
direction, i.e. it should keep asserting only non-emptiness. A
generative fixture would need its own test.

## Don't

Don't implement this as an unconditional change to `schema_to_gbnf`.
The original decision's three signals (SDK convergence, Anthropic API
behavior, the Agora `oneOf`→`null` production observation) all still
hold for the default path. This is an opt-in *addition*, not a
reversal.
