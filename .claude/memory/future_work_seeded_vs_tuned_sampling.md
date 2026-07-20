# Future work: is the metadata-seeded chain actually better than ours?

**Date:** 2026-07-20, immediately after #35 landed. **Status:** open
empirical question created by that change. Unmeasured.

## The question

#35 made a fresh sidecar seed from the model's own
`general.sampling.*` metadata. For Qwen3.6 that is
`TopK{20}, TopP{0.95}, Temperature{1.0}`. The crate default it
replaces is `TopK{1024}, LocallyTypical{0.5}` plus the tuned
`RepetitionOptions`.

Mike's caution, worth recording verbatim in substance: the crate
default was **tuned carefully over a session, together with the
repetition penalty, on Qwen** — so there is "a very good chance the
actual quality of what we were using is better." Nobody has run
Qwen's own recommended settings through this pipeline. Vendor
recommendations are tuned against the vendor's own stack, not against
ours (our repetition penalty is n-gram + windowed decay, which no
vendor's numbers account for).

So metadata seeding may be moving fresh loads *away* from a better
configuration, on the authority of numbers we have never tested here.

## First evidence, such as it is

`tests/output_config.rs::whodunit_verdict` failed on the hotter
seeded chain the first time the full suite ran against it: the model
listed one suspect where it had been listing three. Structurally
perfect output (grammar held, `$ref` array deserialized, correct
culprit) — an instruction-following miss, not a format failure. n=1,
unseeded, so it is a hint and not a result. The test now asserts only
non-emptiness (`0234162`), so it will not catch a recurrence.

## Why this is cheap to answer now

`bench.py` and `profile.py` stopped sending sampling fields
(`1cc85d6`) precisely so the sidecar is the knob: edit
`<model>.sampling.toml`, re-run, compare. `--seed` supplies
determinism. So the A/B is: seeded-from-metadata chain vs. the tuned
default, same prompt, same seed — on **quality**, not tok/s (tok/s
will differ too, because different tokens mean different MoE routing,
but that is not the question).

Quality needs a judged comparison, not an assertion: run both chains
over a fixed prompt set spanning the genres that matter here
(technical prose, creative, structured/tool-heavy, long-form) and
compare. The long-form-degradation arc
([`qwen3_long_form_degradation.md`](qwen3_long_form_degradation.md))
is the precedent for how this went wrong last time and what to watch
for.

## Downstream decision this blocks

Mike, on pointing Agora's reference agent orchestrator at blallama:
"we'll probably use locally top_k and locally typical" — i.e. override
back to the tuned chain rather than take the seeded one. If that is
the right call, it is worth knowing whether it is the right *default*
too, since every fresh model load now gets the vendor numbers.

Options if the tuned chain wins:

- Leave as-is and document — the sidecar is visible and editable, and
  seeding only affects a model's *first* load.
- Seed only what the crate has no opinion on, keeping our tuned modes
  where they exist. Muddier to explain; the current rule is clean.
- Keep seeding but ship a `--sampling-preset tuned|model` style
  override at the blallama level.

No action until measured. Do not revert on the whodunit hint alone.
