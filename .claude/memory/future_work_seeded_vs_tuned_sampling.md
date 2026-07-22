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
default, same prompt, same seed — on **quality**, not tok/s. (Measured
2026-07-22: tok/s is a wash, both ~24 — see below.)

Quality needs a judged comparison, not an assertion: run both chains
over a fixed prompt set spanning the genres that matter here
(technical prose, creative, structured/tool-heavy, long-form) and
compare. The long-form-degradation arc
([`qwen3_long_form_degradation.md`](qwen3_long_form_degradation.md))
is the precedent for how this went wrong last time and what to watch
for.

## Perf axis: settled, and it is a non-issue (2026-07-22)

Mike ran `bench.py` on the seeded chain (`TopK{20}, TopP{0.95},
Temperature{1.0}`): **24 tok/s, not meaningfully moved** from the prior
run on `TopK` ahead of `LocallyTypical`. Both are ~24.

So the two chains cost the same, and the open question narrows to
**quality alone**. The mechanism is worth keeping: `LocallyTypical` was
only ever expensive on an *unpruned* candidate set — put a `TopK` in
front of it and the tail is gone before it runs, so the entropy pass
walks a short list. The new seeded chain prunes harder still
(`k=20` vs `k=1024`), which is why it did not get faster: there was
nothing left to win.

**`partial_sort` earns its keep** — this is the same finding [#33] came
at from the other side. #33 removed a *gratuitous full-vocab* sort;
what remains is a partial sort over a pruned set, and it does not show
up. Do not re-litigate sampler-chain cost on the assumption that
`LocallyTypical` is slow: it is slow only without a `TopK` ahead of it,
and every chain in play has one.

Caveat for anyone comparing logs: tok/s across chains is only loosely
comparable anyway — different tokens mean different MoE expert routing.
"Both ~24" is the claim; a 0.3 tok/s delta between them would not be.

[#33]: https://github.com/mdegans/drama_llama/issues/33

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
