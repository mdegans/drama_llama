# Feature freeze → release (declared 2026-07-17 evening)

Mike's call after the council landed and validated: **feature freeze**,
focus on reliability and design polish, release in a few days. The one
candidate exception is the **disk cache** — design it during the
freeze, land only if it stays boring, genuinely willing to slip it to
0.9 (the multi-slot `PrefixSlot` is already plain serializable data,
so it's additive whenever it happens).

## Release-blocking reliability work

- [#38](https://github.com/mdegans/drama_llama/issues/38) —
  containment: truncated/unparsed tool call seats frame-marker text
  (deferred-grammar violation check, EOG-inside-constraint policy,
  parse-fail containment). The council judge derail, generalized.
- [#37](https://github.com/mdegans/drama_llama/issues/37) —
  region-aware emit ban (specials sampleable inside argument
  strings). Sketch in `future_work_region_aware_emit_ban.md`.
  Both are "a misbehaving model must not poison a transcript" — the
  council demo is the showcase, so these are release-blocking in
  spirit.
- Longer fuzz soaks with the widened corpus (nullable type-arrays now
  generated; `pure --duration 8h` per the fuzzer's own docs). The
  2026-07-17 sweeps: nullable collapse clean, found + fixed two
  pre-existing Class-3 corners (`-0` and 19+-digit integers vs i64).
- Prepublish validation checklist:
  `plan_prepublish_validation_session.md` (from the June unblocking;
  publish on Mike's go).
- Council iteration (late 2026-07-17, post-freeze-declaration): the
  **jester** landed (structural rebuttal phase — sealed round goes to
  the licensed contrarian alone before publication) and WON its debut:
  unanimous-wrong round 1, rebuttal nailed the category error, judge
  ruled Drive. Plus reason-first filings (analysis+verdict, order
  grammar-enforced), anti-stall nudges, bounce-texts-are-prompts fix,
  1024-token seat cap. Known wart: small models burn rounds filing
  meta-diary into their only tool — dissolves under the one-tool
  forced protocol designed on
  [misanthropic#139](https://github.com/mdegans/misanthropic/issues/139)
  (per-round tool_choice / dispatch-once-per-beat; Quirks-aware since
  local grammar forcing is cache-free where Anthropic's isn't). Mike
  is probing contrarian side-effects; swarm.rs also wants a shakedown
  pass under the fixed cache.

## Context for the freeze decision

The 2026-07-17 session shipped the multi-slot cache + council and
then live-run debugging immediately surfaced four reliability finds
(predictor seq-0 corruption, tripwire false positive, nullable-schema
grammar dead-end, containment gap). The codebase's maturity gap is
reliability, not features.
