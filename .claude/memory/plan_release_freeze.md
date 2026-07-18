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
- [#39](https://github.com/mdegans/drama_llama/issues/39) —
  `Session::with_max_tokens`: name collides with the wire param, min
  is silent, and the 1024 default beats misanthropic's 4096 `Prompt`
  default (truncated the jester's first real rebuttal, run five).
  Direction: rename to a ceiling (`with_max_tokens_ceiling` — NOT
  ctx-flavored; it's a generation cap, not context), default
  unlimited, log on clip. Small surface, freeze-appropriate polish.
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
- Council iteration (2026-07-18): **reaction phase — the judge rules
  last**. The 07:11 run showed the rebuttal flipping two advisors and
  the concessions being discarded: publication went to all seats at
  once, the judge's short ruling out-raced the long reactions through
  the session mutex, late filings bounced off `Phase::Idle`. A ruling
  is chat prose — ungateable — so the gate is information ordering:
  rebuttal → one mandatory (free) reaction per advisor → record
  publishes to the judge ALONE. No seat wakes without a duty (missing
  record pieces ride the next duty-bearing mail), so post-publish
  chatter is structurally impossible and rounds cost fewer model
  turns. Plus: lost seats excused not awaited (`Mailbox::send`
  failure prunes at delivery; `/nudge` a corpse to excuse it
  mid-round — the recovery for this run's lawyer death, a live
  #37→#38 instance), and `Court::scan`'s bounce echo defanged
  (`‹tool_call›` — the verbatim quote was itself a poison vector).
  System-role record delivery considered, deferred until misanthropic
  model `Capabilities` (Qwen template likely lacks mid-conversation
  system). Landed `6614ffd`. **Run two of the phase (same day) ended
  with no ruling** — the jester died on its FIRST rebuttal attempt
  (#37→#38 own-emission chain; never rebutted at all), the round hung
  in Rebuttal (excusal was delivery/nudge-triggered only), and the
  artist sparred thirteen bounced rounds with a *hallucinated* jester,
  reciting the briefing's "I stand by my filing" hint as a mantra.
  Fixes in `f92e663`: **a dead jester is fatal** (Mike's call —
  `jester_is_dead`, epitaph + exit(1); advisors stay excusable, now
  reaped on loop death so no phase waits on a corpse; empty book still
  publishes with the gap on the record), the Rebuttal bounce names the
  hallucination outright ("NO rebuttal has been issued yet"), the
  stand-pat hint is deleted (models don't need an invitation to not
  fold), and `log_init` now covers `council=debug` — the jester's raw
  emission was invisible in run two, which is the first thing to look
  at next run (WHY does it emit a frame marker mid-rebuttal: length
  truncation at the 1024 cap vs #37 sampling; the log will tell).
  Run three (same day, `council=debug` now visible): the artist
  hallucinated the rebuttal 21 s after the seal — the priming was the
  *filing ACK's* forward pointer ("after the jester's rebuttal the
  round returns to you"); fixed in `4315aef` (ACKs confirm and stop —
  tool results are prompts, and an ACK that narrates the future is an
  invitation to simulate it).
- **Council v2 (2026-07-18 PM, `0c3be5d`): synchronous, host-driven
  rewrite — guided by `agora/crates/agora-council`.** Runs three and
  four proved the fight was structural: the protocol is synchronous
  and the Chat driver's at-will substrate kept producing hallucinated
  rebuttals (run four's philosopher also hit #27's unforced-path
  XML-dialect emission, unparsed + truncated). Mike's call: rewrite
  as the downstream consumer's shape. Host owns every transcript; one
  forced-tool completion per seat per phase; sealing is call order;
  the human is petitioner + steward (agora `steward::prompt_action`
  analog); judge has no tools and rules last in prose. Properties by
  construction: no hallucinated futures, no wrong-dialect calls
  (tool_choice-forced grammar, cache-free), **corpse impossible at
  the example level** (a failed completion appends nothing — any
  completion error = programmer error, loud nonzero exit; the entire
  corpse machinery from earlier today deleted with the substrate).
  1176 → 660 lines, no tokio, `required-features = ["repl"]`.
  Residual: #37 in-string frame BYTES (grammar-legal) — filings are
  relay-scanned, hit = loud adjourn; closed for real by the
  region-aware emit ban.
- **Library work still queued (releases-blocking as before):** #37
  region-aware emit ban + #38 containment (other consumers, swarm,
  and unforced paths still exposed; council v2 removed the pressure
  but not the defects). Then: the payroll/cache-reads diagnosis —
  `cache w` is hardcoded `Some(0)`, `in` double-counts reads, and
  reads look shallow (tripwire only guards total misses, not hit
  depth; prime suspect is think-strip breaking tip-hash byte
  stability). Council v2's sequential forced-call shape is a cleaner
  probe for that diagnosis than v1 was.

## Context for the freeze decision

The 2026-07-17 session shipped the multi-slot cache + council and
then live-run debugging immediately surfaced four reliability finds
(predictor seq-0 corruption, tripwire false positive, nullable-schema
grammar dead-end, containment gap). The codebase's maturity gap is
reliability, not features.
