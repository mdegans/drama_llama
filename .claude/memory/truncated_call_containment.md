# Truncated tool calls: why prevention runs out, and what containment owes

Design session 2026-07-21 PM (Mike + Claude), settling the remaining
half of [#38](https://github.com/mdegans/drama_llama/issues/38). Defect 1
landed in `6ffe4ae`; defect 3 and the ingest-side upgrade are next.

Mike's framing question, worth preserving because the answer keeps
coming up: *"why is the model even able to emit these in an illegal
context? Can't we make it impossible, with the error as suspenders?"*

## The four classes

Sort every way a frame marker reaches a `Block::Text` and the answer
falls out:

**(a) Model chooses to stop mid-call — ALREADY IMPOSSIBLE.** The
EOG-by-id rule (`sample/grammar.rs:2093-2136`, `sample/state.rs:460`)
lets an EOG candidate survive a live constraint only if its own bytes
*complete* that constraint. This is the "model bails out of a call it
can't finish" path and it is shut. Closed 2026-07-10/11/14.

**(b) Model emits the opener in a free region — LEGAL, MUST STAY.**
Two different things get called "free," and they want opposite
treatment:

- *root-level prose, no constraint* — the opener is how a call
  **begins** on the Auto path. Ban it and Auto tool calling cannot
  start. Pinned by `region_ban_inert_outside_a_constraint`
  (`sample.rs:2261`).
- *a permissive span inside a live grammar* (`.+` JSON string, an
  `until()` raw value) — the opener is poison there, and #37
  (`dd18157`) bans it. This is the sense in which #61's closing comment
  used "free region."

The deeper reason (b) can't be closed: **the token is not illegal when
sampled.** It becomes poison retroactively, because the turn ended
before the call closed. You cannot ban at sampling time a token whose
legality depends on the future.

**(c) Model byte-spells the marker — MOSTLY NOT A VECTOR.** Verified
2026-07-21: the deferred-trigger scan is **byte-based**
(`predictor.rs:960` — `self.text.as_bytes()` windowed against
`spec.activate_after`). So `<`, `tool`, `_call`, `>` as four ordinary
content tokens **activates the deferred grammar exactly like the single
special token would**. It starts a real call; the grammar takes over.
It only bites where no deferred grammar is armed (no tools advertised,
`ToolChoice::None`, empty-trigger dialect) — and there, there is no
legal call to protect, so prevention buys nothing containment doesn't.

**(d) Budget exhaustion mid-call — IRREDUCIBLE.** `max_tokens` or
context-full. A grammar constrains what is **legal**, never whether the
model **finishes**. No emission-side mechanism reaches this.

**So the error isn't suspenders for (d) — it's the only garment.** And
that follows from the data model: half a `serde_json::Value` has no
representation, so a truncated call cannot round-trip, ever. (#61's
closing comment reached the same conclusion independently.)

## Do NOT re-propose: banning the marker as a byte sequence

Considered and rejected this session. Mike supplied the history: the
crate once had a ban set that forbade a word by banning **every token
permutation that could spell it, including individual letters**
(computed in a test, baked into the binary). It worked — and it worked
too well: banning `r` made the `count_letters("strawberry")` tool call
impossible. That machinery is what 0.7 tore out (see CLAUDE.md's
Vocab/VocabKind note; this is the *reason* behind it).

The same shape recurs for a marker veto: **a model that cannot emit
`<tool_call>` cannot explain its own protocol.** Same failure as not
being able to count the r's. Combined with (c) above — where the vector
mostly self-resolves — there is nothing to buy here.

(One correction to the folk version: the cache-rewind objection does
*not* hold. You veto pre-commit, so there is nothing to un-emit. The
objection that stands is collateral, not cost.)

## Why containment cannot key on "the parser degraded"

The first plan for defect 3 was to have `parse_text` report a
degradation and have `Session` error on it. Found by a Plan agent —
but **scoped down by Mike afterwards, and the correction matters**, so
read both halves before citing this.

**The correction (Mike, 2026-07-21): we don't support Llama 3.1.**
`llama31_json()` is the *only* trigger-less dialect in the tree.
`hermes_json()` is also `JsonNative` but carries
`section_start: "<tool_call>\n"`, so its trigger is non-empty — and it
is what `ToolChoiceOptions::default()` hardcodes. On every shipped
dialect (hermes, `qwen_xml`, `gemma4`, harmony) each degradation site
begins at a landmark that *is* a frame marker, so degradation and
poisoning coincide and the degradation-keyed design would have worked.

**Why we still didn't take it.** Not because it breaks today, but
because it depends on "every degradation begins at a frame marker"
staying true as dialects are added — an invariant nothing enforces and
which `llama31_json()` already violates. `scan_text_for_specials` is
exactly coextensive with the ingest guard that will reject the text
anyway (same predicate, no drift), covers byte-spelled markers, and
needs no `Parsed` API change. Robustness, not rescue.

The failure mode below is therefore **latent rather than routine** — it
fires only for a trigger-less dialect. It is still worth understanding,
because it is what the invariant costs when it lapses:

Trigger-less `JsonNative` dialects (Llama 3.1 — `CallSyntax::llama31_json()`,
`dialect/mod.rs:467-476`) have `trigger() == ""`, so the landmark scan
becomes `rest.find(['{', '['])` (`parse.rs:487-496`). **Any brace in
prose is a parse landmark.** Degrading back to `Text` is the documented,
intentional trade (`parse.rs:488-491`). Promoting it to an error breaks:

- **every structured generation** — `Prompt::structured_output` returns
  its JSON payload as prose, which is exactly how `response.json()` gets
  its bytes (`examples/whodunit.rs:109`);
- any prose containing a code snippet or a set-builder `{`.

And `effective_tool_syntax` (`session/mod.rs:5440-5450`) parses **every**
generation through the dialect regardless of whether `prompt.tools` is
set — so this is not confined to tool turns.

**Key on the poisoning predicate instead:** `Session::scan_text_for_specials`
(`session/mod.rs:3067-3084`), the same `special_tokens()` ∩
`tokenize_special(.., parse_special = true)` test that
`check_no_special_injection` will apply at the next ingest. No marker,
no error. Bonus: it is byte-level, so it covers (c) for free.

## The decision: two errors, not one

Both are needed; they serve different populations.

- **Generation-time** (`GrammarViolation`, plus the containment check):
  the cache is warm and the prompt unchanged, so a retry is nearly free.
  This is the only point where a truncated call is learnable *in time to
  just resample*. Removing it would re-introduce #38's own filed
  complaint — the fatal error surfacing one turn after the failure,
  which is what made the council postmortems unreadable.
- **Ingest-time** (`InjectedSpecialToken`, upgraded): rescues a caller
  who already holds poison — from us, another backend, a hand-built
  transcript, a relayed message. Repairable rather than fatal.

Shape agreed for the ingest error (Mike + Claude, both revised from
first drafts):

```rust
/// Public — callers repair with it.
pub struct Violation {
    pub at: misanthropic::prompt::Index,
    pub found: Vec<String>,
}
// InjectedSpecialToken { violations: Vec<Violation>, .. }
```

One `Vec` of pairs, **not** parallel `Vec<Index>` + `Vec<Vec<String>>`
(those desync and the pairing is implicit). `misanthropic::prompt::index`
provides `Index`, `IndexRef`, `IndexMut`, and `impl Index<BlockIndex> for
Prompt` **plus `IndexMut`** — so the caller doesn't just learn *where*,
they repair in place, resubmit, and with the two end-of-prompt
breakpoints pay only for the mutated message.

**`Display` must redact.** A static message is fine; details live in the
fields. This is not fussiness — #38 records the guard's own error text as
a third poisoning vector (`Court::scan` quoted `{piece:?}` verbatim, the
bounce seated as a tool result, and the echo re-poisoned the next
ingest). We relay errors to agents routinely, so any variant carrying
reserved bytes must not print them.

## Remaining work

- **Defect 3**: containment scan over outgoing blocks' free text in
  `run_call`, gated on `emit_specials_ban` (respect the Qwen-VL
  grounding opt-out via `with_emit_specials_ban(false)`).
- **Ingest upgrade**: `find_injected_special_in_prompt`
  (`session/mod.rs:1122-1153`) currently returns **first hit**; it needs
  to return all hits with their `Index`. That is the actual engineering
  here — the error type is the easy part.
- **`GrammarViolation.partial_output`**: `String` → `Content`. It is
  built by joining every `Block::Text` (`session/mod.rs:5133-5140`),
  which loses structure *and* means it can carry the poison bytes while
  its doc invites reuse. API break is fine pre-publish.
- **Out of scope**: #64 (EOG with an open thought on the un-grammared
  lazy span — needs a bounded escape or it manufactures #36);
  per-variant `map_session_err` in blallama (today every variant
  flattens to `500 Unknown`, `bin/blallama/blallama.rs:953-966`);
  `complete_text` has no violation check at all (`session/mod.rs:4396`).

## Live verification (2026-07-21, defect 1 only)

`just example council --verbose`, piped petition (the documented
car-wash trick question), full sealed-round → rebuttal → reaction →
ruling cycle. **Six seats, zero deaths, clean adjournment, exit 0**;
zero occurrences of `reserved special token` / `GrammarViolation` /
bounced mail. Payroll: `in: 32278 | cache w: 27860 | cache r: 4418 |
out: 10319`.

Substantively the council also flipped correctly — unanimous WALK in
round one, the Jester caught that the destination is a *Car* Wash, all
four advisors conceded, judge ruled DRIVE. That is the reaction-phase
ordering working: without it the concessions race the ruling through the
session mutex and get discarded.

**What this did NOT verify:** nothing truncated, so the new check never
fired. This is a no-regression result, not a proof the fix triggers
correctly in the wild — that is covered by
`constraint_incomplete_at_end_sees_activated_deferred` and the model
tests. To exercise it live, run with a deliberately small `max_tokens`
so a filing truncates mid-call; better done alongside defect 3, once
there is a containment error to catch what defect 1 lets through.

Driving the example non-interactively: `printf 'QUESTION\n\n' | just
example council --verbose`. rustyline reads fine from a pipe — first
line is the petition, the empty line is "send to the judge", EOF
adjourns.

## Method note

Three of Mike's recollections were checked against the tree this
session; one was right, one was right-about-the-cell-wrong-about-the-
conclusion, one was wrong in both directions. All three were flagged by
him as memory. The Plan agent then killed the first implementation plan
outright. Neither the design conversation nor the validation pass was
optional — **both changed the outcome.**
