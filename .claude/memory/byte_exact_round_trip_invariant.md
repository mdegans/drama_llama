---
name: byte-exact round-trip is a hard invariant for structured emissions
description: Mike's 2026-07-20 decision — parse→render must be byte-exact for grammar-constrained emissions (reasoning, tool calls); a mismatch is a bug to fix at the grammar layer, not a cache-safe degradation to tolerate. Covers #53 parser fix, the round-trip arc, and #58.
type: project
---

# Byte-exact round-trip: a hard invariant (not a tolerated degradation)

**Decision (Mike, 2026-07-20).** For **grammar-constrained / structured
emissions** (reasoning blocks, tool calls), `parse(raw) → render` must
reproduce `raw` byte-for-byte. A round-trip *mismatch* is a **bug to
fix**, not an acceptable degradation — even though the canonicalization
gate already makes a mismatch cache-*safe*.

**Why "cache-safe" is not good enough.** On a mismatch, `run_call`'s
gate (`session/mod.rs:4694-4741`) skips the auto-tip and falls back to
the token-id LCP walk → a re-prefill. That re-prefill is exactly the
expensive thing the round-trip invariant exists to prevent. Mike's
sharper point: the lost breakpoint may sit on **system+tools**, often
the largest chunk of the prompt (we do add an auto-tip and usually a
user-message breakpoint too, but not always) — so a single "safe"
re-prefill can be a huge token cost. On a 40–60k Agora prompt that is
minutes.

**Scope of the invariant.** Structured output (reasoning, tool calls) —
prose already round-trips verbatim (`Block::Text` renders byte-for-byte)
so it is free there. **Correction (2026-07-20): "every structured path
has a grammar" was wrong.** The lazy/Auto tool path generates the
pre-trigger region (all reasoning) *un-grammared* — the deferred grammar
is suspended until `<tool_call>\n`. So there IS an un-grammared
structured region on a shipping path, and it's exactly where the
residual gap lives (see #53 below).

**Enforcement layer = grammar WHERE the emission is grammared.** On the
forced path the grammar forces canonical shapes and holds round-trip by
construction. On the lazy pre-trigger region it cannot — no grammar is
active — so there the choices are a renderer-preserves-bytes change or a
cache-safe repair, not a grammar fix. The parser fix is correct
defense-in-depth on both.

## State (2026-07-20)

- **#53 functional half: LANDED (`77a4da0`).** The parser was swallowing
  a tool call emitted inside an *unclosed* `<think>` into the reasoning
  text (Thought when pre-opened, Text mid-stream via the #38 `incomplete`
  fallback) and never producing a `Block::ToolUse`. Two sites in
  `src/dialect/parse.rs` now scan for the call trigger before swallowing:
  the pre-opened `None`-close arm in `run`, and the `None` branch of
  `parse_thought`. Marker dialects only (non-empty trigger) — a bare-JSON
  `{`/`[` trigger would false-positive on reasoning-prose braces; no
  shipping JsonNative dialect has reasoning. 8 deterministic parser tests
  (the model fuzzer can't reliably reproduce this on a thinking-disabled
  prompt); streaming==batch invariance verified.

- **#53 round-trip half: the grammar force-close fix is RULED OUT
  (verified 2026-07-20, design investigation + hand-check of code).**
  The earlier framing here — "force `</think>` before a call on the lazy
  path, mirroring `EagerThoughtPreOpened`" — is **unsound**. Two verified
  facts kill it:
  1. **The forced path already forbids it.** `Anchor::Eager` (thinking
     off) → `root ::= ( "<think>" thought_close )? ws calls` (`emit.rs:158`);
     the optional group is all-or-nothing and `thought_close` is a
     non-nullable KMP until-`</think>` rule, so opening `<think>` obligates
     the close. Plus the Qwen template scaffolds a *closed* `<think>\n\n</think>`
     even with thinking off (confirmed from a real render). So an unclosed
     `<think>` before a call is unreachable on Method/Any; the #53 symptom
     the issue describes isn't reachable in its own test config (Method +
     thinking-off). The 8-seed sweep — always `<tool_call>` first, never
     `<think>` — is consistent.
  2. **The Auto/lazy path is the only place it's reachable, and can't be
     grammar-fixed.** `dialect_deferred_grammar_for_prompt` hardcodes
     `Anchor::Lazy` → `root ::= calls` (`emit.rs:143`), *deferred* — it
     activates only after the `<tool_call>\n` trigger fires. Forcing
     `thought_close ws calls` is (a) semantically wrong — Auto calls are
     OPTIONAL (`session/mod.rs:5207-5213`; the model may reason and not
     call), so requiring a call breaks no-call turns; and (b) mechanically
     impossible — a grammar that activates *at* the trigger can't require a
     `</think>` already emitted before it. **There is no cell where a call
     is required ∧ reasoning is un-grammared.** Correct `emit.rs` edit:
     none.

  So byte-exact round-trip for the residual (Auto + unclosed-think) case
  is NOT grammar-achievable. Two options remain, and this is a **decision
  for Mike, not yet made**: (a) a `Block::Thought` "was-unclosed" bit —
  needs a schema change in the local `misanthropic` crate
  (`prompt/message.rs:1017-1037`) and the renderer honoring it; agent
  flags it invasive and arguably off-distribution (re-ingesting an
  unclosed `<think>` isn't a template-expected shape); or (b) accept the
  cache-safe one-turn repair there (contradicts the hard-invariant, but is
  what the codebase already does — see the `emit.rs:130-139` comment). The
  parser fix (77a4da0) still stands as correct defense-in-depth.

- **#58: multi-call round-trip (distinct, also owed).** The round-trip
  *fuzzer* (`tests/session.rs::complete_text_round_trips_through_parse_and_render`)
  is dominated by a different failure than #53: a repeat-prone Qwen (new
  vendor-recommended sampling defaults + repeat penalty exempting special
  tokens) emits 3–20 back-to-back tool calls, and that turn doesn't
  round-trip. ~55% of seeds fail; every seed emitted ≥3 calls — both far
  above the ~10% rate that would point at the model rather than our code
  (Mike's heuristic), so it's likely ours: a renderer call-join or
  arg-canonicalization asymmetry. Isolated to the byte-exact assertion —
  every *structural* Qwen tool-call test (Method/auto/None/forced+
  thinking/cache-survival) is green. See [[completed-work]] and #58.

## What's actually left (revised 2026-07-20 after the investigation)

The grammar-fix step is deleted — it doesn't exist as a sound change
(above). What remains:

1. **#58 multi-call round-trip is the real blocker on the test** (not
   #53's unclosed-think). Leading suspect after seeing the scaffold: the
   Qwen template emits `<think>\n\n</think>` in the *generation prompt*
   but likely NOT (or differently) when re-rendering a *completed*
   assistant turn — a gen-prompt-vs-completed-turn asymmetry that would
   break `strip_prefix` independent of call count. Verify in
   `chat_template.rs` (`append_message`, generation_prompt handling).
   Also the renderer call-join / arg-canonicalization angle.
2. **Decision (Mike):** for the residual Auto+unclosed-think case —
   `misanthropic` "was-unclosed" bit vs accept cache-safe repair. Only
   then does flipping `complete_text_round_trips_through_parse_and_render`
   to a hard assertion make sense; today its failures are #58, not #53.

The round-trip byte-stability is *the* prefix-cache invariant (see
[[plan_tool_dialects]]). But the enforcement layer is NOT always the
grammar — where the emission is un-grammared (the lazy pre-trigger
region), the grammar cannot help, and the honest options are a
renderer-preserves-bytes change or a cache-safe repair.
