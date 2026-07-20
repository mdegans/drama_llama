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

**Scope of the invariant.** Grammar-constrained/structured output only.
Prose already round-trips verbatim (`Block::Text` renders byte-for-byte)
so it is free there. Structured output is where parse→render normalizes
— and every such path has a grammar, so we can force a canonical
emission and hold round-trip by construction. Un-grammared structured
output is the only place it is unachievable, and our tool paths never
hit that.

**Enforcement layer = grammar (forces canonical shapes), with the
parser handling the shape gracefully as defense-in-depth.** The parser
fix alone does *not* deliver round-trip; the grammar does.

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

- **#53 round-trip half: OWED (next focused arc, task #9).** A
  genuinely-unclosed-`<think>` emission still can't round-trip byte-exact
  — the renderer always closes `</think>` (`chat_template.rs:1030-1036`,
  `append_block_text`) and `Block::Thought` carries no "was-unclosed"
  bit. Fix: **force `</think>` before a call on the lazy/non-forced
  path**, mirroring `Anchor::EagerThoughtPreOpened` (`emit.rs:145-149`,
  `root ::= thought_close ws calls`) which already makes the *forced*
  path byte-exact. `Anchor::Lazy` (`emit.rs:142-143`, `root ::= calls`)
  does not constrain reasoning. **Design question before coding:** is
  forcing the close on the lazy path always correct? Anchor selection is
  `dialect_grammar_for_prompt` / `session/mod.rs:5178-5182`; pre-opened
  detection is `render_ends_with_open_reasoning` (`session/mod.rs:5347`).

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

## The arc still to run (task #9)

1. Design pass on anchor selection (lazy-path close-before-call).
2. Grammar fix for #53's unclosed-think (force `</think>` before calls).
3. #58 multi-call round-trip fix (renderer/canonicalization).
4. Flip `complete_text_round_trips_through_parse_and_render` to a **hard
   assertion** ("even if we get a lot of failures to fix" — Mike), i.e.
   stop treating its failures as tolerable flake.

The round-trip byte-stability is *the* prefix-cache invariant (see
[[plan_tool_dialects]]). Don't relax the assert to make it pass — fix
the dialect/grammar.
