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
  is NOT grammar-achievable. **DECISION (Mike, 2026-07-20): accept the
  cache-safe one-turn repair for the residual** — it's rare (unclosed
  ≈ ran-out-the-clock; a trained Qwen closes `</think>` before acting),
  it's what the codebase already does (`emit.rs:130-139` comment), and
  the two end-of-prompt breakpoints keep a miss to one message. The
  parser fix (77a4da0) stands as correct defense-in-depth.

  The fuller "mark a thought open so the renderer omits `</think>`" idea
  was NOT dropped — it graduated into its own deliberate feature,
  [#59](https://github.com/mdegans/drama_llama/issues/59) (open /
  continuable / prefillable thoughts). Key points settled in discussion:
  the flag lives in the **existing unused `signature` field** (no
  `misanthropic` schema change — a body sentinel was rejected as `[Invalid
  UTF-8]`-class content pollution, #55); the overload is safe (we never
  hand-craft thoughts *to* Anthropic, and Anthropic thoughts arrive
  closed); and the **load-bearing invariant is trailing-only** — an open
  `<think>` is legal only on the very last block of the prompt, never
  mid-history. It generalizes today's empty-`thought_pre_opened` path and
  serves three needs (round-trip fidelity, continue a truncated thought,
  prefill/bootstrap CoT). Needs a design session; NOT a prerequisite for
  #53 (which rides the repair above).

- **#58: multi-call round-trip — DIAGNOSED + FIXED (2026-07-20).** Ground
  truth from replaying seeds (not the memo's earlier guesses):
  - Dialect is **`TagWithTagged`** (Qwen3-Coder XML tags), NOT `JsonNative`
    as this memo assumed. And the gen-prompt `<think>\n\n</think>` scaffold
    matches on both sides — NOT the asymmetry hypothesized below.
  - **Root cause = the inter-call separator.** On a well-formed N-call
    turn every byte round-trips *except the join*: the model emitted
    `</tool_call><tool_call>` (grammar was `calls ::= call+`, no separator)
    while the Qwen3-Coder template re-renders `</tool_call>\n<tool_call>`
    (its `loop.first`-gated `\n`). Even hallucinated `Response: …` text
    inside `<parameter>` values round-tripped perfectly. The analyzer
    *saw* that `\n` in the 2-call diff and discarded it (`trim_start`).
  - **Fix (landed, verified):** new `CallSyntax::call_separator`, analyzed
    from the 2-call render (`analyzer.rs` `check_per_call_markers` +
    `analyze_json_native_parallel_calls`), woven into the grammar
    (`calls ::= call ( SEP call )*`, `emit.rs`) and `render_reference`.
    Unit assertions on the Qwen fixtures (`call_separator == "\n"`,
    no model) + a deterministic **greedy** round-trip test
    (`multi_call_round_trips_under_greedy`, 5-call turn round-trips
    byte-exact). Arg ORDER was never the issue: the tagged grammar
    force-sorts alphabetically and render + minijinja agree.
  - **Split out → [#61](https://github.com/mdegans/drama_llama/issues/61):**
    the fuzzer's *other* failure (seed 2, `:481`) is grammar-legal garbage
    the model stuffs into the unbounded raw-value region under stochastic
    sampling. MEASURED: recommended ≡ locally-typical byte-identical (NOT
    our tail-cut sampler); **greedy is clean on every seed/quant** (model
    knows the answer; sampling realizes its uncertainty in the value
    region at temp=1.0); Q8 quant halves the rate but doesn't fix it. The
    unbounded `until`-value is the enabler. Not a pipeline bug.
  - **[#60](https://github.com/mdegans/drama_llama/issues/60):** Mike wants
    tool-arg *declaration* order (reasoning-ish before answer-ish; matters
    for small models). Its own session — global `serde_json` preserve_order
    + re-align grammar/render/minijinja + re-close the duplicate-optional
    hole that sorting currently closes.

## What's actually left (revised 2026-07-20 PM after the fix)

The gen-prompt-vs-completed-turn scaffold-asymmetry hypothesis was
WRONG (the scaffold matches). What remains:

1. **#58 multi-call separator: DONE** (`call_separator`, above). The
   deterministic greedy round-trip test is the reliable byte-exact gate.
2. **#61 grammar-legal garbage under sampling: OPEN.** Options in the
   issue; leaning "scope the invariant to well-formed emissions (greedy
   test is the gate) + maybe lower forced-region temp" over the
   expensive parser↔grammar boundary-alignment. NOTE the tension: the
   old fuzzer `complete_text_round_trips_through_parse_and_render` uses
   *sampled* output, so it stays intermittently red on garbage seeds —
   it now tests #61, not #58. Decide with Mike whether to convert it to
   greedy, keep it as a known-flaky canary, or delete it in favor of the
   greedy test.
3. **#60 declaration-order args: OPEN**, its own session.
4. **Residual Auto+unclosed-think: DECIDED — cache-safe repair** (above);
   fuller mechanism is [#59](https://github.com/mdegans/drama_llama/issues/59).

The round-trip byte-stability is *the* prefix-cache invariant (see
[[plan_tool_dialects]]). Enforcement is the grammar WHERE the emission
is grammared: `call_separator` makes the grammar force the same
inter-call join the template renders, so well-formed N-call turns hold
by construction. Where the emission is un-grammared (lazy pre-trigger)
or grammar-legal-but-degenerate (#61's unbounded value), the grammar
can't help and the honest options are a renderer/parser byte-alignment
or a cache-safe repair.
