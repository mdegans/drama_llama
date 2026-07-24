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
  [#59](https://github.com/mdegans/drama_llama/issues/59), which
  **landed 2026-07-21 — see [[open_thought_blocks]]** for the mechanism
  and its two non-obvious couplings (`render_extended` must merge; the
  pre-opened check is an OR, not a widened scan). One correction to the
  notes below: "the renderer omits the close" is the wrong mechanism —
  the trailing thought is *withheld from the template entirely* and
  appended to the finished generation prompt, because anything that
  reaches Jinja gets its whitespace normalized irreversibly. Key points
  settled in discussion:
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
    byte-exact). Arg ORDER was never the issue: at the time the tagged
    grammar force-sorted alphabetically and render + minijinja agreed.
    (Since 2026-07-24 / #60 the agreed order is schema *declaration*
    order via `preserve_order` — same three-producer agreement, new
    canonical order; dict/Gemma stays alphabetical per `dictsort`.)
  - **Split out → [#61](https://github.com/mdegans/drama_llama/issues/61):**
    the fuzzer's *other* failure (seed 2, `:481`) is grammar-legal garbage
    the model stuffs into the unbounded raw-value region under stochastic
    sampling. MEASURED: recommended ≡ locally-typical byte-identical (NOT
    our tail-cut sampler); **greedy is clean on every seed/quant** (model
    knows the answer; sampling realizes its uncertainty in the value
    region at temp=1.0); Q8 quant halves the rate but doesn't fix it. The
    unbounded `until`-value is the enabler. Not a pipeline bug.
  - **[#60](https://github.com/mdegans/drama_llama/issues/60): LANDED
    2026-07-24.** `serde_json/preserve_order` + `minijinja/preserve_order`
    on unconditionally; grammar iterates `properties` in declaration
    order with required-ness by *membership* in `required:` (never that
    array's order); optionals stay in place — any fixed order closes the
    duplicate-optional hole, alphabetization was never load-bearing.
    `render_reference` renders the input Map's own order, which agrees
    with template re-render by construction and with the grammar for
    model-generated calls (parse order == grammar order). Exceptions &
    hazards that outlive the session: (1) dict/Gemma stays explicitly
    alphabetical — its templates `dictsort`, which sorts regardless of
    the feature (a no-`dictsort` sidecar is a no-op today because the
    gemma4 sniff hard-codes the sorted dict paths;
    [#72](https://github.com/mdegans/drama_llama/issues/72) is the
    probed `dict_sorted` flag that would make such a sidecar opt Gemma
    into declaration order — experiment for OOD behavior first); (2) minijinja's SerializeStruct path still alphabetizes
    Rust *structs* fed to templates — we only feed `serde_json::Value`,
    keep it that way; (3) Anthropic structured outputs reorder
    required-first, so cross-engine tools should declare required fields
    first if identical ordering matters (the misanthropic `#[tool]`
    macro does NOT enforce that — verified 2026-07-24); (4) the
    `serde_json_preserves_insertion_order` canary +
    `declaration_order_*` round-trip tests trip if anyone drops the
    features. Canonical bytes changed (defs envelope now
    type/function + name/description/parameters = ollama training
    shape); 0.8.0-dev-era warm caches won't match — changelogged.

## THE PERMANENT BOUND (settled 2026-07-21, Mike)

**Complete structures round-trip. Incomplete ones round-trip only where
the block type admits openness.** Exactly one does.

- **Thought** = text ⇒ "unclosed" is representable ⇒ [[open_thought_blocks]]
  handles it (trailing-only), and the same mechanism gives us CoT prefill.
- **Tool call** = structure ⇒ `Block::ToolUse` args are a
  `serde_json::Value` ⇒ **half an object has no representation and never
  will.** Not a missing feature — a type-level impossibility. Keeping raw
  bytes instead just recreates #38's poison (frame marker inside a
  `Block::Text` ⇒ ingest rejects ⇒ dead loop).

So for a truncated call the *complete* solution space is #38's two
options: typed error (caller retries) or strip-and-salvage. Stop looking
for a third. Generation can always run out of budget mid-structure — a
grammar constrains what is **legal**, never whether the model **finishes**.

Documented for consumers in `Session`'s rustdoc and `bin/blallama/blallama.rs`
module docs (feature inlined, no issue refs, per Mike). Downstream
mitigation, also recorded there: **two cache breakpoints at the end of the
prompt**, so a miss costs one message rather than everything back to the
previous structural boundary.

## 2026-07-21 PM: #37 region ban — what it actually fixed

**CORRECTION.** `dd18157`'s commit message says "materially closes #61."
That is **overclaimed**; the evidence says something more interesting.

I diffed the two seeds (6, 7) that recovered. Both failed with an
identical signature, and it is **not** #61's "grammar-legal garbage in an
unbounded raw value":

```
orig: ...assistant\n<think>\n\n</think>\n\n\n          ← scaffold, thinking off
foll: ...assistant\n<think>\n<tool_call>\n<function=  ← open think, call inside
```

That is the **#53 unclosed-`<think>`-swallows-call** case. Mechanism:
under `Method` + thinking-off the grammar is
`root ::= ( "<think>" thought_close )? ws calls` (`emit.rs:158`), and
`thought_close` is built by `emit_until_rules` — an **until-region, hence
permissive**. So once `<think>` is emitted, everything to `</think>` is a
free region, where pre-#37 `<tool_call>` was grammar-legal (until accepts
any byte) *and* ban-exempt (marker exemption). The model could open a call
inside an unclosed thought. Post-#37 that special is banned there, and its
bytes don't exit the until-region so the exit exemption doesn't rescue it.

**The lesson worth keeping:** this memo previously concluded the
unclosed-think round-trip failure was "NOT grammar-achievable" and listed
the correct `emit.rs` edit as "none." That was *right* — and it framed the
problem as grammar-or-nothing. It was never a grammar problem. **A sampler
fix existed the whole time.** When a memo says "no fix at layer X,"
re-ask which layer owns the behavior before recording it as unfixable.

Also: **#61's documented repro (seed 2) no longer reproduces on clean
HEAD** — it passed in the control sweep, independent of any change here.

**Controlled A/B** (both sweeps back-to-back, same thermal state — #61
warns heat is a confound, so a one-sided sweep proves nothing):

| seeds | clean HEAD | with region ban |
|---|---|---|
| 1,2,3,4,5,8,12345 | PASS | PASS |
| **6, 7** | **FAIL** | **PASS** |

7/9 → 9/9, and the two recovered seeds are exactly the two failing ones.
Note clean HEAD scored better than #58's historical 4/9 — intervening
fixes (`call_separator`, #53 parser, #59) had already recovered several,
so don't compare against the memo's old numbers, only against a
same-session control.

**Residual, deliberately not closed:** a model can still *byte-spell*
`<tool_call>` inside a string. Ingest re-tokenizes with `parse_special`
and rejects it identically, so relay boundaries still need their own
policy. The ban removes the single-token path, which is the one the
model actually takes.

**Also ruled out while diagnosing this** — recorded so nobody re-derives
it. #61's "Options" section proposes aligning the parser's value
termination with the grammar's. They are *already* aligned:
`parse_tagged_call` cuts at `self.rest().find(&a.value_suffix)`
(`parse.rs:1090`) — first occurrence of the close literal, byte-identical
to the grammar's KMP `until`. The `else` branch (no close found) is what
breaks round-trip: `CallOutcome::Incomplete` → `incomplete()` →
`push_text(tail)` under `Leniency::Final`, collapsing the whole remainder
into one `Block::Text`. So the failure is **non-completion, not boundary
disagreement**, which is the same permanent bound as the section above.

**The reusable lesson:** the permissive-region predicate built for the
constrained-repetition arc (`src/sample/region.rs`) turned out to be the
load-bearing piece of a completely different fix. `ConstraintGuard::build`
returning `Some` *is* the "are frames legal here?" predicate, and
`is_protected` *is* the "is this token an exit delimiter?" predicate.
Before building a region query, check whether that module already answers
it.

Also settled while verifying: **EOG is already forbidden mid-constraint**,
region-independently — `grammar_filter` (`grammar.rs:2129`) and
`accepts_chosen` (`state.rs:460`) both keep a mid-constraint EOG only if
its own bytes *complete* the constraint. So "must close `</think>` before
stopping" holds wherever a grammar is active. This makes **#38's defect 2
stale** (noted on that issue) and scopes the real gap to the un-grammared
lazy pre-trigger span, filed as **#64**.

## What's actually left (revised 2026-07-20 PM after the fix)

The gen-prompt-vs-completed-turn scaffold-asymmetry hypothesis was
WRONG (the scaffold matches). What remains:

1. **#58 multi-call separator: DONE** (`call_separator`, above). The
   deterministic greedy round-trip test is the reliable byte-exact gate.
2. **#61 grammar-legal garbage under sampling: MATERIALLY CLOSED
   2026-07-21** by #37's region ban — see the section above. None of the
   four options in the issue was the answer. **Keep the sampled fuzzer**
   (`complete_text_round_trips_through_parse_and_render`) rather than
   converting it to greedy: Mike, 2026-07-21 — "those tests fuzz so they
   find things we didn't think of," and an increased pass rate is the
   bar, not a green-every-seed guarantee. The deterministic
   `multi_call_round_trips_under_greedy` remains the hard byte-exact gate;
   the fuzzer is the discovery instrument.
3. **#60 declaration-order args: LANDED 2026-07-24** — see the #60
   entry above for the full delta and surviving hazards.
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
