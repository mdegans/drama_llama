# Decision: do not enforce JSON-Schema validator-only constraints in grammar

## TL;DR

`schema_to_gbnf` deliberately does not enforce `minLength`,
`maxLength`, `pattern`, `minimum`, `maximum`, `multipleOf`, `oneOf`,
or `allOf`. These fall through to the permissive type rule (or
`value` for combinators). Don't add support without a strong reason
— the canonical SDK behavior and our own analysis converge on
"describes-but-doesn't-enforce" as the right design.

`anyOf` IS supported and stays. It's the most useful combinator and
Anthropic's own API enforces it correctly.

## Why (the short version)

Grammar-level enforcement of these constraints replaces the model's
*reasoning about value* with structural padding that *looks* valid:

- `pattern: "^[A-Z]{2}_\d{4}$"` → model emits `"AB_0000"`. Pattern
  satisfied, semantics empty. Worst kind of failure: passes
  validation, means nothing.
- `minLength: 5` → model wanted to say "yes" → emits `"yesyy"` to
  fill the bound. Padding garbage.
- `maximum: 10` → model wanted 100, sampled '1','0', forced to
  close → emits `10`. Off by 10×, looks valid.
- `oneOf: [A, B]` → Anthropic's structured generation has been
  observed to force `null` rather than emit anything (Mike's
  observation from Agora MCP tools, 2026-05-12).

The validator catches "wrong shape." It can't catch "right shape,
wrong content." Grammar enforcement of these constraints
*manufactures the appearance of validity without the substance*.

## Why we know this is the right call

Three converging signals:

1. **SDK convergence.** Anthropic's Python, TypeScript, Ruby, and
   PHP SDKs all *strip* `min*`/`max*`/`pattern`/etc. from the schema
   sent to the API and *reword them into the field's `description`*
   ("Must be at least 3 characters"). The model reads the constraint
   as natural language, reasons about it, and the runtime validates
   the result. No grammar-time enforcement anywhere in the stack.
2. **Anthropic API behavior.** `oneOf` and `allOf` are unsupported
   server-side; passing them either has no effect or breaks
   structured generation. `anyOf` is supported and works.
3. **Documented production observation.** Agora's MCP tools using
   `oneOf` saw the model forced to output `null`. This is the
   failure-mode-by-construction we'd be replicating if we enforced
   structurally.

## What this means in practice

- Tool / output schemas that include these keywords compile cleanly;
  the constraints are silently ignored at the grammar level.
- The runtime tool consumer can still validate the model's output
  against the full schema (jsonschema crate or equivalent). If the
  value is out of range, the tool returns an error → conversational
  feedback loop → model retries with corrected value.
- Tool authors document constraints in `description` natural-language
  text. Models reason about it well in practice. Mike has never seen
  a case where a documented-but-unenforced bound failed to be
  respected by the model.

## What's still worth doing

- (Optional, future) Mirror the SDK pattern: pre-process the schema
  to strip these keywords and append the constraint as natural
  language to the `description` field. This would happen in
  misanthropic's `sanitize_for_anthropic`, not in drama_llama. Not a
  priority — schemars-emitted schemas usually have descriptions
  already, and Anthropic-aware authors write the constraint into the
  description manually.
- Document the decision in `src/grammar_compile.rs`'s module header
  so future contributors don't try to add support unaware of the
  reasoning.

## Don't

- Don't add `minLength`/`maxLength`/`pattern`/`minimum`/`maximum`/
  `multipleOf`/`oneOf`/`allOf` enforcement to `schema_to_gbnf`.
  Reverse this decision only with explicit user discussion — the
  failure modes are subtle enough that a "let's just add it"
  rewrite by future-me would re-introduce the deceptive bugs.
- Don't generate these constraints in the fuzzer. They'd produce
  Class 3 "findings" that are intentional non-features, polluting
  the corpus.
- Don't re-litigate this without new information. The decision is
  load-bearing on three independent signals (SDK convergence, API
  behavior, production observation).

## Forward-looking gap

The fuzzer's coverage of `schema_to_gbnf` is now constrained to
features we *do* enforce: `object`/`array`/`primitives` + `enum` +
`const` + `anyOf` + `$ref`. Real bugs in any of those paths would
still surface. If a future feature add (e.g., array-`items`
constraints, additional anyOf shapes) introduces a new
schema-to-grammar branch, add a fuzzer generator branch alongside it
to keep coverage honest.

## Addendum (2026-07-10): `minItems` non-emptiness IS enforced

New information per the "don't re-litigate without new information"
clause: misanthropic's `sanitize_for_anthropic` deliberately passes
`minItems: 0 | 1` through (stripping only values ≥ 2) because
**Anthropic's own structured outputs enforce non-emptiness
server-side**. That puts `minItems: 1` in the API-parity bucket with
`required` and `anyOf`, not the value-bound bucket above.

`schema_to_gbnf` therefore now enforces exactly that much:
`minItems >= 1` compiles to a non-empty array rule; counts beyond 1
remain permissive (forcing N items manufactures filler entries — the
same failure mode as `minLength`). `maxItems` remains unenforced.
The fuzzer generates `minItems: 1` on a third of arrays (never ≥ 2,
which would present the intentional non-feature as a Class 3
finding).

Trigger: Qwen3.6-35B-A3B answered the whodunit structured-output test
with `suspects_considered: []` / `key_evidence: []` while filling
every scalar field richly — the empty-array exit is a real model
behavior that `length(min = 1)` + grammar now closes.
