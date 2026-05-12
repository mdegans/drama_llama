# Class 3 design memo — closing the over-relaxation gap

## What the fuzzer found

After Class 2 was knocked out (exp cap + surrogate validation, both
landed 2026-05-12), Class 3 was the dominant remaining class. After
the dedup signature was coarsened, it collapses to roughly five
underlying patterns, all instances of a single principle:

> **`schema_to_gbnf` is a strict-subset compiler — it enforces
> *required* properties and ignores everything else.**

Per-pattern breakdown (counts from the 2026-05-11 overnight run,
post-coarsening):

1. **Optional-property typing not enforced** (~86 raw → ~3 buckets
   post-dedup). Schema like `{"properties": {"x": {"type": "integer"}}}`
   without `required: ["x"]`. `emit_object_rule` returns
   `{rule_name} ::= object` (the permissive JSON-grammar object) and
   the `x: <typed>` constraint is dropped. `jsonschema` still
   enforces `properties` on present fields, hence the disagreement.
2. **Top-level shape gap** (the "5.999...e+81 is not of type array"
   case). When the *root* schema has `{"type": "array"}` but the
   tool wrapper places it inside an object, the inner property's
   schema is dropped if not required. Same root cause.
3. **Enum on optional property** (32 raw). Same as 1 but with the
   sub-schema being `{"enum": [...]}`.
4. **Const on optional property** (8 raw). Same as 1 with `const`.
5. **anyOf on optional property** (6 raw). Same as 1 with `anyOf`.
6. **Missing required nested property** (1 raw). Schema has nested
   `required` constraints that aren't propagated — likely a bug in
   `emit_object_rule`'s recursion. Worth checking individually.

## The design conversation

Three options, each with real cost:

### Option A — Enforce all declared properties

Change `emit_object_rule` so every property in `properties` (not just
those in `required`) gets a typed slot. Optional properties become
`( ws "," ws "<name>" ws ":" ws <typed> )?` — their *presence* stays
optional but their *type when present* is enforced.

**Cost:** Grammar size grows linearly in the number of declared
optional properties. JSON-Schema-permitted-orderings explode: the
current rule fixes property order to match the schema, which is
already a JSON-spec violation we've accepted (the spec allows any
order). With optionals each present-or-absent, the grammar would
need `2^n` orderings to be order-flexible, or stay order-fixed and
become more brittle.

**Benefit:** Closes ~95% of Class 3 findings.

### Option B — Drop the oracle disagreement

Document Class 3 as "expected over-relaxation in the schema-strict
mode" and stop treating it as a finding class. The grammar is
intentionally a strict-subset compiler; production callers either
mark fields required (and get full enforcement) or accept the gap.

**Cost:** Real disagreements between grammar and schema get hidden.
Specifically, finding #6 above (the actual nested-required bug)
would no longer surface.

**Benefit:** Honest about what the compiler does. No code changes.

### Option C — Tighten only what's cheap

Enforce optional properties only when the optional schema is a
`const` or single-element `enum`. Those are the cases where the
grammar emits a literal (no real size growth) and the cost-benefit
clearly favors enforcement. Leave typed optionals (string/integer/
etc.) as today.

**Cost:** Partial fix. Rules for property-ordering still apply.

**Benefit:** Catches finding categories 3 and 4 (~40 raw / ~5
post-dedup) for nearly zero grammar-size cost.

## Recommendation for next session

**Start with Option C plus the Option B documentation.** Option A is
the right end state but the grammar-size growth and ordering question
deserve their own design pass; doing it as a side fix here would be
rushed. Option C is a low-risk win that closes the const/enum gap
(which is what real tool schemas actually use). Option B gives us
honest oracle classification in the meantime.

**Concrete next-session steps:**

1. Add a `schema_to_gbnf` test for nested required propagation —
   confirm finding #6 above is or isn't a bug. Fix if it is.
2. Implement Option C: emit per-property alternatives when the
   property's schema reduces to a literal (`const`, single-`enum`,
   `null`, `boolean`).
3. Update `examples/grammar_fuzz.rs` Class 3 oracle to skip the
   "optional-property type mismatch" case explicitly (filter out the
   raw error pattern `is not of type` when the violating field is
   not in the schema's `required` array). That removes the noise
   Option A would otherwise need to fix.
4. Re-run fuzzer 5-10 minutes; expect Class 3 to drop into the
   single digits.

## What NOT to do

- Don't try to make the schema-to-grammar compiler total. It's a
  best-effort strict subset; that's intentional and documented in
  the module header (`grammar_compile.rs` line ~27).
- Don't add an `additionalProperties: false` enforcer just because
  jsonschema would flag the bug. Tool schemas in practice often have
  ad-hoc fields the model is allowed to add (per Anthropic's API
  shape).
- Don't enable the model-mode of the fuzzer for this — Class 3 is a
  schema-side issue, not a model-side one. Pure mode is the right
  tool.
