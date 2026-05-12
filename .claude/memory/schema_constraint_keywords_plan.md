# Next-session plan: extend `schema_to_gbnf` for validator-only constraints

## Where this came from

After Option A landed (commit `8a53725`, optional-property type
enforcement), the fuzzer was clean — zero findings across 5 min /
19.7M cases. The schema generator was extended (commit `35c2ee4`)
to produce `minLength` / `maxLength` / `pattern` / `minimum` /
`maximum` constraints; this surfaced **8 unique Class 3 categories**
showing exactly which JSON-Schema keywords `schema_to_gbnf` doesn't
yet model. Each category is a one-line "the grammar fell through to
permissive when the validator was enforcing this constraint" gap.

The gap categories:

1. `"X" does not match "<pattern>"` — `pattern:` (regex) not enforced.
2. `"" is shorter than N character` — `minLength:` not enforced.
3. `"X" is longer than N character` — `maxLength:` not enforced.
4. `N is greater than the maximum of M` — `maximum:` not enforced.
5. `N is less than the minimum of M` — `minimum:` not enforced.
6. `is not valid under any of the schemas listed in the 'anyOf' keyword`
   — `anyOf` with constraint-bearing subschemas inherits gaps 1-5.

## Implementation difficulty (ascending)

### Easy: `minLength` / `maxLength` (strings)

The existing `string` rule is `"\"" char* "\""`. To enforce a length
in `[N, M]`, replace `char*` with N copies of `char` followed by
`(M-N)` optional `char?`s — or use GBNF's `{N,M}` repetition syntax
if our parser supports it. (llama.cpp's parser does; check ours at
`src/sample/grammar.rs` parse_atom for postfix counted repetition.)
If `{}` isn't supported, hand-unroll: `char char char (char (char
(char)?)?)?` for `[3, 6]`. Ugly but mechanical.

Emit a per-property `string_minN_maxM` rule when the schema has
`minLength`/`maxLength`. Counts up to ~50 are realistic.

### Easy: `minimum` / `maximum` (numbers, integers)

These bound integer/decimal values. GBNF can express bounded integer
ranges by partitioning the digit space:
- `[0, 308]` → `[0-9] | [1-9][0-9] | [1-2][0-9][0-9] | 30[0-8]`
- Negative bounds: prefix with `-`, swap range.

The closed-form decomposition is well-known. Worth a small helper
fn that takes (lo: i64, hi: i64) and emits the GBNF rule. Decimals
are harder (range over the rational line) — punt by enforcing only
the integer part of the bound, accept that fractional bounds remain
soft. Document the trade.

### Medium: `pattern` (regex → GBNF)

GBNF can express any regular language by hand-translation. A small
regex → GBNF translator supporting common pattern shapes
(`^[a-z]+$`, `^\d{N}$`, char classes, anchors) covers most
real-world tool schemas. Full regex is overkill; tools rarely use
backreferences or lookaround.

Two-stage approach: implement support for a documented subset, fall
through to permissive `string` for unsupported patterns. The
fuzzer's `gen_primitive` only emits patterns from a tiny test set
today — expand that to drive coverage of the subset we choose.

### Resolved-as-byproduct: `anyOf`

The current emission `rule ::= A | B` is correct. The Class 3
findings under `anyOf` are because one of the subschemas
(`{minimum: 5}`) falls through to permissive, so the union becomes
permissive. Once `minimum` / `maximum` etc. are enforced, the
`anyOf` findings disappear too. Don't fix `anyOf` directly; fix
the subschemas.

## What's *not* worth doing

- `additionalProperties: false` — already accidentally enforced by
  our fixed-shape object grammar. Schema authors can declare it
  redundantly without effect.
- `allOf` — useful in theory but rare in tool schemas. Real tool
  authors compose via `$ref`, not `allOf`. Punt indefinitely.
- `oneOf` — semantically distinct from `anyOf` only when subschemas
  overlap. In practice authors use `anyOf` even when they mean
  `oneOf`. Treat as a synonym for `anyOf`; if anyone files a bug,
  reconsider.
- Numeric `multipleOf` — never seen in the wild for tool schemas.

## Suggested session order

1. `minLength` / `maxLength`. Smallest, most mechanical, ships in
   one focused change with regression tests + fuzzer verification.
2. `minimum` / `maximum` for integers. Helper fn for the range
   decomposition, then plumb into `emit_schema_rule`'s `integer`
   branch.
3. `pattern` (subset). Define the supported subset in the doc
   comment, implement the translator, expand `gen_primitive`'s
   pattern pool to match.

After each: re-run the fuzzer at 5+ min / 6+ threads, expect that
category's findings to drop to zero. Anything that doesn't drop is
the real bug; investigate before moving on.

## Coordination with the runbook

Once any of these land, update
`.claude/memory/grammar_fuzzer_runbook.md`'s
"Limitations / future work" → "Schema generator coverage" section
to mark each category as handled.
