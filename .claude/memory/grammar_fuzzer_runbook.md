# Grammar fuzzer — runbook

## Why this exists

`examples/grammar_fuzz.rs` is a differential fuzzer over the
schema → GBNF → JSON pipeline. Surfaces three bug classes:

* **Class 1** — `Grammar::parse` rejects schema-derived GBNF (other
  than the legitimate `RecursionLimit` defense).
* **Class 2** — grammar accepts bytes that `serde_json::from_slice`
  rejects. The 2026-05-11 unescaped-control-char fix lived here.
* **Class 3** — grammar accepts JSON that `jsonschema` rejects against
  the original schema. Catches "grammar over-relaxes the schema" drift.
* **Class 4** — anything in the pipeline panics. Wraps `run_case` in
  `catch_unwind` so workers don't die.

A pure-Rust schema generator (no model in the loop) drives it. Model
mode is stubbed for a follow-up — pure mode is the high-leverage path.

## Tonight-mode launch (overnight)

```sh
mkdir -p ~/grammar-fuzz-corpus      # or any writable path
cargo run --release --features cli --example grammar_fuzz -- \
    pure --duration 8h --threads 6 --corpus ~/grammar-fuzz-corpus
```

Tested at ~58k cases/sec on 6 threads (M-series Mac). 8h ≈ 1.6B
cases. Findings are dedup'd by **bug-class signature**, not raw
bytes — `5E11`, `5E12`, `5E13` all collapse to one finding for the
"number out of range" bug. The corpus stays small even after
billions of cases.

## Reading findings the next morning

```sh
cargo run --release --features cli --example grammar_fuzz -- \
    report --corpus ~/grammar-fuzz-corpus
# Re-run a single finding to debug it:
cargo run --release --features cli --example grammar_fuzz -- \
    replay ~/grammar-fuzz-corpus/class2_grammar_accepts_serde_rejects/<hash>.json
```

`replay` prints the schema, the offending bytes, the compiled GBNF,
and re-runs the matcher and serde to confirm the bug still
reproduces.

## Tunables

* `--threads N` — defaults to rayon auto-detect; use 4-6 on M-series
  to leave headroom for the rest of the system.
* `--schema-depth N` — recursion depth for the schema generator
  (default 3). 4 surfaces more nested-shape bugs but produces more
  `RecursionLimit` rejections (filtered as not-a-finding).
* `--max-grammar-bytes N` — pre-flight reject schemas whose compiled
  GBNF exceeds this. Default 64 KB.
* `--seed N` — deterministic. Same seed + same code = same case
  sequence per thread (each thread offsets the seed by its tid).

## Known bug classes already found

* **Class 2 — number out of range**: `JSON_GRAMMAR`'s `exp` rule has
  no exponent-magnitude cap, so the grammar admits `5E1234` etc.
  serde uses `f64` and rejects. Fix: tighten `exp ::= [eE] [+\-]?
  [0-9] [0-9]?` (1-2 digits) or convert to a custom rule that bounds
  the magnitude.
* **Class 3 — optional-property gap**: `schema_to_gbnf` drops
  properties not in `required:`, so an object schema with
  `{"properties": {"x": {"const": 5}}}` (no `required`) compiles to
  the permissive `object` rule. `jsonschema` still enforces `const`
  on present fields. Documented behavior, but it's a real
  expressiveness gap — fix would be to emit per-property
  alternatives even when not required.

## Bugs already fixed via fuzzer findings (2026-05-11)

* **`feed_byte` `pending`-clear**: Invalid 4-byte UTF-8 left
  `pending` at capacity; next push panicked on the full ArrayVec.
  Fixed in `src/sample/grammar.rs` (`feed_byte`).
* **GBNF parser unbounded recursion**: `parse_atom` ↔
  `parse_alternates` recursion on `(((...)))` blew the stack.
  Capped at `PARSER_RECURSION_LIMIT = 256` returning the new
  `GrammarError::RecursionLimit`.

Regression tests for both live in `src/sample/grammar.rs` under
`feed_byte_recovers_after_invalid_utf8_at_capacity` and
`parser_caps_recursion_at_deep_nested_groups`.

## Limitations / future work

* **Multi-byte UTF-8 not exercised** — the walker is restricted to
  ASCII (`bitmap_to_bytes` only iterates words 0-1 of the 256-bit
  bitmap). Walking multibyte paths needs a UTF-8-aware walker that
  picks first-byte+continuations atomically.
* **Crash isolation** — uncatchable stack overflows still crash the
  whole process. If the matcher acquires a new pathological case, a
  `while true; do ... ; done` shell loop around the binary (or
  fork-per-case in the fuzzer) is the recovery mechanism.
* **Model mode** — the `model` subcommand is stubbed. Intended
  shape: prompt the loaded GGUF for a JSON Schema, run that through
  the same differential. GPU-bound, composes with `pure`.
* **Better Class 3 dedup** — current `schema_shape` only fingerprints
  the top-level type. Two genuinely different bugs sharing the same
  schema shape can collide. Acceptable noise tradeoff for now.
