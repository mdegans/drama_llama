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

## Bugs found and fixed via this fuzzer

All four landed in commits `a665fd3` → `8a53725` (2026-05-11 to
2026-05-12). The fuzzer runs clean afterwards: 5 min / 19.7M cases /
6 threads → zero findings. Listed in approximate find-order so the
catalogue doubles as a record of what kinds of bugs the differential
oracle catches.

* **`unescaped` rule admitted raw control bytes** — Class 2 (the
  original Agora-side report). Fix in `src/grammar_compile.rs`:
  `unescaped ::= [^"\\\x00-\x1F]` (tighten to RFC 8259 §7).
* **`feed_byte` `pending`-clear** — Class 4. `tinyvec::ArrayVec`
  capacity overflow when an Invalid 4-byte UTF-8 result didn't
  clear the buffer; next push panicked. Fix in
  `src/sample/grammar.rs::feed_byte` (clear on Invalid + on stale
  full-buffer entry).
* **GBNF parser unbounded recursion** — Class 4. `parse_atom` ↔
  `parse_alternates` on `(((...)))` blew the stack uncatchably.
  Capped at `PARSER_RECURSION_LIMIT = 256`; new
  `GrammarError::RecursionLimit` discriminant.
* **`exp` rule unbounded magnitude** — Class 2. Allowed `5E481`
  etc. that overflow `f64`. Capped at 1-2 exponent digits.
* **`escape` rule admitted lone surrogates** — Class 2. `\uD800`
  without a paired low surrogate; `\uD83C` followed by string-close
  ("unexpected end of hex escape"). Split `\u` branch into
  non-surrogate alternative + paired-surrogate alternative.
* **Optional-property typing not enforced** — Class 3 (dominant).
  `schema_to_gbnf` dropped properties not in `required:`, so the
  grammar allowed any value (or no value) for typed optionals.
  Closed via Option A: required first, optionals after, each
  optional wrapped in `( ws "," ws ... )?` with its declared type.

llama.cpp's `json.gbnf` shares two of these bugs (no surrogate
validation in `\u`; no integer-mantissa length cap). Worth filing
upstream when there's time.

## Limitations / future work

* **Multi-byte UTF-8 not exercised** — the walker is restricted to
  ASCII (`bitmap_to_bytes` only iterates words 0-1 of the 256-bit
  bitmap). Walking multibyte paths needs a UTF-8-aware walker that
  picks first-byte+continuations atomically. Until then, the
  surrogate-paired emission rule we just added is exercised only by
  the lib-side regression tests, not the fuzzer.
* **Crash isolation** — uncatchable stack overflows from the matcher
  still crash the process; we caught the obvious sources but
  pathological inputs may still find new ones. A `while true; do
  ... ; done` shell loop around the binary (or fork-per-case in the
  fuzzer) is the recovery mechanism.
* **Model mode** — the `model` subcommand is stubbed. Intended
  shape: prompt the loaded GGUF for a JSON Schema, run that through
  the same differential. GPU-bound, composes with `pure`.
* **Schema generator coverage matches what we enforce.** Generator
  produces `object`/`array`/`primitives` + `enum`/`const`/`anyOf`/
  `$ref` — exactly the subset `schema_to_gbnf` enforces. We
  intentionally don't generate `minLength`/`maxLength`/`pattern`/
  `minimum`/`maximum`/`multipleOf`/`oneOf`/`allOf` because we
  intentionally don't *enforce* them; see
  `.claude/memory/schema_constraint_keywords_decision.md`. If a
  future feature add expands what `schema_to_gbnf` supports, expand
  the generator alongside it to keep coverage honest.
* **Schema-shape dedup is coarse** — `schema_shape` only fingerprints
  the top-level type. Two genuinely different bugs sharing the same
  shape can collide. Currently fine because the corpus is empty;
  revisit if the corpus starts fragmenting again post some future
  feature add.
