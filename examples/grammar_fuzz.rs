//! Differential fuzzer over the schema → grammar → JSON pipeline.
//!
//! Runs forever (or until `--duration`/`--cases` exhausted) generating
//! random JSON Schemas, compiling them via [`schema_to_gbnf`], walking
//! the resulting grammar to emit accepted byte strings, and asserting
//! three invariants:
//!
//! * **Class 1**: every supported schema compiles without panic and the
//!   resulting GBNF round-trips through [`Grammar::parse`].
//! * **Class 2**: every grammar-accepted byte string parses cleanly via
//!   `serde_json::from_slice`. (The bug we caught on 2026-05-11 lived
//!   here — `unescaped` admitted raw control bytes.)
//! * **Class 3**: every grammar-accepted-and-serde-parseable JSON value
//!   also satisfies the original schema as judged by the `jsonschema`
//!   crate. Catches "grammar over-relaxes the schema" drift.
//!
//! Findings are written one-per-file under `--corpus` (default
//! `./fuzz-corpus`), named by SHA-256 of the (class, schema, bytes)
//! triple so re-finding the same bug doesn't fill the disk. Replay any
//! one with `cargo run --release --example grammar_fuzz -- replay
//! <path>`.
//!
//! ## Subcommands
//!
//! * `pure` — CPU-bound, rayon-parallel. The default and the workhorse;
//!   single-digit milliseconds per case lets us hit O(10^7) cases over
//!   eight hours per worker. Run this in one terminal.
//! * `until` — differential fuzzer for
//!   [`emit_until_rules`] (the GBNF
//!   encoding of llama.cpp's `until()` combinator). The in-crate unit
//!   test `until_rules_match_naive_exhaustively` already covers every
//!   string ≤7 chars over a 3-char alphabet; this subcommand covers
//!   the space beyond that — longer strings, random delimiters up to
//!   6 chars (including multi-byte), and a grammar walker that
//!   generates accepted strings the naive matcher must agree with.
//!   See "`until` findings" below for its own class numbering.
//! * `model` — *stub for now*. The intended shape: ask Cogito (or
//!   whatever `models/model.gguf` resolves to) to emit a JSON Schema;
//!   feed the result through the same differential. GPU-bound, so it
//!   composes with `pure` on a separate process. Wiring lands in a
//!   follow-up session — pure mode is the high-leverage path.
//! * `replay <finding.json>` — re-runs a saved finding for debugging.
//!
//! ## `until` findings
//!
//! `until` writes its own finding classes (5–8, `until_*` named)
//! rather than reusing 1–3 — those are schema-pipeline shaped
//! (`schema` + `bytes`), and an `until` case has neither. See
//! `Finding::UntilCompileBroken` / `UntilMismatch` /
//! `UntilWalkerMismatch` / `UntilPanicked`.
//!
//! ## Tonight-mode launch
//!
//! ```sh
//! mkdir -p fuzz-corpus
//! cargo run --release --features cli --example grammar_fuzz -- \
//!     pure --duration 8h --threads 6 --corpus ./fuzz-corpus
//! ```
//!
//! Findings appear under `fuzz-corpus/class{1,2,3}_*/<hash>.json` as
//! they're discovered. Aggregate counts print every 30s.

use std::{
    collections::hash_map::DefaultHasher,
    fs,
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    sync::Arc,
    time::{Duration, Instant},
};

use clap::{Parser, Subcommand};
use drama_llama::{
    emit_until_rules, schema_to_gbnf, Grammar, GrammarError, GrammarState,
    JSON_GRAMMAR,
};
use rand::rngs::SmallRng;
use rand::seq::{IndexedRandom, SliceRandom};
use rand::{RngExt as _, SeedableRng};
use serde_json::{json, Value};

// ───────────────────────────────────────────────────────────────────────
// CLI
// ───────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(version, about = "Differential fuzzer for schema→grammar→JSON")]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// CPU-bound random fuzzer. Multi-threaded via rayon.
    Pure(PureArgs),
    /// Differential fuzzer for `emit_until_rules` (delimiter-terminated
    /// GBNF rules). See the module docs' "`until` findings" section.
    Until(UntilArgs),
    /// Re-run a saved finding. Prints the schema, the bytes, and the
    /// classification. Exit 0 if the finding still reproduces, 1 if it
    /// no longer does.
    Replay(ReplayArgs),
    /// Aggregate stats across the corpus directory.
    Report(ReportArgs),
}

#[derive(Parser, Debug)]
struct PureArgs {
    /// How long to run before exiting cleanly. Accepts s/m/h suffixes
    /// (e.g. `8h`, `45m`, `90s`). Defaults to "until killed."
    #[arg(long, value_parser = parse_duration)]
    duration: Option<Duration>,
    /// Worker thread count. Defaults to rayon's auto-detect.
    #[arg(long)]
    threads: Option<usize>,
    /// Per-case byte budget for the grammar walker. Cases that don't
    /// reach completion in this many bytes are dropped (not counted as
    /// findings). 4 KB is plenty for shallow schemas.
    #[arg(long, default_value_t = 4096)]
    walker_budget: usize,
    /// Maximum recursion depth for the schema generator. Higher values
    /// stress nested $ref / anyOf paths but increase the chance of
    /// generating a grammar deep enough to stack-overflow the matcher
    /// (uncatchable, takes the whole process). 3 is the sweet spot for
    /// overnight runs; 4 surfaces more bugs but trips harder.
    #[arg(long, default_value_t = 3)]
    schema_depth: usize,
    /// Reject schemas whose compiled GBNF source exceeds this many
    /// bytes before even calling `Grammar::parse`. Pathologically
    /// nested schemas can produce multi-megabyte grammars that crash
    /// the matcher on most inputs; this is the cheapest pre-flight
    /// filter.
    #[arg(long, default_value_t = 64 * 1024)]
    max_grammar_bytes: usize,
    /// Where to drop findings.
    #[arg(long, default_value = "fuzz-corpus")]
    corpus: PathBuf,
    /// Stop after this many cases. Defaults to "no cap."
    #[arg(long)]
    cases: Option<u64>,
    /// RNG seed shared across threads (each thread derives a unique
    /// per-thread seed from this). Set this to reproduce a session.
    #[arg(long, default_value_t = 0xC0FFEE)]
    seed: u64,
    /// Trace each case's schema to stderr before running. Useful when
    /// hunting an uncatchable crash — the last line printed is the
    /// schema that crashed. Cripples throughput; use with `--threads 1`.
    #[arg(long)]
    trace_cases: bool,
}

#[derive(Parser, Debug)]
struct UntilArgs {
    /// How long to run before exiting cleanly. Accepts s/m/h suffixes
    /// (e.g. `8h`, `45m`, `90s`). Defaults to "until killed."
    #[arg(long, value_parser = parse_duration)]
    duration: Option<Duration>,
    /// Worker thread count. Defaults to rayon's auto-detect.
    #[arg(long)]
    threads: Option<usize>,
    /// Per-case byte budget for the grammar walker (requirement 4's
    /// generate-then-check direction). Delimiters are ≤6 chars and
    /// direct-feed inputs are ≤~70 bytes, so this only needs enough
    /// headroom to reach one `is_complete()`; 512 is generous.
    #[arg(long, default_value_t = 512)]
    walker_budget: usize,
    /// Where to drop findings.
    #[arg(long, default_value = "fuzz-corpus")]
    corpus: PathBuf,
    /// Stop after this many cases. Defaults to "no cap."
    #[arg(long)]
    cases: Option<u64>,
    /// RNG seed shared across threads (each thread derives a unique
    /// per-thread seed from this). Set this to reproduce a session.
    #[arg(long, default_value_t = 0xC0FFEE)]
    seed: u64,
}

#[derive(Parser, Debug)]
struct ReplayArgs {
    finding: PathBuf,
}

#[derive(Parser, Debug)]
struct ReportArgs {
    #[arg(default_value = "fuzz-corpus")]
    corpus: PathBuf,
}

fn parse_duration(s: &str) -> Result<Duration, String> {
    let (num, suffix) = s.split_at(s.len() - 1);
    let n: u64 = num.parse().map_err(|e| format!("bad number: {e}"))?;
    let secs = match suffix {
        "s" => n,
        "m" => n * 60,
        "h" => n * 3600,
        _ => return Err(format!("unknown duration suffix `{suffix}`")),
    };
    Ok(Duration::from_secs(secs))
}

// ───────────────────────────────────────────────────────────────────────
// Schema generation
// ───────────────────────────────────────────────────────────────────────
//
// Hand-rolled biased random over the JSON Schema subset `schema_to_gbnf`
// understands (object/array/primitives + enum/const/anyOf/$ref). Bias
// targets known-fragile shapes:
//
//   - String enums with control-char/non-ASCII variants (round-trip risk
//     in escape_for_gbnf_string).
//   - Required-field names containing JSON-special characters.
//   - Nested objects 2+ levels deep (exercises counter / sub-rule naming).
//   - Empty `required: []` (the "permissive object" fall-through).
//   - $ref with $defs that the parent never declares (broken-link path
//     through `emit_schema_rule`).

fn gen_schema(rng: &mut SmallRng, depth: usize) -> Value {
    // depth-0 must be a leaf — composite types (object/array/anyOf)
    // recurse into gen_schema(_, depth-1), so picking object at depth 0
    // would loop forever (saturating_sub never goes below 0). The
    // top-level callers pass depth > 0 to ensure tool-call-shaped
    // schemas show up; primitives in deeper positions are fine.
    if depth == 0 {
        return gen_primitive(rng);
    }
    let pick: u32 = rng.random_range(0..100);
    match pick {
        0..15 => gen_primitive(rng),
        15..30 => gen_enum(rng),
        30..38 => gen_const(rng),
        38..55 => gen_object(rng, depth),
        55..65 => gen_array(rng, depth),
        65..72 => gen_any_of(rng, depth),
        _ => gen_primitive(rng),
    }
}

fn gen_primitive(rng: &mut SmallRng) -> Value {
    let t = ["string", "integer", "number", "boolean", "null"]
        .choose(rng)
        .copied()
        .unwrap();
    // A third of primitives render as the nullable type-array shape
    // schemars 1.x emits for `Option<T>` (`"type": ["T", "null"]`) —
    // the shape that dead-ended the council's optional tool parameter
    // (2026-07-17). The compiler collapses it via `effective_type`;
    // Class-2 keeps it honest (collapsed output must still satisfy
    // the nullable schema, which it does by construction).
    if t != "null" && rng.random_range(0..3) == 0 {
        return json!({ "type": [t, "null"] });
    }
    json!({ "type": t })
}

// `minLength` / `maxLength` / `pattern` / `minimum` / `maximum` are
// intentionally NOT generated. drama_llama deliberately doesn't
// enforce them at the grammar level (deceptive padding/truncation
// failure modes — see
// `.claude/memory/schema_constraint_keywords_decision.md`).
// Generating them would just produce "expected non-finding" noise in
// the corpus. The dedup machinery in `validator_error_stem` already
// understands the relevant error shapes if we ever reverse the
// decision and add a generator branch back.

fn gen_enum(rng: &mut SmallRng) -> Value {
    // Mix safe + adversarial enum variants — tries to bait the
    // escape_for_gbnf_string round-trip.
    let n = rng.random_range(1..=5);
    let mut variants = Vec::with_capacity(n);
    for _ in 0..n {
        variants.push(gen_string_literal(rng));
    }
    json!({ "type": "string", "enum": variants })
}

fn gen_const(rng: &mut SmallRng) -> Value {
    // const can be any JSON value; mostly stress strings + ints.
    let v: Value = match rng.random_range(0..4) {
        0 => Value::String(gen_string_literal(rng)),
        1 => json!(rng.random_range(-1000..1000)),
        2 => Value::Bool(rng.random_bool(0.5)),
        _ => Value::Null,
    };
    json!({ "const": v })
}

fn gen_object(rng: &mut SmallRng, depth: usize) -> Value {
    let n = rng.random_range(0..=4);
    let mut props = serde_json::Map::new();
    let mut required: Vec<String> = Vec::new();
    for _ in 0..n {
        let name = gen_field_name(rng);
        if props.contains_key(&name) {
            continue;
        }
        let sub = gen_schema(rng, depth.saturating_sub(1));
        // ~80% required so we hit the fixed-order-required path more
        // often than the permissive-object fallback.
        if rng.random_bool(0.8) {
            required.push(name.clone());
        }
        props.insert(name, sub);
    }
    if required.is_empty() {
        json!({ "type": "object", "properties": props })
    } else {
        json!({
            "type": "object",
            "properties": props,
            "required": required,
        })
    }
}

fn gen_array(rng: &mut SmallRng, depth: usize) -> Value {
    let items = gen_schema(rng, depth.saturating_sub(1));
    // A third of arrays carry `minItems: 1` — the one array-count
    // constraint the grammar enforces (non-emptiness, Anthropic
    // parity). Values > 1 are intentionally never generated: the
    // grammar is deliberately permissive there (see
    // schema_constraint_keywords_decision.md) and the jsonschema
    // oracle would report that intentional non-feature as a Class 3
    // finding.
    if rng.random_range(0..3) == 0 {
        json!({ "type": "array", "items": items, "minItems": 1 })
    } else {
        json!({ "type": "array", "items": items })
    }
}

fn gen_any_of(rng: &mut SmallRng, depth: usize) -> Value {
    let n = rng.random_range(2..=3);
    let mut variants = Vec::with_capacity(n);
    for _ in 0..n {
        variants.push(gen_schema(rng, depth.saturating_sub(1)));
    }
    json!({ "anyOf": variants })
}

/// Field-name generator that includes both the boring case and the
/// escape-stress cases (quote, backslash, multi-byte UTF-8, leading
/// digit, empty). The escape pipeline goes through
/// `escape_for_gbnf_string`; if any of these breaks the GBNF parser,
/// Class 1 catches it.
fn gen_field_name(rng: &mut SmallRng) -> String {
    let pool = [
        "x",
        "y",
        "z",
        "name",
        "id",
        "value",
        "items",
        "options",
        // Adversarial:
        "with\"quote",
        "with\\backslash",
        "新",
        "🌶️",
        "leading_digit_1",
        "1leading",
        "",
    ];
    pool.choose(rng).map(|s| s.to_string()).unwrap_or_default()
}

/// Same idea for string-valued enum / const literals.
fn gen_string_literal(rng: &mut SmallRng) -> String {
    let pool = [
        "alpha",
        "beta",
        "gamma",
        "delta",
        // Adversarial:
        "has\"quote",
        "has\\backslash",
        "has\nnewline",
        "has\ttab",
        "中文",
        "🍓",
    ];
    pool.choose(rng).map(|s| s.to_string()).unwrap_or_default()
}

// ───────────────────────────────────────────────────────────────────────
// `until` generation
// ───────────────────────────────────────────────────────────────────────
//
// `emit_until_rules` is the GBNF encoding of llama.cpp's `until()`
// combinator: strings ending in exactly one occurrence of `delim`,
// with `delim` occurring nowhere else. The unit test
// `until_rules_match_naive_exhaustively` already brute-forces every
// string ≤7 chars over a 3-char alphabet; this generator targets the
// space beyond that — longer strings and a wider, more adversarial
// alphabet mixing plain ASCII with GBNF-metacharacter-adjacent bytes
// (`<`, `/`, `>`, `-`, `\`, `]`, `\n`) and an occasional multi-byte
// codepoint, since the automaton runs on Unicode scalar values.

/// Alphabet shared by delimiter and input generation. Chosen to
/// include GBNF class/literal metacharacters (`-`, `\`, `]`) that
/// `escape_for_gbnf_class` / `escape_for_gbnf_string` must handle, and
/// angle-bracket/slash characters since dialect delimiters are
/// XML-tag-shaped (`</parameter>`).
const UNTIL_ALPHABET: [char; 10] =
    ['a', 'b', 'c', '<', '/', '>', '-', '\\', ']', '\n'];

/// Occasional multi-byte codepoint mixed into both delimiters and
/// inputs — exercises the automaton's Unicode-scalar-value states
/// (`until_rules_unicode_delimiter` covers the happy path; the fuzzer
/// covers combinations that test doesn't enumerate).
const UNTIL_MULTIBYTE: char = '→';

/// Random delimiter, length 1–6. Self-overlapping shapes (`aa`,
/// `aba`) are the ones that stress the KMP fallback logic in
/// `emit_until_rules`, and they show up naturally here since the
/// alphabet is small relative to the length range.
fn gen_until_delim(rng: &mut SmallRng) -> String {
    let len = rng.random_range(1..=6);
    let mut s = String::with_capacity(len * 2);
    for _ in 0..len {
        if rng.random_bool(0.08) {
            s.push(UNTIL_MULTIBYTE);
        } else {
            s.push(*UNTIL_ALPHABET.choose(rng).unwrap());
        }
    }
    s
}

/// Random input, length roughly 0–64 chars, biased so both directions
/// of the naive matcher (`ends_with(delim) && find == last occurrence`)
/// are well represented:
///
/// * splicing a random-length prefix of `delim` (including the whole
///   thing) into the middle of the string — the "partial tease" /
///   "extra occurrence before the end" cases that a naive `[^delim]*`
///   approximation gets wrong.
/// * appending the full delimiter at the end — the accept case.
///
/// Length is a soft target: a splice can push a few chars past it, so
/// actual output can run slightly over 64. That's fine for a fuzzer
/// input distribution.
fn gen_until_input(rng: &mut SmallRng, delim: &str) -> String {
    let delim_chars: Vec<char> = delim.chars().collect();
    let target_len = rng.random_range(0..=64usize);
    let mut chars: Vec<char> =
        Vec::with_capacity(target_len + delim_chars.len());
    while chars.len() < target_len {
        let pick: u32 = rng.random_range(0..100);
        match pick {
            // Splice a prefix of the delimiter (1..=full length) in.
            0..25 => {
                let n = rng.random_range(1..=delim_chars.len());
                chars.extend_from_slice(&delim_chars[..n]);
            }
            25..30 => chars.push(UNTIL_MULTIBYTE),
            _ => chars.push(*UNTIL_ALPHABET.choose(rng).unwrap()),
        }
    }
    // Bias toward accepts: append the full delimiter at the end about
    // half the time (including on top of a trailing partial splice —
    // exercises the KMP fallback-then-complete transition).
    if rng.random_bool(0.5) {
        chars.extend_from_slice(&delim_chars);
    }
    chars.into_iter().collect()
}

// ───────────────────────────────────────────────────────────────────────
// Compile schema → grammar
// ───────────────────────────────────────────────────────────────────────

/// Wrap `schema_to_gbnf` output with a `root ::= args` plus the shared
/// JSON_GRAMMAR. Returns the GBNF source so we can store it alongside a
/// finding.
fn build_schema_grammar_source(schema: &Value) -> String {
    use std::fmt::Write;
    let mut src = String::with_capacity(512);
    let _ = writeln!(&mut src, "root ::= args");
    schema_to_gbnf(schema, "args", &mut src);
    src.push_str(JSON_GRAMMAR);
    src
}

// ───────────────────────────────────────────────────────────────────────
// Grammar walker
// ───────────────────────────────────────────────────────────────────────
//
// Walks an `Arc<Grammar>` by repeatedly:
//   1. computing the first-byte bitmap from the current state
//   2. choosing one set bit uniformly at random
//   3. attempting to advance the matcher by that single byte
//   4. if `advance_bytes` errors (the bitmap is conservative — false
//      positives are possible), retry with a different bit
//   5. once `is_complete()` returns true, stochastically decide whether
//      to stop or keep extending (with a stop probability of ~5% on
//      first-completion to bias toward longer outputs)
//
// Bounded by a byte budget. If the budget is exhausted before reaching
// `is_complete()`, the case is dropped — not a finding, just a dead
// random walk.

fn walk_grammar(
    grammar: Arc<Grammar>,
    rng: &mut SmallRng,
    budget: usize,
) -> Option<Vec<u8>> {
    let mut state = GrammarState::new(grammar);
    let mut out = Vec::with_capacity(64);
    let mut completed_at_least_once = false;
    // If WALKER_TRACE_PATH is set in env, write the byte stream to it
    // as we go. Useful when chasing an uncatchable crash — the file
    // content after the crash is the exact bytes that crashed.
    let trace_path = std::env::var_os("WALKER_TRACE_PATH");
    let mut trace_file: Option<std::fs::File> = trace_path
        .map(|p| std::fs::File::create(p).expect("WALKER_TRACE_PATH create"));
    while out.len() < budget {
        // Stop-probability check. We require at least one completion
        // before we're allowed to stop (so we don't truncate mid-string),
        // and we apply a small bias to keep extending so longer cases
        // get exercised.
        if state.is_complete() {
            completed_at_least_once = true;
            if rng.random_bool(0.20) {
                return Some(out);
            }
        }

        let bitmap = state.first_byte_bitmap();
        let candidates = bitmap_to_bytes(&bitmap);
        if candidates.is_empty() {
            break;
        }
        let mut advanced = false;
        // Reorder candidates randomly so we don't always try byte 0
        // first. Pick at most 8 attempts before giving up — most steps
        // succeed on the first or second try, but a falsely-set bit
        // shouldn't burn 256 probes.
        let mut shuffled = candidates;
        shuffled.shuffle(rng);
        for byte in shuffled.iter().copied().take(8) {
            if let Some(f) = trace_file.as_mut() {
                use std::io::Write;
                let _ = f.write_all(&[byte]);
                let _ = f.flush();
            }
            if state.advance_bytes(&[byte]).is_ok() {
                out.push(byte);
                advanced = true;
                break;
            }
        }
        if !advanced {
            // Every candidate from the bitmap was a false positive.
            // Either we're stuck or the bitmap is broken — bail.
            break;
        }
    }
    if state.is_complete() {
        Some(out)
    } else if completed_at_least_once {
        // The walker ran past a previous completion point but didn't
        // reach a new one before the budget ran out. The output up to
        // the previous completion is still useful — but we don't have
        // a cheap way to recover that point here, so drop the case.
        None
    } else {
        None
    }
}

fn bitmap_to_bytes(bitmap: &[u64; 4]) -> Vec<u8> {
    // ASCII-only: the matcher's `pending` buffer (a fixed-capacity
    // `tinyvec::ArrayVec<[u8; 4]>` for partial UTF-8 codepoints)
    // panics when fed a multi-byte first-byte followed by another
    // multi-byte first-byte instead of a continuation byte. The panic
    // unwinds through deep matcher recursion and trips the
    // (uncatchable) stack guard. Filed as future work — matcher
    // should bound its pending pushes. Until then, sample only ASCII;
    // we lose multibyte coverage but the fuzzer can run cleanly.
    let mut out = Vec::with_capacity(8);
    for word_i in 0..2 {
        let word = bitmap[word_i];
        if word == 0 {
            continue;
        }
        for bit in 0..64 {
            if (word >> bit) & 1 == 1 {
                out.push((word_i * 64 + bit) as u8);
            }
        }
    }
    out
}

// ───────────────────────────────────────────────────────────────────────
// Oracles + finding classification
// ───────────────────────────────────────────────────────────────────────

#[derive(Debug)]
enum Finding {
    /// The schema-derived GBNF failed to parse, or the schema compiler
    /// itself emitted invalid GBNF text.
    CompileBroken {
        schema: Value,
        gbnf_source: String,
        error: String,
    },
    /// Grammar accepted these bytes but `serde_json::from_slice` did
    /// not parse them as valid JSON. This is the class the unescaped
    /// fix targeted.
    GrammarAcceptsSerdeRejects {
        schema: Value,
        bytes: Vec<u8>,
        serde_error: String,
    },
    /// Bytes parsed as JSON but the resulting Value did not satisfy the
    /// original schema. Indicates the grammar is too permissive.
    GrammarAcceptsSchemaRejects {
        schema: Value,
        bytes: Vec<u8>,
        value: Value,
        validation_errors: Vec<String>,
    },
    /// The schema-to-grammar pipeline panicked at some step. Includes
    /// the bytes the walker had produced before the panic (may be
    /// empty if the panic happened inside `schema_to_gbnf` or
    /// `Grammar::parse`).
    PipelinePanicked {
        schema: Value,
        bytes_so_far: Vec<u8>,
        panic_message: String,
        stage: &'static str,
    },
    /// `until` subcommand: `Grammar::parse` failed on the GBNF
    /// `emit_until_rules` emitted for `delim`. Distinct from
    /// `CompileBroken` (schema-shaped) since an `until` case has no
    /// JSON Schema to attach.
    UntilCompileBroken {
        delim: String,
        gbnf_source: String,
        error: String,
    },
    /// `until` subcommand: feeding `input` directly to the compiled
    /// grammar disagreed with the naive matcher
    /// (`input.ends_with(delim) && input.find(delim) == Some(input.len()
    /// - delim.len())`). This is the core differential — a real finding
    /// here is a bug in `emit_until_rules`.
    UntilMismatch {
        delim: String,
        input: String,
        naive: bool,
        by_grammar: bool,
    },
    /// `until` subcommand: the grammar walker generated `generated` and
    /// considered it complete, but the naive matcher does not accept
    /// it. Covers the "generate accepted strings" direction
    /// (requirement 4) that direct-input fuzzing can miss — the walker
    /// explores paths random splicing may never construct.
    UntilWalkerMismatch { delim: String, generated: String },
    /// `until` subcommand: `emit_until_rules`, `Grammar::parse`, or the
    /// matcher panicked while processing `delim`. Mirrors
    /// `PipelinePanicked`'s catch_unwind safety net for the schema
    /// pipeline.
    UntilPanicked {
        delim: String,
        panic_message: String,
        stage: &'static str,
    },
}

impl Finding {
    fn class_dir(&self) -> &'static str {
        match self {
            Finding::CompileBroken { .. } => "class1_compile_broken",
            Finding::GrammarAcceptsSerdeRejects { .. } => {
                "class2_grammar_accepts_serde_rejects"
            }
            Finding::GrammarAcceptsSchemaRejects { .. } => {
                "class3_grammar_accepts_schema_rejects"
            }
            Finding::PipelinePanicked { .. } => "class4_pipeline_panicked",
            Finding::UntilCompileBroken { .. } => "class5_until_compile_broken",
            Finding::UntilMismatch { .. } => "class6_until_mismatch",
            Finding::UntilWalkerMismatch { .. } => {
                "class7_until_walker_mismatch"
            }
            Finding::UntilPanicked { .. } => "class8_until_panicked",
        }
    }

    fn to_json(&self) -> Value {
        match self {
            Finding::CompileBroken {
                schema,
                gbnf_source,
                error,
            } => json!({
                "class": "compile_broken",
                "schema": schema,
                "gbnf_source": gbnf_source,
                "error": error,
            }),
            Finding::GrammarAcceptsSerdeRejects {
                schema,
                bytes,
                serde_error,
            } => json!({
                "class": "grammar_accepts_serde_rejects",
                "schema": schema,
                "bytes_lossy_utf8": String::from_utf8_lossy(bytes),
                "bytes_hex": hex(bytes),
                "serde_error": serde_error,
            }),
            Finding::GrammarAcceptsSchemaRejects {
                schema,
                bytes,
                value,
                validation_errors,
            } => json!({
                "class": "grammar_accepts_schema_rejects",
                "schema": schema,
                "bytes_lossy_utf8": String::from_utf8_lossy(bytes),
                "bytes_hex": hex(bytes),
                "value": value,
                "validation_errors": validation_errors,
            }),
            Finding::PipelinePanicked {
                schema,
                bytes_so_far,
                panic_message,
                stage,
            } => json!({
                "class": "pipeline_panicked",
                "stage": stage,
                "schema": schema,
                "bytes_lossy_utf8": String::from_utf8_lossy(bytes_so_far),
                "bytes_hex": hex(bytes_so_far),
                "panic_message": panic_message,
            }),
            Finding::UntilCompileBroken {
                delim,
                gbnf_source,
                error,
            } => json!({
                "class": "until_compile_broken",
                "delim": delim,
                "delim_hex": hex(delim.as_bytes()),
                "gbnf_source": gbnf_source,
                "error": error,
            }),
            Finding::UntilMismatch {
                delim,
                input,
                naive,
                by_grammar,
            } => json!({
                "class": "until_mismatch",
                "delim": delim,
                "delim_hex": hex(delim.as_bytes()),
                "input": input,
                "input_hex": hex(input.as_bytes()),
                "naive": naive,
                "by_grammar": by_grammar,
            }),
            Finding::UntilWalkerMismatch { delim, generated } => json!({
                "class": "until_walker_mismatch",
                "delim": delim,
                "delim_hex": hex(delim.as_bytes()),
                "generated": generated,
                "generated_hex": hex(generated.as_bytes()),
            }),
            Finding::UntilPanicked {
                delim,
                panic_message,
                stage,
            } => json!({
                "class": "until_panicked",
                "stage": stage,
                "delim": delim,
                "delim_hex": hex(delim.as_bytes()),
                "panic_message": panic_message,
            }),
        }
    }
}

/// Single case run: returns 0+ findings for this (schema, bytes) trial.
/// We don't short-circuit on the first finding so a single broken
/// schema can surface both class 2 and class 3 at once.
///
/// Wraps the whole pipeline in `catch_unwind`. Real panics from the
/// matcher (observed: `tinyvec::ArrayVec` overflow when the walker
/// pushes a long run of UTF-8 first-bytes without continuations)
/// surface as `PipelinePanicked` findings instead of killing the
/// worker thread.
fn run_case(
    rng: &mut SmallRng,
    schema: Value,
    walker_budget: usize,
    max_grammar_bytes: usize,
) -> Vec<Finding> {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    match catch_unwind(AssertUnwindSafe(|| {
        run_case_inner(rng, schema.clone(), walker_budget, max_grammar_bytes)
    })) {
        Ok(findings) => findings,
        Err(panic) => {
            let msg = panic_payload_to_string(panic);
            vec![Finding::PipelinePanicked {
                schema,
                bytes_so_far: vec![],
                panic_message: msg,
                stage: "run_case",
            }]
        }
    }
}

fn run_case_inner(
    rng: &mut SmallRng,
    schema: Value,
    walker_budget: usize,
    max_grammar_bytes: usize,
) -> Vec<Finding> {
    let gbnf_source = build_schema_grammar_source(&schema);
    if gbnf_source.len() > max_grammar_bytes {
        // Pre-flight oversize filter — these grammars routinely crash
        // the matcher with uncatchable stack overflows. Drop silently;
        // the schema generator's depth cap should make this rare.
        return vec![];
    }
    let grammar = match Grammar::parse(&gbnf_source) {
        Ok(g) => g,
        Err(GrammarError::RecursionLimit { .. }) => {
            // The parser bailed defensively on a deeply-nested
            // schema-derived grammar — by design, not a bug. Drop the
            // case silently so we don't fill the corpus with noise.
            // (The schema generator can produce these via stacked
            // anyOf+object recursion; bumping `--schema-depth` raises
            // their frequency.)
            return vec![];
        }
        Err(e) => {
            return vec![Finding::CompileBroken {
                schema,
                gbnf_source,
                error: format!("{e:?}"),
            }];
        }
    };
    let grammar = Arc::new(grammar);

    let Some(bytes) = walk_grammar(grammar.clone(), rng, walker_budget) else {
        return vec![]; // walker bailed; not a finding
    };

    let mut findings = Vec::new();

    // Oracle 1: serde_json. The grammar promises syntactically valid
    // JSON; if serde rejects, we have a Class 2.
    let value: Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(e) => {
            findings.push(Finding::GrammarAcceptsSerdeRejects {
                schema,
                bytes,
                serde_error: format!("{e}"),
            });
            return findings;
        }
    };

    // Oracle 2: jsonschema. The grammar should encode the schema's
    // constraints; if jsonschema flags violations, the grammar
    // over-relaxed.
    //
    // jsonschema requires its compiled validator. Compilation can
    // itself fail on schemas that are valid JSON but not valid JSON
    // Schema (e.g. our adversarial $ref shapes). When validator
    // compilation fails, we *don't* call that a Class 3 — it just means
    // we have no oracle for this schema, so we skip the check.
    //
    // FUZZ_SKIP_JSONSCHEMA=1 disables this oracle entirely. Useful
    // when jsonschema itself stack-overflows on certain schemas (we
    // saw this on `{"items":{"type":"integer"},"type":"array"}` early
    // in development; pinning the cause is a follow-up).
    if std::env::var_os("FUZZ_SKIP_JSONSCHEMA").is_some() {
        return findings;
    }
    if let Ok(validator) = jsonschema::validator_for(&schema) {
        if !validator.is_valid(&value) {
            let errors: Vec<String> = validator
                .iter_errors(&value)
                .map(|e| format!("{e}"))
                .take(8)
                .collect();
            findings.push(Finding::GrammarAcceptsSchemaRejects {
                schema,
                bytes,
                value,
                validation_errors: errors,
            });
        }
    }

    findings
}

/// `until` subcommand: single-case runner. Same `catch_unwind` safety
/// net as `run_case` — the matcher can in principle blow its
/// recursion / `pending`-buffer limits on adversarial input, same as
/// the schema pipeline, so panics surface as `UntilPanicked` findings
/// instead of taking down a worker thread.
fn run_until_case(
    rng: &mut SmallRng,
    delim: &str,
    walker_budget: usize,
) -> Vec<Finding> {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    match catch_unwind(AssertUnwindSafe(|| {
        run_until_case_inner(rng, delim, walker_budget)
    })) {
        Ok(findings) => findings,
        Err(panic) => vec![Finding::UntilPanicked {
            delim: delim.to_string(),
            panic_message: panic_payload_to_string(panic),
            stage: "run_until_case",
        }],
    }
}

/// The two differential directions (requirements 3 and 4):
///
/// 1. Feed a generated `input` directly to the compiled grammar and
///    compare accept/reject against the naive matcher.
/// 2. Walk the grammar to generate strings it considers complete, and
///    assert the naive matcher agrees they're accepted — the
///    direction unit tests can't cover cheaply since it requires
///    exploring the automaton's own reachable states rather than
///    guessing them via random splicing.
fn run_until_case_inner(
    rng: &mut SmallRng,
    delim: &str,
    walker_budget: usize,
) -> Vec<Finding> {
    let mut rules = String::new();
    emit_until_rules("u", delim, &mut rules);
    let gbnf_source = format!("root ::= u\n{rules}");

    let grammar = match Grammar::parse(&gbnf_source) {
        Ok(g) => Arc::new(g),
        Err(e) => {
            return vec![Finding::UntilCompileBroken {
                delim: delim.to_string(),
                gbnf_source,
                error: format!("{e:?}"),
            }];
        }
    };

    let mut findings = Vec::new();

    // Direction 1: direct-feed vs naive matcher.
    let input = gen_until_input(rng, delim);
    let naive = input.ends_with(delim)
        && input.find(delim) == Some(input.len() - delim.len());
    let mut state = GrammarState::new(grammar.clone());
    let by_grammar =
        state.advance_bytes(input.as_bytes()).is_ok() && state.is_complete();
    if by_grammar != naive {
        findings.push(Finding::UntilMismatch {
            delim: delim.to_string(),
            input,
            naive,
            by_grammar,
        });
    }

    // Direction 2: walk the grammar to generate an accepted string,
    // then check the naive matcher agrees. The walker samples
    // ASCII-only first bytes (see `bitmap_to_bytes`), so it never
    // explores a transition through a multi-byte delimiter char — but
    // it still exercises every ASCII-reachable KMP state, which is
    // exactly what direction 1's random splicing can under-sample.
    if let Some(bytes) = walk_grammar(grammar, rng, walker_budget) {
        if let Ok(generated) = String::from_utf8(bytes) {
            let naive_accepts = generated.ends_with(delim)
                && generated.find(delim) == Some(generated.len() - delim.len());
            if !naive_accepts {
                findings.push(Finding::UntilWalkerMismatch {
                    delim: delim.to_string(),
                    generated,
                });
            }
        }
        // Non-UTF8 bytes shouldn't happen (ASCII-only walker sampling
        // plus valid-UTF8 delimiter chars re-encoded), but if it ever
        // does there's nothing to check against a `str`-based naive
        // matcher — skip silently rather than force a spurious finding.
    }

    findings
}

/// Sync-write a one-line trace to FUZZ_TRACE_PATH (default
/// `/tmp/fuzz_case_trace.log`). Used while hunting an uncatchable
/// crash — every line is fsync'd so the file always reflects what
/// happened up to and including the line before the crash.
fn sync_trace(msg: &str) {
    let path = std::env::var("FUZZ_TRACE_PATH")
        .unwrap_or_else(|_| "/tmp/fuzz_case_trace.log".to_string());
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = writeln!(f, "{msg}");
        let _ = f.sync_all();
    }
}

fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(&mut s, "{b:02x}");
    }
    s
}

// ───────────────────────────────────────────────────────────────────────
// Corpus writer (atomic, dedup-by-hash)
// ───────────────────────────────────────────────────────────────────────

fn ensure_corpus_dirs(root: &Path) -> std::io::Result<()> {
    for sub in [
        "class1_compile_broken",
        "class2_grammar_accepts_serde_rejects",
        "class3_grammar_accepts_schema_rejects",
        "class4_pipeline_panicked",
        "class5_until_compile_broken",
        "class6_until_mismatch",
        "class7_until_walker_mismatch",
        "class8_until_panicked",
    ] {
        fs::create_dir_all(root.join(sub))?;
    }
    Ok(())
}

/// Write a finding to `root/<class>/<hash>.json`. Returns `true` iff
/// the file is freshly created (new bug) and `false` iff the same
/// signature was already on disk (dup of a known bug).
///
/// **Dedup signature** is by bug-class fingerprint, NOT raw bytes. Two
/// `serde_reject: number out of range` cases on integer-typed fields
/// of the same schema shape collapse to one file even though the byte
/// streams differ (`5E11` vs `5E12`). Without this, a single bug class
/// produces O(10K) "unique" entries per minute.
fn write_finding(root: &Path, finding: &Finding) -> std::io::Result<bool> {
    let body = finding.to_json();
    let pretty = serde_json::to_vec_pretty(&body).expect("serialize");
    let h = signature_hash(finding);
    let path = root
        .join(finding.class_dir())
        .join(format!("{h:016x}.json"));
    if path.exists() {
        return Ok(false);
    }
    // Atomic write: tmp file alongside, then rename. If we crash, no
    // half-written finding ever appears.
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, &pretty)?;
    fs::rename(&tmp, &path)?;
    Ok(true)
}

/// Hash a finding by its bug-class signature, not its raw payload.
/// Tunable knob: the goal is to map "the same bug expressed two
/// different ways" to the same hash so the corpus doesn't pile up
/// thousands of near-duplicates.
fn signature_hash(f: &Finding) -> u64 {
    let mut h = DefaultHasher::new();
    f.class_dir().hash(&mut h);
    match f {
        Finding::CompileBroken { error, .. } => {
            // `error` includes the parser's structured GrammarError
            // discriminant + cursor position; the position varies
            // per-input but the discriminant is the bug. Hash just the
            // first 32 chars so cursor noise doesn't fragment.
            error.chars().take(32).collect::<String>().hash(&mut h);
        }
        Finding::GrammarAcceptsSerdeRejects {
            schema,
            serde_error,
            ..
        } => {
            // serde_error includes `at line X column Y`; strip that
            // tail and hash just the message stem ("number out of
            // range", "expected ',' or '}'", ...).
            let stem: String = serde_error
                .split(" at ")
                .next()
                .unwrap_or(serde_error)
                .chars()
                .take(96)
                .collect();
            stem.hash(&mut h);
            // Plus the *shape* of the schema (its top-level type), so
            // two different schemas exhibiting the same serde error
            // each get their own bucket.
            schema_shape(schema).hash(&mut h);
        }
        Finding::GrammarAcceptsSchemaRejects {
            schema,
            validation_errors,
            ..
        } => {
            // jsonschema errors are like "<value> is not of type
            // \"integer\"" — the value varies wildly per case but the
            // *bug* is the keyword + target. Strip the leading value
            // before hashing so `5 is not of type "integer"` and
            // `false is not of type "integer"` collapse to one
            // signature.
            let stem = validation_errors
                .first()
                .map(|s| validator_error_stem(s))
                .unwrap_or_default();
            stem.hash(&mut h);
            schema_shape(schema).hash(&mut h);
        }
        Finding::PipelinePanicked {
            panic_message,
            stage,
            ..
        } => {
            stage.hash(&mut h);
            panic_message
                .chars()
                .take(64)
                .collect::<String>()
                .hash(&mut h);
        }
        Finding::UntilCompileBroken { error, .. } => {
            // Same reasoning as `CompileBroken`: hash the error
            // discriminant, not the delimiter that triggered it — many
            // delimiters can hit the same parser bug.
            error.chars().take(32).collect::<String>().hash(&mut h);
        }
        Finding::UntilMismatch {
            delim,
            naive,
            by_grammar,
            ..
        } => {
            // The literal `input` varies per-case but isn't the bug;
            // the bug is characterized by which direction the verdicts
            // disagreed in, plus the delimiter's length and whether it
            // self-overlaps (a border, e.g. "aa"/"aba") — self-overlap
            // is what actually exercises the KMP fallback logic in
            // `emit_until_rules`, so it's the more meaningful bucket
            // than the delimiter's literal characters.
            naive.hash(&mut h);
            by_grammar.hash(&mut h);
            delim.chars().count().hash(&mut h);
            until_delim_has_border(delim).hash(&mut h);
        }
        Finding::UntilWalkerMismatch { delim, .. } => {
            delim.chars().count().hash(&mut h);
            until_delim_has_border(delim).hash(&mut h);
        }
        Finding::UntilPanicked {
            panic_message,
            stage,
            ..
        } => {
            stage.hash(&mut h);
            panic_message
                .chars()
                .take(64)
                .collect::<String>()
                .hash(&mut h);
        }
    }
    h.finish()
}

/// Whether `delim` has a nontrivial border — a proper prefix that is
/// also a suffix (e.g. `"aba"`: `"a"` is both). Borders are exactly
/// what makes the KMP fallback transition in `emit_until_rules`
/// nontrivial (state doesn't reset fully to 0 on a mismatch), so
/// bucketing findings on this is a better dedup signal than the
/// delimiter's literal text.
fn until_delim_has_border(delim: &str) -> bool {
    let chars: Vec<char> = delim.chars().collect();
    let n = chars.len();
    (1..n).any(|len| chars[..len] == chars[n - len..])
}

/// Extract the *keyword tail* from a `jsonschema` validator error.
/// Errors take a few stable shapes; we want the part that names the
/// constraint, not the value that violated it:
///
/// | Raw error | Returned stem |
/// |---|---|
/// | `5 is not of type "integer"` | `is not of type "integer"` |
/// | `8 is not one of ["a","b"]` | `is not one of [...]` |
/// | `-391 was expected` | `was expected` (const mismatch) |
/// | `null is not valid under any of the schemas listed in the 'anyOf' keyword` | `is not valid under any of the schemas listed in the 'anyOf' keyword` |
/// | `"foo" is a required property` | `is a required property` |
/// | `"abc" does not match "<pattern>"` | `does not match "<pattern>"` |
/// | `7 is greater than the maximum of N` | `is greater than the maximum of <N>` |
/// | `2 is less than the minimum of N` | `is less than the minimum of <N>` |
fn validator_error_stem(err: &str) -> String {
    // Find the first " is ", " was ", or " does " — the boundary
    // between value and constraint description in every error shape
    // we've observed.
    for sep in [" is ", " was ", " does "] {
        if let Some(idx) = err.find(sep) {
            let tail = &err[idx + 1..];
            // Collapse value-bearing payloads so two errors that
            // differ only in their interpolated values share a
            // signature. Numeric bounds payloads ("of N") get
            // collapsed too.
            let collapsed = collapse_value_payloads(tail);
            // Singular/plural normalization: "1 character" and "5
            // characters" collapse to the same stem.
            let normalized = collapsed.replace("characters", "character");
            return normalized.chars().take(96).collect();
        }
    }
    err.chars().take(96).collect()
}

/// Consume a sign/decimal/exponent number from a `Peekable<Chars>`,
/// discarding the matched chars. Used by `collapse_value_payloads` to
/// skip past numeric tails when collapsing them.
fn consume_numeric(chars: &mut std::iter::Peekable<std::str::Chars>) {
    while let Some(&peek) = chars.peek() {
        if peek.is_ascii_digit()
            || peek == '-'
            || peek == '+'
            || peek == '.'
            || peek == 'e'
            || peek == 'E'
        {
            chars.next();
        } else {
            break;
        }
    }
}

/// Replace `[...]`, `'...'`, and `of <number>` payloads with
/// placeholders so two errors that differ only in their interpolated
/// values collapse to the same signature.
fn collapse_value_payloads(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '[' {
            out.push_str("[…]");
            for c2 in chars.by_ref() {
                if c2 == ']' {
                    break;
                }
            }
        } else if c == '\'' {
            out.push_str("'…'");
            for c2 in chars.by_ref() {
                if c2 == '\'' {
                    break;
                }
            }
        } else if c == 'o'
            && out.ends_with(' ')
            && chars.clone().take(2).collect::<String>() == "f "
        {
            // " of <number>" → " of <…>". Numeric-bounds errors
            // ("greater than the maximum of 34") would otherwise
            // fragment per the bound value.
            out.push('o');
            chars.next(); // 'f'
            out.push_str("f <…>");
            chars.next(); // ' '
            consume_numeric(&mut chars);
        } else if c.is_ascii_digit() && out.ends_with(' ') {
            // " <digits> character[s]" → " <…> character[s]". minLength/
            // maxLength errors ("is shorter than 5 characters") would
            // otherwise fragment per the bound value.
            let mut peek_chars = chars.clone();
            // Skip past trailing digits.
            while peek_chars.peek().is_some_and(|p| p.is_ascii_digit()) {
                peek_chars.next();
            }
            // If the next bytes are " character" we're in the bound
            // pattern; collapse and skip past the digits.
            let lookahead: String = peek_chars.clone().take(11).collect();
            if lookahead.starts_with(" character") {
                out.push_str("<…>");
                while chars.peek().is_some_and(|p| p.is_ascii_digit()) {
                    chars.next();
                }
            } else {
                out.push(c);
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Coarse fingerprint of a schema's structure. Trades collision risk
/// for dedup quality — schemas with the same top-level type and
/// roughly the same set of property types collapse together.
fn schema_shape(schema: &Value) -> String {
    match schema {
        Value::Object(map) => {
            let t = map.get("type").and_then(|v| v.as_str()).unwrap_or("?");
            let kind = if map.contains_key("anyOf") {
                "anyOf"
            } else if map.contains_key("enum") {
                "enum"
            } else if map.contains_key("const") {
                "const"
            } else {
                t
            };
            kind.to_string()
        }
        _ => "leaf".to_string(),
    }
}

// ───────────────────────────────────────────────────────────────────────
// Pure mode driver
// ───────────────────────────────────────────────────────────────────────

struct Stats {
    cases: AtomicU64,
    findings: AtomicU64,
    new_findings: AtomicU64,
    write_errors: AtomicU64,
}

impl Stats {
    fn new() -> Self {
        Self {
            cases: AtomicU64::new(0),
            findings: AtomicU64::new(0),
            new_findings: AtomicU64::new(0),
            write_errors: AtomicU64::new(0),
        }
    }
}

fn run_pure(args: PureArgs) -> Result<(), Box<dyn std::error::Error>> {
    ensure_corpus_dirs(&args.corpus)?;
    // Pathological grammars can recurse fairly deep in `Grammar::parse`
    // and the matcher; bump the worker stack from rayon's default to
    // give panics-from-recursion more room before they turn into
    // uncatchable stack overflows.
    let mut pool_builder =
        rayon::ThreadPoolBuilder::new().stack_size(128 * 1024 * 1024);
    if let Some(n) = args.threads {
        pool_builder = pool_builder.num_threads(n);
    }
    pool_builder.build_global()?;
    let started = Instant::now();
    let stats = Arc::new(Stats::new());
    let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));

    // Heartbeat thread: every 30 s prints cumulative counts so a tail
    // of stdout makes a reasonable status display.
    let heartbeat_handle = {
        let stats = stats.clone();
        let stop_flag = stop_flag.clone();
        std::thread::spawn(move || {
            while !stop_flag.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_secs(30));
                let cases = stats.cases.load(Ordering::Relaxed);
                let findings = stats.findings.load(Ordering::Relaxed);
                let new_f = stats.new_findings.load(Ordering::Relaxed);
                let werr = stats.write_errors.load(Ordering::Relaxed);
                let elapsed = started.elapsed().as_secs_f64();
                let rate = cases as f64 / elapsed.max(1e-9);
                eprintln!(
                    "[fuzz t+{elapsed:>6.0}s] cases={cases} ({rate:.1}/s) findings={findings} unique={new_f} write_errors={werr}"
                );
            }
        })
    };

    let cases_cap = args.cases;
    let duration_cap = args.duration;
    let walker_budget = args.walker_budget;
    let schema_depth = args.schema_depth;
    let max_grammar_bytes = args.max_grammar_bytes;
    let corpus = args.corpus.clone();
    let seed = args.seed;
    let trace_cases = args.trace_cases;

    rayon::scope(|s| {
        let workers = rayon::current_num_threads();
        for tid in 0..workers {
            let stop_flag = stop_flag.clone();
            let stats = stats.clone();
            let corpus = corpus.clone();
            s.spawn(move |_| {
                let mut rng =
                    SmallRng::seed_from_u64(seed.wrapping_add(tid as u64));
                loop {
                    if stop_flag.load(Ordering::Relaxed) {
                        return;
                    }
                    if let Some(cap) = cases_cap {
                        if stats.cases.load(Ordering::Relaxed) >= cap {
                            stop_flag.store(true, Ordering::Relaxed);
                            return;
                        }
                    }
                    if let Some(d) = duration_cap {
                        if started.elapsed() >= d {
                            stop_flag.store(true, Ordering::Relaxed);
                            return;
                        }
                    }
                    let case_n =
                        stats.cases.fetch_add(1, Ordering::Relaxed) + 1;
                    if trace_cases {
                        sync_trace(&format!("case={case_n} pre-gen"));
                    }
                    // Top-level always an object so the schema looks
                    // like a real tool input_schema. Sub-properties go
                    // through gen_schema with depth-1 and can be any
                    // shape.
                    let schema = gen_object(&mut rng, schema_depth);
                    if trace_cases {
                        sync_trace(&format!(
                            "case={case_n} post-gen schema={}",
                            serde_json::to_string(&schema).unwrap_or_default(),
                        ));
                    }
                    let findings = run_case(
                        &mut rng,
                        schema,
                        walker_budget,
                        max_grammar_bytes,
                    );
                    for f in findings {
                        stats.findings.fetch_add(1, Ordering::Relaxed);
                        match write_finding(&corpus, &f) {
                            Ok(true) => {
                                stats
                                    .new_findings
                                    .fetch_add(1, Ordering::Relaxed);
                                eprintln!(
                                    "[fuzz t+{:>6.0}s] NEW finding: {} -> {}",
                                    started.elapsed().as_secs_f64(),
                                    f.class_dir(),
                                    short_summary(&f),
                                );
                            }
                            Ok(false) => { /* dup, silent */ }
                            Err(e) => {
                                eprintln!("write_finding error: {e}");
                                stats
                                    .write_errors
                                    .fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                }
            });
        }
    });

    stop_flag.store(true, Ordering::Relaxed);
    let _ = heartbeat_handle.join();

    let elapsed = started.elapsed().as_secs_f64();
    let cases = stats.cases.load(Ordering::Relaxed);
    let findings = stats.findings.load(Ordering::Relaxed);
    let new_f = stats.new_findings.load(Ordering::Relaxed);
    eprintln!(
        "\n=== fuzz done in {elapsed:.0}s: cases={cases} ({:.1}/s) findings={findings} unique={new_f} ===",
        cases as f64 / elapsed.max(1e-9)
    );
    Ok(())
}

// ───────────────────────────────────────────────────────────────────────
// `until` mode driver
// ───────────────────────────────────────────────────────────────────────
//
// Structurally the same shape as `run_pure` (rayon scope over worker
// threads, shared `Stats`, periodic heartbeat, duration/cases caps) —
// reuses those exact helpers (`Stats`, `write_finding`,
// `ensure_corpus_dirs`, `short_summary`). The only difference is what
// each case generates and checks: a random delimiter + `run_until_case`
// instead of a random schema + `run_case`.

fn run_until(args: UntilArgs) -> Result<(), Box<dyn std::error::Error>> {
    ensure_corpus_dirs(&args.corpus)?;
    let mut pool_builder =
        rayon::ThreadPoolBuilder::new().stack_size(128 * 1024 * 1024);
    if let Some(n) = args.threads {
        pool_builder = pool_builder.num_threads(n);
    }
    pool_builder.build_global()?;
    let started = Instant::now();
    let stats = Arc::new(Stats::new());
    let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));

    let heartbeat_handle = {
        let stats = stats.clone();
        let stop_flag = stop_flag.clone();
        std::thread::spawn(move || {
            while !stop_flag.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_secs(30));
                let cases = stats.cases.load(Ordering::Relaxed);
                let findings = stats.findings.load(Ordering::Relaxed);
                let new_f = stats.new_findings.load(Ordering::Relaxed);
                let werr = stats.write_errors.load(Ordering::Relaxed);
                let elapsed = started.elapsed().as_secs_f64();
                let rate = cases as f64 / elapsed.max(1e-9);
                eprintln!(
                    "[until t+{elapsed:>6.0}s] cases={cases} ({rate:.1}/s) findings={findings} unique={new_f} write_errors={werr}"
                );
            }
        })
    };

    let cases_cap = args.cases;
    let duration_cap = args.duration;
    let walker_budget = args.walker_budget;
    let corpus = args.corpus.clone();
    let seed = args.seed;

    rayon::scope(|s| {
        let workers = rayon::current_num_threads();
        for tid in 0..workers {
            let stop_flag = stop_flag.clone();
            let stats = stats.clone();
            let corpus = corpus.clone();
            s.spawn(move |_| {
                let mut rng =
                    SmallRng::seed_from_u64(seed.wrapping_add(tid as u64));
                loop {
                    if stop_flag.load(Ordering::Relaxed) {
                        return;
                    }
                    if let Some(cap) = cases_cap {
                        if stats.cases.load(Ordering::Relaxed) >= cap {
                            stop_flag.store(true, Ordering::Relaxed);
                            return;
                        }
                    }
                    if let Some(d) = duration_cap {
                        if started.elapsed() >= d {
                            stop_flag.store(true, Ordering::Relaxed);
                            return;
                        }
                    }
                    stats.cases.fetch_add(1, Ordering::Relaxed);

                    let delim = gen_until_delim(&mut rng);
                    let findings =
                        run_until_case(&mut rng, &delim, walker_budget);
                    for f in findings {
                        stats.findings.fetch_add(1, Ordering::Relaxed);
                        match write_finding(&corpus, &f) {
                            Ok(true) => {
                                stats
                                    .new_findings
                                    .fetch_add(1, Ordering::Relaxed);
                                eprintln!(
                                    "[until t+{:>6.0}s] NEW finding: {} -> {}",
                                    started.elapsed().as_secs_f64(),
                                    f.class_dir(),
                                    short_summary(&f),
                                );
                            }
                            Ok(false) => { /* dup, silent */ }
                            Err(e) => {
                                eprintln!("write_finding error: {e}");
                                stats
                                    .write_errors
                                    .fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                }
            });
        }
    });

    stop_flag.store(true, Ordering::Relaxed);
    let _ = heartbeat_handle.join();

    let elapsed = started.elapsed().as_secs_f64();
    let cases = stats.cases.load(Ordering::Relaxed);
    let findings = stats.findings.load(Ordering::Relaxed);
    let new_f = stats.new_findings.load(Ordering::Relaxed);
    eprintln!(
        "\n=== until done in {elapsed:.0}s: cases={cases} ({:.1}/s) findings={findings} unique={new_f} ===",
        cases as f64 / elapsed.max(1e-9)
    );
    Ok(())
}

fn short_summary(f: &Finding) -> String {
    match f {
        Finding::CompileBroken { error, .. } => {
            format!("compile_broken: {}", first_line(error))
        }
        Finding::GrammarAcceptsSerdeRejects {
            serde_error, bytes, ..
        } => {
            format!(
                "serde_reject: {} | bytes={}",
                first_line(serde_error),
                preview(bytes)
            )
        }
        Finding::GrammarAcceptsSchemaRejects {
            validation_errors,
            bytes,
            ..
        } => {
            let first = validation_errors
                .first()
                .map(|s| first_line(s))
                .unwrap_or_else(|| "(no error)".into());
            format!("schema_reject: {} | bytes={}", first, preview(bytes))
        }
        Finding::PipelinePanicked {
            panic_message,
            stage,
            ..
        } => {
            format!("panic@{stage}: {}", first_line(panic_message))
        }
        Finding::UntilCompileBroken { delim, error, .. } => {
            format!(
                "until_compile_broken: delim={delim:?} {}",
                first_line(error)
            )
        }
        Finding::UntilMismatch {
            delim,
            input,
            naive,
            by_grammar,
        } => {
            format!(
                "until_mismatch: delim={delim:?} input={:?} naive={naive} by_grammar={by_grammar}",
                preview(input.as_bytes())
            )
        }
        Finding::UntilWalkerMismatch { delim, generated } => {
            format!(
                "until_walker_mismatch: delim={delim:?} generated={:?}",
                preview(generated.as_bytes())
            )
        }
        Finding::UntilPanicked {
            delim,
            panic_message,
            stage,
        } => {
            format!(
                "until_panic@{stage}: delim={delim:?} {}",
                first_line(panic_message)
            )
        }
    }
}

fn first_line(s: &str) -> String {
    s.lines().next().unwrap_or(s).chars().take(120).collect()
}

fn preview(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).chars().take(60).collect()
}

// ───────────────────────────────────────────────────────────────────────
// Replay mode
// ───────────────────────────────────────────────────────────────────────

fn run_replay(args: ReplayArgs) -> Result<(), Box<dyn std::error::Error>> {
    let body: Value = serde_json::from_slice(&fs::read(&args.finding)?)?;
    let class = body.get("class").and_then(|v| v.as_str()).unwrap_or("");
    let schema = body.get("schema").cloned().unwrap_or(Value::Null);
    let bytes_hex =
        body.get("bytes_hex").and_then(|v| v.as_str()).unwrap_or("");
    let bytes = hex_to_bytes(bytes_hex);

    println!("=== replay {} ===", args.finding.display());
    println!("class    : {class}");
    println!("schema   : {}", serde_json::to_string_pretty(&schema)?);
    println!("bytes    : {}", String::from_utf8_lossy(&bytes));

    let gbnf_source = build_schema_grammar_source(&schema);
    println!("\n--- compiled GBNF ---\n{gbnf_source}");

    let grammar = match Grammar::parse(&gbnf_source) {
        Ok(g) => Arc::new(g),
        Err(e) => {
            println!("\n[Grammar::parse FAILED] {e:?}");
            return Ok(());
        }
    };
    let mut state = GrammarState::new(grammar);
    match state.advance_bytes(&bytes) {
        Ok(()) if state.is_complete() => {
            println!("\n[grammar accepts bytes — case still reproduces]")
        }
        Ok(()) => println!("\n[grammar partially advanced; not complete]"),
        Err(e) => println!("\n[grammar REJECTS bytes — bug fixed?] {e:?}"),
    }

    if !bytes.is_empty() {
        match serde_json::from_slice::<Value>(&bytes) {
            Ok(v) => println!("[serde_json: OK] {v:#?}"),
            Err(e) => println!("[serde_json: REJECTS] {e}"),
        }
    }
    Ok(())
}

fn hex_to_bytes(s: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len() / 2);
    let bytes = s.as_bytes();
    let mut i = 0;
    while i + 1 < bytes.len() {
        if let (Some(hi), Some(lo)) = (
            (bytes[i] as char).to_digit(16),
            (bytes[i + 1] as char).to_digit(16),
        ) {
            out.push(((hi << 4) | lo) as u8);
        }
        i += 2;
    }
    out
}

// ───────────────────────────────────────────────────────────────────────
// Report mode
// ───────────────────────────────────────────────────────────────────────

fn run_report(args: ReportArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut totals: Vec<(String, usize)> = Vec::new();
    for sub in [
        "class1_compile_broken",
        "class2_grammar_accepts_serde_rejects",
        "class3_grammar_accepts_schema_rejects",
    ] {
        let dir = args.corpus.join(sub);
        let n = fs::read_dir(&dir).map(|it| it.count()).unwrap_or(0);
        totals.push((sub.into(), n));
    }
    println!("=== fuzz-corpus report ({}) ===", args.corpus.display());
    let total: usize = totals.iter().map(|(_, n)| n).sum();
    for (k, n) in &totals {
        println!("  {n:>6}  {k}");
    }
    println!("  ------");
    println!("  {total:>6}  total unique findings");
    Ok(())
}

// ───────────────────────────────────────────────────────────────────────
// Main
// ───────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    match args.cmd {
        Cmd::Pure(a) => run_pure(a),
        Cmd::Until(a) => run_until(a),
        Cmd::Replay(a) => run_replay(a),
        Cmd::Report(a) => run_report(a),
    }
}
