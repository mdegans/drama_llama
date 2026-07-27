# `drama_llama`

<img src="https://raw.githubusercontent.com/mdegans/drama_llama/main/logo.svg" alt="llama with drama mask logo" width="240">

[![CI](https://github.com/mdegans/drama_llama/actions/workflows/ci.yml/badge.svg)](https://github.com/mdegans/drama_llama/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/mdegans/drama_llama/graph/badge.svg)](https://codecov.io/gh/mdegans/drama_llama)
[![tests](https://img.shields.io/badge/tests-635-blue)](#testing)
[![license](https://img.shields.io/badge/license-RAIL--S-lightgrey)](https://github.com/mdegans/drama_llama/blob/main/LICENSE.md)

`drama_llama` runs language models on your own hardware behind an API shaped
like Anthropic's Messages API. It speaks `misanthropic`'s `Prompt`, `Message`
and `Block` types directly — not a lookalike, the same types — so code written
against the Anthropic API drives a local GGUF by swapping the transport and
nothing else.

It is a **work in progress and not intended for production use**. The API
_will_ change.

The part worth your attention is what happens to *structured* output. When you
ask a hosted API for JSON matching a schema, you are asking politely. Here the
schema is compiled to a [GBNF] grammar and enforced inside the sampler, one
token at a time: tokens that would break the schema are removed from the
distribution before a choice is made. Malformed JSON is not unlikely, it is
unreachable.

```rust,no_run
// Compiled and type-checked by CI — not run, since it wants weights. The
// cfg gate keeps the doctest building when these features are off.
#[cfg(all(feature = "llama-cpp", feature = "json-schema"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use drama_llama::{
        FromPath, LlamaCppOptions, LlamaCppSession, Prompt, Role,
    };
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    /// Field order is generation order. `summary` is written first, so
    /// the model has already said what the bug *is* before it has to
    /// commit to a severity — each field is context for the next.
    #[derive(Debug, Serialize, Deserialize, JsonSchema)]
    struct Triage {
        /// One-line, imperative summary of the underlying problem.
        summary: String,
        /// How bad it is, chosen after summarizing.
        severity: Severity,
        /// Concrete, ordered steps to reproduce.
        repro_steps: Vec<String>,
        /// True when the report says the behavior regressed.
        is_regression: bool,
    }

    #[derive(Debug, Serialize, Deserialize, JsonSchema)]
    #[serde(rename_all = "snake_case")]
    #[schemars(rename_all = "snake_case")]
    enum Severity {
        Low,
        Medium,
        High,
        Critical,
    }

    let mut session = LlamaCppSession::from_path_with(
        "models/model.gguf".into(),
        LlamaCppOptions::default().with_n_ctx(8192),
    )?;

    let prompt = Prompt::default()
        .system("You triage incoming bug reports.")
        // A worked exemplar teaches field *depth* — that `repro_steps`
        // should be concrete and non-empty — which a bare schema cannot
        // express. It seeds the schema too, so the two cannot drift apart.
        .add_examples([(
            "Login does nothing in Safari. Started after last week's release.",
            Triage {
                summary: "Login button unresponsive on Safari".into(),
                severity: Severity::High,
                repro_steps: vec![
                    "Open the app in Safari".into(),
                    "Click 'Log in'; observe no network request".into(),
                ],
                is_regression: true,
            },
        )])?
        .add_message((Role::User, "Checkout total shows $0.00 on mobile."))?;

    // The response has the Anthropic shape — content, usage, stop
    // reason — and `.json()` parses its text block, skipping any
    // leading thought blocks. The parse cannot fail on malformed JSON:
    // the model was not able to emit any.
    let triage: Triage = session.complete_response(&prompt)?.json()?;
    println!("{triage:#?}");
    Ok(())
}

#[cfg(not(all(feature = "llama-cpp", feature = "json-schema")))]
fn main() {}
```

The grammar engine underneath is ours — a pure-Rust GBNF parser, matcher and
lazily-built DFA cache, not a call into `llama.cpp`'s. So it is usable on its
own, with no backend and no C dependency at all:

```rust
use drama_llama::{GrammarState, SamplingMode};

let gbnf = r#"
    root ::= "{" ws "\"ok\"" ws ":" ws bool ws "}"
    bool ::= "true" | "false"
    ws   ::= [ \t\n]*
"#;

// As a sampling mode this constrains generation token by token.
let mode = SamplingMode::grammar(gbnf).unwrap();
assert!(matches!(mode, SamplingMode::Grammar(_)));

// The same grammar, driven by hand. `completes_with` asks whether the
// bytes are accepted *and* land in a final state; `accepts_bytes` asks
// only whether they are a legal prefix. Neither mutates the matcher.
let state = GrammarState::from_source(gbnf).unwrap();
assert!(state.completes_with(br#"{"ok": true}"#));
assert!(state.accepts_bytes(br#"{"ok": "#));
assert!(!state.accepts_bytes(br#"{"ok": maybe"#));
```

[GBNF]: https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md

## The layers

You can enter at whichever level you need. Each is a thin, public wrapper over
the one below it.

| Layer | Type | What it gives you |
|---|---|---|
| 5 | `SessionTransport` / `LocalTransport` | Implements `misanthropic::Transport`, so `Chat` loops and agent reactors written for the API drive a local model unchanged. |
| 4 | `Session<B>` | The chat-shaped API: `complete`, `complete_text`, `complete_blocks`, `complete_stream`, `complete_response`. Owns templating, tool dialects, the prefix cache, and grammar resolution. |
| 3 | `Predictor` family | `predict_candidates`, `predict_tokens`, `predict_pieces`, `predict` — iterators. `CandidatePredictor::record_choice` lets *you* pick the token, which is how forced-continuation scoring works. |
| 2 | `Engine<B>` | Decoder + model + optional vision, plus direct KV-cache control (`memory_seq_rm`, `checkpoint_pos`, `restore_to`, …). |
| 1 | `Candidates`, `SamplerConfig` | Every sampling method, translated to Rust. No calls into `llama.cpp`'s sampler chain. |
| 0 | `backend` | `Backend`, `Decoder`, `Model`, `Vision` traits. Compiles with `--no-default-features`: no C dependency. |

## Supported features

| | Feature flag | |
|---|---|---|
| **Structured output** | `json-schema` | A `schemars`-derived type becomes a sampling grammar. Optional `<think>…</think>` preamble, phase-split so the thought runs unconstrained at full speed. |
| **Tool calling** | *(always on)* | Per-model dialects derived by analyzing each model's own chat template, driving both the grammar emitter and the response parser. `ToolChoice::method` is *guaranteed* locally, not requested. Validated for Qwen 3.5/3.6, Gemma 4, gpt-oss (Harmony). |
| **GBNF grammars** | *(always on)* | Pure-Rust parser, matcher and lazy-DFA cache. Sampling checks the one sampled token first and only falls back to an O(vocab) mask on rejection. |
| **Prefix caching** | *(always on, opt-in at runtime)* | Multi-slot, breakpoint-driven, LRU with TTL. Honors Anthropic `cache_control` ephemeral markers. One slot per agent, so an N-agent workload caches N prefixes instead of thrashing one. |
| **Chat templates** | *(always on)* | The model's own Jinja `tokenizer.chat_template`, rendered by `minijinja`. No per-model prompt formats hardcoded here. |
| **Images** | `media`, `mtmd` | `media` is pure Rust (decode via the `image` crate, never `mtmd`'s bundled `stb_image`); `mtmd` adds llama.cpp's multimodal backend. Images render out-of-band through a per-call random sentinel — the projector never sees prompt text. |
| **Sampling** | *(always on)* | Greedy, temperature, top-k, top-p, min-p, tail-free, locally typical, Mirostat v1/v2, plus `SplitP`/`SplitL`/`Deny` which have no llama.cpp counterpart. Chained: each mode narrows the candidate set. |
| **Repetition penalties** | *(always on)* | N-gram based, windowed and decaying, with category exclusions so common English or JSON punctuation isn't penalized. Region-aware inside grammar free-text spans. |
| **HTTP server** | `axum` | `blallama` — an Anthropic-compatible `/v1/messages` server over a local model, with an SSE `/probe` channel. |
| **Accelerators** | `cuda`, `cuda_f16` | Metal is automatic on macOS. |
| **Async** | `tokio` | `SessionTransport`, `FromPath::from_path_async`. |
| **Sidecars** | `toml` | Per-model `sampling.toml` / `dialect.toml` / template files beside the GGUF. |

### Backends

Two, behind one `Backend` trait:

- **`llama-cpp`** (default) — llama.cpp via [`llama-cpp-sys-3`]. CUDA and Metal.
- **`moeflux`** — a Metal-native streaming-MoE runtime, macOS only. Selects
  its model at compile time: exactly one of `moeflux-model-qwen3-6-35b-a3b`,
  `moeflux-model-qwen3-5-a17b`, or `moeflux-model-cogito-v2-671b`. The last is
  ~336 GB at 4-bit and streams experts from SSD, which is how a 671B model runs
  on a 96 GB laptop.

Both can be linked at once. When they are, name the alias (`LlamaCppSession`)
rather than a bare `Session` — a bare `Session::from_path` only infers a
backend when exactly one exists.

[`llama-cpp-sys-3`]: https://github.com/mdegans/llama-cpp-sys

## Examples

Eighteen of them in [`examples/`]. Each carries a module doc explaining not
just what it does but why it is shaped that way. The ones worth reading first:

| Example | |
|---|---|
| [`strawberry`] | Typed tool use with the `#[tool]` macro. Locally, `ToolChoice::method` compiles to a grammar, so the call is guaranteed. |
| [`whodunit`] | Structured output into a typed `CaseFile`, streamed block by block so thoughts arrive as they parse. |
| [`prompt_caching`] | The prefix cache, demonstrated self-referentially: the system prompt embeds this README and the transport's own source, then asks about them. |
| [`swarm`] | Five agents, one GPU, a `#[tool]`-built mail system and a postage ledger. One cache slot per seat. |
| [`whoami`] | The raw `Engine` + `CandidatePredictor` layer: scores candidate model names by forced continuation, reading the distribution instead of the string. |
| [`unhelpful`] | Steering by prefilling the model's *own* reasoning with an unclosed thought block. |

```sh
just example whodunit
cargo run --release --example strawberry --features "tokio,cli,json-schema"
```

[`examples/`]: https://github.com/mdegans/drama_llama/tree/main/examples
[`strawberry`]: https://github.com/mdegans/drama_llama/blob/main/examples/strawberry.rs
[`whodunit`]: https://github.com/mdegans/drama_llama/blob/main/examples/whodunit.rs
[`prompt_caching`]: https://github.com/mdegans/drama_llama/blob/main/examples/prompt_caching.rs
[`swarm`]: https://github.com/mdegans/drama_llama/blob/main/examples/swarm.rs
[`whoami`]: https://github.com/mdegans/drama_llama/blob/main/examples/whoami.rs
[`unhelpful`]: https://github.com/mdegans/drama_llama/blob/main/examples/unhelpful.rs

There are also three binaries: `blallama` (the HTTP server), `regurgitater`
(tests local models for [memorized content]), and `settings_tool` (an egui
sampler-settings editor).

[memorized content]: https://github.com/mdegans/drama_llama/blob/main/bin/regurgitater/README.md

## Testing

635 tests across 28 binaries in the default configuration — 513 that run in
seconds and 122 that load real weights onto a real accelerator. The
model-backed tier is `#[ignore]`d so the fast loop stays fast, and the whole
topology — *which features* × *which tests* — lives in one place,
[`scripts/test.py`]. The justfile delegates to that script, the git hooks call
the justfile, and CI calls the script directly, so the tests that gate a commit
are byte-for-byte the ones that gate a push.

(That is `cargo nextest list`'s count for the `llama-cpp` configuration, not a
grep for `#[test]`. The two disagree, and only one of them is what runs.)

```sh
just setup            # cargo-nextest + cargo-llvm-cov (once)
just install-hooks    # point git at .githooks/ (once)

just test             # the fast tier: no weights, fully parallel
just test ignored     # ONLY the model tests, serialized
just test all         # everything
just test moeflux     # the moeflux configuration, plus cross-backend
just test NAME        # anything matching NAME, any tier, uncaptured

just check            # rustfmt + rustdoc, what the pre-commit hook runs
just permutations     # every feature configuration compiles, test targets too
just doctest          # the doctests, including the ones on this page
just coverage         # instrumented run + report
```

Everything goes through [`cargo-nextest`], which gives each test its own
process. Do **not** use plain `cargo test`: it overlaps test *binaries*, which
`--test-threads=1` does not fix, so two 19 GB models load at once and the OOM
surfaces as a decode failure that reads like a regression.

Run `python3 scripts/test.py --help` for the real interface — and use it
directly on Windows, since the recipe bodies are bash.

[`scripts/test.py`]: https://github.com/mdegans/drama_llama/blob/main/scripts/test.py
[`cargo-nextest`]: https://nexte.st/

## Roadmap

- [ ] Automatic batch scheduling and better parallelism
- [ ] Runtime model-variant selection for moeflux, replacing the compile-time
  feature selection
- [ ] Stream `misanthropic::stream::Event` from `Session::complete_stream`
  ([#26](https://github.com/mdegans/drama_llama/issues/26))
- [ ] Tokenization in the browser
- [ ] Backends beyond llama.cpp and moeflux — an NPU target is the long-term
  goal

See [`CHANGELOG.md`] for what has already landed, and the [issue tracker] for
what is actively broken.

[`CHANGELOG.md`]: https://github.com/mdegans/drama_llama/blob/main/CHANGELOG.md
[issue tracker]: https://github.com/mdegans/drama_llama/issues

## Known issues

- A KV-dirty `llama_decode` failure leaves the cache unreconciled
  ([#52](https://github.com/mdegans/drama_llama/issues/52)).
- A context-full stop is reported as a grammar violation
  ([#36](https://github.com/mdegans/drama_llama/issues/36)).
- moeflux's `memory_seq_cp` / `memory_seq_keep` silently no-op and report
  success ([#42](https://github.com/mdegans/drama_llama/issues/42)).

## Contributing

- Code is poetry. Make it pretty.
- Respect is universal.
- Use `rustfmt` — `just install-hooks` makes that automatic.

## Generative AI Disclosure

- Generative AI, specifically Microsoft's Bing Copilot, GitHub Copilot, and
  Dall-E 3 were used for portions of this project. See inline comments for
  sections where generative AI was used. Completion was also used for getters,
  setters, and some tests. Logos were generated with Dall-E and post processed
  in Inkscape.
- Anthropic's Claude (primarily as Claude Code) is a direct collaborator
  on this project and co-authors commits where it contributed. `git log`
  is the authoritative record — grep for `Co-Authored-By: Claude` — and
  [`CONTRIBUTORS.md`] summarizes the surface areas. As
  of v0.8.0 those include the llama.cpp API migration, the sampling-mode
  suite (JSON, GBNF, tool-choice, structured output), the Jinja chat-
  template renderer, the prompt-caching layer, the grammar matcher
  performance finish line (lazy-DFA cache + thought/JSON phase-split),
  and the `Backend` split that lets the same `Session`/`Engine` surface
  drive either llama.cpp or moeflux's Metal MoE runtime.

[`CONTRIBUTORS.md`]: https://github.com/mdegans/drama_llama/blob/main/CONTRIBUTORS.md
