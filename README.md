# `drama_llama`

<img src="https://raw.githubusercontent.com/mdegans/drama_llama/main/logo.svg" alt="llama with drama mask logo" width="240">

[![CI](https://github.com/mdegans/drama_llama/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mdegans/drama_llama/actions/workflows/ci.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/mdegans/drama_llama/graph/badge.svg)](https://codecov.io/gh/mdegans/drama_llama)
[![tests](https://img.shields.io/badge/tests-1175-blue)](#testing)
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

## Installing

```toml
[dependencies]
drama_llama = "0.9"
```

The default feature is `llama-cpp`, which builds llama.cpp from source through
[`llama-cpp-sys-3`] (CMake, a C++ toolchain and libclang required). Add
features as you need them: `json-schema` for typed structured output, `tokio`
for the async transport, `cuda` on NVIDIA. The minimum supported Rust is 1.97.

To run the server instead of linking the library:

```sh
cargo install drama_llama --bin blallama --features "axum,cli,toml"
```

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
| **Tool calling** | *(always on)* | Per-model dialects derived by analyzing each model's own chat template, driving both the grammar emitter and the response parser. `ToolChoice::method` is *guaranteed* locally, not requested. Parallel calls, with an optional per-turn cap. Validated for the [supported models](#supported-models). |
| **GBNF grammars** | *(always on)* | Pure-Rust parser, matcher and lazy-DFA cache. Sampling checks the one sampled token first and only falls back to an O(vocab) mask on rejection. |
| **Prefix caching** | *(always on, opt-in at runtime)* | Multi-slot, breakpoint-driven, LRU with TTL. Honors Anthropic `cache_control` markers, including request-level automatic caching, with Anthropic's 400s for invalid ones. One slot per agent, so an N-agent workload caches N prefixes instead of thrashing one. Sliding-window (gpt-oss, Gemma 4) and hybrid (Qwen3.6/3.8) models restore from checkpoints under a host-RAM budget. Every reuse decision is logged. |
| **Chat templates** | *(always on)* | Rendered by `minijinja`. A template sidecar wins; otherwise a model whose embedded template we validated gets a *baked*, cache-stable replacement shipped in the crate (see [supported models](#supported-models)); anything else uses its own `tokenizer.chat_template`, with a warning. Prompt content that spells a special token is read as text, not as the token. |
| **Images** | `media`, `mtmd` | `media` is pure Rust (decode via the `image` crate, never `mtmd`'s bundled `stb_image`); `mtmd` adds llama.cpp's multimodal backend. Images render out-of-band through a per-call random sentinel — the projector never sees prompt text. |
| **Sampling** | *(always on)* | Greedy, temperature, top-k, top-p, min-p, tail-free, locally typical, Mirostat v1/v2, plus `SplitP`/`SplitL`/`Deny` which have no llama.cpp counterpart. Chained: each mode narrows the candidate set. |
| **Repetition penalties** | *(always on)* | N-gram based, windowed and decaying, seeded from the conversation's history. Category exclusions (`English`, `Json`, `Markdown`, `Numbers`, `Punctuation`) keep common words, syntax, code fences and numbers unpenalized; a known-id exemption spares faithful copies of ids from the prompt. Region-aware inside grammar free-text spans. |
| **HTTP server** | `axum` | `blallama` — an Anthropic-compatible `/v1/messages`, `/v1/messages/count_tokens` and `/v1/models` server over a directory of local models, with Anthropic's error types and an SSE `/probe` channel. See [Running `blallama`](#running-blallama). |
| **Accelerators** | `cuda`, `cuda_f16` | Metal is automatic on macOS. |
| **Async** | `tokio` | `SessionTransport`, `FromPath::from_path_async`. |
| **Sidecars** | `toml` | Per-model `sampling.toml`, `load.toml`, `dialect.toml`, template and `mmproj` files beside the GGUF. See [Sidecar files](#sidecar-files). |

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

## Supported models

Any GGUF llama.cpp loads will run, and its tool-call dialect is derived from
its own chat template. The models below are supported first-class: the crate
ships a *baked* replacement for each one's chat template, chosen when the
model's embedded template is byte-for-byte the one we validated. A baked
template re-renders what the model wrote byte for byte, which is what lets the
prefix cache continue a conversation instead of re-reading it. Each has its own
model-backed test suite.

| Model | Baked template | Tool calls | Reasoning |
|---|---|---|---|
| Qwen3.6 (35B-A3B) | `qwen3.6-cache-stable` | XML `<tool_call>` | `<think>`, inline |
| Qwen3.8 (27B) | `qwen3.8-cache-stable` | XML `<tool_call>` | `<think>`, as `reasoning_content` |
| Gemma 4 (31B IT) | `gemma4-cache-stable` | `<\|tool_call>` | thinking channel |
| gpt-oss (20b, 120b) | `gptoss-cache-stable`, `gptoss-upstream-cache-stable` | Harmony | analysis channel |
| Mistral Small 4 (119B) | `mistral4-cache-stable` | `[TOOL_CALLS]name[ARGS]` | `[THINK]` |
| cogito (14B, 32B) | `cogito-cache-stable` | Hermes JSON `<tool_call>` | `<think>` when `thinking` is on |

gpt-oss has two detection keys: the Unsloth-patched template and the upstream
OpenAI one. Qwen3.6, Qwen3.8, Gemma 4 and Mistral Small 4 accept images with
an `mmproj` sidecar under the `mtmd` feature. A model whose template we do not
recognize falls back to its embedded template with a warning; one that is
*almost* a baked one is named in the log, so a re-quant that touched its
template is easy to spot. A `<model>.template.jinja` sidecar always wins, and a
sidecar that is a copy of an older bake is warned about at load. The templates
and what each one fixes are documented in [`templates/README.md`].

[`templates/README.md`]: https://github.com/mdegans/drama_llama/blob/main/templates/README.md

## Examples

Twenty-three of them in [`examples/`]. Each carries a module doc explaining not
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

### Running `blallama`

Run `blallama` under a supervisor. It exits on purpose when it can no longer
trust its own process — code **70** after a panic on any thread, **75** after a
backend failure llama.cpp does not recover from in-process (a failed
`llama_decode`, a Metal context left in error state by an out-of-memory
command buffer, a model load that fails after the backend began allocating —
out of memory loading the weights or the KV cache) — rather than unwind through
llama.cpp state or keep serving a wedged backend. A load that fails before
anything is allocated (a missing file, an unsupported architecture, a bad
template) is answered with an error, and the server serves on. Before it goes it logs one `ERROR` line (`"event":"fatal"`,
with the `kind`, `exit_code` and `cause`) and answers the requests in flight
with a 500 `api_error`, which Anthropic's SDKs retry. launchd, systemd
(`Restart=on-failure`) or the restart loop in
[`scripts/blallama-supervise.sh`] all do; the script keeps the arguments, backs
off when it crash-loops, and logs each restart:

```sh
BLALLAMA=target/release/blallama scripts/blallama-supervise.sh models/ --port 11435
```

A clean exit (SIGTERM or Ctrl-C drains in-flight work, code 0) is not
restarted.

[`scripts/blallama-supervise.sh`]: https://github.com/mdegans/drama_llama/blob/main/scripts/blallama-supervise.sh

`blallama` serves every model in one directory: each `.gguf` (projector files
aside) is a model whose API id is its file name, `.gguf` included. One model is
loaded at a time, and a request naming another swaps it in. Requests are served
one at a time; one that arrives while another is generating gets a 529
`overloaded_error`, which Anthropic's SDKs retry.

```sh
cargo run --release --bin blallama --features "axum,cli,toml" -- models/ \
    --n-ctx 131072 --cache-slots 4
```

Point any Anthropic client at `http://127.0.0.1:11435`. It answers
`POST /v1/messages` (streaming or not), `POST /v1/messages/count_tokens`,
`GET /v1/models` and `GET /v1/models/{id}`, with Anthropic's error envelope and
status codes: a request that will fail the same way again is a 400
`invalid_request_error`, a transient failure a 500 `api_error` that the SDKs
retry. A prompt plus `max_tokens` that would not fit the context is refused up
front with Anthropic's `prompt is too long` 400. A turn that breaks its grammar
or schema, or loops on identical tool calls, is redrawn on the warm cache
before an error is returned.

| Flag | Default | |
|---|---|---|
| `--port` | 11435 | Port to listen on. |
| `--n-ctx` | 32768 | Maximum context length for every model, capped at each model's window; a `load.toml` sidecar can lower it. |
| `--cache-slots` | 1 | KV sequences over one cell pool: one cached prefix per concurrent agent. |
| `--swa-full` | off | Size sliding-window layers' KV at the full context instead of the window. |
| `--checkpoint-mib` / `--checkpoint-slot-mib` | 8192 / 4096 | Host RAM the prefix-cache checkpoints may hold, in all and per slot. |
| `--default-model` | none | Serve this model when a request names one that isn't on disk (e.g. a `claude-*` id). |
| `--seed` | fresh | Fixed RNG seed for every request: same prompt, same output. |
| `--no-penalty` | off | Turn the repetition penalty off whatever the sidecar says. |
| `--record-json` / `--probe-stream` | off | Per-token probe records to a JSONL file / an SSE `GET /probe` channel. |
| `--schema-max-*` | see `--help` | Limits on a request's tool and `output_config` schemas, checked before compiling; past one is a 400. |
| `--backend` | `llama-cpp` | `moeflux` in a build with that feature, serving model directories instead. |

### Sidecar files

Per-model settings live in files beside the GGUF, named after it
(`Qwen3.8-27B-UD-Q8_K_XL.gguf` → `Qwen3.8-27B-UD-Q8_K_XL.sampling.toml`). A
sidecar that does not read or parse is logged and ignored.

| File | |
|---|---|
| `<model>.sampling.toml` | Sampling defaults: the mode chain, the repetition penalty, the tool-call cap. Written on first load, seeded from the model's own recommended sampling where it has one; never overwritten. |
| `<model>.load.toml` | Load-time options: `n_ctx` (at most `--n-ctx` and the model's window), `n_ubatch`, and `rope_scale` (YaRN over the original window, stretching it). Never written; unknown keys are an error. |
| `<model>.template.jinja` | Replaces the chat template outright. |
| `<model>.dialect.toml` | Overrides the tool-call dialect derived from the template. |
| `<model>.mmproj.gguf` | The vision projector: image input, with the `mtmd` feature. |

The sampling sidecars for the supported models are versioned in [`models/`].
An abridged one:

```toml
# Qwen3.6-35B-A3B-UD-IQ4_XS.sampling.toml
lazy_grammar = true
# Optional: the most client tool calls one turn may make (absent: no cap).
# The turn ends on the model's own end-of-generation after the last one.
max_tool_calls_per_turn = 3

[[modes]]
[modes.TopK]
k = 1024

[[modes]]
[modes.LocallyTypical]
p = 0.9
min_keep = 3

[repetition]
# Token categories never penalized. `Markdown` spares code fences.
ignored_categories = ["English", "Json", "Markdown", "Numbers", "Punctuation"]
# What an identifier looks like. A faithful copy of one the prompt holds is
# never penalized, and with `id_copy_lock` (default true) a copy eight
# characters into exactly one known hex id must be finished as that id.
id_patterns = ["[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"]
id_copy_lock = true
window_size = 2048
penalty_repeat = 1.06
surgical = true
```

```toml
# Mistral-Small-4-119B-2603-UD-Q4_K_XL.load.toml, under --n-ctx 262144
n_ctx = 131072   # this model's own context, below --n-ctx (window: 1M)
n_ubatch = 2048  # llama.cpp's micro-batch (default 512)
# rope_scale = 4 # YaRN factor; for a model trained unscaled (Qwen3.5+:
#                # 262144 → 1M). Not for one whose GGUF ships scaling.
```

[`models/`]: https://github.com/mdegans/drama_llama/tree/main/models

## Testing

1175 tests across 33 binaries in the default configuration — 985 that run in
seconds and 190 that load real weights onto a real accelerator. The
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
  ([#52](https://github.com/mdegans/drama_llama/issues/52)); `blallama`
  exits on one for its supervisor to restart.
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
