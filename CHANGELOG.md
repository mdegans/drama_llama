# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.0] — Unreleased

Backend split. The chat-style API (`Session`), the engine layer
(`Engine`), the Predictor family, and the binary (`blallama`) are
all generic over a single `Backend` parameter. drama_llama can now
drive either llama.cpp or moeflux's Metal MoE runtime through the
same surface. Runs Cogito-class MoE models on Apple Silicon without
the Anthropic API as a dependency.

### Added

- **`SamplingMode::Deny { range: Range<Token> }`** — sample-time
  mask for forbidden token-id ranges. Constructor:
  `SamplingMode::deny_range(r)`. Filters candidates whose id falls
  in the range out of the set before any downstream mode runs;
  falls back to a single EOS if the range eats every candidate.
  Primary use case: tokenizer reserved/unused vocab tails (Qwen3:
  ~248088..248320). `Session` automatically prepends a Deny mode
  computed once at construction by scanning from the highest vocab
  id downward — empty-piece tokens trivially pass byte-stream
  grammar filters and would otherwise let the model land in a loop
  scattering reserved tokens after a structured response closes.
  See `.claude/memory/grammar_reserved_token_loop.md` for the
  full analysis.
- **`Model::extra_eos_tokens()`** — trait method exposing
  additional EOS-like token ids beyond `eos` and `eot`. Default
  empty; `MoefluxModel` overrides to expose the tail of the
  `eos_token_id` config array (Qwen3 declares `[<|im_end|>,
  <|endoftext|>]`; without this hook, `<|endoftext|>` would never
  reach `add_model_stops`).
- **`Model::display_name()`** — human-readable identifier for
  loaded models. `LlamaCppModel` returns the GGUF basename;
  `MoefluxModel` returns the parent dir's basename (overridden
  by `MoefluxEngine::from_path` to match the discovery-dir name).
- **`backend::Backend` trait** bundling `type Decoder: Decoder + Send`
  and `type Model: Model + Send + Sync` as a single generic
  parameter. Compile-time monomorphization, no `dyn` indirection on
  the hot path. ZST tag impls: `LlamaCppBackend`, `MoefluxBackend`.
- **`Model::display_name(&self) -> Option<String>`** on the trait.
  Both backends populate (GGUF basename / MLX-export dir basename).
  Used by `Session::complete_response` for the `model` field of
  responses, and by `blallama` for model-name matching.
- **`MoefluxEngine::from_path(parent: &Path)`** — convention-based
  wrapper around `from_paths`. Expects `parent/{mlx,artifacts,root}/`
  with sane runtime defaults (`experts_per_tok = 8`, `use_2bit =
  false`). Symmetric with `LlamaCppEngine::from_path` so binaries
  can take a single `--model <path>` arg for either backend. The
  5-arg `from_paths` stays for callers needing explicit paths or
  non-default runtime params.
- **`blallama --backend {llama-cpp|moeflux}`** flag with cfg-gated
  variants. `main()` dispatches once at startup; each backend half
  monomorphizes independently. llama-cpp build accepts only
  `llama-cpp`; moeflux build accepts only `moeflux`; combined build
  accepts both.
- **`drama_llama::sidecar` module** (gated on `feature = "toml"`):
  per-model sampling-config TOML files colocated with the model on
  disk. `Session::from_path*` looks for the sidecar, applies it via
  `with_sample_options`, and writes a default if none exists so
  there's a starting point to edit.
  - **GGUF (llama-cpp)**: sibling `<model>.sampling.toml` next to
    the `.gguf`.
  - **Moeflux**: `parent/sampling.toml` alongside the
    `mlx`/`artifacts`/`root` symlinks.
  - Reset = `rm <sidecar>`; tweak = edit it.
  - `Json`/`Grammar`/`Deny` modes are excluded from sidecars on
    purpose — those are runtime per-request constraints, not
    per-model defaults.
- **`Session::with_sample_options(SampleOptions)`** — wholesale
  setter used by sidecar loading. Sets the post-grammar sampling
  chain, repetition penalty, and any deferred grammar in one shot.
  Auto-extends `repetition.ignored` with the model's special
  tokens (matches `with_repetition` semantics) so a strong rep
  penalty can never lock out EOS / chat-template markers.
- **`Session::with_seed(Option<NonZeroU128>)`** — fixed RNG seed
  forwarded to every `predict_*` call. Makes tuning iteration
  meaningful: same prompt + same seed = same output, so a
  sidecar tweak shows up as a deliberate change rather than
  stochastic noise.
- **`RepetitionOptions::window_size: NonZeroU32`** (default 256)
  and **`RepetitionOptions::decay: f32`** (default 0.95).
  Together they bound the repetition-penalty additive contribution.
  Effective per-n-gram count is now
  `Σ decay^(current_step - position)` over occurrences inside the
  last `window_size` generation steps; bounded above by
  `1 / (1 - decay)`. Pre-fix the additive `count * penalty_freq`
  term grew linearly with generation length and dominated the
  model's natural logit gradient on long generations (~20 logits
  below baseline at 200 steps, ~60 at 600). With the fix the gap
  saturates once the window fills and stays put. See
  `.claude/memory/qwen3_long_form_degradation.md` for the analysis.
- **`NGramStats::evict_outside_window(current_step, window_size)`**
  and **`NGramData::windowed_decayed_count(current_step, decay)`** —
  the primitives backing the windowed-decay penalty path. Maintains
  the `count == positions.len()` invariant on each entry.
- **`blallama --no-penalty`** — force repetition penalty OFF, even
  when the per-model sidecar enables it. For probes, canary runs,
  and any "what does this model do with no penalty" diagnostic.
- **`blallama --seed <u128>`** — fixed RNG seed forwarded to every
  prediction. For tuning iteration where you want sidecar changes
  to show up as deliberate divergences rather than stochastic ones.

### Changed

- **`Session::run_call` now breaks generation on grammar accept**.
  When any active `SamplingMode::Grammar` / `SamplingMode::Json`
  matcher reaches its accept state, the call halts immediately
  instead of continuing to wait for EOS. Belt-and-suspenders with
  the Deny mask: Deny prevents reserved tokens from being sampled;
  break-on-accept terminates cleanly the moment the structured
  output is satisfied. Includes deferred-grammar phase-split paths
  (post-`</think>` JSON matchers terminate the same way once their
  root rule completes).
- **`Engine<D, M>` → `Engine<B: Backend>`.** Type aliases preserve
  the public names: `LlamaCppEngine = Engine<LlamaCppBackend>`,
  `MoefluxEngine = Engine<MoefluxBackend>`. Inherent-method blocks
  on the aliases (state ser/de, log callbacks, `from_path*`, etc.)
  unchanged.
- **Predictor family migrate the same way.** `CandidatePredictor`,
  `TokenPredictor`, `PiecePredictor`, `Predictor` all become
  `<'engine, B>` instead of `<'engine, D, M>`. Iterator-impl `M:
  Sync` bound collapses into Backend's trait-level requirement.
- **`Session<B: Backend>`.** Generic chat-style API. Backend-
  specific constructors (`Session::<LlamaCppBackend>::from_path*`
  with `quiet`; `Session::<MoefluxBackend>::from_path`) live in
  cfg-gated impl blocks. Generic methods (`from_engine`,
  `with_*`, `complete_*`, `engine`, `engine_mut`) live in
  `impl<B: Backend>`.
- **`ChatTemplate::from_model<M: Model>`** and
  `tokenize_with_breakpoints<M: Model>` generalize over the trait.
  `mod chat_template` is no longer gated on `feature = "llama-cpp"`.
- **`mod session` cfg gate** flips from `feature = "llama-cpp"` to
  `any(feature = "llama-cpp", all(feature = "moeflux", target_os
  = "macos"))`.
- **`unsafe impl Send for Engine`** dropped — auto-derive picks it
  up from `B::Decoder: Send` + `B::Model: Send` baked into the
  Backend trait.

### Fixed

- **Qwen3 chat-template thinking-mode forced on by default.**
  `ChatTemplate::render_with` never consulted `prompt.thinking`,
  leaving the Jinja `enable_thinking` variable undefined. Templates
  that gate their `<think>` block on it (Qwen3 family) interpret
  undefined as "thinking on" and emit `<think>\n` after
  `<|im_start|>assistant\n`, forcing the model into thinking mode
  regardless of caller intent. Now derived from
  `prompt.thinking.is_some()` mirroring Anthropic's API semantics
  (`thinking: None` = disabled, `Some(_)` = enabled). Caller-set
  `RenderOptions::with_extra("enable_thinking", _)` continues to win
  for explicit overrides. ollama exhibits the same bug for the same
  reason. Coverage in `tests/template_rendering.rs`.
- **Reserved-token loop on grammar-constrained generation.**
  Tokenizers like Qwen3.5/3.6 carve out a reserved tail of the
  vocab (~248088..248320 for Qwen3) for special-token slots, only
  some of which have registered text content; the rest decode to
  empty strings. Empty-piece tokens contribute zero bytes to a
  byte-stream-driven grammar's matcher and are trivially accepted
  regardless of state, while EOS (`<|im_end|>`) decodes to
  non-empty text the grammar rejects. Result: post-JSON, the model
  could land in a loop scattering reserved tokens until
  `max_tokens` exhausted. Cross-backend testing (A3B on llama.cpp
  vs moeflux) confirms the issue lives at the model/grammar
  layer, not in either backend's decode path. Fixed via the
  `SamplingMode::Deny` mask + `Model::extra_eos_tokens` plumbing
  + grammar-accept-state break described above.
- **Repetition-penalty additive growth on long generations.**
  The additive `count * penalty_freq + penalty_present` term grew
  unboundedly with generation length because `NGramStats` was a
  monotonic frequency map with no eviction. Past ~200 steps the
  additive contribution dominated the model's natural logit
  gradient and content prose collapsed into thesaurus chains or
  fragment loops (the dominant cause of the Qwen3 long-form
  degradation arc). Fixed by replacing the lifetime count with a
  windowed-decay structure: each n-gram tracks the positions of
  its occurrences inside the last `RepetitionOptions::window_size`
  generation steps; the effective count fed to the penalty math
  is `Σ decay^(current_step - position)`, bounded above by
  `1 / (1 - decay)`. With defaults (window=256, decay=0.95) the
  effective count saturates near 20 regardless of how long
  generation runs. `Session::with_repetition` and the per-model
  sampling sidecar are the supported paths to opt in.
  `SampleOptions::default()` ships with `repetition: None` to
  match the historical Session behavior and protect probes from
  silently inheriting a default penalty.

### Removed

- **`blallama --repetition-penalty`** (the v0.7.x band-aid opt-in
  flag). Sampling configuration now comes from the per-model
  sidecar; for force-off probe runs use the new `--no-penalty`
  flag, which overrides the sidecar.

### Migration

- Most callers see no change: `LlamaCppEngine`, `LlamaCppModel`,
  `MoefluxEngine`, etc., are preserved as type aliases / re-exports.
- Callers that explicitly spelled out generic parameters
  (`Engine<LlamaCppDecoder, LlamaCppModel>`) should switch to
  `Engine<LlamaCppBackend>` or just `LlamaCppEngine`.
- `Session` is now `Session<LlamaCppBackend>` (or `Session<MoefluxBackend>`).
  If you stored `Session` in a struct field, parameterize the field.
- `Session::engine()` returns `&Engine<B>` (was `&LlamaCppEngine`).
  For a `Session<LlamaCppBackend>` that's the same type — calls
  unchanged. For ergonomic surface unchanged uses, prefer
  `session.engine().model.display_name()` over the now-llama-cpp-
  only `session.engine().model.file_name()`.

### Notes

- **`blallama` no longer enables the repetition-penalty filter by
  default.** Current `RepetitionOptions::default()` settings
  (`penalty_max_count=1`, `ngram_min_size=1`, `penalty_repeat=1.06`)
  were originally sized for small downstream models in Weave; on the
  larger MoE models drama_llama now drives they over-penalise common
  content tokens during long-form free-text generation and degrade
  output to thesaurus chains or sentence-fragment loops. New
  `--repetition-penalty` flag re-enables the filter for diagnosis.
  Library defaults stay; tuning + broader test coverage tracked
  separately.
- **Upstream moeflux MAX_K bump.** moeflux fork commit `d013a0b`
  raises `MAX_K` from 8 to 16 in `metal_infer/infer.m` (plus the
  combine-shader binding shifts). Without it, A17B (`K=10`) silently
  drops 2 of 10 routed experts per layer per token because the
  `actual_K = (K > MAX_K) ? MAX_K : K` clamp at line 5364 was a
  no-op for A3B (`K=8`) but truncated A17B unconditionally; the
  corresponding routing-weight normalisation already happened over
  the full K, so the dispatched MoE residual was also under-scaled.
  Not the dominant cause of the long-form-degeneration symptom we
  diagnosed (the repetition-penalty defaults above were), but a real
  correctness bug fixed in passing.
- Build matrix: `--no-default-features` (trait layer only),
  `--features llama-cpp,...` (default), `--features
  moeflux-model-qwen3-6-35b-a3b` (moeflux only on macOS), and both
  enabled together. All four combinations build clean.
- Send/Sync trade-offs: `B::Decoder` is required Send (not Sync) —
  `*mut llama_context` is internally mutable. `B::Model` is Send +
  Sync (Iterator impls hand `&Model` to grammar / sampling code
  that fans out across rayon).
- See `.claude/memory/moeflux_disk_convention.md` for the
  forward-looking on-disk layout `MoefluxEngine::from_path`
  expects, and the migration story for current artifacts.

## [0.7.0] — 2026-04-22

Major release. Prompt caching, structured output, grammar-perf finish
line, and a top-to-bottom cleanup pass on the prompt primitives. Requires
`llama-cpp-sys-3` `0.7`, tracking llama.cpp `b8882-5-g82d3f4d3b`.

### Added

- **Prompt caching** (KV-cache reuse across calls). `Engine::prefill` +
  `predict_*_resuming` resumes generation from a populated KV without
  re-decoding the prefix. `Session` tracks previous-turn tokens and
  breakpoints, computes longest-common-prefix `L_hit` with BPE-safety
  backoff, and narrows the KV window on partial reuse. `ChatTemplate`
  supports breakpoint-aware rendering (`render_with_breakpoints`).
  `response::Message` return shape surfaces token usage (input / output /
  cache_read) and a stop-reason. See the `chat_repl` example.
- **Structured output** via `Prompt::output_config`. New `output_config`
  module compiles a `misanthropic::OutputConfig` to a GBNF grammar and a
  `SamplingMode::Grammar`. The shared `grammar_compile` module handles
  `$ref`, `anyOf`, and `const` schema shapes (schemars-emitted schemas
  round-trip cleanly). `Session::complete_*` routes the compiled grammar
  through `SampleOptions::modes`. New `json-schema` feature adds typed
  helpers: `Prompt::structured_output::<T>()`, `OutputConfig::for_type::<T>()`.
- **Thought/JSON phase-split.** New `DeferredGrammar` and
  `SampleOptions::deferred_grammar` let a grammar stay suspended until a
  trigger byte sequence appears in the predictor's output, then get
  promoted into `modes`. `OutputConfigOptions::phase_split` (default `true`)
  compiles a JSON-only grammar triggered by `</think>` — grammar filtering
  is skipped entirely during the thought preamble. `CompiledOutputConfig::
  {Single, Deferred}` + `compile_output_config` / `compile_prompt_output_config`
  expose the phase-split-aware compiler. Legacy `grammar_for_output_config` /
  `grammar_for_prompt` remain as the unified-grammar path. `TokenPredictor`
  drives promotion; post-trigger tail bytes are fed through
  `GrammarState::advance_bytes` so the matcher lines up with the model.
- **Lazy-DFA grammar cache.** `DfaCache` interns canonical `StackState`
  values into `StateId`s and memoizes one-byte transitions + first-byte
  bitmaps. Hot path becomes a `DashMap` lookup; misses pay the current
  `feed_byte` + intern cost. Shared across clones of `GrammarState` via
  `Arc`. Default-on; disable via `DRAMA_LLAMA_DFA_CACHE=0`. Extended
  `GrammarStats` with `dfa_states` / `dfa_transition_hits|misses` /
  `dfa_bitmap_hits|misses`.
- **Grammar matcher profiling.** Opt-in per-call stats via
  `DRAMA_LLAMA_GRAMMAR_STATS=1`. `grammar_stats_snapshot()` /
  `grammar_stats_reset()` return cumulative counts of filter calls,
  candidate survival at each prefilter stage, stack depth, and wall-clock.
- **Tool-choice constrained generation.** `grammar_for_tool_choice`
  emits GBNF for `ToolChoice::{Auto, Any, Method}` with optional
  `wrap_tags` and an `allow_thought` preamble. Session priority is
  `tool_choice > output_config > none`.
- **`Session::from_path_with_n_ctx`** — construct a session with a custom
  KV context size without crafting unsafe FFI params.
- **`blallama` example** — small `/v1/messages` server.
- **Examples**: `whodunit` (structured output integration),
  `chat_repl` (prompt caching demo), `--no-grammar` and `--phase-split`
  flags on `whodunit` for baseline measurements.

### Changed

- **Prompt primitives are misanthropic-native.** `Message` / `Content` /
  `Block` / `Role` come from misanthropic and are aliased to `'static`;
  `Prompt` is a thin wrapper. `RenderOptions::with_extra<V: Serialize>` is
  now generic over serializable extras.
- **`ChatTemplate`** renders via minijinja + pycompat. Handles
  `raise_exception` and a `strftime_now` subset.
- **Sampling chain now applies grammar in parallel.** The per-candidate
  `grammar_filter` loop runs under Rayon (`3.5×` on complex grammars).
  Requires `unsafe impl Sync for Model` — post-load model state is
  immutable.
- **Grammar matcher** refactored for throughput: 256-bit first-byte
  acceptance bitmap prefilter; stack storage moved to
  `TinyVec<[Position; 8]>`; `StackState` split from `GrammarState` so the
  hot clone path doesn't bump the `Arc<Grammar>` refcount; fast-path
  `expand` skips alloc + sort + dedup when every stack is at a yield
  point; tail-call optimization in `expand` bounds stack depth for
  right-recursive rules like `.+`.
- **Repetition penalty rewrite (surgical/"B2")**. New `IgnoreCategory`
  variants for JSON / Punctuation; special tokens (EOS / EOT /
  ignored_stopwords) auto-added to the repetition ignore list. Ignored
  fields moved to `BTreeSet`.
- **`rocket::serde`** indirection dropped from the library.
- **`Session::complete_*` setup paths** polished — `complete_text` /
  `complete_stream` / `complete_blocks` / `complete` / `complete_response`
  all flow through the same prepare-call path.

### Removed

- **`Vocab` / `VocabKind` subsystem** and `data/banned.rs`. Content
  filtering belongs in the consuming app, not the library. See the note
  in `CLAUDE.md` — the Eric Hartford uncensored model check in
  `Model::from_file` stays.
- **`llama_params_fit` / `llama_memory_breakdown_print`** vanished from
  upstream llama.cpp between `b8809` and `b8882`; neither was exposed by
  this crate.

### Fixed

- `session: merge adjacent prose blocks on batch return`
  (`9b62626`).
- `example(whodunit): strip EOS piece from raw text before JSON parse`
  (`4361556`).
- `tool: add strict: None for new Method.strict field` (`c250d31`).

### Performance

On the `whodunit` workload (Qwen 3 8B Q8_0, structured output with
thought preamble):

| config                             | tok/s |
| ---------------------------------- | ----- |
| unconstrained (`--no-grammar`)     | ~20.0 |
| v0.6.2 grammar-constrained         | ~0.7  |
| v0.7.0 after bitmap + TCO + etc.   | 10.1  |
| v0.7.0 with DFA, no phase-split    | 8.9   |
| v0.7.0 with `--phase-split` + DFA  | **17.6** |

Phase-split on + DFA on: phase 1 thought runs at the unconstrained
ceiling (~21.5 tok/s, zero grammar filter calls) and phase 2 JSON at
~13.0 tok/s with 99.8% DFA transition hit rate. Workloads with wide
free-form `.+` regions inside JSON (some Agora reactor shapes) should
flip `DRAMA_LLAMA_DFA_CACHE=0`.

### Notes

- `cargo publish` for this crate is still gated on misanthropic 1.0
  landing on crates.io. Published as a git tag only.
- Known pre-existing test failures: `candidates::tests::test_apply_entropy`,
  `candidates::tests::test_sample_tail_free` are `todo!()` stubs;
  `model::tests::test_model`, `model::tests::test_model_desc` assume a
  Llama-family model and fail when `models/model.gguf` points at Qwen.
- egui 0.34 deprecation warnings (`clamp_range`, `id_source`) are left
  for a follow-up PR.

[0.7.0]: https://github.com/mdegans/drama_llama/releases/tag/v0.7.0
