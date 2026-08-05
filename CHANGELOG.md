# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **The repetition penalty now sees history (#106) — three new
  `RepetitionOptions` seeding flags, all default ON.** For an
  all-structured workload (an agent whose every turn is tool calls
  and `output_config` JSON) the penalty was effectively disabled
  end-to-end: the prose fold excluded tool results and tool-call
  args, and the constrained-region accumulator started empty every
  call — near-identical sequential posts from similar agents were
  the symptom. Now `seed_tool_results` folds tool-result text (how
  agents read thread context), `seed_tool_args` folds the string
  values of prior tool-call arguments (how agents emit; keys,
  numbers and booleans stay excluded), and `seed_constrained_regions`
  clones the folded corpus into the constrained accumulator at
  generation start — after every breakpoint snapshot, step rebased —
  so grammar free regions feel prompt-history pressure from token
  one. Cold-prefill ≡ cache-resume is untouched (the seed re-derives
  from the prompt identically on both paths; oracle-pinned including
  the constrained fields). Escape hatches: `set_seed_tool_results` /
  `set_seed_tool_args` / `set_seed_constrained_regions`, all
  sidecar-settable per model.
- **Penalty defaults retuned for thread-corpus reach**: `window_size`
  256 → 2048, `decay` 0.95 → 0.99, `penalty_freq` 0.125 → 0.025. The
  decay/freq pair moves together, so the saturated additive cap
  (≈ 2.6) is unchanged — 5× the decayed-term reach at the same
  ceiling. Note existing sidecars serialize the full options table
  and therefore pin the old numbers; delete or edit
  `<model>.sampling.toml` to pick up the new defaults. The
  digit-echo pins (a `"3"` echoed from a tool result) stay green on
  all four model suites: short tool echoes seed nothing (blocks
  shorter than `ngram_max_size` produce no windows) and surgical
  mode gates single occurrences.

### Fixed

- **The post-generation tip anchors continuations again (#96).** The
  prefix cache's two lookups composed as hash-keyed first, LCP only on
  a total hash miss — and every continuation re-renders its old
  `cache_control` markers to identical partials, so the hash path
  always matched something and capped reuse at the last explicit
  marker. The tip (the internal anchor at the KV head, past every
  marker) was never consulted unless the client happened to mark the
  re-ingested assistant turn. Every drama_llama release with hash-keyed
  reuse re-prefilled the entire final turn on every continuation.
  `slot_l_hit` now runs both lookups and takes the larger offer; both
  prove their prefix in both coordinate spaces, so the max is always
  sound. Two debug lines make future losses visible: one when a live
  tip loses the pick, one when `tip_extension` declines to build a tip.

### Added

- **#96 regression suite, per model.** A shared scenario
  (`tests/common/tip.rs`) runs the downstream agent shape — sliding
  marker window, forced tool-call turns, honest tool results — plus
  the issue's probe (a continuation adding no new `cache_control`
  anywhere), asserting each call resumes past the *entire* previous
  prompt: a read that far can only come from the tip. Wired into the
  Qwen (`session_cache`), Gemma 4, gpt-oss, and Mistral Small 4
  suites; green on all four. No golden text — CI runs smaller family
  members (`session_gptoss` gained `DRAMA_LLAMA_GPTOSS_MODEL` for
  that).

## [0.8.3] — 2026-07-25

### Added

- **`llama_cpp::gpu_device_names`** — the GPU-class compute devices ggml
  actually discovered, as opposed to the ones the build asked for. Those
  differ in a way nothing else reported: a `feature = "cuda"` build links
  CUDA and then silently runs every model on the CPU when the driver is
  unusable, and from inside the process that is indistinguishable from a
  box with no card.

- **A test asserting a `cuda` build found a GPU.** The model tier stays
  **green** through CPU fallback — correctness doesn't depend on the
  device — and timing doesn't give it away either, because only the
  generation-heavy tests slow down while load-dominated ones get *faster*
  (no VRAM upload, weights already in page cache). The suite's wall clock
  can therefore move in either direction, which makes an explicit device
  assertion the only reliable signal.

  Caught for real on 2026-07-25: an unattended upgrade of
  `nvidia-driver-580-server` (580.159.03 → 580.173.02) left the old kernel
  module loaded against new userspace libraries. `nvidia-smi` failed
  outright, `cuInit` returned 804 (`CUDA_ERROR_SYSTEM_DRIVER_MISMATCH`),
  ggml found zero CUDA devices, and the entire model tier passed on the
  CPU with nothing in CI saying so. The new test fails in 0.04 s and names
  the diagnosis.

  Set `DRAMA_LLAMA_ALLOW_CPU_FALLBACK=1` to exercise the CPU path
  deliberately — a legitimate thing to want; silently getting it is not.

  The test's `cuda` gate is on its **body**, not on `#[test]`, so it is
  listed in every configuration and `nextest list` returns the same count
  everywhere. Gating the attribute makes README.md's hand-counted numbers
  mean one thing under `llama-cpp` (605) and another under
  `llama-cpp-cpu` (604) — which is how this PR first went red. House rule
  now, and the reason `just check` can verify the badge against whichever
  target dir is warm instead of building the cpu one.

- **A `nvidia-smi` preflight on `test.py run`** for `cuda` configurations
  on Linux. Reaches the same verdict as the test above roughly 40 minutes
  earlier — before a cold cuda build rather than after it. It is not a
  replacement: a working `nvidia-smi` says nothing about a llama.cpp that
  cmake quietly configured without CUDA, so ggml's own device list stays
  the authority. Honours `DRAMA_LLAMA_ALLOW_CPU_FALLBACK=1`.

### Fixed

- **The pre-commit hook actually gates the README badge now.** 6d4a9e2
  put that gate in `scripts/test.py check` and claimed the hook was
  covered; it was not. The hook runs `just check`, whereas that function
  is `just permutations` — which the hook deliberately skips as too slow
  — so the gate never ran once, and a stale badge reached CI in #82. It
  lives in the `just check` recipe now.

- **CI's `gpu` step no longer swallows a failing `nvidia-smi`.** It pipes
  into `tee` under the default `bash -e`, which takes its status from
  `tee`, so the exact driver-mismatch state the step exists to record
  reported green. It sets `pipefail` now.

## [0.8.2] — 2026-07-24

### Fixed

- **A truncation sampler can no longer starve the deferred (lazy)
  tool-call grammar** ([#76]). An eager grammar — the forced
  `tool_choice` path — is prepended into `SamplerConfig::modes` by the
  session so it masks the full vocab *before* `LocallyTypical` / `TopP`
  / `TopK` narrow it. The deferred grammar, which is not in `modes`,
  was applied after the entire fold instead, so it only ever saw what
  truncation left behind — frequently a single token. When that lone
  survivor was illegal, `grammar_filter` force-EOS'd and generation
  ended mid-structure, surfacing as
  `SessionError::GrammarViolation`, even though thousands of legal
  tokens existed in the full candidate set.

  Because only `tool_choice: auto` builds a deferred grammar, this
  presented as *intermittent* tool-call failures on local models under
  auto while forced tool choice was unaffected, and it was independent
  of `max_tokens` (reproduced identically at 2048, 4096, 8192, and
  16384). `apply_modes` now runs the deferred grammar first, at the
  same point an eager grammar runs. Reproduced end-to-end on
  gpt-oss-20b via a multi-round agentic tool session, where it failed
  on the first acting round every time and passes 8/8 rounds after.

## [0.8.1] — 2026-07-24

### Fixed

- **Tip resume into a new assistant turn no longer forces an
  immediate EOS** (the 0-output-token second-round bug). A cached
  tip's `SamplerState` carries constraint-matcher positions on
  grammar identity — correct for assistant-prefill / partial-
  completion resume, but wrong when the new call seats the generated
  turn plus a tool result: the matcher arrived parked at
  tool-call-complete, where the only legal continuation is the turn
  terminator, so every second acting round of an agentic session
  generated zero tokens (reproduced on both Qwen 3.6 and gpt-oss —
  template-independent). `Session::build_initial_state` now resets
  matchers (eager, JSON, and deferred) to their grammar roots when
  the fold has messages past the resume cursor
  (`matcher_carry_valid`); the prose stream (rng, mirostat `mu`,
  n-gram stats) still carries. `hash_cache_smoke` — which exercised
  the exact failing shape but asserted only cache-read counters —
  now also asserts round 2 produces output: cache stats are not a
  proxy for generation.

## [0.8.0] — 2026-07-24

Backend split. The chat-style API (`Session`), the engine layer
(`Engine`), the Predictor family, and the binary (`blallama`) are
all generic over a single `Backend` parameter. drama_llama can now
drive either llama.cpp or moeflux's Metal MoE runtime through the
same surface. Runs Cogito-class MoE models on Apple Silicon without
the Anthropic API as a dependency.

Three further arcs land on top of the split:

- **Image input** via llama.cpp's mtmd ([#31]). A backend-agnostic
  `Vision<D>` trait, a safe `Mtmd` wrapper, and a cache-aware
  `Session` media path let vision models take `Block::Image` input.
  Images are rendered out-of-band through a per-call random sentinel
  — mtmd never sees prompt text — and the prefix cache accounts in
  M-RoPE cell space so an image mid-prompt doesn't invalidate the
  KV walk. Gated on `feature = "mtmd"` (or pure-Rust `feature =
  "media"`).
- **Per-model tool-call dialects** ([#30], absorbing [#29]). A
  `CallSyntax` derived by differentially analyzing each model's chat
  template drives both the GBNF grammar emitter and the response
  parser, so `Session` speaks a model's *native* tool-call format
  instead of one imposed shape. Qwen3.5/3.6 (XML-ish), Gemma 4
  (`TagWithDict`, causal announce-then-call), and gpt-oss (Harmony
  channels) ship as validated dialects. Round-trip byte-stability is
  the cache invariant.
- **Lazy grammar checking** ([#28]). Grammar-constrained sampling
  now samples first and checks the one sampled token
  (`GrammarState::accepts_bytes`, O(piece)), falling back to a full
  O(vocab) mask only on rejection — instead of masking the whole
  vocabulary every step.

### Added

- **The README is now the crate-level documentation** (#66), mounted
  with `#![doc = include_str!("../README.md")]`, so the front page of
  the repository and the front page of docs.rs cannot drift apart. It
  was rewritten from the v0.3-era text it still carried, which
  advertised a `bin/dittomancer` deleted in `57016dd` and a hardcoded
  n-gram blocklist removed in `c2d8579`. Its two `rust` blocks are
  doctests: a `no_run` structured-output walkthrough that is
  type-checked on every CI run, and an executed one driving the GBNF
  matcher standalone.
- **Coverage** (#70) — `scripts/test.py coverage`, `just coverage`,
  and a CI job uploading to Codecov. Runs the tests under
  cargo-llvm-cov once and re-reads that data per output format
  (`--lcov`, `--json`, `--html`, and a per-file table). Defaults to
  `-t all` because the fast tier alone reports every generation path
  as dead code. Read the headline percentage as an upper bound:
  llvm-cov filters by file, so `#[cfg(test)] mod tests` inside `src/`
  cannot be excluded from it.
- **`scripts/test.py doctest`** / `just doctest`, wired into `just
  check` and its own CI job. cargo-nextest has no doctest support
  (nextest-rs/nextest#16), so until now every recipe in the repo ran
  exactly zero of them — which stopped being acceptable once the
  README became a doctest. The pre-commit hook and `ci.yml` both
  skipped `.md`-only changes; both now treat `README.md` and
  `TERMS_OF_USE.md` as source, since `src/lib.rs` `include_str!`s
  them.
- **`FromPath::Options`** — an associated type carrying whatever a
  backend needs to be told at load time, so generic code can ask for
  a context size. `LlamaCppOptions` (`n_ctx`, `cache_slots`,
  `flash_attn`, `no_gpu`, `numa`) and `MoefluxOptions` (`use_2bit`);
  both `serde`-serializable, and `clap::Args`-flattenable under
  `feature = "cli"` for single-backend binaries. Every field unset
  means the backend's own default, so `from_path` is unchanged
  behaviour. `FromPath` gained `from_path_with` (the constructor),
  keeps `from_path` (default options) and `from_path_async` (the
  same on tokio's blocking pool) as provided methods, and is no
  longer `tokio`-gated.
- **`cli::BackendArgs`** — the union of every compiled-in backend's
  load knobs plus the `--backend` selector, for binaries that pick
  their backend at run time (clap flattens at compile time, so such
  a binary cannot flatten `B::Options` directly). Narrows to a
  concrete backend's options with `TryFrom`, returning
  `cli::UnsupportedOptions` when a flag names something that backend
  has no notion of. Also `cli::DEFAULT_N_CTX` (32768), the value
  this repo's own front-ends default to.
- **`Backend::set_log_callback` / `Backend::clear_log_callback`** —
  route a backend's native log stream wherever the application wants,
  returning `Result<(), NotImplemented>` with a default body that
  errors. Lets an application install a sink *before* loading a
  model, which is when llama.cpp is loudest and the only point at
  which the noise can be caught. `LogLevel` moved to
  `crate::backend` (its `Other` variant now carries `u32`) so it
  compiles with no backend feature enabled.
- **`blallama --n-ctx` / `--cache-slots` / `--use-2bit`**, via the
  flattened `BackendArgs`.

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
- **`Model::eog_tokens()`** — trait method exposing the model's
  complete end-of-generation set, and the single authority for what
  stops a prediction. `LlamaCppModel` returns libllama's
  `special_eog_ids` verbatim (`llama_vocab_is_eog`); `MoefluxModel`
  composes it from the `eos_token_id` config array (Qwen3 declares
  `[<|im_end|>, <|endoftext|>]`, and the secondary decodes to an
  empty piece, so missing it means an invisible loop to
  `max_tokens`). Never derive a stop set from `eos()`/`eot()` — see
  *Changed* and *Fixed*.
- **`Model::display_name(&self) -> Option<String>`** — human-readable
  identifier on the trait. `LlamaCppModel` returns the GGUF basename;
  `MoefluxModel` returns the parent dir's basename (overridden
  by `MoefluxEngine::from_path` to match the discovery-dir name).
  Used by `Session::complete_response` for the `model` field of
  responses, and by `blallama` for model-name matching.
- **`backend::Backend` trait** bundling `type Decoder: Decoder + Send`
  and `type Model: Model + Send + Sync` as a single generic
  parameter. Compile-time monomorphization, no `dyn` indirection on
  the hot path. ZST tag impls: `LlamaCppBackend`, `MoefluxBackend`.
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
- **`Session::with_sample_options(SamplerConfig)`** — wholesale
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
- **Probe mode** — `Engine::set_probe_hook` installs a per-token
  `ProbeHook` observer on the prediction loop, so a consumer (canary
  suite, batch evaluation, Weave) can watch each yielded token
  without forking the predictor iterator. `ProbeCtx` carries the
  sampled token, its position, and the effective sampler
  configuration; hooks self-declare their snapshot appetite via
  `ProbeHook::snapshot_opts` — the default `None` skips the per-token
  softmax/sort entirely, while `Some(SnapshotOpts)` populates a rich
  `Candidates` `Snapshot` (`Candidates::capture_snapshot`:
  pre-everything top-k probabilities and ranks, optional entropy)
  for cross-validating external behavior against internal
  disposition. `blallama` exposes both recorders: `--record-json
  <FILE>` appends one JSONL record per token, and `--probe-stream`
  mounts a `/probe` SSE endpoint streaming per-request
  `session_start` / `token` / `session_end` events.
- **`SessionTransport` / `LocalTransport`** ([#48]) — a
  `misanthropic::Transport` over a locally-owned `Session`, so
  anything generic over a transport (the upstream `Chat` driver
  included) drives local inference exactly as it would the API
  client. Completions run on tokio's blocking pool behind an async
  mutex, one at a time — cloning the transport clones the *handle*,
  and every clone serializes through the same session and KV cache,
  the honest shape of one local model on one GPU. `LocalTransport`
  is the erasure: `Arc<dyn LocalTransport>` serves both `Prompt` and
  `CachedPrompt` and carries `scan_text_for_specials` through the
  erased type, which is how the examples went backend-generic (see
  `.claude/memory/examples_erase_at_transport.md` for why the type
  is the transport, not a boxed `Session`).
- **Multi-slot prefix cache.** The v0.7 single-sequence prompt cache
  grew slots: up to `PrefixCacheConfig::max_slots` cached prefixes
  (clamped to the engine's sequence count), so a council of agents
  each keep a warm prefix instead of evicting one another. Enable
  with `Session::with_prefix_cache(true)`, tune with
  `Session::with_prefix_cache_config`. Slots evict
  least-recently-used against a cell budget
  (`PrefixCacheConfig::capacity_cells`, default the engine's
  `n_ctx`), and breakpoints honor Anthropic-style TTLs (5m default,
  1h opt-in) with refresh-on-read semantics: an idle breakpoint past
  its TTL loses its snapshot, and a fully-expired slot is evicted.
- **`Model::recommended_sampling()`**, with **`SamplingParams` /
  `Mirostat`** and **`apply_request_sampling`** — each backend
  reports the sampling settings the model recommends for itself
  (GGUF `general.sampling.*` for llama.cpp; returning empty is
  expected, not exceptional — gpt-oss carries no such keys). Seeds a
  fresh sampling sidecar, and serves as the fallback tier for
  per-request sampling: `apply_request_sampling` patches a single
  unambiguous knob into the existing chain, otherwise rebuilds a
  canonical `TopK → TopP → MinP → Temperature` chain from the
  request layered over the model's recommendation — keeping
  constraint modes as a prefix and never touching repetition or
  `banned_specials`, which are emission-side protocol integrity a
  remote client must not be able to switch off.
- **`ToolChoice::None` is enforced** ([#44]) — "the model must not
  use any tool" now bans the dialect's tool-call opener for that
  call alone (the standing emit-ban exempts the opener so
  Auto/Any/Method can call), leaving the tool definitions rendered
  and the cached prefix intact. Use cases: prefix-preserving
  interviews, and forcing prose out of a tool loop. The `chat`
  example grew `--tool-choice <auto|any|none|method:NAME>`.

#### Image input — mtmd ([#31])

- **`feature = "media"` and `feature = "mtmd"`** — two-tier gating.
  `media` is the pure-Rust image layer (decode via the `image` crate
  — never mtmd's bundled `stb_image`, a deliberate CVE-posture
  choice — plus the conversions into the frozen `Image` pixel
  record); it compiles without `llama-cpp`, so moeflux-only builds
  get typed "media unsupported" errors from `NoVision` rather than
  silent drops. `mtmd` adds the llama.cpp multimodal backend on top
  (libmtmd bindings + the safe `Mtmd` wrapper).
- **`backend::Vision<D: Decoder>` trait** — the backend-agnostic
  image-input capability, generic over the decoder. Placeholder-
  typed by design: `tokenize_image` takes an `ImageInfo` (dims +
  identity hash, no pixels), while `prefill_image` requires a full
  `Image` and the decoder — encoding a placeholder is unrepresentable
  in the type. `NoVision` is the uninhabited impl for backends
  without vision, so generic `Session` code compiles for every
  backend.
- **`backend::Image` / `ImageInfo` / `MediaSpan` / `MediaChunk`** —
  the frozen pixel record (`Image::from_rgb8`, sha256 identity via
  `Image::id`) and the placeholder / span types the `Vision` trait
  and the media-aware cache traffic in.
- **`llama_cpp::Mtmd`** (`feature = "mtmd"`) — safe wrapper owning
  the `mtmd_context`, implementing `Vision<LlamaCppDecoder>`.
  `MtmdParams` is the small stable construction subset; typed error
  ladder (`MtmdNewError` / `MtmdTokenizeError` / `MtmdPrefillError`
  / `MtmdError`), all `Send + Sync`. The media eval loop is Rust-
  owned (`EmbdBatch`, a pre-KV `NaN` guard, explicit M-RoPE position
  planes) rather than delegated to mtmd's C helper.
- **`Session` media path** — `Block::Image` is accepted at ingest,
  decoded through `media`, and rendered out-of-band via a per-call
  random sentinel so mtmd never tokenizes prompt text (injection-
  proof by construction). The prefix cache is media-aware end to
  end: `CacheEntry` (token vs. media sentinel), `EntryPos`
  (entry↔position translation), and cell-space accounting so an
  image's M-RoPE cell span participates in the longest-common-prefix
  KV walk instead of forcing a full reprefill.

#### Per-model tool-call dialects ([#30], absorbs [#29])

- **`drama_llama::dialect` module** — template-derived tool-call
  formats. `CallSyntax` (with `ReasoningSyntax` / `ContentSyntax` /
  `FunctionSyntax` / `ArgumentsSyntax` / `CallIdSyntax` /
  `JsonFields` and the `Family` / `ReasoningMode` / `ContentMode`
  axes) is the single description that drives both emission and
  parsing.
- **`analyze_template` / `vocab_cross_check`** — the differential
  analyzer that derives a `CallSyntax` from a model's chat template
  (probe-first, llama.cpp-validated), cross-checked against the
  model's vocab so emitted markers are real tokens.
- **`grammar_source` / `render_reference` / `validate_representable`
  / `EmitOptions` / `Anchor`** — the GBNF emitter half: a
  `CallSyntax` compiles to a grammar that constrains generation to
  the model's native call shape.
- **`parse_text` / `StreamParser` / `ParseStatus` / `Parsed` /
  `Leniency`** — the parser half: the model's emitted envelope is
  re-ingested back into typed tool-call blocks, byte-stable with
  what the grammar emitted (the cache invariant).
- **`emit_until_rules`** (in `grammar_compile`) — GBNF encoding of
  llama.cpp's `until()` combinator (KMP-DFA complement), the
  grammar-engine primitive dialects need to consume "everything up
  to the closing tag." Exhaustively differential-tested against a
  naive matcher.
- **`Session::with_dialect` / `Session::dialect`** — a dialect is
  analyzed once at load and thereafter drives grammar construction
  and response parsing. Shipped dialects: Qwen3.5/3.6 (XML-ish,
  native format from [#29]), Gemma 4 (`TagWithDict`, causal
  announce-then-call render), gpt-oss (Harmony channel format; see
  the `dialect::harmony` submodule).
- **Chat-template sidecar** — an optional `<model>.chat_template.jinja`
  sibling overrides the GGUF-embedded template (used to ship the
  cache-stable Gemma 4 / gpt-oss templates without patching the
  model file).

#### Lazy grammar checking ([#28])

- **Sample-then-check grammar constraint** — `SamplingMode::Grammar`
  / `Json` now sample a token from the *unmasked* distribution and
  validate just that token with `GrammarState::accepts_bytes`
  (O(piece)); only on rejection does the full O(vocab) mask-and-
  resample path run. Common case drops from per-step vocab masking
  to a single byte-run check.
- **`SamplerConfig::banned_specials: Vec<Token>`** — emit-side
  special-token mask applied before sampling, so a dialect's illegal
  control tokens never reach the candidate set (opt-out via
  `Session::with_emit_specials_ban(false)` for e.g. Qwen-VL
  grounding markers). Falls back to a resample when the ban would
  empty the set.
- **Exit interviews for agents** — the `council` example grew
  `--dump [DIR]`, archiving each seat's complete prompt to
  `<DIR>/<seat>.json` on adjournment, and the `chat_repl` example
  became `chat`: `--load` reseats a dumped prompt so you can
  interview the agent about the run. Loaded tools are kept verbatim
  (their schemas are debug context); every tool call is printed but
  only bash executes (`--add-bash`, Docker-sandboxed `RichBash`
  driven without a `ToolBox`); everything else is answered with a
  stub receipt so the transcript stays wire-legal. `--clear-tools`
  strips tools for a prose-only interview.

### Changed

- **Tool arguments render in schema declaration order** ([#60]).
  `serde_json/preserve_order` and `minijinja/preserve_order` are now
  enabled unconditionally, so JSON maps keep insertion order
  end-to-end: schemars-derived `properties` arrive in field
  declaration order, the GBNF grammar emits arguments in that order
  (letting a model condition later arguments on earlier,
  reasoning-ish ones), the parser preserves it, and minijinja's
  `tojson` re-renders it unchanged — matching llama.cpp. Required
  vs. optional is now classified by *membership* in `required:`,
  never by that array's order, and optionals sit in place in the
  grammar. The dict family (Gemma 4) is the deliberate exception:
  its model-shipped templates pipe arguments through `| dictsort`,
  so that family stays explicitly alphabetical. Behavior-breaking
  for caches: the canonical bytes of rendered tool definitions and
  tool calls change (previously alphabetized), so warm prefix caches
  from earlier 0.8.0 dev builds will not match. Enabling the
  features ourselves also closes a feature-unification hazard where
  a downstream crate enabling `serde_json/preserve_order` would have
  silently broken the old sorted-order assumptions. Note for tools
  served to both a local backend and the Anthropic API: Anthropic's
  structured outputs reorder required properties first, so declare
  required fields before optional ones if identical ordering across
  engines matters.

- **Pre-publish API hardening** (pre-crates.io review). Every public
  error enum, the growth-prone enums (`LogLevel`, `MediaChunk`,
  `Family` — whose docs already promise an `Instructed` variant —
  `ReasoningMode`, `ContentMode`, `CallIdPosition`,
  `ReasoningReingest`), and the options/config structs
  (`LlamaCppOptions`, `MoefluxOptions`, `BackendArgs`,
  `SamplerConfig`, `SamplingParams`, `PrefixCacheConfig`,
  `EmitOptions`, `CallSyntax` and its field-structs) are
  `#[non_exhaustive]`: adding a variant or field later must not be a
  breaking change. Construction is `Default` + pub-field mutation or
  the `with_*` builders (`BackendArgs` gained the `Default` its clap
  attributes already advertised; `MoefluxOptions::use_2bit()` is now
  `with_use_2bit(bool)`, matching the family). The `moeflux` module
  itself went private with curated re-exports, mirroring
  `llama_cpp`: `MoefluxDecoder::ctx()`/`ctx_mut()` leaked
  `moeflux::Ctx` — a third-party type pinned at a pre-release — into
  the public API, and the cmdbuf/`eos_raw` diagnostics were public.
  `MoefluxEngineError` and `PrefetchStats` joined the crate-root
  exports because public signatures name them.
- **`SampleOptions` split into `SamplerConfig` + `SamplerState`.**
  The old type conflated immutable configuration with live per-call
  run-state — grammar matcher positions behind `Arc<Mutex<_>>`, ~80
  lines of custom serde hooks (inconsistent between the `Json` and
  `Grammar` variants), and a ~113-line mutex-locking `PartialEq`.
  `SamplerConfig` is now a pure value: derive-serializable,
  `PartialEq`, no interior mutability, and `banned_specials` became a
  plain `Vec<Token>` in the same move. `SamplerState` gathers
  everything a run accumulates: matcher positions, mirostat `mu`, the
  n-gram stats, and the working RNG — now `rand_pcg::Pcg64Mcg`, whose
  single-`u128` state can actually be serialized (the previous
  `xorshift::Xoroshiro128` could not expose its state, which is the
  point).
  The effective config is the sole constructor of a fresh state
  (`SamplerConfig::init_state`), and `Engine::predict_tokens` /
  `predict_pieces` / `predict` (plus their `_resuming` variants) take
  a new `initial_state: Option<SamplerState>` parameter to resume a
  caller-owned state. `Session` caches the state at breakpoints
  alongside the KV snapshot, and the per-call seed default flipped to
  `None`, making the seed encode resume/fork/fresh: no seed + cache
  hit resumes the exact stream, no seed + miss draws a fresh random
  seed, an explicit seed forks deterministically. Serialize → restore
  → continue is bit-exact — `NGramStats` moved to `BTreeMap` so
  iteration order (and thus float accumulation) survives a round
  trip. See `.claude/memory/design_sampler_config_state_split.md`
  for the full design.
- **`tracing` is now a non-optional dependency** (with its `log`
  feature), and the crate's own diagnostics — sidecar read/write
  failures, chat-template dialect analysis failures — emit
  `tracing::warn!` instead of writing to stderr unconditionally.
  `RUST_LOG` now governs them on every backend.
- **`Session::from_path_sync` is now `FromPath::from_path`** (the
  trait must be in scope). `Session::from_path_with_n_ctx` and
  `LlamaCppEngine::from_path_with_n_ctx` remain as shorthand for the
  common case.

- **`Model::extra_eos_tokens` → `Model::eog_tokens`**, and it now
  returns the *whole* end-of-generation set (`eos` and `eot`
  included) rather than the extras beyond them. For the llama.cpp
  backend that set is libllama's `special_eog_ids` verbatim —
  `llama_vocab_is_eog`, quirks and per-family workarounds included.
  It is the single authority for both "does emitting this end the
  turn" and "may this token be masked while a constraint is open";
  callers must not derive a stop set from `eos()`/`eot()`, which are
  labels the vocab applies, not statements about behavior (see
  *Fixed*). Backends that have no `is_eog` oracle report their own
  truth: `MoefluxModel` composes it from the tokenizer config, where
  `eot` genuinely does terminate a turn.
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
- **`llama-cpp-sys-3` 0.7 → 0.8.1.** Picks up the upstream cmake
  `mtmd` target, libmtmd bindgen, and packaging that back the new
  `mtmd` feature.
- **`misanthropic` alpha.3 → alpha.12.** Adds the image content-block
  types `Session` needs to accept `Block::Image`; the `image` /
  `jpeg` / `png` sub-features are pulled in by drama_llama's `media`
  feature.
- **`Session` tool-call termination is constraint-owned.** With a
  dialect active, the emitted call's own close marker (not a raw
  sampled EOG) terminates the turn: EOG and empty-piece tokens are
  masked while a constraint is live, the repetition penalty is
  suspended across structural emission (region-aware within free-text
  spans — see below), and the recorded tip is the canonical close
  token rather than whatever EOG happened to be sampled — so the next
  turn's cache walk stays byte-stable.
- **Repetition penalty now applies inside grammar free-text regions.**
  The penalty was previously suspended for the entire span of any
  active byte-constraint — which also silenced it inside the free
  islands where the model writes prose (JSON string bodies, `until()`
  spans), letting small models loop a paragraph verbatim inside a
  forced tool-call argument. Suspension is now scoped to *structural*
  emission (delimiters, keys, tags); inside permissive regions the
  penalty runs against a call-local n-gram accumulator, with
  region-exit tokens (the closing quote, merged `",` pieces) left
  unpenalized so the model can always leave the region. Default-on;
  opt out with `RepetitionOptions::set_constrained_regions(false)`,
  which restores the pre-feature blanket suspension exactly. ([#43])
- **Default sampler chain prepends a top-k 1024 cut before
  locally-typical.** The stock `SamplerConfig` now applies a top-k
  1024 pre-cut ahead of the locally-typical stage (typical mass
  concentrates in the head, so the cut trims the tail cheaply).
  Output for streams pinned by seed against the previous default
  chain will differ.

### Fixed

- **`blallama` served every llama.cpp model at `n_ctx = 512`.** It is
  generic over the backend, so it could only reach `FromPath`, and
  `FromPath` carried nothing but a path — leaving llama.cpp's own
  512-token default in place with no way to override it. Its
  `session_ready` log line had been reporting this all along. Fixed
  by `FromPath::Options`; `--n-ctx` now defaults to 32768.
- **`--seed` did not exist on any example.** The field in
  `CommonArgs` was missing its `#[arg(long)]` attribute, so clap made
  it a positional argument instead of a flag.

- **Cache usage counters are honest now** ([#40]).
  `cache_creation_input_tokens` had been hardcoded `Some(0)` since the
  original caching commit; it now reports the prompt tokens newly
  decoded into the cache this call (`input − read`, per the Anthropic
  field semantics — every decoded token lands in the slot's
  tip/breakpoint snapshots). With the prefix cache **disabled**, both
  cache counters are now `None` ("not reported") instead of `Some(0)`,
  so consumers can finally distinguish cache-off from a healthy cold
  call. `input_tokens` stays the full prompt. Additionally,
  `complete_response`'s `Message.usage` is now the *same* `Usage` the
  session records as `last_usage` (carried through `CallOutcome`)
  instead of an identical second build.
- **The constructor-default repetition penalty no longer penalizes
  specials.** `from_engine` seeded `SamplerConfig::default()` without
  the specials injection the `with_repetition` / `with_sample_options`
  setters apply, and the per-call assembly discards
  `add_model_stops`' injection — so a session that never routed
  through those setters (no sidecar on disk, sidecar parse error,
  `from_engine` directly) penalized its own EOG/framing tokens,
  making every turn less likely to end than the last. Injected at
  construction now; all paths protected.
- **Harmony turns died at the end of their reasoning block —
  `eot` is not a stop token.** libllama auto-detects the EOT token
  *by text*, and `"<|end|>"` is on that match list, so gpt-oss's
  `eot()` is `<|end|>` — its in-stream *channel separator*. libllama
  then removes `<|end|>` from `special_eog_ids` precisely so the model
  can close an analysis channel and keep going, and leaves
  `special_eot_id` pointing at it; upstream stays consistent because
  its generation loop only ever asks `llama_vocab_is_eog`. drama_llama
  instead rebuilt the stop set by hand as `{eos} ∪ {eot} ∪ extras`, in
  seven places, dragging `<|end|>` back in. Unconstrained, a Harmony
  turn stopped dead after its analysis block (one lone `Block::Thought`
  came back — no answer, no tool call); under a tool grammar, where the
  same set is masked while the constraint is incomplete, the model
  could not emit the token that closes the channel and rambled to
  `max_tokens`. `<|end|>`'s piece was also being stripped from the
  surfaced text, which would have left the dialect parser with an
  unterminated reasoning block. Fixed by deleting the union: see
  `Model::eog_tokens` under *Changed*.
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
  `SamplingMode::Deny` mask + `Model::eog_tokens` plumbing
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
  generation runs. With the growth bounded,
  `SamplerConfig::default()` now ships the penalty **on** — the
  unbounded additive term was the reason it had been off — with
  `SamplerConfig::greedy()`, the per-model sidecar, and blallama's
  `--no-penalty` as the opt-outs.
- **Special-token injection through prompt content.** `Session`
  rejects `Block::Text` content bearing chat-format control tokens
  (`<|im_end|>` and friends) or media markers at ingest via
  `check_no_special_injection` — a framing token inside content is
  an accident or an injection, never meaning, and letting it through
  desynchronizes the KV cache, the block parser, and the marker-
  count contract. This is format-integrity enforcement, not content
  filtering: `Session` owns it, `Engine`/the raw predictor stay
  permissive for callers deliberately hand-feeding control tokens.
  The guard scans `ToolUse` surfaces too — tool name, call id, every
  string leaf *and key* of the `input` (`ServerToolUse` included),
  and `ToolResult.tool_use_id` — because templates render all of
  them verbatim and ingest tokenizes with specials enabled, so a
  special piece in any of them became real control tokens; the
  [#37] relay scenario is exactly a tool-use-shaped payload.
- **Ingest injection guard no longer false-positives on `add_bos`
  vocabs.** The guard keyed off raw tokenization including a leading
  BOS the caller never wrote; on `add_bos` vocabs that flagged clean
  content. Now compared against the content's own token span.
- **Grammar exit-marker EOG exemption + until-delimiter trim.** A
  completed constraint whose close marker *is* an EOG-adjacent token
  no longer double-terminates or trims the closing delimiter out of
  the emitted text; incomplete-constraint violations surface as
  errors instead of silent truncation.
- **A safe-code use-after-free through `pub Engine.model`** ([#54]).
  `llama_context` keeps a reference to its model for its whole life,
  but nothing tied the decoder's lifetime to the model's — only
  `Engine`'s field declaration order kept drops sound, and
  `engine.model = other` through the `pub` field freed the weights
  under a live context. Two lines, all safe code. `LlamaCppModel` is
  now a cheap-clone refcounted handle (a private inner owns the
  pointer); the decoder and `Mtmd` each hold a clone, so the weights
  structurally outlive anything referencing them. `Engine.model` is
  private behind `Engine::model()` — now for coherence rather than
  safety: swapping the model would leave the tokenizer disagreeing
  with a KV cache built from the previous weights. See *Migration*.
- **The auto-tip was silently defeated on `add_bos` vocabs.** The
  canonical close token was tokenized with `add_special = true`, so
  on Gemma / Llama-3-style vocabs the stored tip ended `[.., BOS,
  close]`; the next call's longest-common-prefix walk stopped at the
  BOS and fell back to the last breakpoint on every such model.
- **Dialect analyzer / parser hardening from the pre-publish
  review.** The template diff-split and its common-prefix/suffix
  helpers now round byte-wise match lengths down to char boundaries
  in both strings — two renders diverging inside a multi-byte
  character (any non-English template) panicked `Session` load,
  outside the analyzer's `catch_unwind`. `heal_json` copied
  quoted-string bytes `as char` (a Latin-1 reinterpretation),
  mojibaking non-ASCII tool arguments into valid JSON that serde
  then accepted — silent corruption on every heal path.
  `parse_tagged_call` gained a per-iteration progress guard: a
  degenerate `CallSyntax` (reachable via the analyzer's fallback)
  consumed zero bytes per iteration forever; zero progress is now
  `Malformed`. The `Malformed`-degrade path and
  `extract_args_markers` no longer slice mid-character or chop real
  content on non-ASCII input.
- **An over-`CAPACITY` `ngram_max_size` panicked prompt seeding.**
  The prose-seeding fold now clamps to `NGram::CAPACITY` like the
  live penalty pass does; previously `with_repetition` with a large
  `ngram_max_size` plus any `complete_*` over a long-enough prose
  block hit an `unwrap` in the fold.

### Removed

- **`cli::Args`** and **`LlamaCppEngine::from_cli`** — superseded by
  `LlamaCppOptions`, which is the same idea with the missing knobs,
  serde support, and no CLI dependency. `regurgitater` wraps it in
  its own `Parser` struct.
- **`Session::from_path_cpu_only`**, **`Session::from_path_with_flash_attention`**,
  **`Session::from_path_with_cache_slots`**, and the matching
  `LlamaCppEngine::from_path_cpu_only` /
  `from_path_with_flash_attention` / `from_path_with_n_ctx_and_seqs`
  — all expressible as `from_path_with(path, options)`. The first two
  had no callers at all.

- **`blallama --repetition-penalty`** (the v0.7.x band-aid opt-in
  flag). Sampling configuration now comes from the per-model
  sidecar; for force-off probe runs use the new `--no-penalty`
  flag, which overrides the sidecar.
- **`StopWords` and the `*_stopwords` methods** — the pure-rename
  shims deprecated in 0.7.0 (`type StopWords = IgnoreCategory` and
  the four `ignored_stopwords`-family methods on
  `RepetitionOptions`). Live replacements have existed since 0.7.0,
  and 0.8.0 is a breaking release, so they go now rather than riding
  through another one. The TOML sidecar's legacy `ignored_stopwords`
  key keeps working — that back-compat lives in a serde alias on the
  field, not in the removed API.

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
  `session.engine().model().display_name()` over the now-llama-cpp-
  only `session.engine().model().file_name()`.
- `Engine.model` is no longer a `pub` field ([#54], see *Fixed*);
  use the `Engine::model()` accessor.
- `Session::from_path_sync(p)` → `Session::from_path(p)`, with
  `use drama_llama::FromPath;` — it is a trait method now.
- The specialized constructors become one call with an options
  struct. `from_path_with_cache_slots(p, 4096, 3)` becomes:

  ```rust
  Session::from_path_with(
      p,
      LlamaCppOptions::default().with_n_ctx(4096).with_cache_slots(3),
  )
  ```

  Note `cache_slots: Some(1)` is still not the same as leaving it
  unset — it switches the KV cache to unified, exactly as the old
  three-argument constructor did.
- Loading in async code: `from_path_async(path, options)` runs the
  load on tokio's blocking pool. The old `FromPath::from_path` was
  async; the new one is sync.

### Notes

- **`blallama` takes its repetition-penalty setting from the
  per-model sidecar**, not from a flag. The v0.7.x defaults
  (`penalty_max_count=1`, `ngram_min_size=1`, `penalty_repeat=1.06`)
  were sized for small downstream models in Weave; on the larger MoE
  models drama_llama now drives they over-penalise common content
  tokens during long-form free-text generation and degrade output to
  thesaurus chains or sentence-fragment loops. `--no-penalty`
  overrides the sidecar to force the filter OFF for probe and canary
  runs; there is no flag to force it on, because the sidecar is where
  that decision now lives.

  (This bullet said "no longer enables the filter by default … new
  `--repetition-penalty` flag re-enables it", carried over from
  v0.7.x. That flag was removed in this same release — see *Removed*
  — and the sentence had been contradicting the entry three sections
  up ever since.)
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
- **Dev workflow: `justfile` + cargo-nextest** (`just setup` installs
  nextest). `just test` runs the fast suite GPU-accelerated,
  `just test ignored` runs only the long-running `#[ignore]`d
  GPU/model tests, `just test all` runs both, and `just test cpu`
  runs CPU-only — the recipes are thin wrappers over
  `scripts/test.py`, which owns the configuration × tier topology so
  the justfile, the hooks, and CI cannot drift ([#68]). CUDA is
  auto-enabled on Linux (Metal is automatic on macOS) — deliberately
  kept OUT of the crate's default features so a bare `cargo build`
  stays portable. A 30B-class model barely fits once on a 24GB card,
  so the model tests must run one-at-a-time; the nextest `full`
  profile caps `test-threads = 1` (see `.config/nextest.toml`) and
  `just test ignored` / `just test all` run under it, so the whole
  set is serialized without a fragile per-test filter. GPU vs. CPU
  builds use separate target dirs to avoid evicting each other's
  llama.cpp build.
- Send/Sync trade-offs: `B::Decoder` is required Send (not Sync) —
  `*mut llama_context` is internally mutable. `B::Model` is Send +
  Sync (Iterator impls hand `&Model` to grammar / sampling code
  that fans out across rayon).
- See `.claude/memory/moeflux_disk_convention.md` for the
  forward-looking on-disk layout `MoefluxEngine::from_path`
  expects, and the migration story for current artifacts.
- **Known issues** (tracked, not fixed in this release):
  - Context-full during generation ends the stream silently and, on
    the grammar path, misreports as a `GrammarViolation` ([#36]).
  - A tool call truncated by the token budget can seat frame-marker
    text in the transcript; containment (not prevention) is the
    planned fix ([#38]).
  - On the un-grammared lazy path, nothing forbids EOG while a
    thought block is open, so a model can end its turn mid-thought
    ([#64]).

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

[0.8.0]: https://github.com/mdegans/drama_llama/releases/tag/v0.8.0
[0.7.0]: https://github.com/mdegans/drama_llama/releases/tag/v0.7.0
[#28]: https://github.com/mdegans/drama_llama/issues/28
[#36]: https://github.com/mdegans/drama_llama/issues/36
[#38]: https://github.com/mdegans/drama_llama/issues/38
[#64]: https://github.com/mdegans/drama_llama/issues/64
[#29]: https://github.com/mdegans/drama_llama/issues/29
[#30]: https://github.com/mdegans/drama_llama/issues/30
[#31]: https://github.com/mdegans/drama_llama/issues/31
[#37]: https://github.com/mdegans/drama_llama/issues/37
[#40]: https://github.com/mdegans/drama_llama/issues/40
[#43]: https://github.com/mdegans/drama_llama/issues/43
[#44]: https://github.com/mdegans/drama_llama/issues/44
[#48]: https://github.com/mdegans/drama_llama/issues/48
[#54]: https://github.com/mdegans/drama_llama/issues/54
[#60]: https://github.com/mdegans/drama_llama/issues/60
[#68]: https://github.com/mdegans/drama_llama/issues/68
[#76]: https://github.com/mdegans/drama_llama/issues/76
