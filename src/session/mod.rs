//! High-level ergonomic wrapper around [`LlamaCppEngine`] for chat-style tool-using
//! inference.
//!
//! [`Session`] is to local inference what [`misanthropic::Client::message`] is
//! to the Anthropic API: given a [`Prompt`], get back a
//! [`response::Message`](misanthropic::response::Message) via
//! [`Session::complete_response`], typed [`Block`]s via
//! [`Session::complete_blocks`], or raw bytes via [`Session::complete_text`].
//! The caller builds their [`Prompt`] with misanthropic's normal builders and
//! lets `Session` handle rendering, grammar enforcement, sampling, streaming
//! block parsing, and — opt-in — prefix-cache reuse across calls.
//!
//! ```no_run
//! use drama_llama::{Prompt, Session};
//!
//! let mut session = Session::from_path("models/model.gguf".into())
//!     .unwrap()
//!     .quiet();
//! let prompt = Prompt::default(); // + system, messages, tools, etc.
//! let raw = session.complete_text(&prompt).unwrap();
//! println!("{raw}");
//! ```
//!
//! # What `Session` does for you
//!
//! * Renders the prompt through the model's embedded Jinja chat template (via
//!   [`ChatTemplate`]).
//! * Compiles any [`ToolChoice`] into a [`SamplingMode::Grammar`] via
//!   [`grammar_for_prompt`], and **prepends** it to the caller's sampling chain
//!   each call. [`Session::with_sampling`] only replaces the user portion — it
//!   can't override the grammar.
//! * Tokenizes, runs the predictor, collects the result.
//! * Streams or batches [`Block`]s via [`Session::complete_stream`] /
//!   [`Session::complete_blocks`]; returns a full
//!   [`response::Message`](misanthropic::response::Message) via
//!   [`Session::complete_response`].
//! * Optionally reuses KV state across calls when the caller opts in via
//!   [`Session::with_prefix_cache`] (see below).
//!
//! # Prefix caching
//!
//! Local inference has no "cache creation" cost in the Anthropic sense — the
//! whole prompt is decoded on every call anyway — but it *does* pay a linear
//! prefill cost in tokens. When successive calls share a long prefix (system
//! + tools + early turns), re-prefilling those positions wastes work. The
//! opt-in prefix cache keeps the KV state from the previous call around and, on
//! the next call, computes the longest common prefix of `new_tokens` and
//! `prev_tokens`, clipped to the nearest `cache_control` breakpoint declared in
//! the prompt, and resumes generation from that position via
//! [`LlamaCppEngine::predict_pieces_resuming`].
//!
//! The contract:
//!
//! * **Opt-in.** Default is off — existing callers are unaffected. Enable with
//!   [`Session::with_prefix_cache(true)`](Session::with_prefix_cache).
//! * **Breakpoint-driven.** The cache only honors positions the caller
//!   explicitly marked with a `cache_control` on a [`Block`], [`Tool`], or
//!   [`tool::Use`](misanthropic::tool::Use) /
//!   [`tool::Result`](misanthropic::tool::Result). Without breakpoints, every
//!   call is a full re-prefill.
//! * **Single sequence.** All prefill/decode uses `seq_id = 0`. Parallel
//!   conversation threads need one [`Session`] each.
//! * **Thread swap = clear.** When swapping conversation threads or reloading
//!   system/tools outside the `cache_control` contract, call
//!   [`Session::clear_prefix_cache`] to zero both the cache metadata and the KV
//!   state. The library can't detect semantic-level context swaps on its own.
//!
//! Usage statistics matching the Anthropic API shape are tracked on every
//! `complete_*` call: see [`Session::last_usage`] and [`Session::total_usage`].
//!
//! [`misanthropic::Client::message`]:
//!     https://docs.rs/misanthropic/latest/misanthropic/struct.Client.html#method.message
//! [`ToolChoice`]: crate::ToolChoice
//! [`Block`]: crate::Block
//! [`Tool`]: crate::Tool

use std::{num::NonZeroUsize, path::PathBuf};

use misanthropic::response::Usage;

use crate::{
    backend::{Backend, Model}, chat_template::tokenize_with_breakpoints,
    grammar_for_prompt, output_config, ChatTemplate, ChatTemplateError, Engine,
    OutputConfigError, OutputConfigOptions, PredictOptions, Prompt,
    RenderOptions, RepetitionOptions, SampleOptions, SamplingMode, Token,
    ToolChoiceError, ToolChoiceOptions,
};

#[cfg(feature = "llama-cpp")]
use crate::{silence_logs, LlamaCppBackend, NewError};

#[cfg(all(feature = "moeflux", target_os = "macos"))]
use crate::{moeflux::engine::MoefluxEngineError, MoefluxBackend};

mod parse;
pub use parse::{parse_completion, BlockParser};

/// Errors from [`Session`].
#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    /// llama.cpp engine setup failed (model load or context init).
    /// Only emitted by `Session<LlamaCppBackend>::from_path*`
    /// constructors.
    #[cfg(feature = "llama-cpp")]
    #[error("llama.cpp engine setup: {0}")]
    LlamaCppEngine(#[from] NewError),
    /// Moeflux engine setup failed (artifact discovery, MLX parse, or
    /// `mf_init_model`). Only emitted by
    /// `Session<MoefluxBackend>::from_path`.
    #[cfg(all(feature = "moeflux", target_os = "macos"))]
    #[error("moeflux engine setup: {0}")]
    MoefluxEngine(#[from] MoefluxEngineError),
    /// The model has no embedded `tokenizer.chat_template`, or the template
    /// failed to compile.
    #[error("chat template: {0}")]
    ChatTemplate(#[from] ChatTemplateError),
    /// [`ToolChoice`] couldn't be compiled into a grammar — the referenced tool
    /// doesn't exist, the schema is malformed, etc.
    ///
    /// [`ToolChoice`]: crate::ToolChoice
    #[error("tool choice: {0}")]
    ToolChoice(#[from] ToolChoiceError),
    /// [`OutputConfig`] couldn't be compiled into a grammar — the schema is
    /// malformed or uses an unsupported `OutputFormat` variant.
    ///
    /// [`OutputConfig`]: misanthropic::prompt::output::OutputConfig
    #[error("output config: {0}")]
    OutputConfig(#[from] OutputConfigError),
    /// Grammar-forced generation ended without producing a parseable tool call.
    /// Usually means the model was truncated by `max_tokens` before closing the
    /// `</tool_call>` tag, or (less commonly) the grammar itself has a gap that
    /// let the model produce bytes the parser couldn't interpret as a tool
    /// call.
    #[error("grammar violation: forced tool call did not produce a tool_use block; partial_output={partial_output:?}")]
    GrammarViolation {
        /// Any prose / thought blocks that streamed before the violation was
        /// detected. Callers can surface this to the user or log it for
        /// debugging.
        partial_output: String,
    },
}

/// Default maximum tokens per [`Session::complete_text`] call. Users override
/// via [`Session::with_max_tokens`].
const DEFAULT_MAX_TOKENS: usize = 1024;

/// Per-session prefix-cache state.
///
/// Tracks the full prompt tokens from the last cache-participating `complete_*`
/// call, the indices within those tokens where `cache_control` breakpoints
/// landed (sorted ascending), and the number of tokens actually reused on the
/// last call. Generation tokens are **not** stored — they're overwritten on the
/// next call whose prompt extends past the reused prefix.
///
/// Private to the session module; callers interact through
/// [`Session::with_prefix_cache`] / [`Session::clear_prefix_cache`] /
/// [`Session::last_usage`].
struct PrefixCache {
    /// Full prompt tokens from the last cache-participating call. Generation
    /// tokens are NOT stored here.
    prev_tokens: Vec<Token>,
    /// Token indices in `prev_tokens` where `cache_control` breakpoints landed.
    /// Sorted ascending.
    prev_breakpoints: Vec<usize>,
    /// Tokens reused in the last call. `0` = full re-prefill.
    last_reused_tokens: usize,
}

impl PrefixCache {
    /// Fresh, empty cache.
    fn new() -> Self {
        Self {
            prev_tokens: Vec::new(),
            prev_breakpoints: Vec::new(),
            last_reused_tokens: 0,
        }
    }

    /// Zero every field. Called from [`Session::clear_prefix_cache`].
    fn clear(&mut self) {
        self.prev_tokens.clear();
        self.prev_breakpoints.clear();
        self.last_reused_tokens = 0;
    }
}

/// Length of the longest prefix shared between `a` and `b`, in tokens.
fn longest_common_prefix_len(a: &[Token], b: &[Token]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// Cache-reuse length for a call.
///
/// Given the previously-cached `prev_tokens`, the newly-rendered `new_tokens`,
/// and the new call's breakpoint token indices (sorted ascending), compute
/// `L_hit`: the largest breakpoint index that is
///
/// 1. less than or equal to the common-prefix length of the two token streams,
///    with one token of BPE-boundary safety (to avoid reusing a position whose
///    successor might tokenize differently); and
/// 2. strictly greater than zero (we only reuse at breakpoints).
///
/// Returns `0` when no breakpoint is eligible — the caller should treat that as
/// a full re-prefill. Pure function, tested directly.
fn compute_l_hit(
    prev_tokens: &[Token],
    new_tokens: &[Token],
    new_breakpoints: &[usize],
) -> usize {
    let lcp = longest_common_prefix_len(prev_tokens, new_tokens);
    // BPE-boundary safety: back off by one token so a breakpoint falling
    // exactly at the prefix end can't reuse a position whose successor might
    // re-tokenize differently once more context is added.
    let safe = if lcp == 0 { 0 } else { lcp - 1 };
    new_breakpoints
        .iter()
        .rev()
        .find(|&&bp| bp <= safe && bp > 0)
        .copied()
        .unwrap_or(0)
}

/// One generated-position entry in a [`Session::top_k_trace`] dump.
///
/// Mirrors the shape of ollama's `choices[].logprobs.content[]` so
/// trace-vs-trace diffs don't need an intermediate normalization step.
#[derive(Debug, Clone)]
pub struct TokenTrace {
    /// 0-indexed position in the generated sequence.
    pub position: usize,
    /// Top-k candidates **after grammar filtering** (if the prompt's
    /// `tool_choice` compiled to one), sorted by logit descending. Entry 0 is
    /// the greedy argmax that was committed to advance generation.
    pub top_k: Vec<TopKEntry>,
}

/// One candidate row inside a [`TokenTrace`].
#[derive(Debug, Clone)]
pub struct TopKEntry {
    /// Vocabulary id.
    pub token: Token,
    /// Raw logit from the model (pre-softmax).
    pub logit: f32,
    /// Decoded string for this token (via `LlamaCppModel::token_to_piece`).
    pub piece: String,
}

/// Chat-style inference session: owns an [`Engine`] + [`ChatTemplate`]
/// plus the builder-configured defaults for each `complete_*` call.
///
/// Generic over a [`Backend`] so the same chat-style surface drives
/// either llama.cpp ([`LlamaCppBackend`]) or moeflux
/// ([`MoefluxBackend`]). Backend-specific constructors
/// (`Session::<B>::from_path*`) live in specialized impl blocks; the
/// rest of the API is generic.
pub struct Session<B: Backend> {
    engine: Engine<B>,
    template: ChatTemplate,
    tool_choice_opts: ToolChoiceOptions,
    output_config_opts: OutputConfigOptions,
    render_opts: RenderOptions,
    /// User's sampling chain WITHOUT any grammar. Grammar is prepended
    /// transiently inside `complete_*`. Defaults to
    /// `[SamplingMode::locally_typical()]`.
    sample_modes: Vec<SamplingMode>,
    /// Optional repetition penalty. Defaults to `None`: chat-style use wants
    /// the model to be able to repeat natural short tokens (words, punctuation,
    /// digits that appeared in the context — especially important for
    /// tool-result follow-ups where the answer IS the digit). Story generation
    /// can opt in via [`Session::with_repetition`].
    repetition: Option<RepetitionOptions>,
    max_tokens: NonZeroUsize,
    /// Prefix-cache state. `Some` iff the caller opted in via
    /// [`Session::with_prefix_cache(true)`](Session::with_prefix_cache).
    /// `None` means every call is a full re-prefill (the pre-0.7
    /// behavior).
    prefix_cache: Option<PrefixCache>,
    /// [`Usage`] from the most recent `complete_*` call. Zeroed on
    /// construction; overwritten on every call.
    last_usage: Usage,
    /// Cumulative [`Usage`] across every `complete_*` call on this
    /// `Session`. Zeroed on construction; never reset except by
    /// dropping and rebuilding the `Session`.
    total_usage: Usage,
}

// llama.cpp-specific constructors. Available only when the
// `llama-cpp` feature is enabled.
#[cfg(feature = "llama-cpp")]
impl Session<LlamaCppBackend> {
    /// Load a model from disk and wire up the chat template.
    pub fn from_path(path: PathBuf) -> Result<Self, SessionError> {
        let engine = crate::LlamaCppEngine::from_path(path)?;
        Self::from_engine(engine)
    }

    /// Load a model from disk with an explicit Flash Attention policy.
    ///
    /// Diagnostic escape hatch for output-divergence debugging — see
    /// [`FlashAttention`](crate::FlashAttention) for the when and why.
    pub fn from_path_with_flash_attention(
        path: PathBuf,
        fa: crate::FlashAttention,
    ) -> Result<Self, SessionError> {
        let engine =
            crate::LlamaCppEngine::from_path_with_flash_attention(path, fa)?;
        Self::from_engine(engine)
    }

    /// Load a model from disk with an explicit KV context size.
    ///
    /// [`Self::from_path`] inherits llama.cpp's default `n_ctx = 512`,
    /// which truncates chat and structured-output workloads well
    /// before they finish. Use this builder when the prompt plus the
    /// generation cap ([`Self::with_max_tokens`]) can exceed 512
    /// tokens — which is almost always for reasoning-capable models.
    /// Typical values: 4096 – 16384.
    pub fn from_path_with_n_ctx(
        path: PathBuf,
        n_ctx: u32,
    ) -> Result<Self, SessionError> {
        let engine = crate::LlamaCppEngine::from_path_with_n_ctx(path, n_ctx)?;
        Self::from_engine(engine)
    }

    /// Load a model CPU-only (zero GPU layers). Diagnostic path for
    /// isolating GPU-kernel divergence.
    pub fn from_path_cpu_only(path: PathBuf) -> Result<Self, SessionError> {
        let engine = crate::LlamaCppEngine::from_path_cpu_only(path)?;
        Self::from_engine(engine)
    }

    /// Silence llama.cpp's log spew (model load progress, KV cache
    /// setup, compute buffer sizing, etc.). Process-global effect —
    /// calling it on any [`Session`] silences logs for every
    /// subsequent inference in the process.
    ///
    /// llama.cpp-specific. The [`restore_default_logs`](crate::restore_default_logs)
    /// free function flips the flag back.
    pub fn quiet(self) -> Self {
        silence_logs();
        self
    }
}

// Moeflux-specific constructor. Available only on macOS with the
// `moeflux` feature enabled.
#[cfg(all(feature = "moeflux", target_os = "macos"))]
impl Session<MoefluxBackend> {
    /// Load a moeflux model from a parent directory using the
    /// drama_llama folder convention: `parent/mlx/`,
    /// `parent/artifacts/`, `parent/root/` (the experts dir).
    /// Defaults `experts_per_tok = 8`, `use_2bit = false` — the Qwen3
    /// MoE 4-bit setup. Power users who need explicit paths or
    /// non-default runtime params can construct a
    /// [`crate::MoefluxEngine`] directly via `MoefluxEngine::from_paths`
    /// and hand it to [`Self::from_engine`].
    pub fn from_path(parent: PathBuf) -> Result<Self, SessionError> {
        let engine = crate::MoefluxEngine::from_path(&parent)?;
        Self::from_engine(engine)
    }
}

impl<B: Backend> Session<B> {
    /// Wrap an already-constructed [`Engine`]. Useful when the engine
    /// was built with custom parameters (specific context size, GPU
    /// layout, moeflux runtime knobs, ...).
    pub fn from_engine(engine: Engine<B>) -> Result<Self, SessionError> {
        let template = ChatTemplate::from_model(&engine.model)?;
        Ok(Self {
            engine,
            template,
            tool_choice_opts: ToolChoiceOptions::default(),
            output_config_opts: OutputConfigOptions::default(),
            render_opts: RenderOptions::default().with_generation_prompt(true),
            sample_modes: vec![SamplingMode::locally_typical()],
            repetition: None,
            max_tokens: NonZeroUsize::new(DEFAULT_MAX_TOKENS).unwrap(),
            prefix_cache: None,
            last_usage: Usage::default(),
            total_usage: Usage::default(),
        })
    }

    /// Enable (or replace) repetition penalty. Default is `None`
    /// because chat flows need the model to repeat natural tokens —
    /// in particular, digits that appeared in a tool result (the
    /// answer is often the same digit the tool just returned).
    ///
    /// Opt in for story generation, poetry, or anywhere loop-
    /// prevention matters. See [`RepetitionOptions`] for parameters.
    ///
    /// The full set of model special tokens (EOS, EOT, BOS,
    /// chat-template markers like `<|start_header_id|>` /
    /// `<|eot_id|>`, tool-call markers like `<|python_tag|>`) is
    /// added to `opts.ignored` before storing — a strong repetition
    /// penalty on those would prevent the model from ever closing a
    /// turn or emitting a valid tool call.
    pub fn with_repetition(mut self, mut opts: RepetitionOptions) -> Self {
        opts.extend_ignored(self.engine.model.special_tokens());
        self.repetition = Some(opts);
        self
    }

    /// Clear any repetition penalty — the explicit "no penalty"
    /// state, equivalent to the default.
    pub fn without_repetition(mut self) -> Self {
        self.repetition = None;
        self
    }

    /// Override the defaults used when compiling [`ToolChoice`] into a grammar
    /// (e.g. `wrap_tags`, `arguments_field`, `allow_thought`, `strict_schema`).
    ///
    /// Cogito / Qwen / Hermes templates want `wrap_tags =
    /// Some(("<tool_call>\n", "\n</tool_call>"))`, `arguments_field =
    /// "arguments"`, and `allow_thought = true`. See [`ToolChoiceOptions`] for
    /// defaults.
    ///
    /// [`ToolChoice`]: crate::ToolChoice
    pub fn with_tool_choice_opts(mut self, opts: ToolChoiceOptions) -> Self {
        self.tool_choice_opts = opts;
        self
    }

    /// Override the defaults used when compiling
    /// [`Prompt::output_config`] into a grammar — today just whether an
    /// optional `<think>...</think>` block is permitted before the
    /// JSON body. Defaults are `allow_thought: true`, which is usually
    /// what you want for reasoning-capable models.
    ///
    /// Unlike [`Self::with_tool_choice_opts`], this only matters when
    /// the prompt has `output_config` set; it's otherwise a no-op.
    ///
    /// [`Prompt::output_config`]: misanthropic::Prompt::output_config
    pub fn with_output_config_opts(
        mut self,
        opts: OutputConfigOptions,
    ) -> Self {
        self.output_config_opts = opts;
        self
    }

    /// Override the defaults used when rendering the prompt through the chat
    /// template. The generation-prompt flag is forced to `true` regardless —
    /// `Session` is always rendering for live inference, never archival.
    pub fn with_render_opts(mut self, opts: RenderOptions) -> Self {
        self.render_opts = opts.with_generation_prompt(true);
        self
    }

    /// Replace the user-specified sampling chain. Grammar is prepended
    /// transiently inside `complete_*` when [`Prompt::tool_choice`] is
    /// `Some(Method | Any)`, so this signature intentionally does NOT accept a
    /// grammar mode — set grammar via [`Prompt::tool_choice`] +
    /// [`with_tool_choice_opts`] instead.
    ///
    /// Passing an empty iterator is valid: the model will sample with no
    /// post-grammar filters at all.
    ///
    /// [`Prompt::tool_choice`]: crate::Prompt
    /// [`with_tool_choice_opts`]: Self::with_tool_choice_opts
    pub fn with_sampling<I>(mut self, modes: I) -> Self
    where
        I: IntoIterator<Item = SamplingMode>,
    {
        self.sample_modes = modes.into_iter().collect();
        self
    }

    /// Set the maximum tokens generated per `complete_*` call.
    ///
    /// This is a *defensive* generation cap. Every `Prompt` carries its
    /// own [`Prompt::max_tokens`] (Anthropic-API required field,
    /// `NonZeroU32`), and the effective cap for any call is
    /// `min(prompt.max_tokens, self.max_tokens)` — per-request wins when
    /// it's smaller, Session's cap clips it when the request asks for
    /// more than this Session is willing to emit. Set this high (or to
    /// the model's `n_ctx`) if you want Prompt's value to always win.
    ///
    /// This is a *generation* cap, independent of the engine's KV
    /// context size (`n_ctx`). If the prompt plus `n` exceeds the
    /// engine's configured `n_ctx`, generation truncates at the KV
    /// cache boundary regardless of this value — reached via
    /// [`Self::from_path_with_n_ctx`] or by constructing an [`LlamaCppEngine`]
    /// directly.
    pub fn with_max_tokens(mut self, n: NonZeroUsize) -> Self {
        self.max_tokens = n;
        self
    }

    /// Effective generation cap for a single call: the minimum of
    /// `prompt.max_tokens` (per-request, Anthropic-API-required) and
    /// `self.max_tokens` (Session-level defensive ceiling).
    ///
    /// Both inputs are `NonZero`, so the minimum is also `NonZero`.
    fn effective_max_tokens(&self, prompt: &Prompt) -> NonZeroUsize {
        let req = prompt.max_tokens.get() as usize;
        let cap = self.max_tokens.get();
        NonZeroUsize::new(req.min(cap))
            .expect("min of two NonZero values is NonZero")
    }

    /// Enable (or disable) prefix-cache reuse across `complete_*`
    /// calls.
    ///
    /// Default is disabled — existing callers are unaffected unless
    /// they opt in. When enabled, `Session` honors `cache_control`
    /// breakpoints on [`Block`](crate::Block)s,
    /// [`tool::Method`](misanthropic::tool::Method)s,
    /// [`tool::Result`](misanthropic::tool::Result)s, and
    /// [`tool::Use`](misanthropic::tool::Use)s, resuming generation
    /// from the longest prefix shared with the previous call (clipped
    /// to the nearest declared breakpoint).
    ///
    /// Enabling when already enabled is a no-op; disabling clears any
    /// cached prefix metadata AND the KV cache (delegates to
    /// [`Self::clear_prefix_cache`]).
    pub fn with_prefix_cache(mut self, on: bool) -> Self {
        if on {
            if self.prefix_cache.is_none() {
                self.prefix_cache = Some(PrefixCache::new());
            }
        } else if self.prefix_cache.is_some() {
            self.clear_prefix_cache();
            self.prefix_cache = None;
        }
        self
    }

    /// Clear both the cached prefix metadata AND the KV cache.
    ///
    /// Call when swapping conversation threads or reloading
    /// system/tools outside the `cache_control` contract — the
    /// library can't detect semantic-level context swaps on its own,
    /// and silently reusing stale KV state across unrelated
    /// conversations would produce incoherent output.
    ///
    /// No-op on the KV side if the prefix cache is disabled, but
    /// still safe to call.
    pub fn clear_prefix_cache(&mut self) {
        if let Some(cache) = self.prefix_cache.as_mut() {
            cache.clear();
        }
        self.engine.memory_clear();
    }

    /// The [`Usage`] from the most recent `complete_*` call. Zeroed
    /// at [`Session`] construction; overwritten on every call.
    ///
    /// For local inference, `cache_creation_input_tokens` is always
    /// `Some(0)` — there's no asymmetric creation-vs-read cost like
    /// the Anthropic API has. `cache_read_input_tokens` is the number
    /// of prompt tokens reused from the previous call's KV state, or
    /// `Some(0)` when caching is disabled or the call missed.
    pub fn last_usage(&self) -> &Usage {
        &self.last_usage
    }

    /// Cumulative [`Usage`] across every `complete_*` call on this
    /// [`Session`]. Zeroed at construction; never reset except by
    /// dropping and rebuilding the `Session`. Follows misanthropic's
    /// [`Usage: AddAssign<Usage>`][aa] convention — cache counters
    /// saturate to `Some(total)` once any call produces a value.
    ///
    /// [aa]: misanthropic::response::Usage
    pub fn total_usage(&self) -> &Usage {
        &self.total_usage
    }

    /// Borrow the underlying [`Engine`] — useful when the caller needs
    /// raw predictor access for something `Session` doesn't expose yet
    /// (e.g. custom stop-sequence management). Concretely this returns
    /// `&LlamaCppEngine` or `&MoefluxEngine` depending on `B`, since
    /// those are type aliases for `Engine<...Backend>`.
    pub fn engine(&self) -> &Engine<B> {
        &self.engine
    }

    /// Mutable borrow of the underlying [`Engine`]. Handy for KV-cache
    /// manipulation across turns.
    pub fn engine_mut(&mut self) -> &mut Engine<B> {
        &mut self.engine
    }

    /// Borrow the compiled chat template.
    pub fn template(&self) -> &ChatTemplate {
        &self.template
    }

    /// Shared setup for every `complete_*` entry point: render the
    /// prompt through the chat template, tokenize with
    /// `parse_special=true` (so `<|im_start|>` etc. resolve to their
    /// single special-token IDs), and build the effective sampling
    /// chain — grammar from [`Prompt::tool_choice`] prepended,
    /// optionally followed by [`Self::with_sampling`]'s user filters.
    ///
    /// `include_user_sampling = true` for production calls
    /// ([`Self::complete_text`] / [`Self::complete_stream`]).
    /// `include_user_sampling = false` for diagnostic calls
    /// ([`Self::top_k_trace`]) that want the raw grammar-filtered
    /// candidate distribution without user-filter shaping.
    ///
    /// Returns the token ids and the [`SamplingMode`] chain; callers
    /// wire them into whatever predictor / `PredictOptions` shape they
    /// need.
    ///
    /// [`Prompt::tool_choice`]: crate::Prompt
    fn prepare_call(
        &mut self,
        prompt: &Prompt,
        include_user_sampling: bool,
    ) -> Result<
        (
            Vec<Token>,
            Vec<SamplingMode>,
            Option<crate::DeferredGrammar>,
        ),
        SessionError,
    > {
        let rendered = self.template.render_with(prompt, &self.render_opts)?;
        // parse_special=true: the rendered prompt contains chat markers
        // (`<|im_start|>`, `<|im_end|>`, etc.) that must tokenize to
        // their single special-token IDs, not to the individual ASCII
        // characters. Passing false causes `<|im_start|>` to tokenize
        // as 6 tokens instead of 1, producing a completely different
        // input for the model — diagnosed as the cause of cogito's
        // wrong-letter + loop behavior in strawberry.
        let tokens = self.engine.model.tokenize(&rendered, true);

        // Grammar (if any) is prepended so it runs first and narrows
        // candidates down to grammar-legal tokens before user filters
        // further shape the distribution. A deferred grammar is carried
        // separately (not in `modes`) — it stays suspended until
        // `TokenPredictor` sees its trigger in the output.
        let (grammar_mode, deferred) = match resolve_grammar(
            prompt,
            &self.tool_choice_opts,
            &self.output_config_opts,
        )? {
            None => (None, None),
            Some(crate::CompiledOutputConfig::Single(g)) => (Some(g), None),
            Some(crate::CompiledOutputConfig::Deferred(d)) => (None, Some(d)),
        };
        let modes: Vec<SamplingMode> = if include_user_sampling {
            grammar_mode
                .into_iter()
                .chain(self.sample_modes.iter().cloned())
                .collect()
        } else {
            grammar_mode.into_iter().collect()
        };
        Ok((tokens, modes, deferred))
    }

    /// Cache-aware superset of [`Self::prepare_call`]: renders the
    /// prompt **with** cache breakpoints, tokenizes both the full
    /// render and each partial via
    /// [`tokenize_with_breakpoints`](crate::chat_template::tokenize_with_breakpoints),
    /// and returns the full token stream, the breakpoint token
    /// indices (sorted ascending), and the sampling-mode chain.
    ///
    /// When the caller has not enabled prefix caching
    /// ([`Self::with_prefix_cache(false)`](Self::with_prefix_cache)),
    /// this function skips the partial-render + tokenize passes and
    /// returns an empty breakpoint list — breakpoints are never
    /// consulted in that mode anyway, so computing them is wasted
    /// work.
    fn prepare_call_cached(
        &mut self,
        prompt: &Prompt,
        include_user_sampling: bool,
    ) -> Result<
        (
            Vec<Token>,
            Vec<usize>,
            Vec<SamplingMode>,
            Option<crate::DeferredGrammar>,
        ),
        SessionError,
    > {
        let (tokens, breakpoints) = if self.prefix_cache.is_some() {
            let rendered = self
                .template
                .render_with_breakpoints(prompt, &self.render_opts)?;
            tokenize_with_breakpoints(&self.engine.model, &rendered)
        } else {
            // Fast path: single render + tokenize, no partials.
            let rendered =
                self.template.render_with(prompt, &self.render_opts)?;
            (self.engine.model.tokenize(&rendered, true), Vec::new())
        };

        let (grammar_mode, deferred) = match resolve_grammar(
            prompt,
            &self.tool_choice_opts,
            &self.output_config_opts,
        )? {
            None => (None, None),
            Some(crate::CompiledOutputConfig::Single(g)) => (Some(g), None),
            Some(crate::CompiledOutputConfig::Deferred(d)) => (None, Some(d)),
        };
        let modes: Vec<SamplingMode> = if include_user_sampling {
            grammar_mode
                .into_iter()
                .chain(self.sample_modes.iter().cloned())
                .collect()
        } else {
            grammar_mode.into_iter().collect()
        };
        Ok((tokens, breakpoints, modes, deferred))
    }

    /// Prefix-cache KV-state setup shared by every batch `complete_*`
    /// entry point.
    ///
    /// Given the newly-tokenized prompt and its breakpoint indices,
    /// computes `L_hit` (tokens reusable from the previous call's KV
    /// state), narrows the KV cache to `[0, L_hit)` via
    /// [`LlamaCppEngine::memory_seq_rm`] (or clears it entirely on miss), and
    /// returns the suffix of `new_tokens` the predictor must decode
    /// plus the reuse length. The caller still owns whether / when to
    /// update `self.prefix_cache` — batch callers do it *after*
    /// success; streaming callers do it *before* the predictor borrow.
    ///
    /// This function touches the KV cache but nothing else on `self`
    /// beyond the engine.
    fn kv_setup_for_call(
        &mut self,
        new_tokens: &[Token],
        new_breakpoints: &[usize],
    ) -> (Vec<Token>, usize) {
        let l_hit = match self.prefix_cache.as_ref() {
            Some(cache) if !cache.prev_tokens.is_empty() => {
                compute_l_hit(&cache.prev_tokens, new_tokens, new_breakpoints)
            }
            _ => 0,
        };
        if l_hit > 0 {
            // Narrow the KV cache to the reused prefix. Anything past
            // `l_hit` on seq 0 (old generation + old suffix) is dropped
            // so the resuming prefill can write the new suffix into
            // fresh positions.
            self.engine.memory_seq_rm(0, l_hit as i32, -1);
        } else {
            // Full re-prefill: wipe everything.
            self.engine.memory_clear();
        }
        let suffix = new_tokens[l_hit..].to_vec();
        (suffix, l_hit)
    }

    /// Build a [`Usage`] for one `complete_*` call. `Option` fields
    /// are always populated — locally we know both cache counters
    /// exactly, so recording them explicitly (even as `Some(0)`) is
    /// more informative than `None` and keeps
    /// [`Usage::AddAssign`](std::ops::AddAssign) behavior well-
    /// defined across calls.
    fn make_usage(
        prompt_tokens: usize,
        cache_read: usize,
        output_tokens: usize,
    ) -> Usage {
        Usage {
            input_tokens: prompt_tokens as u64,
            cache_creation_input_tokens: Some(0),
            cache_read_input_tokens: Some(cache_read as u64),
            output_tokens: output_tokens as u64,
        }
    }

    /// After a batch call succeeds, update [`self.prefix_cache`] to
    /// describe the current KV state: full prompt tokens, breakpoint
    /// indices, and actual reuse length. No-op when caching is off.
    fn record_cache_hit(
        &mut self,
        new_tokens: Vec<Token>,
        new_breakpoints: Vec<usize>,
        l_hit: usize,
    ) {
        if let Some(cache) = self.prefix_cache.as_mut() {
            cache.prev_tokens = new_tokens;
            cache.prev_breakpoints = new_breakpoints;
            cache.last_reused_tokens = l_hit;
        }
    }

    /// After a batch call fails, invalidate [`self.prefix_cache`] and
    /// wipe the KV state — partial decodes may have left the cache
    /// inconsistent with `prev_tokens`.
    fn record_cache_miss_on_error(&mut self) {
        self.engine.memory_clear();
        if let Some(cache) = self.prefix_cache.as_mut() {
            cache.clear();
        }
    }

    /// Record usage for the current call onto [`self.last_usage`]
    /// (overwrite) and [`self.total_usage`] (accumulate).
    fn record_usage(&mut self, usage: Usage) {
        self.last_usage = usage;
        self.total_usage += usage;
    }

    /// Debug escape hatch. Renders the prompt → tokenizes → runs the
    /// predictor → concatenates pieces into a `String`.
    ///
    /// # What this method is for
    ///
    /// Verifying the round-trip invariant: a [`response::Message`][rm]
    /// produced by [`Self::complete_response`] must re-render through
    /// [`ChatTemplate`] to exactly the bytes this method returns for
    /// the same `prompt`. That's the "complete* and complete_text are
    /// two views of the same bytes" contract.
    ///
    /// Beyond testing, prefer [`Self::complete_response`] (returns a
    /// full [`response::Message`][rm] with usage + stop reason) or
    /// [`Self::complete`] (returns a typed [`AssistantMessage`]).
    ///
    /// # Grammar
    ///
    /// Grammar is prepended per-call: if [`grammar_for_prompt`] returns
    /// `Some(grammar)`, the effective sampling chain is `[grammar,
    /// ...self.sample_modes.iter().cloned()]`. This happens automatically
    /// whenever `prompt.tool_choice` is `Some(Method | Any)` and the tool list
    /// is non-empty.
    ///
    /// # Prefix caching
    ///
    /// Participates in prefix-cache reuse when
    /// [`Self::with_prefix_cache`] is enabled — no opt-out. Callers
    /// that need bit-exact repeat output across calls should use
    /// greedy sampling, as today.
    ///
    /// [rm]: misanthropic::response::Message
    pub fn complete_text(
        &mut self,
        prompt: &Prompt,
    ) -> Result<String, SessionError> {
        let (tokens, breakpoints, modes, deferred_grammar) =
            self.prepare_call_cached(prompt, true)?;
        let prompt_tokens = tokens.len();

        let (suffix, l_hit) = self.kv_setup_for_call(&tokens, &breakpoints);

        let mut predict_opts =
            PredictOptions::default().add_model_stops(&self.engine.model);
        predict_opts.n = self.effective_max_tokens(prompt);
        predict_opts.sample_options = SampleOptions {
            modes,
            repetition: self.repetition.clone(),
            deferred_grammar: deferred_grammar.clone(),
        };

        // Count pieces as we consume them — one piece equals one
        // generated token before any post-hoc stop-string trimming
        // the predictor does.
        let mut generated_count: usize = 0;
        let mut text = String::new();
        let mut predictor = if l_hit > 0 {
            self.engine
                .predict_pieces_resuming(suffix, l_hit, 0, predict_opts)
        } else {
            self.engine.predict_pieces(suffix, predict_opts)
        };
        while let Some(piece) = predictor.next() {
            generated_count += 1;
            text.push_str(&piece);
        }
        // Drop the predictor so it releases the engine borrow — we
        // need `&self.engine` for `trim_eos` below.
        drop(predictor);

        let trimmed = trim_eos(&text, &self.engine).to_string();

        self.record_cache_hit(tokens, breakpoints, l_hit);
        let usage = Self::make_usage(prompt_tokens, l_hit, generated_count);
        self.record_usage(usage);

        Ok(trimmed)
    }

    /// Stream [`Block`]s as they're generated.
    ///
    /// Each iterator yield is one fully-resolved block. Prose is flushed as
    /// soon as enough bytes arrive to disambiguate it from a tag prefix;
    /// `<think>…</think>` and `<tool_call>…</tool_call>` are emitted when their
    /// closing tag arrives. Malformed JSON inside a well-framed tool_call falls
    /// back to a `Block::Text` (see [`BlockParser`] for the parser contract).
    ///
    /// The returned iterator borrows `self` — only one stream can be live at a
    /// time. Drop it before calling another `complete_*`.
    ///
    /// # Prefix caching
    ///
    /// Participates in prefix-cache reuse when enabled. Cache
    /// metadata (prev_tokens, prev_breakpoints, reused count) is
    /// updated **before** the predictor borrow — iterating or
    /// dropping the returned [`BlockStream`] does not mutate cache
    /// state. That's correct: `prev_tokens` describes *prompt* KV,
    /// and the next call's
    /// [`kv_setup_for_call`](Session::kv_setup_for_call) truncates any
    /// generation tokens that leaked past the reused prefix.
    ///
    /// Output-token count is not known until the stream is consumed,
    /// so [`Self::last_usage`]'s `output_tokens` is set to 0 for
    /// streaming calls. Input counts (`input_tokens`,
    /// `cache_read_input_tokens`) are accurate. Callers who need an
    /// output count should count pieces themselves or use a batch
    /// entry point.
    ///
    /// # Errors
    ///
    /// Iteration itself doesn't produce per-item errors; all setup failures
    /// (template render, grammar compile) surface as the outer `Err`.
    /// Grammar-violation checks live on the batch methods — streaming callers
    /// see whatever partial output the model produced.
    pub fn complete_stream<'s>(
        &'s mut self,
        prompt: &Prompt,
    ) -> Result<BlockStream<'s, B>, SessionError> {
        let (tokens, breakpoints, modes, deferred_grammar) =
            self.prepare_call_cached(prompt, true)?;
        let prompt_tokens = tokens.len();

        let (suffix, l_hit) = self.kv_setup_for_call(&tokens, &breakpoints);

        // Streaming: the cache must be updated BEFORE the predictor
        // borrows `&mut self.engine`, because the returned stream
        // holds that borrow for the lifetime of iteration. Usage
        // follows the same ordering — output count stays 0 because we
        // can't count pieces from here.
        self.record_cache_hit(tokens, breakpoints, l_hit);
        let usage = Self::make_usage(prompt_tokens, l_hit, 0);
        self.record_usage(usage);

        let mut eos_pieces: std::collections::BTreeSet<String> =
            std::collections::BTreeSet::new();
        eos_pieces.insert(
            self.engine.model.token_to_piece(self.engine.model.eos()),
        );
        let eot_id = self.engine.model.eot();
        if eot_id >= 0 {
            eos_pieces.insert(self.engine.model.token_to_piece(eot_id));
        }
        for extra in self.engine.model.extra_eos_tokens() {
            if extra >= 0 {
                eos_pieces.insert(self.engine.model.token_to_piece(extra));
            }
        }
        eos_pieces.remove("");

        let mut predict_opts =
            PredictOptions::default().add_model_stops(&self.engine.model);
        predict_opts.n = self.effective_max_tokens(prompt);
        predict_opts.sample_options = SampleOptions {
            modes,
            repetition: self.repetition.clone(),
            deferred_grammar: deferred_grammar.clone(),
        };

        let predictor = if l_hit > 0 {
            self.engine
                .predict_pieces_resuming(suffix, l_hit, 0, predict_opts)
        } else {
            self.engine.predict_pieces(suffix, predict_opts)
        };
        Ok(BlockStream {
            predictor,
            parser: BlockParser::new(),
            pending: std::collections::VecDeque::new(),
            eos_pieces,
            drained: false,
        })
    }

    /// Run a batch call end-to-end: cache setup, prediction, cache
    /// bookkeeping, usage accounting, stop-reason inference. The
    /// single source of truth for [`Self::complete_blocks`],
    /// [`Self::complete`], and [`Self::complete_response`].
    ///
    /// Returns a [`CallOutcome`] with everything a caller could
    /// reasonably need to build an API-shaped response. On error,
    /// invalidates the prefix cache AND the KV cache — partial
    /// decodes may have left them inconsistent.
    fn run_call(
        &mut self,
        prompt: &Prompt,
    ) -> Result<CallOutcome, SessionError> {
        use crate::ToolChoice;
        let forced_tool_call = matches!(
            prompt.tool_choice,
            Some(ToolChoice::Method { .. }) | Some(ToolChoice::Any)
        );

        let (tokens, breakpoints, modes, deferred_grammar) =
            self.prepare_call_cached(prompt, true)?;
        let prompt_tokens = tokens.len();

        let (suffix, l_hit) = self.kv_setup_for_call(&tokens, &breakpoints);

        // Pieces we drop from the surfaced output: the primary EOS,
        // the EOT (if distinct), every extra-EOS the model declares
        // (e.g. Qwen3's `<|endoftext|>`), and the invalid-UTF-8
        // sentinel. Pre-decode once so the inner loop is a hash
        // lookup. Empty pieces are kept out of the set — empty is
        // also what a stuck-on-secondary-EOS loop emits, but we'd
        // rather rely on the new `extra_eos_tokens` plumbing in
        // PredictOptions::add_model_stops to halt the loop cleanly
        // than silently swallow every empty piece.
        let mut eos_pieces: std::collections::BTreeSet<String> =
            std::collections::BTreeSet::new();
        eos_pieces.insert(
            self.engine.model.token_to_piece(self.engine.model.eos()),
        );
        let eot_id = self.engine.model.eot();
        if eot_id >= 0 {
            eos_pieces.insert(self.engine.model.token_to_piece(eot_id));
        }
        for extra in self.engine.model.extra_eos_tokens() {
            if extra >= 0 {
                eos_pieces.insert(self.engine.model.token_to_piece(extra));
            }
        }
        eos_pieces.remove("");

        let mut predict_opts =
            PredictOptions::default().add_model_stops(&self.engine.model);
        predict_opts.n = self.effective_max_tokens(prompt);
        predict_opts.sample_options = SampleOptions {
            modes,
            repetition: self.repetition.clone(),
            deferred_grammar: deferred_grammar.clone(),
        };

        // Collect generated pieces + count tokens inline. We also
        // track the concatenated raw-text buffer so stop-sequence
        // matching can inspect it post-hoc.
        let mut generated_count: usize = 0;
        let mut raw_text = String::new();
        let mut blocks: Vec<crate::Block> = Vec::new();
        let mut parser = BlockParser::new();

        let predictor = if l_hit > 0 {
            self.engine
                .predict_pieces_resuming(suffix, l_hit, 0, predict_opts)
        } else {
            self.engine.predict_pieces(suffix, predict_opts)
        };

        for piece in predictor {
            if eos_pieces.contains(&piece) || piece == "[Invalid UTF-8]" {
                continue;
            }
            generated_count += 1;
            raw_text.push_str(&piece);
            let emitted = parser.push(&piece);
            blocks.extend(emitted);
        }
        blocks.extend(parser.finish());
        // The parser emits one block per resolved prose chunk for
        // streaming friendliness. For batch callers (complete_blocks /
        // complete / complete_response), collapse adjacent same-kind
        // prose so `[Text, Text, Text]` becomes `[Text]` — semantically
        // identical, and lets the `FromIterator<Block>` path flatten
        // single-Text outputs into Content::SinglePart.
        let blocks = merge_adjacent_prose(blocks);

        // Cache + usage bookkeeping, then grammar-violation check.
        // Check last so a violation still records the work that was
        // done — usage numbers are correct either way.
        self.record_cache_hit(tokens, breakpoints, l_hit);
        let usage = Self::make_usage(prompt_tokens, l_hit, generated_count);
        self.record_usage(usage);

        if forced_tool_call
            && !blocks
                .iter()
                .any(|b| matches!(b, crate::Block::ToolUse { .. }))
        {
            let partial = blocks
                .iter()
                .filter_map(|b| match b {
                    crate::Block::Text { text, .. } => Some(text.as_ref()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("");
            // Grammar violation is a call failure — invalidate cache
            // + KV to avoid stale reuse next call.
            self.record_cache_miss_on_error();
            return Err(SessionError::GrammarViolation {
                partial_output: partial,
            });
        }

        let (stop_reason, stop_sequence) = infer_stop_reason(
            &blocks,
            &raw_text,
            generated_count,
            self.effective_max_tokens(prompt),
            prompt.stop_sequences.as_deref(),
        );

        // Diagnostic dump of the unparsed text + per-piece breakdown.
        // Off by default; enable with `RUST_LOG=drama_llama::session=debug`.
        // Useful when generation hits `max_tokens` with valid
        // grammar-shaped output but the post-grammar tail is opaque
        // (whitespace? content?). Escape into Debug form so newlines
        // and tabs are visible. Gated on the `axum` feature (which
        // pulls in tracing); the library doesn't otherwise depend on
        // it.
        #[cfg(feature = "axum")]
        {
            if tracing::enabled!(tracing::Level::DEBUG) {
                tracing::debug!(
                    event = "raw_generation",
                    generated_tokens = generated_count,
                    raw_text_bytes = raw_text.len(),
                    raw_text_debug = %format!("{:?}", raw_text),
                );
            }
        }

        // `raw_text` was consumed by stop-sequence inference; not
        // exported. Drop explicitly so the allocation is released
        // before the outcome is handed back to the caller.
        drop(raw_text);

        Ok(CallOutcome {
            blocks,
            prompt_tokens,
            cache_read_tokens: l_hit,
            generated_tokens: generated_count,
            stop_reason,
            stop_sequence,
        })
    }

    /// Batch variant of [`Self::complete_stream`]: collect every emitted block
    /// into a `Vec`, then run the grammar-violation check.
    ///
    /// # Errors
    ///
    /// Returns [`SessionError::GrammarViolation`] when the prompt's
    /// [`ToolChoice`] is `Method | Any` (grammar-forced) but the resulting
    /// block stream contains no [`Block::ToolUse`] — e.g. the model was
    /// truncated by `max_tokens` before closing the `</tool_call>` tag.
    ///
    /// [`ToolChoice`]: crate::ToolChoice
    pub fn complete_blocks(
        &mut self,
        prompt: &Prompt,
    ) -> Result<Vec<crate::Block>, SessionError> {
        Ok(self.run_call(prompt)?.blocks)
    }

    /// Greedy-driven diagnostic: render the prompt, decode it, then
    /// greedy-sample up to [`Session::with_max_tokens`] tokens, recording the
    /// **top-k candidates + their logits + decoded pieces** at every generated
    /// position.
    ///
    /// Grammar from the prompt's [`ToolChoice`] is applied each step exactly as
    /// production does, so the returned top-k is the same candidate set the
    /// real sampler would see. User [`SamplingMode`]s are deliberately **not**
    /// applied — they shape the final pick, not the candidate distribution we
    /// want to inspect. The committed token at each position is the argmax of
    /// the post-grammar candidates (i.e. what [`SamplingMode::Greedy`] would
    /// pick).
    ///
    /// Intended for diffing against external engines that expose logprobs (e.g.
    /// ollama's `/v1/chat/completions` with `logprobs: true, top_logprobs: N`)
    /// to localize wrong-argmax bugs to either our decode pipeline or upstream
    /// llama.cpp.
    ///
    /// # Prefix cache interaction
    ///
    /// Invalidates any prefix-cache state (calls
    /// [`Self::clear_prefix_cache`] internally) because the underlying
    /// [`LlamaCppEngine::predict_candidates`] path unconditionally clears the
    /// KV cache. Without this invalidation, a subsequent cached call
    /// would read stale `prev_tokens` metadata against a wiped KV.
    ///
    /// [`ToolChoice`]: crate::ToolChoice
    pub fn top_k_trace(
        &mut self,
        prompt: &Prompt,
        k: usize,
    ) -> Result<Vec<TokenTrace>, SessionError> {
        use crate::sample::grammar as grammar_mod;
        use crate::Sorted;

        self.clear_prefix_cache();
        // `top_k_trace` is diagnostic / offline — it iterates candidates
        // directly without going through the predictor, so there is no one
        // to drive deferred-grammar promotion. Drop the deferred grammar
        // on the floor (matches legacy behaviour of ignoring output_config
        // phase-split in this path).
        let (tokens, modes, _deferred) = self.prepare_call(prompt, false)?;

        let k_nz = NonZeroUsize::new(k.max(1)).unwrap();
        let eos = self.engine.model.eos();

        let mut predictor = self
            .engine
            .predict_candidates(tokens, self.effective_max_tokens(prompt));
        let mut trace: Vec<TokenTrace> = Vec::new();
        let mut position: usize = 0;

        while let Some(cands) = predictor.next() {
            let filtered = modes.iter().fold(cands, |c, mode| match mode {
                SamplingMode::Grammar(state) => {
                    let mut locked = state.lock().expect(
                        "SamplingMode::Grammar mutex poisoned in \
                         top_k_trace; matcher state unrecoverable.",
                    );
                    grammar_mod::grammar_filter(
                        c,
                        &mut locked,
                        &predictor.engine.model,
                    )
                }
                _ => c,
            });

            let sorted = filtered.sort(Sorted::ByLogit { k: k_nz });
            let top_k: Vec<TopKEntry> = sorted
                .iter()
                .map(|d| TopKEntry {
                    token: d.id,
                    logit: d.logit,
                    piece: predictor.engine.model.token_to_piece(d.id),
                })
                .collect();

            let chosen = match top_k.first() {
                Some(e) => e.token,
                None => break,
            };

            trace.push(TokenTrace { position, top_k });
            position += 1;

            if chosen == eos {
                break;
            }

            grammar_mod::advance_all(&modes, chosen, &predictor.engine.model);
            predictor.record_choice(chosen);
        }

        Ok(trace)
    }

    /// Batch variant returning a role-typed [`AssistantMessage`][am]. Routed
    /// through misanthropic's [`AssistantMessage: FromIterator<Block>`][am-fi]
    /// so single- block outputs flatten to [`Content::SinglePart`] and multi-
    /// block outputs stay [`Content::MultiPart`] — the crate-level convention,
    /// not one we reinvent here.
    ///
    /// Returning [`AssistantMessage`][am] rather than the bare [`Message`][m]
    /// is deliberate: it's statically impossible to paste a `Session::complete`
    /// return value in as a user turn. Need a bare [`Message`][m]?
    /// `assistant.into()` — the [`From`] impl is zero-cost.
    ///
    /// [am]: misanthropic::prompt::message::AssistantMessage
    /// [am-fi]: misanthropic::prompt::message::AssistantMessage
    /// [m]: crate::Message
    /// [`Content::SinglePart`]: crate::Content::SinglePart
    /// [`Content::MultiPart`]: crate::Content::MultiPart
    pub fn complete(
        &mut self,
        prompt: &Prompt,
    ) -> Result<crate::AssistantMessage, SessionError> {
        let blocks = self.complete_blocks(prompt)?;
        Ok(blocks.into_iter().collect())
    }

    /// Batch-complete returning a full
    /// [`response::Message`][rm] with content, usage, stop reason,
    /// and stop sequence populated.
    ///
    /// This is the shape downstream consumers (agent reactors,
    /// observability tooling, anything that mirrors the Anthropic
    /// Messages API response) want, so it gets a dedicated method
    /// rather than forcing callers to manually stitch together the
    /// outputs of [`Self::complete`] and [`Self::last_usage`].
    ///
    /// The existing [`Self::complete`] / [`Self::complete_blocks`] /
    /// [`Self::complete_text`] methods remain as shape-narrowed views
    /// of the same work; all four share [`Self::run_call`] under the
    /// hood.
    ///
    /// # Field filling
    ///
    /// * `id`: new UUID v4
    /// * `model`: [`model::Id::Custom`][cu] wrapping the result of
    ///   [`LlamaCppModel::desc`](crate::LlamaCppModel::desc).
    /// * `content`: [`AssistantMessage`] via
    ///   [`FromIterator<Block>`](std::iter::FromIterator).
    /// * `stop_reason`: inferred by [`infer_stop_reason`] — see its
    ///   docs for the mapping.
    /// * `stop_sequence`: the matched sequence when
    ///   `stop_reason == StopSequence`, else `None`.
    /// * `usage`: same shape as [`Self::last_usage`].
    ///
    /// [rm]: misanthropic::response::Message
    /// [cu]: misanthropic::model::Id::Custom
    pub fn complete_response(
        &mut self,
        prompt: &Prompt,
    ) -> Result<misanthropic::response::Message<'static>, SessionError> {
        let outcome = self.run_call(prompt)?;
        let inner: crate::AssistantMessage =
            outcome.blocks.into_iter().collect();
        let usage = Self::make_usage(
            outcome.prompt_tokens,
            outcome.cache_read_tokens,
            outcome.generated_tokens,
        );
        Ok(misanthropic::response::Message {
            id: std::borrow::Cow::Owned(uuid::Uuid::new_v4().to_string()),
            inner,
            model: self
                .engine
                .model
                .display_name()
                .unwrap_or_else(|| "unknown".to_string())
                .into(),
            stop_reason: outcome.stop_reason,
            stop_sequence: outcome.stop_sequence.map(std::borrow::Cow::Owned),
            usage,
        })
    }
}

/// Resolve the single grammar (if any) that should constrain
/// generation for `prompt`. Priority:
///
/// 1. `prompt.tool_choice` (when set and not `Auto`) — compiled via
///    [`grammar_for_prompt`] with `tool_choice_opts`. Always produces a
///    unified `Single` grammar (tool-choice has no thought preamble today).
/// 2. `prompt.output_config` — compiled via
///    [`output_config::compile_prompt_output_config`]; may return either a
///    `Single` unified grammar or a `Deferred` phase-split grammar
///    depending on `output_config_opts.phase_split`.
/// 3. `None` — generation is unconstrained.
///
/// Tool-choice wins when both are set: tool schemas *are* structured
/// output, and the model can only commit to one terminal shape per
/// turn. Lifted out of [`Session`] so the priority rule is testable
/// without instantiating an engine.
fn resolve_grammar(
    prompt: &Prompt,
    tool_choice_opts: &ToolChoiceOptions,
    output_config_opts: &OutputConfigOptions,
) -> Result<Option<crate::CompiledOutputConfig>, SessionError> {
    if let Some(g) = grammar_for_prompt(prompt, tool_choice_opts)? {
        return Ok(Some(crate::CompiledOutputConfig::Single(g)));
    }
    if let Some(c) = output_config::compile_prompt_output_config(
        prompt,
        output_config_opts,
    )? {
        return Ok(Some(c));
    }
    Ok(None)
}

/// Everything [`Session::run_call`] produces about one batch call —
/// shared by [`Session::complete_blocks`] / [`Session::complete`] /
/// [`Session::complete_response`] so each can project out the shape
/// it wants without duplicating the run itself.
struct CallOutcome {
    /// Parsed blocks from the completion.
    blocks: Vec<crate::Block>,
    /// Full prompt token length — input for the
    /// [`Usage`](misanthropic::response::Usage) `input_tokens` field.
    prompt_tokens: usize,
    /// Tokens reused from the prefix cache (0 on miss).
    cache_read_tokens: usize,
    /// Tokens emitted by the predictor (pre-trim, pre-stop-string
    /// truncation). This is the count the model actually generated.
    generated_tokens: usize,
    /// Inferred [`StopReason`](misanthropic::response::StopReason),
    /// or `None` if ambiguous.
    stop_reason: Option<misanthropic::response::StopReason>,
    /// The exact stop string that matched, if any. Populated only
    /// when `stop_reason == Some(StopSequence)`.
    stop_sequence: Option<String>,
}

/// Infer a [`StopReason`](misanthropic::response::StopReason) from a
/// completed batch call.
///
/// Priority (highest first):
///
/// 1. `ToolUse` — any [`Block::ToolUse`](crate::Block::ToolUse) in
///    the block stream. Anthropic-style: tool calls terminate the
///    assistant turn.
/// 2. `StopSequence` — `raw_text` ends with one of
///    `prompt.stop_sequences`. The matched sequence is returned as
///    the second tuple element.
/// 3. `MaxTokens` — `generated_tokens == max_tokens.get()`.
/// 4. `EndTurn` — the last block is a [`Block::Text`](crate::Block::Text)
///    (i.e. we successfully closed out on prose, not mid-tag).
/// 5. `None` — ambiguous; the caller can log or surface as `null` in
///    API wire output.
///
/// The check order prefers semantic signals (tool use, stop
/// sequence) over mechanical ones (token limit) so tool-call-forced
/// flows and caller-supplied stop strings are never mis-labeled as
/// `MaxTokens`.
/// Collapse runs of adjacent same-kind prose blocks. The streaming
/// [`BlockParser`] emits one [`Block::Text`] per resolved prose chunk
/// and one [`Block::Thought`] per tagged chunk for streaming
/// friendliness; batch callers want those coalesced before the
/// [`FromIterator<Block>`] flattening path decides
/// [`Content::SinglePart`] vs [`Content::MultiPart`].
///
/// Tool-use and tool-result blocks are discrete units and pass through
/// unchanged, as do any other non-prose variants.
///
/// [`Content::SinglePart`]: crate::Content::SinglePart
/// [`Content::MultiPart`]: crate::Content::MultiPart
fn merge_adjacent_prose(blocks: Vec<crate::Block>) -> Vec<crate::Block> {
    use crate::Block;
    use std::borrow::Cow;
    let mut out: Vec<Block> = Vec::with_capacity(blocks.len());
    for block in blocks {
        match (out.last_mut(), block) {
            (
                Some(Block::Text { text: prev, .. }),
                Block::Text { text: new, .. },
            ) => {
                *prev = Cow::Owned(format!("{prev}{new}"));
            }
            (
                Some(Block::Thought { thought: prev, .. }),
                Block::Thought { thought: new, .. },
            ) => {
                *prev = Cow::Owned(format!("{prev}{new}"));
            }
            (_, block) => out.push(block),
        }
    }
    out
}

fn infer_stop_reason(
    blocks: &[crate::Block],
    raw_text: &str,
    generated_tokens: usize,
    max_tokens: NonZeroUsize,
    stop_sequences: Option<&[std::borrow::Cow<'static, str>]>,
) -> (Option<misanthropic::response::StopReason>, Option<String>) {
    use misanthropic::response::StopReason;

    if blocks
        .iter()
        .any(|b| matches!(b, crate::Block::ToolUse { .. }))
    {
        return (Some(StopReason::ToolUse), None);
    }

    if let Some(stops) = stop_sequences {
        for s in stops {
            if !s.is_empty() && raw_text.ends_with(s.as_ref()) {
                return (Some(StopReason::StopSequence), Some(s.to_string()));
            }
        }
    }

    if generated_tokens >= max_tokens.get() {
        return (Some(StopReason::MaxTokens), None);
    }

    match blocks.last() {
        Some(crate::Block::Text { .. })
        | Some(crate::Block::Thought { .. }) => {
            (Some(StopReason::EndTurn), None)
        }
        _ => (None, None),
    }
}

/// Streaming [`Iterator`] over [`crate::Block`]s, produced by
/// [`Session::complete_stream`]. Yields each block as soon as its closing tag
/// (or tag-prefix ambiguity resolution) arrives.
///
/// Drops trailing EOS and `[Invalid UTF-8]` pieces the predictor emits at
/// stream end — those are artifacts of token-to-string conversion, not model
/// output.
pub struct BlockStream<'engine, B: Backend> {
    predictor: crate::PiecePredictor<'engine, B>,
    parser: BlockParser,
    pending: std::collections::VecDeque<crate::Block>,
    /// EOS-like piece texts (primary EOS, EOT, every
    /// `extra_eos_tokens` declared by the model) — filtered out of
    /// the stream since they're sentinels, not content the caller
    /// wants to see.
    eos_pieces: std::collections::BTreeSet<String>,
    drained: bool,
}

impl<'engine, B: Backend> Iterator for BlockStream<'engine, B> {
    type Item = crate::Block;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(block) = self.pending.pop_front() {
                return Some(block);
            }
            if self.drained {
                return None;
            }
            match self.predictor.next() {
                Some(piece) => {
                    // Skip the sentinel pieces — they aren't content.
                    // Everything else goes through the parser.
                    if self.eos_pieces.contains(&piece) || piece == "[Invalid UTF-8]" {
                        continue;
                    }
                    let blocks = self.parser.push(&piece);
                    self.pending.extend(blocks);
                }
                None => {
                    self.drained = true;
                    // Drain the parser — any trailing prose / partial
                    // tag contents become final blocks.
                    let final_blocks =
                        std::mem::take(&mut self.parser).finish();
                    self.pending.extend(final_blocks);
                }
            }
        }
    }
}

/// Strip trailing EOS piece and the `[Invalid UTF-8]` marker predictors emit
/// for byte-fallback tokens at stream end. Matches what
/// `examples/strawberry.rs` does by hand today.
fn trim_eos<'a, B: Backend>(text: &'a str, engine: &Engine<B>) -> &'a str {
    let eos_piece = engine.model.token_to_piece(engine.model.eos());
    text.trim_end_matches(eos_piece.as_str())
        .trim_end_matches("[Invalid UTF-8]")
        .trim_end()
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // Pure-Rust helper tests — no model, no KV, no #[ignore].
    // -----------------------------------------------------------------

    /// `longest_common_prefix_len` covers the edge shapes we rely on:
    /// empty inputs, identical inputs, one-token-different,
    /// one-shorter, and totally-disjoint. Token ids are arbitrary
    /// `i32`s — the function doesn't care about the vocab.
    #[test]
    fn test_longest_common_prefix_len() {
        assert_eq!(longest_common_prefix_len(&[], &[]), 0);
        assert_eq!(longest_common_prefix_len(&[1, 2, 3], &[]), 0);
        assert_eq!(longest_common_prefix_len(&[], &[1, 2, 3]), 0);
        assert_eq!(
            longest_common_prefix_len(&[1, 2, 3], &[1, 2, 3]),
            3,
            "identical",
        );
        assert_eq!(
            longest_common_prefix_len(&[1, 2, 3, 4], &[1, 2, 3, 9]),
            3,
            "one-different",
        );
        assert_eq!(
            longest_common_prefix_len(&[1, 2, 3], &[1, 2, 3, 4, 5]),
            3,
            "one-shorter",
        );
        assert_eq!(
            longest_common_prefix_len(&[1, 2, 3], &[9, 8, 7]),
            0,
            "disjoint",
        );
    }

    /// No breakpoints → no eligible reuse point → `L_hit == 0`,
    /// even when the common prefix is long.
    #[test]
    fn test_l_hit_computation_no_breakpoints() {
        let prev: Vec<Token> = (0..20).collect();
        let new_: Vec<Token> = (0..10).chain(100..110).collect();
        assert_eq!(longest_common_prefix_len(&prev, &new_), 10);
        assert_eq!(compute_l_hit(&prev, &new_, &[]), 0);
    }

    /// With breakpoints at [5, 8, 12] and a common prefix of 10, the
    /// BPE-safe cap is `10 - 1 = 9`. The largest breakpoint ≤ 9 is
    /// `8`, so `L_hit == 8`.
    #[test]
    fn test_l_hit_computation_with_breakpoint() {
        let prev: Vec<Token> = (0..20).collect();
        let new_: Vec<Token> = (0..10).chain(100..110).collect();
        let breakpoints = vec![5, 8, 12];
        assert_eq!(longest_common_prefix_len(&prev, &new_), 10);
        assert_eq!(compute_l_hit(&prev, &new_, &breakpoints), 8);
    }

    /// Common prefix of 5 with a breakpoint exactly at 5: BPE-safe cap
    /// is 4, nothing ≤ 4 is in the breakpoint list, so `L_hit == 0`.
    /// This guards against the one-token-boundary trap where
    /// resuming exactly at the prefix end is unsafe.
    #[test]
    fn test_l_hit_computation_bpe_backoff() {
        let prev: Vec<Token> = (0..10).collect();
        let new_: Vec<Token> =
            (0..5).chain(200..205).chain(300..305).collect();
        let breakpoints = vec![5];
        assert_eq!(longest_common_prefix_len(&prev, &new_), 5);
        assert_eq!(compute_l_hit(&prev, &new_, &breakpoints), 0);
    }

    /// When the common prefix is zero, `L_hit` must also be zero,
    /// regardless of breakpoint placement.
    #[test]
    fn test_l_hit_zero_common_prefix() {
        let prev = vec![10, 20, 30];
        let new_ = vec![40, 50, 60];
        let breakpoints = vec![1, 2, 3];
        assert_eq!(compute_l_hit(&prev, &new_, &breakpoints), 0);
    }

    /// Empty previous tokens — first call against a cold cache —
    /// always lands at `L_hit == 0`.
    #[test]
    fn test_l_hit_empty_prev() {
        let prev: Vec<Token> = Vec::new();
        let new_: Vec<Token> = vec![1, 2, 3, 4, 5];
        let breakpoints = vec![1, 3];
        assert_eq!(compute_l_hit(&prev, &new_, &breakpoints), 0);
    }

    /// No tool_choice and no output_config → no grammar constraint.
    #[test]
    fn test_resolve_grammar_none_when_neither_set() {
        let prompt = Prompt::default();
        let got = resolve_grammar(
            &prompt,
            &ToolChoiceOptions::default(),
            &OutputConfigOptions::default(),
        )
        .expect("resolve");
        assert!(got.is_none());
    }

    /// Only output_config is set → output-config grammar is used.
    /// Verify by sniffing the compiled GBNF source for the
    /// `output_schema` rule name the output_config builder emits.
    /// Default `OutputConfigOptions` has `phase_split=true`; since
    /// `compile_prompt_output_config` auto-disables phase_split when
    /// `prompt.thinking.is_none()`, the prompt here opts into
    /// thinking so the Deferred path is exercised.
    #[test]
    fn test_resolve_grammar_output_config_when_no_tool_choice() {
        use misanthropic::prompt::thinking::{Kind, Thinking};
        use std::num::NonZeroU32;
        let prompt = Prompt::default()
            .json_schema(serde_json::json!({
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            }))
            .thinking(Thinking {
                budget_tokens: NonZeroU32::new(1024).unwrap(),
                kind: Kind::Enabled,
            });
        let got = resolve_grammar(
            &prompt,
            &ToolChoiceOptions::default(),
            &OutputConfigOptions::default(),
        )
        .expect("resolve");
        let crate::CompiledOutputConfig::Deferred(deferred) =
            got.expect("some compiled config")
        else {
            panic!("expected Deferred variant (phase_split defaults on)");
        };
        assert_eq!(deferred.activate_after.as_slice(), b"</think>");
        let SamplingMode::Grammar(state) = deferred.grammar else {
            panic!("deferred.grammar must be SamplingMode::Grammar");
        };
        let source = state.lock().unwrap().grammar().source().to_string();
        assert!(
            source.contains("output_schema"),
            "expected output_config grammar, got: {source}"
        );
        // Phase-split emits JSON-only grammar; thought rules are
        // handled entirely at predictor level.
        assert!(
            !source.contains("think_body"),
            "phase-split grammar must not contain thought rules, got: \
             {source}"
        );
    }

    /// Opt out of `phase_split` — the unified thought+JSON grammar comes
    /// back under `Single`.
    #[test]
    fn test_resolve_grammar_output_config_single_when_phase_split_off() {
        let prompt = Prompt::default().json_schema(serde_json::json!({
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }));
        let got = resolve_grammar(
            &prompt,
            &ToolChoiceOptions::default(),
            &OutputConfigOptions {
                allow_thought: true,
                phase_split: false,
            },
        )
        .expect("resolve");
        let crate::CompiledOutputConfig::Single(SamplingMode::Grammar(
            state,
        )) = got.expect("some compiled config")
        else {
            panic!("expected Single(Grammar) variant");
        };
        let source = state.lock().unwrap().grammar().source().to_string();
        assert!(source.contains("output_schema"));
        assert!(source.contains("think_body"));
    }

    /// Both tool_choice and output_config set → tool_choice wins.
    /// Verify by sniffing for tool_choice's `name_choice` rule (which
    /// output_config never emits).
    #[test]
    fn test_resolve_grammar_tool_choice_wins_over_output_config() {
        use std::borrow::Cow;
        let tool = crate::Tool {
            name: Cow::Borrowed("foo"),
            description: Cow::Borrowed(""),
            schema: serde_json::json!({"type": "object"}),
            cache_control: None,
            strict: None,
        };
        let prompt = Prompt {
            functions: Some(vec![tool]),
            tool_choice: Some(crate::ToolChoice::Method {
                name: "foo".into(),
            }),
            ..Prompt::default()
        }
        .json_schema(serde_json::json!({
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }));
        let got = resolve_grammar(
            &prompt,
            &ToolChoiceOptions::default(),
            &OutputConfigOptions::default(),
        )
        .expect("resolve");
        let crate::CompiledOutputConfig::Single(SamplingMode::Grammar(
            state,
        )) = got.expect("some compiled config")
        else {
            panic!("expected Single(Grammar) variant for tool_choice");
        };
        let source = state.lock().unwrap().grammar().source().to_string();
        assert!(
            source.contains("name_choice"),
            "expected tool_choice grammar, got: {source}"
        );
        assert!(
            !source.contains("output_schema"),
            "tool_choice grammar must not leak output_config rules, got: {source}"
        );
    }

    /// `PrefixCache::new()` starts with every field zeroed, and
    /// `reset()` returns a populated cache to that state. This is the
    /// invariant `Session::clear_prefix_cache` relies on.
    #[test]
    fn test_prefix_cache_reset_zeroes_state() {
        let mut cache = PrefixCache::new();
        assert!(cache.prev_tokens.is_empty());
        assert!(cache.prev_breakpoints.is_empty());
        assert_eq!(cache.last_reused_tokens, 0);

        cache.prev_tokens = vec![1, 2, 3];
        cache.prev_breakpoints = vec![1, 2];
        cache.last_reused_tokens = 2;

        cache.clear();
        assert!(cache.prev_tokens.is_empty());
        assert!(cache.prev_breakpoints.is_empty());
        assert_eq!(cache.last_reused_tokens, 0);
    }

    /// Stop-reason inference: tool use wins over everything. When a
    /// `ToolUse` block is present, the stop reason must be `ToolUse`
    /// even if `generated_tokens == max_tokens` or a stop sequence
    /// technically matches — semantics beat bookkeeping.
    #[test]
    fn test_infer_stop_reason_tool_use_wins() {
        use misanthropic::response::StopReason;
        use misanthropic::tool::Use;
        let blocks = vec![
            crate::Block::Text {
                text: "ok".into(),
                cache_control: None,
            },
            crate::Block::ToolUse {
                call: Use {
                    id: "id".into(),
                    name: "t".into(),
                    input: serde_json::json!({}),
                    cache_control: None,
                },
            },
        ];
        let max = NonZeroUsize::new(8).unwrap();
        let (reason, seq) = infer_stop_reason(&blocks, "ok", 8, max, None);
        assert_eq!(reason, Some(StopReason::ToolUse));
        assert_eq!(seq, None);
    }

    /// Stop sequence matching — the matched string is returned as the
    /// tuple's second element and the reason is `StopSequence`.
    #[test]
    fn test_infer_stop_reason_stop_sequence() {
        use misanthropic::response::StopReason;
        let blocks = vec![crate::Block::Text {
            text: "hello STOP".into(),
            cache_control: None,
        }];
        let stops = vec![std::borrow::Cow::Borrowed("STOP")];
        let max = NonZeroUsize::new(128).unwrap();
        let (reason, seq) =
            infer_stop_reason(&blocks, "hello STOP", 3, max, Some(&stops));
        assert_eq!(reason, Some(StopReason::StopSequence));
        assert_eq!(seq.as_deref(), Some("STOP"));
    }

    /// Hitting `max_tokens` without a tool call and without a stop
    /// match reports `MaxTokens`.
    #[test]
    fn test_infer_stop_reason_max_tokens() {
        use misanthropic::response::StopReason;
        let blocks = vec![crate::Block::Text {
            text: "truncated".into(),
            cache_control: None,
        }];
        let max = NonZeroUsize::new(16).unwrap();
        let (reason, seq) =
            infer_stop_reason(&blocks, "truncated", 16, max, None);
        assert_eq!(reason, Some(StopReason::MaxTokens));
        assert_eq!(seq, None);
    }

    /// Clean text-block finish with room to spare → `EndTurn`.
    #[test]
    fn test_infer_stop_reason_end_turn() {
        use misanthropic::response::StopReason;
        let blocks = vec![crate::Block::Text {
            text: "done.".into(),
            cache_control: None,
        }];
        let max = NonZeroUsize::new(64).unwrap();
        let (reason, _) = infer_stop_reason(&blocks, "done.", 5, max, None);
        assert_eq!(reason, Some(StopReason::EndTurn));
    }

    /// Default [`Usage`] is the all-zero shape [`Session`] starts
    /// with. This guards against accidentally changing misanthropic's
    /// `Usage: Default` convention out from under us.
    #[test]
    fn test_usage_default_is_zero() {
        let u = Usage::default();
        assert_eq!(u.input_tokens, 0);
        assert_eq!(u.output_tokens, 0);
        assert_eq!(u.cache_creation_input_tokens, None);
        assert_eq!(u.cache_read_input_tokens, None);
    }

    /// `make_usage` is the function both batch + streaming paths use
    /// to stamp [`Usage`] values. It must always populate both cache
    /// counters (even at zero) so `Usage::AddAssign` accumulates
    /// them across calls instead of hitting the `None.or(Some(rhs))`
    /// first-value edge case.
    #[test]
    fn test_make_usage_populates_cache_counters() {
        let u = Session::<crate::LlamaCppBackend>::make_usage(100, 42, 10);
        assert_eq!(u.input_tokens, 100);
        assert_eq!(u.cache_read_input_tokens, Some(42));
        assert_eq!(u.cache_creation_input_tokens, Some(0));
        assert_eq!(u.output_tokens, 10);
    }

    // -----------------------------------------------------------------
    // Session builder tests — require a model to construct `Session`,
    // so they live behind #[ignore] like every other session-level
    // test in the crate.
    // -----------------------------------------------------------------

    fn model_path() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models/model.gguf")
    }

    #[test]
    #[ignore = "long running, requires models/model.gguf"]
    fn test_with_prefix_cache_default_off() {
        let session = Session::from_path(model_path()).unwrap().quiet();
        assert!(
            session.prefix_cache.is_none(),
            "default Session must have prefix cache disabled",
        );
        let on = session.with_prefix_cache(true);
        assert!(on.prefix_cache.is_some());
    }

    #[test]
    #[ignore = "long running, requires models/model.gguf"]
    fn test_last_and_total_usage_zero_initially() {
        let session = Session::from_path(model_path()).unwrap().quiet();
        assert_eq!(session.last_usage(), &Usage::default());
        assert_eq!(session.total_usage(), &Usage::default());
    }

    /// `RepetitionOptions::default()` includes `IgnoreCategory::Punctuation`
    /// so prose punctuation (`.`, `,`, etc.) is never penalized — penalty
    /// accumulating on `.` biases toward run-on sentences. After
    /// `Session::with_repetition(default)`, the category must still be
    /// in `ignored_categories` so the drain inside
    /// `apply_sample_repetition_ngram` materializes the punctuation
    /// tokens into `ignored` on first sample call.
    #[test]
    #[ignore = "long running, requires models/model.gguf"]
    fn test_default_repetition_ignores_punctuation_category() {
        let session = Session::from_path(model_path()).unwrap().quiet();
        let with_rep = session.with_repetition(RepetitionOptions::default());
        let rep = with_rep.repetition.as_ref().expect("repetition set");
        assert!(
            rep.ignored_categories()
                .contains(&crate::IgnoreCategory::Punctuation),
            "default must include Punctuation category, got {:?}",
            rep.ignored_categories(),
        );
    }

    /// `with_repetition` must plumb every special token (CONTROL +
    /// USER_DEFINED) into `opts.ignored` so a strong repetition
    /// penalty never suppresses chat-template or tool-call markers
    /// the model needs to close a turn. Regression guard for the
    /// bug where Session built `PredictOptions` *before* assigning
    /// repetition, so `add_model_stops`'s ignored-list injection
    /// silently no-op'd (and for the earlier EOS/EOT-only fix that
    /// missed modern chat templates).
    #[test]
    #[ignore = "long running, requires models/model.gguf"]
    fn test_with_repetition_adds_special_tokens_to_ignored() {
        let session = Session::from_path(model_path()).unwrap().quiet();
        let eos = session.engine.model.eos();
        let eot = session.engine.model.eot();
        let specials = session.engine.model.special_tokens();

        let with_rep = session.with_repetition(RepetitionOptions::default());
        let rep = with_rep.repetition.as_ref().expect("repetition set");
        let ignored = rep.ignored();

        assert!(
            ignored.contains(&crate::NGram::from(eos)),
            "EOS ({}) must be in ignored",
            eos,
        );
        if eot != eos && eot >= 0 {
            assert!(
                ignored.contains(&crate::NGram::from(eot)),
                "EOT ({}) must be in ignored when distinct",
                eot,
            );
        }
        for &t in &specials {
            assert!(
                ignored.contains(&crate::NGram::from(t)),
                "special token {} must be in ignored",
                t,
            );
        }
        // Modern chat-tuned models have several specials beyond EOS/EOT
        // (start_header, end_header, eot_id, eom_id, python_tag, ...).
        // Sanity check that the sweep isn't silently returning only a
        // couple — actual count varies by model.
        println!("special_tokens count = {}", specials.len());
    }

    #[test]
    #[ignore = "long running, requires models/model.gguf"]
    fn test_clear_prefix_cache_zeroes_state() {
        let mut session = Session::from_path(model_path())
            .unwrap()
            .quiet()
            .with_prefix_cache(true);
        // Force some "used" state so we know clear actually zeros.
        if let Some(cache) = session.prefix_cache.as_mut() {
            cache.prev_tokens = vec![1, 2, 3];
            cache.prev_breakpoints = vec![1, 2];
            cache.last_reused_tokens = 2;
        }
        session.clear_prefix_cache();
        let cache = session
            .prefix_cache
            .as_ref()
            .expect("clear does not drop the cache, only zeros it");
        assert!(cache.prev_tokens.is_empty());
        assert!(cache.prev_breakpoints.is_empty());
        assert_eq!(cache.last_reused_tokens, 0);
    }

    // -----------------------------------------------------------------
    // End-to-end prefix-cache integration tests. All `#[ignore]` —
    // require models/model.gguf and wall-clock time.
    // -----------------------------------------------------------------

    /// Build a [`Prompt`] with system + tools + one cached user
    /// message, producing at least one cache breakpoint.
    fn cached_prompt(user_msg: &'static str) -> Prompt {
        use misanthropic::prompt::message::{
            Block as MBlock, CacheControl, Content as MContent,
        };
        use std::borrow::Cow;
        let user_block = MBlock::Text {
            text: Cow::Borrowed(user_msg),
            cache_control: Some(CacheControl::Ephemeral { ttl: None }),
        };
        Prompt {
            system: Some(MContent::SinglePart(Cow::Borrowed(
                "You are a helpful assistant. Keep replies short.",
            ))),
            messages: vec![crate::Message {
                role: crate::Role::User,
                content: MContent::MultiPart(vec![user_block]),
            }],
            ..Prompt::default()
        }
    }

    /// Two back-to-back [`Session::complete_response`] calls on the
    /// exact same cached prompt must produce a cache hit on the
    /// second call (`usage.cache_read_input_tokens > 0`).
    #[test]
    #[ignore = "long running, requires models/model.gguf"]
    fn test_cache_hit_on_identical_prompts() {
        let mut session = Session::from_path(model_path())
            .unwrap()
            .quiet()
            .with_prefix_cache(true)
            .with_sampling(std::iter::empty());
        let prompt = cached_prompt("Pick a number 1-10.");

        let first = session.complete_response(&prompt).unwrap();
        assert_eq!(
            first.usage.cache_read_input_tokens,
            Some(0),
            "first call has nothing to read",
        );

        let second = session.complete_response(&prompt).unwrap();
        let read = second.usage.cache_read_input_tokens.unwrap_or(0);
        assert!(
            read > 0,
            "second identical call must hit the cache; got read={read}",
        );
    }

    /// Two prompts with identical system + tools but diverging last
    /// user messages: second call must reuse at least the
    /// system-boundary worth of tokens.
    #[test]
    #[ignore = "long running, requires models/model.gguf"]
    fn test_cache_hit_on_shared_system_diverging_last_message() {
        let mut session = Session::from_path(model_path())
            .unwrap()
            .quiet()
            .with_prefix_cache(true)
            .with_sampling(std::iter::empty());

        let first_prompt = cached_prompt("Say 'A'.");
        let second_prompt = cached_prompt("Say 'B'.");

        let _ = session.complete_response(&first_prompt).unwrap();
        let second = session.complete_response(&second_prompt).unwrap();
        let read = second.usage.cache_read_input_tokens.unwrap_or(0);
        assert!(
            read > 0,
            "shared-system call must reuse the system boundary; got {read}",
        );
    }

    /// Prompt with no `cache_control` markers: second call has
    /// nothing to reuse, so `cache_read_input_tokens == 0`.
    #[test]
    #[ignore = "long running, requires models/model.gguf"]
    fn test_cache_miss_no_breakpoints() {
        use misanthropic::prompt::message::Content as MContent;
        use std::borrow::Cow;
        let mut session = Session::from_path(model_path())
            .unwrap()
            .quiet()
            .with_prefix_cache(true)
            .with_sampling(std::iter::empty());
        let prompt = Prompt {
            system: Some(MContent::SinglePart(Cow::Borrowed(
                "You are a helpful assistant.",
            ))),
            messages: vec![crate::Message {
                role: crate::Role::User,
                content: MContent::SinglePart(Cow::Borrowed("Hello.")),
            }],
            ..Prompt::default()
        };

        let _ = session.complete_response(&prompt).unwrap();
        let second = session.complete_response(&prompt).unwrap();
        assert_eq!(
            second.usage.cache_read_input_tokens,
            Some(0),
            "no breakpoints = no reuse",
        );
    }

    /// [`Session::clear_prefix_cache`] must invalidate the cache so
    /// the next call misses even if the prompt is identical to the
    /// one that populated the cache.
    #[test]
    #[ignore = "long running, requires models/model.gguf"]
    fn test_clear_invalidates_cache() {
        let mut session = Session::from_path(model_path())
            .unwrap()
            .quiet()
            .with_prefix_cache(true)
            .with_sampling(std::iter::empty());
        let prompt = cached_prompt("Count to 3.");

        let _ = session.complete_response(&prompt).unwrap();
        session.clear_prefix_cache();
        let after = session.complete_response(&prompt).unwrap();
        assert_eq!(
            after.usage.cache_read_input_tokens,
            Some(0),
            "post-clear call must miss",
        );
    }
}
