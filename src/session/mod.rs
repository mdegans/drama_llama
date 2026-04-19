//! High-level ergonomic wrapper around [`Engine`] for chat-style tool-using
//! inference.
//!
//! [`Session`] is to local inference what [`misanthropic::Client::message`] is
//! to the Anthropic API: given a [`Prompt`], get back a [`Message`] (or, while
//! Phase 1 ships, raw bytes via [`Session::complete_text`]). The caller builds
//! their [`Prompt`] with misanthropic's normal builders and lets `Session`
//! handle rendering, grammar enforcement, sampling, and (in later phases) block
//! parsing.
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
//!
//! # What it doesn't (yet)
//!
//! * **Phase 2**: streaming block parser — `<think>`/`<tool_call>` scanning
//!   plus serde-typed tool-call JSON.
//! * **Phase 3**: [`Session::complete`] returning a
//!   [`misanthropic::prompt::Message`], [`Session::complete_blocks`] returning
//!   `Vec<Block>`, and a byte-for-byte round-trip invariant test: re-rendering
//!   `complete()`'s output through the template must reproduce
//!   `complete_text()`'s bytes exactly.
//!
//! [`misanthropic::Client::message`]:
//!     https://docs.rs/misanthropic/latest/misanthropic/struct.Client.html#method.message
//! [`ToolChoice`]: crate::ToolChoice

use std::{num::NonZeroUsize, path::PathBuf};

use crate::{
    engine::NewError, grammar_for_prompt, silence_logs, ChatTemplate,
    ChatTemplateError, Engine, PredictOptions, Prompt, RenderOptions,
    RepetitionOptions, SampleOptions, SamplingMode, ToolChoiceError,
    ToolChoiceOptions,
};

mod parse;
pub use parse::{parse_completion, BlockParser};

/// Errors from [`Session`].
#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    /// Model load / llama.cpp init failure.
    #[error("engine setup: {0}")]
    Engine(#[from] NewError),
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

/// Default maximum tokens per [`Session::complete_text`] call. Users
/// override via [`Session::with_max_tokens`].
const DEFAULT_MAX_TOKENS: usize = 1024;

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
    pub token: llama_cpp_sys_3::llama_token,
    /// Raw logit from the model (pre-softmax).
    pub logit: f32,
    /// Decoded string for this token (via `Model::token_to_piece`).
    pub piece: String,
}

/// Chat-style inference session: owns an [`Engine`] + [`ChatTemplate`]
/// plus the builder-configured defaults for each `complete_*` call.
//
// Cloning is cheap for everything except the [`Engine`] — which isn't
// [`Clone`] — so [`Session`] isn't either. Create one per model load;
// call `complete_*` many times.
//
// NOTE(mdegans): I'm actually fine using Arc for `engine`.
pub struct Session {
    engine: Engine,
    template: ChatTemplate,
    tool_choice_opts: ToolChoiceOptions,
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
}

impl Session {
    /// Load a model from disk and wire up the chat template.
    pub fn from_path(path: PathBuf) -> Result<Self, SessionError> {
        let engine = Engine::from_path(path)?;
        Self::from_engine(engine)
    }

    /// Load a model from disk with an explicit Flash Attention policy.
    ///
    /// Diagnostic escape hatch for output-divergence debugging — see
    /// [`FlashAttention`] for the when and why.
    pub fn from_path_with_flash_attention(
        path: PathBuf,
        fa: crate::FlashAttention,
    ) -> Result<Self, SessionError> {
        let engine = Engine::from_path_with_flash_attention(path, fa)?;
        Self::from_engine(engine)
    }

    /// Load a model CPU-only (zero GPU layers). Diagnostic path for
    /// isolating GPU-kernel divergence.
    pub fn from_path_cpu_only(path: PathBuf) -> Result<Self, SessionError> {
        let engine = Engine::from_path_cpu_only(path)?;
        Self::from_engine(engine)
    }

    /// Wrap an already-constructed [`Engine`]. Useful when the engine
    /// was built via [`Engine::new`] with custom context parameters.
    pub fn from_engine(engine: Engine) -> Result<Self, SessionError> {
        let template = ChatTemplate::from_model(&engine.model)?;
        Ok(Self {
            engine,
            template,
            tool_choice_opts: ToolChoiceOptions::default(),
            render_opts: RenderOptions::default().with_generation_prompt(true),
            sample_modes: vec![SamplingMode::locally_typical()],
            repetition: None,
            max_tokens: NonZeroUsize::new(DEFAULT_MAX_TOKENS).unwrap(),
        })
    }

    /// Enable (or replace) repetition penalty. Default is `None`
    /// because chat flows need the model to repeat natural tokens —
    /// in particular, digits that appeared in a tool result (the
    /// answer is often the same digit the tool just returned).
    ///
    /// Opt in for story generation, poetry, or anywhere loop-
    /// prevention matters. See [`RepetitionOptions`] for parameters.
    pub fn with_repetition(mut self, opts: RepetitionOptions) -> Self {
        self.repetition = Some(opts);
        self
    }

    /// Clear any repetition penalty — the explicit "no penalty"
    /// state, equivalent to the default.
    pub fn without_repetition(mut self) -> Self {
        self.repetition = None;
        self
    }

    /// Silence llama.cpp's log spew (model load progress, KV cache setup,
    /// compute buffer sizing, etc.). This is a process-global effect — calling
    /// it on any [`Session`] silences logs for every subsequent inference in
    /// the process.
    ///
    /// Consumer of `Session`. The [`restore_default_logs`] free function flips
    /// the flag back.
    ///
    /// [`restore_default_logs`]: crate::restore_default_logs
    pub fn quiet(self) -> Self {
        silence_logs();
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
    pub fn with_max_tokens(mut self, n: NonZeroUsize) -> Self {
        self.max_tokens = n;
        self
    }

    /// Borrow the underlying [`Engine`] — useful when the caller needs raw
    /// predictor access for something `Session` doesn't expose yet (e.g. custom
    /// stop-sequence management).
    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// Mutable borrow of the underlying [`Engine`]. Handy for KV-cache
    /// manipulation across turns.
    pub fn engine_mut(&mut self) -> &mut Engine {
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
        (Vec<llama_cpp_sys_3::llama_token>, Vec<SamplingMode>),
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
        // further shape the distribution.
        let grammar = grammar_for_prompt(prompt, &self.tool_choice_opts)?;
        let modes: Vec<SamplingMode> = if include_user_sampling {
            grammar
                .into_iter()
                .chain(self.sample_modes.iter().cloned())
                .collect()
        } else {
            grammar.into_iter().collect()
        };
        Ok((tokens, modes))
    }

    /// Debug escape hatch. Renders the prompt → tokenizes → runs the
    /// predictor → concatenates pieces into a `String`.
    ///
    /// # What this method is for
    ///
    /// Verifying the round-trip invariant: once [`Session::complete`] lands in
    /// Phase 3, a `Message` produced by `complete(&prompt)` must re-render
    /// through [`ChatTemplate`] to exactly the bytes this method returns for
    /// the same `prompt`. That's the "complete and complete_text are two views
    /// of the same bytes" contract.
    ///
    /// Beyond testing, prefer [`Session::complete`] (Phase 3) which returns a
    /// parsed [`AssistantMessage`] with typed blocks.
    ///
    /// # Grammar
    ///
    /// Grammar is prepended per-call: if [`grammar_for_prompt`] returns
    /// `Some(grammar)`, the effective sampling chain is `[grammar,
    /// ...self.sample_modes.iter().cloned()]`. This happens automatically
    /// whenever `prompt.tool_choice` is `Some(Method | Any)` and the tool list
    /// is non-empty.
    ///
    /// [`Message`]: crate::prompt::AssistantMessage
    pub fn complete_text(
        &mut self,
        prompt: &Prompt,
    ) -> Result<String, SessionError> {
        let (tokens, modes) = self.prepare_call(prompt, true)?;

        let mut predict_opts =
            PredictOptions::default().add_model_stops(&self.engine.model);
        predict_opts.n = self.max_tokens;
        predict_opts.sample_options = SampleOptions {
            modes,
            repetition: self.repetition.clone(),
        };

        let text: String =
            self.engine.predict_pieces(tokens, predict_opts).collect();

        Ok(trim_eos(&text, &self.engine).to_string())
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
    /// # Errors
    ///
    /// Iteration itself doesn't produce per-item errors; all setup failures
    /// (template render, grammar compile) surface as the outer `Err`.
    /// Grammar-violation checks live on the batch methods — streaming callers
    /// see whatever partial output the model produced.
    pub fn complete_stream<'s>(
        &'s mut self,
        prompt: &Prompt,
    ) -> Result<BlockStream<'s>, SessionError> {
        let (tokens, modes) = self.prepare_call(prompt, true)?;
        let eos_piece =
            self.engine.model.token_to_piece(self.engine.model.eos());

        let mut predict_opts =
            PredictOptions::default().add_model_stops(&self.engine.model);
        predict_opts.n = self.max_tokens;
        predict_opts.sample_options = SampleOptions {
            modes,
            repetition: self.repetition.clone(),
        };

        let predictor = self.engine.predict_pieces(tokens, predict_opts);
        Ok(BlockStream {
            predictor,
            parser: BlockParser::new(),
            pending: std::collections::VecDeque::new(),
            eos_piece,
            drained: false,
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
        use crate::ToolChoice;
        let forced_tool_call = matches!(
            prompt.tool_choice,
            Some(ToolChoice::Method { .. }) | Some(ToolChoice::Any)
        );
        let blocks: Vec<_> = self.complete_stream(prompt)?.collect();
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
            return Err(SessionError::GrammarViolation {
                partial_output: partial,
            });
        }
        Ok(blocks)
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
    /// [`ToolChoice`]: crate::ToolChoice
    pub fn top_k_trace(
        &mut self,
        prompt: &Prompt,
        k: usize,
    ) -> Result<Vec<TokenTrace>, SessionError> {
        use crate::sample::grammar as grammar_mod;
        use crate::Sorted;

        let (tokens, modes) = self.prepare_call(prompt, false)?;

        let k_nz = NonZeroUsize::new(k.max(1)).unwrap();
        let eos = self.engine.model.eos();

        let mut predictor =
            self.engine.predict_candidates(tokens, self.max_tokens);
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
            if predictor.record_choice(chosen).is_err() {
                break;
            }
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
}

/// Streaming [`Iterator`] over [`crate::Block`]s, produced by
/// [`Session::complete_stream`]. Yields each block as soon as its closing tag
/// (or tag-prefix ambiguity resolution) arrives.
///
/// Drops trailing EOS and `[Invalid UTF-8]` pieces the predictor emits at
/// stream end — those are artifacts of token-to-string conversion, not model
/// output.
pub struct BlockStream<'engine> {
    predictor: crate::PiecePredictor<'engine>,
    parser: BlockParser,
    pending: std::collections::VecDeque<crate::Block>,
    /// EOS piece text — we filter it out of the stream since it's a
    /// sentinel, not content the caller wants to see.
    eos_piece: String,
    drained: bool,
}

impl<'engine> Iterator for BlockStream<'engine> {
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
                    if piece == self.eos_piece || piece == "[Invalid UTF-8]" {
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
fn trim_eos<'a>(text: &'a str, engine: &Engine) -> &'a str {
    let eos_piece = engine.model.token_to_piece(engine.model.eos());
    text.trim_end_matches(eos_piece.as_str())
        .trim_end_matches("[Invalid UTF-8]")
        .trim_end()
}
