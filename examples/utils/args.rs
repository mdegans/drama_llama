//! Shared command-line flags for the examples.
//!
//! Examples `#[command(flatten)]`: [`CommonArgs`] into their own.
//!
//! # Why the examples name no backend
//!
//! [`CommonArgs::transport`] hands back an
//! [`Arc<dyn LocalTransport>`](drama_llama::LocalTransport) — a
//! [`Transport`](misanthropic::Transport) with its backend erased. The single
//! `match` that names `LlamaCppBackend` or `MoefluxBackend` lives in
//! [`TransportBuilder::build`] and nowhere else, so an example body is
//! backend-agnostic without carrying a generic parameter. `--backend` picks
//! at runtime among whatever this build compiled in.
//!
//! # Flags vs. builder arguments
//!
//! Settings split three ways here, and the split is deliberate:
//!
//! - **Universal** — nothing to configure. The prefix cache is switched on by
//!   [`SessionTransport::new`](drama_llama::SessionTransport::new); llama.cpp
//!   log spew is silenced at construction.
//! - **User preference** — a flag on [`CommonArgs`], one default for every
//!   example: `--model`, `--backend`, `--n-ctx`, `--seed`, `--max-tokens`,
//!   `--system`, `--verbose`.
//! - **Example requirement** — a [`TransportBuilder`] argument, set in code,
//!   deliberately *not* a flag. `strawberry` has to echo the digit from its
//!   tool result, so its `without_repetition` is load-bearing; exposing it as
//!   `--repetition` would let a user silently break the demo and get a wrong
//!   letter count out of an example whose whole job is a correct one.

use std::num::{NonZeroU128, NonZeroU32};
use std::path::PathBuf;

use clap::{Args as ClapArgs, Parser};
use drama_llama::cli::{default_backend_kind, BackendKind};
use misanthropic::Prompt;

/// Default `--n-ctx`. Named so the moeflux arm can tell "the user asked for a
/// context size" from "the user said nothing" — an `Option` would do the same
/// job but would drop the default out of `--help`, where it is worth seeing.
const DEFAULT_N_CTX: u32 = 8192;

/// Flags shared by most examples. `#[command(flatten)]` into example args.
#[derive(ClapArgs, Debug, Clone)]
pub struct CommonArgs {
    /// Path to the model: a `.gguf` file for `--backend llama-cpp`, or the
    /// parent of `mlx/` + `artifacts/` + `root/` for `--backend moeflux`.
    #[arg(short, long, default_value_os_t = default_model_path())]
    pub model: PathBuf,

    /// Inference backend. Choices are whichever backends this build compiled
    /// in, so a single-backend build offers exactly one.
    #[arg(long, value_enum, default_value_t = default_backend_kind())]
    pub backend: BackendKind,

    /// Override `max_tokens` (also applied to the session).
    #[arg(long)]
    pub max_tokens: Option<NonZeroU32>,

    /// RNG seed for new generation. Unset means random.
    pub seed: Option<NonZeroU128>,

    /// Max context length. llama-cpp only — moeflux takes its context length
    /// from the model export.
    #[arg(long, default_value_t = DEFAULT_N_CTX)]
    pub n_ctx: u32,

    /// Override the example's built-in system prompt.
    #[arg(long)]
    pub system: Option<String>,

    /// Verbose output (the example decides what that means).
    #[arg(long)]
    pub verbose: bool,
}

/// `{CARGO_MANIFEST_DIR}/models/model.gguf`
pub fn default_model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf")
}

impl CommonArgs {
    /// Apply whichever of `--max-tokens` / `--system` the user set onto
    /// `prompt`, leaving the example's own defaults for the rest. (`--model`
    /// is consumed by [`Self::transport`], not the prompt.)
    pub fn configure(&self, mut prompt: Prompt) -> Prompt {
        if let Some(max_tokens) = self.max_tokens {
            prompt = prompt.max_tokens(max_tokens);
        }
        if let Some(system) = &self.system {
            prompt = prompt.system(system.clone());
        }
        prompt
    }

    /// Start building this example's transport. See [`TransportBuilder`] for
    /// the knobs; `.build()` (async examples) or `.build_blocking()` (sync
    /// ones) finishes it.
    ///
    /// ```no_run
    /// # let common: utils::CommonArgs = todo!();
    /// let transport = common.transport().cache_slots(5).build()?;
    /// # Ok::<_, drama_llama::SessionError>(())
    /// ```
    #[cfg(feature = "tokio")]
    pub fn transport(&self) -> TransportBuilder<'_> {
        TransportBuilder {
            args: self,
            cache_slots: 1,
            repetition: true,
            render_opts: None,
        }
    }
}

/// Builds the example's erased transport. Everything settable here is an
/// *example requirement* rather than a user preference — see the module docs
/// for why that distinction decides flag vs. argument.
#[cfg(feature = "tokio")]
pub struct TransportBuilder<'a> {
    args: &'a CommonArgs,
    cache_slots: u32,
    repetition: bool,
    render_opts: Option<drama_llama::RenderOptions>,
}

#[cfg(feature = "tokio")]
impl TransportBuilder<'_> {
    /// Size the KV pool for `slots` concurrent cached prefixes — one per
    /// agent, so agent switches reuse instead of re-prefilling. What the
    /// multi-agent examples (swarm, council) want.
    pub fn cache_slots(mut self, slots: u32) -> Self {
        self.cache_slots = slots.max(1);
        self
    }

    /// Turn the repetition penalty off. For flows that must re-emit a short
    /// context token verbatim — a digit echoed back from a tool result — where
    /// the penalty can talk the model out of the right answer.
    pub fn without_repetition(mut self) -> Self {
        self.repetition = false;
        self
    }

    /// Override the chat-template render options (generation prompt, template
    /// `extra`s such as `enable_thinking`).
    pub fn render_opts(mut self, opts: drama_llama::RenderOptions) -> Self {
        self.render_opts = Some(opts);
        self
    }

    /// Load the model and wrap it in the erased transport.
    ///
    /// This is the one place in the examples that names a concrete backend.
    pub fn build(
        self,
    ) -> Result<
        std::sync::Arc<dyn drama_llama::LocalTransport>,
        drama_llama::SessionError,
    > {
        use drama_llama::SessionTransport;

        match self.args.backend {
            #[cfg(feature = "llama-cpp")]
            BackendKind::LlamaCpp => {
                let session = if self.cache_slots > 1 {
                    drama_llama::LlamaCppSession::from_path_with_cache_slots(
                        self.args.model.clone(),
                        self.args.n_ctx,
                        self.cache_slots,
                    )?
                } else {
                    // Deliberately *not* `..._with_cache_slots(.., 1)`: that
                    // constructor also flips `kv_unified`, so it is not the
                    // identity at one slot.
                    drama_llama::LlamaCppSession::from_path_with_n_ctx(
                        self.args.model.clone(),
                        self.args.n_ctx,
                    )?
                };
                // `quiet` is llama.cpp-specific (it silences that library's
                // log spew, process-globally), hence in this arm rather than
                // the shared finisher.
                Ok(std::sync::Arc::new(SessionTransport::new(
                    self.finish(session.quiet()),
                )))
            }
            #[cfg(all(feature = "moeflux", target_os = "macos"))]
            BackendKind::Moeflux => {
                self.warn_llama_cpp_only_knobs();
                let session = drama_llama::Session::<
                    drama_llama::MoefluxBackend,
                >::from_path_sync(
                    self.args.model.clone()
                )?;
                Ok(std::sync::Arc::new(SessionTransport::new(
                    self.finish(session),
                )))
            }
        }
    }

    /// [`Self::build`] wrapped for examples whose `main` is synchronous.
    pub fn build_blocking(
        self,
    ) -> Result<BlockingTransport, BlockingTransportError> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        // The load itself is sync, but `SessionTransport::new` is not tied to
        // a runtime, so this needs no `block_on`.
        let inner = self.build()?;
        Ok(BlockingTransport { runtime, inner })
    }

    /// The backend-agnostic half of session setup.
    fn finish<B: drama_llama::Backend>(
        &self,
        session: drama_llama::Session<B>,
    ) -> drama_llama::Session<B> {
        let mut session = session.with_seed(self.args.seed);
        if !self.repetition {
            session = session.without_repetition();
        }
        if let Some(opts) = self.render_opts.clone() {
            session = session.with_render_opts(opts);
        }
        // The `--max-tokens` override rides on the prompt via
        // [`CommonArgs::configure`], the sole generation cap since
        // `Session::with_max_tokens` was removed.
        session
    }

    /// Say so rather than silently dropping knobs the chosen backend has no
    /// notion of. `Session<MoefluxBackend>` has one constructor and no KV
    /// slot concept, so both of these are genuinely inapplicable — but a user
    /// who typed `--n-ctx` deserves to know it did nothing.
    #[cfg(all(feature = "moeflux", target_os = "macos"))]
    fn warn_llama_cpp_only_knobs(&self) {
        if self.args.n_ctx != DEFAULT_N_CTX {
            log::warn!(
                "--n-ctx {} ignored: the moeflux backend takes its context \
                 length from the model export",
                self.args.n_ctx,
            );
        }
        if self.cache_slots > 1 {
            log::warn!(
                "{} cache slots requested but ignored: moeflux has one \
                 physical stream (its seq_id is a namespace label), so \
                 per-agent KV slots do not exist",
                self.cache_slots,
            );
        }
    }
}

/// Failure building a [`BlockingTransport`]: loading the model, or standing
/// up the runtime it needs.
///
/// Typed rather than boxed because the examples' `main`s box to different
/// things — `Box<dyn Error>` in some, `Box<dyn Error + Send + Sync>` in
/// others — and neither of those converts into the other, so any single
/// boxed error type would compile in half the call sites.
#[cfg(feature = "tokio")]
#[derive(Debug, thiserror::Error)]
pub enum BlockingTransportError {
    #[error(transparent)]
    Session(#[from] drama_llama::SessionError),
    #[error("could not start the tokio runtime: {0}")]
    Runtime(#[from] std::io::Error),
}

/// A [`LocalTransport`](drama_llama::LocalTransport) driven from synchronous
/// code — what the examples that never grew an async `main` use.
///
/// The runtime is not optional ceremony: `SessionTransport` completes on
/// tokio's blocking pool, so *some* runtime has to be present whether or not
/// the example wants async. Owning a current-thread one here keeps that fact
/// in one place instead of sprinkling `block_on` through the call sites.
#[cfg(feature = "tokio")]
pub struct BlockingTransport {
    runtime: tokio::runtime::Runtime,
    inner: std::sync::Arc<dyn drama_llama::LocalTransport>,
}

#[cfg(feature = "tokio")]
impl BlockingTransport {
    /// One completion, start to finish.
    pub fn send(
        &self,
        prompt: &Prompt,
    ) -> Result<misanthropic::response::Message, drama_llama::SessionError>
    {
        use misanthropic::Transport;
        self.runtime
            .block_on(Transport::<Prompt>::send(&*self.inner, prompt))
    }

    /// Scan `text` for content that would tokenize to a reserved chat-framing
    /// special, returning the first offender's `(id, piece)`. Use it on
    /// anything relayed between agents or typed by a human: framing bytes in
    /// content are rejected at ingest, and catching them here turns a fatal
    /// error into a rephrase.
    pub fn scan_text_for_specials(
        &self,
        text: &str,
    ) -> Option<(drama_llama::Token, String)> {
        self.runtime
            .block_on(self.inner.scan_text_for_specials(text))
    }

    /// The erased transport itself, for handing to something async.
    pub fn transport(&self) -> std::sync::Arc<dyn drama_llama::LocalTransport> {
        self.inner.clone()
    }
}

/// Extra flag for chat-loop examples: the consecutive-tool-call cap.
#[derive(ClapArgs, Debug, Clone)]
pub struct ChatArgs {
    /// Cap consecutive tool-call rounds within one user beat (default: 8).
    #[arg(long)]
    pub max_tool_calls: Option<usize>,
}

impl ChatArgs {
    /// Apply `--max-tool-calls` onto a [`Chat`](super::Chat) if set, else
    /// leave the driver's default.
    pub fn configure<S, T: misanthropic::Transport>(
        &self,
        chat: super::Chat<S, T>,
    ) -> super::Chat<S, T> {
        match self.max_tool_calls {
            Some(max) => chat.max_consecutive_tool_calls(max),
            None => chat,
        }
    }
}

/// The common chat-example shape — [`CommonArgs`] + [`ChatArgs`] + a prompt.
/// Examples needing nothing more can `Args::parse()` directly.
#[derive(Parser, Debug)]
pub struct Args {
    #[command(flatten)]
    pub common: CommonArgs,

    #[command(flatten)]
    pub chat: ChatArgs,

    /// The initial user prompt / question.
    #[arg(short, long)]
    pub prompt: Option<String>,
}
