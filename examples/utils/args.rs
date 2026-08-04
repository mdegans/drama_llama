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
//!   [`SessionTransport::new`](drama_llama::SessionTransport::new).
//! - **User preference** — a flag on [`CommonArgs`], one default for every
//!   example: `--model`, `--seed`, `--max-tokens`, `--system`, `--verbose`,
//!   plus the backend's own knobs via a flattened
//!   [`BackendArgs`](drama_llama::cli::BackendArgs) (`--backend`, `--n-ctx`,
//!   `--cache-slots`, `--use-2bit`).
//! - **Example requirement** — a [`TransportBuilder`] argument, set in code,
//!   deliberately *not* a flag. `strawberry` has to echo the digit from its
//!   tool result, so its `without_repetition` is load-bearing; exposing it as
//!   `--repetition` would let a user silently break the demo and get a wrong
//!   letter count out of an example whose whole job is a correct one.
//!
//! A knob the chosen backend has no notion of is an **error**, not a warning
//! — `--backend moeflux --n-ctx 4096` refuses to run rather than quietly
//! ignoring the context size. See
//! [`UnsupportedOptions`](drama_llama::cli::UnsupportedOptions).

use std::num::{NonZeroU128, NonZeroU32};
use std::path::PathBuf;

use clap::{Args as ClapArgs, Parser};
use drama_llama::cli::{BackendArgs, BackendKind};
use misanthropic::Prompt;

/// Flags shared by most examples. `#[command(flatten)]` into example args.
#[derive(ClapArgs, Debug, Clone)]
pub struct CommonArgs {
    /// Path to the model: a `.gguf` file for `--backend llama-cpp`, or the
    /// parent of `mlx/` + `artifacts/` + `root/` for `--backend moeflux`.
    #[arg(short, long, default_value_os_t = default_model_path())]
    pub model: PathBuf,

    /// Which backend, and how to load it.
    #[command(flatten)]
    pub load: BackendArgs,

    /// Override `max_tokens` (also applied to the session).
    #[arg(long)]
    pub max_tokens: Option<NonZeroU32>,

    /// RNG seed for new generation. Unset means random.
    #[arg(long)]
    pub seed: Option<NonZeroU128>,

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
    /// # Ok::<_, BuildError>(())
    /// ```
    #[cfg(feature = "tokio")]
    pub fn transport(&self) -> TransportBuilder<'_> {
        TransportBuilder {
            args: self,
            cache_slots: 1,
            repetition: true,
            render_opts: None,
            output_config_opts: None,
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
    output_config_opts: Option<drama_llama::OutputConfigOptions>,
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

    /// Override how `Prompt::output_config` compiles to a grammar. What
    /// `soul_forge` uses to reject the optional `<think>` preamble: a base
    /// model has no trained habit of closing one, so the default
    /// `allow_thought: true` limb is a trap rather than a feature there.
    pub fn output_config_opts(
        mut self,
        opts: drama_llama::OutputConfigOptions,
    ) -> Self {
        self.output_config_opts = Some(opts);
        self
    }

    /// Load the model and wrap it in the erased transport.
    ///
    /// This is the one place in the examples that names a concrete backend,
    /// and therefore the one place that can narrow
    /// [`BackendArgs`](drama_llama::cli::BackendArgs) to a concrete backend's
    /// options.
    pub fn build(
        self,
    ) -> Result<std::sync::Arc<dyn drama_llama::LocalTransport>, BuildError>
    {
        use drama_llama::{FromPath, SessionTransport};

        match self.args.load.backend {
            #[cfg(feature = "llama-cpp")]
            BackendKind::LlamaCpp => {
                let mut options =
                    drama_llama::LlamaCppOptions::try_from(&self.args.load)?;
                // `cache_slots` is an example *requirement* (swarm and
                // council want one slot per agent) rather than a user
                // preference, so it is applied here — but only where the user
                // expressed none, so an explicit `--cache-slots` still wins.
                // `n_ctx` needs no such handling: `BackendArgs` carries the
                // default eagerly.
                if self.cache_slots > 1 {
                    options.cache_slots.get_or_insert(self.cache_slots);
                }
                let session = drama_llama::LlamaCppSession::from_path_with(
                    self.args.model.clone(),
                    options,
                )?;
                Ok(std::sync::Arc::new(SessionTransport::new(
                    self.finish(session),
                )))
            }
            #[cfg(all(feature = "moeflux", target_os = "macos"))]
            BackendKind::Moeflux => {
                let options =
                    drama_llama::MoefluxOptions::try_from(&self.args.load)?;
                // Not an error, unlike a `--cache-slots` the user typed:
                // this one is the example's own request and the user never
                // asked for it. Still worth saying, because the consequence
                // is a re-prefill on every agent switch rather than a cache
                // hit — a large difference in a multi-agent example, and an
                // otherwise invisible one.
                if self.cache_slots > 1 {
                    log::info!(
                        "this example asked for {} KV cache slots; moeflux \
                         has one physical stream (its seq_id is a namespace \
                         label), so agent switches will re-prefill",
                        self.cache_slots,
                    );
                }
                let session = drama_llama::Session::<
                    drama_llama::MoefluxBackend,
                >::from_path_with(
                    self.args.model.clone(), options
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
        if let Some(opts) = self.output_config_opts.clone() {
            session = session.with_output_config_opts(opts);
        }
        // The `--max-tokens` override rides on the prompt via
        // [`CommonArgs::configure`], the sole generation cap since
        // `Session::with_max_tokens` was removed.
        session
    }
}

/// Failure building the transport: narrowing the flags to the chosen
/// backend, or loading the model itself.
///
/// Typed rather than boxed because the examples' `main`s box to different
/// things — `Box<dyn Error>` in some, `Box<dyn Error + Send + Sync>` in
/// others — and neither of those converts into the other, so any single
/// boxed error type would compile in half the call sites.
#[cfg(feature = "tokio")]
#[derive(Debug, thiserror::Error)]
pub enum BuildError {
    #[error(transparent)]
    Session(#[from] drama_llama::SessionError),
    /// A flag the chosen backend has no notion of. Fatal on purpose — see
    /// the module docs.
    #[error(transparent)]
    Options(#[from] drama_llama::cli::UnsupportedOptions),
}

/// [`BuildError`] plus the runtime a [`BlockingTransport`] has to stand up.
#[cfg(feature = "tokio")]
#[derive(Debug, thiserror::Error)]
pub enum BlockingTransportError {
    #[error(transparent)]
    Build(#[from] BuildError),
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
