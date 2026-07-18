//! Shared command-line flags for the examples.
//!
//! Composed with clap's `#[command(flatten)]`: an example derives its own
//! `Parser` and flattens [`CommonArgs`] (and [`ChatArgs`] for chat-loop
//! examples), adding only its own arguments. Examples that need nothing
//! beyond the common shape can parse [`Args`] directly.

use std::num::NonZeroU32;
use std::path::PathBuf;

use clap::{Args as ClapArgs, Parser};
use misanthropic::Prompt;

/// Flags shared by most examples. Flatten into an example's `Parser` and
/// apply with [`configure`](CommonArgs::configure).
#[derive(ClapArgs, Debug, Clone)]
pub struct CommonArgs {
    /// Path to a GGUF model. Defaults to the repo's `models/model.gguf`
    /// symlink.
    #[arg(short, long, default_value_os_t = default_model_path())]
    pub model: PathBuf,

    /// Override `max_tokens` (also applied to the session).
    #[arg(long)]
    pub max_tokens: Option<NonZeroU32>,

    /// Context length (KV cells) for the session. Examples with many
    /// cache slots override the default upward via `mut_arg` (see
    /// council) so `--help` always shows the real value.
    #[arg(long, default_value_t = 8192)]
    pub n_ctx: u32,

    /// Override the example's built-in system prompt.
    #[arg(long)]
    pub system: Option<String>,

    /// Verbose output (the example decides what that means).
    #[arg(long)]
    pub verbose: bool,
}

/// The repo-conventional model path: `models/model.gguf` next to
/// `Cargo.toml`, usually a symlink to a real GGUF.
pub fn default_model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf")
}

impl CommonArgs {
    /// Apply whichever of `--max-tokens` / `--system` the user set onto
    /// `prompt`, leaving the example's own defaults for the rest. (`--model`
    /// is consumed by [`session`](Self::session), not the prompt.)
    pub fn configure(&self, mut prompt: Prompt) -> Prompt {
        if let Some(max_tokens) = self.max_tokens {
            prompt = prompt.max_tokens(max_tokens);
        }
        if let Some(system) = &self.system {
            prompt = prompt.system(system.clone());
        }
        prompt
    }

    /// Load the session for `--model`, quiet, honoring `--max-tokens`.
    #[cfg(feature = "llama-cpp")]
    pub fn session(
        &self,
    ) -> Result<drama_llama::LlamaCppSession, drama_llama::SessionError> {
        let session = drama_llama::LlamaCppSession::from_path_with_n_ctx(
            self.model.clone(),
            self.n_ctx,
        )?;
        Ok(self.finish_session(session))
    }

    /// [`Self::session`], built for `slots` concurrent cached prefixes
    /// (multi-sequence unified KV — see
    /// `LlamaCppSession::from_path_with_cache_slots`). What multi-agent
    /// examples (swarm, council) want: one slot per agent, so agent
    /// switches stop thrashing the prefix cache.
    ///
    /// The prefix cache itself is switched **on** here: the library
    /// constructor only shapes the KV pool (its docs say "combine with
    /// `with_prefix_cache`"), and forgetting the combine step means
    /// every call cold-prefills while the payroll shows zero reads and
    /// the cache tripwire — which lives inside cache selection — can
    /// never fire. That was the council's zero-cache-reads bug.
    #[cfg(feature = "llama-cpp")]
    pub fn session_with_cache_slots(
        &self,
        slots: u32,
    ) -> Result<drama_llama::LlamaCppSession, drama_llama::SessionError> {
        let session = drama_llama::LlamaCppSession::from_path_with_cache_slots(
            self.model.clone(),
            self.n_ctx,
            slots,
        )?
        .with_prefix_cache(true);
        Ok(self.finish_session(session))
    }

    #[cfg(feature = "llama-cpp")]
    fn finish_session(
        &self,
        mut session: drama_llama::LlamaCppSession,
    ) -> drama_llama::LlamaCppSession {
        session = session.quiet();
        if let Some(max_tokens) = self.max_tokens {
            session = session.with_max_tokens(
                std::num::NonZeroUsize::new(max_tokens.get() as usize)
                    .expect("NonZeroU32 fits NonZeroUsize"),
            );
        }
        session
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
