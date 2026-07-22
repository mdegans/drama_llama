use std::path::PathBuf;

use clap::Parser;

use llama_cpp_sys_3::{
    llama_context_params, llama_model_default_params, llama_model_params,
};

#[derive(Debug, Parser)]
pub struct Args {
    /// Path to the model
    #[arg(short, long)]
    pub model: PathBuf,
    /// Context size
    #[arg(short, long, default_value_t = 1024)]
    pub context: u32,
    /// Disable on-by-default GPU acceleration
    #[arg(short, long, default_value_t = false)]
    pub no_gpu: bool,
}

impl Args {
    /// Create `llama_model_params` from `Args`. Defaults are used for fields
    /// not specified in `Args`.
    pub fn model_params(&self) -> llama_model_params {
        self.into()
    }

    /// Create `llama_context_params` from `Args`. Defaults are used for fields
    /// not specified in `Args`.
    pub fn context_params(&self) -> llama_context_params {
        self.into()
    }
}

impl From<&Args> for llama_model_params {
    fn from(args: &Args) -> Self {
        // Safety: This returns POD and makes no allocations for the pointer
        // fields, which are optional and initialized to null.
        let mut params = unsafe { llama_model_default_params() };
        params.n_gpu_layers = if args.no_gpu { 0 } else { 1000 };

        params
    }
}

impl From<&Args> for llama_context_params {
    fn from(args: &Args) -> Self {
        // Upgrades ggml's 4-thread library default to all cores.
        let mut params = crate::LlamaCppEngine::default_context_params();
        params.n_ctx = args.context;

        params
    }
}

/// Inference backend selector for a `--backend` flag. Variants are cfg-gated
/// to whichever crate features are enabled, so a single-backend build gets a
/// single-variant flag rather than one that can name a backend that isn't
/// there.
///
/// Shared by `bin/blallama` and the examples' `CommonArgs` so there is one
/// answer to "which backends does this build have" rather than one per
/// consumer.
///
/// Note the `cli` feature implies `llama-cpp` (see `Cargo.toml`), so
/// [`Self::LlamaCpp`] is always present wherever this type compiles today.
/// The cfg is kept anyway: it is what the enum *means*, and it is what makes
/// splitting `cli` into `clap`-only a one-line change rather than a hunt.
#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum BackendKind {
    #[cfg(feature = "llama-cpp")]
    LlamaCpp,
    #[cfg(all(feature = "moeflux", target_os = "macos"))]
    Moeflux,
}

/// Default `--backend` value: prefer llama-cpp when both backends are
/// compiled in (it has been blallama's default for its whole life, and it is
/// the backend the example models are packaged for).
pub const fn default_backend_kind() -> BackendKind {
    #[cfg(feature = "llama-cpp")]
    {
        BackendKind::LlamaCpp
    }
    #[cfg(all(
        all(feature = "moeflux", target_os = "macos"),
        not(feature = "llama-cpp"),
    ))]
    {
        BackendKind::Moeflux
    }
}
