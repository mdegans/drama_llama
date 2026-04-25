//! [`MoefluxEngine`] type alias + a `from_paths` constructor that
//! opens the decoder against the moeflux artifacts and the model
//! against the MLX export directory in one call.

use std::path::Path;

use thiserror::Error;

use crate::{
    moeflux::{
        decoder::{MoefluxDecoder, MoefluxError},
        model::{MoefluxModel, MoefluxModelError},
        MoefluxBackend,
    },
    Engine,
};

/// Convenience alias for the moeflux-backed pair. Use
/// `MoefluxEngine::from_paths(...)` when you want the moeflux backend
/// without spelling out `Engine<MoefluxBackend>`.
pub type MoefluxEngine = Engine<MoefluxBackend>;

/// Errors from [`MoefluxEngine::from_paths`]. Wraps both the
/// model-side (HF tokenizer / config) and decoder-side (mf_init_model)
/// failure modes.
#[derive(Debug, Error)]
pub enum MoefluxEngineError {
    /// Model-side failure (tokenizer or config parse).
    #[error(transparent)]
    Model(#[from] MoefluxModelError),
    /// Decoder-side failure (mf_init_model).
    #[error(transparent)]
    Decoder(#[from] MoefluxError),
}

impl MoefluxEngine {
    /// Open both halves of a moeflux engine.
    ///
    /// - `mlx_dir` — HF / MLX export directory containing
    ///   `tokenizer.json`, `config.json`, `tokenizer_config.json`,
    ///   and (optionally) `chat_template.jinja`. The
    ///   [`MoefluxModel`] reads from here.
    /// - `artifacts_dir` — moeflux artifacts directory containing
    ///   `model_weights.bin`, `model_weights.json`, `vocab.bin`.
    /// - `experts_dir` — directory containing `packed_experts/`.
    /// - `experts_per_tok` — MoE top-K at inference.
    /// - `use_2bit` — select the 2-bit packed-experts layout.
    pub fn from_paths(
        mlx_dir: &Path,
        artifacts_dir: &Path,
        experts_dir: &Path,
        experts_per_tok: u32,
        use_2bit: bool,
    ) -> Result<Self, MoefluxEngineError> {
        let model = MoefluxModel::from_mlx_dir(mlx_dir)?;
        let decoder = MoefluxDecoder::open(
            &artifacts_dir.join("model_weights.bin"),
            &artifacts_dir.join("model_weights.json"),
            &artifacts_dir.join("vocab.bin"),
            experts_dir,
            experts_per_tok,
            use_2bit,
        )?;
        Ok(Self { decoder, model })
    }

    /// Open a moeflux engine from a single parent directory using the
    /// drama_llama folder convention:
    ///
    /// - `parent/mlx/` — MLX export (tokenizer, config, optional
    ///   `chat_template.jinja`).
    /// - `parent/artifacts/` — moeflux artifacts directory (contains
    ///   `model_weights.bin`, `model_weights.json`, `vocab.bin`).
    /// - `parent/root/` — the experts directory (contains
    ///   `packed_experts/`).
    ///
    /// Defaults `experts_per_tok = 8`, `use_2bit = false` — the Qwen3
    /// MoE 4-bit setup. Power users who need explicit paths or
    /// non-default runtime params can use [`Self::from_paths`]
    /// directly.
    ///
    /// This is the constructor `blallama` uses on the moeflux side so
    /// `--model <path>` is symmetric with the llama.cpp path. The
    /// convention is forward-looking — current on-disk artifacts use
    /// flat sibling layout (`<stem>-mlx-4bit/`, `<stem>-artifacts/`,
    /// `<stem>-root/`); migration via the moeflux conversion script
    /// is tracked in `.claude/memory/`.
    pub fn from_path(
        parent: &Path,
    ) -> Result<Self, MoefluxEngineError> {
        let mlx_dir = parent.join("mlx");
        let artifacts_dir = parent.join("artifacts");
        let experts_dir = parent.join("root");
        let mut engine = Self::from_paths(
            &mlx_dir,
            &artifacts_dir,
            &experts_dir,
            8,
            false,
        )?;
        // Override the model's display name to the parent dir's
        // basename — that's what blallama-style discovery flows
        // address the model by, and what they expect to see echoed
        // back in the `/v1/messages` response. Without this the
        // name would be the MLX dir's basename (e.g.
        // `Qwen3.5-397B-A17B-4bit`), which doesn't match the
        // discovery-dir entry name (`qwen3-5-a17b`).
        if let Some(name) =
            parent.file_name().map(|s| s.to_string_lossy().into_owned())
        {
            engine.model.set_name(name);
        }
        Ok(engine)
    }
}
