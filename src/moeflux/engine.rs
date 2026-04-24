//! [`MoefluxEngine`] type alias + a `from_paths` constructor that
//! opens the decoder against the moeflux artifacts and the model
//! against the MLX export directory in one call.

use std::path::Path;

use thiserror::Error;

use crate::{
    moeflux::{
        decoder::{MoefluxDecoder, MoefluxError},
        model::{MoefluxModel, MoefluxModelError},
    },
    Engine,
};

/// Convenience alias for the moeflux-backed pair. Use
/// `MoefluxEngine::from_paths(...)` when you want the moeflux backend
/// without spelling out `Engine<MoefluxDecoder, MoefluxModel>`.
pub type MoefluxEngine = Engine<MoefluxDecoder, MoefluxModel>;

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
}
