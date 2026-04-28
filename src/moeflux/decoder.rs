//! moeflux-backed [`Decoder`]: wraps a single `moeflux::Ctx`, owns the
//! logit scratch buffer, and maps `mf_*` errors into a typed enum.

use std::path::Path;

use moeflux::{CheckpointError, Ctx, Error as MfError};
use thiserror::Error;

use crate::{
    backend::{Decoder, MemoryRmError},
    Token,
};

/// Errors from the moeflux decoder path.
#[derive(Debug, Error)]
pub enum MoefluxError {
    /// A path argument contained an interior NUL and could not be
    /// passed through the C FFI.
    #[error("path contained an interior NUL byte")]
    PathHasNul,
    /// `mf_init_model` returned NULL — missing files, mmap failure,
    /// Metal unavailable, or vocab parse failure. moeflux does not
    /// distinguish these; check stderr for the C-side diagnostic.
    #[error("moeflux: mf_init_model returned NULL")]
    InitFailed,
    /// `mf_eval_prompt` / `mf_eval_token` returned nonzero. State is
    /// undefined; call [`Decoder::memory_clear`] before continuing.
    #[error("moeflux: eval call returned nonzero")]
    EvalFailed,
    /// State snapshot save/load failed (shape mismatch, truncated
    /// buffer, or underlying FFI error).
    #[error("moeflux: state save/load failed")]
    StateFailed,
}

impl From<MfError> for MoefluxError {
    fn from(e: MfError) -> Self {
        match e {
            MfError::PathHasNul => Self::PathHasNul,
            MfError::InitFailed => Self::InitFailed,
            MfError::EvalFailed => Self::EvalFailed,
            MfError::StateFailed | MfError::StateBufferTooSmall { .. } => {
                Self::StateFailed
            }
        }
    }
}

/// moeflux-backed decoder. Owns a [`moeflux::Ctx`] and a scratch
/// `Vec<f32>` sized to the model's vocab — `mf_eval_*` writes into a
/// caller-supplied buffer, but [`Decoder::prefill`] and
/// [`Decoder::step`] return borrowed slices, so the buffer lives on
/// the struct.
///
/// Single-threaded per instance: `Ctx` is `Send` but not `Sync` (see
/// moeflux.h).
#[derive(Debug)]
pub struct MoefluxDecoder {
    ctx: Ctx,
    /// Logit scratch, sized to `ctx.n_vocab()` at construction and
    /// never resized. `Decoder::prefill` / `Decoder::step` return
    /// `&self.logits[..]`.
    logits: Vec<f32>,
}

impl MoefluxDecoder {
    /// Open a model and allocate the logit scratch buffer.
    ///
    /// Arguments mirror [`moeflux::Ctx::open`]:
    ///
    /// - `weights` — `model_weights.bin`.
    /// - `manifest` — `model_weights.json` (sibling).
    /// - `vocab` — `vocab.bin` (produced by moeflux's `export_vocab.py`).
    /// - `experts_dir` — directory containing `packed_experts/`.
    /// - `experts_per_tok` — MoE top-K at inference.
    /// - `use_2bit` — select `packed_experts_2bit/` layout.
    pub fn open(
        weights: &Path,
        manifest: &Path,
        vocab: &Path,
        experts_dir: &Path,
        experts_per_tok: u32,
        use_2bit: bool,
    ) -> Result<Self, MoefluxError> {
        let ctx = Ctx::open(
            weights,
            manifest,
            vocab,
            experts_dir,
            experts_per_tok,
            use_2bit,
        )?;
        let n_vocab = ctx.n_vocab();
        Ok(Self {
            ctx,
            logits: vec![0.0f32; n_vocab],
        })
    }

    /// Vocabulary size as reported by moeflux. Must equal the vocab
    /// size [`crate::moeflux::MoefluxModel`] was loaded for, or logit
    /// indices disagree between the two backends.
    pub fn n_vocab(&self) -> usize {
        self.ctx.n_vocab()
    }

    /// Canonical EOS token id baked into the compile-time variant
    /// (`EOS_TOKEN_1` in `model_variant.h`).
    pub fn eos_raw(&self) -> i32 {
        self.ctx.eos()
    }

    /// Display name of the compiled-in model variant.
    pub fn model_name(&self) -> &'static str {
        self.ctx.model_name()
    }

    /// Underlying moeflux context for callers that need state
    /// snapshot / restore (moeflux supports prefix-cache re-use via
    /// `state_save` / `state_load`; this crate does not yet expose
    /// those through the Decoder trait).
    pub fn ctx(&self) -> &Ctx {
        &self.ctx
    }

    /// Mutable access to the underlying moeflux context.
    pub fn ctx_mut(&mut self) -> &mut Ctx {
        &mut self.ctx
    }
}

impl Decoder for MoefluxDecoder {
    type Error = MoefluxError;

    fn prefill(
        &mut self,
        tokens: &[Token],
        start_pos: usize,
        seq_id: i32,
    ) -> Result<&[f32], Self::Error> {
        if tokens.is_empty() {
            return Ok(&[]);
        }
        self.ctx
            .eval_prompt(tokens, start_pos, seq_id, &mut self.logits)?;
        Ok(&self.logits[..])
    }

    fn step(
        &mut self,
        token: Token,
        pos: usize,
        seq_id: i32,
    ) -> Result<&[f32], Self::Error> {
        self.ctx.eval_token(token, pos, seq_id, &mut self.logits)?;
        Ok(&self.logits[..])
    }

    fn n_ctx(&self) -> u32 {
        // moeflux reports usize; the Decoder trait uses u32 to match
        // llama.cpp. Saturating-cast: realistic n_ctx values fit.
        self.ctx.n_ctx().try_into().unwrap_or(u32::MAX)
    }

    fn memory_clear(&mut self) {
        self.ctx.memory_clear();
    }

    fn memory_seq_rm(
        &mut self,
        seq_id: i32,
        p0: i32,
        p1: i32,
    ) -> bool {
        self.ctx.memory_seq_rm(seq_id, p0, p1)
    }

    fn memory_seq_cp(&mut self, _src: i32, _dst: i32, _p0: i32, _p1: i32) {
        // moeflux's C API does not expose cross-sequence copy; it only
        // maintains a single implicit sequence per ctx. Callers that
        // need KV-fork semantics should fall back to state_save /
        // state_load on a second Ctx. Silently no-op rather than
        // panicking: Session's prefix-cache path tolerates a backend
        // without sequence copy.
    }

    fn memory_seq_keep(&mut self, _seq_id: i32) {
        // Same reasoning as `memory_seq_cp` — moeflux has no notion of
        // multiple sequences to drop.
    }

    fn memory_seq_pos_max(&mut self, seq_id: i32) -> i32 {
        self.ctx.memory_seq_pos_max(seq_id)
    }

    /// Forwards to [`Ctx::checkpoint_pos`]. moeflux's recurrence is
    /// GPU-resident and folds full history; the snapshot reads it
    /// back via the existing `state_save` wire format. The trait
    /// surface is infallible — `state_save` errors only if linear
    /// buffers aren't initialized, which can't happen here because
    /// drama_llama's Session always prefills before checkpointing.
    /// We surface that contract violation as a panic.
    fn checkpoint_pos(&mut self, _seq_id: i32, pos: i32) {
        self.ctx
            .checkpoint_pos(pos)
            .expect("checkpoint_pos before first prefill (Session bug)");
    }

    fn restore_to(
        &mut self,
        _seq_id: i32,
        pos: i32,
    ) -> Result<(), MemoryRmError> {
        match self.ctx.restore_to(pos) {
            Ok(()) => Ok(()),
            Err(CheckpointError::NoCheckpoint { pos }) => {
                Err(MemoryRmError::NoCheckpoint { pos })
            }
            // A reload error from a buffer we wrote ourselves with
            // `state_save` is a moeflux-internal bug; surface as
            // BackendUnsupported so Session falls back gracefully
            // rather than crashing.
            Err(CheckpointError::Snapshot(_)) => {
                Err(MemoryRmError::BackendUnsupported { pos })
            }
        }
    }
}
