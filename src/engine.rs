//! Backend-agnostic [`Engine`]: pairs a [`Decoder`] with a [`Model`]
//! through a single [`Backend`] type parameter and routes predictor
//! construction.
//!
//! All llama.cpp-specific machinery (context creation, Flash-Attention
//! toggles, state ser/de, log callbacks) lives in `crate::llama_cpp`.
//! The `Engine<B>` type here only knows trait methods.

use crate::{
    backend::{Backend, Decoder, MemoryRmError},
    predictor::{CandidatePredictor, PiecePredictor, TokenPredictor},
    PredictOptions, Predictor, ProbeHook, Token,
};

use std::num::NonZeroUsize;

/// An `Engine` encompasses everything needed to run inferences. It
/// bundles a [`crate::Decoder`] (context + KV cache) with a
/// [`crate::Model`] (weights + tokenizer) via a single [`Backend`]
/// parameter. Use the [`crate::LlamaCppEngine`] /
/// [`crate::MoefluxEngine`] type aliases for the common backends.
///
/// Field declaration order (`decoder` before `model`) matters for
/// Drop: Rust drops fields in declaration order, so the decoder's
/// context is freed — and the backend teared down if it was the last
/// decoder — before the model is freed. This matches llama.cpp's
/// expected ordering.
///
/// `Engine<B>` is `Send` whenever `B::Decoder` and `B::Model` are —
/// which they are by `Backend`'s associated-type bounds. No manual
/// unsafe impl needed; auto-derive does the right thing.
pub struct Engine<B: Backend> {
    pub(crate) decoder: B::Decoder,
    /// The model. Public so callers (e.g. Session) can tokenize, look
    /// up special tokens, and render chat templates without going
    /// through Engine forwarding methods.
    pub model: B::Model,
    /// Optional per-token probe-mode hook. See [`crate::ProbeHook`]
    /// and [`Self::set_probe_hook`].
    pub(crate) probe_hook: Option<Box<dyn ProbeHook>>,
}

impl<B: Backend> std::fmt::Debug for Engine<B>
where
    B::Decoder: std::fmt::Debug,
    B::Model: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
            .field("decoder", &self.decoder)
            .field("model", &self.model)
            .field("probe_hook", &self.probe_hook.as_ref().map(|_| "Box<dyn ProbeHook>"))
            .finish()
    }
}

impl<B: Backend> Engine<B> {
    /// Install (or remove) a per-token probe-mode hook. The hook is
    /// invoked synchronously inside [`crate::TokenPredictor`]'s
    /// iterator after each token is sampled; see [`crate::ProbeHook`]
    /// for the contract. Pass `None` to clear an installed hook.
    pub fn set_probe_hook(
        &mut self,
        hook: Option<Box<dyn ProbeHook>>,
    ) {
        self.probe_hook = hook;
    }

    /// Context length (tokens).
    pub fn n_ctx(&self) -> u32 {
self.decoder.n_ctx()
    }

    /// Clear the KV cache.
    pub fn memory_clear(&mut self) {
self.decoder.memory_clear()
    }

    /// Remove KV entries for `seq_id` in position range `[p0, p1)`.
    pub fn memory_seq_rm(
        &mut self,
        seq_id: i32,
        p0: i32,
        p1: i32,
    ) -> bool {
self.decoder.memory_seq_rm(seq_id, p0, p1)
    }

    /// Copy KV entries between sequences in `[p0, p1)`.
    pub fn memory_seq_cp(&mut self, src: i32, dst: i32, p0: i32, p1: i32) {
self.decoder.memory_seq_cp(src, dst, p0, p1)
    }

    /// Keep only `seq_id`'s entries, drop all others.
    pub fn memory_seq_keep(&mut self, seq_id: i32) {
self.decoder.memory_seq_keep(seq_id)
    }

    /// Largest position present in KV for `seq_id`.
    pub fn memory_seq_pos_max(&mut self, seq_id: i32) -> i32 {
self.decoder.memory_seq_pos_max(seq_id)
    }

    /// Snapshot decoder state at sequence position `pos`. See
    /// [`Decoder::checkpoint_pos`] — backends like moeflux capture
    /// recurrent state for later lossless rewind; backends with
    /// per-cell preserved state (llama.cpp) no-op.
    pub fn checkpoint_pos(&mut self, seq_id: i32, pos: i32) {
        self.decoder.checkpoint_pos(seq_id, pos);
    }

    /// Rewind decoder state to a previously-snapshotted position.
    /// See [`Decoder::restore_to`] — `Err(NoCheckpoint)` signals the
    /// caller should fall back to `memory_clear` + full re-prefill.
    pub fn restore_to(
        &mut self,
        seq_id: i32,
        pos: i32,
    ) -> Result<(), MemoryRmError> {
        self.decoder.restore_to(seq_id, pos)
    }

    /// Prefill `tokens` at positions `[start_pos, start_pos +
    /// tokens.len())` on `seq_id`. Does not clear the KV cache.
    /// Thin forward over [`Decoder::prefill`] — used by the
    /// chunked-prefill path in `Session` so each cache-breakpoint
    /// chunk can be flushed before its [`Engine::checkpoint_pos`]
    /// call. Returns `Ok(())` on a non-empty slice; an empty slice
    /// is a no-op.
    pub fn prefill_chunk(
        &mut self,
        tokens: &[Token],
        start_pos: usize,
        seq_id: i32,
    ) -> Result<(), <B::Decoder as Decoder>::Error> {
        if tokens.is_empty() {
            return Ok(());
        }
        self.decoder.prefill(tokens, start_pos, seq_id)?;
        Ok(())
    }

    /// Iterator that yields [`crate::Candidates`] until `n` tokens
    /// have been produced or the end of context is reached. KV cache
    /// is cleared before starting.
    pub fn predict_candidates<'a>(
        &'a mut self,
        tokens: Vec<Token>,
        n: NonZeroUsize,
    ) -> CandidatePredictor<'a, B> {
        CandidatePredictor::new(self, tokens, n)
    }

    /// Iterator that predicts a sequence of tokens.
    pub fn predict_tokens<'a>(
        &'a mut self,
        tokens: Vec<Token>,
        options: PredictOptions,
    ) -> TokenPredictor<'a, B> {
        TokenPredictor::new(self, tokens, options)
    }

    /// Iterator that predicts a sequence of pieces (strings).
    pub fn predict_pieces<'a>(
        &'a mut self,
        tokens: Vec<Token>,
        options: PredictOptions,
    ) -> PiecePredictor<'a, B> {
        PiecePredictor::new(self, tokens, options)
    }

    /// Iterator that predicts both tokens and pieces.
    pub fn predict<'a>(
        &'a mut self,
        tokens: Vec<Token>,
        options: PredictOptions,
    ) -> Predictor<'a, B> {
        Predictor::new(self, tokens, options)
    }

    /// Resume candidate prediction from a KV cache the caller has
    /// already populated for positions `[0, start_pos)` on `seq_id`.
    /// The Predictor internally prefills `tokens` at those positions
    /// and begins sampling from the last prefilled position.
    pub fn predict_candidates_resuming<'a>(
        &'a mut self,
        tokens: Vec<Token>,
        start_pos: usize,
        seq_id: i32,
        n: NonZeroUsize,
    ) -> CandidatePredictor<'a, B> {
        CandidatePredictor::new_resuming(self, tokens, start_pos, seq_id, n)
    }

    /// Resume token prediction from a pre-populated KV cache.
    pub fn predict_tokens_resuming<'a>(
        &'a mut self,
        tokens: Vec<Token>,
        start_pos: usize,
        seq_id: i32,
        options: PredictOptions,
    ) -> TokenPredictor<'a, B> {
        TokenPredictor::new_resuming(
            self, tokens, start_pos, seq_id, options,
        )
    }

    /// Resume piece prediction from a pre-populated KV cache.
    pub fn predict_pieces_resuming<'a>(
        &'a mut self,
        tokens: Vec<Token>,
        start_pos: usize,
        seq_id: i32,
        options: PredictOptions,
    ) -> PiecePredictor<'a, B> {
        PiecePredictor::new_resuming(
            self, tokens, start_pos, seq_id, options,
        )
    }
}
