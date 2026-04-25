//! Backend-agnostic [`Engine`]: pairs a [`Decoder`] with a [`Model`]
//! through a single [`Backend`] type parameter and routes predictor
//! construction.
//!
//! All llama.cpp-specific machinery (context creation, Flash-Attention
//! toggles, state ser/de, log callbacks) lives in `crate::llama_cpp`.
//! The `Engine<B>` type here only knows trait methods.

use crate::{
    backend::{Backend, Decoder},
    predictor::{CandidatePredictor, PiecePredictor, TokenPredictor},
    PredictOptions, Predictor, Token,
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
#[derive(Debug)]
pub struct Engine<B: Backend> {
    pub(crate) decoder: B::Decoder,
    /// The model. Public so callers (e.g. Session) can tokenize, look
    /// up special tokens, and render chat templates without going
    /// through Engine forwarding methods.
    pub model: B::Model,
}

impl<B: Backend> Engine<B> {
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
