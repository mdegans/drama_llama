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
/// parameter. Use the `LlamaCppEngine` / `MoefluxEngine` type aliases
/// (feature-gated) for the common backends.
///
/// **Why the two live in one type:** a decoder's context borrows from
/// the model on backends that have one (llama.cpp's `llama_context`
/// keeps a `const llama_model &` for its whole life), so their
/// lifetimes have to coincide. Co-owning them here is what makes that
/// true by construction — it is the reason this type exists, not an
/// incidental convenience. That rationale was lost once and cost a
/// use-after-free reachable from safe code: the `model` field had
/// become `pub`, so `engine.model = other` dropped the weights in
/// place while the decoder still pointed at them (issue #54). Keep
/// both fields non-public, and if a future backend needs the model
/// reachable in some new way, add an accessor rather than reopening
/// the field.
///
/// Field declaration order (`vision`, then `decoder`, then `model`)
/// is the natural teardown order — vision context, then decoder
/// context (tearing the backend down if it was the last decoder),
/// then the model — but it is no longer load-bearing. It used to be:
/// a backend whose contexts borrow the model relied on this order and
/// nothing else, so a routine field reorder was a silent
/// use-after-free (issue #54). Backends now hold their own handle on
/// the model (llama.cpp: `LlamaCppModel` is a refcounted handle that
/// `LlamaCppDecoder` and `Mtmd` each clone), so the weights outlive
/// whatever refers to them regardless of the order here.
///
/// `Engine<B>` is `Send` whenever `B::Vision`, `B::Decoder` and
/// `B::Model` are — which they are by `Backend`'s associated-type
/// bounds. No manual unsafe impl needed; auto-derive does the right
/// thing.
pub struct Engine<B: Backend> {
    /// Optional vision (image-input) capability. `None` until loaded
    /// (llama.cpp: an mmproj sidecar or [`Engine::set_vision`]);
    /// statically always `None` for backends whose `B::Vision` is
    /// the uninhabited [`crate::NoVision`].
    pub(crate) vision: Option<B::Vision>,
    pub(crate) decoder: B::Decoder,
    /// The model. Read it through [`Engine::model`].
    ///
    /// Deliberately not public: replacing it would leave a tokenizer
    /// and chat template that disagree with a KV cache built from the
    /// *previous* model — silent garbage output rather than an error.
    /// (Before the model became a refcounted handle it was worse than
    /// that: assignment dropped the old model in place while the
    /// decoder's context still pointed at it.)
    pub(crate) model: B::Model,
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
            .field("vision", &self.vision.as_ref().map(|_| "B::Vision"))
            .field("decoder", &self.decoder)
            .field("model", &self.model)
            .field(
                "probe_hook",
                &self.probe_hook.as_ref().map(|_| "Box<dyn ProbeHook>"),
            )
            .finish()
    }
}

impl<B: Backend> Engine<B> {
    /// Install (or remove) a per-token probe-mode hook. The hook is
    /// invoked synchronously inside [`crate::TokenPredictor`]'s
    /// iterator after each token is sampled; see [`crate::ProbeHook`]
    /// for the contract. Pass `None` to clear an installed hook.
    pub fn set_probe_hook(&mut self, hook: Option<Box<dyn ProbeHook>>) {
        self.probe_hook = hook;
    }

    /// The model: tokenization, vocab introspection, chat template,
    /// metadata. See [`crate::Model`].
    pub fn model(&self) -> &B::Model {
        &self.model
    }

    /// Context length (tokens).
    pub fn n_ctx(&self) -> u32 {
        self.decoder.n_ctx()
    }

    /// Maximum number of distinct KV sequences the decoder supports.
    /// See [`Decoder::n_seq_max`].
    pub fn n_seq_max(&self) -> u32 {
        self.decoder.n_seq_max()
    }

    /// The vision capability, if one is loaded. `None` means images
    /// are unsupported on this engine (either the backend has no
    /// vision path at all, or no projector was loaded).
    pub fn vision(&self) -> Option<&B::Vision> {
        self.vision.as_ref()
    }

    /// Install (or remove) the vision capability, returning the
    /// previous one. Backend-specific constructors (e.g. the mmproj
    /// sidecar auto-load) normally handle this; the setter exists for
    /// arbitrary-path loads and tests.
    pub fn set_vision(
        &mut self,
        vision: Option<B::Vision>,
    ) -> Option<B::Vision> {
        std::mem::replace(&mut self.vision, vision)
    }

    /// Split borrow for the media prefill path:
    /// [`crate::Vision::prefill_image`] needs `&mut` both halves at
    /// once, which two separate accessors can't hand out.
    pub fn vision_and_decoder(
        &mut self,
    ) -> (Option<&mut B::Vision>, &mut B::Decoder) {
        (self.vision.as_mut(), &mut self.decoder)
    }

    /// Clear the KV cache.
    pub fn memory_clear(&mut self) {
        self.decoder.memory_clear()
    }

    /// Remove KV entries for `seq_id` in position range `[p0, p1)`.
    pub fn memory_seq_rm(&mut self, seq_id: i32, p0: i32, p1: i32) -> bool {
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

    /// Drop a single named snapshot at `(seq_id, pos)`. See
    /// [`Decoder::forget_pos`] — used by `Session`'s prefix-cache
    /// to release orphaned snapshots (replaced tips, no-longer-set
    /// breakpoints) without touching other state.
    pub fn forget_pos(
        &mut self,
        seq_id: i32,
        pos: i32,
    ) -> Result<(), MemoryRmError> {
        self.decoder.forget_pos(seq_id, pos)
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
    ///
    /// `initial_state`: `Some` resumes a caller-owned
    /// [`SamplerState`](crate::SamplerState) — the predictor skips
    /// state construction and prompt seeding, continuing the given
    /// stream. `None` builds a fresh state from `options` (seeded by
    /// `options.seed`, random when absent). Same on every `predict_*`
    /// below.
    pub fn predict_tokens<'a>(
        &'a mut self,
        tokens: Vec<Token>,
        options: PredictOptions,
        initial_state: Option<crate::SamplerState>,
    ) -> TokenPredictor<'a, B> {
        TokenPredictor::new(self, tokens, options, initial_state)
    }

    /// Iterator that predicts a sequence of pieces (strings).
    pub fn predict_pieces<'a>(
        &'a mut self,
        tokens: Vec<Token>,
        options: PredictOptions,
        initial_state: Option<crate::SamplerState>,
    ) -> PiecePredictor<'a, B> {
        PiecePredictor::new(self, tokens, options, initial_state)
    }

    /// Iterator that predicts both tokens and pieces.
    pub fn predict<'a>(
        &'a mut self,
        tokens: Vec<Token>,
        options: PredictOptions,
        initial_state: Option<crate::SamplerState>,
    ) -> Predictor<'a, B> {
        Predictor::new(self, tokens, options, initial_state)
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
        initial_state: Option<crate::SamplerState>,
    ) -> TokenPredictor<'a, B> {
        TokenPredictor::new_resuming(
            self,
            tokens,
            start_pos,
            seq_id,
            options,
            initial_state,
        )
    }

    /// Resume piece prediction from a pre-populated KV cache.
    pub fn predict_pieces_resuming<'a>(
        &'a mut self,
        tokens: Vec<Token>,
        start_pos: usize,
        seq_id: i32,
        options: PredictOptions,
        initial_state: Option<crate::SamplerState>,
    ) -> PiecePredictor<'a, B> {
        PiecePredictor::new_resuming(
            self,
            tokens,
            start_pos,
            seq_id,
            options,
            initial_state,
        )
    }
}
