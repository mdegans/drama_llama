//! Per-token probe-mode callback hook for `Engine`.
//!
//! Lets a downstream consumer (Agora reactor, Weave, batch
//! evaluation) observe each yielded token without forking
//! [`crate::TokenPredictor`]'s iterator. Hook fires once per token
//! after sampling but before [`crate::CandidatePredictor::record_choice`]
//! advances the KV cache.
//!
//! ## Schema rationale
//!
//! The fields here are the minimum set required to populate a
//! provider-trust baseline-record (see
//! `.claude/memory/provider_trust_discipline.md` for the methodology
//! the Council uses to compare drama_llama's outputs against external
//! provider baselines).
//!
//! ## Threading
//!
//! Hooks are owned by the [`crate::Engine`] and called from whichever
//! thread is driving prediction. The trait requires [`Send`] so the
//! engine can be moved across threads but does not require [`Sync`] —
//! hooks aren't shared between concurrent engines (the Council runs
//! a separate engine per parallel agent, not a shared hook across
//! engines).

use crate::{SampleOptions, Token};

/// Per-token observer for [`crate::TokenPredictor`].
///
/// Implementations record / forward the per-token state passed in via
/// [`ProbeCtx`]. The hook should be cheap — it runs synchronously on
/// the prediction loop's hot path.
pub trait ProbeHook: Send {
    /// Called once per yielded token. The token has been sampled and
    /// will be the next [`Iterator::Item`] returned by
    /// [`crate::TokenPredictor`]; the KV cache has not yet advanced
    /// past this token (`n_cur` is the position the token will land
    /// at on the next decode step).
    fn on_token(&mut self, ctx: ProbeCtx<'_>);
}

/// Per-token state passed to [`ProbeHook::on_token`].
#[non_exhaustive]
pub struct ProbeCtx<'a> {
    /// The token that was just sampled.
    pub token: Token,
    /// Position the token will land at on the next decode step. Equal
    /// to the index in the prefill+generation token sequence.
    pub n_cur: usize,
    /// Sampler chain configuration used to choose `token`. Includes
    /// repetition penalty, sampling modes (top-k / top-p / mirostat /
    /// grammar / etc.), and any deferred-grammar state.
    pub sample_options: &'a SampleOptions,
}
