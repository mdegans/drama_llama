//! Per-token probe-mode callback hook for `Engine`.
//!
//! Lets a downstream consumer (Agora reactor, Weave, batch evaluation,
//! canary suite) observe each yielded token without forking
//! [`crate::TokenPredictor`]'s iterator. Hook fires once per token
//! after sampling but before [`crate::CandidatePredictor::record_choice`]
//! advances the KV cache.
//!
//! ## Two appetites
//!
//! Hooks self-declare snapshot appetite via [`ProbeHook::snapshot_opts`]:
//!
//! - **Default `None`** — predictor skips the per-token softmax/sort
//!   entirely. Cheapest path. Used by lightweight recorders that only
//!   want `(token, n_cur)` and a wall-clock timestamp (e.g. blallama's
//!   `JsonlProbeRecorder`).
//! - **`Some(SnapshotOpts { ... })`** — predictor calls
//!   [`crate::Candidates::capture_snapshot`] on the pre-everything
//!   candidates and populates the rich fields on [`ProbeCtx`]
//!   ([`ProbeCtx::snapshot`], [`ProbeCtx::sampled_p`],
//!   [`ProbeCtx::sampled_rank`]). Used by the canary suite for
//!   cross-validation between external behavior and internal
//!   disposition.
//!
//! ## Threading
//!
//! Hooks are owned by the [`crate::Engine`] and called from whichever
//! thread is driving prediction. The trait requires [`Send`] so the
//! engine can be moved across threads but does not require [`Sync`] —
//! hooks aren't shared between concurrent engines (the Council runs
//! a separate engine per parallel agent, not a shared hook across
//! engines).

use crate::{SampleOptions, Snapshot, SnapshotOpts, Token};

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

    /// Self-declared snapshot appetite. Default `None` → predictor
    /// skips the per-token softmax/sort entirely. Override to `Some`
    /// to opt into rich capture.
    ///
    /// Probed once per `next()` call; cheap to call.
    fn snapshot_opts(&self) -> Option<SnapshotOpts> {
        None
    }
}

/// Per-token state passed to [`ProbeHook::on_token`].
///
/// Existing-field semantics ([`Self::token`], [`Self::n_cur`],
/// [`Self::sample_options`]) are preserved from earlier versions; the
/// rich fields ([`Self::snapshot`], [`Self::piece`],
/// [`Self::generation_index`]) are populated when the hook returned
/// `Some(SnapshotOpts)` from [`ProbeHook::snapshot_opts`].
///
/// To recover the pre-everything probability and rank of the sampled
/// token, call [`Snapshot::lookup_p`] / [`Snapshot::lookup_rank`] on
/// [`Self::snapshot`] with [`Self::token`].
#[non_exhaustive]
#[derive(Copy, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct ProbeCtx<'a> {
    /// The token that was just sampled (post-chain).
    pub token: Token,
    /// Position the token will land at on the next decode step. Equal
    /// to the index in the prefill+generation token sequence.
    pub n_cur: usize,
    /// Sampler chain configuration used to choose `token`. Includes
    /// repetition penalty, sampling modes (top-k / top-p / mirostat /
    /// grammar / etc.), and any deferred-grammar state.
    ///
    /// Skipped from serde — grammar `Arc<Mutex<...>>` doesn't serialize
    /// cleanly, and consumers who want a digest can pull it explicitly.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub sample_options: &'a SampleOptions,
    /// Pre-everything candidates snapshot. `None` when the hook's
    /// [`ProbeHook::snapshot_opts`] returned `None` — predictor skipped
    /// capture. `Some` when capture ran.
    pub snapshot: Option<&'a Snapshot>,
    /// Decoded piece for [`Self::token`]. Empty for tokens whose
    /// detokenization yields no visible bytes.
    pub piece: &'a str,
    /// 0-indexed position of this token in the generated sequence (not
    /// counting prefilled prompt tokens). Convenience for consumers
    /// building per-position records.
    pub generation_index: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send<T: Send>() {}

    #[test]
    fn probe_hook_object_is_send() {
        // `ProbeHook: Send` is load-bearing for engines that move
        // across threads. The default-impl `snapshot_opts` doesn't
        // change that, but a future trait-surface change could; this
        // smoke test catches the regression at compile time.
        assert_send::<Box<dyn ProbeHook>>();
    }
}
