//! moeflux-backed [`Decoder`]: wraps a single `moeflux::Ctx`, owns the
//! logit scratch buffer, and maps `mf_*` errors into a typed enum.

use std::path::Path;

use moeflux::{Ctx, Error as MfError};
use thiserror::Error;

use crate::{
    backend::{Decoder, MemoryRmError},
    snapshot_store::SnapshotStore,
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
    /// The moeflux library was built for a different model than the
    /// weights it was pointed at (`moeflux-model-*` feature
    /// mismatch). The string carries moeflux's descriptive
    /// diagnostic, including the rebuild remedy. Surfaced cleanly at
    /// open time rather than as an opaque prefill panic.
    #[error("{0}")]
    ModelMismatch(String),
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
            MfError::ModelMismatch { expected, detail } => Self::ModelMismatch(
                format!("moeflux: binary built for {expected} — {detail}"),
            ),
        }
    }
}

/// Per-phase prefetch counters. The underlying moeflux `PrefetchState`
/// counter is a single (hits, misses) tuple covering both prefill and
/// decode; we split here by snapshotting the moeflux counter before
/// and after each [`Decoder::prefill`] / [`Decoder::step`] call so
/// blallama can log the steady-state decode hit rate separately from
/// prefill (which often has very different token-to-token expert
/// stability).
#[derive(Copy, Clone, Debug, Default)]
pub struct PrefetchStats {
    pub prefill_hits: u64,
    pub prefill_misses: u64,
    pub decode_hits: u64,
    pub decode_misses: u64,
}

/// moeflux-backed decoder. Owns a [`moeflux::Ctx`] and a scratch
/// `Vec<f32>` sized to the model's vocab — `mf_eval_*` writes into a
/// caller-supplied buffer, but [`Decoder::prefill`] and
/// [`Decoder::step`] return borrowed slices, so the buffer lives on
/// the struct.
///
/// Single-threaded per instance: `Ctx` is `Send` but not `Sync` (see
/// moeflux.h).
///
/// # Sequences are a namespace, not a partition
///
/// moeflux physically maintains **one** stream — `Ctx` ignores
/// `seq_id`. This wrapper realizes the [`Decoder`] trait's `(seq_id,
/// pos)` keying on top of it: [`Decoder::checkpoint_pos`] serializes
/// the whole `Ctx` state (`state_save`) into a wrapper-owned
/// [`SnapshotStore`] keyed `(seq_id, pos)`, and
/// [`Decoder::restore_to`] loads the blob back (`state_load`) — the
/// blob *is* the entire history, so loading it is the sequence
/// switch. [`Self::active_seq`] tracks which sequence the live
/// stream currently embodies; see the per-method docs for how calls
/// targeting an *inactive* sequence behave. Cost note: switching
/// sequences is a whole-state GPU upload — far cheaper than a
/// re-prefill, but not free like llama.cpp's resident-cell switch.
#[derive(Debug)]
pub struct MoefluxDecoder {
    ctx: Ctx,
    /// Logit scratch, sized to `ctx.n_vocab()` at construction and
    /// never resized. `Decoder::prefill` / `Decoder::step` return
    /// `&self.logits[..]`.
    logits: Vec<f32>,
    /// Per-phase prefetch counters. See [`PrefetchStats`].
    prefetch_stats: PrefetchStats,
    /// Wrapper-owned `(seq_id, pos)` → whole-`Ctx`-state blobs.
    /// Replaces `Ctx`'s internal position-keyed checkpoint map, which
    /// cannot distinguish sequences. Host-RAM budget = blob count
    /// (LRU-capped in [`SnapshotStore`]) × `state_size` — the *dual
    /// budget*: `Session`'s cell budget bounds bookkeeping, this cap
    /// bounds actual memory here.
    snapshots: SnapshotStore,
    /// The sequence the live stream currently embodies. `-1` = fresh /
    /// unowned (no eval since construction or [`Decoder::memory_clear`]).
    ///
    /// **The guard this enables is load-bearing**: mutating calls
    /// targeting a different sequence must not touch the live stream —
    /// most critically [`Decoder::memory_seq_rm`], which `Session`'s
    /// multi-slot cache calls to evict *inactive* slots. Without the
    /// guard, evicting slot B while slot A is generating would wipe
    /// A's state mid-flight.
    active_seq: i32,
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
    /// - `use_2bit` — select `packed_experts_2bit/` layout.
    ///
    /// MoE top-K is not a parameter: it is model shape, fixed by the
    /// compiled moeflux variant (`num_experts_per_tok`).
    pub fn open(
        weights: &Path,
        manifest: &Path,
        vocab: &Path,
        experts_dir: &Path,
        use_2bit: bool,
    ) -> Result<Self, MoefluxError> {
        let ctx = Ctx::open(weights, manifest, vocab, experts_dir, use_2bit)?;
        let n_vocab = ctx.n_vocab();
        Ok(Self {
            ctx,
            logits: vec![0.0f32; n_vocab],
            prefetch_stats: PrefetchStats::default(),
            snapshots: SnapshotStore::default(),
            active_seq: -1,
        })
    }

    /// Reset the live stream for a switch to `seq_id`, if it currently
    /// embodies a *different* sequence. The wrapper's snapshot store
    /// is untouched — it's what makes the departed sequence
    /// recoverable.
    fn switch_stream_to(&mut self, seq_id: i32) {
        if self.active_seq != seq_id {
            if self.active_seq != -1 {
                self.ctx.memory_clear();
            }
            self.active_seq = seq_id;
        }
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

    /// Underlying moeflux context. The [`Decoder`] trait's snapshot
    /// surface (`checkpoint_pos` / `restore_to` / `forget_pos`) is
    /// realized wrapper-side over `state_save` / `state_load` — direct
    /// `Ctx` access here is for diagnostics and tests, not state
    /// management (mixing `Ctx::checkpoint_pos` with the wrapper's
    /// store would split the snapshot namespace).
    pub fn ctx(&self) -> &Ctx {
        &self.ctx
    }

    /// Mutable access to the underlying moeflux context.
    pub fn ctx_mut(&mut self) -> &mut Ctx {
        &mut self.ctx
    }

    /// Per-phase prefetch counters accumulated since the last
    /// [`Self::reset_prefetch_stats`].
    pub fn prefetch_stats(&self) -> PrefetchStats {
        self.prefetch_stats
    }

    /// Zero the per-phase counters AND the underlying moeflux counter
    /// so the snapshot deltas in [`Decoder::prefill`] / [`Decoder::step`]
    /// start from zero.
    pub fn reset_prefetch_stats(&mut self) {
        self.prefetch_stats = PrefetchStats::default();
        self.ctx.reset_prefetch_stats();
    }

    /// Zero the moeflux per-label cmdbuf timing stats — call before a
    /// measured prefill so the breakdown is scoped to it.
    pub fn reset_cmdbuf_stats(&self) {
        self.ctx.reset_cmdbuf_stats();
    }

    /// Log the moeflux per-label cmdbuf timing breakdown, rows sorted
    /// by total CPU wait descending (dominant work first). Most useful
    /// under `MOEFLUX_PROFILE_PER_OP`, where each [`Op`] commits as its
    /// own labeled cmdbuf so the labels are per-Op. A no-op when no
    /// labeled commit has run.
    pub fn log_cmdbuf_stats(&self) {
        let mut stats = self.ctx.cmdbuf_stats();
        if stats.is_empty() {
            return;
        }
        stats.sort_by(|a, b| b.1.cpu_wait_ns.cmp(&a.1.cpu_wait_ns));
        let total_ns: u64 = stats.iter().map(|(_, s)| s.cpu_wait_ns).sum();
        tracing::info!(
            event = "moeflux_cmdbuf_stats",
            labels = stats.len(),
            total_ms = total_ns as f64 / 1e6,
        );
        for (label, s) in &stats {
            tracing::info!(
                event = "moeflux_cmdbuf_op",
                label,
                count = s.count,
                wait_ms = s.cpu_wait_ns as f64 / 1e6,
                pct = if total_ns > 0 {
                    100.0 * s.cpu_wait_ns as f64 / total_ns as f64
                } else {
                    0.0
                },
            );
        }
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
        // A prefill targeting a different sequence than the live
        // stream is a fresh start for that sequence (a mid-history
        // resume must come through `restore_to`, which switches
        // `active_seq` itself).
        debug_assert!(
            seq_id == self.active_seq || start_pos == 0,
            "cross-sequence prefill at pos {start_pos} without restore \
             (Session bug)",
        );
        self.switch_stream_to(seq_id);
        let (h0, m0) = self.ctx.prefetch_stats();
        let result =
            self.ctx
                .eval_prompt(tokens, start_pos, seq_id, &mut self.logits);
        let (h1, m1) = self.ctx.prefetch_stats();
        self.prefetch_stats.prefill_hits += h1.saturating_sub(h0);
        self.prefetch_stats.prefill_misses += m1.saturating_sub(m0);
        result?;
        Ok(&self.logits[..])
    }

    fn step(
        &mut self,
        token: Token,
        pos: usize,
        seq_id: i32,
    ) -> Result<&[f32], Self::Error> {
        // A step always extends the live stream; stepping an inactive
        // sequence would silently decode against another agent's
        // history.
        debug_assert_eq!(
            seq_id, self.active_seq,
            "step on inactive sequence (Session bug)",
        );
        let (h0, m0) = self.ctx.prefetch_stats();
        let result = self.ctx.eval_token(token, pos, seq_id, &mut self.logits);
        let (h1, m1) = self.ctx.prefetch_stats();
        self.prefetch_stats.decode_hits += h1.saturating_sub(h0);
        self.prefetch_stats.decode_misses += m1.saturating_sub(m0);
        result?;
        Ok(&self.logits[..])
    }

    fn n_ctx(&self) -> u32 {
        // moeflux reports usize; the Decoder trait uses u32 to match
        // llama.cpp. Saturating-cast: realistic n_ctx values fit.
        self.ctx.n_ctx().try_into().unwrap_or(u32::MAX)
    }

    fn n_seq_max(&self) -> u32 {
        // A namespace, not a partition (see the struct docs): any
        // number of sequences can exist as snapshot blobs, just never
        // concurrently resident. Report "unbounded" and let Session's
        // slot config choose its own cap.
        u32::MAX
    }

    fn memory_clear(&mut self) {
        self.ctx.memory_clear();
        // Full clear is `Session`'s thread-swap hammer: every
        // sequence's snapshots die with the KV.
        self.snapshots.clear();
        self.active_seq = -1;
    }

    /// Truncate `seq_id`'s live state — **only if it is the active
    /// sequence** (or `seq_id < 0`, "any sequence" per the trait).
    /// A removal targeting an inactive sequence is a no-op returning
    /// `true`: the live stream belongs to someone else, and the
    /// inactive sequence's actual storage is its snapshot blobs, which
    /// `Session` frees separately via [`Decoder::forget_pos`]. Without
    /// this guard, evicting an inactive cache slot would wipe the
    /// active agent's state.
    fn memory_seq_rm(&mut self, seq_id: i32, p0: i32, p1: i32) -> bool {
        if seq_id >= 0 && seq_id != self.active_seq {
            return true;
        }
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

    /// Serialize the whole `Ctx` state (`state_save`) into the
    /// wrapper's `(seq_id, pos)` snapshot store. moeflux's recurrence
    /// is GPU-resident and folds full history; the snapshot reads it
    /// back via the `state_save` wire format, so restoring it later is
    /// lossless. The trait surface is infallible — `state_save` errors
    /// only if linear buffers aren't initialized, which can't happen
    /// here because drama_llama's Session always prefills before
    /// checkpointing. We surface that contract violation as a panic.
    ///
    /// Only the *active* sequence can be checkpointed: the blob is a
    /// serialization of the live stream, and checkpointing an inactive
    /// sequence would label another agent's state with this key.
    fn checkpoint_pos(&mut self, seq_id: i32, pos: i32) {
        debug_assert_eq!(
            seq_id, self.active_seq,
            "checkpoint_pos on inactive sequence (Session bug)",
        );
        let mut buf = vec![0u8; self.ctx.state_size()];
        self.ctx
            .state_save(&mut buf)
            .expect("checkpoint_pos before first prefill (Session bug)");
        self.snapshots.insert((seq_id, pos), buf);
    }

    /// Load the blob stored at `(seq_id, pos)` back into the live
    /// stream (`state_load`) — the blob is the entire history, so this
    /// *is* the sequence switch; `active_seq` becomes `seq_id`.
    /// Snapshots on `seq_id` at positions `> pos` are dropped per the
    /// trait contract (other sequences' snapshots are untouched — the
    /// per-seq keying is exactly what makes that safe).
    fn restore_to(
        &mut self,
        seq_id: i32,
        pos: i32,
    ) -> Result<(), MemoryRmError> {
        let Some(bytes) = self.snapshots.get((seq_id, pos)) else {
            return Err(MemoryRmError::NoCheckpoint { pos });
        };
        // A reload error from a buffer we wrote ourselves with
        // `state_save` is a moeflux-internal bug; surface as
        // BackendUnsupported so Session falls back gracefully rather
        // than crashing. `state_load` preflights before mutating, so
        // the live stream is unchanged on error.
        if self.ctx.state_load(bytes).is_err() {
            return Err(MemoryRmError::BackendUnsupported { pos });
        }
        self.active_seq = seq_id;
        self.snapshots.invalidate_after(seq_id, pos);
        Ok(())
    }

    /// Drop the blob at `(seq_id, pos)` without touching any other
    /// snapshot or the live stream.
    ///
    /// `Session` calls this in two paths: tip replacement
    /// (`record_cache_hit` evicting the previous tip) and orphan
    /// pruning in `kv_setup_and_chunk_prefill` (forgetting prior
    /// breakpoints that are no longer set in the current call). Both
    /// are essential — without them, the LRU eventually evicts the
    /// system+tools snapshot, the most valuable cross-agent anchor.
    ///
    /// Idempotent: returns `Ok(())` even when no entry exists at
    /// `pos`, since both eviction paths may legitimately call this on
    /// positions already pruned by `restore_to`'s forward-drop.
    fn forget_pos(
        &mut self,
        seq_id: i32,
        pos: i32,
    ) -> Result<(), MemoryRmError> {
        self.snapshots.forget((seq_id, pos));
        Ok(())
    }
}
