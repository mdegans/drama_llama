//! Checkpoint-based rewind for llama.cpp sequences a KV truncate cannot
//! rewind on its own: sliding-window attention (gpt-oss, Gemma 4) and
//! recurrent / hybrid layers (Qwen3.6, Qwen3.8).
//!
//! FFI-free. [`LlamaCppDecoder`](super::LlamaCppDecoder) hands its
//! context over as a [`SeqMemory`], so the restore rules here are unit
//! tested against a simulated cache rather than a model.
//!
//! # Why a truncate is not enough
//!
//! A dense attention layer keeps every position, so dropping the tail
//! (`llama_memory_seq_rm(seq, pos, -1)`) leaves exactly the state the
//! model had at `pos`. Two kinds of layer break that:
//!
//! - **Sliding-window attention** keeps only the last `n_swa` positions
//!   live, and llama.cpp recycles the masked cells below them — for
//!   *any* sequence, idle ones included, and even with a full-size SWA
//!   cache (`find_slot` takes a masked cell as readily as an empty one).
//!   A sequence that has moved past `pos` may no longer hold the window
//!   *below* `pos`. Live (gpt-oss, 2026-10-01): an idle slot's window
//!   at its breakpoint was recycled by its neighbours, and the restore
//!   fell back to a full 10k-token re-prefill.
//! - **Recurrent layers** hold one state per sequence, the one after
//!   its last token; there is nothing to truncate back to.
//!
//! # Partial checkpoints
//!
//! llama.cpp's own server answers both with *context checkpoints*: a
//! `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` state holds just what a
//! truncate cannot rewind — the SWA cells still in the window, or the
//! recurrent state — and restores on top of the truncated dense KV.
//! So a checkpoint's size does not grow with the prefix (gpt-oss: 128
//! cells, ≈ 4.5 MiB), where a whole-sequence snapshot carries the
//! entire prefix again.
//!
//! The price is a dependency: a partial checkpoint at `pos` is only
//! valid while the dense KV *below* `pos` is the one it was taken over.
//! [`Checkpoints`] drops it the moment anything removes, copies over or
//! replaces that KV — every such call goes through the decoder, which
//! reports it here. Loading a stale one would hand the model the
//! window or recurrent state of a different history (#91's cardinal
//! sin), so invalidation errs wide.

use std::time::Instant;

use crate::{backend::MemoryRmError, snapshot_store::SnapshotStore};

/// One mebibyte, the unit the budget's options and flags are given in.
const MIB: usize = 1 << 20;

/// How a llama.cpp sequence rewinds to an earlier position — what
/// [`Decoder::checkpoint_pos`](crate::Decoder::checkpoint_pos) stores
/// and what [`Decoder::restore_to`](crate::Decoder::restore_to)
/// restores from. See
/// [`LlamaCppDecoder::checkpointing`](crate::LlamaCppDecoder::checkpointing).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Checkpointing {
    /// A truncate alone: every layer keeps every position (dense
    /// attention — cogito, Mistral 4). Nothing is stored.
    Off,
    /// Checkpoints hold only what a truncate cannot rewind — the
    /// sliding-window cells, the recurrent state — and restore on top
    /// of the truncated KV. See the `llama_cpp::checkpoint` module docs.
    Partial,
    /// Checkpoints hold the whole sequence and restore it wholesale:
    /// self-contained, and as large as the prefix. What a dense model
    /// takes when snapshots are forced on, and the safe choice for a
    /// hybrid whose attention also slides (see [`Self::for_model`]).
    Whole,
}

impl Checkpointing {
    /// The mode a model needs, from what llama.cpp reports about it:
    /// `llama_model_is_recurrent`, `llama_model_is_hybrid` and
    /// `llama_model_n_swa`.
    ///
    /// A hybrid with a sliding window takes [`Self::Whole`]: llama.cpp's
    /// plain hybrid memory leaves its attention out of a partial state
    /// altogether, which is only sound when that attention is dense.
    pub fn for_model(recurrent: bool, hybrid: bool, n_swa: u32) -> Self {
        match (recurrent, hybrid, n_swa) {
            (true, _, _) => Self::Partial,
            (_, true, 0) => Self::Partial,
            (_, true, _) => Self::Whole,
            (_, _, 0) => Self::Off,
            _ => Self::Partial,
        }
    }
}

/// The host RAM a decoder's checkpoints may hold — they live outside
/// the KV cache, and on a unified-memory Mac they compete with the
/// weights and the KV for the same memory. Past either bound the least
/// recently used checkpoints are evicted (the anchors they held then
/// restore through a lower one), and a checkpoint larger than a bound
/// on its own is not taken.
///
/// The default, 8 GiB in all and 4 GiB per sequence (prefix-cache
/// slot), is sized for a 96 GB Mac serving a 30 – 70 GB model with its
/// KV. Only Gemma 4 comes near it: ≈ 800 MiB a checkpoint, ten in all
/// and five per slot, where the count cap alone would allow ≈ 19 GiB at
/// four slots. Everything else in the fleet fits far below it (gpt-oss
/// 4.5 MiB, Qwen3.6 63 MiB, Qwen3.8 150 MiB a checkpoint). Set it with
/// [`LlamaCppOptions::checkpoint_mib`](crate::LlamaCppOptions::checkpoint_mib)
/// and
/// [`LlamaCppOptions::checkpoint_slot_mib`](crate::LlamaCppOptions::checkpoint_slot_mib),
/// or [`LlamaCppDecoder::set_checkpoint_budget`](crate::LlamaCppDecoder::set_checkpoint_budget).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct CheckpointBudget {
    /// Bytes across every sequence.
    pub total: usize,
    /// Bytes for any one sequence, so one busy slot cannot evict every
    /// other slot's anchors.
    pub per_seq: usize,
}

impl CheckpointBudget {
    /// The default total, in MiB: 8 GiB.
    pub const DEFAULT_TOTAL_MIB: u32 = 8 * 1024;
    /// The default per-sequence bound, in MiB: 4 GiB.
    pub const DEFAULT_PER_SEQ_MIB: u32 = 4 * 1024;

    /// A budget of `total` bytes, at most `per_seq` of them for any one
    /// sequence.
    pub fn new(total: usize, per_seq: usize) -> Self {
        Self { total, per_seq }
    }

    /// [`Self::new`], in MiB.
    pub fn from_mib(total: u32, per_seq: u32) -> Self {
        Self::new(total as usize * MIB, per_seq as usize * MIB)
    }
}

impl Default for CheckpointBudget {
    fn default() -> Self {
        Self::from_mib(Self::DEFAULT_TOTAL_MIB, Self::DEFAULT_PER_SEQ_MIB)
    }
}

/// The slice of a llama.cpp context's sequence memory that
/// [`Checkpoints`] drives. One method per `llama_memory_*` /
/// `llama_state_seq_*_ext` call, with llama.cpp's semantics.
pub(crate) trait SeqMemory {
    /// `llama_memory_seq_rm`: remove `[p0, p1)` (negative = unbounded).
    /// `false` when the memory refuses the range (a recurrent layer
    /// asked to forget part of its history), and then nothing changed.
    fn seq_rm(&mut self, seq: i32, p0: i32, p1: i32) -> bool;
    /// `llama_memory_seq_pos_min`, `-1` when the sequence is empty.
    fn seq_pos_min(&mut self, seq: i32) -> i32;
    /// `llama_memory_seq_pos_max`, `-1` when the sequence is empty.
    fn seq_pos_max(&mut self, seq: i32) -> i32;
    /// Serialize `seq` — only its non-truncatable part when `partial`.
    /// Empty on failure.
    fn save(&mut self, seq: i32, partial: bool) -> Vec<u8>;
    /// Load bytes from [`Self::save`] into `seq`, replacing the part
    /// they cover. `false` when llama.cpp rejects them, which leaves
    /// that part of `seq` cleared.
    fn load(&mut self, seq: i32, bytes: &[u8], partial: bool) -> bool;
}

/// The checkpoints of one llama.cpp context, and the rules for taking,
/// restoring and invalidating them. See the [module docs](self).
#[derive(Debug)]
pub(crate) struct Checkpoints {
    store: SnapshotStore,
    mode: Checkpointing,
    /// What the model needs; [`Self::force`] can only add to it.
    native: Checkpointing,
    /// `llama_model_n_swa`: how far back a position's attention reaches
    /// (`0` = unbounded, dense).
    n_swa: u32,
}

impl Checkpoints {
    /// Checkpoints for a model that needs `mode`, at most `cap` of them
    /// (see [`crate::snapshot_store::cap_for_sequences`]) within the
    /// default [`CheckpointBudget`].
    pub(crate) fn new(mode: Checkpointing, n_swa: u32, cap: usize) -> Self {
        let mut this = Self {
            store: SnapshotStore::with_cap(cap),
            mode,
            native: mode,
            n_swa,
        };
        this.set_budget(CheckpointBudget::default());
        this
    }

    /// Bound the bytes held, evicting down to the new bounds now.
    pub(crate) fn set_budget(&mut self, budget: CheckpointBudget) {
        self.store.set_byte_limits(budget.total, budget.per_seq);
    }

    pub(crate) fn mode(&self) -> Checkpointing {
        self.mode
    }

    pub(crate) fn len(&self) -> usize {
        self.store.len()
    }

    /// Bytes held across every sequence.
    pub(crate) fn bytes(&self) -> usize {
        self.store.bytes()
    }

    /// Force checkpointing on or off. On, a dense model takes
    /// [`Checkpointing::Whole`] snapshots (rewind insurance); a model
    /// that needs checkpoints keeps its own mode. Off drops them all.
    pub(crate) fn force(&mut self, enabled: bool) {
        self.mode = match (enabled, self.native) {
            (false, _) => Checkpointing::Off,
            (true, Checkpointing::Off) => Checkpointing::Whole,
            (true, native) => native,
        };
        if !enabled {
            self.store.clear();
        }
    }

    /// Store a checkpoint of `seq` at `pos`, which must be its head —
    /// the sequence holds exactly `[0, pos)`. Anything else would file
    /// another position's state under `pos`, so it is skipped (and
    /// logged): the anchor then restores through a lower one.
    ///
    /// A partial checkpoint already stored at `(seq, pos)` is kept as
    /// is: it survived invalidation, so the KV under it is the one it
    /// was taken over, and saving again would copy the same state.
    pub(crate) fn checkpoint(
        &mut self,
        mem: &mut impl SeqMemory,
        seq: i32,
        pos: i32,
    ) {
        if self.mode == Checkpointing::Off {
            return;
        }
        let head = mem.seq_pos_max(seq) + 1;
        if head != pos {
            tracing::warn!(
                target: "drama_llama::snapshot_store",
                event = "cache_degrade",
                reason = "checkpoint_skipped",
                cause = "off_head",
                seq_id = seq,
                pos,
                head,
                "prefix cache: no checkpoint taken at {pos}: the sequence's \
                 head is {head}, and only the head can be saved; a rewind \
                 there falls to a lower anchor",
            );
            return;
        }
        if self.mode == Checkpointing::Partial && self.store.refresh((seq, pos))
        {
            return;
        }
        let started = Instant::now();
        let bytes = mem.save(seq, self.mode == Checkpointing::Partial);
        if bytes.is_empty() {
            tracing::warn!(
                target: "drama_llama::snapshot_store",
                event = "cache_degrade",
                reason = "checkpoint_skipped",
                cause = "save_failed",
                seq_id = seq,
                pos,
                "prefix cache: no checkpoint taken at {pos}: llama.cpp \
                 failed to serialize the sequence; a rewind there falls to \
                 a lower anchor",
            );
            return;
        }
        let len = bytes.len();
        if self.store.insert((seq, pos), bytes) {
            tracing::debug!(
                target: "drama_llama::snapshot_store",
                event = "checkpoint_taken",
                seq_id = seq,
                pos,
                bytes = len,
                ms = started.elapsed().as_secs_f64() * 1e3,
                held_bytes = self.store.bytes(),
                held = self.store.len(),
                "checkpoint taken",
            );
        }
    }

    /// Rewind `seq` to `pos`: a truncate when that alone restores the
    /// state the model had there, else the checkpoint stored at `pos`.
    /// On success every checkpoint above `pos` is dropped (its future
    /// is gone); on failure lower ones stay restorable, which is what
    /// `Session`'s restore ladder relies on.
    ///
    /// A partial checkpoint the truncate made unnecessary is taken
    /// there and then if none is stored, since the head is `pos`: with
    /// a window-sized cache the neighbours recycle the window below an
    /// anchor soon after, and without one the anchor would restore
    /// once and then fail.
    pub(crate) fn restore(
        &mut self,
        mem: &mut impl SeqMemory,
        seq: i32,
        pos: i32,
    ) -> Result<(), MemoryRmError> {
        let started = Instant::now();
        let truncated = mem.seq_rm(seq, pos, -1);
        if truncated {
            // The KV above `pos` is gone whatever happens next.
            self.invalidate_partial(|s, p| s == seq && p > pos);
        }
        if truncated && self.window_intact(mem, seq, pos) {
            self.store.invalidate_after(seq, pos);
            tracing::debug!(
                target: "drama_llama::snapshot_store",
                event = "checkpoint_restore",
                restore_via = "truncate",
                seq_id = seq,
                pos,
                bytes = 0,
                ms = started.elapsed().as_secs_f64() * 1e3,
                "rewound by truncation",
            );
            if self.mode == Checkpointing::Partial {
                self.checkpoint(mem, seq, pos);
            }
            return Ok(());
        }
        let Some(bytes) = self.store.take((seq, pos)) else {
            return Err(MemoryRmError::NoCheckpoint { pos });
        };
        let len = bytes.len();
        let loaded = match self.mode {
            // The checkpoint first: a recurrent layer refuses the
            // truncate until its state is back at `pos`.
            Checkpointing::Partial => {
                mem.load(seq, &bytes, true) && mem.seq_rm(seq, pos, -1)
            }
            _ => {
                mem.seq_rm(seq, -1, -1);
                mem.load(seq, &bytes, false)
            }
        };
        if loaded && mem.seq_pos_max(seq) == pos - 1 {
            // Still valid — it survives its own restore so `Session` can
            // rewind to the same anchor again.
            self.store.insert((seq, pos), bytes);
            self.store.invalidate_after(seq, pos);
            tracing::debug!(
                target: "drama_llama::snapshot_store",
                event = "checkpoint_restore",
                restore_via = "checkpoint",
                seq_id = seq,
                pos,
                bytes = len,
                ms = started.elapsed().as_secs_f64() * 1e3,
                "rewound by loading a checkpoint",
            );
            Ok(())
        } else {
            // llama.cpp rejected bytes we serialized ourselves, or the
            // KV under them is not there. `Session`'s ladder tries the
            // next anchor below; each rung reloads its own checkpoint.
            Err(MemoryRmError::BackendUnsupported { pos })
        }
    }

    /// Whether `seq`, freshly truncated to `pos`, already holds the
    /// state the model had at `pos`: its head is `pos`, and with a
    /// sliding window, every position the next token attends to is
    /// still there.
    ///
    /// The token at `pos` attends to the cells `p` with
    /// `pos - p < n_swa` (`llama_hparams::is_masked_swa`, standard
    /// window), so the window is `[pos - n_swa + 1, pos)`, and llama.cpp
    /// keeps a sequence's cells contiguous from `pos_min`. The cell at
    /// `pos - n_swa` is masked for that token, and `find_slot` recycles
    /// it (`llama-kv-cache.cpp`, "SWA mask") as soon as a neighbour
    /// needs a cell, so it is not required: llama-server's one cell of
    /// margin would send every idle slot's rewind to its own head to a
    /// checkpoint load once a neighbour took that one cell. Conservative
    /// for chunked and symmetric windows, which reach back less far. A
    /// miss costs a checkpoint load, never correctness.
    fn window_intact(
        &self,
        mem: &mut impl SeqMemory,
        seq: i32,
        pos: i32,
    ) -> bool {
        if mem.seq_pos_max(seq) != pos - 1 {
            return false;
        }
        if self.n_swa == 0 || pos == 0 {
            return true;
        }
        let floor = (pos - self.n_swa as i32 + 1).max(0);
        (0..=floor).contains(&mem.seq_pos_min(seq))
    }

    /// Drop the checkpoint at `(seq, pos)`, if any.
    pub(crate) fn forget(&mut self, seq: i32, pos: i32) {
        self.store.forget((seq, pos));
    }

    /// Drop every checkpoint: the whole KV was cleared or replaced.
    pub(crate) fn clear(&mut self) {
        self.store.clear();
    }

    /// The KV of `seq` (every sequence when negative) changed from
    /// position `p0` on (from the start when negative): partial
    /// checkpoints above `p0` no longer sit on the KV they were taken
    /// over. One *at* `p0` still does — `[0, p0)` is untouched.
    pub(crate) fn invalidate_from(&mut self, seq: i32, p0: i32) {
        let p0 = p0.max(0);
        self.invalidate_partial(|s, p| (seq < 0 || s == seq) && p > p0);
    }

    /// Every sequence but `seq` was dropped.
    pub(crate) fn keep_only(&mut self, seq: i32) {
        self.invalidate_partial(|s, _| s != seq);
    }

    /// Drop the checkpoints matching `stale` — partial ones only. A
    /// whole-sequence snapshot restores wholesale, so it outlives any
    /// change to the KV under it.
    fn invalidate_partial(&mut self, mut stale: impl FnMut(i32, i32) -> bool) {
        if self.mode != Checkpointing::Partial {
            return;
        }
        let before = self.store.len();
        self.store.retain(|s, p| !stale(s, p));
        let count = before - self.store.len();
        if count > 0 {
            // DEBUG: the KV under them changed, which a rewind or a
            // slot reset does on every call. An anchor dropped here
            // shows up later as a `restore_failed`; this says why.
            tracing::debug!(
                target: "drama_llama::snapshot_store",
                event = "checkpoint_invalidated",
                count,
                held = self.store.len(),
                "dropped {count} partial checkpoint(s) whose KV changed",
            );
        }
    }

    #[cfg(test)]
    fn contains(&self, seq: i32, pos: i32) -> bool {
        self.store.contains((seq, pos))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, HashMap};

    /// The layers a [`SimMemory`] simulates.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Layers {
        /// Dense attention only (cogito, Mistral 4).
        Dense,
        /// Dense layers plus sliding-window ones, as llama.cpp's iSWA
        /// cache keeps them (gpt-oss, Gemma 4): two cell sets,
        /// positions reported from the window's.
        Iswa { n_swa: i32 },
        /// Dense attention plus a recurrent state (Qwen3.6, Qwen3.8).
        Hybrid,
    }

    /// A content-tracking stand-in for one llama.cpp context. Cells
    /// carry the token decoded into them, so [`Self::sees`] can tell a
    /// faithful restore from one that hands the model somebody else's
    /// window or state.
    struct SimMemory {
        layers: Layers,
        /// Dense cells per sequence: position → token.
        dense: HashMap<i32, BTreeMap<i32, u32>>,
        /// Sliding-window cells per sequence (iSWA only).
        swa: HashMap<i32, BTreeMap<i32, u32>>,
        /// Recurrent state per sequence: last position and the tokens
        /// folded into it (hybrid only).
        recurrent: HashMap<i32, (i32, Vec<u32>)>,
        /// Saved states, by the id `save` returns.
        saved: Vec<Saved>,
    }

    #[derive(Clone)]
    struct Saved {
        dense: Option<BTreeMap<i32, u32>>,
        swa: Option<BTreeMap<i32, u32>>,
        recurrent: Option<(i32, Vec<u32>)>,
    }

    impl SimMemory {
        fn new(layers: Layers) -> Self {
            Self {
                layers,
                dense: HashMap::new(),
                swa: HashMap::new(),
                recurrent: HashMap::new(),
                saved: Vec::new(),
            }
        }

        /// Decode `tokens` onto `seq` from `start`.
        fn decode(&mut self, seq: i32, start: i32, tokens: &[u32]) {
            for (i, &t) in tokens.iter().enumerate() {
                let pos = start + i as i32;
                self.dense.entry(seq).or_default().insert(pos, t);
                match self.layers {
                    Layers::Iswa { .. } => {
                        self.swa.entry(seq).or_default().insert(pos, t);
                    }
                    Layers::Hybrid => {
                        let (last, folded) = self
                            .recurrent
                            .entry(seq)
                            .or_insert((-1, Vec::new()));
                        assert_eq!(*last, pos - 1, "recurrent gap");
                        *last = pos;
                        folded.push(t);
                    }
                    Layers::Dense => {}
                }
            }
        }

        /// Recycle `seq`'s window cells below `keep_from` the way
        /// llama.cpp's `find_slot` does for a neighbour's decode: only
        /// cells masked for the sequence's head, `head - p >= n_swa`
        /// (`is_masked_swa`) — so up to and including `head - n_swa`.
        fn recycle(&mut self, seq: i32, keep_from: i32) {
            let Layers::Iswa { n_swa } = self.layers else {
                panic!("only a sliding window recycles");
            };
            let cells = self.swa.entry(seq).or_default();
            let head = cells.keys().max().copied().unwrap_or(-1) + 1;
            assert!(keep_from <= head - n_swa + 1, "would recycle live cells");
            cells.retain(|&p, _| p >= keep_from);
        }

        /// The tokens the model attends to at `seq`'s next position, or
        /// `None` when what it would see is not one consistent history
        /// — a hole in the dense prefix, a window cell missing or from
        /// different content, a recurrent state of another prefix.
        fn sees(&self, seq: i32) -> Option<Vec<u32>> {
            let empty = BTreeMap::new();
            let dense = self.dense.get(&seq).unwrap_or(&empty);
            let head = dense.keys().max().map_or(0, |p| p + 1);
            let tokens: Vec<u32> = (0..head)
                .map(|p| dense.get(&p).copied())
                .collect::<Option<_>>()?;
            match self.layers {
                Layers::Dense => {}
                Layers::Iswa { n_swa } => {
                    let swa = self.swa.get(&seq).unwrap_or(&empty);
                    if swa.keys().any(|&p| p >= head) {
                        return None;
                    }
                    for p in (head - n_swa + 1).max(0)..head {
                        if swa.get(&p) != Some(&tokens[p as usize]) {
                            return None;
                        }
                    }
                }
                Layers::Hybrid => {
                    let (last, folded) = self
                        .recurrent
                        .get(&seq)
                        .cloned()
                        .unwrap_or((-1, vec![]));
                    if last != head - 1 || folded != tokens {
                        return None;
                    }
                }
            }
            Some(tokens)
        }
    }

    impl SeqMemory for SimMemory {
        fn seq_rm(&mut self, seq: i32, p0: i32, p1: i32) -> bool {
            let p0 = p0.max(0);
            let p1 = if p1 < 0 { i32::MAX } else { p1 };
            if self.layers == Layers::Hybrid {
                let last = self.recurrent.get(&seq).map(|r| r.0);
                if p0 == 0 && p1 == i32::MAX {
                    self.recurrent.remove(&seq);
                } else if last.is_some_and(|l| p0 <= l && l < p1) {
                    // Recurrent memory refuses a partial erase, and the
                    // hybrid touches nothing then.
                    return false;
                }
            }
            for cells in [&mut self.dense, &mut self.swa] {
                if let Some(c) = cells.get_mut(&seq) {
                    c.retain(|&p, _| p < p0 || p >= p1);
                }
            }
            true
        }

        fn seq_pos_min(&mut self, seq: i32) -> i32 {
            let first = |m: &HashMap<i32, BTreeMap<i32, u32>>| {
                m.get(&seq)
                    .and_then(|c| c.keys().next().copied())
                    .unwrap_or(-1)
            };
            match self.layers {
                Layers::Dense => first(&self.dense),
                Layers::Iswa { .. } => first(&self.swa),
                Layers::Hybrid => {
                    let recurrent =
                        self.recurrent.get(&seq).map_or(-1, |r| r.0);
                    first(&self.dense).max(recurrent)
                }
            }
        }

        fn seq_pos_max(&mut self, seq: i32) -> i32 {
            let last = |m: &HashMap<i32, BTreeMap<i32, u32>>| {
                m.get(&seq)
                    .and_then(|c| c.keys().next_back().copied())
                    .unwrap_or(-1)
            };
            match self.layers {
                Layers::Dense => last(&self.dense),
                Layers::Iswa { .. } => last(&self.swa),
                Layers::Hybrid => {
                    let recurrent =
                        self.recurrent.get(&seq).map_or(-1, |r| r.0);
                    last(&self.dense).min(recurrent)
                }
            }
        }

        fn save(&mut self, seq: i32, partial: bool) -> Vec<u8> {
            let dense = self.dense.get(&seq).cloned().unwrap_or_default();
            let saved = match self.layers {
                // A dense cache ignores the flag.
                Layers::Dense => Saved {
                    dense: Some(dense),
                    swa: None,
                    recurrent: None,
                },
                Layers::Iswa { n_swa } => {
                    // Only the cells unmasked for the head are written.
                    let swa = self.swa.get(&seq).cloned().unwrap_or_default();
                    let head = swa.keys().max().map_or(0, |p| p + 1);
                    let live =
                        swa.into_iter().filter(|(p, _)| head - 1 - p < n_swa);
                    Saved {
                        dense: (!partial).then_some(dense),
                        swa: Some(live.collect()),
                        recurrent: None,
                    }
                }
                Layers::Hybrid => Saved {
                    dense: (!partial).then_some(dense),
                    swa: None,
                    recurrent: self.recurrent.get(&seq).cloned(),
                },
            };
            self.saved.push(saved);
            (self.saved.len() as u32).to_le_bytes().to_vec()
        }

        fn load(&mut self, seq: i32, bytes: &[u8], _partial: bool) -> bool {
            let id = u32::from_le_bytes(bytes.try_into().unwrap()) as usize;
            let saved = self.saved[id - 1].clone();
            if let Some(dense) = saved.dense {
                self.dense.insert(seq, dense);
            }
            if let Some(swa) = saved.swa {
                self.swa.insert(seq, swa);
            }
            if let Some(recurrent) = saved.recurrent {
                self.recurrent.insert(seq, recurrent);
            }
            true
        }
    }

    fn tokens(n: u32) -> Vec<u32> {
        (1000..1000 + n).collect()
    }

    /// A model's layers, decoded with `n` tokens on seq 0, and the
    /// checkpoints its mode takes.
    fn rig(layers: Layers, n: u32) -> (SimMemory, Checkpoints) {
        let (mode, n_swa) = match layers {
            Layers::Dense => (Checkpointing::for_model(false, false, 0), 0),
            Layers::Iswa { n_swa } => (
                Checkpointing::for_model(false, false, n_swa as u32),
                n_swa as u32,
            ),
            Layers::Hybrid => (Checkpointing::for_model(false, true, 0), 0),
        };
        let mut mem = SimMemory::new(layers);
        mem.decode(0, 0, &tokens(n));
        (mem, Checkpoints::new(mode, n_swa, 16))
    }

    #[test]
    fn modes_follow_what_llama_cpp_reports() {
        use Checkpointing::*;
        // cogito, Mistral 4: dense.
        assert_eq!(Checkpointing::for_model(false, false, 0), Off);
        // gpt-oss (128), Gemma 4 (1024): sliding window.
        assert_eq!(Checkpointing::for_model(false, false, 128), Partial);
        // Qwen3.6, Qwen3.8: hybrid, dense attention.
        assert_eq!(Checkpointing::for_model(false, true, 0), Partial);
        assert_eq!(Checkpointing::for_model(true, false, 0), Partial);
        // A hybrid whose attention slides: a partial state would leave
        // that attention out.
        assert_eq!(Checkpointing::for_model(false, true, 4096), Whole);
    }

    /// The live gpt-oss miss (2026-10-01): an idle slot's window at its
    /// breakpoint was recycled by its neighbours, so the truncate could
    /// not restore it — and nothing was checkpointed, because a
    /// sliding-window model was treated as dense. With the window
    /// checkpointed at the anchor, the rewind lands there.
    #[test]
    fn a_recycled_window_restores_from_its_checkpoint() {
        let (mut mem, mut ckpt) = rig(Layers::Iswa { n_swa: 8 }, 40);
        mem.seq_rm(0, 20, -1);
        ckpt.checkpoint(&mut mem, 0, 20);
        mem.decode(0, 20, &tokens(60)[20..]);
        mem.recycle(0, 50);

        assert_eq!(ckpt.restore(&mut mem, 0, 20), Ok(()));
        assert_eq!(mem.sees(0), Some(tokens(20)));
        assert!(ckpt.contains(0, 20), "kept for the next rewind");
    }

    /// The silent half of the same bug: the truncate left the head at
    /// the anchor but only *part* of the window below it. Checking the
    /// head alone called that a lossless rewind, and the model would
    /// have attended over a window with a hole in it.
    #[test]
    fn a_truncate_that_leaves_half_a_window_is_not_a_rewind() {
        let (mut mem, mut ckpt) = rig(Layers::Iswa { n_swa: 8 }, 40);
        mem.recycle(0, 17);
        // Without a checkpoint the anchor is unrestorable — reported,
        // so the ladder goes lower, never silently "restored".
        assert_eq!(
            ckpt.restore(&mut mem, 0, 20),
            Err(MemoryRmError::NoCheckpoint { pos: 20 }),
        );
        assert_eq!(mem.sees(0), None, "half a window is not a state");
    }

    /// A window the neighbours never reached truncates losslessly, as a
    /// dense cache does: no checkpoint load, and none needed.
    #[test]
    fn an_intact_window_truncates_without_a_checkpoint() {
        let (mut mem, mut ckpt) = rig(Layers::Iswa { n_swa: 8 }, 40);
        assert_eq!(ckpt.restore(&mut mem, 0, 20), Ok(()));
        assert_eq!(mem.sees(0), Some(tokens(20)));
    }

    /// The window boundary is llama.cpp's: the token at `pos` reads
    /// `[pos - n_swa + 1, pos)`, and the cell at `pos - n_swa` is masked
    /// for it — recyclable by any neighbour. An idle slot that lost
    /// only that cell still rewinds to its own head by truncation; one
    /// cell more and the window has a hole, which fails closed.
    #[test]
    fn the_window_ends_where_llama_cpp_masks_it() {
        let n_swa = 8;
        // Idle at its head (40): a neighbour took the masked cell 32.
        let (mut mem, mut ckpt) = rig(Layers::Iswa { n_swa }, 40);
        mem.recycle(0, 40 - n_swa + 1);
        assert_eq!(ckpt.restore(&mut mem, 0, 40), Ok(()));
        assert_eq!(mem.sees(0), Some(tokens(40)));

        // Rewinding to 20 needs [13, 20): 13 is the last cell it may
        // lose to the neighbours' recycling.
        let (mut mem, mut ckpt) = rig(Layers::Iswa { n_swa }, 40);
        mem.recycle(0, 20 - n_swa + 1);
        assert_eq!(ckpt.restore(&mut mem, 0, 20), Ok(()));
        assert_eq!(mem.sees(0), Some(tokens(20)));

        let (mut mem, mut ckpt) = rig(Layers::Iswa { n_swa }, 40);
        mem.recycle(0, 20 - n_swa + 2);
        assert_eq!(
            ckpt.restore(&mut mem, 0, 20),
            Err(MemoryRmError::NoCheckpoint { pos: 20 }),
        );
        assert_eq!(mem.sees(0), None, "cell 13 is read at 20");
    }

    /// An anchor restored by truncation — its checkpoint LRU-evicted,
    /// or never taken — is checkpointed on the way, since its head is
    /// right there: with a window-sized cache the neighbours recycle
    /// that window soon after, and the anchor would otherwise restore
    /// once and then fail.
    #[test]
    fn a_truncated_anchor_is_checkpointed_for_next_time() {
        let (mut mem, mut ckpt) = rig(Layers::Iswa { n_swa: 8 }, 40);
        assert_eq!(ckpt.restore(&mut mem, 0, 20), Ok(()));
        assert!(ckpt.contains(0, 20));

        mem.decode(0, 20, &tokens(60)[20..]);
        mem.recycle(0, 50);
        assert_eq!(ckpt.restore(&mut mem, 0, 20), Ok(()));
        assert_eq!(mem.sees(0), Some(tokens(20)));
    }

    /// Taking a checkpoint that is already stored copies nothing: it
    /// survived invalidation, so it is the state at the head.
    #[test]
    fn a_stored_partial_checkpoint_is_not_taken_twice() {
        let (mut mem, mut ckpt) = rig(Layers::Hybrid, 10);
        ckpt.checkpoint(&mut mem, 0, 10);
        ckpt.checkpoint(&mut mem, 0, 10);
        assert_eq!(mem.saved.len(), 1);
    }

    /// The byte budget bounds what is held, least recently used first:
    /// a restore is a use, so the anchor a slot just rewound to outlives
    /// an idle slot's older one.
    #[test]
    fn checkpoints_stay_within_their_byte_budget() {
        let (mut mem, mut ckpt) = rig(Layers::Hybrid, 10);
        mem.decode(1, 0, &tokens(10));
        // The sim's states are 4 bytes each: room for two.
        ckpt.set_budget(CheckpointBudget::new(8, 8));
        ckpt.checkpoint(&mut mem, 0, 10);
        ckpt.checkpoint(&mut mem, 1, 10);
        assert_eq!(ckpt.restore(&mut mem, 0, 10), Ok(()));
        mem.decode(1, 10, &tokens(20)[10..]);
        ckpt.checkpoint(&mut mem, 1, 20);
        assert_eq!((ckpt.len(), ckpt.bytes()), (2, 8));
        assert!(ckpt.contains(0, 10), "restored, so recently used");
        assert!(!ckpt.contains(1, 10), "the least recently used goes");
        assert!(ckpt.contains(1, 20));
    }

    /// A whole-sequence snapshot (a dense model with snapshots forced
    /// on; a hybrid whose attention slides) restores wholesale through
    /// `restore`: it outlives a wipe of the KV under it, and replaces
    /// both caches of a sliding-window model.
    #[test]
    fn a_whole_snapshot_restores_through_restore() {
        let (mut mem, mut ckpt) = rig(Layers::Dense, 20);
        ckpt.force(true);
        assert_eq!(ckpt.mode(), Checkpointing::Whole);
        ckpt.checkpoint(&mut mem, 0, 20);
        mem.decode(0, 20, &tokens(30)[20..]);
        mem.seq_rm(0, -1, -1);
        ckpt.invalidate_from(0, -1);
        assert_eq!(ckpt.restore(&mut mem, 0, 20), Ok(()));
        assert_eq!(mem.sees(0), Some(tokens(20)));

        let mut mem = SimMemory::new(Layers::Iswa { n_swa: 8 });
        mem.decode(0, 0, &tokens(20));
        let mut ckpt = Checkpoints::new(Checkpointing::Whole, 8, 16);
        ckpt.checkpoint(&mut mem, 0, 20);
        mem.decode(0, 20, &tokens(60)[20..]);
        mem.recycle(0, 50);
        assert_eq!(ckpt.restore(&mut mem, 0, 20), Ok(()));
        assert_eq!(mem.sees(0), Some(tokens(20)));
        assert!(ckpt.contains(0, 20), "kept for the next rewind");
    }

    /// Dense models are untouched: nothing is stored, the truncate is
    /// the rewind, at any position.
    #[test]
    fn a_dense_model_rewinds_by_truncation_alone() {
        let (mut mem, mut ckpt) = rig(Layers::Dense, 40);
        assert_eq!(ckpt.mode(), Checkpointing::Off);
        ckpt.checkpoint(&mut mem, 0, 40);
        assert_eq!(ckpt.len(), 0);
        assert_eq!(ckpt.restore(&mut mem, 0, 33), Ok(()));
        assert_eq!(mem.sees(0), Some(tokens(33)));
        assert_eq!(ckpt.restore(&mut mem, 0, 7), Ok(()));
        assert_eq!(mem.sees(0), Some(tokens(7)));
    }

    /// Hybrid: the recurrent state comes back from the checkpoint, the
    /// attention KV from the truncate — and a rung that fails keeps the
    /// lower ones restorable (the restore ladder's contract).
    #[test]
    fn a_hybrid_restores_each_rung_of_the_ladder() {
        let (mut mem, mut ckpt) = rig(Layers::Hybrid, 10);
        ckpt.checkpoint(&mut mem, 0, 10);
        mem.decode(0, 10, &tokens(20)[10..]);
        ckpt.checkpoint(&mut mem, 0, 20);
        mem.decode(0, 20, &tokens(30)[20..]);

        // The head itself needs no checkpoint.
        assert_eq!(ckpt.restore(&mut mem, 0, 30), Ok(()));
        assert_eq!(mem.sees(0), Some(tokens(30)));
        // An anchor with none fails, and leaves the lower ones alone.
        assert_eq!(
            ckpt.restore(&mut mem, 0, 25),
            Err(MemoryRmError::NoCheckpoint { pos: 25 }),
        );
        assert_eq!(ckpt.restore(&mut mem, 0, 20), Ok(()));
        assert_eq!(mem.sees(0), Some(tokens(20)));
        assert_eq!(ckpt.restore(&mut mem, 0, 10), Ok(()));
        assert_eq!(mem.sees(0), Some(tokens(10)));
        assert!(!ckpt.contains(0, 20), "its future is gone");
    }

    /// A partial checkpoint sits on the KV below it. Once that KV is
    /// rewritten — a rewind below it, then a different continuation —
    /// loading it would graft the old history's state onto the new one;
    /// it must be gone instead.
    #[test]
    fn a_partial_checkpoint_dies_with_the_kv_under_it() {
        for layers in [Layers::Hybrid, Layers::Iswa { n_swa: 8 }] {
            let (mut mem, mut ckpt) = rig(layers, 30);
            mem.seq_rm(0, 20, -1);
            if layers == Layers::Hybrid {
                // A recurrent state cannot be truncated; start over.
                mem.seq_rm(0, -1, -1);
                mem.decode(0, 0, &tokens(20));
            }
            ckpt.checkpoint(&mut mem, 0, 20);
            assert!(ckpt.contains(0, 20), "{layers:?}");

            // The client edits history at 15: the session wipes and
            // re-prefills (every removal reaches `invalidate_from`).
            mem.seq_rm(0, -1, -1);
            ckpt.invalidate_from(0, -1);
            let edited: Vec<u32> = (0..30).map(|t| t + 5000).collect();
            mem.decode(0, 0, &edited);

            assert!(!ckpt.contains(0, 20), "{layers:?}");
            assert_eq!(
                ckpt.restore(&mut mem, 0, 20),
                if layers == Layers::Hybrid {
                    Err(MemoryRmError::NoCheckpoint { pos: 20 })
                } else {
                    // The new history's window is intact.
                    Ok(())
                },
                "{layers:?}",
            );
        }
    }

    /// Invalidation is positional: a removal from `p0` leaves the
    /// checkpoints at or below it, other sequences, and — for a whole
    /// snapshot, which restores wholesale — everything.
    #[test]
    fn invalidation_spares_what_the_change_did_not_touch() {
        let mut ckpt = Checkpoints::new(Checkpointing::Partial, 0, 16);
        let mut mem = SimMemory::new(Layers::Hybrid);
        for seq in 0..2 {
            for pos in [10, 20, 30] {
                mem.seq_rm(seq, -1, -1);
                mem.decode(seq, 0, &tokens(pos as u32));
                ckpt.checkpoint(&mut mem, seq, pos);
            }
        }
        ckpt.invalidate_from(0, 20);
        assert!(ckpt.contains(0, 10) && ckpt.contains(0, 20));
        assert!(!ckpt.contains(0, 30));
        assert!(ckpt.contains(1, 30));
        ckpt.invalidate_from(-1, 10);
        assert!(!ckpt.contains(0, 20) && !ckpt.contains(1, 20));
        assert!(ckpt.contains(1, 10));
        ckpt.keep_only(0);
        assert!(!ckpt.contains(1, 10) && ckpt.contains(0, 10));

        let mut whole = Checkpoints::new(Checkpointing::Whole, 0, 16);
        mem.seq_rm(0, -1, -1);
        mem.decode(0, 0, &tokens(10));
        whole.checkpoint(&mut mem, 0, 10);
        whole.invalidate_from(0, -1);
        assert!(whole.contains(0, 10), "a whole snapshot stands alone");
    }

    /// A checkpoint is taken at the head or not at all: filing another
    /// position's state under `pos` would restore the wrong prefix.
    #[test]
    fn a_checkpoint_off_the_head_is_skipped() {
        let (mut mem, mut ckpt) = rig(Layers::Hybrid, 30);
        ckpt.checkpoint(&mut mem, 0, 20);
        assert_eq!(ckpt.len(), 0);
        ckpt.checkpoint(&mut mem, 0, 30);
        assert_eq!(ckpt.len(), 1);
    }

    /// The fleet, as a `vocab_only` load (CPU, no tensors) shows it:
    /// the sliding-window and hybrid models checkpoint, the dense ones
    /// stay on truncation alone. Qwen3.8 is hybrid (`qwen35`: gated
    /// delta-net layers), not dense. Such a load skips hyperparameters,
    /// so `llama_model_n_swa` reads `0` here; the window comes from the
    /// GGUF key it is loaded from (the real load's value is asserted in
    /// `tests/swa_checkpoint.rs`). Reads
    /// `$DRAMA_LLAMA_MODEL_DIR`, else `models/`; each missing GGUF is
    /// skipped, loudly.
    #[test]
    #[ignore = "needs the fleet GGUFs (vocab-only loads, CPU)"]
    fn the_fleet_checkpoints_as_its_layers_need() {
        use Checkpointing::*;
        let fleet = [
            ("gpt-oss-120b-MXFP4.gguf", Partial, 128),
            ("gemma-4-31B-it-qat-UD-Q4_K_XL.gguf", Partial, 1024),
            ("Qwen3.6-35B-A3B-UD-IQ4_XS.gguf", Partial, 0),
            ("Qwen3.8-27B-UD-Q8_K_XL.gguf", Partial, 0),
            ("cogito-32b.gguf", Off, 0),
            ("Mistral-Small-4-119B-2603-UD-Q4_K_XL.gguf", Off, 0),
        ];
        let models = std::env::var_os("DRAMA_LLAMA_MODEL_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("models")
            });
        let mut checked = 0;
        for (file, mode, n_swa) in fleet {
            let path = models.join(file);
            if !path.exists() {
                eprintln!("SKIP: {} absent", path.display());
                continue;
            }
            let mut params = crate::LlamaCppOptions::default().model_params();
            params.vocab_only = true;
            params.n_gpu_layers = 0;
            let model = crate::LlamaCppModel::from_file(path, Some(params))
                .expect("vocab-only load");
            assert_eq!(model.n_swa(), 0, "{file}: hparams skipped");
            let arch = model.get_meta("general.architecture").unwrap();
            let key = format!("{arch}.attention.sliding_window");
            let window: u32 = model
                .get_meta(key.as_str())
                .map_or(0, |w| w.parse().expect("an integer"));
            let got = Checkpointing::for_model(
                model.is_recurrent(),
                model.is_hybrid(),
                window,
            );
            assert_eq!((got, window), (mode, n_swa), "{file}");
            checked += 1;
        }
        assert!(checked > 0, "no fleet GGUF under {}", models.display());
    }

    /// Forcing snapshots on gives a dense model whole ones (the old
    /// behaviour); a model that needs checkpoints keeps its own kind,
    /// and forcing off drops them.
    #[test]
    fn forcing_snapshots_keeps_a_models_own_mode() {
        let (mut mem, mut dense) = rig(Layers::Dense, 10);
        dense.force(true);
        assert_eq!(dense.mode(), Checkpointing::Whole);
        dense.checkpoint(&mut mem, 0, 10);
        assert_eq!(dense.len(), 1);
        dense.force(false);
        assert_eq!((dense.mode(), dense.len()), (Checkpointing::Off, 0));

        let (_, mut swa) = rig(Layers::Iswa { n_swa: 8 }, 10);
        swa.force(true);
        assert_eq!(swa.mode(), Checkpointing::Partial);
    }
}
