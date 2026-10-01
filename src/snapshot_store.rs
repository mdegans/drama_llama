//! Bounded LRU of serialized per-sequence decoder states, keyed by
//! `(seq_id, pos)`. Pure bookkeeping — FFI-free, so the eviction /
//! invalidation logic is unit-testable without a model. Shared by the
//! llama.cpp decoder (sliding-window, recurrent and hybrid models — see
//! `llama_cpp::checkpoint`) and the moeflux decoder (whole-`Ctx`
//! `state_save` blobs realizing `(seq, pos)` keying over a physically
//! single stream).

use std::collections::{HashMap, VecDeque};

/// Floor on retained snapshots, and the cap for a decoder that cannot
/// size one from its sequence count (moeflux, whose `n_seq_max` is a
/// namespace bound, not a slot count).
pub(crate) const MAX_SEQ_SNAPSHOTS: usize = 16;

/// Snapshots one prefix-cache slot can hold at its peak: one per
/// `cache_control` breakpoint ([`MAX_CACHE_CONTROLS`], the automatic
/// one included) plus two tips — the outgoing one, still restorable
/// while the turn it anchored generates, and the incoming one taken at
/// that turn's end.
///
/// [`MAX_CACHE_CONTROLS`]: crate::MAX_CACHE_CONTROLS
pub(crate) const SNAPSHOTS_PER_SLOT: usize =
    crate::chat_template::MAX_CACHE_CONTROLS + 2;

/// The snapshot cap for a decoder serving `n_seq` sequences: a full
/// [`SNAPSHOTS_PER_SLOT`] set for every slot, never below
/// [`MAX_SEQ_SNAPSHOTS`].
///
/// The LRU is shared by every sequence, so a flat cap below this lets
/// one slot's newest snapshot evict *another* slot's oldest — its
/// system anchor, the most valuable one. A flat 16 held three slots
/// before automatic caching added a fourth anchor per slot; with
/// `--cache-slots 4` on a hybrid model it sat exactly at the cap, and
/// the next anchor cost a conversation its prefix.
///
/// The trade-off is host RAM, `cap × (largest snapshot)`. On llama.cpp
/// a snapshot holds only what a KV truncate cannot rewind, so its size
/// does not grow with the prefix: the recurrent state (Qwen3.6 ≈ 63 MiB,
/// Qwen3.8 ≈ 150 MiB) or the sliding-window cells (gpt-oss ≈ 4.5 MiB,
/// Gemma 4 ≈ 800 MiB). At four slots that is at most 24 snapshots —
/// ≈ 19 GiB on Gemma 4, more than a unified-memory Mac can spare next
/// to the weights, so llama.cpp's store also has a byte budget
/// ([`SnapshotStore::set_byte_limits`],
/// [`crate::llama_cpp::checkpoint::CheckpointBudget`]). Eviction logs
/// each dropped snapshot's size and which bound it hit.
///
/// llama.cpp-only (moeflux keeps [`MAX_SEQ_SNAPSHOTS`]), hence the cfg
/// on the lint.
#[cfg_attr(not(feature = "llama-cpp"), allow(dead_code))]
pub(crate) fn cap_for_sequences(n_seq: usize) -> usize {
    n_seq
        .saturating_mul(SNAPSHOTS_PER_SLOT)
        .max(MAX_SEQ_SNAPSHOTS)
}

/// Bounded LRU of serialized decoder states.
///
/// A stored value is serialized state for its key's sequence, taken
/// when that sequence held exactly positions `[0, pos)`. Whether it
/// goes stale is the owner's concern: a whole-sequence blob (moeflux,
/// or llama.cpp's forced snapshots) restores wholesale and stays
/// restorable regardless of later KV mutations, while a llama.cpp
/// *partial* checkpoint restores on top of the KV below `pos` and must
/// be dropped as soon as that KV changes ([`Self::retain`]).
///
/// Three bounds, each enforced by evicting the least recently used
/// snapshot it covers: a count ([`cap_for_sequences`]), a byte total,
/// and bytes per sequence (unbounded until [`Self::set_byte_limits`]).
/// Each sequence's lowest snapshot goes last, though: it is the slot's
/// first anchor — the system prompt and tools every later request of
/// its conversation shares — and the most valuable, yet the LRU order
/// alone ranks it lowest, since a restore refreshes only the anchor it
/// lands on and later anchors are taken after it. Under Gemma 4's
/// ≈ 800 MiB checkpoints a slot's byte share holds five, one fewer than
/// a full set, so pure LRU evicted exactly that anchor first.
#[derive(Debug)]
pub(crate) struct SnapshotStore {
    map: HashMap<(i32, i32), Vec<u8>>,
    /// Insertion order, oldest first. Re-inserting an existing key
    /// refreshes its position.
    order: VecDeque<(i32, i32)>,
    /// Eviction cap. See [`cap_for_sequences`].
    cap: usize,
    /// Bytes across every sequence.
    max_bytes: usize,
    /// Bytes for any one sequence, so one busy slot cannot take the
    /// whole budget from the others.
    max_seq_bytes: usize,
}

impl Default for SnapshotStore {
    fn default() -> Self {
        Self::with_cap(MAX_SEQ_SNAPSHOTS)
    }
}

impl SnapshotStore {
    /// An empty store holding at most `cap` snapshots (at least one).
    pub(crate) fn with_cap(cap: usize) -> Self {
        Self {
            map: HashMap::new(),
            order: VecDeque::new(),
            cap: cap.max(1),
            max_bytes: usize::MAX,
            max_seq_bytes: usize::MAX,
        }
    }

    /// Bound the bytes held in total and per sequence, evicting the
    /// least recently used snapshots now if the store is already over.
    #[cfg_attr(not(feature = "llama-cpp"), allow(dead_code))]
    pub(crate) fn set_byte_limits(&mut self, total: usize, per_seq: usize) {
        self.max_bytes = total;
        self.max_seq_bytes = per_seq;
        let seqs: Vec<i32> = self.order.iter().map(|k| k.0).collect();
        for seq in seqs {
            while self.seq_bytes(seq) > self.max_seq_bytes
                && self.evict_oldest(|s| s == seq, "seq_bytes")
            {}
        }
        while self.bytes() > self.max_bytes
            && self.evict_oldest(|_| true, "bytes")
        {}
    }

    /// Insert (or replace) the snapshot at `key`, evicting the least
    /// recently used ones until it fits every bound. `false` when it
    /// cannot fit at all — larger than a byte limit on its own — and
    /// was dropped (and logged) instead; nothing else is evicted then.
    pub(crate) fn insert(&mut self, key: (i32, i32), bytes: Vec<u8>) -> bool {
        // A replaced snapshot's bytes stop counting against the new one.
        self.forget(key);
        let len = bytes.len();
        let limit = self.max_seq_bytes.min(self.max_bytes);
        if len > limit {
            tracing::warn!(
                target: "drama_llama::snapshot_store",
                event = "cache_degrade",
                reason = "snapshot_over_budget",
                seq_id = key.0,
                pos = key.1,
                bytes = len,
                limit,
                "snapshot (seq {}, pos {}) is {len} bytes, over the {limit} \
                 byte budget on its own; not stored, so a rewind there \
                 falls to a lower anchor",
                key.0,
                key.1,
            );
            return false;
        }
        while self.seq_bytes(key.0) + len > self.max_seq_bytes
            && self.evict_oldest(|s| s == key.0, "seq_bytes")
        {}
        while self.bytes() + len > self.max_bytes
            && self.evict_oldest(|_| true, "bytes")
        {}
        self.map.insert(key, bytes);
        self.order.push_back(key);
        while self.map.len() > self.cap && self.evict_oldest(|_| true, "count")
        {
        }
        true
    }

    /// Evict the least recently used snapshot whose sequence matches
    /// `seq` — any but a sequence's lowest while one is left (see
    /// [`SnapshotStore`]) — logging which `bound` forced it. `false`
    /// when none matches.
    fn evict_oldest(
        &mut self,
        mut seq: impl FnMut(i32) -> bool,
        bound: &'static str,
    ) -> bool {
        let floor =
            |s: i32| self.map.keys().filter(|k| k.0 == s).map(|k| k.1).min();
        let Some(at) = self
            .order
            .iter()
            .position(|k| seq(k.0) && floor(k.0) != Some(k.1))
            .or_else(|| self.order.iter().position(|k| seq(k.0)))
        else {
            return false;
        };
        let oldest = self.order.remove(at).expect("position is in range");
        let bytes = self.map.remove(&oldest).map_or(0, |b| b.len());
        // A restore target is gone, but not necessarily a reuse: the
        // anchor that sat here restores through a lower one (Session's
        // restore ladder), and Session logs what that cost if a request
        // ever asks for it (`restore_failed`, WARN past a few hundred
        // tokens). So INFO here.
        tracing::info!(
            target: "drama_llama::snapshot_store",
            event = "cache_degrade",
            reason = "snapshot_evicted",
            seq_id = oldest.0,
            pos = oldest.1,
            bytes,
            bound,
            cap = self.cap,
            max_bytes = self.max_bytes,
            max_seq_bytes = self.max_seq_bytes,
            "snapshot store over its {bound} bound; dropped the least \
             recently used snapshot (seq {}, pos {}, {bytes} bytes)",
            oldest.0,
            oldest.1,
        );
        true
    }

    /// Bytes held across every sequence.
    pub(crate) fn bytes(&self) -> usize {
        self.map.values().map(Vec::len).sum()
    }

    /// Bytes held for `seq`.
    fn seq_bytes(&self, seq: i32) -> usize {
        self.map
            .iter()
            .filter(|((s, _), _)| *s == seq)
            .map(|(_, b)| b.len())
            .sum()
    }

    /// Borrow the snapshot at `key`, if any, refreshing its LRU
    /// position (a read is a use — the caller is about to restore it,
    /// which makes it the most current state).
    ///
    /// Only the moeflux wrapper borrows (its `Ctx` is a disjoint
    /// field); llama.cpp must `take` + re-insert around its `&mut
    /// self` FFI call — hence the cfg on the lint.
    #[cfg_attr(
        not(all(feature = "moeflux", target_os = "macos")),
        allow(dead_code)
    )]
    pub(crate) fn get(&mut self, key: (i32, i32)) -> Option<&Vec<u8>> {
        self.refresh(key);
        self.map.get(&key)
    }

    /// Mark the snapshot at `key` most recently used. `false` when
    /// there is none.
    pub(crate) fn refresh(&mut self, key: (i32, i32)) -> bool {
        if !self.map.contains_key(&key) {
            return false;
        }
        self.order.retain(|k| *k != key);
        self.order.push_back(key);
        true
    }

    /// Remove and return the snapshot at `key`, if any.
    ///
    /// The mirror of [`Self::get`]: llama.cpp takes + re-inserts around
    /// its `&mut self` FFI call, so this is llama.cpp-only and dead in a
    /// moeflux-only build — hence the cfg on the lint, pointing the
    /// opposite way to `get`'s.
    #[cfg_attr(not(feature = "llama-cpp"), allow(dead_code))]
    pub(crate) fn take(&mut self, key: (i32, i32)) -> Option<Vec<u8>> {
        let bytes = self.map.remove(&key)?;
        self.order.retain(|k| *k != key);
        Some(bytes)
    }

    /// Drop the snapshot at `key`. Idempotent.
    pub(crate) fn forget(&mut self, key: (i32, i32)) {
        if self.map.remove(&key).is_some() {
            self.order.retain(|k| *k != key);
        }
    }

    /// Drop every snapshot on `seq_id` at positions strictly greater
    /// than `pos` — the "futures are invalid after a rewind" rule from
    /// [`crate::backend::Decoder::restore_to`].
    pub(crate) fn invalidate_after(&mut self, seq_id: i32, pos: i32) {
        self.retain(|s, p| s != seq_id || p <= pos);
    }

    /// Keep only the snapshots whose `(seq_id, pos)` satisfies `keep`.
    pub(crate) fn retain(&mut self, mut keep: impl FnMut(i32, i32) -> bool) {
        self.map.retain(|&(s, p), _| keep(s, p));
        self.order.retain(|&(s, p)| keep(s, p));
    }

    /// Whether a snapshot is stored at `key`. Does not touch the LRU
    /// order — asking is not a use. Read by llama.cpp's media-tip
    /// bookkeeping, so it is dead in a moeflux-only build.
    #[cfg_attr(not(any(test, feature = "llama-cpp")), allow(dead_code))]
    pub(crate) fn contains(&self, key: (i32, i32)) -> bool {
        self.map.contains_key(&key)
    }

    /// Drop everything.
    pub(crate) fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
    }

    /// Number of live snapshots. Read only by llama.cpp's
    /// `checkpoint_count`, so it is dead in a moeflux-only build.
    #[cfg_attr(not(feature = "llama-cpp"), allow(dead_code))]
    pub(crate) fn len(&self) -> usize {
        self.map.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store_with(keys: &[(i32, i32)]) -> SnapshotStore {
        let mut s = SnapshotStore::default();
        for &k in keys {
            s.insert(k, vec![0u8; 4]);
        }
        s
    }

    #[test]
    fn snapshot_store_insert_take_forget_roundtrip() {
        let mut s = SnapshotStore::default();
        s.insert((0, 10), vec![1, 2, 3]);
        assert_eq!(s.len(), 1);
        assert_eq!(s.take((0, 10)), Some(vec![1, 2, 3]));
        assert_eq!(s.len(), 0);
        assert_eq!(s.take((0, 10)), None);
        // forget is idempotent on absent keys.
        s.forget((0, 10));
        s.forget((0, 10));
    }

    #[test]
    fn snapshot_store_replace_refreshes_lru_position() {
        // Fill to capacity, then re-insert the oldest key. The next
        // eviction must hit the second-oldest, not the refreshed one.
        let keys: Vec<(i32, i32)> =
            (0..MAX_SEQ_SNAPSHOTS as i32).map(|i| (0, i)).collect();
        let mut s = store_with(&keys);
        s.insert((0, 0), vec![9]); // refresh oldest
        s.insert((0, 999), vec![8]); // overflow → evict (0, 1)
        assert_eq!(s.len(), MAX_SEQ_SNAPSHOTS);
        assert_eq!(s.take((0, 1)), None, "second-oldest should be evicted");
        assert_eq!(s.take((0, 0)), Some(vec![9]), "refreshed key survives");
    }

    #[test]
    fn snapshot_store_get_refreshes_lru_position() {
        // Reading a snapshot marks it recently-used: fill to cap, get
        // the oldest, overflow — the eviction must skip the read key.
        let keys: Vec<(i32, i32)> =
            (0..MAX_SEQ_SNAPSHOTS as i32).map(|i| (0, i)).collect();
        let mut s = store_with(&keys);
        assert!(s.get((0, 0)).is_some());
        s.insert((0, 999), vec![8]); // overflow → evict (0, 1)
        assert_eq!(s.take((0, 1)), None, "second-oldest should be evicted");
        assert!(s.take((0, 0)).is_some(), "read key survives");
    }

    #[test]
    fn snapshot_store_evicts_oldest_beyond_cap() {
        let keys: Vec<(i32, i32)> = (0..(MAX_SEQ_SNAPSHOTS as i32 + 3))
            .map(|i| (0, i))
            .collect();
        let mut s = store_with(&keys);
        assert_eq!(s.len(), MAX_SEQ_SNAPSHOTS);
        // The three oldest past the sequence's first anchor are gone;
        // that anchor and the newest three are present.
        assert_eq!(s.take((0, 1)), None);
        assert_eq!(s.take((0, 2)), None);
        assert_eq!(s.take((0, 3)), None);
        assert!(s.take((0, 0)).is_some(), "the first anchor goes last");
        assert!(s.take((0, MAX_SEQ_SNAPSHOTS as i32 + 2)).is_some());
    }

    /// A full set of anchors on every slot fits: one slot's newest
    /// snapshot never evicts another slot's system anchor. Four slots
    /// of a hybrid model (the live Qwen3.6 `--cache-slots 4` setup)
    /// used to sit exactly at the flat cap of 16.
    #[test]
    fn snapshot_store_cap_holds_every_slots_anchors() {
        assert_eq!(cap_for_sequences(0), MAX_SEQ_SNAPSHOTS);
        assert_eq!(cap_for_sequences(1), MAX_SEQ_SNAPSHOTS);
        assert_eq!(cap_for_sequences(4), 4 * SNAPSHOTS_PER_SLOT);
        assert!(cap_for_sequences(4) > MAX_SEQ_SNAPSHOTS);

        let slots = 4;
        let mut s = SnapshotStore::with_cap(cap_for_sequences(slots));
        // Every slot: its system anchor first, then the other
        // breakpoints and both tips, round-robin as agents interleave.
        for pos in 1..=SNAPSHOTS_PER_SLOT as i32 {
            for seq in 0..slots as i32 {
                s.insert((seq, pos * 100), vec![0u8; 4]);
            }
        }
        assert_eq!(s.len(), slots * SNAPSHOTS_PER_SLOT);
        for seq in 0..slots as i32 {
            assert!(s.get((seq, 100)).is_some(), "seq {seq} lost its anchor");
        }
    }

    #[test]
    fn snapshot_store_invalidate_after_is_per_sequence() {
        let mut s = store_with(&[(0, 5), (0, 10), (0, 20), (1, 15)]);
        s.invalidate_after(0, 10);
        // (0, 20) dropped: strictly greater than pos on seq 0.
        assert_eq!(s.take((0, 20)), None);
        // (0, 10) kept: boundary is inclusive.
        assert!(s.take((0, 10)).is_some());
        assert!(s.take((0, 5)).is_some());
        // Other sequences untouched.
        assert!(s.take((1, 15)).is_some());
    }

    #[test]
    fn snapshot_store_retain_filters_map_and_order_together() {
        let mut s = store_with(&[(0, 5), (1, 5), (1, 9), (2, 1)]);
        s.retain(|seq, _| seq == 1);
        assert_eq!(s.len(), 2);
        assert!(!s.contains((0, 5)) && !s.contains((2, 1)));
        assert!(s.contains((1, 5)) && s.contains((1, 9)));
        // The LRU order lost the same keys: filling to the cap evicts
        // only what is still stored, oldest first.
        for pos in 100..(100 + MAX_SEQ_SNAPSHOTS as i32 - 1) {
            s.insert((3, pos), vec![0]);
        }
        assert_eq!(s.len(), MAX_SEQ_SNAPSHOTS);
        assert!(
            !s.contains((1, 9)),
            "the oldest survivor but a first anchor"
        );
        assert!(s.contains((1, 5)), "seq 1's first anchor");
    }

    /// The byte total evicts least recently used first, across
    /// sequences: Gemma 4's ≈ 800 MiB checkpoints would otherwise fill
    /// a count cap of 24 with ≈ 19 GiB.
    #[test]
    fn snapshot_store_evicts_by_bytes_lru_first() {
        let mut s = SnapshotStore::with_cap(64);
        s.set_byte_limits(100, 100);
        assert!(s.insert((0, 1), vec![0; 40]));
        assert!(s.insert((1, 1), vec![0; 40]));
        assert!(s.get((0, 1)).is_some(), "a read is a use");
        assert!(s.insert((2, 1), vec![0; 40]));
        assert!(!s.contains((1, 1)), "the least recently used goes");
        assert!(s.contains((0, 1)) && s.contains((2, 1)));
        assert_eq!(s.bytes(), 80);
    }

    /// One slot over its share evicts only its own snapshots — never
    /// another slot's system anchor, nor its own while another is left.
    #[test]
    fn snapshot_store_seq_budget_spares_other_slots() {
        let mut s = SnapshotStore::with_cap(64);
        s.set_byte_limits(1000, 100);
        s.insert((1, 1), vec![0; 50]);
        for pos in 1..=3 {
            s.insert((0, pos), vec![0; 40]);
        }
        assert!(!s.contains((0, 2)), "seq 0's oldest past its first");
        assert!(s.contains((0, 1)) && s.contains((0, 3)));
        assert!(s.contains((1, 1)), "older, but another slot's");
        assert_eq!(s.seq_bytes(0), 80);
    }

    /// The live Gemma 4 shape: a slot's byte share holds one fewer
    /// checkpoint than a full set of anchors, and restores refresh only
    /// the anchor they land on. The first anchor survives every later
    /// one the slot takes; with no other left, it goes too.
    #[test]
    fn snapshot_store_keeps_each_slots_first_anchor_last() {
        let mut s = SnapshotStore::with_cap(64);
        s.set_byte_limits(1000, 5 * 10);
        for pos in 1..=SNAPSHOTS_PER_SLOT as i32 + 3 {
            s.insert((0, pos * 100), vec![0; 10]);
            // A restore to the latest anchor refreshes it alone.
            s.refresh((0, pos * 100));
        }
        assert!(s.contains((0, 100)), "the system anchor");
        assert_eq!(s.seq_bytes(0), 50);
        s.set_byte_limits(1000, 10);
        assert_eq!(s.len(), 1);
        assert!(s.contains((0, 100)), "the last to go");
        s.set_byte_limits(1000, 0);
        assert_eq!(s.len(), 0, "and it goes when nothing else is left");
    }

    /// A snapshot over a limit on its own is refused, and costs the
    /// others nothing; replacing a key never counts its old bytes.
    #[test]
    fn snapshot_store_refuses_what_cannot_fit() {
        let mut s = SnapshotStore::with_cap(64);
        s.set_byte_limits(100, 60);
        s.insert((0, 1), vec![0; 50]);
        assert!(!s.insert((1, 1), vec![0; 61]), "over the per-slot limit");
        assert!(s.contains((0, 1)));
        assert!(s.insert((0, 1), vec![0; 60]), "a replacement fits");
        assert_eq!((s.len(), s.bytes()), (1, 60));
    }

    /// Lowering the limits evicts down to them at once — each slot's
    /// first anchor last.
    #[test]
    fn snapshot_store_lowered_limits_apply_now() {
        let mut s = store_with(&[(0, 1), (0, 2), (1, 1), (1, 2)]);
        s.set_byte_limits(usize::MAX, 4);
        assert!(!s.contains((0, 2)) && !s.contains((1, 2)));
        s.set_byte_limits(4, 4);
        assert_eq!(s.len(), 1);
        assert!(s.contains((1, 1)), "the most recent first anchor survives");
    }

    #[test]
    fn snapshot_store_clear_empties_map_and_order() {
        let mut s = store_with(&[(0, 1), (0, 2)]);
        s.clear();
        assert_eq!(s.len(), 0);
        // Insert after clear must not resurrect stale order entries.
        s.insert((0, 3), vec![1]);
        assert_eq!(s.len(), 1);
        assert!(s.take((0, 3)).is_some());
    }
}
