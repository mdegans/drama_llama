//! Bounded LRU of serialized per-sequence decoder states, keyed by
//! `(seq_id, pos)`. Pure bookkeeping — FFI-free, so the eviction /
//! invalidation logic is unit-testable without a model. Shared by the
//! llama.cpp decoder (recurrent / hybrid models, `get_state_seq`
//! blobs) and the moeflux decoder (whole-`Ctx` `state_save` blobs
//! realizing `(seq, pos)` keying over a physically single stream).

use std::collections::{HashMap, VecDeque};

/// Default cap on retained snapshots. Anthropic's cache budget is 4
/// explicit breakpoints; `Session` adds one internal tip per cache
/// slot. 16 leaves generous slack for multi-sequence use before the
/// LRU starts evicting.
pub(crate) const MAX_SEQ_SNAPSHOTS: usize = 16;

/// Bounded LRU of serialized decoder states.
///
/// A stored value is the *entire* serialized state for its key's
/// sequence, taken when that sequence held exactly positions
/// `[0, pos)`. Restoring one replaces the sequence wholesale, so
/// entries stay restorable regardless of later KV mutations;
/// invalidation exists to keep the `Decoder` trait's rewind semantics
/// ("futures are dropped") uniform across backends, not because the
/// bytes go stale.
#[derive(Debug)]
pub(crate) struct SnapshotStore {
    map: HashMap<(i32, i32), Vec<u8>>,
    /// Insertion order, oldest first. Re-inserting an existing key
    /// refreshes its position.
    order: VecDeque<(i32, i32)>,
    /// Eviction cap. See [`MAX_SEQ_SNAPSHOTS`].
    cap: usize,
}

impl Default for SnapshotStore {
    fn default() -> Self {
        Self {
            map: HashMap::new(),
            order: VecDeque::new(),
            cap: MAX_SEQ_SNAPSHOTS,
        }
    }
}

impl SnapshotStore {
    /// Insert (or replace) the snapshot at `key`, evicting the oldest
    /// entries beyond the cap.
    pub(crate) fn insert(&mut self, key: (i32, i32), bytes: Vec<u8>) {
        if self.map.insert(key, bytes).is_some() {
            self.order.retain(|k| *k != key);
        }
        self.order.push_back(key);
        while self.map.len() > self.cap {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            self.map.remove(&oldest);
        }
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
        if self.map.contains_key(&key) {
            self.order.retain(|k| *k != key);
            self.order.push_back(key);
        }
        self.map.get(&key)
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
        self.map.retain(|&(s, p), _| s != seq_id || p <= pos);
        self.order.retain(|&(s, p)| s != seq_id || p <= pos);
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
        // The three oldest are gone; the newest three are present.
        assert_eq!(s.take((0, 0)), None);
        assert_eq!(s.take((0, 1)), None);
        assert_eq!(s.take((0, 2)), None);
        assert!(s.take((0, MAX_SEQ_SNAPSHOTS as i32 + 2)).is_some());
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
