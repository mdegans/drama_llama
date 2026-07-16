use core::slice;
use std::{
    collections::{BTreeMap, VecDeque},
    ops::Index,
};
use tinyvec::ArrayVec;

use crate::{utils::cold, Token};

#[derive(Debug, thiserror::Error)]
pub enum NGramNewError {
    #[error("Need at least one token to create an NGram.")]
    NotEnoughTokens,
    #[error("Number of tokens exceeds NGram::CAPACITY.")]
    TooManyTokens,
}

static_assertions::assert_impl_all!(NGramNewError: Send, Sync);

/// An immutable N-gram of tokens.
///
/// # Note
/// * The minimum length of an Ngram is 1 (a unigram).
/// * The capacity of the Ngram is fixed to [`NGram::CAPACITY`].
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Hash, Ord)]
#[repr(transparent)]
pub struct NGram {
    data: ArrayVec<[Token; Self::CAPACITY]>,
}

static_assertions::assert_impl_all!(NGram: Send, Sync);

impl NGram {
    pub const CAPACITY: usize = 7;

    /// Create an Ngram from a slice of [`Token`]. This can fail in cases
    /// where there are either no tokens or `window.len()` >
    /// [`NGram::CAPACITY`].
    pub fn try_from_tokens(window: &[Token]) -> Result<Self, NGramNewError> {
        if window.is_empty() {
            cold(); // this branch is unlikely
            return Err(NGramNewError::NotEnoughTokens);
        }

        let mut data = ArrayVec::new();
        for &token in window {
            match data.try_push(token) {
                None => {} // successful insert
                Some(_) => {
                    cold();
                    return Err(NGramNewError::TooManyTokens);
                } // full
            };
        }

        let new = Self { data };

        debug_assert_eq!(new.len(), window.len().min(Self::CAPACITY));

        Ok(new)
    }

    /// Iterate over the tokens in the Ngram.
    pub fn iter(&self) -> slice::Iter<'_, Token> {
        self.data.iter()
    }

    /// Get a slice of the tokens in the Ngram.
    pub fn as_slice(&self) -> &[Token] {
        self.data.as_slice()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub const fn capacity(&self) -> usize {
        Self::CAPACITY
    }
}

impl TryFrom<&[Token]> for NGram {
    type Error = NGramNewError;

    fn try_from(window: &[Token]) -> Result<Self, Self::Error> {
        Self::try_from_tokens(window)
    }
}

impl From<Token> for NGram {
    fn from(token: Token) -> Self {
        // Unwrap can never panic because the only two failure conditions for
        // construction are when the len is 0 or > CAPACITY. We are creating a
        // unigram.
        Self::try_from_tokens(&[token]).unwrap()
    }
}

impl Index<usize> for NGram {
    type Output = Token;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
/// Metadata about an Ngram.
///
/// Invariant: `count == positions.len()`. Both grow on
/// [`NGramStats::add`] and shrink on
/// [`NGramStats::evict_outside_window`], [`NGramStats::remove_one`],
/// or [`NGramStats::remove_every`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NGramData {
    /// Currently-tracked occurrences of the associated [`NGram`].
    /// Equals `positions.len()`.
    count: usize,
    /// Absolute generation step of each currently-tracked occurrence,
    /// in insertion order. Used by windowed-decay penalty math.
    /// Serialized: a restored snapshot must continue the exact same
    /// windowed-decay math (`count == positions.len()` invariant and
    /// `windowed_decayed_count` both depend on it).
    positions: VecDeque<u64>,
    /// Weight of the Ngram. This can be used for anything.
    weight: f64,
}

impl NGramData {
    pub const fn count(&self) -> usize {
        self.count
    }

    pub const fn weight(&self) -> f64 {
        self.weight
    }

    /// Sum of `decay^(current_step - position)` over every currently-tracked
    /// occurrence. Bounded above by `1 / (1 - decay)` for sustained
    /// repetition with `decay < 1.0` — the structural fix that keeps the
    /// repetition-penalty additive term from diverging on long generations.
    ///
    /// For windowed semantics, call
    /// [`NGramStats::evict_outside_window`] beforehand so older positions
    /// have already been dropped.
    pub fn windowed_decayed_count(&self, current_step: u64, decay: f32) -> f32 {
        self.positions
            .iter()
            .map(|&p| {
                let age = current_step.saturating_sub(p);
                // i32::MAX is comfortably more than any reasonable window;
                // huge ages decay to ~0 anyway.
                decay.powi(age.min(i32::MAX as u64) as i32)
            })
            .sum()
    }

    /// Add (or subtract) a weight from this Ngram.
    ///
    /// * It is guaranteed that the weight will never be NaN or infinite in
    ///   either bound.
    pub fn add_weight(&mut self, weight: f64) {
        let old = self.weight;
        self.weight += weight;
        if self.weight.is_infinite() || self.weight.is_nan() {
            cold(); // this branch is unlikely
                    // Copilot suggested it, and it's a good idea, since these
                    // checks are likely to happen in a very tight loop, but
                    // they are necessary because it only takes two float values
                    // to create Inf or NaN. In our codebase it would be a
                    // programmer error for this to happen but we are going to
                    // expose this to the public API, so we need to ensure that
                    // it's not possible to create an invalid weight.
            self.weight = old;
        }
    }

    /// Multiply the weight by a factor.
    ///
    /// * It is guaranteed that the weight will never be NaN or infinite in
    ///   either bound.
    pub fn mul_weight(&mut self, factor: f64) {
        let old = self.weight;
        self.weight *= factor;
        if self.weight.is_infinite() || self.weight.is_nan() {
            cold(); // this branch is unlikely
            self.weight = old;
        }
    }
}

/// A map of [`NGram`] metadata.
///
/// It keeps track of:
/// * The total number of [`NGram`]s that have been added.
/// * The total number of tokens that have been added.
/// * [`NGramData`] for each [`NGram`] including count, positions, and weight.
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct NGramStats {
    // BTreeMap, not HashMap: both penalty passes iterate this map and
    // accumulate additive float penalties, so iteration order must be
    // deterministic for a serialized-then-restored state to continue the
    // exact same stream. It also makes serialized blobs canonical.
    // On the wire this is a sequence of (NGram, NGramData) pairs —
    // human-readable formats (JSON, TOML) reject non-string map keys.
    #[cfg_attr(feature = "serde", serde(with = "data_serde"))]
    data: BTreeMap<NGram, NGramData>,
    ngram_count: usize,
    token_count: usize,
}

/// `NGramStats::data` as a pair-sequence on the wire. BTreeMap
/// iteration keeps the emitted sequence canonical.
#[cfg(feature = "serde")]
mod data_serde {
    use super::*;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(
        map: &BTreeMap<NGram, NGramData>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(map.iter())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<BTreeMap<NGram, NGramData>, D::Error> {
        Ok(Vec::<(NGram, NGramData)>::deserialize(deserializer)?
            .into_iter()
            .collect())
    }
}

impl NGramStats {
    /// Create a new, empty NGramStats.
    pub fn new() -> Self {
        NGramStats {
            data: BTreeMap::new(),
            ngram_count: 0,
            token_count: 0,
        }
    }

    /// Add an [`NGram`] observed at `current_step`, updating the
    /// [`NGramData`]'s count and position queue. Returns a mutable
    /// reference to the updated [`NGramData`].
    ///
    /// `current_step` is the absolute generation step at which the
    /// occurrence happened; it feeds windowed-decay penalty math via
    /// [`NGramData::windowed_decayed_count`] and
    /// [`NGramStats::evict_outside_window`].
    pub fn add(&mut self, key: NGram, current_step: u64) -> &mut NGramData {
        self.ngram_count += 1;
        self.token_count += key.len();

        let entry = self.data.entry(key).or_insert(NGramData::default());
        entry.count += 1;
        entry.positions.push_back(current_step);

        entry
    }

    /// Drop every recorded position older than
    /// `current_step - window_size` from every tracked n-gram. Entries
    /// whose position queue empties are removed entirely. Maintains the
    /// `count == positions.len()` invariant on each remaining entry and
    /// keeps `total_ngram_count` / `total_token_count` consistent with
    /// the eviction.
    ///
    /// Cheap when the window holds steady — each call only pops the few
    /// front positions that just rolled out, not the full queue. Pair
    /// with [`NGramData::windowed_decayed_count`] for the bounded-additive
    /// penalty path.
    pub fn evict_outside_window(
        &mut self,
        current_step: u64,
        window_size: u32,
    ) {
        let cutoff = current_step.saturating_sub(window_size as u64);
        let stats_ngram_count = &mut self.ngram_count;
        let stats_token_count = &mut self.token_count;

        self.data.retain(|ngram, data| {
            while let Some(&front) = data.positions.front() {
                if front < cutoff {
                    data.positions.pop_front();
                    data.count -= 1;
                    *stats_ngram_count -= 1;
                    *stats_token_count -= ngram.len();
                } else {
                    break;
                }
            }
            !data.positions.is_empty()
        });
    }

    /// Get data about an Ngram.
    pub fn get(&self, key: &NGram) -> Option<&NGramData> {
        self.data.get(key)
    }

    /// Remove a single instance of a particular [`NGram`] from the map. Returns
    /// the current data associated with the [`NGram`] if it was present or None
    /// in cases where the [`NGram`] was not present.
    pub fn remove_one(&mut self, key: &NGram) -> Option<NGramData> {
        if let Some(data) = self.data.get_mut(key) {
            self.ngram_count -= 1;
            self.token_count -= key.len();

            data.count -= 1;
            // Pop the oldest recorded position to keep
            // `count == positions.len()`.
            data.positions.pop_front();

            if data.count == 0 {
                self.data.remove(key)
            } else {
                Some(data.clone())
            }
        } else {
            None
        }
    }

    /// Remove every one of a particualar [`NGram`] from the map. Returns the
    /// data associated with the [`NGram`] if it was present (as of removal).
    pub fn remove_every(&mut self, key: &NGram) -> Option<NGramData> {
        if let Some(data) = self.data.remove(key) {
            self.ngram_count -= data.count;
            self.token_count -= key.len() * data.count;

            Some(data)
        } else {
            None
        }
    }

    /// Get the total count of all Ngrams ever added
    pub fn total_ngram_count(&self) -> usize {
        self.ngram_count
    }

    /// Get the total count of all tokens ever added
    pub fn total_token_count(&self) -> usize {
        self.token_count
    }

    /// Average ngram length.
    pub fn avg_ngram_length(&self) -> f64 {
        self.token_count as f64 / self.ngram_count as f64
    }

    /// Iterate over all Ngrams and their data.
    pub fn iter(&self) -> impl Iterator<Item = (&NGram, &NGramData)> {
        self.data.iter()
    }

    /// Iterate over ngrams where the count is greater than or equal to the
    /// given value.
    pub fn iter_count_ge(
        &self,
        count: usize,
    ) -> impl Iterator<Item = (&NGram, &NGramData)> {
        self.iter().filter(move |(_, data)| data.count >= count)
    }
}

impl Into<BTreeMap<NGram, NGramData>> for NGramStats {
    fn into(self) -> BTreeMap<NGram, NGramData> {
        self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// Test NGram::new, len and capacity
    fn test_ngram_from_tokens() {
        let err = NGram::try_from_tokens(&[]);
        assert!(matches!(err, Err(NGramNewError::NotEnoughTokens)));
        for i in 1..=NGram::CAPACITY {
            let tokens = (1..=i as i32).collect::<Vec<_>>();
            let ngram = NGram::try_from_tokens(&tokens).unwrap();
            assert_eq!(ngram.len(), i);
            assert_eq!(ngram.capacity(), NGram::CAPACITY);
        }
        let err = NGram::try_from_tokens(&[1; NGram::CAPACITY + 1]);
        assert!(matches!(err, Err(NGramNewError::TooManyTokens)));
    }

    #[test]
    fn test_ngram_iter() {
        let ngram = NGram::try_from_tokens(&[1, 2, 3, 4, 5, 6, 7]).unwrap();
        let mut iter = ngram.iter();
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), Some(&4));
        assert_eq!(iter.next(), Some(&5));
        assert_eq!(iter.next(), Some(&6));
        assert_eq!(iter.next(), Some(&7));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_ngram_as_slice() {
        let ngram = NGram::try_from_tokens(&[1, 2, 3, 4, 5, 6, 7]).unwrap();
        assert_eq!(ngram.as_slice(), &[1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_ngram_data() {
        let mut data = NGramData::default();
        assert_eq!(data.count(), 0);
        assert_eq!(data.weight(), 0.0);
        data.add_weight(1.0);
        assert_approx_eq!(data.weight(), 1.0, 1e-6);
        data.add_weight(-0.5);
        assert_approx_eq!(data.weight(), 0.5, 1e-6);
        data.mul_weight(2.0);
        assert_approx_eq!(data.weight(), 1.0, 1e-6);
        data.add_weight(f64::INFINITY);
        assert_approx_eq!(data.weight(), 1.0, 1e-6);
        data.add_weight(f64::NEG_INFINITY);
        assert_approx_eq!(data.weight(), 1.0, 1e-6);
        data.add_weight(f64::NAN);
        assert_approx_eq!(data.weight(), 1.0, 1e-6);
    }

    /// A serde round-trip must reconstruct the stats exactly —
    /// including `positions`, which the windowed-decay penalty math
    /// reads. Derived `PartialEq` is the comparator, same as the
    /// eventual SamplerState bit-exact test.
    #[cfg(feature = "serde")]
    #[test]
    fn test_ngram_stats_serde_round_trip() {
        let mut stats = NGramStats::new();

        for step in 0..8u64 {
            let ngram = NGram::try_from_tokens(&[step as Token, 1, 2]).unwrap();
            stats.add(ngram, step);
        }
        stats.evict_outside_window(8, 4);

        let json = serde_json::to_string(&stats).unwrap();
        let restored: NGramStats = serde_json::from_str(&json).unwrap();
        assert_eq!(stats, restored);
        // And a second serialization is byte-identical (canonical blobs
        // — BTreeMap ordering).
        assert_eq!(json, serde_json::to_string(&restored).unwrap());
    }

    #[test]
    fn test_ngram_stats() {
        let mut stats = NGramStats::new();
        let a = NGram::try_from_tokens(&[0, 1, 2, 3, 4, 5, 6]).unwrap();
        let b = NGram::try_from_tokens(&[0, 1, 2, 3, 4]).unwrap();

        let count = stats.add(a, 0).count;
        assert_eq!(count, 1);
        assert_eq!(stats.total_ngram_count(), 1);
        assert_eq!(stats.total_token_count(), 7);
        assert_approx_eq!(stats.avg_ngram_length(), 7.0, 1e-6);

        let count = stats.add(b, 0).count;
        assert_eq!(count, 1);
        assert_eq!(stats.total_ngram_count(), 2);
        assert_eq!(stats.total_token_count(), 12);
        assert_approx_eq!(stats.avg_ngram_length(), 6.0, 1e-6);

        let count = stats.add(b, 0).count;
        assert_eq!(count, 2);
        assert_eq!(stats.total_ngram_count(), 3);
        assert_eq!(stats.total_token_count(), 17);
        assert_approx_eq!(stats.avg_ngram_length(), 5.666666666666667, 1e-6);

        let data = stats.remove_every(&b).unwrap();
        assert_eq!(data.count(), 2);
        assert_eq!(data.weight(), 0.0);
        assert_eq!(stats.total_ngram_count(), 1);
        assert_eq!(stats.total_token_count(), 7);
        assert_approx_eq!(stats.avg_ngram_length(), 7.0, 1e-6);

        let _ = stats.add(b, 0);
        let count = stats.add(b, 0).count;
        assert_eq!(count, 2);
        assert_eq!(stats.total_ngram_count(), 3);
        assert_eq!(stats.total_token_count(), 17);
        assert_approx_eq!(stats.avg_ngram_length(), 5.666666666666667, 1e-6);

        let data = stats.remove_one(&b).unwrap();
        assert_eq!(data.count, 1);
        assert_eq!(stats.total_ngram_count(), 2);
        assert_eq!(stats.total_token_count(), 12);
        assert_approx_eq!(stats.avg_ngram_length(), 6.0, 1e-6);
    }

    /// Cover the `count == positions.len()` invariant under the
    /// windowed-decay path (`add` records position, `evict_outside_window`
    /// drops old positions, `remove_one` decrements both).
    #[test]
    fn ngram_stats_windowed_decay_invariants() {
        let mut stats = NGramStats::new();
        let g = NGram::from(7 as Token);

        // Add the same unigram at three different generation steps.
        stats.add(g, 10);
        stats.add(g, 20);
        stats.add(g, 30);
        assert_eq!(stats.get(&g).unwrap().count(), 3);

        // Effective count at step 30 with decay 1.0 (no decay) is just
        // the number of in-window occurrences.
        let eff = stats.get(&g).unwrap().windowed_decayed_count(30, 1.0);
        assert_approx_eq!(eff, 3.0, 1e-6);

        // With decay 0.9 and ages [20, 10, 0], effective count is
        // 0.9^20 + 0.9^10 + 1.0.
        let expected = 0.9_f32.powi(20) + 0.9_f32.powi(10) + 1.0;
        let eff = stats.get(&g).unwrap().windowed_decayed_count(30, 0.9);
        assert_approx_eq!(eff, expected, 1e-5);

        // Evict positions older than `current_step - window_size`.
        // window=15 at step 30 keeps positions >= 15, so [20, 30] stay
        // and [10] is dropped.
        stats.evict_outside_window(30, 15);
        let data = stats.get(&g).unwrap();
        assert_eq!(data.count(), 2, "count should match remaining positions");
        assert_eq!(stats.total_ngram_count(), 2);
        assert_eq!(stats.total_token_count(), 2);

        // Far-future eviction empties the entry entirely.
        stats.evict_outside_window(1_000_000, 100);
        assert!(stats.get(&g).is_none());
        assert_eq!(stats.total_ngram_count(), 0);
        assert_eq!(stats.total_token_count(), 0);
    }
}
