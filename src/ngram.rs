use core::slice;
use llama_cpp_sys::llama_token;
use std::{collections::HashMap, ops::Index};
use tinyvec::ArrayVec;

use crate::{candidates::Sorted, utils::cold, Candidates};

#[derive(Debug, thiserror::Error)]
pub enum NGramNewError {
    #[error("Need at least one token to create an NGram.")]
    NotEnoughTokens,
    #[error("Number of tokens exceeds NGram::CAPACITY.")]
    TooManyTokens,
}

/// An immutable N-gram of tokens.
///
/// # Note
/// * The minimum length of an Ngram is 1 (a unigram).
/// * The capacity of the Ngram is fixed to [`NGram::CAPACITY`].
#[cfg_attr(
    feature = "serde",
    derive(rocket::serde::Deserialize, rocket::serde::Serialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "rocket::serde"))]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Hash, Ord)]
#[repr(transparent)]
pub struct NGram {
    data: ArrayVec<[llama_token; Self::CAPACITY]>,
}

impl NGram {
    pub const CAPACITY: usize = 7;

    /// Create an Ngram from a slice of [`llama_token`]. This can fail in cases
    /// where there are either no tokens or `window.len()` >
    /// [`NGram::CAPACITY`].
    pub fn try_from_tokens(
        window: &[llama_token],
    ) -> Result<Self, NGramNewError> {
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
    pub fn iter(&self) -> slice::Iter<'_, llama_token> {
        self.data.iter()
    }

    /// Get a slice of the tokens in the Ngram.
    pub fn as_slice(&self) -> &[llama_token] {
        self.data.as_slice()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub const fn capacity(&self) -> usize {
        Self::CAPACITY
    }
}

impl TryFrom<&[llama_token]> for NGram {
    type Error = NGramNewError;

    fn try_from(window: &[llama_token]) -> Result<Self, Self::Error> {
        Self::try_from_tokens(window)
    }
}

impl From<llama_token> for NGram {
    fn from(token: llama_token) -> Self {
        // Unwrap can never panic because the only two failure conditions for
        // construction are when the len is 0 or > CAPACITY. We are creating a
        // unigram.
        Self::try_from_tokens(&[token]).unwrap()
    }
}

impl Index<usize> for NGram {
    type Output = llama_token;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

#[cfg_attr(
    feature = "serde",
    derive(rocket::serde::Deserialize, rocket::serde::Serialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "rocket::serde"))]
/// Metadata about an Ngram.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct NGramData {
    /// How many times the associated [`NGram`] has been added to the
    /// [`NGramStats`]
    count: usize,
    /// The sum of the probabilities of the individual tokens in the Ngram.
    cum_prob: f64,
    /// Weight of the Ngram. This can be used for anything.
    weight: f64,
}

impl NGramData {
    pub const fn count(&self) -> usize {
        self.count
    }

    pub const fn cum_prob(&self) -> f64 {
        self.cum_prob
    }

    pub fn avg_cum_prob(&self) -> f64 {
        self.cum_prob / self.count as f64
    }

    pub const fn weight(&self) -> f64 {
        self.weight
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
/// * [`NGramData`] for each [`NGram`] including count, cum_prob, and weight.
#[cfg_attr(feature = "serde", derive(rocket::serde::Serialize))]
// TODO: Implement Deserialize
#[cfg_attr(feature = "serde", serde(crate = "rocket::serde"))]
#[derive(Debug)]
pub struct NGramStats {
    data: HashMap<NGram, NGramData>,
    ngram_count: usize,
    token_count: usize,
}

impl NGramStats {
    /// Create a new, empty NGramStats.
    pub fn new() -> Self {
        NGramStats {
            data: HashMap::new(),
            ngram_count: 0,
            token_count: 0,
        }
    }

    /// Add an [`NGram`] to self, updating the [`NGramData`]'s count and
    /// cum_prob. Returns a mutable reference to the updated [`NGramData`].
    ///
    /// # Note
    /// * This function applies softmax to the candidates if it hasn't been
    ///   applied already.
    /// * This function sorts the candidates by id.
    pub fn add(
        &mut self,
        key: NGram,
        candidates: &mut Candidates,
    ) -> &mut NGramData {
        // This only applies softmax if it hasn't been applied already, like on
        // the first call.
        candidates.apply_softmax(None);

        // This doesn't invalidate the softmax state, and is not applied to the
        // candidates if it has already been applied.
        candidates.sort(Sorted::ById {
            k: candidates.len(),
        });

        self.ngram_count += 1;
        self.token_count += key.len();

        let entry = self.data.entry(key).or_insert(NGramData::default());
        entry.count += 1;
        // Accumulated probability of the NGram
        let cum_prob: f64 = key
            .iter()
            .map(|&token| candidates[token.abs() as usize].p as f64)
            .sum();

        entry.cum_prob += cum_prob;

        entry
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

            data.cum_prob -= data.avg_cum_prob();
            data.count -= 1;

            if data.count == 0 {
                self.data.remove(key)
            } else {
                Some(*data)
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

impl Into<HashMap<NGram, NGramData>> for NGramStats {
    fn into(self) -> HashMap<NGram, NGramData> {
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
        assert_eq!(data.cum_prob(), 0.0);
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

    #[test]
    fn test_ngram_stats() {
        let n: usize = 10;
        let mut stats = NGramStats::new();
        let a = NGram::try_from_tokens(&[0, 1, 2, 3, 4, 5, 6]).unwrap();
        let b = NGram::try_from_tokens(&[0, 1, 2, 3, 4]).unwrap();

        let mut c = Candidates::new(n).unwrap();
        for i in 0..n {
            c.data[i].id = i as llama_token;
            c.data[i].logit = -(i as f32 / n as f32);
        }
        c.apply_softmax(None);
        c.sort(Sorted::ById { k: c.len() });

        let data = stats.add(a, &mut c);
        assert_eq!(data.count(), 1);
        assert_eq!(stats.total_ngram_count(), 1);
        assert_eq!(stats.total_token_count(), 7);
        assert_approx_eq!(stats.avg_ngram_length(), 7.0, 1e-6);
        let expected_cum_prob: f64 = c.iter().take(7).map(|t| t.p as f64).sum();
        assert_approx_eq!(
            stats.get(&a).unwrap().cum_prob(),
            expected_cum_prob,
            1e-6
        );
        assert_approx_eq!(
            stats.get(&a).unwrap().avg_cum_prob(),
            expected_cum_prob,
            1e-6
        );

        let data = stats.add(b, &mut c);
        assert_eq!(data.count(), 1);
        assert_eq!(stats.total_ngram_count(), 2);
        assert_eq!(stats.total_token_count(), 12);
        let expected_cum_prob: f64 = c.iter().take(5).map(|t| t.p as f64).sum();
        assert_approx_eq!(
            stats.get(&b).unwrap().cum_prob(),
            expected_cum_prob,
            1e-6
        );
        assert_approx_eq!(stats.avg_ngram_length(), 6.0, 1e-6);

        let count = stats.add(b, &mut c).count();
        assert_eq!(count, 2);
        assert_eq!(stats.total_ngram_count(), 3);
        assert_eq!(stats.total_token_count(), 17);
        assert_approx_eq!(
            stats.get(&b).unwrap().avg_cum_prob(),
            expected_cum_prob,
            1e-6
        );
        let expected_cum_prob: f64 = expected_cum_prob * count as f64;
        assert_approx_eq!(
            stats.get(&b).unwrap().cum_prob(),
            expected_cum_prob,
            1e-6
        );
        assert_approx_eq!(stats.avg_ngram_length(), 5.666666666666667, 1e-6);

        let data = stats.remove_every(&b).unwrap();
        assert_approx_eq!(data.cum_prob(), expected_cum_prob, 1e-6);
        assert_eq!(data.count(), 2);
        assert_eq!(data.weight(), 0.0);
        assert_eq!(stats.total_ngram_count(), 1);
        assert_eq!(stats.total_token_count(), 7);
        assert_approx_eq!(stats.avg_ngram_length(), 7.0, 1e-6);

        let _ = stats.add(b, &mut c);
        let count = stats.add(b, &mut c).count();
        let expected_cum_prob: f64 = c.iter().take(5).map(|t| t.p as f64).sum();
        assert_eq!(count, 2);
        assert_eq!(stats.total_ngram_count(), 3);
        assert_eq!(stats.total_token_count(), 17);
        assert_approx_eq!(
            stats.get(&b).unwrap().avg_cum_prob(),
            expected_cum_prob,
            1e-6
        );
        let expected_cum_prob: f64 = expected_cum_prob * count as f64;
        assert_approx_eq!(
            stats.get(&b).unwrap().cum_prob(),
            expected_cum_prob,
            1e-6
        );
        assert_approx_eq!(stats.avg_ngram_length(), 5.666666666666667, 1e-6);

        let data = stats.remove_one(&b).unwrap();
        let expected_cum_prob: f64 = c.iter().take(5).map(|t| t.p as f64).sum();
        assert_eq!(data.count, 1);
        assert_approx_eq!(data.cum_prob, expected_cum_prob, 1e-6);
        assert_eq!(stats.total_ngram_count(), 2);
        assert_eq!(stats.total_token_count(), 12);
        assert_approx_eq!(
            stats.get(&b).unwrap().cum_prob(),
            expected_cum_prob,
            1e-6
        );
        assert_approx_eq!(stats.avg_ngram_length(), 6.0, 1e-6);
    }
}
