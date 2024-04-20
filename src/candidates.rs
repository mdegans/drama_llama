//! Candidate token container and sampling methods.
//!
//! This is not publicly exposed because we plan to refactor the Candidate API
//! to be more functional and less imperative. The idea originally was to avoid
//! unnecessary copying of data, but it's not clear that this is a good idea
//! because in most cases the data is copied anyway.
//!
//! In the future, almost all sampling methods will take self by value and
//! return a truncated set of candidates. This will make the API more functional
//! and allow chaining of methods. It will also make it easier to reason about
//! the code and avoid bugs. It will also make it easier to implement a new
//! iterator yielding candidates, rather than the final token. Exposing this
//! will give users much more flexibility in how they use the library and how to
//! sample tokens.

use std::{
    borrow::Cow,
    num::{NonZeroUsize, TryFromIntError},
    ops::Index,
};

use partial_sort::PartialSort;

use llama_cpp_sys_3::{llama_token, llama_token_data, llama_token_data_array};

use crate::{
    model::Vocab,
    ngram::NGramStats,
    sample::{predict_token, SampleError},
    Probability, RepetitionOptions, SampleOptions,
};

/// Sort state of the candidates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sorted {
    /// The candidates may or may not be sorted.
    Unknown,
    /// The candidates are sorted because there is only one candidate left.
    One,
    /// The candidates are sorted until index `k` by id.
    ById { k: NonZeroUsize },
    /// The candidates are sorted until index `k` in order of most likely to
    /// least likely. This also means the candidates are sorted by probability.
    ByLogit { k: NonZeroUsize },
}

impl Default for Sorted {
    fn default() -> Self {
        Self::Unknown
    }
}

impl Sorted {
    /// Get the number of candidates that are sorted to by id. None is
    /// returned if the state is not [`Sorted::ByLogit`] or [`Sorted::One`].
    pub fn by_id(&self) -> Option<NonZeroUsize> {
        match self {
            Self::ById { k } => Some(*k),
            Self::One => Some(NonZeroUsize::new(1).unwrap()),
            _ => None,
        }
    }
    /// Get the number of candidates that are sorted to by logit. None is
    /// returned if the state is not [`Sorted::ByLogit`] or [`Sorted::One`].
    pub fn by_logit(&self) -> Option<NonZeroUsize> {
        match self {
            Self::ByLogit { k } => Some(*k),
            Self::One => Some(NonZeroUsize::new(1).unwrap()),
            _ => None,
        }
    }
}

/// A container for candidate tokens.
///
/// It is guaranteed that:
/// * The number of candidates is at least 1 and at most `i32::MAX`.
pub struct Candidates {
    /// A llama.cpp array of candidate tokens. This contains a pointer to the
    /// data in `data`, the size of the array, and a boolean indicating whether
    /// the array is sorted entirely by logit.
    pub(crate) arr: llama_token_data_array,
    /// In the initial state, or unsorted state, `sorted_until` is None. When
    /// it is not None, it is guaranteed to be non-zero and less than or equal
    /// to self.len()
    pub(crate) sort_state: Sorted,
    /// A cached state of whether the softmax has been applied. This is set
    /// to the index of the last candidate that was softmaxed. If the softmax
    /// has not been applied, this is set to None.
    pub(crate) softmax_applied_to: Option<NonZeroUsize>,
    /// It is guaranteed that `data.len() == arr.size` and `data.as_mut_ptr() ==
    /// arr.data`. It is also guaranteed that `data.len() <= i32::MAX` and there
    /// is at least one candidate `llama_token_data`.
    // This must always be last because it is dropped last, and it has to
    // outlive `arr` because `arr` contains raw pointers to the data in `data`.
    pub(crate) data: Vec<llama_token_data>,
}

// Safety: Candidates is Send because the only field that contains a raw pointer
// is `arr` and it is guaranteed that the data it points to is owned by the
// Candidates struct. The data is never moved or dropped while the raw pointer
// is still alive. All other fields are Send.
unsafe impl Send for Candidates {}
// Candidates is not Sync because of data races that could occur if the data is
// modified from multiple threads. We would need to add locking to make it Sync.

#[derive(Debug, thiserror::Error)]
pub enum CandidatesNewError<N> {
    #[error("Could not convert `{0}` to i32.")]
    OutOfRange(N),
    #[error("Number of candidates must be > 0. Got `{0}`.")]
    NotEnoughCandidates(i32),
}

impl Candidates {
    /// Create a new Candidates container with a given vocabulary size. The IDs
    /// of the candidates will be in the range [0, n_vocab]).
    // TODO: Custom type for the input. An i32 that is always supposed to be
    // positive is the way it's wr>itten in the c++ api, so we're stuck with it
    // for now... But we can make an immutable wrapper around it that is
    // guaranteed to always be positive.
    pub fn new<N>(n_vocab: N) -> Result<Self, CandidatesNewError<N>>
    where
        N: TryInto<i32, Error = TryFromIntError> + Copy,
    {
        // check upper bound
        let n_vocab: i32 = n_vocab
            .try_into()
            .map_err(|_| CandidatesNewError::OutOfRange(n_vocab))?;
        // check lower bound
        if n_vocab < 1 {
            return Err(CandidatesNewError::NotEnoughCandidates(n_vocab));
        }

        Ok(Self::from_i32(n_vocab))
    }

    // This is private because we don't want to expose the ability to create a
    // Candidates container with a negative number of candidates because it is
    // used as an array index in the c++ api (and ours) which could lead to
    // memory unsafety. It is checked on debug automatically, but not on
    // release.
    fn from_i32(n_vocab: i32) -> Self {
        let mut data: Vec<llama_token_data> = (0_i32..n_vocab)
            .map(|id| llama_token_data {
                id,
                logit: 0.0,
                p: 0.0,
            })
            .collect();

        let arr = llama_token_data_array {
            size: data.len(),
            data: data.as_mut_ptr(),
            sorted: false,
        };

        let sort_state = if data.len() == 1 {
            // Special case.
            Sorted::One
        } else {
            // We are sorted by id because we just initialized the data in
            // sorted order.
            Sorted::ById {
                k: data.len().try_into().unwrap(),
            }
        };

        Self {
            data,
            arr,
            sort_state,
            softmax_applied_to: None,
        }
    }

    /// Create a new Candidates container from a Vec of `llama_token_data`.
    ///
    /// Time complexity is O(n) where n is the number of candidates because the
    /// candidates are checked for sorting. See
    /// [`Candidates::from_vec_unchecked`] for a version that does not check if
    /// the candidates are sorted.
    ///
    /// # Panics
    /// * If `data` is empty.
    /// * If the IDs in `data` are not contiguous from 0 to n-1.
    // TODO: Make fallible and return a Result.
    pub fn from_vec(mut data: Vec<llama_token_data>) -> Self {
        assert!(!data.is_empty());

        let mut arr = llama_token_data_array {
            size: data.len(),
            data: data.as_mut_ptr(),
            sorted: false,
        };

        let sort_state = if data.len() == 1 {
            // Special case.
            Sorted::One
        } else {
            let mut by_logit = true;
            let mut by_id = true;
            let mut id_seen: Vec<bool> = vec![false; data.len()];

            *id_seen.get_mut(data[0].id as usize).unwrap() = true;
            for pair in data.windows(2) {
                if *id_seen.get(pair[1].id as usize).unwrap() {
                    panic!("Duplicate ID: {}", pair[1].id);
                }
                *id_seen.get_mut(pair[1].id as usize).unwrap() = true;
                if by_id && pair[0].id > pair[1].id {
                    by_id = false;
                }
                if by_logit && pair[0].logit < pair[1].logit {
                    by_logit = false;
                }
            }

            assert!(id_seen.iter().all(|&b| b));

            if by_id {
                Sorted::ById {
                    k: data.len().try_into().unwrap(),
                }
            } else if by_logit {
                arr.sorted = true;
                Sorted::ByLogit {
                    k: data.len().try_into().unwrap(),
                }
            } else {
                Sorted::Unknown
            }
        };

        Self {
            data,
            arr,
            sort_state,
            softmax_applied_to: None,
        }
    }

    /// Create a new Candidates container from a [`Vec`] without checking if the
    /// candidates are sorted. This is unsafe because ids are used to index into
    /// the candidates and if they are out of bounds, it could lead to memory
    /// unsafety in release mode.
    ///
    /// Time complexity is O(1).
    ///
    /// # Safety
    /// * The IDs in `data` must be contiguous from 0 to n-1. If they are
    ///   duplicated, this could lead to unsound behavior. If they are out of
    ///   bounds, this could lead to memory unsafety.
    pub unsafe fn from_vec_unchecked(mut data: Vec<llama_token_data>) -> Self {
        let arr = llama_token_data_array {
            size: data.len(),
            data: data.as_mut_ptr(),
            sorted: false,
        };

        let sort_state = if data.len() == 1 {
            // Special case.
            Sorted::One
        } else {
            Sorted::Unknown
        };

        Self {
            data,
            arr,
            sort_state,
            softmax_applied_to: None,
        }
    }

    /// Capacity of the container.
    pub fn capacity(&self) -> usize {
        self.data.len()
    }

    /// Returns the number of candidates.
    pub fn len(&self) -> NonZeroUsize {
        self.arr.size.min(self.data.len()).try_into().unwrap()
    }

    /// Get the [`Sorted`] state of the candidates.
    pub fn is_sorted(&self) -> Sorted {
        self.sort_state
    }

    /// Get a slice of sorted candidates. If k is out of upper bounds, all
    /// candidates will be returned.
    ///
    /// If [`SortState::Unknown`] is requested, an empty slice is returned. If
    /// [`SortState::One`] is requested, the first candidate is returned if
    /// there is only one candidate, otherwise an empty slice is returned.
    ///
    /// In general, it does not make sense to request the two above states, but
    /// it is allowed.
    ///
    /// If candidates are not yet sorted, sorting is performed on the k elements
    /// which can be up to O(n log n) time complexity in the worst case. If the
    /// candidates are already sorted to at least `k`, this method's time
    /// complexity is O(1).
    pub fn sorted(&mut self, desired: Sorted) -> &[llama_token_data] {
        let k = match desired {
            Sorted::ById { k } => k.get(),
            Sorted::ByLogit { k } => k.get(),
            Sorted::Unknown | Sorted::One => {
                panic!("Invalid state requested: {desired:?}")
            }
        }
        .min(self.data.len());

        self.sort(desired);

        &self.data[0..k]
    }

    /// Get a slice of sorted candidates. If k is out of upper bounds, all
    /// candidates will be returned. K cannot be zero.
    ///
    /// The returned slice is guaranteed to be non-empty because a class
    /// invariant is that the number of candidates is at least 1.
    ///
    /// If candidates are not yet sorted, sorting is performed on the k elements
    /// which can be up to O(n log n) time complexity in the worst case. If the
    /// candidates are already sorted to at least `k`, this method's time
    /// complexity is O(1).
    ///
    /// # Safety
    /// This method is unsafe because it returns a mutable view of the data. It
    /// is the caller's responsibility to ensure that the ids are contiguous
    /// from 0 to n-1.
    ///
    /// # Note:
    /// Because this returns a mutable view of the data, this method invalidates
    /// cached [`Sorted`] states and softmax state. The next time one of the
    /// methods requiring these states is called, they will be recalculated.
    pub unsafe fn sorted_mut(
        &mut self,
        desired: Sorted,
    ) -> &mut [llama_token_data] {
        self.sort(desired);

        let k = match self.sort_state {
            Sorted::ById { k } => k.get(),
            Sorted::ByLogit { k } => k.get(),
            Sorted::Unknown | Sorted::One => {
                panic!("Invalid state requested: {desired:?}")
            }
        }
        .min(self.data.len());

        // We have to invalidate the sort immediately after sorting, because we
        // are returning a mutable reference to the data. It's likely that the
        // user will want to change the logits of the candidates, and we can't
        // guarantee that the sort will be valid after that.
        self.arr.sorted = false;
        self.sort_state = Sorted::Unknown;
        self.softmax_applied_to = None;

        &mut self.data[0..k]
    }

    /// Sort the candidates by the desired [`Sorted`] state. If the state is
    /// already fulfilled, this method does nothing.
    ///
    /// # Panics
    /// * If SortState::One is requested and there is more than one candidate.
    ///   In general it doesn't make sense to request this state, but it is
    ///   allowed.
    // It's this way because without it the method would have to return a Result
    // (since none of the other states change the number of candidates) and this
    // way it's more ergonomic.
    pub fn sort(&mut self, desired: Sorted) {
        // Todo: We could avoid sorting entirely with a cached associative
        // array. This would be a tradeoff between memory and time complexity.
        // As it is this isn't terrible because we cache repeated calls.
        if self.len().get() == 1 {
            self.sort_state = Sorted::One;
            self.arr.sorted = true;
            return;
        }

        self.sort_state = match desired {
            Sorted::ById { k } => {
                let k = k.min(self.len());

                if let Some(cached_k) = self.sort_state.by_id() {
                    if cached_k >= k {
                        // Enough candidates are already sorted by id.
                        return;
                    }
                }

                self.data.partial_sort(k.get(), |a, b| {
                    b.id.partial_cmp(&a.id).unwrap()
                });

                Sorted::ById { k }
            }
            Sorted::ByLogit { k } => {
                let k = k.min(self.len());

                if self.arr.sorted {
                    // All the candidates are already sorted by logit. This is
                    // what the underlying library considers `arr.sorted` to
                    // mean and what we use by convention since we expose the
                    // same data structure for use in the underlying library.
                    return;
                }

                if let Some(cached_k) = self.sort_state.by_logit() {
                    if cached_k.get() >= k.get() {
                        return;
                    }
                }

                self.data.partial_sort(k.get(), |a, b| {
                    b.logit.partial_cmp(&a.logit).unwrap()
                });

                if k == self.len() {
                    self.arr.sorted = true;
                }

                Sorted::ByLogit { k }
            }
            Sorted::Unknown => return,
            Sorted::One => {
                if self.len().get() != 1 {
                    panic!("Invalid sort state requested: {desired:?} because there is more than one candidate.");
                }
                self.arr.sorted = true;
                Sorted::One
            }
        };
    }

    /// Returns a slice of the candidates.
    pub fn as_slice(&self) -> &[llama_token_data] {
        &self.data[0..self.len().get()]
    }

    /// Returns a mutable slice of the candidates.
    pub fn as_mut_slice(&mut self) -> &mut [llama_token_data] {
        let len = self.len().get();
        &mut self.data[0..len]
    }

    /// Get a `llama_token_data_array` referencing the internal data.
    pub fn as_llama_token_data_array(&mut self) -> &llama_token_data_array {
        assert_eq!(self.arr.data, self.data.as_mut_ptr());
        self.arr.size = self.len().get();
        &self.arr
    }

    /// Get a mutable `llama_token_data_array` referencing the internal data.
    ///
    /// Changing the size of the returned array is allowed. If it is out of
    /// range it will be truncated. Sorting the array and changing the member
    /// values is allowed.
    ///
    /// Changing the sort state is allowed. Methods that require a sorted state
    /// by logit will respect the new sort state. If the array is not really
    /// sorted by logit, this may lead to unexpected results, but not memory
    /// unsafety.
    ///
    /// Changing the pointers is not recommended at all. There isn't a good
    /// reason to do this **will lead to a panic** on next access.
    ///
    /// # Panics
    /// * if the `.data` pointer is changed to anything other than
    ///   `self.data.as_mut_ptr()` which isn't public.
    pub fn as_mut_llama_token_data_array(
        &mut self,
    ) -> &mut llama_token_data_array {
        assert_eq!(self.arr.data, self.data.as_mut_ptr());
        self.arr.size = self.len().get();
        &mut self.arr
    }

    /// Sample the most likely candidate from the candidates.
    ///
    /// If the candidates are already sorted by logit this method's time
    /// complexity is O(1). Otherwise, it is O(n).
    pub fn sample_token_greedy(&self) -> llama_token_data {
        if self.is_sorted().by_logit().is_some() || self.len().get() == 1 {
            // We are sorted by at least one candidate.
            *self.as_slice().first().unwrap()
        } else {
            *self
                .as_slice()
                .iter()
                .max_by(|a, b| a.logit.partial_cmp(&b.logit).unwrap())
                .unwrap()
        }
    }

    /// Sorts candidate tokens by their logits in descending order and calculate
    /// probabilities for top `k` candidates. Native Rust implementation.
    ///
    /// If `k` is greater than the number of candidates or `k` is `None`, all
    /// candidates are softmaxed.
    ///
    /// Time complexity is O(k log k) in cases where the candidates are not
    /// already sorted up to at least k elements by logit. Otherwise, it is
    /// O(k). In cases where the candidates are already softmaxed to k elements,
    /// the time complexity is O(1).
    ///
    /// # Note
    /// * This is a translation of `llama_sample_softmax`.
    /// * Summation of exp(logit) is calculated in f64 for precision.
    pub fn apply_softmax(&mut self, k: Option<NonZeroUsize>) {
        if let Some(cached_k) = self.softmax_applied_to {
            match k {
                // If the softmax has already been applied to the same number of
                // candidates, we don't need to do it again.
                Some(k) => {
                    if k.min(self.len()) == cached_k {
                        return;
                    }
                }
                None => {
                    if cached_k == self.len() {
                        return;
                    }
                }
            };
        }

        let k = k.map(|k| k.min(self.len())).unwrap_or(self.len());

        self.sort(Sorted::ByLogit { k });

        // Unwrap can never panic because a class invariant is that there is
        // at least one candidiate.
        let max_logit = self.data[..k.get()].first().unwrap().logit;
        let cum_sum: f64 =
            self.data[..k.get()].iter_mut().fold(0.0, |sum, token| {
                token.p = (token.logit - max_logit).exp();
                sum + token.p as f64
            });
        let cum_sum: f32 = cum_sum as f32;
        for token in &mut self.data[..k.get()] {
            token.p /= cum_sum as f32;
        }

        // We're not changing the logits so we don't need to sort again.
        debug_assert!(self.data[..k.get()]
            .windows(2)
            .all(|pair| pair[0].p >= pair[1].p));
        self.softmax_applied_to = Some(k);
    }

    /// Top-k sampling. Returns the top `k` most likely tokens. If `k` is
    /// greater than the number of candidates, all candidates are returned.
    ///
    /// Time complexity is O(k log k) if the candidates are not already sorted
    /// by logit to at least `k` elements. Otherwise, it is O(1).
    ///
    /// # Note
    /// * This method will sort the candidates by logit to at least `k` elements
    ///   if they are not already sorted to at least `k`.
    pub fn top_k(&mut self, k: NonZeroUsize) -> &[llama_token_data] {
        self.sorted(Sorted::ByLogit { k })
    }

    /// Top-p sampling. Returns the top tokens whose cumulative probability is
    /// less than or equal to `p`.
    ///
    /// It is guaranteed that at least `min_keep` tokens are returned even if no
    /// candidates fulfill the condition *however* in the case where the number
    /// of candidates is less than `min_keep`, all candidates are returned to a
    /// minimum of 1.
    ///
    /// Time complexity is O(n) where softmax has already been applied.
    /// Otherwise, it is O(n log n).
    ///
    /// # Note
    /// * This method will sort the candidates if they are not already sorted.
    /// * This method will apply the softmax if it has not been applied yet.
    pub fn top_p(
        &mut self,
        p: Probability<f64>,
        min_keep: NonZeroUsize,
    ) -> &[llama_token_data] {
        let min_keep = min_keep.min(self.len()).get();
        let mut sum: f64 = 0.0;
        let max_p: f64 = p.into_f();

        self.apply_softmax(None);

        for (i, data) in self.data.iter_mut().enumerate() {
            sum += data.p as f64;
            if sum >= max_p && i >= min_keep {
                return &self.data[..i];
            }
        }

        &self.data[..min_keep]
    }

    /// Min-p sampling. Returns the tokens whose logits are greater than or
    /// equal to the minimum logit required to select `min_keep` tokens with
    /// probability `p`. None is returned if no tokens can be selected.
    ///
    /// If the candidates are already sorted, this method's complexity is O(n).
    /// Otherwise, it is O(n log n).
    ///
    /// # Note
    /// * This is a translation of `llama_sample_min_p`, however it does not
    ///   reduce the number of candidates in self.data, rather returning a slice
    ///   or vector of the selected tokens (as needed).
    /// * This *may* sort the candidates by logit if they are not already
    ///   sorted.
    pub fn min_p(
        &mut self,
        p: Probability<f32>,
        min_keep: NonZeroUsize,
    ) -> Cow<'_, [llama_token_data]> {
        if self.data.is_empty() {
            unreachable!("Class invariant violated: There must be at least one candidate.")
        }

        let p: f32 = p.into_f();

        // If the candidates are not sorted, use the unsorted implementation
        if !self.arr.sorted {
            let mut max_logit: f32 = f32::MIN;

            for candidate in self.data.iter() {
                max_logit = max_logit.max(candidate.logit);
            }

            let min_logit = max_logit + p.ln();
            let filtered_tokens: Vec<llama_token_data> = self
                .data
                .iter()
                .cloned()
                .filter(|token| token.logit >= min_logit)
                .collect();

            if filtered_tokens.len() >= min_keep.get() {
                return Cow::Owned(filtered_tokens);
            }
        }

        self.sort(Sorted::ByLogit { k: self.len() });

        let min_logit = self.data.first().unwrap().logit + p.ln();

        // skip 1 because the first token is always selected
        for (i, token) in self.data.iter().enumerate().skip(1) {
            if token.logit < min_logit && i >= min_keep.get() {
                Some(Cow::Borrowed(&self.data[..i]));
            }
        }

        unreachable!(
            "Class invariant violated: There must be at least one candidate."
        )
    }

    /// Tail free sampling. Returns the tokens whose second derivatives are
    /// greater than or equal to `z`.
    ///
    /// Time complexity is O(n) where softmax has already been applied to n.
    /// Otherwise, it is O(n log n).
    ///
    /// # Note
    /// * This is a translation of `llama_sample_tail_free`.
    /// * This method may apply the softmax if it has not been applied yet.
    pub fn tail_free(
        &mut self,
        z: Probability<f32>,
        min_keep: NonZeroUsize,
    ) -> &[llama_token_data] {
        if self.data.len() <= 2 {
            return &self.data;
        }

        let z = z.into_f();

        self.apply_softmax(None);

        // Compute the first and second derivatives
        let first_derivatives: Vec<f32> = self
            .data
            .windows(2)
            .map(|pair| pair[0].p - pair[1].p)
            .collect();
        // Absolute value of the second derivatives
        let mut second_derivatives: Vec<f32> = first_derivatives
            .windows(2)
            .map(|pair| (pair[0] - pair[1]).abs())
            .collect();

        let sum: f64 = second_derivatives
            .iter()
            .fold(0.0, |sum, x| sum + f64::from(*x));

        if sum > 1e-6 {
            for value in second_derivatives.iter_mut() {
                *value /= sum as f32;
            }
        } else {
            let len: f32 = second_derivatives.len() as f32;
            for value in second_derivatives.iter_mut() {
                *value = 1.0 / len;
            }
        }

        let mut sum: f64 = 0.0;
        for (i, value) in second_derivatives.iter().enumerate() {
            sum += f64::from(*value);
            if sum > f64::from(z) && i >= min_keep.get() {
                return &self.data[..i];
            }
        }

        &self.data
    }

    /// Locally typical sampling.
    ///
    /// Time complexity is O(n) where softmax has already been applied to n.
    /// Otherwise, it is O(n log n).
    ///
    /// # Note
    /// * This is a translation of `llama_sample_typical`.
    /// * This method may apply the softmax if it has not been applied yet.
    ///
    /// As described in: https://arxiv.org/abs/2202.00666
    pub fn locally_typical(
        &mut self,
        p: Probability<f32>,
        min_keep: NonZeroUsize,
    ) -> Vec<llama_token_data> {
        if self.data.is_empty() {
            unreachable!("Class invariant violated: There must be at least one candidate.")
        }

        let min_keep: usize = min_keep.get();

        self.apply_softmax(None);

        let entropy: f64 = self.data.iter().fold(0.0, |sum, token| {
            sum - f64::from(token.p) * f64::from(token.p).ln()
        });

        // Compute the absolute difference between negative log probability and
        // entropy for each candidate
        let shifted_scores: Vec<f64> = self
            .data
            .iter()
            .map(|token| (f64::from(token.p).ln() - entropy).abs())
            .collect();

        // Sort tokens based on the shifted scores and their corresponding
        // indices
        let mut indices: Vec<usize> = (0..self.data.len()).collect();
        indices.sort_by(|&a, &b| {
            shifted_scores[a].partial_cmp(&shifted_scores[b]).unwrap()
        });

        // Compute the cumulative sum of the probabilities of the sorted tokens
        let mut sum: f64 = 0.0;
        let mut last_idx: usize = indices.len();
        for (i, index) in indices.iter().enumerate() {
            sum += f64::from(self.data[*index].p);
            if sum >= f64::from(p.into_f()) && i >= min_keep - 1 {
                last_idx = i + 1;
                break;
            }
        }

        indices
            .iter()
            .take(last_idx)
            .map(|&index| self.data[index])
            .collect()
    }

    /// Mirostat sampling.
    ///
    /// Time complexity is O(m) where softmax has already been applied to n.
    /// Otherwise, it is O(n log n).
    ///
    /// This is a translation of `llama_sample_token_mirostat`.
    pub fn mirostat(
        &mut self,
        rng: &mut xorshift::Xoroshiro128,
        tau: f32,
        eta: f32,
        m: NonZeroUsize,
        mu: &mut f32,
    ) -> llama_token {
        self.apply_softmax(None);

        if self.len() == NonZeroUsize::new(1).unwrap() {
            return self[0].id;
        }

        let x: llama_token_data;

        // Estimate s_hat using the most probable m tokens
        let s_hat;
        let mut t_i;
        let mut b_i;
        let mut sum_ti_bi = 0.0;
        let mut sum_ti_sq = 0.0;
        for i in 0..self.len().min(m).get() {
            t_i = ((i + 2) as f32 / (i + 1) as f32).ln();
            b_i = (self[i].p / self[i + 1].p).ln();
            sum_ti_bi += t_i * b_i;
            sum_ti_sq += t_i * b_i;
        }
        s_hat = sum_ti_bi / sum_ti_sq;

        // Compute k from the estimated s_hat and target suprise value
        let epsilon_hat = s_hat - 1.0;
        let k: f32 = ((epsilon_hat * (*mu * *mu))
            / (1.0 - (-epsilon_hat).powf(self.capacity() as f32)))
        .powf(1.0 / s_hat);
        let k: NonZeroUsize = (k as usize).try_into().unwrap();

        // Sample the next word X using top-k sampling
        let mut sample = self.top_k(k).to_vec();
        x = predict_token(rng, &mut sample);

        // Compute error as the difference between observed suprise and target
        // surprise value
        let observed_surprise = -(x.p.log2());
        let e = observed_surprise - tau;

        // Update mu using the learning rate and error
        *mu = *mu - eta * e;

        x.id
    }

    /// Mirostat V.2 sampling.
    ///
    /// Time complexity is O(n) where softmax has already been applied to n.
    /// Otherwise, it is O(n log n).
    ///
    /// This is a translation of `llama_sample_token_mirostat_v2`
    pub fn mirostat_v2(
        &mut self,
        rng: &mut xorshift::Xoroshiro128,
        tau: f32,
        eta: f32,
        mu: &mut f32,
    ) -> llama_token {
        self.apply_softmax(None);

        let x: llama_token_data;

        // Truncate the words with surprise values greater than mu
        let mut end: usize = self.len().get();
        for (i, candidate) in self.iter().enumerate() {
            if candidate.p.log2() > *mu {
                end = i.max(1);
                break;
            }
        }

        x = predict_token(rng, &mut self.data[..end]);

        // Compute error as the difference between observed surprise and target
        // surprise value
        let observed_surprise = -(x.p.log2());
        let e = observed_surprise - tau;

        // Update mu using the learning rate and error
        *mu = *mu - eta * e;

        x.id
    }

    /// Entropy sampling.
    ///
    /// Time complexity is O(n) where softmax has already been applied to n.
    /// Otherwise, it is O(n log n).
    pub fn apply_entropy(
        &mut self,
        min_temp: f64,
        max_temp: f64,
        exp_val: f64,
    ) {
        if min_temp < 0.0
            || max_temp < 0.0
            || min_temp > max_temp
            || self.data.len() <= 1
        {
            return;
        }

        // Calculate maximum possible entropy
        let max_entropy: f64 = -(1.0 / self.data.len() as f64).ln();

        self.apply_softmax(None);

        // Calculate entropy of the softmax probabilities
        let mut entropy: f64 = 0.0;
        for token in self.data.iter() {
            // ensure no log(0)
            if token.p > 0.0 {
                entropy -= f64::from(token.p) * f64::from(token.p).ln();
            }
        }

        // Normalize entropy
        let normalized_entropy: f64 = entropy / max_entropy;

        // Map the normalized entropy to the temperature range using the power
        // function
        let dyn_temp: f64 =
            min_temp + (max_temp - min_temp) * normalized_entropy.powf(exp_val);

        // Apply the temperature to the logits
        for token in self.data.iter_mut() {
            token.logit /= dyn_temp as f32;
        }

        // What we did above doesn't change the order of the candidates, so we
        // don't need to sort again.
        debug_assert!(self
            .data
            .windows(2)
            .all(|pair| pair[0].logit >= pair[1].logit));
    }

    /// Split P sampling. Sort the candidates by probability and returns the top
    /// tokens split at the point where the difference between probabilities is
    /// greatest.
    ///
    /// Time complexity is O(max_keep) where softmax has already been applied to
    /// max_keep. Otherwise, it is O(max_keep log max_keep). `max_keep` defaults
    /// to the number of candidates.
    ///
    /// # Panics
    /// * if `min_keep` is greater than `max_keep`.
    pub fn split_p(
        &mut self,
        min_keep: NonZeroUsize,
        max_keep: Option<NonZeroUsize>,
    ) -> &[llama_token_data] {
        self.apply_softmax(max_keep);

        let min_keep = min_keep.min(self.len()).get();
        let max_keep = max_keep
            .map(|k| k.min(self.len()))
            .unwrap_or(self.len())
            .get();

        assert!(
            min_keep <= max_keep,
            "min_keep must be less than or equal to max_keep"
        );

        let mut max_diff = 0.0;
        let mut split_idx = 0;
        for (i, pair) in self.data[..max_keep].windows(2).enumerate() {
            let diff = (pair[0].p - pair[1].p).abs();
            if diff > max_diff {
                max_diff = diff;
                split_idx = i + 1;
            }
        }

        &self.data[..split_idx.max(min_keep)]
    }

    /// Split L sampling. Sort the candidates by logit and returns the top
    /// tokens split at the point where the difference between logits is
    /// greatest.
    ///
    /// Time complexity is O(max_keep) where softmax has already been applied to
    /// max_keep. Otherwise, it is O(max_keep log max_keep). `max_keep` defaults
    /// to the number of candidates.
    ///
    /// # Panics
    /// * if `min_keep` is greater than `max_keep`.
    pub fn split_l(
        &mut self,
        min_keep: NonZeroUsize,
        max_keep: Option<NonZeroUsize>,
    ) -> &[llama_token_data] {
        let max_keep =
            max_keep.map(|k| k.min(self.len())).unwrap_or(self.len());
        self.sort(Sorted::ByLogit { k: max_keep });

        let min_keep = min_keep.get();
        let max_keep = max_keep.get();

        assert!(
            min_keep <= max_keep,
            "min_keep must be less than or equal to max_keep"
        );

        let mut max_diff = 0.0;
        let mut split_idx = 0;
        for (i, pair) in self.data[..max_keep].windows(2).enumerate() {
            let diff = (pair[0].logit - pair[1].logit).abs();
            if diff > max_diff {
                max_diff = diff;
                split_idx = i + 1;
            }
        }

        &self.data[..split_idx.max(min_keep)]
    }

    /// Sample a token from the candidates using [`SampleOptions`].
    pub fn sample_token(
        &mut self,
        tokens: &[llama_token],
        vocab: &Vocab,
        opts: &SampleOptions,
        freq_map: &mut NGramStats,
        rng: &mut xorshift::Xoroshiro128,
        mu: &mut Option<f32>,
    ) -> Result<llama_token, SampleError> {
        crate::sample::sample_token(
            tokens, self, vocab, opts, freq_map, rng, mu,
        )
    }

    /// This is a translation of `llama_sample_repetition_penalties`. It can be
    /// used in that mode or with n-grams. See [SampleRepetionOptions] for
    /// details.
    ///
    /// # Note
    /// * This method may apply the softmax if it has not been applied yet.
    /// * This method may sort the candidates if they are not already sorted.
    /// * This method may change the logits of the candidates.
    ///
    /// # Panics
    /// todo
    pub fn penalize_repetition(
        &mut self,
        tokens: &[llama_token],
        opts: &RepetitionOptions,
        freq_map: &mut NGramStats,
    ) -> Result<(), crate::sample::RepetitionError> {
        crate::sample::apply_sample_repetition_ngram(
            self, tokens, opts, freq_map,
        )
    }

    /// Iterates over the candidates.
    pub fn iter(&self) -> std::slice::Iter<llama_token_data> {
        self.data.iter()
    }

    /// Iterates over the candidates mutably. This does not change the sort
    /// state, so it's private.
    pub(crate) fn iter_mut(&mut self) -> std::slice::IterMut<llama_token_data> {
        self.data.iter_mut()
    }
}

impl Index<usize> for Candidates {
    type Output = llama_token_data;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let n = NonZeroUsize::new(10).unwrap();
        let candidates = Candidates::new(n.get()).unwrap();
        assert_eq!(candidates.data.len(), n.get());
        assert_eq!(candidates.len(), n);
        assert_eq!(candidates.capacity(), n.get());
        assert_eq!(Some(n.try_into().unwrap()), candidates.is_sorted().by_id());

        for i in 0..n.get() {
            assert_eq!(candidates[i].id, i as i32);
            assert_eq!(candidates[i].logit, 0.0);
            assert_eq!(candidates[i].p, 0.0);
        }
    }

    #[test]
    fn test_from_vec() {
        // Test with a vector sorted by id
        let v = (0..10)
            .map(|i| llama_token_data {
                id: i as i32,
                logit: 0.0,
                p: 0.0,
            })
            .collect::<Vec<_>>();
        let candidates = Candidates::from_vec(v);
        assert_eq!(
            candidates.is_sorted().by_id(),
            Some(10.try_into().unwrap())
        );

        // Test with a vector sorted by logit
        let v = (0..10)
            .map(|i| llama_token_data {
                id: 9 - i as i32,
                logit: -(i as f32),
                p: 0.0,
            })
            .collect::<Vec<_>>();
        let candidates = Candidates::from_vec(v);
        assert_eq!(
            candidates.is_sorted().by_logit(),
            Some(10.try_into().unwrap())
        );

        // Test with a vector that is not sorted
        let v = (0..10)
            .map(|i| llama_token_data {
                id: 9 - i as i32,
                logit: i as f32,
                p: 0.0,
            })
            .collect::<Vec<_>>();
        let candidates = Candidates::from_vec(v);
        assert_eq!(candidates.is_sorted(), Sorted::Unknown);
    }

    #[test]
    fn test_indexing() {
        let n = 10;
        let mut candidates = Candidates::new(n).unwrap();

        for i in 0..n {
            let data = &mut candidates.data[i];
            // we only need to test the id, since the other fields are on the
            // same struct and we're only testing for presence. Unless somehow
            // `new` is broken, this should always be `i`. It is also
            assert_eq!(data.id, i as i32);
        }

        for i in 0..n {
            assert_eq!(candidates[i].id, i as i32);
        }
    }

    #[test]
    fn test_sorted() {
        let n: usize = 10;
        let mut candidates = Candidates::new(n).unwrap();

        for i in 1..n + 1 {
            candidates.data[i - 1].logit = -(i as f32 / n as f32);
        }

        candidates.data.reverse();

        let sorted = candidates.sorted(Sorted::ByLogit {
            k: NonZeroUsize::new(3).unwrap(),
        });
        assert_eq!(sorted.len(), 3);

        assert_eq!(sorted[0].logit, -0.1);
        assert_eq!(sorted[1].logit, -0.2);
        assert_eq!(sorted[2].logit, -0.3);
    }

    #[test]
    fn test_iter() {
        let n: usize = 10;
        let mut candidates = Candidates::new(n).unwrap();

        for token in candidates.iter_mut() {
            assert_eq!(token.logit, 0.0);
            token.logit = token.id as f32;
        }

        for data in candidates.iter() {
            assert_eq!(data.logit, data.id as f32);
        }
    }

    #[test]
    fn test_apply_softmax() {
        let n: usize = 10;
        let mut c = Candidates::new(n).unwrap();

        for i in 1..n + 1 {
            c.data[n - i].logit = -(i as f32 / n as f32) + 0.5;
        }

        c.apply_softmax(None);

        // Check sort is by logit
        assert_eq!(
            c.is_sorted(),
            Sorted::ByLogit {
                k: NonZeroUsize::new(n).unwrap()
            }
        );
        for pair in c.data.windows(2) {
            assert!(pair[0].logit >= pair[1].logit);
        }

        // Check that the probabilities sum to 1
        let sum: f64 = c.iter().map(|d| d.p as f64).sum();
        assert_approx_eq!(sum, 1.0, 1e-5);
    }

    #[test]
    fn test_sample_token_greedy() {
        let n: usize = 10;
        let mut c = Candidates::new(n).unwrap();

        for i in 0..n {
            c.data[i].logit = -(i as f32).exp();
        }

        let token = c.sample_token_greedy();
        assert_eq!(token.id, 0);
    }

    #[test]
    fn test_top_p() {
        let n: usize = 10;
        let p: Probability<f64> = 0.5.try_into().unwrap();
        let mut c = Candidates::new(n).unwrap();

        for i in 1..n + 1 {
            c.data[n - i].logit = -(i as f32 / n as f32) + 0.5;
        }

        let tokens = c.top_p(p, 1.try_into().unwrap());
        assert_eq!(tokens.len(), 3);
        assert_approx_eq!(tokens[0].logit, 0.4, 1e-5);
        assert_approx_eq!(tokens[1].logit, 0.3, 1e-5);
        assert_approx_eq!(tokens[2].logit, 0.2, 1e-5);

        let sum: f64 = tokens.iter().map(|d| d.p as f64).sum();
        assert!(sum <= p.into_f());
    }

    #[test]
    fn test_min_p() {
        let n: usize = 10;
        let mut c = Candidates::new(n).unwrap();

        for i in 1..n + 1 {
            c.data[i - 1].logit = -(i as f32 / n as f32);
        }

        let tokens = c.min_p(0.8.try_into().unwrap(), 1.try_into().unwrap());
        assert_eq!(tokens.len(), 3);

        assert_eq!(tokens[0].logit, -0.1);
        assert_eq!(tokens[1].logit, -0.2);
        assert_eq!(tokens[2].logit, -0.3);
    }

    #[test]
    fn test_sample_typical() {
        let n: usize = 10;
        let mut c = Candidates::new(n).unwrap();

        for i in 1..n + 1 {
            c.data[i - 1].logit = -(i as f32 / n as f32);
        }

        let tokens =
            c.locally_typical(0.5.try_into().unwrap(), 1.try_into().unwrap());
        assert_eq!(tokens.len(), 4);

        assert_eq!(tokens[0].logit, -0.1);
        assert_eq!(tokens[1].logit, -0.2);
        assert_eq!(tokens[2].logit, -0.3);
        assert_eq!(tokens[3].logit, -0.4);
    }

    #[test]
    fn test_split_p() {
        let n: usize = 10;
        let mut c = Candidates::new(n).unwrap();

        for i in 0..5 {
            c.data[i].logit = -0.1;
        }
        for i in 5..10 {
            c.data[i].logit = -0.2;
        }

        let tokens = c.split_p(1.try_into().unwrap(), None);
        assert_eq!(tokens.len(), 5);
        assert_approx_eq!(tokens[0].p, 0.10499584, 1e-5);
        assert_approx_eq!(tokens[4].p, 0.10499584, 1e-5);
    }

    #[test]
    fn test_split_l() {
        let n: usize = 10;
        let mut c = Candidates::new(n).unwrap();

        for i in 0..5 {
            c.data[i].logit = -0.1;
        }
        for i in 5..10 {
            c.data[i].logit = -0.2;
        }

        let tokens = c.split_l(1.try_into().unwrap(), None);
        assert_eq!(tokens.len(), 5);
        assert_eq!(tokens[0].logit, -0.1);
        assert_eq!(tokens[4].logit, -0.1);
    }

    #[test]
    fn test_apply_entropy() {
        todo!();
    }

    #[test]
    fn test_sample_tail_free() {
        todo!();
    }
}
