//! Candidate token container and sampling methods.

use std::{
    num::{NonZeroUsize, TryFromIntError},
    ops::{Deref, Index},
};

use partial_sort::PartialSort;

use llama_cpp_sys_3::{llama_token, llama_token_data, llama_token_data_array};

use crate::{
    model::Vocab,
    ngram::NGramStats,
    sample::{choose_candidate, SampleError},
    Probability, RepetitionOptions, SampleOptions,
};
/// Sort state of the candidates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sorted {
    /// The candidates may or may not be sorted.
    Unknown,
    /// The candidates are sorted because there is only one candidate left. In
    /// this case they are sorted by id, logit, and probability.
    One,
    /// The candidates are sorted until index `k` by id.
    ById { k: NonZeroUsize },
    /// The candidates are sorted until index `k` in order of most likely to
    /// least likely. This also means the candidates are sorted by probability,
    /// although probabilities may not yet be calculated.
    ///
    /// See [`Candidates::softmax`] for calculating probabilities.
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

pub struct TokenDataArray<'a> {
    arr: llama_token_data_array,
    candidates: &'a mut Candidates,
}

impl TokenDataArray<'_> {
    /// Get the number of candidates in the array.
    pub fn len(&self) -> NonZeroUsize {
        self.candidates.len().min(self.arr.size.try_into().unwrap())
    }

    /// Get a slice of the candidates in the array.
    ///
    /// # Panics
    /// * If the arr.size has been modified to be out of bounds.
    pub fn as_slice(&self) -> &[llama_token_data] {
        assert!(self.arr.size == self.len().get());
        unsafe { std::slice::from_raw_parts(self.arr.data, self.len().get()) }
    }

    /// Get a mutable slice of the candidates in the array.
    pub fn as_mut_slice(&mut self) -> &mut [llama_token_data] {
        self.arr.size = self.len().get();
        self.candidates.sort_state = Sorted::Unknown;
        self.candidates.softmax_applied_to = None;
        unsafe {
            std::slice::from_raw_parts_mut(self.arr.data, self.len().get())
        }
    }

    /// Get the inner `llama_token_data_array` as a reference.
    ///
    /// # Panics
    /// * If the arr.size has been modified to be out of bounds.
    pub fn as_ref(&self) -> &llama_token_data_array {
        assert!(self.arr.size == self.len().get());
        &self.arr
    }

    /// Get the inner `llama_token_data_array` as a mutable reference.
    ///
    /// # Safety
    /// * The `arr.size` is guaranteed to be valid as long as the Candidates
    ///   struct is not modified.
    /// * The Candidates struct must outlive the pointer in the array since
    ///   it points to the candidate data.
    /// * If `arr.size` is shrunk, the Candidates struct must be truncated to
    ///   the new size. Growing the candidates is not allowed and will cause
    ///   a panic or truncation on the next access.
    pub fn as_mut(&mut self) -> &mut llama_token_data_array {
        self.arr.size = self.len().get();
        self.candidates.sort_state = Sorted::Unknown;
        self.candidates.softmax_applied_to = None;
        &mut self.arr
    }

    /// Convert into the inner `llama_token_data_array`.
    ///
    /// # Safety
    /// * The `arr.size` is guaranteed to be valid as long as the Candidates
    ///   struct is not modified.
    /// * The Candidates struct must outlive the array.
    /// * Prefer methods that hold a reference to the Candidates struct.
    /// * If `arr.size` is shrunk, the Candidates struct must be truncated to
    ///   the new size. Growing the candidates is not allowed and will cause
    ///   a panic or truncation on the next access.
    pub fn into_inner(mut self) -> llama_token_data_array {
        self.arr.size = self.len().get();
        self.candidates.sort_state = Sorted::Unknown;
        self.candidates.softmax_applied_to = None;
        self.arr
    }

    /// Get a pointer to the inner [`llama_token_data_array`]. Provided for
    /// compatability with [`llama_cpp_sys_3`] and friends.
    ///
    /// # Panics
    /// * If the arr.size has been modified to be out of bounds.
    ///
    ///
    /// # Safety
    /// * The `arr.size` is guaranteed to be valid as long as the Candidates
    ///   struct is not modified.
    /// * The Candidates struct and self must outlive the pointer in the array.
    /// * Prefer methods that hold a reference to the Candidates struct.
    pub fn as_ptr(&self) -> *const llama_token_data_array {
        assert!(self.arr.size == self.len().get());
        &self.arr
    }

    /// Get a mutable pointer to the inner [`llama_token_data_array`]. Provided
    /// for compatability with [`llama_cpp_sys_3`] and friends.
    ///
    /// # Safety
    /// * The `arr.size` is guaranteed to be valid as long as the Candidates
    ///   struct is not modified.
    /// * The Candidates struct and self must outlive the pointer in the array.
    /// * Prefer methods that hold a reference to the Candidates struct.
    /// * If `arr.size` is shrunk, the Candidates struct must be truncated to
    ///   the new size. Growing the candidates is not allowed and will cause
    ///   a panic or truncation on the next access.
    pub fn as_mut_ptr(&mut self) -> *mut llama_token_data_array {
        self.arr.size = self.len().get();
        self.candidates.sort_state = Sorted::Unknown;
        self.candidates.softmax_applied_to = None;
        &mut self.arr
    }
}

impl Deref for TokenDataArray<'_> {
    type Target = llama_token_data_array;

    fn deref(&self) -> &Self::Target {
        &self.arr
    }
}

/// A container for candidate tokens.
///
/// It is guaranteed, when not using unsafe methods, that:
/// * The number of candidates is **at least 1** and at most `i32::MAX`.
pub struct Candidates {
    /// Cached state of whether the candidates are sorted, by what, and to what
    /// index. This is used to avoid unnecessary sorting.
    pub(crate) sort_state: Sorted,
    /// A cached state whether, and to what index, the softmax has been applied.
    /// This does not guarantee that the candidates are sorted by logit.
    pub(crate) softmax_applied_to: Option<NonZeroUsize>,
    /// The actual candidate tokens.
    pub(crate) data: Vec<llama_token_data>,
}

static_assertions::assert_impl_all!(Candidates: Send, Sync);

#[derive(Debug, thiserror::Error, derive_more::From)]
pub enum CandidatesNewError {
    #[error("Could not convert vocabulary size to i32 because: `{0}`.")]
    OutOfRange(TryFromIntError),
    #[error("Number of candidates must be > 0. Got `{0}`.")]
    NotEnoughCandidates(i32),
}

static_assertions::assert_impl_all!(CandidatesNewError: Send, Sync);

impl Candidates {
    /// Create a new Candidates container with a given vocabulary size. The IDs
    /// of the candidates will be in the range [0, n_vocab].
    pub fn new<N>(n_vocab: N) -> Result<Self, CandidatesNewError>
    where
        N: TryInto<i32, Error = TryFromIntError> + Copy,
    {
        // check upper bound
        let n_vocab: i32 = n_vocab.try_into()?;
        // check lower bound
        if n_vocab < 1 {
            return Err(CandidatesNewError::NotEnoughCandidates(n_vocab));
        }

        Ok(Self::from_i32(n_vocab))
    }

    // This is private because we don't want to expose the ability to create a
    // Candidates container with a negative number of candidates because it is
    // used as an array index in the c++ api (and ours) which could lead to
    // memory unsafety.
    fn from_i32(n_vocab: i32) -> Self {
        let data: Vec<llama_token_data> = (0_i32..n_vocab)
            .map(|id| llama_token_data {
                id,
                logit: 0.0,
                p: 0.0,
            })
            .collect();

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
            sort_state,
            softmax_applied_to: None,
        }
    }

    /// Create a new Candidates container from a Vec of `llama_token_data`.
    ///
    /// Time complexity is O(n) where n is the number of candidates because the
    /// candidates are checked for sorting. See
    /// [`Candidates::from_vec_unchecked`] for a version that does not check if
    /// the candidates are sorted. Partial sort is not checked (yet).
    ///
    /// # Panics
    /// * If `data` is empty.
    /// * If the IDs in `data` are not contiguous from 0 to n-1.
    // TODO: Make fallible and return a Result.
    pub fn from_vec(data: Vec<llama_token_data>) -> Self {
        assert!(!data.is_empty());

        let sort_state = if data.len() == 1 {
            // Special case.
            Sorted::One
        } else {
            let mut by_logit = true;
            let mut by_id = true;

            for pair in data.windows(2) {
                if by_id && pair[0].id > pair[1].id {
                    by_id = false;
                }
                if by_logit && pair[0].logit < pair[1].logit {
                    by_logit = false;
                }
            }

            if by_id {
                Sorted::ById {
                    k: data.len().try_into().unwrap(),
                }
            } else if by_logit {
                Sorted::ByLogit {
                    k: data.len().try_into().unwrap(),
                }
            } else {
                Sorted::Unknown
            }
        };

        Self {
            data,
            sort_state,
            softmax_applied_to: None,
        }
    }

    /// Convert from an iterable of logit values. The IDs are assigned in order
    /// of the logit values starting from 0.
    ///
    /// # Panics
    /// * If the iterator is empty.
    pub fn from_logits<T>(it: T) -> Self
    where
        T: IntoIterator<Item = f32>,
    {
        Self::from_iter(
            (0..)
                .zip(it.into_iter())
                .map(|(id, logit)| llama_token_data { id, logit, p: 0.0 }),
        )
    }

    /// Create a new Candidates container from a [`Vec`] without checking if the
    /// candidates are sorted.
    ///
    /// The sort state is set to [`Sorted::Unknown`] and the softmax applied to
    /// is set to None.
    // This is const because Vec has const constructors and there is no reason
    // not to make it const. As of writing, the const constructors only allow
    // empty Vecs, but this may change in the future since C++ has constexpr
    // constructors for std::vector since C++20. By the time Rust has it,
    // Vec::len will probably also be const and this will be able to be checked
    // at compile time. Creating a const container of candidates may be useful
    // for intializing const candidate masks or other uses.
    pub const fn from_vec_unchecked(data: Vec<llama_token_data>) -> Self {
        Self {
            data,
            sort_state: Sorted::Unknown,
            softmax_applied_to: None,
        }
    }

    /// Create a new Candidates container from a [`Vec`] without checking if the
    /// candidates are sorted. This is safe, but if the candidates are not
    /// really sorted or softmaxed, it will break things. It will not, however,
    /// cause memory unsafety.
    pub const fn from_vec_unchecked_full(
        data: Vec<llama_token_data>,
        sort_state: Sorted,
        softmax_applied_to: Option<NonZeroUsize>,
    ) -> Self {
        Self {
            data,
            sort_state,
            softmax_applied_to,
        }
    }

    /// Convert into `llama_token_data_array`, leaking self.
    ///
    /// This is the most efficient way to pass the candidates to the C API since
    /// when the struct is recreated, the sort state is respected. For a
    /// slightly safer way to pass the candidates to the C API, see
    /// [`Candidates::as_token_data_array`] and the [`TokenDataArray`] wrapper.
    ///
    /// # Safety
    /// * This **leaks** the inner data so it must be converted back into a
    ///   Candidates struct using [`Candidates::from_llama_token_data_array`]
    ///   **or there will be a memory leak**. [`Vec::from_raw_parts`] can also
    ///   be used to take ownership of the data.
    pub fn into_llama_token_data_array(mut self) -> llama_token_data_array {
        let size = self.len().get();
        let data = self.data.as_mut_ptr();
        let sorted =
            self.is_sorted().by_logit().is_some_and(|n| n.get() == size);
        std::mem::forget(self);
        llama_token_data_array { size, data, sorted }
    }

    /// Create a Candidates container from `llama_token_data_array`. This will
    /// *take ownership* of the array and free it when the Candidates struct is
    /// dropped.
    ///
    /// If `arr.sorted` is true, the candidates are assumed to be sorted by
    /// logit entirely. This is not checked on release builds, but it is on
    /// debug builds at O(n) time complexity.
    ///
    /// If the softmax state is known, it should be passed in to avoid repeated
    /// unnecessary softmax calculations.
    ///
    /// # Panics
    /// * In debug builds, if the candidates are not sorted by logit.
    /// * If the array's data is null.
    /// * If the array's size is 0.
    ///
    /// # Safety
    /// * The array must have been created by
    ///   [`Candidates::into_llama_token_data_array`] or by using the global
    ///   allocator. If `arr.data` was allocated by another allocator, this
    ///   method should not be used.
    /// * (a copy of the) array must not be used after calling this method.
    /// * `arr.size` must be within the bounds of `arr.data`.
    /// * If the array's data is owned by another struct, dropping the
    ///   Candidates struct will cause a double free.
    pub unsafe fn from_llama_token_data_array(
        arr: llama_token_data_array,
        softmax_applied_to: Option<NonZeroUsize>,
    ) -> Self {
        assert!(!arr.data.is_null());
        let data = Vec::from_raw_parts(arr.data, arr.size, arr.size);

        if arr.sorted {
            debug_assert!(data.windows(2).all(|w| w[0].logit >= w[1].logit));
        }

        Self {
            data,
            sort_state: if arr.sorted {
                Sorted::ByLogit {
                    k: arr.size.try_into().unwrap(),
                }
            } else {
                Sorted::Unknown
            },
            softmax_applied_to,
        }
    }

    /// Returns the number of candidates. This is guaranteed to be at least 1
    /// and at most the vocabulary size the candidates were created with.
    pub fn len(&self) -> NonZeroUsize {
        self.data.len().try_into().unwrap()
    }

    /// Returns `Some(llama_token_data)` if there is only one candidate.
    pub fn is_one(&self) -> Option<llama_token_data> {
        if self.len().get() == 1 {
            Some(self.data[0])
        } else {
            None
        }
    }

    /// Get the [`Sorted`] state of the candidates. This can be use to check
    /// whether the candidates are sorted, by what, and to what index.
    ///
    /// ```rust
    /// # use std::num::NonZeroUsize;
    /// use drama_llama::{Candidates, Sorted};
    ///
    /// let k = NonZeroUsize::new(10).unwrap();
    /// let candidates = Candidates::new(k.get()).unwrap();
    /// assert!(candidates.is_sorted().by_id().is_some_and(|n| n == k));
    /// assert_eq!(candidates.is_sorted(), Sorted::ById { k });
    /// ```
    pub const fn is_sorted(&self) -> Sorted {
        self.sort_state
    }

    /// Truncate the candidates to a new length. If the new length is greater
    /// than the current length, this does nothing.
    ///
    /// Note that this does not change the allocated capacity of the candidates.
    pub fn truncate(mut self, new_len: NonZeroUsize) -> Self {
        self.data.truncate(self.len().min(new_len).get());
        self
    }

    /// Select a candidate by index, throwing out the rest, leaving only the
    /// selected candidate in self.
    pub fn select(mut self, index: usize) -> Self {
        self.data = vec![self[index]];
        self.sort_state = Sorted::One;
        self.softmax_applied_to = None;
        self
    }

    /// Sort the candidates by the desired [`Sorted`] state. If the state is
    /// already fulfilled this method does nothing. If [`Sorted::Unknown`] is
    /// requested, the sort state is reset to [`Sorted::Unknown`]. Generally,
    /// this doens't make sense to request, but it is allowed.
    ///
    /// **Truncates to `k.`**
    ///
    /// Time complexity is O(n log k) where n is the number of candidates and k
    /// is the number of candidates to sort to (partial sort). If the candidates
    /// are already sorted to at least k elements, the time complexity is O(1).
    ///
    /// # Panics
    /// * If SortState::One is requested and there is more than one candidate
    ///   because it would be impossible to know what to sort by in order to
    ///   satisfy the request.
    // It's this way because without it the method would have to return a Result
    // (since none of the other states change the number of candidates) and this
    // way it's more ergonomic.
    pub fn sort(mut self, desired: Sorted) -> Self {
        if self.len().get() == 1 {
            self.sort_state = Sorted::One;
            return self;
        }

        self.sort_state = match desired {
            Sorted::ById { k } => {
                let k = k.min(self.len());

                if let Some(cached_k) = self.sort_state.by_id() {
                    if cached_k >= k {
                        // Enough candidates are already sorted by id.
                        self.sort_state = desired;
                        return self.truncate(k);
                    }
                }

                self.data.partial_sort(k.get(), |a, b| {
                    b.id.partial_cmp(&a.id).unwrap()
                });

                Sorted::ById { k }
            }
            Sorted::ByLogit { k } => {
                let k = k.min(self.len());

                if let Some(cached_k) = self.sort_state.by_logit() {
                    if cached_k.get() >= k.get() {
                        self.sort_state = desired;
                        return self.truncate(k);
                    }
                }

                self.data.partial_sort(k.get(), |a, b| {
                    b.logit.partial_cmp(&a.logit).unwrap()
                });

                Sorted::ByLogit { k }
            }
            // This doesn't make sense to request, but it's allowed becuase in
            // some cases (for example unchecked construction) it may be useful.
            Sorted::Unknown => Sorted::Unknown,
            Sorted::One => {
                if self.len().get() != 1 {
                    panic!("Invalid sort state requested: {desired:?} because there is more than one candidate.");
                }
                Sorted::One
            }
        };

        let k = match self.sort_state {
            Sorted::ById { k } => k,
            Sorted::ByLogit { k } => k,
            _ => unreachable!(),
        };

        if self.len().get() == 1 {
            self.sort_state = Sorted::One;
        }

        self.truncate(k)
    }

    /// Returns a slice of the candidates as [`llama_token_data`].
    pub fn as_slice(&self) -> &[llama_token_data] {
        &self.data[0..self.len().get()]
    }

    /// Returns a mutable slice of the candidates as [`llama_token_data`].
    ///
    /// Invalidates cached:
    /// * Sort state
    /// * Softmax state
    pub fn as_mut_slice(&mut self) -> &mut [llama_token_data] {
        let len = self.len().get();
        self.sort_state = Sorted::Unknown;
        self.softmax_applied_to = None;
        &mut self.data[0..len]
    }

    /// Get a [`TokenDataArray`] referencing the internal data. It provides
    /// methods for working with the candidates using the C API.
    ///
    /// If a method is called that modifies the candidates, the internal sort
    /// state and softmax state will be invalidated automatically.
    pub fn as_token_data_array<'a>(&'a mut self) -> TokenDataArray<'a> {
        TokenDataArray {
            arr: llama_token_data_array {
                data: self.data.as_mut_ptr(),
                sorted: self
                    .is_sorted()
                    .by_logit()
                    .is_some_and(|n| n == self.len()),
                size: self.len().get(),
            },
            candidates: self,
        }
    }

    /// Sample the most likely candidate from the candidates. This is guaranteed
    /// to return a single candidate. [`Candidates::is_one`] can be used to get
    /// and unwrap the candidate without risk of panic.
    ///
    /// If the candidates are already sorted by logit this method's time
    /// complexity is O(1). Otherwise, it is O(n).
    pub fn sample_token_greedy(self) -> Candidates {
        if self.is_sorted().by_logit().is_some() || self.len().get() == 1 {
            // We are sorted by at least one candidate.
            self.truncate(NonZeroUsize::new(1).unwrap())
        } else {
            // Unsorted implementation, find the max logit. Since this is a
            // partial sort, the complexity is O(n).
            self.sort(Sorted::ByLogit {
                k: NonZeroUsize::new(1).unwrap(),
            })
        }
    }

    /// Sorts candidate tokens by their logits to k in descending order and
    /// calculate probabilities for top `k` candidates. **Truncates to `k`.**
    ///
    /// If `k` is greater than the number of candidates or `k` is `None`, all
    /// candidates are softmaxed.
    ///
    /// Time complexity is O(n log k) in cases where the candidates are not
    /// already sorted up to at least k elements by logit. Otherwise, it is
    /// O(k). In cases where the candidates are already softmaxed to k elements,
    /// the time complexity is O(1).
    ///
    /// # Note
    /// * This is a translation of `llama_sample_softmax`.
    /// * Summation of exp(logit) is calculated in f64 for precision. This
    ///   likely doesn't matter, but with many small values, it could.
    pub fn softmax(self, k: Option<NonZeroUsize>) -> Self {
        let k = k.map(|k| k.min(self.len())).unwrap_or(self.len());

        if let Some(cached_k) = self.softmax_applied_to {
            if k == cached_k {
                // We may be softmaxed to k, but unsorted. Since we state that
                // this function sorts to k, we need to sort to k. This is a
                // no-op if we are already sorted to at least k.
                return self.sort(Sorted::ByLogit { k });
            }
        }

        let mut new = self.sort(Sorted::ByLogit { k }).truncate(k);

        // Unwrap can never panic because a class invariant is that there is
        // at least one candidiate.
        let max_logit = new.data.first().unwrap().logit;
        let cum_sum: f64 = new.iter_mut().fold(0.0, |sum, token| {
            token.p = (token.logit - max_logit).exp();
            sum + token.p as f64
        });
        let cum_sum: f32 = cum_sum as f32;
        for token in &mut new.data {
            token.p /= cum_sum as f32;
        }

        // We're not changing the logits so we don't need to sort again.

        new.softmax_applied_to = Some(k);

        new
    }

    /// Top-k sampling. Returns the top `k` most likely tokens. If `k` is
    /// greater than the number of candidates, all candidates are returned.
    ///
    /// Time complexity is O(n log k) if the candidates are not already sorted
    /// by logit to at least `k` elements. Otherwise, it is O(1).
    ///
    /// # Note
    /// * This method will sort the candidates by logit to at least `k` elements
    ///   if they are not already sorted to at least `k`. This is equivalent to
    ///   calling `sort(Sorted::ByLogit { k })`.
    pub fn top_k(self, k: NonZeroUsize) -> Self {
        self.sort(Sorted::ByLogit { k })
    }

    /// Top-p sampling (nucleus). Returns the top tokens whose cumulative
    /// probability is less than or equal to `p`.
    ///
    /// It is guaranteed at least one token is selected.
    ///
    /// <https://arxiv.org/abs/1904.09751>
    ///
    /// Time complexity is O(n) where softmax has already been applied.
    /// Otherwise, it is O(n log n).
    ///
    /// # Note
    /// * This method will sort the candidates by logit if they are not already.
    /// * This method will apply the softmax if it has not been applied yet.
    pub fn top_p(self, p: Probability<f64>, min_keep: NonZeroUsize) -> Self {
        if self.data.len() == 1 {
            return self;
        }

        let min_keep = min_keep.min(self.len());
        let mut sum: f64 = 0.0;
        let max_p: f64 = p.into_f();

        let new = self.softmax(None);

        for (i, data) in new.iter().enumerate() {
            sum += data.p as f64;
            if sum >= max_p && i >= min_keep.get() {
                return new.truncate(i.try_into().unwrap());
            }
        }

        new
    }

    /// Min-p sampling.
    ///
    /// As described in: <https://github.com/ggerganov/llama.cpp/pull/3841>
    ///
    /// It is guaranteed at least one token is selected.
    ///
    /// If the candidates are already sorted, this method's complexity is O(n).
    /// Otherwise, it is O(n log n).
    ///
    /// # Note
    /// * This is a translation of `llama_sample_min_p`.
    /// * This *may* sort the candidates by logit if they are not already
    ///   sorted.
    pub fn min_p(
        self,
        p: Probability<f32>,
        mut min_keep: NonZeroUsize,
    ) -> Self {
        if self.data.is_empty() {
            unreachable!("Class invariant violated: There must be at least one candidate.")
        }
        if self.data.len() == 1 {
            return self;
        }

        min_keep = min_keep.min(self.len());

        let p: f32 = p.into_f();

        // TODO: This has just been refactored so it should be checked against
        // the original implementation.

        // If the candidates are not sorted, use the unsorted implementation
        if !self.is_sorted().by_logit().is_some_and(|k| k == self.len()) {
            let mut max_logit: f32 = f32::MAX;
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
                return Self::from_vec_unchecked(filtered_tokens);
            }
        }

        let k = self.len();
        let new = self.sort(Sorted::ByLogit { k });

        // min logit for p_i >= p * p_max
        let min_logit = new.data.first().unwrap().logit + p.ln();

        // skip 1 because the first token is always selected
        for (i, token) in new.iter().enumerate().skip(1) {
            if token.logit < min_logit && i >= min_keep.get() {
                // This can never panic because we just skipped the first token.
                return new.truncate(i.try_into().unwrap());
            }
        }

        unreachable!(
            "Class invariant violated: There must be at least one candidate."
        )
    }

    /// Tail free sampling.
    ///
    /// <https://www.trentonbricken.com/Tail-Free-Sampling/>
    ///
    /// Time complexity is O(n) where softmax has already been applied to n.
    /// Otherwise, it is O(n log n).
    ///
    /// # Note
    /// * This is a translation of `llama_sample_tail_free`.
    /// * This method may apply the softmax if it has not been applied yet.
    pub fn tail_free(
        self,
        z: Probability<f32>,
        mut min_keep: NonZeroUsize,
    ) -> Self {
        if self.data.len() <= 2 {
            return self;
        }

        min_keep = min_keep.min(self.len());

        let z = z.into_f();

        let new = self.softmax(None);

        // Compute the first and second derivatives
        let first_derivatives: Vec<f32> = new
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
                return new.truncate(i.try_into().unwrap());
            }
        }

        new
    }

    /// Locally typical sampling.
    ///
    /// <https://arxiv.org/abs/2202.00666>
    ///
    /// Time complexity is O(n) where softmax has already been applied to n.
    /// Otherwise, it is O(n log n).
    ///
    /// # Note
    /// * This is a translation of `llama_sample_typical`.
    /// * This method will apply the softmax if it has not been applied yet.
    pub fn locally_typical(
        self,
        p: Probability<f32>,
        min_keep: NonZeroUsize,
    ) -> Self {
        if self.data.is_empty() {
            unreachable!("Class invariant violated: There must be at least one candidate.")
        }
        if self.data.len() == 1 {
            return self;
        }

        let min_keep: usize = min_keep.min(self.len()).get();

        let new = self.softmax(None);

        let entropy: f64 = new.iter().fold(0.0, |sum, token| {
            sum - f64::from(token.p) * f64::from(token.p).ln()
        });

        // Compute the absolute difference between negative log probability and
        // entropy for each candidate
        let shifted_scores: Vec<f64> = new
            .iter()
            .map(|token| {
                if token.p != 0.0 {
                    (f64::from(token.p).ln() - entropy)
                        .abs()
                        .min(f64::MAX)
                        .max(f64::MIN)
                } else {
                    // If p is 0, the entropy is 0 and the score is infinity.
                    f64::MAX
                }
            })
            .collect();

        // Sort tokens based on the shifted scores and their corresponding
        // indices.
        let mut indices: Vec<usize> = (0..new.data.len()).collect();
        indices.sort_by(|&a, &b| {
            shifted_scores[a].partial_cmp(&shifted_scores[b]).unwrap()
        });

        // Compute the cumulative sum of the probabilities of the sorted tokens
        let mut sum: f64 = 0.0;
        let mut new_len: usize = indices.len();
        for (i, index) in indices.iter().enumerate() {
            sum += f64::from(new.data[*index].p);
            if sum >= f64::from(p.into_f()) && i >= min_keep - 1 {
                new_len = i + 1;
                break;
            }
        }

        indices
            .iter()
            .take(new_len)
            .map(|&index| new.data[index])
            .collect()
    }

    /// Mirostat sampling. Guaranteed to return a single candidate.
    ///
    /// Time complexity is O(**max_keep**) where softmax has already been
    /// applied to n. Otherwise, it is O(n log n).
    ///
    /// # Note
    /// * This is a translation of `llama_sample_token_mirostat`.
    /// * The public API will likely change since this sampling method is
    ///   very different from the others and doesn't return a Candidate
    ///   container.
    pub fn mirostat(
        self,
        rng: &mut xorshift::Xoroshiro128,
        tau: f32,
        eta: f32,
        max_keep: Option<NonZeroUsize>,
        opt_mu: &mut Option<f32>,
    ) -> Candidates {
        if self.data.is_empty() {
            unreachable!("Class invariant violated: There must be at least one candidate.")
        }
        if self.data.len() == 1 {
            return self;
        }

        let new = self.softmax(None);
        let m = max_keep
            .unwrap_or(NonZeroUsize::new(100).unwrap())
            .min(new.len())
            .get();
        // mu is initialized to 2 * tau if not provided
        let mu = opt_mu.unwrap_or(2.0 * tau);

        // Estimate s_hat using the most probable m tokens
        let s_hat;
        let mut t_i;
        let mut b_i;
        let mut sum_ti_bi = 0.0;
        let mut sum_ti_sq = 0.0;
        for i in 0..m {
            t_i = ((i + 2) as f32 / (i + 1) as f32).ln();
            b_i = (new[i].p / new[i + 1].p).ln();
            sum_ti_bi += t_i * b_i;
            sum_ti_sq += t_i * b_i;
        }
        s_hat = sum_ti_bi / sum_ti_sq;

        // Compute k from the estimated s_hat and target suprise value
        let epsilon_hat = s_hat - 1.0;
        let k: f32 = ((epsilon_hat * (mu * mu))
            / (1.0 - (-epsilon_hat).powf(new.len().get() as f32)))
        .powf(1.0 / s_hat);
        let k: NonZeroUsize = (k as usize).try_into().unwrap();

        // Sample the next word X using top-k sampling
        // TODO: in the future we may return the candidates here to give more
        // control over how a token is chosen and to make the Candidate API more
        // consistent.
        let new = choose_candidate(rng, new.top_k(k));

        // Compute error as the difference between observed suprise and target
        // surprise value
        let observed_surprise = -(new.is_one().unwrap().p.log2());
        let e = observed_surprise - tau;

        // Update mu using the learning rate and error
        *opt_mu = Some(mu - eta * e);

        new
    }

    /// Mirostat V.2 sampling. Guaranteed to return a single Candidate.
    ///
    /// Time complexity is O(**max_keep**) where softmax has already been
    /// applied to n. Otherwise, it is O(n log n).
    ///
    /// # Note
    /// * This is a translation of `llama_sample_token_mirostat_v2` but differs
    /// slightly in that `max_keep` is supported. If the original behavior is
    /// desired, set `max_keep` to `self.len()`. The defualt is 100 like v1 with
    /// the rationale that more than 100 candidates is likely to be too many.
    pub fn mirostat_v2(
        self,
        rng: &mut xorshift::Xoroshiro128,
        tau: f32,
        eta: f32,
        max_keep: Option<NonZeroUsize>,
        opt_mu: &mut Option<f32>,
    ) -> Candidates {
        if self.data.is_empty() {
            unreachable!("Class invariant violated: There must be at least one candidate.")
        }
        if self.data.len() == 1 {
            return self;
        }
        let new = self.softmax(None);
        // mu is initialized to 2 * tau if not provided
        let mu = opt_mu.unwrap_or(2.0 * tau);
        let m = max_keep
            .unwrap_or(NonZeroUsize::new(100).unwrap())
            .min(new.len())
            .get();

        // Truncate the words with surprise values greater than mu
        let mut end: usize = m;
        for (i, candidate) in new.data[..m].iter().enumerate().skip(1) {
            if candidate.p.log2() > mu {
                end = i;
                break;
            }
        }

        let new = choose_candidate(rng, new.truncate(end.try_into().unwrap()));

        // Compute error as the difference between observed surprise and target
        // surprise value
        let observed_surprise = -(new.is_one().unwrap().p.log2());
        let e = observed_surprise - tau;

        // Update mu using the learning rate and error
        *opt_mu = Some(mu - eta * e);

        new
    }

    /// Entropy sampling.
    ///
    /// Time complexity is O(n) where softmax has already been applied to n.
    /// Otherwise, it is O(n log n).
    pub fn apply_entropy(
        self,
        min_temp: f64,
        max_temp: f64,
        exp_val: f64,
    ) -> Self {
        if min_temp < 0.0
            || max_temp < 0.0
            || min_temp > max_temp
            || self.data.len() <= 1
        {
            return self;
        }

        // Calculate maximum possible entropy
        let max_entropy: f64 = -(1.0 / self.len().get() as f64).ln();

        let mut new = self.softmax(None);

        // Calculate entropy of the softmax probabilities
        let mut entropy: f64 = 0.0;
        for token in new.iter() {
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
        for token in new.data.iter_mut() {
            token.logit /= dyn_temp as f32;
        }

        // What we did above doesn't change the order of the candidates, so we
        // don't need to sort again.
        debug_assert!(new
            .data
            .windows(2)
            .all(|pair| pair[0].logit >= pair[1].logit));

        new
    }

    /// Split P sampling. Softmax to `max_keep` and return the top tokens split
    /// at the point where the difference between probabilities is greatest.
    ///
    /// Time complexity is O(max_keep) where softmax has already been applied to
    /// max_keep. Otherwise, it is O(max_keep log max_keep). `max_keep` defaults
    /// to the number of candidates.
    ///
    /// It is guaranteed that at least one token is selected.
    ///
    /// # Panics
    /// * if `min_keep` is greater than `max_keep`.
    ///
    /// # Note
    /// * This applies the softmax to `max_keep` if it has not been applied yet.
    // Note: I wrote this after observing the point where candidates are split
    // most of the time. This is my own method and not a part of `llama.cpp`.
    // Likewise `split_l` is also my own method. The name is inspired by split
    // pea soup and subject to change.
    //
    // I don't know how this will perform in practice. It's just an idea based
    // on observation and intuition. This is similar to nucleus sampling but
    // instead of using probability mass, it uses a difference in probability
    // (or logit) to split the candidates. - mdegans
    pub fn split_p(
        self,
        min_keep: NonZeroUsize,
        max_keep: Option<NonZeroUsize>,
    ) -> Self {
        if self.data.is_empty() {
            unreachable!("Class invariant violated: There must be at least one candidate.")
        }
        if self.data.len() == 1 {
            return self;
        }
        let new = self.softmax(max_keep);

        let min_keep = min_keep.min(new.len()).get();
        let max_keep = max_keep
            .map(|k| k.min(new.len()))
            .unwrap_or(new.len())
            .get();

        assert!(
            min_keep <= max_keep,
            "min_keep must be less than or equal to max_keep"
        );

        let mut max_diff = 0.0;
        let mut split_idx = 0;
        for (i, pair) in new.data[..max_keep].windows(2).enumerate() {
            let diff = (pair[0].p - pair[1].p).abs();
            if diff > max_diff {
                max_diff = diff;
                split_idx = i + 1;
            }
        }

        new.truncate(split_idx.max(min_keep).try_into().unwrap())
    }

    /// Split L sampling. Sort the candidates by logit to `max_keep` and return
    /// the top tokens split at the point where the difference between logits is
    /// greatest.
    ///
    /// Time complexity is O(max_keep). `max_keep` defaults to the number of
    /// candidates.
    ///
    /// It is guaranteed that at least one token is selected.
    ///
    /// # Panics
    /// * if `min_keep` is greater than `max_keep`.
    // I wrote this after `split_p` and it's based on the same idea, but since
    // the relationship between logit and probability is not linear, it's not
    // guaranteed to return the same results, or even good results... but it is
    // probably faster than most other methods. - mdegans
    pub fn split_l(
        self,
        min_keep: NonZeroUsize,
        max_keep: Option<NonZeroUsize>,
    ) -> Self {
        if self.data.is_empty() {
            unreachable!("Class invariant violated: There must be at least one candidate.")
        }
        if self.data.len() == 1 {
            return self;
        }
        let max_keep =
            max_keep.map(|k| k.min(self.len())).unwrap_or(self.len());
        let new = self.sort(Sorted::ByLogit { k: max_keep });

        let min_keep = min_keep.get();
        let max_keep = max_keep.get();

        assert!(
            min_keep <= max_keep,
            "min_keep must be less than or equal to max_keep"
        );

        let mut max_diff = 0.0;
        let mut split_idx = 0;
        for (i, pair) in new.data[..max_keep].windows(2).enumerate() {
            let diff = (pair[0].logit - pair[1].logit).abs();
            if diff > max_diff {
                max_diff = diff;
                split_idx = i + 1;
            }
        }

        new.truncate(split_idx.max(min_keep).try_into().unwrap())
    }

    /// Sample a token from the candidates using [`SampleOptions`].
    pub fn sample_token(
        self,
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

    /// Apply repetition penalties to the candidates. This code is inspired by
    /// [`llama_sample_repetition_penalties`] but supports n-grams and an
    /// [`NGramStats`] object to keep track of n-gram frequencies as well as
    /// other stats like the number of tokens processed.
    ///
    /// # Note
    /// * This method may apply the softmax if it has not been applied yet.
    /// * This method may sort the candidates if they are not already sorted.
    /// * This method may change the logits of the candidates.
    ///
    /// [`llama_sample_repetition_penalties`]: llama_cpp_sys_3::llama_sample_repetition_penalties
    pub fn penalize_repetition(
        self,
        tokens: &[llama_token],
        opts: &RepetitionOptions,
        freq_map: &mut NGramStats,
    ) -> Result<Candidates, crate::sample::RepetitionError> {
        crate::sample::apply_sample_repetition_ngram(
            self, tokens, opts, freq_map,
        )
    }

    /// Iterates over the candidates.
    pub fn iter(&self) -> std::slice::Iter<llama_token_data> {
        self.data.iter()
    }

    /// Iterates over the candidates mutably. This invalidates the sort state
    /// and softmax state.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<llama_token_data> {
        self.sort_state = Sorted::Unknown;
        self.softmax_applied_to = None;
        self.data.iter_mut()
    }

    /// Convert from an iterator without checking class invariants.
    pub fn from_iter_unchecked<T>(it: T) -> Self
    where
        T: IntoIterator<Item = llama_token_data>,
    {
        let data: Vec<llama_token_data> = it.into_iter().collect();
        Self::from_vec_unchecked(data)
    }
}

impl Index<usize> for Candidates {
    type Output = llama_token_data;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl FromIterator<llama_token_data> for Candidates {
    fn from_iter<T: IntoIterator<Item = llama_token_data>>(iter: T) -> Self {
        Self::from_vec(iter.into_iter().collect())
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

        let sorted = candidates.sort(Sorted::ByLogit {
            k: NonZeroUsize::new(3).unwrap(),
        });
        assert_eq!(sorted.len(), NonZeroUsize::new(3).unwrap());

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

        let c = c.softmax(None);

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
        assert_eq!(token.is_one().unwrap().id, 0);
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
        assert_eq!(tokens.len(), NonZeroUsize::new(3).unwrap());
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
        assert_eq!(tokens.len(), NonZeroUsize::new(3).unwrap());

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
        assert_eq!(tokens.len(), NonZeroUsize::new(4).unwrap());

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
        assert_eq!(tokens.len(), NonZeroUsize::new(5).unwrap());
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
        assert_eq!(tokens.len(), NonZeroUsize::new(5).unwrap());
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
