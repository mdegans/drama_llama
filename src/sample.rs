use crate::{
    candidates::Sorted,
    data::stopwords::StopWords,
    model::Vocab,
    ngram::{NGram, NGramStats},
    Candidates, Probability,
};

use llama_cpp_sys_3::llama_token;
use xorshift::Rng;

use std::num::{NonZeroU8, NonZeroUsize};

#[cfg_attr(
    feature = "serde",
    derive(rocket::serde::Deserialize, rocket::serde::Serialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "rocket::serde"))]
/// Options determining how raw logits are turned into a token. This is used by
/// [`Candidates::sample_token`] and associated functions.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct SampleOptions {
    /// Sampling modes to apply in order. Greedy, Mirostat, and MirostatV2 are
    /// guaranteed to return a single token, so they should be the last mode.
    // TODO: There may be a way to refactor mirostat and mirostat v2 to return
    // candidates instead of a single token. Issue is they rely on a suprise
    // value that is calculated in the function after the token is chosen, so
    // this would have to occur at the beginning of the function, but not on the
    // first call. It's doable, but it's a bit of a pain. It may be worth it.
    pub modes: Vec<SamplingMode>,
    /// Repetition penalty options. If this is `None`, no repetition penalty is
    /// applied. This is applied before the sampling modes, so it may be used
    /// with any of them, including greedy.
    pub repetition: Option<RepetitionOptions>,
}

impl SampleOptions {
    /// Greedy sampling. No repetition penalty.
    pub fn greedy() -> Self {
        Self {
            modes: vec![SamplingMode::Greedy],
            repetition: None,
        }
    }
}

#[cfg_attr(
    feature = "serde",
    derive(rocket::serde::Deserialize, rocket::serde::Serialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "rocket::serde"))]
#[derive(Clone, Debug, PartialEq, Default)]
// TODO: add `min_keep` and `mad_keep` to all the sampling modes since it's
// doable and it would be nice to have a more consistent API.
pub enum SamplingMode {
    /// Greedy sampling. The most likely next token is always chosen. Not very
    /// useful unless you want to regurgitate the training data.
    #[default]
    Greedy,
    /// Top-p sampling. A token is chosen from the top tokens whose cumulative
    /// probability is greater than or equal to `p`.
    TopP {
        /// Reasonable values are between 0.9 and 0.95. Higher means more
        /// diversity, but potentially less coherent.
        p: Probability<f64>,
        /// Minimum number of candidates to keep per token.
        min_keep: NonZeroUsize,
    },
    /// A token is chosen from the top `k` tokens. This is not very good.
    /// Reasonable values are between 30 and 40.
    TopK {
        /// The top `k` tokens are kept.
        k: NonZeroUsize,
    },
    /// Min-p sampling. `p` sets the minimum probability to keep a token. Below
    /// that the tail is cut off. `p` is scaled by the top token's probability
    /// to balance diversity and quality.
    ///
    /// It is described in detail in the following pull request:
    /// <https://github.com/ggerganov/llama.cpp/pull/3841>
    MinP {
        /// The minimum probability to keep a token. This is scaled by the top
        /// token's probability. Reasonable values are 0.05 to 0.3. Higher means
        /// less diversity.
        p: Probability<f32>,
        min_keep: NonZeroUsize,
    },
    /// Tail free sampling.
    ///
    /// "TFS first converts logits output by a model into probabilities using
    /// the softmax function before sorting them in descending order. It then
    /// calculates the first and second derivatives. As the tokens are discrete,
    /// this can be found with subtraction. The magnitude of each second
    /// derivative is then taken and normalized so that they sum to 1. Finally,
    /// a threshold z is used to determine what part of the cumulative
    /// distribution of the second derivative weights to define the “tail” of
    /// the distribution to be at."
    ///
    /// <https://www.trentonbricken.com/Tail-Free-Sampling/.>
    TailFree {
        /// Reasonable values are between 0.25 and 0.75. The higher, the more
        /// diverse the output, but also potentially less coherent.
        // TODO(mdegans): verify this is correct, read the article. From the
        // figures, it seems correct, but the colors are hard to distinguish
        // (for me).
        z: Probability<f32>,
        /// Minimum number of candidates to keep per token.
        min_keep: NonZeroUsize,
    },
    /// Locally typical sampling.
    ///
    /// "First, we compute the conditional entropy, which is an O(|V|)
    /// operation. Second, we sort words by their absolute distance from H(pb(·|
    /// Y <t = y<t)), which can be done in O(|V| log |V|) time with standard
    /// sorting algorithms. Finally, we greedily take words from this list until
    /// their cumulative probability exceeds the threshold `p` , which again
    /// takes O(|V|) time. Thus, creating our altered distribution has time
    /// complexity O(|V| log |V|)."
    ///
    /// <https://arxiv.org/pdf/2202.00666.pdf>
    LocallyTypical {
        /// Probability. Reasonable values are between 0.2 and 0.95. For story
        /// generation, lower is better. For summarization, higher is better.
        p: Probability<f32>,
        /// Minimum number of candidates to keep per token.
        min_keep: NonZeroUsize,
    },
    /// Mirostat sampling.
    ///
    /// "a neural text decoding algorithm that directly controls the perplexity
    /// of the generated text over a wide range of text length. Notably, for
    /// longer texts and certain ranges of input parameters, top-k and top-p
    /// sampling fall into boredom and confusion traps which cause low-quality
    /// texts; Mirostat avoids both traps."
    ///
    /// <https://arxiv.org/pdf/2007.14966.pdf>
    Mirostat {
        /// Tau. Target entropy. A good value is 3.0 according to this paper:
        /// <https://arxiv.org/pdf/2202.00666.pdf>
        ///
        /// `llama.cpp` uses a default of 5.0.
        tau: f32,
        /// Eta. Learning rate. A good value is 0.1.
        eta: f32,
        /// Maximum number of candidates to keep. In the original paper and code
        /// the default is 100 and the name is `m`.
        max_keep: Option<NonZeroUsize>,
    },
    /// Mirostat V.2 sampling.
    ///
    /// "Here we provide an alternate algorithm for perplexity control, Alg. 2,
    /// which does not depend on the distribution of the underlying LM. In this
    /// sense, Alg. 2 controls perplexity in more general sequential generative
    /// models than Alg. 1 where the underlying distribution may not be Zipfian.
    /// In our work, we choose Alg. 1 since it has only an additional constant
    /// time complexity compared to top-k sampling. Whereas Alg. 2 has
    /// additional time complexity that depends on target cross-entropy rate and
    /// vocabulary size, which may vary with different LMs."
    ///
    /// # Note:
    /// * The bit about time complexity is not relevant to this implementation
    ///   since we truncate the candidates to a fixed size like v1.
    ///
    /// <https://arxiv.org/pdf/2007.14966.pdf>
    MirostatV2 {
        /// Tau. Target entropy. A good value is 3.0 according to the paper and
        /// HF's experiments in <https://arxiv.org/pdf/2202.00666.pdf>
        ///
        /// `llama.cpp` uses a default of 5.0.
        tau: f32,
        /// Eta. Learning rate. A good value is 0.1.
        eta: f32,
        /// Maximum number of candidates to keep. Defaults to 100. The original
        /// implementation does not support this. If identical behavior is
        /// desired, set this to the vocabulary size.
        max_keep: Option<NonZeroUsize>,
    },
    /// Split P sampling. This cuts the tail off where the difference between
    /// adjacent probabilities is greatest, where the slope is steepest.
    SplitP {
        /// Minimum number of candidates to keep.
        min_keep: NonZeroUsize,
        /// Maximum number of candidates to keep.
        max_keep: Option<NonZeroUsize>,
    },
    /// Split L sampling. This cuts the tail off where the difference between
    /// adjacent logits is greatest, where the slope is steepest.
    SplitL {
        /// Minimum number of candidates to keep.
        min_keep: NonZeroUsize,
        /// Maximum number of candidates to keep.
        max_keep: Option<NonZeroUsize>,
    },
}

impl SamplingMode {
    /// Default top-p sampling: p = 0.9 with no minimum keep.
    pub const fn top_p() -> Self {
        Self::TopP {
            p: Probability { p: 0.9 },
            // Verbosity because const unwrap is not stable for no good reason.
            // the code is literally this for Option<T>:
            min_keep: match NonZeroUsize::new(1) {
                Some(min_keep) => min_keep,
                None => panic!("NonZeroUsize::new(1) failed"),
            },
        }
    }

    /// Default top-k sampling: k = 35.
    pub const fn top_k() -> Self {
        Self::TopK {
            k: match NonZeroUsize::new(35) {
                Some(k) => k,
                None => panic!("NonZeroUsize::new(35) failed"),
            },
        }
    }

    /// Default min-p sampling: p = 0.05 with no minimum keep.
    pub const fn min_p() -> Self {
        Self::MinP {
            p: Probability { p: 0.05 },
            min_keep: match NonZeroUsize::new(1) {
                Some(min_keep) => min_keep,
                None => panic!("NonZeroUsize::new(1) failed"),
            },
        }
    }

    /// Default tail free sampling: z = 0.5 with no minimum keep.
    pub const fn tail_free() -> Self {
        Self::TailFree {
            z: Probability { p: 0.5 },
            min_keep: match NonZeroUsize::new(1) {
                Some(min_keep) => min_keep,
                None => panic!("NonZeroUsize::new(1) failed"),
            },
        }
    }

    /// Default locally typical sampling: p = 0.5 with no minimum keep.
    pub const fn locally_typical() -> Self {
        Self::LocallyTypical {
            p: Probability { p: 0.5 },
            min_keep: match NonZeroUsize::new(1) {
                Some(min_keep) => min_keep,
                None => panic!("NonZeroUsize::new(1) failed"),
            },
        }
    }

    /// Default mirostat sampling: tau = 3.0, eta = 0.1, max_keep = 100.
    pub const fn mirostat() -> Self {
        Self::Mirostat {
            tau: 3.0,
            eta: 0.1,
            max_keep: match NonZeroUsize::new(100) {
                Some(max_keep) => Some(max_keep),
                None => panic!("NonZeroUsize::new(100) failed"),
            },
        }
    }

    /// Default mirostat v2 sampling: tau = 3.0, eta = 0.1, max_keep = 100.
    pub const fn mirostat_v2() -> Self {
        Self::MirostatV2 {
            tau: 3.0,
            eta: 0.1,
            max_keep: match NonZeroUsize::new(100) {
                Some(max_keep) => Some(max_keep),
                None => panic!("NonZeroUsize::new(100) failed"),
            },
        }
    }

    /// Default split p sampling: min_keep = 1, max_keep = 50.
    pub const fn split_p() -> Self {
        Self::SplitP {
            min_keep: match NonZeroUsize::new(1) {
                Some(min_keep) => min_keep,
                None => panic!("NonZeroUsize::new(1) failed"),
            },
            max_keep: match NonZeroUsize::new(50) {
                Some(max_keep) => Some(max_keep),
                None => panic!("NonZeroUsize::new(50) failed"),
            },
        }
    }

    /// Default split l sampling: min_keep = 1, max_keep = 50.
    pub const fn split_l() -> Self {
        Self::SplitL {
            min_keep: match NonZeroUsize::new(1) {
                Some(min_keep) => min_keep,
                None => panic!("NonZeroUsize::new(1) failed"),
            },
            max_keep: match NonZeroUsize::new(50) {
                Some(max_keep) => Some(max_keep),
                None => panic!("NonZeroUsize::new(50) failed"),
            },
        }
    }
}

#[derive(Debug, thiserror::Error, derive_more::From)]
pub enum SampleError {
    #[error("Sampling failed because of a repetition error: {err}")]
    RepetitionError { err: RepetitionError },
}

#[cfg_attr(
    feature = "serde",
    derive(rocket::serde::Deserialize, rocket::serde::Serialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "rocket::serde"))]
/// Options for `apply_sample_repetition_penalties`.
#[derive(Clone, Debug, PartialEq)]
pub struct RepetitionOptions {
    /// [`NGram`]s to ignore. These are never penalized.
    pub(crate) ignored: Vec<NGram>,
    // TODO: Add this back in when we have a way to prune the ngram stats. We
    // will likely have to add `last_seen` to the `NGramData` struct. However
    // this will make the code O(n) instead of O(1) for each token, which is as
    // bad as the original C++ code. Not great, not terrible. Also not a
    // priority. There may also be a better structure than a HashMap for the
    // NGramStats.
    // /// Penalty of last n tokens. Outside of this window, NGram penalties are
    // /// forgotten. This is useful for penalizing repetition in a sliding window
    // /// of tokens.
    // ///
    // /// Be default the [`NgramStats`] is never pruned and will continue to
    // /// grow forever as more tokens are added.
    // pub(crate) penalty_last_n: NonZeroUsize,
    /// The maximum number of times an item can be repeated before it is
    /// penalized. This can be used to allow some n-grams to be repeated in
    /// situations where repetition is desirable for the purpose, like song
    /// lyrics.
    pub(crate) penalty_max_count: NonZeroU8,
    /// NGram minimum size. Max size is [`NGram::CAPACITY`] by default.
    pub(crate) ngram_min_size: NonZeroU8,
    /// NGram maximum size, capped at [`NGram::CAPACITY`].
    pub(crate) ngram_max_size: NonZeroU8,
    /// Repetition penalty. A reasonable value is 1.15.
    pub(crate) penalty_repeat: f32,
    /// The penalty for the frequency of the n-grams. This subtracts the
    /// frequency of the n-gram from the logit of the penalized token in the
    /// n-gram.
    pub(crate) penalty_freq: f32,
    /// The penalty for the presence of the n-grams. This subtracts the presence
    /// of the n-gram from the logit of the penalized token in the n-gram.
    pub(crate) penalty_present: f32,
}

impl Default for RepetitionOptions {
    fn default() -> Self {
        Self {
            ignored: vec![NGram::from(13)],
            penalty_max_count: NonZeroU8::new(1).unwrap(),
            ngram_min_size: NonZeroU8::new(1).unwrap(),
            ngram_max_size: NonZeroU8::new(4).unwrap(),
            penalty_repeat: 1.15,
            penalty_freq: 0.0,
            penalty_present: 0.0,
        }
    }
}

impl RepetitionOptions {
    /// Tokens to ignore. These tokens are never penalized.
    ///
    /// # Notes:
    /// * any tokens in [`PredictOptions::stop_sequences`], including EOS, are
    /// automatically ignored.
    /// * The returned slice is guaranteed to be sorted.
    ///
    /// [`PredictOptions::stop_sequences`]: crate::PredictOptions::stop_sequences
    pub fn ignored(&self) -> &[NGram] {
        &self.ignored
    }

    /// Ngrams to ignore. These ngrams are never penalized. The input can be any
    /// iterable of any type that can be converted into an ngram.
    ///
    /// # Notes:
    /// * any tokens in [`PredictOptions::stop_sequences`], including EOS, are
    /// automatically ignored.
    ///
    /// [`PredictOptions::stop_sequences`]: crate::PredictOptions::stop_sequences
    pub fn set_ignored<It, Ng>(mut self, ignored: It) -> Self
    where
        It: IntoIterator<Item = Ng>,
        Ng: Into<NGram>,
    {
        // If the type is the same, the compiler will optimize this to a no-op,
        // at least hopefully, since we can reason about it and compiler
        // developers are smart.
        self.ignored = ignored.into_iter().map(|t| t.into()).collect();
        self.ignored.sort();
        self
    }

    /// Extend the list of ignored ngrams. Input can be any iterable of any type
    /// that can be converted into an [`NGram`].
    ///
    /// # Deprecation:
    /// In the future, to be more consistent, this will take and return `self`.
    pub fn extend_ignored<It, Ng>(&mut self, ignored: It)
    where
        It: IntoIterator<Item = Ng>,
        Ng: Into<NGram>,
    {
        self.ignored.extend(ignored.into_iter().map(|t| t.into()));
        self.ignored.sort();
    }

    /// Add a token to the list of ignored tokens. that can be converted into an
    /// [`NGram`].
    ///
    /// Prefer using [`extend_ignored`] or [`set_ignored`] for multiple tokens.
    ///
    /// # Deprecation:
    /// In the future, to be more consistent, this will take and return `self`.
    ///
    /// [`extend_ignored`]: Self::extend_ignored
    /// [`set_ignored`]: Self::set_ignored
    pub fn add_ignored<I>(&mut self, ngram: I)
    where
        I: Into<NGram>,
    {
        self.ignored.push(ngram.into());
        self.ignored.sort();
    }

    /// Ignore [`StopWords`]. This should not be confused with stop sequences.
    /// These are commonly used words that are often ignored in text generation.
    pub fn ignore_stopwords(
        mut self,
        stopwords: StopWords,
        model: &crate::Model,
    ) -> Self {
        self.extend_ignored(stopwords.into_tokens(model));
        self
    }

    // TODO: ngram penality options

    /// The maximum number of times an item can be repeated before it is
    /// penalized.
    pub fn penalty_max_count(&self) -> NonZeroU8 {
        self.penalty_max_count
    }

    /// Set the maximum number of times an item can be repeated before it is
    /// penalized.
    pub fn set_penalty_max_count(
        mut self,
        penalty_max_count: NonZeroU8,
    ) -> Self {
        self.penalty_max_count = penalty_max_count;
        self
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RepetitionError {
    #[error("Too few candidates ({actual}). The minimum is {min}. It cannot be less than the n-gram size.")]
    TooFewCandidates { min: usize, actual: usize },
    #[error("The penalize n-gram index ({penalize_ngram_index}) is greater than or equal to the n-gram size ({n_gram_size}).")]
    PenalizeNgramIndexBounds {
        n_gram_size: usize,
        penalize_ngram_index: u8,
    },
}

fn ngram_is_ignored(ngram: NGram, ignored: &[NGram]) -> bool {
    if ignored.len() > 64 {
        // binary search is faster for large lists
        ignored.binary_search(&ngram).is_ok()
    } else {
        ignored.contains(&ngram)
    }
}

/// Apply repetition penalties to the candidates.
///
/// This is mostly a translation of the C++ code in `llama.cpp` with support for
/// n-grams. [`NGramStats`] is used to store the n-gram statistics.
pub fn apply_sample_repetition_ngram(
    candidates: Candidates,
    tokens: &[llama_token],
    opts: &RepetitionOptions,
    freq_map: &mut NGramStats,
) -> Result<Candidates, RepetitionError> {
    let k = candidates.len();
    let mut candidates = candidates.sort(Sorted::ById { k });

    let RepetitionOptions {
        ignored,
        ngram_max_size,
        ngram_min_size,
        penalty_freq,
        penalty_max_count,
        penalty_present,
        mut penalty_repeat,
    } = opts;

    let ngram_min_size: usize = ngram_min_size
        .get()
        .max(1)
        .min(NGram::CAPACITY as u8)
        .try_into()
        .unwrap();
    let ngram_max_size: usize = ngram_max_size
        .get()
        .max(1)
        .min(NGram::CAPACITY as u8)
        .try_into()
        .unwrap();
    let penalty_max_count: usize = penalty_max_count.get().into();

    // Iterate over possible n-grams at the end of the tokens in order of
    // smallest to largest. Larger n-grams are more penalized than smaller.
    for slice in (ngram_min_size..ngram_max_size)
        .filter_map(|n| tokens.get(tokens.len() - n..))
    {
        let ngram = NGram::try_from(slice).unwrap();

        // If either the ngram or the penalized token is ignored, skip the ngram
        if ngram_is_ignored(ngram, &ignored) {
            continue;
        }

        // Search from the end of the slice for a token that is not ignored.
        let mut penalized_token = None;
        for &token in slice.iter().rev() {
            if ngram_is_ignored(token.into(), &ignored) {
                continue;
            }
            penalized_token = Some(token);
        }
        let penalized_token = match penalized_token {
            Some(token) => token as usize,
            // If there are no tokens that are not ignored, skip the ngram.
            None => continue,
        };

        // This counts the ngram and gets mutable data about it.
        // TODO: we added as weight member to the NGramData but it is unused. We
        // can use the weighted probability to penalize the n-gram in addition
        // to the count.
        let data = freq_map.add(ngram, &candidates);

        let candidate = &mut candidates.data[penalized_token];

        // The logic here is copied from the c++ code in llama.cpp.. which was
        // broken. It was fixed in the c++ so we fix it here. It looked wrong.
        if data.count() > penalty_max_count {
            // dbg!(count, &ngram, &candidate);
            if candidate.logit <= 0.0 {
                candidate.logit *= penalty_repeat.powf(ngram.len() as f32);
            } else {
                candidate.logit /= penalty_repeat;
            }
        }

        candidate.logit -= (data.count() as f32) * penalty_freq
            + (data.count() as f32) * penalty_present;

        // We penalize longer ngrams more. Because we work backwards, this will
        // penalize the first token in the ngram the most.
        penalty_repeat *= penalty_repeat;

        // We have modified the logit values, so we need to reapply softmax.
        candidates.softmax_applied_to = None;
    }

    Ok(candidates)
}

/// Sample a token from the candidates.
pub(crate) fn sample_token(
    tokens: &[llama_token],
    mut candidates: Candidates,
    vocab: &Vocab,
    opts: &SampleOptions,
    freq_map: &mut NGramStats,
    rng: &mut xorshift::Xoroshiro128,
    mu: &mut Option<f32>,
) -> Result<llama_token, SampleError> {
    // Ban tokens and ngrams from the candidates.
    let min_logit = if candidates
        .is_sorted()
        .by_logit()
        .is_some_and(|until| until == candidates.len())
    {
        candidates.data.last().unwrap().logit
    } else {
        candidates
            .data
            .iter()
            .min_by(|a, b| a.logit.partial_cmp(&b.logit).unwrap())
            .unwrap()
            .logit
    };
    for (token, allowed) in candidates.iter_mut().zip(vocab.allowed_tokens()) {
        if !allowed {
            // The individual token is banned
            token.logit = min_logit;
        } else {
            if let Some(banned) = vocab.banned() {
                // There are some banned ngrams
                if let Some(&last_token) = tokens.last() {
                    // and there is a previous token
                    let ngram = [last_token, token.id];
                    if banned.as_slice().binary_search(&ngram).is_ok() {
                        // And the ngram is banned, ban the token
                        // TODO: we could also remove the previous tokens from
                        // the tokens but that doesn't suit our current design.
                        // FIXME: it does now but there is still work to do.
                        token.logit = min_logit;
                    }
                }
            }
        }
    }

    // Apply any repetition penalties to the candidates. This also applies the
    // softmax and sorts the candidates by logit where the most likely token is
    // first.
    if let Some(repetition) = &opts.repetition {
        candidates = apply_sample_repetition_ngram(
            candidates, tokens, repetition, freq_map,
        )?;
    }

    // Fold candidates, applying the sampling modes in order.
    let filtered =
        opts.modes
            .iter()
            .cloned()
            .fold(candidates, |candidates, mode| match mode {
                SamplingMode::Greedy => candidates.sample_token_greedy(),
                SamplingMode::TopP { p, min_keep } => {
                    candidates.top_p(p, min_keep)
                }
                SamplingMode::TopK { k } => candidates.top_k(k),
                SamplingMode::MinP { p, min_keep } => {
                    candidates.min_p(p, min_keep)
                }
                SamplingMode::TailFree { z, min_keep } => {
                    candidates.tail_free(z, min_keep)
                }
                SamplingMode::LocallyTypical { p, min_keep } => {
                    candidates.locally_typical(p, min_keep)
                }
                SamplingMode::Mirostat { tau, eta, max_keep } => {
                    candidates.mirostat(rng, tau, eta, max_keep, mu)
                }
                SamplingMode::MirostatV2 { tau, eta, max_keep } => {
                    candidates.mirostat_v2(rng, tau, eta, max_keep, mu)
                }
                SamplingMode::SplitP { min_keep, max_keep } => {
                    candidates.split_p(min_keep, max_keep)
                }
                SamplingMode::SplitL { min_keep, max_keep } => {
                    candidates.split_l(min_keep, max_keep)
                }
            });

    Ok(choose_candidate(rng, filtered.softmax(None))
        .is_one()
        .unwrap()
        .id)
}

/// Apply the softmax function to the remaining candidates and select a single
/// candidate. This function is guaranteed to leave the candidates with only
/// one token.
// TODO: better name
pub(crate) fn choose_candidate(
    rng: &mut xorshift::Xoroshiro128,
    candidates: Candidates,
) -> Candidates {
    if candidates.len().get() == 1 {
        return candidates;
    }

    let candidates = candidates.softmax(None);

    // Pick a token based on the probabilities
    let val = rng.gen_range(0.0, 1.0);
    let mut cum_prob = 0.0;
    for (i, token) in candidates.iter().enumerate() {
        cum_prob += token.p;
        if cum_prob > val {
            return candidates.select(i);
        }
    }

    // This can happen because of floating point errors
    let last = candidates.len().get() - 1;
    candidates.select(last)
}
