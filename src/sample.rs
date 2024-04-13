use crate::{
    candidates::Sorted,
    data::stopwords::StopWords,
    model::Vocab,
    ngram::{NGram, NGramStats},
    Candidates, Probability,
};

use llama_cpp_sys::{llama_token, llama_token_data};
use xorshift::Rng;

use std::{
    borrow::Cow,
    num::{NonZeroU8, NonZeroUsize},
};

#[cfg_attr(
    feature = "serde",
    derive(rocket::serde::Deserialize, rocket::serde::Serialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "rocket::serde"))]
/// Options for [`sample`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct SampleOptions {
    pub mode: SamplingMode,
    pub repetition: Option<RepetitionOptions>,
}

impl SampleOptions {
    /// Greedy sampling. No repetition penalty.
    pub fn greedy() -> Self {
        Self {
            mode: SamplingMode::Greedy,
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
pub enum SamplingMode {
    /// Greedy sampling. The most likely next token is always chosen.
    /// Probability is always 0.0 since it is not calculated.
    #[default]
    Greedy,
    /// Top-p sampling. Returns the top tokens whose cumulative probability is
    /// less than or equal to `p`.
    TopP {
        /// The top tokens whose cumulative probability exceeds `p` are kept.
        p: Probability<f64>,
        /// Minimum number of candidates to keep per token.
        min_keep: NonZeroUsize,
    },
    /// Top-k sampling. Returns the top `k` most likely tokens. If `k` is
    /// greater than the number of candidates, all candidates are returned.
    TopK {
        /// The top `k` tokens are kept.
        k: NonZeroUsize,
    },
    /// Min-p sampling. Returns the tokens whose logits are greater than or
    /// equal to the minimum logit required to select `min_keep` tokens with
    /// probability `p`. None is returned if no tokens can be selected.
    MinP {
        /// The minimum probability to keep a token.
        p: Probability<f32>,
        min_keep: NonZeroUsize,
    },
    /// Tail free sampling. Returns the tokens whose second derivatives are
    /// greater than or equal to `z`.
    TailFree {
        /// The minimum probability to keep a token.
        z: Probability<f32>,
        /// Minimum number of candidates to keep per token.
        min_keep: NonZeroUsize,
    },
    /// Locally typical sampling.
    LocallyTypical {
        /// Probability
        p: Probability<f32>,
        /// Minimum number of candidates to keep per token.
        min_keep: NonZeroUsize,
    },
    /// Mirostat sampling.
    Mirostat {
        /// Tau
        tau: f32,
        /// Eta
        eta: f32,
        /// M
        m: NonZeroUsize,
        /// Mu
        mu: f32,
    },
    /// Mirostat V.2 sampling.
    MirostatV2 {
        /// Tau
        tau: f32,
        /// Eta
        eta: f32,
        /// Mu
        mu: f32,
    },
    /// Split P sampling.
    SplitP {
        /// Minimum number of candidates to keep.
        min_keep: NonZeroUsize,
        /// Maximum number of candidates to keep.
        max_keep: Option<NonZeroUsize>,
    },
    /// Split L sampling.
    SplitL {
        /// Minimum number of candidates to keep.
        min_keep: NonZeroUsize,
        /// Maximum number of candidates to keep.
        max_keep: Option<NonZeroUsize>,
    },
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
    pub fn extend_ignored<It, Ng>(&mut self, ignored: It)
    where
        It: IntoIterator<Item = Ng>,
        Ng: Into<NGram>,
    {
        self.ignored.extend(ignored.into_iter().map(|t| t.into()));
        self.ignored.sort();
    }

    /// Add a token to the list of ignored tokens.
    ///
    /// # Notes:
    /// * The time complexity of this addition is O(n log n), so prefer to use
    /// [`set_ignored`] or [`extend_ignored`] if you have multiple tokens to
    /// add.
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
    candidates: &mut Candidates,
    tokens: &[llama_token],
    opts: &RepetitionOptions,
    freq_map: &mut NGramStats,
) -> Result<(), RepetitionError> {
    candidates.sort(Sorted::ById {
        k: candidates.len(),
    });

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
        let count = freq_map.add(ngram, candidates).count();

        let candidate = &mut candidates.data[penalized_token];

        // The logic here is copied from the c++ code in llama.cpp.. which was
        // broken. It was fixed in the c++ so we fix it here. It looked wrong.
        if count > penalty_max_count {
            // dbg!(count, &ngram, &candidate);
            if candidate.logit <= 0.0 {
                candidate.logit *= penalty_repeat.powf(ngram.len() as f32);
            } else {
                candidate.logit /= penalty_repeat;
            }
        }

        candidate.logit -=
            (count as f32) * penalty_freq + (count as f32) * penalty_present;

        // We penalize longer ngrams more. Because we work backwards, this will
        // penalize the first token in the ngram the most.
        penalty_repeat *= penalty_repeat;
    }

    // We have modified the logit values, so we need to reapply softmax.
    candidates.arr.sorted = false;
    candidates.softmax_applied_to = None;

    Ok(())
}

/// Sample a token from the candidates.
pub(crate) fn sample_token(
    tokens: &[llama_token],
    candidates: &mut Candidates,
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
        apply_sample_repetition_ngram(
            candidates, tokens, repetition, freq_map,
        )?;
    }

    let filtered: Cow<'_, [llama_token_data]> = match opts.mode {
        SamplingMode::Greedy => return Ok(candidates.sample_token_greedy().id),
        SamplingMode::TopP { p, min_keep } => {
            candidates.top_p(p, min_keep).into()
        }
        SamplingMode::TopK { k } => candidates.top_k(k).into(),
        SamplingMode::MinP { p, min_keep } => candidates.min_p(p, min_keep),
        SamplingMode::TailFree { z, min_keep } => {
            candidates.tail_free(z, min_keep).into()
        }
        SamplingMode::LocallyTypical { p, min_keep } => {
            candidates.locally_typical(p, min_keep).into()
        }
        SamplingMode::Mirostat {
            tau,
            eta,
            m,
            mu: initial_mu,
        } => {
            let mu = mu.get_or_insert(initial_mu);
            return Ok(candidates.mirostat(rng, tau, eta, m, mu));
        }
        SamplingMode::MirostatV2 {
            tau,
            eta,
            mu: initial_mu,
        } => {
            let mu = mu.get_or_insert(initial_mu);
            return Ok(candidates.mirostat_v2(rng, tau, eta, mu));
        }
        SamplingMode::SplitP { min_keep, max_keep } => {
            candidates.split_p(min_keep, max_keep).into()
        }
        SamplingMode::SplitL { min_keep, max_keep } => {
            candidates.split_l(min_keep, max_keep).into()
        }
    };

    let mut filtered = filtered.to_vec();

    // Check that the filtered candidates are sorted by logit
    debug_assert!(filtered.windows(2).all(|w| w[0].logit >= w[1].logit));

    Ok(predict_token(rng, &mut filtered).id)
}

/// Apply the softmax function to the remaining candidates and choose one based
/// on weighted probabilities.
pub(crate) fn predict_token(
    rng: &mut xorshift::Xoroshiro128,
    tokens: &mut [llama_token_data],
) -> llama_token_data {
    // Recalculate probabilities
    let max_logit = tokens.first().unwrap().logit;
    let cum_prob = tokens.iter_mut().fold(0.0, |sum, token| {
        token.p = (token.logit - max_logit).exp();
        sum + token.p
    });
    for token in tokens.iter_mut() {
        token.p /= cum_prob;
    }

    // Pick a token based on the probabilities
    let val = rng.gen_range(0.0, 1.0);
    let mut cum_prob = 0.0;
    for token in tokens.iter() {
        debug_assert!(token.p >= 0.0);
        debug_assert!(token.p <= 1.0);
        cum_prob += token.p;
        if val < cum_prob {
            return *token;
        }
    }

    // This can happen because of floating point errors
    *tokens.last().unwrap()
}
