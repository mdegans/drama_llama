//! Repetition-penalty sampling helpers.
//!
//! Extracted from `sample.rs` in the JSON-grammar refactor; semantics
//! unchanged.

use crate::{
    candidates::Sorted,
    data::stopwords::StopWords,
    ngram::{NGram, NGramStats},
    Candidates,
};

use llama_cpp_sys_3::llama_token;

use std::num::NonZeroU8;

#[cfg(feature = "egui")]
use super::DELETE_ICON;

#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize)
)]
/// Options for `apply_sample_repetition_penalties`.
#[derive(Clone, Debug, PartialEq)]
pub struct RepetitionOptions {
    /// Stopwords to ignore, by language. These are never penalized.
    #[cfg_attr(feature = "serde", serde(default))]
    pub(crate) ignored_stopwords: Vec<StopWords>,
    /// [`NGram`]s to ignore. These are never penalized.
    #[cfg_attr(feature = "serde", serde(default))]
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
    /// Surgical mode. Instead of penalizing all seen tokens, only penalize the
    /// token that would *complete* a repeated n-gram pattern. For example, if
    /// "The New" was just generated and "The New York" is a known trigram,
    /// penalize "York" specifically to prevent the phrase from completing.
    /// This is more targeted than the default broad approach.
    #[cfg_attr(feature = "serde", serde(default))]
    pub(crate) surgical: bool,
}

impl Default for RepetitionOptions {
    fn default() -> Self {
        Self {
            ignored_stopwords: vec![],
            ignored: vec![],
            penalty_max_count: NonZeroU8::new(1).unwrap(),
            ngram_min_size: NonZeroU8::new(1).unwrap(),
            ngram_max_size: NonZeroU8::new(4).unwrap(),
            penalty_repeat: 1.15,
            penalty_freq: 0.0,
            penalty_present: 0.0,
            surgical: false,
        }
    }
}

impl RepetitionOptions {
    /// Stopwords to ignore (by language). These are never penalized.
    pub fn ignored_stopwords(&self) -> &[StopWords] {
        &self.ignored_stopwords
    }

    /// Set the stopwords to ignore. These are never penalized.
    pub fn set_ignored_stopwords(
        mut self,
        ignored_stopwords: Vec<StopWords>,
    ) -> Self {
        self.ignored_stopwords = ignored_stopwords;
        self
    }

    /// Extend the list of ignored stopwords.
    pub fn extend_ignored_stopwords<It>(&mut self, ignored_stopwords: It)
    where
        It: IntoIterator<Item = StopWords>,
    {
        self.ignored_stopwords.extend(ignored_stopwords);
    }

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

    /// The multiplicative penalty applied to repeated tokens. `1.0` is
    /// no penalty; `1.15` is the old aggressive default; `1.05`–`1.1`
    /// is a reasonable chat range.
    pub fn penalty_repeat(&self) -> f32 {
        self.penalty_repeat
    }

    /// Set the multiplicative penalty applied to repeated tokens.
    pub fn set_penalty_repeat(mut self, penalty_repeat: f32) -> Self {
        self.penalty_repeat = penalty_repeat;
        self
    }

    /// Draw [`egui::Ui`] for [`RepetitionOptions`] without the outer
    /// [`egui::CollapsingHeader`].
    #[cfg(feature = "egui")]
    pub fn draw_inner(&mut self, ui: &mut egui::Ui) -> egui::Response {
        // FIXME: for internationalization, we should put all the strings in a
        // separate file and use gettext or similar. There may be something
        // better from the Rust ecosystem, but I'm not aware of it.
        const STOPWORD_HELP: &str = "Stopwords are common words that are often ignored in NLP tasks. They shouldn't be confused with stop sequences, which are used to terminate generation. These options allow you to ignore stopwords for the purpose of penalizing repetition. The idea is to allow a higher repetition penalty without penalizing common words. This is experimental.";

        // Ignored Stopwords
        if !self.ignored_stopwords.is_empty() {
            ui.label("Stopwords ignored for:")
                .on_hover_text_at_pointer(STOPWORD_HELP);
            let mut to_remove = None;
            for (i, language) in self.ignored_stopwords.iter().enumerate() {
                ui.horizontal(|ui| {
                    if ui
                        .add(egui::Button::image(DELETE_ICON))
                        .on_hover_text_at_pointer("Remove this language from the list of ignored stopwords.")
                        .clicked() {
                        to_remove = Some(i);
                    };
                    ui.label(language.as_str())
                });
            }
            if let Some(i) = to_remove {
                self.ignored_stopwords.remove(i);
            }
        }
        if self.ignored_stopwords.len() < StopWords::ALL.len() {
            egui::ComboBox::from_label("to ignore stopwords for")
                .selected_text("Select a language...")
                .show_ui(ui, |ui| {
                    for language in StopWords::ALL {
                        if !self.ignored_stopwords.contains(&language) {
                            if ui
                                .selectable_label(false, language.as_str())
                                .clicked()
                            {
                                self.ignored_stopwords.push(language);
                                self.ignored_stopwords.sort();
                            }
                        }
                    }
                })
                .response
                .on_hover_text_at_pointer(STOPWORD_HELP);
        }

        // Ignored ngrams
        if !self.ignored.is_empty() {
            ui.label("Ignored NGrams").on_hover_text_at_pointer("These ngrams are never penalized. Common words can be put here to prevent them from being penalized. This is experimental.");
            let mut to_remove = None;
            for (i, ngram) in self.ignored.iter().enumerate() {
                ui.horizontal(|ui| {
                    if ui
                        .add(egui::Button::image(DELETE_ICON))
                        .on_hover_text_at_pointer("Remove this ngram from the list of ignored ngrams.")
                        .clicked() {
                        to_remove = Some(i);
                    };
                    ui.label(format!("{:?}", ngram))
                });
            }
            // TODO: add a way to add ngrams. This requires a buffer to store
            // the input. We can use &mut function argument for this since we
            // don't have a place on this struct to store it. We could also
            // serde(skip) it.

            if let Some(i) = to_remove {
                self.ignored.remove(i);
            }
        }

        // Penalty max count
        let mut resp = ui.horizontal(|ui| {
            let mut max_count = self.penalty_max_count.get();
            let inner = ui.label("Penalty Max Count") |
            ui.add(egui::DragValue::new(&mut max_count).clamp_range(1..=255))
                .on_hover_text_at_pointer("The maximum number of times an item can be repeated before it is penalized.");

            self.penalty_max_count = NonZeroU8::new(max_count.clamp(1, 255)).unwrap();

            inner
        }).inner;

        // Ngram min size
        resp |= ui
            .horizontal(|ui| {
                // NonZeroU8 does not implement emath::Numeric, so we have to
                // convert and then back again.
                let mut min_size = self.ngram_min_size.get();
                let inner = ui.label("Ngram Min Size")
                    | ui.add(
                        egui::DragValue::new(&mut min_size)
                            .clamp_range(1..=NGram::CAPACITY as u8),
                    )
                    .on_hover_text_at_pointer(
                        "The minimum size of the n-gram to penalize (in tokens).",
                    );

                self.ngram_min_size =
                    NonZeroU8::new(min_size.clamp(1, NGram::CAPACITY as u8))
                        .unwrap();

                inner
            })
            .inner;

        // Ngram max size
        resp |= ui
            .horizontal(|ui| {
                let mut max_size = self.ngram_max_size.get();
                let inner = ui.label("Ngram Max Size")
                    | ui.add(
                        egui::DragValue::new(&mut max_size)
                            .clamp_range(1..=NGram::CAPACITY as u8),
                    )
                    .on_hover_text_at_pointer(
                        "The maximum size of the n-gram to penalize (in tokens).",
                    );

                self.ngram_max_size =
                    NonZeroU8::new(max_size.clamp(1, NGram::CAPACITY as u8))
                        .unwrap();

                inner
            })
            .inner;

        // Penalty repeat
        resp |= ui
            .horizontal(|ui| {
                ui.label("Repeat penalty")
                    | ui.add(egui::Slider::new(&mut self.penalty_repeat, 1.0..=2.0))
                        .on_hover_text_at_pointer(
                            "The penalty for repeating an n-gram. A reasonable value is 1.15. The value should be slightly less than other software packages because we're penalizing ngrams, not just tokens, and the penalties stack.",
                        )
            })
            .inner;

        // Penalty freq
        resp |= ui
            .horizontal(|ui| {
                ui.label("Frequency penalty")
                    | ui.add(egui::Slider::new(&mut self.penalty_freq, 0.0..=1.0))
                        .on_hover_text_at_pointer(
                            "The penalty for the frequency of the n-grams. This subtracts the frequency of the n-gram from the logit of the penalized token in the n-gram. The value should be slightly less than other software packages because we're penalizing ngrams, not just tokens, and the penalties stack.",
                        )
            })
            .inner;

        // Penalty present
        resp |= ui
            .horizontal(|ui| {
                ui.label("Presence penalty")
                    | ui.add(egui::Slider::new(&mut self.penalty_present, 0.0..=1.0))
                        .on_hover_text_at_pointer(
                            "The penalty for the presence of the n-grams. This subtracts the presence of the n-gram from the logit of the penalized token in the n-gram. The value should be slightly less than other software packages because we're penalizing ngrams, not just tokens, and the penalties stack.",
                        )
            })
            .inner;

        // Surgical mode
        resp |= ui.checkbox(&mut self.surgical, "Surgical")
            .on_hover_text_at_pointer(
                "When enabled, only penalize the token that would complete a repeated n-gram pattern. \
                 For example, if \"The New\" was just generated and \"The New York\" has appeared before, \
                 penalize \"York\" specifically. When disabled, all previously-seen tokens are penalized.",
            );

        resp
    }

    /// Draw [`egui::Ui`] for [`RepetitionOptions`].
    #[cfg(feature = "egui")]
    pub fn draw(&mut self, ui: &mut egui::Ui) -> egui::Response {
        let resp = egui::CollapsingHeader::new("Repetition Options")
            .show(ui, |ui| self.draw_inner(ui));

        let header_response = resp.header_response.on_hover_text_at_pointer(
            "Options for penalizing repetition in the generated text.",
        );
        resp.body_response.unwrap_or(header_response)
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

static_assertions::assert_impl_all!(RepetitionError: Send, Sync);

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
/// Two-phase approach: first, count trailing n-grams to update the frequency
/// map. Then, iterate over all tracked n-grams and penalize the last
/// non-ignored token in each one. This ensures all previously-seen tokens get
/// penalized, not just the most recently generated one.
///
/// Originally inspired by `llama.cpp`'s repetition penalties, extended with
/// n-gram support. Rewritten by Claude (Anthropic) to fix a design issue where
/// penalties were only applied to the trailing token.
pub fn apply_sample_repetition_ngram(
    candidates: Candidates,
    tokens: &[llama_token],
    opts: &mut RepetitionOptions,
    freq_map: &mut NGramStats,
    model: &crate::Model,
) -> Result<Candidates, RepetitionError> {
    let k = candidates.len();
    let n_vocab = k.get();
    let mut candidates = candidates.sort(Sorted::ById { k });

    let RepetitionOptions {
        ignored_stopwords,
        ignored,
        ngram_max_size,
        ngram_min_size,
        penalty_freq,
        penalty_max_count,
        penalty_present,
        penalty_repeat,
        surgical,
    } = opts;

    // Add ignored stopwords to the ignored ngrams if they are not already
    // there.
    while let Some(stopwords) = ignored_stopwords.pop() {
        let newly_ignored = stopwords.into_tokens(model);
        ignored.sort();
        for token in newly_ignored {
            let ngram = NGram::from(token);
            if !ngram_is_ignored(ngram, ignored) {
                ignored.push(ngram);
            }
        }
    }

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

    // Phase 1: Count trailing n-grams to update the frequency map with newly
    // generated tokens.
    for slice in (ngram_min_size..=ngram_max_size).filter_map(|n| {
        tokens
            .len()
            .checked_sub(n)
            .and_then(|start| tokens.get(start..))
    }) {
        let ngram = NGram::try_from(slice).unwrap();
        if ngram_is_ignored(ngram, &ignored) {
            continue;
        }
        freq_map.add(ngram, &candidates);
    }

    if *surgical {
        // Phase 2 (surgical): For each n-gram in the frequency map with len > 1,
        // check if the current trailing tokens match the n-gram's prefix. If so,
        // penalize the LAST token in the n-gram — the one that would complete
        // the repeated pattern. For unigrams, penalize the token directly (same
        // as broad mode).
        //
        // Example: freq_map contains [The, New, York] with count=2.
        // Trailing tokens are [..., The, New]. The prefix [The, New] matches,
        // so we penalize "York" to prevent the trigram from completing.
        for (ngram, data) in freq_map.iter() {
            if ngram_is_ignored(*ngram, &ignored) {
                continue;
            }

            let slice = ngram.as_slice();

            if slice.len() == 1 {
                // Unigram: penalize directly if seen enough times.
                let token = slice[0] as usize;
                if token >= n_vocab {
                    continue;
                }
                if ngram_is_ignored(slice[0].into(), &ignored) {
                    continue;
                }
                let candidate = &mut candidates.data[token];
                if data.count() > penalty_max_count {
                    if candidate.logit <= 0.0 {
                        candidate.logit *= *penalty_repeat;
                    } else {
                        candidate.logit /= *penalty_repeat;
                    }
                }
                candidate.logit -=
                    (data.count() as f32) * *penalty_freq + *penalty_present;
            } else {
                // Multi-token n-gram: check if trailing tokens match the prefix.
                let prefix = &slice[..slice.len() - 1];
                let completion_token = *slice.last().unwrap() as usize;
                if completion_token >= n_vocab {
                    continue;
                }
                if ngram_is_ignored((*slice.last().unwrap()).into(), &ignored) {
                    continue;
                }

                // Check if the trailing tokens end with this prefix.
                let matches = tokens.len() >= prefix.len()
                    && tokens[tokens.len() - prefix.len()..] == *prefix;

                if matches && data.count() > penalty_max_count {
                    let candidate = &mut candidates.data[completion_token];
                    let scaled_penalty =
                        penalty_repeat.powf(ngram.len() as f32);
                    if candidate.logit <= 0.0 {
                        candidate.logit *= scaled_penalty;
                    } else {
                        candidate.logit /= scaled_penalty;
                    }
                    candidate.logit -= (data.count() as f32) * *penalty_freq
                        + *penalty_present;
                }
            }
        }
    } else {
        // Phase 2 (broad): Penalize ALL tracked n-grams. For each n-gram,
        // penalize the last non-ignored token. This ensures all previously-seen
        // tokens get their logits reduced.
        for (ngram, data) in freq_map.iter() {
            if ngram_is_ignored(*ngram, &ignored) {
                continue;
            }

            // Find the last non-ignored token in the n-gram to penalize.
            let mut penalized_token = None;
            for &token in ngram.as_slice().iter().rev() {
                if ngram_is_ignored(token.into(), &ignored) {
                    continue;
                }
                penalized_token = Some(token);
                break;
            }
            let penalized_token = match penalized_token {
                Some(token) if (token as usize) < n_vocab => token as usize,
                _ => continue,
            };

            let candidate = &mut candidates.data[penalized_token];

            // Multiplicative penalty: scales with n-gram size. Only fires when
            // count exceeds max_count.
            if data.count() > penalty_max_count {
                let scaled_penalty = penalty_repeat.powf(ngram.len() as f32);
                if candidate.logit <= 0.0 {
                    candidate.logit *= scaled_penalty;
                } else {
                    candidate.logit /= scaled_penalty;
                }
            }

            // Additive penalties: frequency scales with count, presence is
            // binary.
            candidate.logit -=
                (data.count() as f32) * *penalty_freq + *penalty_present;
        }
    }

    // We have modified logit values.
    candidates.softmax_applied_to = None;

    Ok(candidates)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Candidates;
    use llama_cpp_sys_3::llama_token_data;
    use std::path::PathBuf;

    /// Helper: create candidates sorted by id with given logits.
    fn make_candidates(logits: &[f32]) -> Candidates {
        let data: Vec<llama_token_data> = logits
            .iter()
            .enumerate()
            .map(|(i, &logit)| llama_token_data {
                id: i as i32,
                logit,
                p: 0.0,
            })
            .collect();
        Candidates::from_vec(data)
    }

    fn load_model() -> crate::Model {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("models/model.gguf");
        crate::Model::from_file(path, None).unwrap()
    }

    /// Simulate multiple generation steps, applying the penalty at each step.
    /// Returns (logits_after, freq_map) so callers can inspect both.
    fn run_penalty_steps(
        logits: &[f32],
        token_history: &[llama_token],
        opts: &mut RepetitionOptions,
        steps: usize,
        model: &crate::Model,
    ) -> (Vec<f32>, NGramStats) {
        let mut freq_map = NGramStats::new();
        let mut result_logits = logits.to_vec();

        for _ in 0..steps {
            let candidates = make_candidates(&result_logits);
            let result = apply_sample_repetition_ngram(
                candidates,
                token_history,
                opts,
                &mut freq_map,
                model,
            )
            .unwrap();
            result_logits = result.iter().map(|c| c.logit).collect();
        }

        (result_logits, freq_map)
    }

    #[test]
    #[ignore = "requires model"]
    fn test_repetition_penalty_basic() {
        let model = load_model();
        // 10-token vocab, token 5 has highest logit
        let logits: Vec<f32> =
            (0..10).map(|i| if i == 5 { 2.0 } else { -1.0 }).collect();
        let tokens = vec![5_i32];

        // Use presence penalty so the effect is visible on first repeat.
        let mut opts = RepetitionOptions {
            penalty_repeat: 1.15,
            penalty_freq: 0.0,
            penalty_present: 0.5,
            ..RepetitionOptions::default()
        };

        let (after, _) =
            run_penalty_steps(&logits, &tokens, &mut opts, 1, &model);

        println!("=== Basic unigram penalty (presence=0.5) ===");
        for i in 0..10 {
            if (logits[i] - after[i]).abs() > 1e-6 {
                println!(
                    "  token {}: {:.4} -> {:.4} (delta {:.4})",
                    i,
                    logits[i],
                    after[i],
                    after[i] - logits[i]
                );
            }
        }

        // Token 5 should have been penalized
        assert!(
            after[5] < logits[5],
            "token 5 should be penalized: before={}, after={}",
            logits[5],
            after[5]
        );
        // Other tokens should be unchanged
        for i in 0..10 {
            if i != 5 {
                assert_eq!(
                    logits[i], after[i],
                    "token {} should be unchanged",
                    i
                );
            }
        }
    }

    #[test]
    #[ignore = "requires model"]
    fn test_repetition_penalty_frequency_vs_presence() {
        let model = load_model();
        let logits: Vec<f32> =
            (0..10).map(|i| if i == 3 { 2.0 } else { -1.0 }).collect();
        // Token 3 is the last token (unigram match)
        let tokens = vec![3_i32];

        // Frequency only — after 3 steps, count=3, so freq subtracts 3*0.1=0.3
        let mut freq_opts = RepetitionOptions {
            penalty_repeat: 1.0,
            penalty_freq: 0.1,
            penalty_present: 0.0,
            ..RepetitionOptions::default()
        };
        let (after_freq, _) =
            run_penalty_steps(&logits, &tokens, &mut freq_opts, 3, &model);

        // Presence only — after 3 steps, each subtracts 0.5, total 1.5
        let mut pres_opts = RepetitionOptions {
            penalty_repeat: 1.0,
            penalty_freq: 0.0,
            penalty_present: 0.5,
            ..RepetitionOptions::default()
        };
        let (after_pres, _) =
            run_penalty_steps(&logits, &tokens, &mut pres_opts, 3, &model);

        println!("=== Frequency vs Presence after 3 steps ===");
        println!(
            "  freq (0.1):     {:.4} -> {:.4} (delta {:.4})",
            logits[3],
            after_freq[3],
            after_freq[3] - logits[3]
        );
        println!(
            "  presence (0.5): {:.4} -> {:.4} (delta {:.4})",
            logits[3],
            after_pres[3],
            after_pres[3] - logits[3]
        );

        // Both should reduce the logit
        assert!(after_freq[3] < logits[3], "freq should reduce logit");
        assert!(after_pres[3] < logits[3], "presence should reduce logit");

        // Frequency accumulates: step 1 subtracts 1*0.1, step 2 starts from
        // the already-reduced logit and subtracts 2*0.1, etc. (count grows)
        // Presence is constant per step: always subtracts 0.5.
        let freq_delta = (logits[3] - after_freq[3]).abs();
        let pres_delta = (logits[3] - after_pres[3]).abs();
        println!(
            "  freq total delta: {:.4}, pres total delta: {:.4}",
            freq_delta, pres_delta
        );
        // Presence: 3 steps * 0.5 = 1.5
        assert_approx_eq!(pres_delta, 1.5, 0.01);
    }

    #[test]
    #[ignore = "requires model"]
    fn test_repetition_penalty_multiplicative() {
        let model = load_model();
        // The penalty applies to the LAST non-ignored token in each trailing
        // n-gram. So we test positive and negative logits by putting each as
        // the last token in separate runs.

        let mut opts = RepetitionOptions {
            penalty_repeat: 1.5,
            penalty_freq: 0.0,
            penalty_present: 0.0,
            penalty_max_count: NonZeroU8::new(1).unwrap(),
            ngram_min_size: NonZeroU8::new(1).unwrap(),
            ngram_max_size: NonZeroU8::new(1).unwrap(),
            ..RepetitionOptions::default()
        };

        // Positive logit: token 3 (logit=2.0) is the last token
        let logits_pos: Vec<f32> =
            (0..10).map(|i| if i == 3 { 2.0 } else { 0.1 }).collect();
        let (after_pos_1, _) =
            run_penalty_steps(&logits_pos, &[3], &mut opts, 1, &model);
        let (after_pos_2, _) =
            run_penalty_steps(&logits_pos, &[3], &mut opts, 2, &model);

        // Negative logit: token 7 (logit=-2.0) is the last token
        let logits_neg: Vec<f32> =
            (0..10).map(|i| if i == 7 { -2.0 } else { 0.1 }).collect();
        let (after_neg_1, _) =
            run_penalty_steps(&logits_neg, &[7], &mut opts, 1, &model);
        let (after_neg_2, _) =
            run_penalty_steps(&logits_neg, &[7], &mut opts, 2, &model);

        println!("=== Multiplicative penalty (unigrams, penalty=1.5) ===");
        println!("Positive logit (token 3, logit=2.0):");
        println!(
            "  step 1 (count=1): {:.4} -> {:.4}",
            logits_pos[3], after_pos_1[3]
        );
        println!(
            "  step 2 (count=2): {:.4} -> {:.4} (expect {:.4})",
            logits_pos[3],
            after_pos_2[3],
            2.0 / 1.5
        );
        println!("Negative logit (token 7, logit=-2.0):");
        println!(
            "  step 1 (count=1): {:.4} -> {:.4}",
            logits_neg[7], after_neg_1[7]
        );
        println!(
            "  step 2 (count=2): {:.4} -> {:.4} (expect {:.4})",
            logits_neg[7],
            after_neg_2[7],
            -2.0 * 1.5
        );

        // Step 1: count=1, not > max_count(1), no penalty
        assert_eq!(logits_pos[3], after_pos_1[3], "step 1 pos: no penalty");
        assert_eq!(logits_neg[7], after_neg_1[7], "step 1 neg: no penalty");

        // Step 2: count=2 > 1, penalty fires
        // Positive logit divided by penalty
        assert!(
            after_pos_2[3] < logits_pos[3],
            "positive logit should decrease"
        );
        // Negative logit multiplied by penalty (more negative)
        assert!(
            after_neg_2[7] < logits_neg[7],
            "negative logit should become more negative"
        );
    }

    #[test]
    #[ignore = "requires model"]
    fn test_repetition_penalty_escalation() {
        let model = load_model();
        // Show how penalty_repeat squaring escalates across ngram sizes.
        let logits: Vec<f32> = (0..10).map(|_| 1.0).collect();
        let tokens = vec![1, 2, 3, 4];

        let mut opts = RepetitionOptions {
            penalty_repeat: 1.15,
            penalty_freq: 0.0,
            penalty_present: 0.0,
            penalty_max_count: NonZeroU8::new(1).unwrap(),
            ngram_min_size: NonZeroU8::new(1).unwrap(),
            ngram_max_size: NonZeroU8::new(4).unwrap(),
            ..RepetitionOptions::default()
        };

        // 2 steps: first counts, second penalizes
        let (after, freq_map) =
            run_penalty_steps(&logits, &tokens, &mut opts, 2, &model);

        println!("=== Escalation across ngram sizes (penalty=1.15) ===");
        println!("Token logits after 2 steps:");
        for i in 0..10 {
            if (logits[i] - after[i]).abs() > 1e-6 {
                println!(
                    "  token {}: {:.4} -> {:.6} (ratio {:.4}x)",
                    i,
                    logits[i],
                    after[i],
                    logits[i] / after[i]
                );
            }
        }

        println!("\nPenalty_repeat value at each ngram size:");
        let mut p = 1.15_f32;
        for size in 1..=4 {
            println!("  size {}: penalty_repeat = {:.6}", size, p);
            p *= p;
        }

        println!("\nNGram frequency map:");
        for (ngram, data) in freq_map.iter() {
            println!(
                "  {:?}: count={}, cum_prob={:.4}",
                ngram.as_slice(),
                data.count(),
                data.cum_prob()
            );
        }
    }
}
