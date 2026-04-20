//! Repetition-penalty sampling helpers.
//!
//! Extracted from `sample.rs` in the JSON-grammar refactor; semantics
//! unchanged.

use crate::{
    candidates::Sorted,
    data::{ignore_category::IgnoreCategory, StopWords},
    ngram::{NGram, NGramStats},
    Candidates,
};

use llama_cpp_sys_3::llama_token;

use std::num::NonZeroU8;

#[cfg(feature = "egui")]
use super::DELETE_ICON;

#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
/// Options for `apply_sample_repetition_penalties`.
#[derive(Clone, Debug, PartialEq)]
pub struct RepetitionOptions {
    /// Sets of tokens to ignore, by language. These are never penalized.
    #[cfg_attr(feature = "serde", serde(default, alias = "ignored_stopwords"))]
    pub(crate) ignored_categories: Vec<IgnoreCategory>,
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
    /// Surgical mode. For each repeating n-gram, penalize the earliest token
    /// that would extend the match in the trailing history. If no prefix has
    /// been re-emitted yet, this is the first token of the n-gram — blocking
    /// the phrase at its entry point rather than only at its completion. If
    /// that target token is in the ignored set, the n-gram is skipped
    /// entirely (we do not fall through to the next token, which would
    /// produce odd partial completions).
    ///
    /// Example: with `[The, New, York]` as a known repeating trigram, this
    /// penalizes "The" before the phrase starts, "New" if "The" was just
    /// emitted, or "York" if "The New" was just emitted. Lowercase stopwords
    /// like "the" are skipped via the ignored set; uppercase "The" is not,
    /// so proper-noun repetition is blocked at the first token.
    #[cfg_attr(feature = "serde", serde(default))]
    pub(crate) surgical: bool,
}

impl Default for RepetitionOptions {
    fn default() -> Self {
        Self {
            ignored_categories: vec![],
            ignored: vec![],
            penalty_max_count: NonZeroU8::new(1).unwrap(),
            ngram_min_size: NonZeroU8::new(1).unwrap(),
            ngram_max_size: NonZeroU8::new(4).unwrap(),
            penalty_repeat: 1.06,
            penalty_freq: 0.1,
            penalty_present: 0.1,
            surgical: false,
        }
    }
}

impl RepetitionOptions {
    /// [`IgnoreCategory`]s of tokens. These are never penalized.
    pub fn ignored_categories(&self) -> &[IgnoreCategory] {
        &self.ignored_categories
    }

    /// Use [`RepetitionOptions::ignored_categories`] instead
    #[allow(deprecated)]
    #[deprecated]
    pub fn ignored_stopwords(&self) -> &[StopWords] {
        self.ignored_categories()
    }

    /// Set [`IgnoreCategory`]s to ignore. These are never penalized.
    pub fn set_ignored_categories(
        mut self,
        ignored_categories: Vec<IgnoreCategory>,
    ) -> Self {
        self.ignored_categories = ignored_categories;
        self
    }

    /// Use [`RepetitionOptions::set_ignored_categories`] instead
    #[allow(deprecated)]
    #[deprecated]
    pub fn set_ignored_stopwords(
        self,
        ignored_stopwords: Vec<StopWords>,
    ) -> Self {
        self.set_ignored_categories(ignored_stopwords)
    }

    /// Extend the list of [`IgnoreCategory`]
    pub fn extend_ignored_categories<It>(&mut self, ignored_categories: It)
    where
        It: IntoIterator<Item = IgnoreCategory>,
    {
        self.ignored_categories.extend(ignored_categories);
    }

    /// Use [`RepetitionOptions::extend_ignored_categories`] instead
    #[allow(deprecated)]
    #[deprecated]
    pub fn extend_stopwords<It>(&mut self, stopwords: It)
    where
        It: IntoIterator<Item = StopWords>,
    {
        self.ignored_categories.extend(stopwords);
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

    /// Ignore [`IgnoreCategory`]s. These are commonly used tokens that are
    /// often ignored in text generation.
    pub fn ignore_categories(
        mut self,
        ignore_categories: IgnoreCategory,
        model: &crate::Model,
    ) -> Self {
        self.extend_ignored(ignore_categories.into_tokens(model));
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
        const IGNORE_CATEGORY_HELP: &str = "Common tokens that are often ignored in NLP tasks. These options allow you to ignore sets of tokens for the purpose of penalizing repetition. The idea is to allow a higher repetition penalty without penalizing common words. This is experimental.";

        // IgnoreCategories
        if !self.ignored_categories.is_empty() {
            ui.label("Token categories ignored for:")
                .on_hover_text_at_pointer(IGNORE_CATEGORY_HELP);
            let mut to_remove = None;
            for (i, language) in self.ignored_categories.iter().enumerate() {
                ui.horizontal(|ui| {
                    if ui
                        .add(egui::Button::image(DELETE_ICON))
                        .on_hover_text_at_pointer("Remove this language from the list of ignored token sets.")
                        .clicked() {
                        to_remove = Some(i);
                    };
                    ui.label(language.as_str())
                });
            }
            if let Some(i) = to_remove {
                self.ignored_categories.remove(i);
            }
        }
        if self.ignored_categories.len() < IgnoreCategory::ALL.len() {
            egui::ComboBox::from_label("to ignore token categories for")
                .selected_text("Select a language...")
                .show_ui(ui, |ui| {
                    for language in IgnoreCategory::ALL {
                        if !self.ignored_categories.contains(&language) {
                            if ui
                                .selectable_label(false, language.as_str())
                                .clicked()
                            {
                                self.ignored_categories.push(language);
                                self.ignored_categories.sort();
                            }
                        }
                    }
                })
                .response
                .on_hover_text_at_pointer(IGNORE_CATEGORY_HELP);
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

/// Pick the token to penalize for surgical mode.
///
/// Finds the longest `k` in `[0, ngram.len() - 1]` such that `tokens` ends
/// with `ngram[..k]` — i.e. how much of the n-gram has already been re-emitted.
/// Returns `ngram[k]`: the next token that would extend the match (or the
/// first token, if no prefix has been emitted yet).
///
/// Returns `None` when `ngram[k]` is in `ignored`. We deliberately do not
/// advance to `ngram[k+1]` in that case — penalizing a later token produces
/// odd partial completions, e.g. ignoring the lowercase "the" stopword in
/// `[the, cat, sat]` and penalizing "cat" allows "the <other> sat".
fn surgical_target(
    ngram: &NGram,
    tokens: &[llama_token],
    ignored: &[NGram],
) -> Option<llama_token> {
    let slice = ngram.as_slice();
    let max_k = slice.len().saturating_sub(1);
    let mut best_k = 0;
    for k in (1..=max_k).rev() {
        if tokens.len() >= k && tokens[tokens.len() - k..] == slice[..k] {
            best_k = k;
            break;
        }
    }
    let target = slice[best_k];
    if ngram_is_ignored(target.into(), ignored) {
        None
    } else {
        Some(target)
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
        ignored_categories,
        ignored,
        ngram_max_size,
        ngram_min_size,
        penalty_freq,
        penalty_max_count,
        penalty_present,
        penalty_repeat,
        surgical,
    } = opts;

    // Add ignored categories of tokens to the ignored ngrams if they are not
    // already there.
    while let Some(cats) = ignored_categories.pop() {
        let newly_ignored = cats.into_tokens(model);
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
        // Phase 2 (surgical): For each repeating n-gram, penalize the earliest
        // token that would extend the match — slice[k], where k is the longest
        // prefix already re-emitted in the trailing history. With k=0 (no
        // prefix re-emitted) we penalize the first token to prevent the phrase
        // from starting; with k>0 we penalize the next continuation.
        //
        // Example: freq_map contains [The, New, York] with count=2.
        //   tokens end in []           -> k=0, penalize "The"  (prevent start)
        //   tokens end in [The]        -> k=1, penalize "New"  (prevent continuation)
        //   tokens end in [The, New]   -> k=2, penalize "York" (prevent completion)
        //
        // TokenCategory are case-sensitive via tokenization: lowercase "the" is
        // in the ignored set, uppercase "The" is not — so proper-noun phrases
        // get blocked at their entry point.
        for (ngram, data) in freq_map.iter() {
            if ngram_is_ignored(*ngram, &ignored) {
                continue;
            }
            if data.count() <= penalty_max_count {
                continue;
            }
            let target = match surgical_target(ngram, tokens, &ignored) {
                Some(t) => t as usize,
                None => continue,
            };
            if target >= n_vocab {
                continue;
            }

            let candidate = &mut candidates.data[target];
            let scaled_penalty = penalty_repeat.powf(ngram.len() as f32);
            if candidate.logit <= 0.0 {
                candidate.logit *= scaled_penalty;
            } else {
                candidate.logit /= scaled_penalty;
            }
            candidate.logit -=
                (data.count() as f32) * *penalty_freq + *penalty_present;
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

    // --- surgical_target tests (no model required) -----------------------

    fn ngram(tokens: &[llama_token]) -> NGram {
        NGram::try_from(tokens).unwrap()
    }

    /// k=0: no prefix re-emitted yet — penalize the first token of the
    /// n-gram. This blocks proper-noun repetitions like "The New York Times"
    /// at entry, before "The" is even selected.
    #[test]
    fn surgical_target_k0_penalizes_first_token() {
        let ng = ngram(&[10, 20, 30]);
        let tokens = &[99, 88, 77];
        assert_eq!(surgical_target(&ng, tokens, &[]), Some(10));
    }

    /// k=1: first token re-emitted — penalize the next continuation to
    /// prevent the phrase from growing past its first token.
    #[test]
    fn surgical_target_k1_penalizes_second_token() {
        let ng = ngram(&[10, 20, 30]);
        let tokens = &[99, 10];
        assert_eq!(surgical_target(&ng, tokens, &[]), Some(20));
    }

    /// k=n-1: full prefix re-emitted — penalize the final completion token.
    #[test]
    fn surgical_target_k_full_penalizes_completion() {
        let ng = ngram(&[10, 20, 30]);
        let tokens = &[10, 20];
        assert_eq!(surgical_target(&ng, tokens, &[]), Some(30));
    }

    /// Longest prefix match wins when the history happens to match multiple
    /// prefix lengths (here the history ends in [10, 10], both k=1 and k=2
    /// for an n-gram of [10, 10, 30]).
    #[test]
    fn surgical_target_picks_longest_prefix_match() {
        let ng = ngram(&[10, 10, 30]);
        let tokens = &[10, 10];
        assert_eq!(surgical_target(&ng, tokens, &[]), Some(30));
    }

    /// If the target token is in the ignored set, return None — do NOT
    /// fall through to slice[k+1]. That fall-through was rejected because
    /// it produces odd partial completions (e.g. "The New Potato" when the
    /// lowercase "the" stopword is skipped).
    #[test]
    fn surgical_target_skips_entirely_when_target_ignored() {
        let ng = ngram(&[10, 20, 30]);
        let ignored = vec![NGram::from(10_i32)];
        let tokens: &[llama_token] = &[];
        assert_eq!(surgical_target(&ng, tokens, &ignored), None);
    }

    /// Ignored check applies at whichever k is selected, not only k=0.
    #[test]
    fn surgical_target_ignored_continuation_skips() {
        let ng = ngram(&[10, 20, 30]);
        let ignored = vec![NGram::from(20_i32)];
        let tokens = &[10];
        assert_eq!(surgical_target(&ng, tokens, &ignored), None);
    }

    /// Unigram: always k=0, target is the only token.
    #[test]
    fn surgical_target_unigram() {
        let ng = ngram(&[42]);
        let tokens = &[99];
        assert_eq!(surgical_target(&ng, tokens, &[]), Some(42));
        let ignored = vec![NGram::from(42_i32)];
        assert_eq!(surgical_target(&ng, tokens, &ignored), None);
    }

    /// History shorter than the prefix shouldn't panic or over-match.
    #[test]
    fn surgical_target_short_history() {
        let ng = ngram(&[10, 20, 30]);
        // Empty history: only k=0 is possible.
        assert_eq!(surgical_target(&ng, &[], &[]), Some(10));
        // History of length 1 that doesn't match prefix[0]: k=0.
        assert_eq!(surgical_target(&ng, &[99], &[]), Some(10));
        // History of length 1 matching prefix[0]: k=1.
        assert_eq!(surgical_target(&ng, &[10], &[]), Some(20));
    }

    /// A prefix match that doesn't reach the end of history: not a match.
    /// (We only look at the trailing suffix, not arbitrary substrings.)
    #[test]
    fn surgical_target_only_trailing_suffix_counts() {
        let ng = ngram(&[10, 20, 30]);
        // [10, 20] appears earlier, but history ends in [99]: k=0.
        let tokens = &[10, 20, 99];
        assert_eq!(surgical_target(&ng, tokens, &[]), Some(10));
    }
}
