//! Repetition-penalty sampling helpers.
//!
//! Extracted from `sample.rs` in the JSON-grammar refactor; semantics
//! unchanged.

#[allow(deprecated)]
use crate::{
    candidates::Sorted,
    data::{ignore_category::IgnoreCategory, StopWords},
    ngram::{NGram, NGramStats},
    Candidates, Token,
};

use std::{
    collections::BTreeSet,
    num::{NonZeroU32, NonZeroU8},
};

#[cfg(feature = "egui")]
use super::DELETE_ICON;

/// Options for [`apply_sample_repetition_ngram`].
///
/// # Why this config is the odd one out
///
/// Unlike every other options struct in the crate ([`SampleOptions`],
/// [`PredictOptions`], `RenderOptions`, `ToolChoiceOptions`,
/// `OutputConfigOptions` — all of which expose public fields), the fields here
/// are `pub(crate)` and reached through accessors. This is **deliberate**: it
/// is the crate's only config with a cross-field invariant —
/// [`ngram_min_size`](Self::ngram_min_size) `<=`
/// [`ngram_max_size`](Self::ngram_max_size). A violated pair makes the
/// generation-time `(min..=max)` range empty, silently disabling the penalty.
///
/// Do **not** "fix the inconsistency" by making the fields `pub`. That would let
/// an external caller construct an inverted pair, reopening the hole that the
/// three guards here close together:
/// - **Setters clamp** with "just-set wins" — see
///   [`set_ngram_min_size`](Self::set_ngram_min_size).
/// - **`Deserialize` rejects** an inverted pair with
///   [`RepetitionError::InvalidNgramRange`] (via `#[serde(try_from = ...)]` —
///   keep it; don't revert to a plain derive). Strict rather than lenient
///   because a hand-edited sidecar is read every run and never rewritten, so a
///   silent clamp would hide the breakage forever.
/// - **The apply site `debug_assert!`s** the invariant, surfacing any
///   construction path that bypassed validation instead of masking it.
///
/// This struct is the template for the next config that earns a cross-field
/// invariant before 1.0.
///
/// [`SampleOptions`]: crate::SampleOptions
/// [`PredictOptions`]: crate::PredictOptions
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "serde", serde(try_from = "RepetitionOptionsShadow"))]
#[derive(Clone, Debug, PartialEq)]
pub struct RepetitionOptions {
    /// Sets of tokens to ignore, by language. These are never penalized.
    pub(crate) ignored_categories: BTreeSet<IgnoreCategory>,
    /// [`NGram`]s to ignore. These are never penalized.
    pub(crate) ignored: BTreeSet<NGram>,
    /// Sliding-window size. Only n-gram occurrences within the last
    /// `window_size` generation steps contribute to the penalty. Older
    /// occurrences are evicted by [`NGramStats::evict_outside_window`].
    /// This — together with [`Self::decay`] — bounds the additive penalty
    /// term so long generations don't have their natural logit gradient
    /// dominated by `count * penalty_freq`.
    pub(crate) window_size: NonZeroU32,
    /// Per-step decay applied to in-window occurrences when computing the
    /// effective count for the penalty. The effective count contributed by one
    /// occurrence at age `a` is `decay^a`; the sum over all in-window
    /// occurrences is bounded above by `1 / (1 - decay)` for sustained
    /// repetition. Use `1.0` to disable decay (all in-window occurrences count
    /// fully); recommended range `0.95..=0.99`.
    pub(crate) decay: f32,
    /// The maximum number of times an item can be repeated before it is
    /// penalized. This can be used to allow some n-grams to be repeated in
    /// situations where repetition is desirable for the purpose, like song
    /// lyrics.
    pub(crate) penalty_max_count: NonZeroU8,
    /// NGram minimum size. Max size is [`NGram::CAPACITY`] by default.
    pub(crate) ngram_min_size: NonZeroU8,
    /// NGram maximum size, capped at [`NGram::CAPACITY`].
    pub(crate) ngram_max_size: NonZeroU8,
    /// Repetition penalty. A reasonable value is 1.05-1.10. Keep in mind
    /// overlapping n-gram penalties stack so this is stronger than with most
    /// inference packages. Raise this to the power of the overlap to get a
    /// comparable number.
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
    /// the phrase at its entry point rather than the entire completion.
    ///
    /// Example: ["go", "fuck", "yourself"] would only penalize "go", ensuring
    /// another insult entirely such as ["eat", "shit", "buddy"]. Note that
    /// penalties *do* overlap so if bigrams *and* trigrams are penalized,
    /// "fuck" will also be penalized, and "go" *doubly* so, but not "yourself".
    pub(crate) surgical: bool,
}

/// Deserialize-only mirror of [`RepetitionOptions`]. The real struct routes its
/// `Deserialize` through this via `#[serde(try_from = ...)]` so the
/// `ngram_min_size <= ngram_max_size` invariant is validated on the way in
/// (see [`TryFrom`] impl below). It carries every field-level serde attribute
/// (`default`, `alias`); the real struct deliberately has none so the two can't
/// drift in opposite directions. Never serialized — only ever constructed by
/// serde during deserialization.
#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct RepetitionOptionsShadow {
    #[serde(default, alias = "ignored_stopwords")]
    ignored_categories: BTreeSet<IgnoreCategory>,
    #[serde(default)]
    ignored: BTreeSet<NGram>,
    #[serde(default = "default_window_size")]
    window_size: NonZeroU32,
    #[serde(default = "default_decay")]
    decay: f32,
    penalty_max_count: NonZeroU8,
    ngram_min_size: NonZeroU8,
    ngram_max_size: NonZeroU8,
    penalty_repeat: f32,
    penalty_freq: f32,
    penalty_present: f32,
    #[serde(default)]
    surgical: bool,
}

#[cfg(feature = "serde")]
impl TryFrom<RepetitionOptionsShadow> for RepetitionOptions {
    type Error = RepetitionError;

    fn try_from(s: RepetitionOptionsShadow) -> Result<Self, Self::Error> {
        if !Self::ngram_range_ordered(s.ngram_min_size, s.ngram_max_size) {
            // Strict (not lenient-clamp): a hand-edited sidecar is read every
            // run and never rewritten, so a silent fix would hide the breakage
            // forever. serde maps this `Display` error through
            // `de::Error::custom`. See the struct-level docs.
            return Err(RepetitionError::InvalidNgramRange {
                min: s.ngram_min_size,
                max: s.ngram_max_size,
            });
        }
        Ok(Self {
            ignored_categories: s.ignored_categories,
            ignored: s.ignored,
            window_size: s.window_size,
            decay: s.decay,
            penalty_max_count: s.penalty_max_count,
            ngram_min_size: s.ngram_min_size,
            ngram_max_size: s.ngram_max_size,
            penalty_repeat: s.penalty_repeat,
            penalty_freq: s.penalty_freq,
            penalty_present: s.penalty_present,
            surgical: s.surgical,
        })
    }
}

/// Default window size — long enough to catch genuine paragraph-scale
/// repetition, short enough that the bounded additive contribution
/// (`window_size * penalty_freq` worst-case before decay) stays well
/// inside the model's natural logit gradient.
fn default_window_size() -> NonZeroU32 {
    NonZeroU32::new(256).unwrap()
}

/// Default per-step decay. Caps sustained-repetition effective count at
/// `1 / (1 - 0.95) = 20` regardless of how long generation runs.
fn default_decay() -> f32 {
    0.95
}

impl Default for RepetitionOptions {
    fn default() -> Self {
        // Defaults are tuned for Qwen3.5 a17b and a3b
        Self {
            // English stopwords + JSON syntax tokens are common-by- design;
            // penalising them just biases the model away from natural prose /
            // structured output for no anti-loop benefit. Punctuation is also
            // default-on for the same reason — prose `. , ; : ! ?` have no
            // lexical variety, so accumulating penalty on `.` biases toward
            // run-ons. Users can override by calling
            // `set_ignored_categories(vec![])`.
            ignored_categories: BTreeSet::from([
                IgnoreCategory::English,
                IgnoreCategory::Json,
                IgnoreCategory::Punctuation,
            ]),
            ignored: BTreeSet::new(),
            window_size: default_window_size(),
            decay: default_decay(),
            penalty_max_count: NonZeroU8::new(1).unwrap(),
            ngram_min_size: NonZeroU8::new(1).unwrap(),
            ngram_max_size: NonZeroU8::new(4).unwrap(),
            // 1.05 — `1.05^4 ≈ 1.22` keeps the 4-gram stacked multiplicative
            // inside the model's natural top-k spread. 1.06 (the prior default)
            // → `1.06^4 ≈ 1.27` was already pushing factual tokens out of
            // contention on big-model prose.
            penalty_repeat: 1.05,
            // 0.125 (was 0.1) and 0.0625 (was 0.1) — the saturated additive
            // contribution is bounded by `1 / (1 - decay) * penalty_freq +
            // penalty_present` ≈ 2.6 at these defaults. Comfortable inside any
            // model's natural top-k spread.
            penalty_freq: 0.125,
            penalty_present: 0.0625,
            // Surgical-on (was off): only penalises the *next- extension* token
            // of a recurring n-gram, not every trailing token of every tracked
            // n-gram. On big-vocab models this preserves digits and proper
            // nouns even when the surrounding bigrams repeat. Small-vocab
            // models that prefer broader penalty pressure can opt out via
            // per-model sidecar.
            surgical: true,
        }
    }
}

/// Which ngram-size bound was just set, for [`RepetitionOptions::normalize_ngram_range`].
#[derive(Clone, Copy)]
enum NgramBound {
    Min,
    Max,
}

impl RepetitionOptions {
    /// The crate's one cross-field invariant: the n-gram size range must be
    /// ordered. A violated pair makes the generation-time `(min..=max)` range
    /// empty, silently disabling the penalty.
    #[inline]
    fn ngram_range_ordered(min: NonZeroU8, max: NonZeroU8) -> bool {
        min <= max
    }

    /// Restore `ngram_min_size <= ngram_max_size` in place with "just-set field
    /// wins" semantics: the bound that was just assigned is authoritative, and
    /// the other is raised/lowered to meet it. Used by the clamping ngram-size
    /// setters and by the egui tail (to close immediate-mode one-frame lag).
    fn normalize_ngram_range(&mut self, just_set: NgramBound) {
        if Self::ngram_range_ordered(self.ngram_min_size, self.ngram_max_size) {
            return;
        }
        match just_set {
            NgramBound::Min => self.ngram_max_size = self.ngram_min_size,
            NgramBound::Max => self.ngram_min_size = self.ngram_max_size,
        }
    }

    /// [`IgnoreCategory`]s of tokens. These are never penalized.
    pub fn ignored_categories(&self) -> &BTreeSet<IgnoreCategory> {
        &self.ignored_categories
    }

    /// Use [`RepetitionOptions::ignored_categories`] instead.
    #[allow(deprecated)]
    #[deprecated(since = "0.7.0", note = "renamed to `ignored_categories`")]
    pub fn ignored_stopwords(&self) -> &BTreeSet<StopWords> {
        self.ignored_categories()
    }

    /// Set [`IgnoreCategory`]s to ignore. These are never penalized.
    pub fn set_ignored_categories<It>(mut self, ignored_categories: It) -> Self
    where
        It: IntoIterator<Item = IgnoreCategory>,
    {
        self.ignored_categories = ignored_categories.into_iter().collect();
        self
    }

    /// Use [`RepetitionOptions::set_ignored_categories`] instead.
    #[allow(deprecated)]
    #[deprecated(since = "0.7.0", note = "renamed to `set_ignored_categories`")]
    pub fn set_ignored_stopwords<It>(self, ignored_stopwords: It) -> Self
    where
        It: IntoIterator<Item = StopWords>,
    {
        self.set_ignored_categories(ignored_stopwords)
    }

    /// Extend the list of [`IgnoreCategory`]
    pub fn extend_ignored_categories<It>(&mut self, ignored_categories: It)
    where
        It: IntoIterator<Item = IgnoreCategory>,
    {
        self.ignored_categories.extend(ignored_categories);
    }

    /// Use [`RepetitionOptions::extend_ignored_categories`] instead.
    #[allow(deprecated)]
    #[deprecated(
        since = "0.7.0",
        note = "renamed to `extend_ignored_categories`"
    )]
    pub fn extend_ignored_stopwords<It>(&mut self, stopwords: It)
    where
        It: IntoIterator<Item = StopWords>,
    {
        self.extend_ignored_categories(stopwords);
    }

    /// Ngrams to ignore. These are never penalized.
    ///
    /// # Notes:
    /// * any tokens in [`PredictOptions::stop_sequences`], including EOS, are
    /// automatically ignored.
    ///
    /// [`PredictOptions::stop_sequences`]: crate::PredictOptions::stop_sequences
    pub fn ignored(&self) -> &BTreeSet<NGram> {
        &self.ignored
    }

    /// Set the set of ignored ngrams. These are never penalized. The input can
    /// be any iterable of any type that can be converted into an [`NGram`].
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
        self.ignored = ignored.into_iter().map(|t| t.into()).collect();
        self
    }

    /// Extend the set of ignored ngrams. Input can be any iterable of any type
    /// that can be converted into an [`NGram`]. Duplicates are silently
    /// dropped by the underlying [`BTreeSet`].
    ///
    /// # Deprecation:
    /// In the future, to be more consistent, this will take and return `self`.
    pub fn extend_ignored<It, Ng>(&mut self, ignored: It)
    where
        It: IntoIterator<Item = Ng>,
        Ng: Into<NGram>,
    {
        self.ignored.extend(ignored.into_iter().map(|t| t.into()));
    }

    /// Add a token to the set of ignored tokens. The input can be anything
    /// convertible into an [`NGram`].
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
        self.ignored.insert(ngram.into());
    }

    /// Ignore [`IgnoreCategory`]s. These are commonly used tokens that are
    /// often ignored in text generation.
    pub fn ignore_categories<M: crate::backend::Model>(
        mut self,
        ignore_categories: IgnoreCategory,
        model: &M,
    ) -> Self {
        self.extend_ignored(ignore_categories.into_tokens(model));
        self
    }

    /// Use [`RepetitionOptions::ignore_categories`] instead.
    #[allow(deprecated)]
    #[deprecated(since = "0.7.0", note = "renamed to `ignore_categories`")]
    pub fn ignore_stopwords<M: crate::backend::Model>(
        self,
        stopwords: StopWords,
        model: &M,
    ) -> Self {
        self.ignore_categories(stopwords, model)
    }

    /// Minimum n-gram size to penalize, in tokens. Always `<=`
    /// [`ngram_max_size`](Self::ngram_max_size).
    pub fn ngram_min_size(&self) -> NonZeroU8 {
        self.ngram_min_size
    }

    /// Set the minimum n-gram size. Clamped to uphold the
    /// `ngram_min_size <= ngram_max_size` invariant with **"just-set wins"**:
    /// if the new minimum exceeds the current maximum, the maximum is raised to
    /// match. (Values are additionally clamped to `[1, NGram::CAPACITY]` at
    /// apply time.)
    ///
    /// # Examples (starting from the default `min=1, max=4`)
    /// ```text
    /// .set_ngram_min_size(6).set_ngram_max_size(2)  // -> (6,6) then (2,2)
    /// .set_ngram_max_size(2).set_ngram_min_size(6)  // -> (1,2) then (6,6)
    /// ```
    /// The end state is order-dependent because the most recently set bound
    /// always wins; both results are valid ordered pairs.
    pub fn set_ngram_min_size(mut self, ngram_min_size: NonZeroU8) -> Self {
        self.ngram_min_size = ngram_min_size;
        self.normalize_ngram_range(NgramBound::Min);
        self
    }

    /// Maximum n-gram size to penalize, in tokens (capped at
    /// [`NGram::CAPACITY`] at apply time). Always `>=`
    /// [`ngram_min_size`](Self::ngram_min_size).
    pub fn ngram_max_size(&self) -> NonZeroU8 {
        self.ngram_max_size
    }

    /// Set the maximum n-gram size. Clamped to uphold the
    /// `ngram_min_size <= ngram_max_size` invariant with **"just-set wins"**:
    /// if the new maximum is below the current minimum, the minimum is lowered
    /// to match. See [`set_ngram_min_size`](Self::set_ngram_min_size) for
    /// worked ordering examples.
    pub fn set_ngram_max_size(mut self, ngram_max_size: NonZeroU8) -> Self {
        self.ngram_max_size = ngram_max_size;
        self.normalize_ngram_range(NgramBound::Max);
        self
    }

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

    /// The additive frequency penalty. Subtracts the windowed, decay-weighted
    /// count of the n-gram from the penalized token's logit.
    pub fn penalty_freq(&self) -> f32 {
        self.penalty_freq
    }

    /// Set the additive frequency penalty.
    pub fn set_penalty_freq(mut self, penalty_freq: f32) -> Self {
        self.penalty_freq = penalty_freq;
        self
    }

    /// The additive presence penalty. Subtracted once from the penalized
    /// token's logit whenever the n-gram is present in the window.
    pub fn penalty_present(&self) -> f32 {
        self.penalty_present
    }

    /// Set the additive presence penalty.
    pub fn set_penalty_present(mut self, penalty_present: f32) -> Self {
        self.penalty_present = penalty_present;
        self
    }

    /// Whether surgical mode is enabled. See the field docs for the
    /// next-extension-token targeting behavior.
    pub fn surgical(&self) -> bool {
        self.surgical
    }

    /// Enable or disable surgical mode.
    pub fn set_surgical(mut self, surgical: bool) -> Self {
        self.surgical = surgical;
        self
    }

    /// Sliding-window size in generation steps. See field docs.
    pub fn window_size(&self) -> NonZeroU32 {
        self.window_size
    }

    /// Set the sliding-window size in generation steps.
    pub fn set_window_size(mut self, window_size: NonZeroU32) -> Self {
        self.window_size = window_size;
        self
    }

    /// Per-step decay applied to in-window occurrences. See field docs.
    pub fn decay(&self) -> f32 {
        self.decay
    }

    /// Set the per-step decay. Recommended range `0.95..=0.99`; clamped
    /// internally to `(0.0, 1.0]`.
    pub fn set_decay(mut self, decay: f32) -> Self {
        self.decay = decay;
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
            let mut to_remove: Option<IgnoreCategory> = None;
            for category in self.ignored_categories.iter() {
                ui.horizontal(|ui| {
                    if ui
                        .add(egui::Button::image(DELETE_ICON))
                        .on_hover_text_at_pointer("Remove this category from the list of ignored token sets.")
                        .clicked() {
                        to_remove = Some(*category);
                    };
                    ui.label(category.as_str())
                });
            }
            if let Some(category) = to_remove {
                self.ignored_categories.remove(&category);
            }
        }
        if self.ignored_categories.len() < IgnoreCategory::ALL.len() {
            egui::ComboBox::from_label("to ignore token categories for")
                .selected_text("Select a category...")
                .show_ui(ui, |ui| {
                    for category in IgnoreCategory::ALL {
                        if !self.ignored_categories.contains(&category) {
                            if ui
                                .selectable_label(false, category.as_str())
                                .clicked()
                            {
                                self.ignored_categories.insert(category);
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
            let mut to_remove: Option<NGram> = None;
            for ngram in self.ignored.iter() {
                ui.horizontal(|ui| {
                    if ui
                        .add(egui::Button::image(DELETE_ICON))
                        .on_hover_text_at_pointer("Remove this ngram from the list of ignored ngrams.")
                        .clicked() {
                        to_remove = Some(*ngram);
                    };
                    ui.label(format!("{:?}", ngram))
                });
            }
            // TODO: add a way to add ngrams. This requires a buffer to store
            // the input. We can use &mut function argument for this since we
            // don't have a place on this struct to store it. We could also
            // serde(skip) it.

            if let Some(ngram) = to_remove {
                self.ignored.remove(&ngram);
            }
        }

        // Penalty max count
        let mut resp = ui.horizontal(|ui| {
            let mut max_count = self.penalty_max_count.get();
            let inner = ui.label("Penalty Max Count") |
            ui.add(egui::DragValue::new(&mut max_count).range(1..=255))
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
                // Dynamic upper bound keeps `min <= max` within the frame; the
                // tail normalize closes the one-frame lag (see end of fn).
                let max_bound = self.ngram_max_size.get();
                let inner = ui.label("Ngram Min Size")
                    | ui.add(
                        egui::DragValue::new(&mut min_size)
                            .range(1..=max_bound),
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
                // Dynamic lower bound keeps `min <= max` within the frame.
                let min_bound = self.ngram_min_size.get();
                let inner = ui.label("Ngram Max Size")
                    | ui.add(
                        egui::DragValue::new(&mut max_size)
                            .range(min_bound..=NGram::CAPACITY as u8),
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

        // Window size — how many recent generation steps contribute to
        // the windowed-decay penalty. Older occurrences are evicted.
        resp |= ui
            .horizontal(|ui| {
                let mut window = self.window_size.get();
                let inner = ui.label("Window size")
                    | ui.add(egui::DragValue::new(&mut window).range(1..=8192))
                        .on_hover_text_at_pointer(
                            "Sliding-window size in generation steps. Only \
                         occurrences within the last N steps contribute \
                         to the penalty. Bounds the additive term so \
                         long generations don't have their logit \
                         gradient dominated.",
                        );
                self.window_size = NonZeroU32::new(window.max(1)).unwrap();
                inner
            })
            .inner;

        // Decay — per-step multiplicative weight on aging occurrences.
        resp |= ui
            .horizontal(|ui| {
                ui.label("Decay")
                    | ui.add(
                        egui::Slider::new(&mut self.decay, 0.5..=1.0)
                            .step_by(0.005),
                    )
                    .on_hover_text_at_pointer(
                        "Per-step decay applied to in-window \
                         occurrences. Sustained-repetition effective \
                         count is bounded above by 1 / (1 - decay). \
                         0.95 caps it at ~20; 0.99 at ~100; 1.0 \
                         disables decay (all in-window occurrences \
                         count fully).",
                    )
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

        // Close the immediate-mode one-frame lag: each ngram-size slider's
        // range derives from the *other* field's value as of frame start, so a
        // drag that lowers max below an already-committed min can leave an
        // inverted pair for a single frame. This is frame-ordering, not a data
        // race (egui is single-threaded). Treat max as authoritative here,
        // mirroring `set_ngram_max_size`'s "just-set wins".
        self.normalize_ngram_range(NgramBound::Max);

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
    #[error("ngram_min_size ({min}) must be <= ngram_max_size ({max})")]
    InvalidNgramRange { min: NonZeroU8, max: NonZeroU8 },
}

static_assertions::assert_impl_all!(RepetitionError: Send, Sync);

fn ngram_is_ignored(ngram: NGram, ignored: &BTreeSet<NGram>) -> bool {
    ignored.contains(&ngram)
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
    tokens: &[Token],
    ignored: &BTreeSet<NGram>,
) -> Option<Token> {
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
pub fn apply_sample_repetition_ngram<M: crate::backend::Model>(
    candidates: Candidates,
    tokens: &[Token],
    opts: &mut RepetitionOptions,
    freq_map: &mut NGramStats,
    model: &M,
) -> Result<Candidates, RepetitionError> {
    let k = candidates.len();
    let n_vocab = k.get();
    let mut candidates = candidates.sort(Sorted::ById { k });

    let RepetitionOptions {
        ignored_categories,
        ignored,
        window_size,
        decay,
        ngram_max_size,
        ngram_min_size,
        penalty_freq,
        penalty_max_count,
        penalty_present,
        penalty_repeat,
        surgical,
    } = opts;

    // Drain ignored categories into the ignored set. BTreeSet::insert
    // silently handles duplicates; no manual dedup check needed.
    while let Some(cats) = ignored_categories.pop_first() {
        for token in cats.into_tokens(model) {
            ignored.insert(NGram::from(token));
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
    // Invariant established by strict deserialize, the clamping ngram-size
    // setters, and dynamic GUI ranges: no construction path can produce
    // min > max. CAPACITY-clamping both to `[1, CAPACITY]` above cannot invert
    // an already-ordered pair, so this assert documents the contract and
    // surfaces validation holes in tests rather than masking them (the old
    // min/max swap silently disabled the penalty on an inverted pair).
    debug_assert!(
        ngram_min_size <= ngram_max_size,
        "ngram_min_size ({ngram_min_size}) > ngram_max_size \
         ({ngram_max_size}): a construction path bypassed RepetitionOptions \
         validation"
    );
    // `effective_count > penalty_max_count` triggers the multiplicative
    // term (analogue of the old raw-count threshold).
    let penalty_max_count_f = penalty_max_count.get() as f32;
    let window_size = window_size.get();
    // Clamp decay to (0, 1]; out-of-range values would either wedge the
    // sum to NaN/inf or invert the decay direction.
    let decay = decay.clamp(f32::MIN_POSITIVE, 1.0);

    // The "current step" we attribute to occurrences added in this call.
    // Trailing n-grams all live at the most recent token position; using a
    // single value per call keeps eviction cheap and matches what
    // production sees (one penalty pass per generation step).
    let current_step = tokens.len() as u64;

    // Drop any positions that fell out of the window since the last call,
    // before recording the new ones. Cheap: only pops the front entries
    // that just rolled out.
    freq_map.evict_outside_window(current_step, window_size);

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
        freq_map.add(ngram, &candidates, current_step);
    }

    if *surgical {
        // Phase 2 (surgical): For each repeating n-gram, penalize the earliest
        // token that would extend the match — slice[k], where k is the longest
        // prefix already re-emitted in the trailing history. With k=0 (no
        // prefix re-emitted) we penalize the first token to prevent the phrase
        // from starting; with k>0 we penalize the next continuation.
        //
        // Example: freq_map contains [The, New, York] with effective_count=2.
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
            let effective = data.windowed_decayed_count(current_step, decay);
            if effective <= penalty_max_count_f {
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
            candidate.logit -= effective * *penalty_freq + *penalty_present;
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
            let effective = data.windowed_decayed_count(current_step, decay);

            // Multiplicative penalty: scales with n-gram size. Only fires when
            // effective count exceeds max_count.
            if effective > penalty_max_count_f {
                let scaled_penalty = penalty_repeat.powf(ngram.len() as f32);
                if candidate.logit <= 0.0 {
                    candidate.logit *= scaled_penalty;
                } else {
                    candidate.logit /= scaled_penalty;
                }
            }

            // Additive penalties: frequency scales with the windowed,
            // decay-weighted count (bounded above by `1 / (1 - decay)`),
            // presence is binary.
            candidate.logit -= effective * *penalty_freq + *penalty_present;
        }
    }

    // We have modified logit values.
    candidates.softmax_applied_to = None;

    Ok(candidates)
}

/// Invariant / builder / deserialize tests that need no model — run under any
/// test build (the integration suite below is gated behind `llama-cpp`).
#[cfg(test)]
mod invariant_tests {
    use super::*;

    fn nz(n: u8) -> NonZeroU8 {
        NonZeroU8::new(n).unwrap()
    }

    /// "just-set wins": setting min above max raises max, then setting max
    /// below the new min lowers min to match → (2, 2).
    #[test]
    fn set_min_then_max_just_set_wins() {
        let o = RepetitionOptions::default()
            .set_ngram_min_size(nz(6))
            .set_ngram_max_size(nz(2));
        assert_eq!(o.ngram_min_size(), nz(2));
        assert_eq!(o.ngram_max_size(), nz(2));
    }

    /// Reverse order lands at (6, 6) — the most recently set bound wins.
    #[test]
    fn set_max_then_min_just_set_wins() {
        let o = RepetitionOptions::default()
            .set_ngram_max_size(nz(2))
            .set_ngram_min_size(nz(6));
        assert_eq!(o.ngram_min_size(), nz(6));
        assert_eq!(o.ngram_max_size(), nz(6));
    }

    /// Property: no setter ordering for any pair in `1..=CAPACITY` can leave
    /// an inverted range.
    #[test]
    fn setters_never_invert() {
        for min in 1..=NGram::CAPACITY as u8 {
            for max in 1..=NGram::CAPACITY as u8 {
                let a = RepetitionOptions::default()
                    .set_ngram_min_size(nz(min))
                    .set_ngram_max_size(nz(max));
                assert!(a.ngram_min_size() <= a.ngram_max_size());
                let b = RepetitionOptions::default()
                    .set_ngram_max_size(nz(max))
                    .set_ngram_min_size(nz(min));
                assert!(b.ngram_min_size() <= b.ngram_max_size());
            }
        }
    }

    /// Each new getter returns what the matching setter set.
    #[test]
    fn new_getters_roundtrip() {
        let o = RepetitionOptions::default()
            .set_penalty_freq(0.3)
            .set_penalty_present(0.4)
            .set_surgical(false)
            .set_ngram_min_size(nz(2))
            .set_ngram_max_size(nz(5));
        assert_eq!(o.penalty_freq(), 0.3);
        assert_eq!(o.penalty_present(), 0.4);
        assert!(!o.surgical());
        assert_eq!(o.ngram_min_size(), nz(2));
        assert_eq!(o.ngram_max_size(), nz(5));
    }

    // Deserialize wiring is exercised through the real-world TOML path.
    #[cfg(feature = "toml")]
    mod de {
        use super::*;

        // Every field without a serde default — required in TOML.
        const REQUIRED: &str = "\
            penalty_max_count = 1\n\
            ngram_min_size = 1\n\
            ngram_max_size = 4\n\
            penalty_repeat = 1.05\n\
            penalty_freq = 0.125\n\
            penalty_present = 0.0625\n";

        /// Strict: an inverted range is rejected at deserialize, not silently
        /// clamped. Regression guard for the dropped apply-site swap.
        #[test]
        fn rejects_inverted_ngram_range() {
            let doc = "\
                penalty_max_count = 1\n\
                ngram_min_size = 4\n\
                ngram_max_size = 2\n\
                penalty_repeat = 1.05\n\
                penalty_freq = 0.125\n\
                penalty_present = 0.0625\n";
            let err =
                ::toml::from_str::<RepetitionOptions>(doc).unwrap_err();
            let msg = err.to_string();
            assert!(
                msg.contains("ngram_min_size") && msg.contains("must be <="),
                "expected InvalidNgramRange message, got: {msg}"
            );
        }

        /// Omitted defaulted fields fall back to their serde defaults. Note
        /// `ignored_categories` defaults to an *empty* set (the field-level
        /// `#[serde(default)]`), not the rich `RepetitionOptions::default()`.
        #[test]
        fn partial_applies_defaults() {
            let o: RepetitionOptions =
                ::toml::from_str(REQUIRED).unwrap();
            assert_eq!(o.window_size().get(), 256);
            assert_eq!(o.decay(), 0.95);
            assert!(!o.surgical());
            assert!(o.ignored().is_empty());
            assert!(o.ignored_categories().is_empty());
        }

        /// The `ignored_stopwords` legacy key still maps onto
        /// `ignored_categories`.
        #[test]
        fn ignored_stopwords_alias() {
            let doc =
                format!("{REQUIRED}ignored_categories = [\"English\"]\n");
            let o: RepetitionOptions =
                ::toml::from_str(&doc).unwrap();
            assert!(o.ignored_categories().contains(&IgnoreCategory::English));

            let doc =
                format!("{REQUIRED}ignored_stopwords = [\"Json\"]\n");
            let o: RepetitionOptions =
                ::toml::from_str(&doc).unwrap();
            assert!(o.ignored_categories().contains(&IgnoreCategory::Json));
        }

        /// A valid config (including the default) round-trips through TOML.
        #[test]
        fn default_roundtrips() {
            let s = ::toml::to_string(&RepetitionOptions::default()).unwrap();
            let o: RepetitionOptions = ::toml::from_str(&s).unwrap();
            assert_eq!(o, RepetitionOptions::default());
        }
    }
}

#[cfg(all(test, feature = "llama-cpp"))]
mod tests {
    use super::*;
    use crate::{Candidates, TokenData};
    use std::path::PathBuf;

    /// Helper: create candidates sorted by id with given logits.
    fn make_candidates(logits: &[f32]) -> Candidates {
        let data: Vec<TokenData> = logits
            .iter()
            .enumerate()
            .map(|(i, &logit)| TokenData {
                id: i as i32,
                logit,
                p: 0.0,
            })
            .collect();
        Candidates::from_vec(data)
    }

    fn load_model() -> crate::LlamaCppModel {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("models/model.gguf");
        crate::LlamaCppModel::from_file(path, None).unwrap()
    }

    /// Simulate multiple generation steps, applying the penalty at each step.
    /// Returns (logits_after, freq_map) so callers can inspect both.
    fn run_penalty_steps(
        logits: &[f32],
        token_history: &[Token],
        opts: &mut RepetitionOptions,
        steps: usize,
        model: &crate::LlamaCppModel,
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
        // Empty ignore set: the vocab here is synthetic (ids 0..10), and
        // the default categories resolve against the *real* model's
        // tokenizer — whether id 5 collides with a stopword is an
        // accident of which model.gguf is linked.
        let mut opts = RepetitionOptions {
            penalty_repeat: 1.15,
            penalty_freq: 0.0,
            penalty_present: 0.5,
            ignored_categories: BTreeSet::new(),
            ..RepetitionOptions::default()
        };

        // Two steps: `penalty_max_count` defaults to 1, meaning the
        // penalty deliberately fires from the *second* sighting of an
        // n-gram (the v0.8.0 anti-fact-swap tuning). One step leaves
        // the logit untouched by design.
        let (after, _) =
            run_penalty_steps(&logits, &tokens, &mut opts, 2, &model);

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
        // Empty ignore set: synthetic vocab; see test_repetition_penalty_basic.
        let mut freq_opts = RepetitionOptions {
            penalty_repeat: 1.0,
            penalty_freq: 0.1,
            penalty_present: 0.0,
            ignored_categories: BTreeSet::new(),
            ..RepetitionOptions::default()
        };
        let (after_freq, _) =
            run_penalty_steps(&logits, &tokens, &mut freq_opts, 3, &model);

        // Presence only — after 3 steps, each subtracts 0.5, total 1.5
        let mut pres_opts = RepetitionOptions {
            penalty_repeat: 1.0,
            penalty_freq: 0.0,
            penalty_present: 0.5,
            ignored_categories: BTreeSet::new(),
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

        // `penalty_max_count` defaults to 1, so step 1 (count=1) is
        // exempt and the penalty fires on steps 2 and 3 only.
        // Frequency scales with the count when it fires: 2*0.1 + 3*0.1.
        // Presence is constant per firing step: 2 * 0.5.
        let freq_delta = (logits[3] - after_freq[3]).abs();
        let pres_delta = (logits[3] - after_pres[3]).abs();
        println!(
            "  freq total delta: {:.4}, pres total delta: {:.4}",
            freq_delta, pres_delta
        );
        assert_approx_eq!(freq_delta, 0.5, 0.01);
        assert_approx_eq!(pres_delta, 1.0, 0.01);
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

    /// Long-generation regression test: with the windowed-decay
    /// `RepetitionOptions::default()` (window_size=256, decay=0.95) the
    /// popular-vs-rare logit gap **converges** instead of growing
    /// linearly. Before the fix the additive `count * penalty_freq`
    /// term grew unboundedly with generation length (~20 logits below
    /// the rare token at step 200, ~60 at step 600). With the fix the
    /// windowed effective count is bounded above by `1 / (1 - decay) =
    /// 20` per tracked n-gram, so once the window fills the gap
    /// saturates and stays put.
    ///
    /// The exact saturated magnitude depends on how many n-gram sizes
    /// stack on the same target token in the test scenario (broad mode
    /// applies size-1..=size-4 penalties in parallel for a single
    /// repeating token). The contract this test enforces is the
    /// structural one: gap at step 4*window_size ≈ gap at step
    /// window_size, within a small tolerance. That proves the additive
    /// term is no longer dominating the logit gradient.
    ///
    /// Mirrors production: each call gets fresh logits from the model;
    /// only `freq_map` is shared across calls; `tokens` grows by one
    /// popular emission per step so positions advance and eviction
    /// kicks in once the window fills.
    #[test]
    #[ignore = "long-running; uses real model"]
    fn long_generation_does_not_swamp_popular_token_logit() {
        let model = load_model();
        let n_vocab = 1024;
        let popular: Token = 100;
        let rare: Token = 200;
        let baseline: Vec<f32> = (0..n_vocab).map(|_| 1.0).collect();
        let mut tokens: Vec<Token> = Vec::new();

        let mut opts = RepetitionOptions::default();
        let window = opts.window_size().get() as usize;
        let mut freq_map = NGramStats::new();

        let snapshot = |result: &Candidates| -> f32 {
            let pop = result
                .iter()
                .find(|c| c.id == popular)
                .map(|c| c.logit)
                .unwrap();
            let rar = result
                .iter()
                .find(|c| c.id == rare)
                .map(|c| c.logit)
                .unwrap();
            rar - pop
        };

        // Snapshot the gap at three points: just past the first full
        // window (saturation), 2x window, and 4x window. If the bug
        // were back, the 4x measurement would dominate the 1x
        // measurement; with the fix they should be within rounding.
        let mut gap_at_window = 0.0_f32;
        let mut gap_at_2window = 0.0_f32;
        let mut gap_at_4window = 0.0_f32;
        for step in 0..(window * 4) {
            tokens.push(popular);
            let candidates = make_candidates(&baseline);
            let result = apply_sample_repetition_ngram(
                candidates,
                &tokens,
                &mut opts,
                &mut freq_map,
                &model,
            )
            .unwrap();
            // step is zero-indexed, so step + 1 is the iteration count.
            if step + 1 == window {
                gap_at_window = snapshot(&result);
            } else if step + 1 == window * 2 {
                gap_at_2window = snapshot(&result);
            } else if step + 1 == window * 4 {
                gap_at_4window = snapshot(&result);
            }
        }

        println!("gap @ 1x window ({}): {gap_at_window:.4}", window);
        println!("gap @ 2x window ({}): {gap_at_2window:.4}", window * 2);
        println!("gap @ 4x window ({}): {gap_at_4window:.4}", window * 4);
        if let Some((_, data)) = freq_map.iter().next() {
            println!(
                "Trailing-unigram tracked: {} positions, \
                 windowed-decayed: {:.2}",
                data.count(),
                data.windowed_decayed_count(tokens.len() as u64, opts.decay())
            );
        }

        // The structural contract: once the window has filled, the gap
        // should be saturated. After another 3x window of the same
        // token, the gap should not grow meaningfully. Pre-fix, the
        // additive term grew ~0.4 logits per step, so 3x window ≈ 300
        // additional steps would have added ~120 logits.
        let growth = gap_at_4window - gap_at_window;
        assert!(
            growth.abs() < 0.5,
            "gap should have saturated by 1x window and held there; \
             gap@1x={gap_at_window:.4}, gap@4x={gap_at_4window:.4}, \
             growth={growth:.4}. Pre-fix growth at this scale was \
             dozens of logits (linear-additive penalty)."
        );
    }

    /// Companion to [`long_generation_does_not_swamp_popular_token_logit`]:
    /// once the popular n-gram stops being emitted, the gap should
    /// shrink as in-window occurrences age out under the decay. By the
    /// time `window_size` more steps have passed without re-emission,
    /// the effective count has fallen to zero and the popular token's
    /// logit returns to baseline.
    #[test]
    #[ignore = "long-running; uses real model"]
    fn gap_decays_after_popular_ngram_exits_window() {
        let model = load_model();
        let n_vocab = 1024;
        let popular: Token = 100;
        let rare: Token = 200;
        let other: Token = 300;
        let baseline: Vec<f32> = (0..n_vocab).map(|_| 1.0).collect();
        let mut tokens: Vec<Token> = Vec::new();

        let mut opts = RepetitionOptions::default();
        let window = opts.window_size().get() as usize;
        let mut freq_map = NGramStats::new();

        // Phase A: saturate. Emit `popular` for `window` steps so the
        // unigram's effective count converges near `1 / (1 - decay)`.
        for _ in 0..window {
            tokens.push(popular);
            let candidates = make_candidates(&baseline);
            let _ = apply_sample_repetition_ngram(
                candidates,
                &tokens,
                &mut opts,
                &mut freq_map,
                &model,
            )
            .unwrap();
        }

        // Snapshot the saturated gap.
        let saturated_candidates = make_candidates(&baseline);
        let saturated = apply_sample_repetition_ngram(
            saturated_candidates,
            &tokens,
            &mut opts,
            &mut freq_map,
            &model,
        )
        .unwrap();
        let popular_sat = saturated
            .iter()
            .find(|c| c.id == popular)
            .map(|c| c.logit)
            .unwrap();
        let rare_sat = saturated
            .iter()
            .find(|c| c.id == rare)
            .map(|c| c.logit)
            .unwrap();
        let gap_sat = rare_sat - popular_sat;

        // Phase B: stop emitting popular. Emit a different token for
        // `window` more steps; the popular n-gram should age out of the
        // window entirely.
        for _ in 0..window + 4 {
            tokens.push(other);
            let candidates = make_candidates(&baseline);
            let _ = apply_sample_repetition_ngram(
                candidates,
                &tokens,
                &mut opts,
                &mut freq_map,
                &model,
            )
            .unwrap();
        }

        let final_candidates = make_candidates(&baseline);
        let final_result = apply_sample_repetition_ngram(
            final_candidates,
            &tokens,
            &mut opts,
            &mut freq_map,
            &model,
        )
        .unwrap();
        let popular_final = final_result
            .iter()
            .find(|c| c.id == popular)
            .map(|c| c.logit)
            .unwrap();
        let rare_final = final_result
            .iter()
            .find(|c| c.id == rare)
            .map(|c| c.logit)
            .unwrap();
        let gap_final = rare_final - popular_final;

        println!(
            "saturated gap: {gap_sat:.4}, decayed gap after {} idle \
             steps: {gap_final:.4}",
            window + 4
        );

        // After the popular n-gram has fully aged out, the gap should
        // be effectively zero (popular returns to baseline). Allow a
        // little slack for the trailing-token presence penalty if any
        // popular position is still inside the window.
        assert!(
            gap_final < gap_sat / 2.0,
            "gap should have decayed after popular n-gram exited the \
             window; saturated={gap_sat:.4}, final={gap_final:.4}"
        );
    }

    // --- surgical_target tests (no model required) -----------------------

    fn ngram(tokens: &[Token]) -> NGram {
        NGram::try_from(tokens).unwrap()
    }

    fn ignored_of(tokens: &[Token]) -> BTreeSet<NGram> {
        tokens.iter().map(|&t| NGram::from(t)).collect()
    }

    /// k=0: no prefix re-emitted yet — penalize the first token of the
    /// n-gram. This blocks proper-noun repetitions like "The New York Times"
    /// at entry, before "The" is even selected.
    #[test]
    fn surgical_target_k0_penalizes_first_token() {
        let ng = ngram(&[10, 20, 30]);
        let tokens = &[99, 88, 77];
        assert_eq!(surgical_target(&ng, tokens, &BTreeSet::new()), Some(10));
    }

    /// k=1: first token re-emitted — penalize the next continuation to
    /// prevent the phrase from growing past its first token.
    #[test]
    fn surgical_target_k1_penalizes_second_token() {
        let ng = ngram(&[10, 20, 30]);
        let tokens = &[99, 10];
        assert_eq!(surgical_target(&ng, tokens, &BTreeSet::new()), Some(20));
    }

    /// k=n-1: full prefix re-emitted — penalize the final completion token.
    #[test]
    fn surgical_target_k_full_penalizes_completion() {
        let ng = ngram(&[10, 20, 30]);
        let tokens = &[10, 20];
        assert_eq!(surgical_target(&ng, tokens, &BTreeSet::new()), Some(30));
    }

    /// Longest prefix match wins when the history happens to match multiple
    /// prefix lengths (here the history ends in [10, 10], both k=1 and k=2
    /// for an n-gram of [10, 10, 30]).
    #[test]
    fn surgical_target_picks_longest_prefix_match() {
        let ng = ngram(&[10, 10, 30]);
        let tokens = &[10, 10];
        assert_eq!(surgical_target(&ng, tokens, &BTreeSet::new()), Some(30));
    }

    /// If the target token is in the ignored set, return None — do NOT
    /// fall through to slice[k+1]. That fall-through was rejected because
    /// it produces odd partial completions (e.g. "The New Potato" when the
    /// lowercase "the" stopword is skipped).
    #[test]
    fn surgical_target_skips_entirely_when_target_ignored() {
        let ng = ngram(&[10, 20, 30]);
        let ignored = ignored_of(&[10]);
        let tokens: &[Token] = &[];
        assert_eq!(surgical_target(&ng, tokens, &ignored), None);
    }

    /// Ignored check applies at whichever k is selected, not only k=0.
    #[test]
    fn surgical_target_ignored_continuation_skips() {
        let ng = ngram(&[10, 20, 30]);
        let ignored = ignored_of(&[20]);
        let tokens = &[10];
        assert_eq!(surgical_target(&ng, tokens, &ignored), None);
    }

    /// Unigram: always k=0, target is the only token.
    #[test]
    fn surgical_target_unigram() {
        let ng = ngram(&[42]);
        let tokens = &[99];
        assert_eq!(surgical_target(&ng, tokens, &BTreeSet::new()), Some(42));
        let ignored = ignored_of(&[42]);
        assert_eq!(surgical_target(&ng, tokens, &ignored), None);
    }

    /// History shorter than the prefix shouldn't panic or over-match.
    #[test]
    fn surgical_target_short_history() {
        let ng = ngram(&[10, 20, 30]);
        let empty = BTreeSet::new();
        // Empty history: only k=0 is possible.
        assert_eq!(surgical_target(&ng, &[], &empty), Some(10));
        // History of length 1 that doesn't match prefix[0]: k=0.
        assert_eq!(surgical_target(&ng, &[99], &empty), Some(10));
        // History of length 1 matching prefix[0]: k=1.
        assert_eq!(surgical_target(&ng, &[10], &empty), Some(20));
    }

    /// A prefix match that doesn't reach the end of history: not a match.
    /// (We only look at the trailing suffix, not arbitrary substrings.)
    #[test]
    fn surgical_target_only_trailing_suffix_counts() {
        let ng = ngram(&[10, 20, 30]);
        // [10, 20] appears earlier, but history ends in [99]: k=0.
        let tokens = &[10, 20, 99];
        assert_eq!(surgical_target(&ng, tokens, &BTreeSet::new()), Some(10));
    }
}
