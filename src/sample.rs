use crate::{model::Vocab, ngram::NGramStats, Candidates, Probability};

use llama_cpp_sys_3::llama_token;
use xorshift::Rng;

use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

pub(crate) mod grammar;
mod json;
mod repetition;

pub use grammar::{Grammar, GrammarError, GrammarState};
pub use json::{JsonError, JsonState};
pub use repetition::{
    apply_sample_repetition_ngram, RepetitionError, RepetitionOptions,
};

/// Serialize a `SamplingMode::Json` parser state.
///
/// Locks the mutex momentarily and delegates to [`JsonState`]'s derive. The
/// wrapping `Arc<Mutex<…>>` is discarded on the wire — deserialization
/// reconstructs a fresh Arc/Mutex around the decoded state.
#[cfg(feature = "serde")]
fn serialize_json_state<S>(
    arc: &Arc<Mutex<JsonState>>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: rocket::serde::Serializer,
{
    use rocket::serde::Serialize;
    // If the mutex is poisoned, fall back to a fresh state. This keeps
    // serialization from panicking in degraded conditions; the cost is that
    // the serialized snapshot will be "fresh" instead of the actual state.
    // Better than taking down the serializer.
    match arc.lock() {
        Ok(guard) => guard.serialize(serializer),
        Err(_) => JsonState::new().serialize(serializer),
    }
}

/// Deserialize a `SamplingMode::Json` parser state into a fresh
/// `Arc<Mutex<JsonState>>`.
#[cfg(feature = "serde")]
fn deserialize_json_state<'de, D>(
    deserializer: D,
) -> Result<Arc<Mutex<JsonState>>, D::Error>
where
    D: rocket::serde::Deserializer<'de>,
{
    use rocket::serde::Deserialize;
    let state = JsonState::deserialize(deserializer)?;
    Ok(Arc::new(Mutex::new(state)))
}

/// Serialize a `SamplingMode::Grammar` by writing out the GBNF source
/// text only. The matcher's mid-parse position is NOT serialized — on
/// deserialize a fresh matcher is constructed from the source. This
/// mirrors how the Json variant reconstructs its `Arc<Mutex<…>>` wrapping
/// but accepts that grammar position info is lost across reload.
#[cfg(feature = "serde")]
fn serialize_grammar<S>(
    arc: &Arc<Mutex<GrammarState>>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: rocket::serde::Serializer,
{
    use rocket::serde::Serialize;
    let source = match arc.lock() {
        Ok(guard) => guard.grammar().source().to_owned(),
        Err(poisoned) => poisoned.into_inner().grammar().source().to_owned(),
    };
    source.serialize(serializer)
}

/// Deserialize a `SamplingMode::Grammar` by re-parsing the GBNF source
/// text and constructing a fresh matcher.
#[cfg(feature = "serde")]
fn deserialize_grammar<'de, D>(
    deserializer: D,
) -> Result<Arc<Mutex<GrammarState>>, D::Error>
where
    D: rocket::serde::Deserializer<'de>,
{
    use rocket::serde::{de::Error, Deserialize};
    let source = String::deserialize(deserializer)?;
    let grammar = Grammar::parse(&source).map_err(D::Error::custom)?;
    Ok(Arc::new(Mutex::new(GrammarState::new(Arc::new(grammar)))))
}

#[cfg(feature = "egui")]
pub(crate) const DELETE_ICON: egui::ImageSource<'static> =
    egui::include_image!("../assets/ui/images/delete.png");

#[cfg_attr(
    feature = "serde",
    derive(rocket::serde::Deserialize, rocket::serde::Serialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "rocket::serde"))]
/// Options determining how raw logits are turned into a token. This is used by
/// [`Candidates::sample_token`] and associated functions.
#[derive(Clone, Debug, PartialEq)]
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

    /// Draw [`egui::Ui`] for [`SampleOptions`] but without the outer
    /// [`egui::CollapsingHeader`].
    #[cfg(feature = "egui")]
    pub fn draw_inner(&mut self, ui: &mut egui::Ui) -> egui::Response {
        let collaping_resp = egui::CollapsingHeader::new("Sampling Modes")
            .show(ui, |ui| {
                // TODO: The user should be able to drag and drop these to reorder them.
                // Otherwise they must remove and re-add them in the desired order.
                // TODO: Test combinations of modes since I am fairly sure some make the
                // assumption all candidates are present and may crash if they are not.
                // TODO: Add temperature to the sampling modes. It's an oversight that
                // it's not here. It's not a priority since it's not very useful on it's
                // own, but it could be in combination with other modes.
                let mut remove = None;
                let n_modes = self.modes.len();
                for (i, mode) in self.modes.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        if n_modes > 1 {
                            // We have at least one mode. We don't want to remove
                            // the last one, even though this can work.
                            if ui
                                .add(egui::Button::image(DELETE_ICON))
                                .on_hover_text_at_pointer(
                                    "Remove this sampling mode.",
                                )
                                .clicked()
                            {
                                remove = Some(i);
                            }
                        }

                        mode.draw(ui, i)
                    })
                    .inner;
                }

                // We could use a Vec, but it's unlikely that a user can delete two
                // modes within a single frame, so this is fine.
                if let Some(i) = remove {
                    if self.modes.len() > 1 {
                        // It should't be possible to remove the last mode, but we check
                        // anyway. There may be cases where data races could cause this,
                        // and I can't prove it's impossible.
                        self.modes.remove(i);
                    }
                }

                // Add a combo box to add a new sampling modes.
                egui::ComboBox::from_label("to add to the above list.")
                    .selected_text("Choose a mode...")
                    .show_ui(ui, |ui| {
                        for mode in SamplingMode::ALL {
                            if ui
                                .selectable_label(false, mode.name())
                                .on_hover_text_at_pointer(mode.help())
                                .clicked()
                            {
                                self.modes.push(mode);
                            }
                        }
                        // JSON is constructed at runtime (it owns an
                        // Arc<Mutex<…>>), so it can't live in the const
                        // `ALL` array. Append it separately.
                        let json_sample = SamplingMode::json();
                        if ui
                            .selectable_label(false, json_sample.name())
                            .on_hover_text_at_pointer(json_sample.help())
                            .clicked()
                        {
                            self.modes.push(json_sample);
                        }
                        // Grammar is the same story. We seed with a
                        // permissive placeholder (`root ::= .+`); the
                        // caller is expected to swap in a real grammar
                        // via `SamplingMode::grammar_from_file`.
                        let grammar_sample =
                            SamplingMode::grammar("root ::= .+").expect(
                                "placeholder grammar `root ::= .+` must \
                                 parse",
                            );
                        if ui
                            .selectable_label(false, grammar_sample.name())
                            .on_hover_text_at_pointer(grammar_sample.help())
                            .clicked()
                        {
                            self.modes.push(grammar_sample);
                        }
                    });
            });

        // Message when the header's text is hovered.
        let mut resp = collaping_resp.header_response
                    .on_hover_text_at_pointer("Add or remove sampling modes. These are applied in top-down order. The idea is to start with all possible candidates for the next token (the entire vocabulary) and reduce them to a single token. If any tokens are left at the end of this list, one will be chosen at random, weighted by their probability. Greedy, Mirostat, and MirostatV2 are guaranteed to return a single token, so they should be the last mode. Any sampling mode encountering a single token will return that token.");

        // Repetition options.
        let mut repetition_enabled = self.repetition.is_some();
        resp |= ui
            .checkbox(&mut repetition_enabled, "Repetition Penalties")
            .on_hover_text_at_pointer(
                "Apply penalties to reduce repetition in the output.",
            );

        if repetition_enabled {
            let repetition =
                self.repetition.get_or_insert(RepetitionOptions::default());
            resp |= repetition.draw(ui);
        } else {
            self.repetition = None;
        }

        resp
    }

    /// Draw [`egui::Ui`] for [`SampleOptions`]. This lets the user add or remove
    /// sampling modes, ensuring there is always at least one. It also allows
    /// the user to set repetition options.
    #[cfg(feature = "egui")]
    pub fn draw(&mut self, ui: &mut egui::Ui) -> egui::Response {
        // FIXME: This nesting is verging on illegible. This function should be
        // split up or lambdas should be used to reduce nesting.

        let resp = egui::CollapsingHeader::new("Sampling Options")
            .show(ui, |ui| self.draw_inner(ui));

        let header_response = resp
            .header_response
            .on_hover_text_at_pointer("Options for sampling tokens.");

        resp.body_response.unwrap_or(header_response)
    }
}

impl Default for SampleOptions {
    fn default() -> Self {
        Self {
            modes: vec![SamplingMode::locally_typical()],
            repetition: RepetitionOptions::default().into(),
        }
    }
}

#[cfg_attr(
    feature = "serde",
    derive(rocket::serde::Deserialize, rocket::serde::Serialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "rocket::serde"))]
#[derive(Clone, Debug)]
// TODO: add `min_keep` and `mad_keep` to all the sampling modes since it's
// doable and it would be nice to have a more consistent API.
pub enum SamplingMode {
    /// Greedy sampling. The most likely next token is always chosen. Not very
    /// useful unless you want to regurgitate the training data.
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
    TopK {
        /// The top `k` tokens are kept. Reasonable values are between 30 and
        /// 40.
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
    /// <https://www.trentonbricken.com/Tail-Free-Sampling/>
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
    /// JSON-constrained sampling. Rejects any candidate whose bytes would
    /// produce invalid JSON.
    ///
    /// # Termination
    ///
    /// On zero valid candidates, this mode forces a single EOS candidate.
    /// Two cases trigger this:
    ///
    /// * **Success** — the document has been closed; the strict post-complete
    ///   rule rejects all further bytes (including whitespace, so the model
    ///   can't burn its token budget on trailing ws). The parser is
    ///   **auto-reset** before returning EOS, so the next generation on the
    ///   same mode instance starts fresh.
    /// * **Grammar violation** — the parser is mid-parse but no candidate
    ///   token extends it. State is preserved for inspection.
    ///
    /// **For generation to actually stop**, EOS must be in
    /// [`PredictOptions::stop_sequences`]. Use
    /// [`PredictOptions::add_model_stops`] (or add EOS explicitly) —
    /// otherwise the predictor will keep sampling past EOS and the
    /// auto-reset will let the model emit another JSON document.
    ///
    /// # State sharing
    ///
    /// [`JsonState`] is wrapped in `Arc<Mutex<…>>` so cloning
    /// `SampleOptions` keeps the parser state shared across clones (one
    /// parser per generation). To start a fresh generation manually, either
    /// rebuild the mode or call [`JsonState::reset`] through the mutex. The
    /// auto-reset on success-termination covers the common case.
    ///
    /// # Placement
    ///
    /// Place early in the chain: it prunes the candidate set before top-p /
    /// top-k / mirostat run over it. Leading whitespace before the opening
    /// `{`/`[`/etc. is accepted (models often emit a newline after the
    /// prompt's trailing colon); trailing whitespace after the document
    /// closes is rejected (see Termination above).
    ///
    /// # Serde
    ///
    /// Under `serde`, the parse state is serialized — reloading
    /// mid-generation will resume with the parser aligned to the
    /// already-emitted tokens.
    ///
    /// [`PredictOptions::stop_sequences`]: crate::PredictOptions::stop_sequences
    /// [`PredictOptions::add_model_stops`]: crate::PredictOptions::add_model_stops
    Json(
        #[cfg_attr(
            feature = "serde",
            serde(
                serialize_with = "serialize_json_state",
                deserialize_with = "deserialize_json_state"
            )
        )]
        Arc<Mutex<JsonState>>,
    ),
    /// GBNF-constrained sampling. Rejects any candidate whose bytes would
    /// violate the grammar.
    ///
    /// # Termination
    ///
    /// Same as [`SamplingMode::Json`]: on zero valid candidates the filter
    /// forces a single EOS candidate. Two cases trigger this:
    ///
    /// * **Success** — the grammar has reached an accept state; all further
    ///   tokens are rejected. The matcher is **auto-reset** before
    ///   returning EOS, so the next generation on the same mode instance
    ///   starts fresh.
    /// * **Grammar violation** — the matcher is mid-parse but no candidate
    ///   token extends it. State is preserved for inspection.
    ///
    /// **For generation to actually stop**, EOS must be in
    /// [`PredictOptions::stop_sequences`]. Use
    /// [`PredictOptions::add_model_stops`] — otherwise the auto-reset will
    /// let the model emit another document after EOS.
    ///
    /// # Construction
    ///
    /// Use [`SamplingMode::grammar`] to parse GBNF source, or
    /// [`SamplingMode::grammar_from_file`] to load a `.gbnf` file.
    ///
    /// # Serde
    ///
    /// Under `serde`, only the GBNF source text is serialized. On
    /// deserialize the source is re-parsed and the matcher starts fresh —
    /// mid-parse position is NOT preserved across reload.
    ///
    /// [`PredictOptions::stop_sequences`]: crate::PredictOptions::stop_sequences
    /// [`PredictOptions::add_model_stops`]: crate::PredictOptions::add_model_stops
    Grammar(
        #[cfg_attr(
            feature = "serde",
            serde(
                serialize_with = "serialize_grammar",
                deserialize_with = "deserialize_grammar"
            )
        )]
        Arc<Mutex<GrammarState>>,
    ),
}

impl SamplingMode {
    // TODO: Figure out a way to statically assert that the length of this list
    // is equal to the number of variants in SamplingMode.
    pub const ALL: [Self; 10] = [
        Self::Greedy,
        Self::top_p(),
        Self::top_k(),
        Self::min_p(),
        Self::tail_free(),
        Self::locally_typical(),
        Self::mirostat(),
        Self::mirostat_v2(),
        Self::split_p(),
        Self::split_l(),
    ];

    /// Construct a fresh JSON-constrained sampling mode at the root of a
    /// document.
    pub fn json() -> Self {
        Self::Json(Arc::new(Mutex::new(JsonState::new())))
    }

    /// Construct a GBNF-constrained sampling mode from a GBNF source
    /// string. Returns the parse error if the grammar is malformed.
    pub fn grammar(source: &str) -> Result<Self, GrammarError> {
        let grammar = Arc::new(Grammar::parse(source)?);
        Ok(Self::Grammar(Arc::new(Mutex::new(GrammarState::new(
            grammar,
        )))))
    }

    /// Construct a GBNF-constrained sampling mode by loading a `.gbnf`
    /// file from disk. Returns an I/O or parse error on failure.
    pub fn grammar_from_file(
        path: impl AsRef<std::path::Path>,
    ) -> Result<Self, GrammarError> {
        let grammar = Arc::new(Grammar::from_file(path)?);
        Ok(Self::Grammar(Arc::new(Mutex::new(GrammarState::new(
            grammar,
        )))))
    }

    /// The name of the sampling mode.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Greedy => "Greedy",
            Self::TopP { .. } => "Top-P",
            Self::TopK { .. } => "Top-K",
            Self::MinP { .. } => "Min-P",
            Self::TailFree { .. } => "Tail Free",
            Self::LocallyTypical { .. } => "Locally Typical",
            Self::Mirostat { .. } => "Mirostat",
            Self::MirostatV2 { .. } => "Mirostat V2",
            Self::SplitP { .. } => "Split P",
            Self::SplitL { .. } => "Split L",
            Self::Json(_) => "JSON",
            Self::Grammar(_) => "Grammar",
        }
    }

    /// A help message for the sampling mode (but not it's parameters)
    pub fn help(&self) -> &'static str {
        match self {
            Self::Greedy => "The most likely next token is always chosen. Not very useful unless you want to regurgitate the training data.",
            Self::TopP { .. } => "A token is chosen from the top tokens whose cumulative probability is greater than or equal to `p`.",
            Self::TopK { .. } => "A token is chosen from the top `k` tokens. This is not very good.",
            Self::MinP { .. } => "Min-p sampling. `p` sets the minimum probability to keep a token. Below that the tail is cut off. `p` is scaled by the top token's probability to balance diversity and quality.",
            Self::TailFree { .. } => "Tail free sampling. Described here: https://www.trentonbricken.com/Tail-Free-Sampling/",
            Self::LocallyTypical { .. } => "Locally typical sampling is one of the best sampling methods. described here: https://arxiv.org/pdf/2202.00666.pdf",
            Self::Mirostat { .. } => "Mirostat sampling. Described here: https://arxiv.org/pdf/2007.14966.pdf",
            Self::MirostatV2 { .. } => "Mirostat v2 sampling. Described here: https://arxiv.org/pdf/2007.14966.pdf",
            Self::SplitP { .. } => "Cuts off the tail where the difference between adjacent probabilities is greatest.",
            Self::SplitL { .. } => "Cuts off the tail where the difference between adjacent logits is greatest.",
            Self::Json(_) => "Constrains generation to valid JSON. Place early in the chain so it prunes candidates before top-p/top-k. On grammar violation, forces EOS and terminates.",
            Self::Grammar(_) => "Constrains generation to a GBNF grammar. Place early in the chain so it prunes candidates before top-p/top-k. On grammar violation, forces EOS and terminates.",
        }
    }

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

    /// Draw [`egui::Ui`], but without the outer collapsible header.
    #[cfg(feature = "egui")]
    pub fn draw_inner(&mut self, ui: &mut egui::Ui) -> egui::Response {
        const MIN_KEEP_MIN: usize = 1;
        const MIN_KEEP_MAX: usize = 4096;

        // Helper function to draw min/max_keep DragValue.
        let keep_helper = |name: &str,
                           min_keep: &mut NonZeroUsize,
                           ui: &mut egui::Ui| {
            let mut val = min_keep.get();
            let resp = ui.horizontal(|ui| {
                ui.label(name) |
                ui.add(egui::DragValue::new(&mut val).clamp_range(MIN_KEEP_MIN..=MIN_KEEP_MAX))
                    .on_hover_text_at_pointer("Min/Max number of candidates to keep per token. Useful in combination with other sampling modes.")
            }).inner;
            *min_keep =
                NonZeroUsize::new(val.clamp(MIN_KEEP_MIN, MIN_KEEP_MAX))
                    .unwrap();
            resp
        };

        match self {
            // This is a big verbose, but we're trying to make sure a response
            // is available for each branch. This way the caller can use it to
            // detect clicks, changes, etc.
            Self::Greedy => ui.separator(),
            Self::TopP { p, min_keep } => {
                let mut inner = ui.add(egui::Slider::new(&mut p.p, 0.0..=1.0).text("P"))
                    .on_hover_text_at_pointer("0.9-0.95 is a good range for creative uses. Higher is more diverse, but potentially less coherent. 0.0 will give the same result as greedy.");
                p.p = p.p.clamp(0.0, 1.0);
                inner |= keep_helper("Min Keep", min_keep, ui);
                inner
            }
            Self::TopK { k } => {
                let mut val = k.get();
                let inner = ui
                    .add(
                        egui::Slider::new(
                            &mut val,
                            MIN_KEEP_MIN..=MIN_KEEP_MAX,
                        )
                        .text("K"),
                    )
                    .on_hover_text_at_pointer(
                        "Reasonable values are between 30 and 40.",
                    );
                *k = NonZeroUsize::new(val.clamp(MIN_KEEP_MIN, MIN_KEEP_MAX))
                    .unwrap();
                inner
            }
            Self::MinP { p, min_keep } => {
                let inner = ui.add(
                            egui::Slider::new(&mut p.p, 0.0..=1.0).text("P"),
                        ).on_hover_text_at_pointer("Reasonable values are 0.05 to 0.3. Higher means less diversity.")
                        | keep_helper("Min Keep", min_keep, ui);
                p.p = p.p.clamp(0.0, 1.0);

                inner
            }
            Self::TailFree { z, min_keep } => {
                let inner = ui.add(egui::Slider::new(&mut z.p, 0.0..=1.0).text("Z")).on_hover_text_at_pointer("Reasonable values are between 0.25 and 0.75. Higher is more diverse but potentially less coherent.")
                    | keep_helper("Min Keep", min_keep, ui);
                z.p = z.p.clamp(0.0, 1.0);

                inner
            }
            Self::LocallyTypical { p, min_keep } => {
                let inner = ui.add(egui::Slider::new(&mut p.p, 0.0..=1.0).text("P")).on_hover_text_at_pointer("Reasonable values are between 0.2 and 0.95. For story generation, lower is better. For summarization, higher is better.")
                    | keep_helper("Min Keep", min_keep, ui);
                p.p = p.p.clamp(0.0, 1.0);

                inner
            }
            Self::Mirostat { tau, eta, max_keep }
            | Self::MirostatV2 { tau, eta, max_keep } => {
                let mut max_keep_enabled = max_keep.is_some();
                let inner = ui.add(egui::Slider::new(tau, 0.0..=10.0).text("Tau")).on_hover_text_at_pointer("Target entropy. A good value is 3.0 according to this paper: https://arxiv.org/pdf/2202.00666.pdf")
                    | ui.add(egui::Slider::new(eta, 0.0..=1.0).text("Eta")).on_hover_text_at_pointer("Learning rate. A good value is 0.1.")
                    | ui.checkbox(&mut max_keep_enabled, "Limit max candidates.").on_hover_text_at_pointer("If unset, the maximum number of candidates to keep is 100.");

                if max_keep_enabled {
                    let max_keep =
                        max_keep.get_or_insert(NonZeroUsize::new(50).unwrap());
                    keep_helper("Max Keep", max_keep, ui);
                } else {
                    *max_keep = None;
                }

                inner
            }
            Self::SplitP { min_keep, max_keep } => {
                let mut max_keep_enabled = max_keep.is_some();
                let inner = keep_helper("Min Keep", min_keep, ui)
                    | ui.checkbox(&mut max_keep_enabled, "Limit max candidates.").on_hover_text_at_pointer("If unset, the maximum number of candidates to keep is 50.");

                if max_keep_enabled {
                    let max_keep =
                        max_keep.get_or_insert(NonZeroUsize::new(50).unwrap());
                    keep_helper("Max Keep", max_keep, ui);
                } else {
                    *max_keep = None;
                }

                inner
            }
            Self::SplitL { min_keep, max_keep } => {
                let mut max_keep_enabled = max_keep.is_some();
                let inner = keep_helper("Min Keep", min_keep, ui)
                    | ui.checkbox(&mut max_keep_enabled, "Limit max candidates.").on_hover_text_at_pointer("If unset, the maximum number of candidates to keep is 50.");

                if max_keep_enabled {
                    let max_keep =
                        max_keep.get_or_insert(NonZeroUsize::new(50).unwrap());
                    keep_helper("Max Keep", max_keep, ui);
                } else {
                    *max_keep = None;
                }

                inner
            }
            Self::Json(state) => {
                let (complete, depth) = match state.lock() {
                    Ok(s) => (s.is_complete(), s.stack_depth()),
                    Err(_) => (false, 0),
                };
                let status = if complete {
                    "JSON: complete".to_string()
                } else {
                    format!("JSON: in progress (depth {depth})")
                };
                let resp = ui.label(status).on_hover_text_at_pointer(
                    "JSON grammar constraint. Filters candidates to those that keep output valid JSON. On violation, forces EOS.",
                );
                if ui
                    .button("Reset parser")
                    .on_hover_text_at_pointer(
                        "Restart the parser at the root of a new document.",
                    )
                    .clicked()
                {
                    if let Ok(mut s) = state.lock() {
                        s.reset();
                    }
                }
                resp
            }
            Self::Grammar(state) => {
                let (complete, depth, rule_count) = match state.lock() {
                    Ok(s) => (
                        s.is_complete(),
                        s.stack_depth(),
                        s.grammar().rule_count(),
                    ),
                    Err(_) => (false, 0, 0),
                };
                let status = if complete {
                    format!("Grammar: complete ({rule_count} rules)")
                } else {
                    format!(
                        "Grammar: in progress ({depth} active, {rule_count} rules)"
                    )
                };
                let resp = ui.label(status).on_hover_text_at_pointer(
                    "GBNF grammar constraint. Filters candidates to those that extend the grammar. On violation, forces EOS.",
                );
                if ui
                    .button("Reset matcher")
                    .on_hover_text_at_pointer(
                        "Restart the matcher at the root rule.",
                    )
                    .clicked()
                {
                    if let Ok(mut s) = state.lock() {
                        s.reset();
                    }
                }
                resp
            }
        }
    }

    /// Draw [`egui::Ui`] for sampling mode.
    ///
    /// The index is used to generate a unique id for the collapsible header.
    #[cfg(feature = "egui")]
    pub fn draw(&mut self, ui: &mut egui::Ui, index: usize) -> egui::Response {
        let resp = egui::CollapsingHeader::new(self.name())
            // We need an id because it's possible (but likely pointless) to
            // have two identical sampling modes in the list.
            .id_source((index, self.name()))
            .show(ui, |ui| self.draw_inner(ui));

        let header_resp =
            resp.header_response.on_hover_text_at_pointer(self.help());
        resp.body_response.unwrap_or(header_resp)
    }
}

impl Default for SamplingMode {
    fn default() -> Self {
        Self::locally_typical()
    }
}

impl PartialEq for SamplingMode {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Greedy, Self::Greedy) => true,
            (
                Self::TopP {
                    p: p1,
                    min_keep: k1,
                },
                Self::TopP {
                    p: p2,
                    min_keep: k2,
                },
            ) => p1 == p2 && k1 == k2,
            (Self::TopK { k: k1 }, Self::TopK { k: k2 }) => k1 == k2,
            (
                Self::MinP {
                    p: p1,
                    min_keep: k1,
                },
                Self::MinP {
                    p: p2,
                    min_keep: k2,
                },
            ) => p1 == p2 && k1 == k2,
            (
                Self::TailFree {
                    z: z1,
                    min_keep: k1,
                },
                Self::TailFree {
                    z: z2,
                    min_keep: k2,
                },
            ) => z1 == z2 && k1 == k2,
            (
                Self::LocallyTypical {
                    p: p1,
                    min_keep: k1,
                },
                Self::LocallyTypical {
                    p: p2,
                    min_keep: k2,
                },
            ) => p1 == p2 && k1 == k2,
            (
                Self::Mirostat {
                    tau: t1,
                    eta: e1,
                    max_keep: m1,
                },
                Self::Mirostat {
                    tau: t2,
                    eta: e2,
                    max_keep: m2,
                },
            ) => t1 == t2 && e1 == e2 && m1 == m2,
            (
                Self::MirostatV2 {
                    tau: t1,
                    eta: e1,
                    max_keep: m1,
                },
                Self::MirostatV2 {
                    tau: t2,
                    eta: e2,
                    max_keep: m2,
                },
            ) => t1 == t2 && e1 == e2 && m1 == m2,
            (
                Self::SplitP {
                    min_keep: a1,
                    max_keep: b1,
                },
                Self::SplitP {
                    min_keep: a2,
                    max_keep: b2,
                },
            ) => a1 == a2 && b1 == b2,
            (
                Self::SplitL {
                    min_keep: a1,
                    max_keep: b1,
                },
                Self::SplitL {
                    min_keep: a2,
                    max_keep: b2,
                },
            ) => a1 == a2 && b1 == b2,
            // Two `Json` variants compare equal if they share the same Arc
            // (identity) OR if their locked states are structurally equal. A
            // poisoned mutex compares unequal — we can't safely inspect it.
            (Self::Json(a), Self::Json(b)) => {
                if Arc::ptr_eq(a, b) {
                    return true;
                }
                match (a.lock(), b.lock()) {
                    (Ok(a), Ok(b)) => *a == *b,
                    _ => false,
                }
            }
            (Self::Grammar(a), Self::Grammar(b)) => {
                if Arc::ptr_eq(a, b) {
                    return true;
                }
                match (a.lock(), b.lock()) {
                    (Ok(a), Ok(b)) => *a == *b,
                    _ => false,
                }
            }
            _ => false,
        }
    }
}

#[derive(Debug, thiserror::Error, derive_more::From)]
pub enum SampleError {
    #[error("Sampling failed because of a repetition error: {err}")]
    RepetitionError { err: RepetitionError },
}

static_assertions::assert_impl_all!(SampleError: Send, Sync);

/// Sample a token from the candidates.
pub(crate) fn sample_token(
    tokens: &[llama_token],
    mut candidates: Candidates,
    vocab: &Vocab,
    opts: &mut SampleOptions,
    freq_map: &mut NGramStats,
    rng: &mut xorshift::Xoroshiro128,
    mu: &mut Option<f32>,
    model: &crate::Model,
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
    if let Some(repetition) = &mut opts.repetition {
        candidates = apply_sample_repetition_ngram(
            candidates, tokens, repetition, freq_map, model,
        )?;
    }

    // Fold candidates, applying the sampling modes in order.
    //
    // `Arc::clone` on `SamplingMode::Json` is cheap and preserves shared-state
    // semantics across the fold — the cloned Arc still points at the same
    // Mutex that the advance-step below will update.
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
                SamplingMode::Json(state) => {
                    // A poisoned mutex here means a prior panic occurred
                    // while advancing the parser — the state is unknown,
                    // so silently emitting non-JSON would violate the
                    // `SamplingMode::Json` contract. Fail fast instead.
                    let mut locked = state.lock().expect(
                        "SamplingMode::Json mutex poisoned; parser state \
                         is unrecoverable. Rebuild the mode with \
                         SamplingMode::json() and retry.",
                    );
                    json::json_filter(candidates, &mut locked, model)
                }
                SamplingMode::Grammar(state) => {
                    let mut locked = state.lock().expect(
                        "SamplingMode::Grammar mutex poisoned; matcher \
                         state is unrecoverable. Rebuild the mode with \
                         SamplingMode::grammar(...) and retry.",
                    );
                    grammar::grammar_filter(candidates, &mut locked, model)
                }
            });

    let chosen = choose_candidate(rng, filtered.softmax(None))
        .is_one()
        .unwrap()
        .id;

    // Post-selection: advance every JSON and grammar parser in the chain
    // by the chosen token. Multiple grammar constraints in the chain all
    // observe the same chosen token and advance in lockstep.
    json::advance_all(&opts.modes, chosen, model);
    grammar::advance_all(&opts.modes, chosen, model);

    Ok(chosen)
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
