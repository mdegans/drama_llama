use crate::{ngram::NGramStats, Candidates, Probability, Token};

use rand::RngExt as _;

use std::num::NonZeroUsize;

pub(crate) mod grammar;
mod json;
mod repetition;
pub(crate) mod state;

pub use grammar::{
    grammar_stats_enabled, grammar_stats_reset, grammar_stats_snapshot,
    CompiledGrammar, Grammar, GrammarError, GrammarState, GrammarStats,
};
pub use json::{JsonError, JsonState};
pub use repetition::{
    apply_sample_repetition_ngram, RepetitionError, RepetitionOptions,
};
pub use state::SamplerState;

#[cfg(feature = "egui")]
pub(crate) const DELETE_ICON: egui::ImageSource<'static> =
    egui::include_image!("../assets/ui/images/delete.png");

#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
/// Options determining how raw logits are turned into a token. This is used by
/// [`Candidates::sample_token`] and associated functions.
#[derive(Clone, Debug, PartialEq)]
pub struct SamplerConfig {
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
    /// Optional grammar that stays dormant until a byte sequence appears in
    /// the predictor's output, then activates (its matcher lives in
    /// [`SamplerState`], flagged active on trigger). Used to skip
    /// grammar filtering during free-form preambles like `<think>…</think>`
    /// and only activate it for the structured portion that follows.
    /// See [`DeferredGrammar`].
    #[cfg_attr(feature = "serde", serde(default))]
    pub deferred_grammar: Option<DeferredGrammar>,
    /// Sample-then-check for `Grammar`/`Json` modes: sample *without* the
    /// grammar filters, then verify just the chosen token's piece with
    /// `accepts_bytes` (O(piece) instead of O(vocab) per token). If the
    /// piece is illegal, fall back to the full masked path over the
    /// pre-fold candidates — so output is always grammar-legal.
    ///
    /// **Semantics differ from masked-first by design** (this is the same
    /// accept-if-legal shape llama.cpp ships): the unconstrained winner
    /// keeps its seat if legal; masked-first instead removes illegal mass
    /// *before* truncation samplers run. Streams from the two modes
    /// diverge, deliberately. Both are fully deterministic: fixed seed →
    /// identical stream, and exactly one RNG draw is consumed per emitted
    /// token on either path. `false` keeps the masked path (which *is*
    /// the fallback implementation), modulo one deliberate post-v0.8.0
    /// fix in both modes: completed constraints no longer keep
    /// empty-piece tokens, so generation terminates at document end
    /// instead of looping on invisible reserved tokens (see
    /// `json_filter`).
    ///
    /// **Default `true`** (#28 step 3, flipped 2026-07-16): the masked
    /// path burns a full-vocab rayon sweep per constrained token even
    /// with the DFA cache; sample-then-check is O(piece) in the
    /// steady state with identical grammar-legality guarantees. Greedy
    /// streams are unchanged by the flip (the fallback replay *is* the
    /// masked path, so an illegal argmax converges to the same pick);
    /// sampled streams under constraints differ, deliberately.
    ///
    /// Has no effect when `modes` contains no `Grammar`/`Json` mode.
    #[cfg_attr(feature = "serde", serde(default = "default_lazy_grammar"))]
    pub lazy_grammar: bool,
    /// Emit-side special-token ban (sorted ids): specials the active
    /// chat dialect never legitimately emits — turn-open markers,
    /// reserved-vocab controls — everything except EOG tokens and the
    /// dialect's own in-stream markers. Checked on the SAMPLED token
    /// (O(log n) per token, the accept-then-mask shape) with a full
    /// masked resample only on a hit, so free prose can't smuggle
    /// chat-framing tokens into the transcript (the emission-side
    /// sibling of `Session`'s ingest injection guard). The banned
    /// token's byte *text* stays expressible through ordinary
    /// tokenization — this bans the control token id, not the
    /// characters. Empty (the default) disables the check. Runtime
    /// wiring set by `Session` per call from the model's specials ×
    /// the analyzed dialect.
    #[cfg_attr(feature = "serde", serde(default))]
    pub banned_specials: Vec<Token>,
}

/// A grammar that starts suspended and activates once a specific byte
/// sequence appears in the predictor's generated text. Activation is driven
/// by `TokenPredictor`: when the trigger is found in the accumulated output,
/// the matcher in [`SamplerState`] is flagged active and any bytes emitted
/// after the trigger are fed into it so it lines up with the model's byte
/// position.
///
/// Typical use: reactor / structured-output workloads where the model emits
/// `<think>…</think>` then JSON. The JSON grammar is the deferred one;
/// `activate_after` is `b"</think>"`. During thought the grammar filter is
/// skipped entirely, restoring pure-inference tok/s. See
/// `src/output_config.rs` for the compiler that builds one from an
/// `OutputConfig`.
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct DeferredGrammar {
    /// The grammar to promote. Pure spec — the matcher that starts
    /// walking it on activation lives in [`SamplerState`], created at
    /// `init_state` and flagged active by the predictor when a trigger
    /// fires.
    pub grammar: CompiledGrammar,
    /// Byte sequences (any-of) whose appearance in the predictor's
    /// accumulated text triggers promotion. Matched anywhere in the
    /// trailing window (same sizing as stop-strings), not just at the
    /// exact text end. Most dialects have exactly one; Harmony's
    /// tool-call header has several legal shapes
    /// ([`CallSyntax::triggers`](crate::CallSyntax::triggers)).
    pub activate_after: Vec<Vec<u8>>,
    /// Whether the trigger bytes themselves are fed into the grammar
    /// state at promotion (llama.cpp lazy-pattern semantics: the
    /// grammar root *starts with* the trigger, e.g. a `<tool_call>\n`
    /// wrap-open). `false` keeps the original behavior — only bytes
    /// *after* the trigger feed in (e.g. `</think>` triggering a
    /// JSON-body grammar).
    pub feed_trigger: bool,
}

/// The `lazy_grammar` default — `true` — shared by `Default`,
/// `greedy()`, and serde (a config that omits the field gets the same
/// answer as a constructed one).
const fn default_lazy_grammar() -> bool {
    true
}

impl SamplerConfig {
    /// Greedy sampling. No repetition penalty.
    pub fn greedy() -> Self {
        Self {
            modes: vec![SamplingMode::Greedy],
            repetition: None,
            deferred_grammar: None,
            lazy_grammar: default_lazy_grammar(),
            banned_specials: Vec::new(),
        }
    }

    /// Draw [`egui::Ui`] for [`SamplerConfig`] but without the outer
    /// [`egui::CollapsingHeader`].
    #[cfg(feature = "egui")]
    pub fn draw_inner(&mut self, ui: &mut egui::Ui) -> egui::Response {
        let collaping_resp = egui::CollapsingHeader::new("Sampling Modes")
            .show(ui, |ui| {
                // TODO: The user should be able to drag and drop these to reorder them.
                // Otherwise they must remove and re-add them in the desired order.
                // TODO: Test combinations of modes since I am fairly sure some make the
                // assumption all candidates are present and may crash if they are not.
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

    /// Draw [`egui::Ui`] for [`SamplerConfig`]. This lets the user add or remove
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

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            modes: vec![SamplingMode::locally_typical()],
            // On by default as of v0.8.0. The long-form degradation that
            // originally forced this off (qwen3 long-form arc) was the
            // unbounded additive `count * penalty_freq` term; the windowed
            // decay (see `RepetitionOptions::decay`) bounds it, and the
            // retuned + surgical defaults were validated to leave big-model
            // prose intact across technical / creative / social genres.
            // NOTE: chat/tool-result flows that re-emit a short token from
            // context (e.g. a digit the tool returned) now see a gentle
            // penalty; if that proves a problem, opt back out via a sidecar
            // or `SamplerConfig::greedy()`.
            repetition: Some(RepetitionOptions::default()),
            deferred_grammar: None,
            lazy_grammar: default_lazy_grammar(),
            banned_specials: Vec::new(),
        }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Debug, PartialEq)]
// TODO: add `min_keep` and `mad_keep` to all the sampling modes since it's
// doable and it would be nice to have a more consistent API.
pub enum SamplingMode {
    /// Greedy sampling. The most likely next token is always chosen. Not very
    /// useful unless you want to regurgitate the training data.
    Greedy,
    /// Temperature scaling. Divides every logit by `t`, flattening
    /// (`t > 1.0`) or sharpening (`t < 1.0`) the distribution before
    /// whatever follows in the chain. Composable: place before a
    /// truncation mode (top-p/top-k) in the conventional order.
    Temperature {
        /// `t <= 0.0` collapses to greedy (argmax), matching `llama.cpp`.
        /// `1.0` is a no-op. Reasonable values are 0.2 (focused) to 1.5
        /// (diverse).
        t: f32,
    },
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
    ///   can't burn its token budget on trailing ws).
    /// * **Grammar violation** — the parser is mid-parse but no candidate
    ///   token extends it.
    ///
    /// **For generation to actually stop**, EOS must be in
    /// [`PredictOptions::stop_sequences`]. Use
    /// [`PredictOptions::add_model_stops`] (or add EOS explicitly).
    ///
    /// # State
    ///
    /// This variant is pure config — the JSON grammar is built in, so it
    /// carries nothing. The parser position ([`JsonState`]) lives in
    /// [`SamplerState`], fresh per call via `SamplerConfig::init_state`.
    ///
    /// # Placement
    ///
    /// Place early in the chain: it prunes the candidate set before top-p /
    /// top-k / mirostat run over it. Leading whitespace before the opening
    /// `{`/`[`/etc. is accepted (models often emit a newline after the
    /// prompt's trailing colon); trailing whitespace after the document
    /// closes is rejected (see Termination above).
    ///
    /// [`PredictOptions::stop_sequences`]: crate::PredictOptions::stop_sequences
    /// [`PredictOptions::add_model_stops`]: crate::PredictOptions::add_model_stops
    Json,
    /// GBNF-constrained sampling. Rejects any candidate whose bytes would
    /// violate the grammar.
    ///
    /// # Termination
    ///
    /// Same as [`SamplingMode::Json`]: on zero valid candidates the filter
    /// forces a single EOS candidate. Two cases trigger this:
    ///
    /// * **Success** — the grammar has reached an accept state; all further
    ///   tokens are rejected.
    /// * **Grammar violation** — the matcher is mid-parse but no candidate
    ///   token extends it.
    ///
    /// **For generation to actually stop**, EOS must be in
    /// [`PredictOptions::stop_sequences`]. Use
    /// [`PredictOptions::add_model_stops`].
    ///
    /// # Construction
    ///
    /// Use [`SamplingMode::grammar`] to parse GBNF source, or
    /// [`SamplingMode::grammar_from_file`] to load a `.gbnf` file.
    ///
    /// # State
    ///
    /// [`CompiledGrammar`] is pure config: compiled rules + the shared
    /// lazy-DFA cache. The matcher position lives in [`SamplerState`],
    /// fresh per call via `SamplerConfig::init_state`.
    ///
    /// # Serde
    ///
    /// Only the GBNF source text is serialized; deserialization
    /// re-parses it (deterministic compile, so matcher positions
    /// serialized alongside in `SamplerState` stay index-consistent).
    ///
    /// [`PredictOptions::stop_sequences`]: crate::PredictOptions::stop_sequences
    /// [`PredictOptions::add_model_stops`]: crate::PredictOptions::add_model_stops
    Grammar(CompiledGrammar),
    /// Forbid every token id in `range` from sampling. Logits for
    /// matching candidates are dropped before downstream modes run.
    ///
    /// Primary use case: tokenizer reserved/unused vocab tail
    /// (Qwen3.5: 248000..248320). Such slots decode to empty
    /// strings — they contribute zero bytes to a grammar's byte
    /// stream and so are trivially accepted by GBNF / JSON
    /// constraints, even when the structured response has already
    /// completed. Without this mask, the model can land in a loop
    /// scattering reserved tokens until `max_tokens` runs out (the
    /// grammar's "all-candidates-rejected → force EOS" fallback
    /// never fires because empty-piece tokens keep the candidate
    /// set non-empty). With the mask, the grammar sees a candidate
    /// set without empty-piece tokens, rejects every byte-emitting
    /// candidate after the structure closes, and the EOS fallback
    /// triggers naturally.
    ///
    /// # Placement
    ///
    /// Place at the **start** of the chain so the deny applies
    /// before grammar, top-p, etc. reason about the candidate set.
    /// `Session` prepends a Deny mode from the model's reserved
    /// vocab range automatically; callers building chains by hand
    /// should mirror that.
    Deny {
        /// Half-open range `[start, end)` of token ids to forbid.
        range: std::ops::Range<Token>,
    },
}

impl SamplingMode {
    // TODO: Figure out a way to statically assert that the length of this list
    // is equal to the number of variants in SamplingMode.
    pub const ALL: [Self; 11] = [
        Self::Greedy,
        Self::temperature(),
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

    /// Construct a [`SamplingMode::Deny`] from a token-id range.
    /// Convenience wrapper so callers don't need to spell out the
    /// struct field syntax.
    pub fn deny_range(range: std::ops::Range<Token>) -> Self {
        Self::Deny { range }
    }

    /// Construct a fresh JSON-constrained sampling mode at the root of a
    /// document.
    pub fn json() -> Self {
        Self::Json
    }

    /// Construct a GBNF-constrained sampling mode from a GBNF source
    /// string. Returns the parse error if the grammar is malformed.
    pub fn grammar(source: &str) -> Result<Self, GrammarError> {
        Ok(Self::Grammar(CompiledGrammar::parse(source)?))
    }

    /// Construct a GBNF-constrained sampling mode by loading a `.gbnf`
    /// file from disk. Returns an I/O or parse error on failure.
    pub fn grammar_from_file(
        path: impl AsRef<std::path::Path>,
    ) -> Result<Self, GrammarError> {
        Ok(Self::Grammar(CompiledGrammar::from_file(path)?))
    }

    /// The name of the sampling mode.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Greedy => "Greedy",
            Self::Temperature { .. } => "Temperature",
            Self::TopP { .. } => "Top-P",
            Self::TopK { .. } => "Top-K",
            Self::MinP { .. } => "Min-P",
            Self::TailFree { .. } => "Tail Free",
            Self::LocallyTypical { .. } => "Locally Typical",
            Self::Mirostat { .. } => "Mirostat",
            Self::MirostatV2 { .. } => "Mirostat V2",
            Self::SplitP { .. } => "Split P",
            Self::SplitL { .. } => "Split L",
            Self::Json => "JSON",
            Self::Grammar(_) => "Grammar",
            Self::Deny { .. } => "Deny",
        }
    }

    /// A help message for the sampling mode (but not it's parameters)
    pub fn help(&self) -> &'static str {
        match self {
            Self::Greedy => "The most likely next token is always chosen. Not very useful unless you want to regurgitate the training data.",
            Self::Temperature { .. } => "Divides every logit by `t`, flattening (t > 1.0) or sharpening (t < 1.0) the distribution. t <= 0.0 collapses to greedy.",
            Self::TopP { .. } => "A token is chosen from the top tokens whose cumulative probability is greater than or equal to `p`.",
            Self::TopK { .. } => "A token is chosen from the top `k` tokens. This is not very good.",
            Self::MinP { .. } => "Min-p sampling. `p` sets the minimum probability to keep a token. Below that the tail is cut off. `p` is scaled by the top token's probability to balance diversity and quality.",
            Self::TailFree { .. } => "Tail free sampling. Described here: https://www.trentonbricken.com/Tail-Free-Sampling/",
            Self::LocallyTypical { .. } => "Locally typical sampling is one of the best sampling methods. described here: https://arxiv.org/pdf/2202.00666.pdf",
            Self::Mirostat { .. } => "Mirostat sampling. Described here: https://arxiv.org/pdf/2007.14966.pdf",
            Self::MirostatV2 { .. } => "Mirostat v2 sampling. Described here: https://arxiv.org/pdf/2007.14966.pdf",
            Self::SplitP { .. } => "Cuts off the tail where the difference between adjacent probabilities is greatest.",
            Self::SplitL { .. } => "Cuts off the tail where the difference between adjacent logits is greatest.",
            Self::Json => "Constrains generation to valid JSON. Place early in the chain so it prunes candidates before top-p/top-k. On grammar violation, forces EOS and terminates.",
            Self::Grammar(_) => "Constrains generation to a GBNF grammar. Place early in the chain so it prunes candidates before top-p/top-k. On grammar violation, forces EOS and terminates.",
            Self::Deny { .. } => "Forbids every candidate whose token id falls in `range`. Place at the start of the chain — typically used for the model's reserved/empty-piece vocab tail.",
        }
    }

    /// Default temperature scaling: t = 0.8 (the `llama.cpp` default).
    pub const fn temperature() -> Self {
        Self::Temperature { t: 0.8 }
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
                ui.add(egui::DragValue::new(&mut val).range(MIN_KEEP_MIN..=MIN_KEEP_MAX))
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
            Self::Temperature { t } => {
                let inner = ui
                    .add(egui::Slider::new(t, 0.0..=2.0).text("T"))
                    .on_hover_text_at_pointer(
                        "Higher is more diverse, lower is more focused. 1.0 is a no-op; 0.0 is greedy.",
                    );
                *t = t.clamp(0.0, 2.0);
                inner
            }
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
            Self::Json => ui
                .label("JSON grammar (built in)")
                .on_hover_text_at_pointer(
                    "JSON grammar constraint. Filters candidates to those that keep output valid JSON. On violation, forces EOS. Live parser position is per-call state, not shown here.",
                ),
            Self::Grammar(compiled) => {
                let rule_count = compiled.grammar.rule_count();
                ui.label(format!("Grammar ({rule_count} rules)"))
                    .on_hover_text_at_pointer(
                        "GBNF grammar constraint. Filters candidates to those that extend the grammar. On violation, forces EOS. Live matcher position is per-call state, not shown here.",
                    )
            }
            Self::Deny { range } => {
                ui.label(format!(
                    "Deny: {}..{} ({} ids)",
                    range.start,
                    range.end,
                    (range.end - range.start).max(0)
                ))
                .on_hover_text_at_pointer(
                    "Forbids every candidate whose token id falls in this range. Typically used for the model's reserved/empty-piece vocab tail.",
                )
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
            .id_salt((index, self.name()))
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

#[derive(Debug, thiserror::Error, derive_more::From)]
pub enum SampleError {
    #[error("Sampling failed because of a repetition error: {err}")]
    RepetitionError { err: RepetitionError },
}

static_assertions::assert_impl_all!(SampleError: Send, Sync);

/// Construction of a fresh [`SamplerState`] from the effective config —
/// the sole constructor (config is the authority; the only other door
/// is validated deserialization).
impl SamplerConfig {
    /// Build the per-call run-state this config implies: fresh matchers
    /// at each grammar's root, an inactive deferred matcher when a
    /// deferred grammar is configured, a seeded working RNG, empty
    /// n-gram stats, and the resolved repetition ignore set.
    pub fn init_state<M: crate::backend::Model>(
        &self,
        seed: u128,
        model: &M,
    ) -> SamplerState {
        use crate::sample::state::{DeferredMatcher, MatcherState};
        SamplerState {
            matchers: self
                .modes
                .iter()
                .map(|m| match m {
                    SamplingMode::Grammar(compiled) => {
                        MatcherState::Grammar(compiled.root_state())
                    }
                    SamplingMode::Json => MatcherState::Json(JsonState::new()),
                    _ => MatcherState::Stateless,
                })
                .collect(),
            deferred: self.deferred_grammar.as_ref().map(|d| DeferredMatcher {
                active: false,
                matcher: d.grammar.root_state(),
            }),
            mu: None,
            rng: rand_pcg::Pcg64Mcg::new(seed),
            ngram_stats: NGramStats::new(),
            resolved_ignored: self
                .repetition
                .as_ref()
                .map(|r| r.resolved_ignored(model))
                .unwrap_or_default(),
        }
    }
}

/// Sample a token from the candidates.
///
/// Does NOT advance the constraint matchers: whether the chosen token
/// continues generation is the caller's call, and a token that
/// terminates it must never mutate `state` (tip invariant). Callers
/// that keep generating follow up with [`SamplerState::advance`].
pub(crate) fn sample_token<M: crate::backend::Model + Sync>(
    tokens: &[Token],
    mut candidates: Candidates,
    opts: &SamplerConfig,
    state: &mut SamplerState,
    model: &M,
) -> Result<Token, SampleError> {
    // Suspend the repetition penalty while any byte-constraint in the
    // chain is active and incomplete. Constrained output is
    // format-bound: the grammar necessarily repeats structural tokens
    // (`\n`, `</`, tag words — ordinary text tokens, not on the
    // special-token ignore list), and exiting a permissive region
    // (raw until() value, JSON string) requires emitting an exact
    // multi-byte delimiter built from exactly those tokens. With the
    // penalty live, each repetition crushes the delimiter's logits
    // further, systematically steering sampled generation away from
    // the only exit — observed on Qwen3.6 as tool calls thrashing
    // inside the second parameter value until `max_tokens` (greedy
    // was immune; its margins dwarf the penalty). Anti-degeneration
    // heuristics are for free prose; the constraint owns the shape
    // here. Side effect (accepted, arguably desirable): stats
    // ingestion lives inside the penalty pass, so tokens emitted
    // during the constrained span never enter the stats — structural
    // markers don't seed penalties against later prose. Free-running
    // spans before a lazy trigger and after completion are penalized
    // as usual.
    if !state.constrained_incomplete() {
        if let Some(repetition) = &opts.repetition {
            // Split borrow: the pass reads the resolved ignore set and
            // mutates the stats accumulator — disjoint state fields.
            let SamplerState {
                ngram_stats,
                resolved_ignored,
                ..
            } = &mut *state;
            candidates = apply_sample_repetition_ngram(
                candidates,
                tokens,
                repetition,
                resolved_ignored,
                ngram_stats,
            )?;
        }
    }

    let lazy = opts.lazy_grammar && state.has_active_constraint();
    let banned = opts.banned_specials.as_slice();

    // Fallback snapshots (lazy-grammar check and/or emit-side specials
    // ban): `Pcg64Mcg` is a single `u128` of state (Clone), `mu` is a
    // plain `Option<f32>`, and the pre-fold candidates clone is a
    // straight memcpy of the vector. Restoring these and replaying the
    // fold consumes the identical RNG draw sequence on either path, so
    // a fixed seed yields the same stream every run regardless of how
    // many checks fall back.
    let snapshot = if lazy || !banned.is_empty() {
        Some((state.rng.clone(), state.mu, candidates.clone()))
    } else {
        None
    };

    let filtered = apply_modes(candidates, opts, state, model, lazy);
    let mut chosen = choose_candidate(&mut state.rng, filtered.softmax(None))
        .is_one()
        .unwrap()
        .id;

    if let Some((rng_snap, mu_snap, saved)) = snapshot.as_ref().filter(|_| lazy)
    {
        // Verify just the chosen token's piece against every constraint —
        // O(piece bytes) vs the O(vocab) filter sweep. Matching the
        // masked filters: empty pieces are rejected outright (zero
        // bytes never advance a matcher — reserved-token loops), and
        // end-of-generation tokens are rejected BY ID while the
        // constraint is incomplete (byte-acceptance can't catch EOG
        // inside permissive regions like raw until() values — see
        // `grammar_filter`). The fallback rerun below then applies
        // the same policy vocab-wide, reaching force-EOS termination
        // when nothing legal remains.
        let t0 = grammar::grammar_stats_enabled().then(std::time::Instant::now);
        let legal = state.accepts_chosen(opts, chosen, model);
        if let Some(t0) = t0 {
            grammar::record_lazy(legal, t0.elapsed().as_micros() as u64);
        }
        if !legal {
            // Rejected: restore the pre-fold state and rerun the exact
            // masked path, grammar filters included — with the forced-EOS
            // termination those filters provide when nothing legal
            // remains.
            state.rng = rng_snap.clone();
            state.mu = *mu_snap;
            let filtered =
                apply_modes(saved.clone(), opts, state, model, false);
            chosen = choose_candidate(&mut state.rng, filtered.softmax(None))
                .is_one()
                .unwrap()
                .id;
        }
    }

    // Emit-side special-token ban (#31 item 9): the sampled id is
    // checked against the dialect ban set — O(log n) per token, the
    // same accept-then-mask shape as the lazy path above. On a hit
    // (pathological, not steady-state) restore the pre-fold state,
    // drop every banned id from the candidates, and rerun the full
    // masked path. The emission is never silently rewritten: the
    // resample IS the mask, applied before anything commits, and a
    // vacuously-empty candidate set forces EOS exactly like
    // `SamplingMode::Deny`.
    let banned = opts.banned_specials.as_slice();
    if !banned.is_empty() && banned.binary_search(&chosen).is_ok() {
        if let Some((rng_snap, mu_snap, saved)) = snapshot {
            state.rng = rng_snap;
            state.mu = mu_snap;
            let kept: Vec<crate::TokenData> = saved
                .as_slice()
                .iter()
                .filter(|td| banned.binary_search(&td.id).is_err())
                .copied()
                .collect();
            let cleaned = if kept.is_empty() {
                Candidates::from_vec(vec![crate::TokenData {
                    id: model.eos(),
                    logit: 0.0,
                    p: 0.0,
                }])
            } else {
                Candidates::from_vec_unchecked(kept)
            };
            let filtered = apply_modes(cleaned, opts, state, model, false);
            chosen = choose_candidate(&mut state.rng, filtered.softmax(None))
                .is_one()
                .unwrap()
                .id;
        }
    }

    // NOTE: constraint matchers are deliberately NOT advanced here.
    // The caller decides whether the chosen token continues generation
    // and calls `state.advance` only then (tip invariant: a token that
    // terminates generation never mutates the sampler state — it is
    // absent from the cache entries and the KV alike, so the state must
    // not carry its bytes either). See `TokenPredictor::next`.
    Ok(chosen)
}

/// Fold `candidates` through `opts.modes` in order (plus the activated
/// deferred grammar, if any — it runs last, matching its old
/// pushed-to-the-end promotion position). With `skip_constraints`, the
/// `Grammar`/`Json` arms pass candidates through untouched (the lazy
/// fast path); every other mode still runs, so `Deny` reserved-token
/// protection, truncation samplers, and mirostat behave identically on
/// both paths.
fn apply_modes<M: crate::backend::Model + Sync>(
    candidates: Candidates,
    opts: &SamplerConfig,
    state: &mut SamplerState,
    model: &M,
    skip_constraints: bool,
) -> Candidates {
    use crate::sample::state::MatcherState;
    debug_assert_eq!(opts.modes.len(), state.matchers.len());
    // Split borrow: matchers are read by the constraint arms while
    // rng/mu feed mirostat — disjoint fields of the same state.
    let SamplerState {
        matchers,
        deferred,
        mu,
        rng,
        ..
    } = &mut *state;
    let mut candidates = opts.modes.iter().zip(matchers.iter()).fold(
        candidates,
        |candidates, (mode, matcher)| {
            if skip_constraints
                && matches!(mode, SamplingMode::Json | SamplingMode::Grammar(_))
            {
                return candidates;
            }
            match (mode, matcher) {
                (SamplingMode::Greedy, _) => candidates.sample_token_greedy(),
                (SamplingMode::Temperature { t }, _) => {
                    candidates.temperature(*t)
                }
                (SamplingMode::TopP { p, min_keep }, _) => {
                    candidates.top_p(*p, *min_keep)
                }
                (SamplingMode::TopK { k }, _) => candidates.top_k(*k),
                (SamplingMode::MinP { p, min_keep }, _) => {
                    candidates.min_p(*p, *min_keep)
                }
                (SamplingMode::TailFree { z, min_keep }, _) => {
                    candidates.tail_free(*z, *min_keep)
                }
                (SamplingMode::LocallyTypical { p, min_keep }, _) => {
                    candidates.locally_typical(*p, *min_keep)
                }
                (SamplingMode::Mirostat { tau, eta, max_keep }, _) => {
                    candidates.mirostat(rng, *tau, *eta, *max_keep, mu)
                }
                (SamplingMode::MirostatV2 { tau, eta, max_keep }, _) => {
                    candidates.mirostat_v2(rng, *tau, *eta, *max_keep, mu)
                }
                (SamplingMode::SplitP { min_keep, max_keep }, _) => {
                    candidates.split_p(*min_keep, *max_keep)
                }
                (SamplingMode::SplitL { min_keep, max_keep }, _) => {
                    candidates.split_l(*min_keep, *max_keep)
                }
                (SamplingMode::Json, MatcherState::Json(parser)) => {
                    json::json_filter(candidates, parser, model)
                }
                (
                    SamplingMode::Grammar(compiled),
                    MatcherState::Grammar(matcher),
                ) => grammar::grammar_filter(
                    candidates, compiled, matcher, model,
                ),
                (SamplingMode::Json, _) | (SamplingMode::Grammar(_), _) => {
                    // Config/state kind mismatch: impossible via
                    // init_state; deserialized state is validated
                    // before use. Pass through rather than panic.
                    debug_assert!(false, "matcher/mode kind mismatch");
                    candidates
                }
                (SamplingMode::Deny { range }, _) => {
                    let kept: Vec<crate::TokenData> = candidates
                        .as_slice()
                        .iter()
                        .filter(|td| !range.contains(&td.id))
                        .copied()
                        .collect();
                    if kept.is_empty() {
                        // Vacuously denied. Force EOS so the predictor's
                        // stop machinery can halt generation.
                        Candidates::from_vec(vec![crate::TokenData {
                            id: model.eos(),
                            logit: 0.0,
                            p: 0.0,
                        }])
                    } else {
                        Candidates::from_vec_unchecked(kept)
                    }
                }
            }
        },
    );
    if !skip_constraints {
        if let (Some(d), Some(spec)) =
            (deferred.as_ref(), opts.deferred_grammar.as_ref())
        {
            if d.active {
                candidates = grammar::grammar_filter(
                    candidates,
                    &spec.grammar,
                    &d.matcher,
                    model,
                );
            }
        }
    }
    candidates
}

/// Apply the softmax function to the remaining candidates and select a single
/// candidate. This function is guaranteed to leave the candidates with only
/// one token.
// TODO: better name
pub(crate) fn choose_candidate(
    rng: &mut rand_pcg::Pcg64Mcg,
    candidates: Candidates,
) -> Candidates {
    if candidates.len().get() == 1 {
        return candidates;
    }

    let candidates = candidates.softmax(None);

    // Pick a token based on the probabilities
    let val = rng.random_range(0.0..1.0);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Token, TokenData};
    use rand::Rng as _;

    /// Minimal `Model` for exercising `sample_token` without a GGUF on
    /// disk. Token id indexes straight into `PIECES`; id 0 is EOS with an
    /// empty piece (like Qwen's secondary EOS variants decode to).
    struct MockModel;

    const PIECES: &[&str] = &["", "a", "b", "c", "x", "", "a", "b"];
    const EOS: Token = 0;
    const A: Token = 1;
    const B: Token = 2;
    const C: Token = 3;
    const X: Token = 4;
    /// A reserved-style token whose piece is empty but which is NOT
    /// EOS — the Qwen3.6 shape behind the post-complete budget-burn
    /// loop.
    const RSV: Token = 5;
    /// An extra-EOS token whose piece renders as byte-legal text
    /// ("a") — the `<|im_end|>`-inside-until() shape: byte-acceptance
    /// alone cannot reject it, only the EOG-by-id rule can.
    const EOG_A: Token = 6;
    /// An extra-EOS token whose piece ("b") COMPLETES the test
    /// grammar from the mid-parse state — the dialect-exit-marker
    /// shape (Gemma 4's `<|tool_response>` is both grammar-required
    /// exit bytes and a vocab EOG).
    const EOG_B: Token = 7;

    impl crate::backend::Model for MockModel {
        type Error = std::convert::Infallible;

        fn n_vocab(&self) -> i32 {
            PIECES.len() as i32
        }
        fn bos(&self) -> Token {
            EOS
        }
        fn eos(&self) -> Token {
            EOS
        }
        fn eot(&self) -> Token {
            EOS
        }
        fn special_tokens(&self) -> Vec<Token> {
            vec![EOS, EOG_A]
        }
        fn eog_tokens(&self) -> Vec<Token> {
            vec![EOS, EOG_A, EOG_B]
        }
        fn max_token_len(&self) -> usize {
            1
        }
        fn tokenize(&self, _input: &str, _special: bool) -> Vec<Token> {
            unimplemented!("not needed by sample_token")
        }
        fn token_to_piece(&self, token: Token) -> String {
            PIECES[token as usize].to_string()
        }
        fn token_to_piece_ref(&self, token: Token, buf: &mut Vec<u8>) {
            buf.clear();
            buf.extend_from_slice(PIECES[token as usize].as_bytes());
        }
        fn context_size(&self) -> i32 {
            4096
        }
        fn chat_template_source(&self) -> Option<String> {
            None
        }
        fn get_meta(&self, _key: &str) -> Option<String> {
            None
        }
    }

    /// `root ::= "ab"` — accepts exactly the string "ab".
    const AB_GRAMMAR: &str = r#"root ::= "ab""#;

    fn opts_with_grammar(lazy: bool) -> SamplerConfig {
        SamplerConfig {
            modes: vec![
                SamplingMode::grammar(AB_GRAMMAR).expect("test grammar parses")
            ],
            repetition: None,
            deferred_grammar: None,
            lazy_grammar: lazy,
            ..SamplerConfig::default()
        }
    }

    /// Candidates from (id, logit) pairs.
    fn cands(pairs: &[(Token, f32)]) -> Candidates {
        Candidates::from_vec(
            pairs
                .iter()
                .map(|&(id, logit)| TokenData { id, logit, p: 0.0 })
                .collect(),
        )
    }

    const TEST_SEED: u128 = (1337u128 << 64) | 42;

    fn state_for(opts: &SamplerConfig) -> SamplerState {
        opts.init_state(TEST_SEED, &MockModel)
    }

    /// Sample and advance — the non-terminal-token path, as the
    /// predictor loop drives it. Tests exercising the tip invariant
    /// (terminal tokens must not advance) call `sample_token` raw.
    fn sample(
        candidates: Candidates,
        opts: &SamplerConfig,
        state: &mut SamplerState,
    ) -> Token {
        let chosen = sample_token(&[], candidates, opts, state, &MockModel)
            .expect("sample_token");
        state.advance(opts, chosen, &MockModel);
        chosen
    }

    /// True iff the (single) grammar matcher accepts `bytes` from its
    /// current position.
    fn grammar_accepts(
        opts: &SamplerConfig,
        state: &SamplerState,
        bytes: &[u8],
    ) -> bool {
        use crate::sample::state::MatcherState;
        match (&opts.modes[0], &state.matchers[0]) {
            (SamplingMode::Grammar(compiled), MatcherState::Grammar(s)) => {
                s.accepts_bytes(&compiled.grammar, bytes)
            }
            _ => unreachable!("first mode is not a grammar"),
        }
    }

    /// Emit-side special-token ban (#31 item 9): a banned sampled id
    /// triggers the masked resample (never surfaced), a non-banned
    /// winner is untouched, and a vacuously-banned candidate set
    /// forces EOS.
    #[test]
    fn banned_specials_masked_resample() {
        let opts = SamplerConfig {
            modes: vec![SamplingMode::Greedy],
            repetition: None,
            deferred_grammar: None,
            lazy_grammar: false,
            banned_specials: vec![X],
        };
        let mut state = state_for(&opts);

        // Banned winner is masked; next-best is sampled instead.
        let picked =
            sample(cands(&[(X, 10.0), (A, 5.0), (B, 1.0)]), &opts, &mut state);
        assert_eq!(picked, A, "banned winner must be masked, not surfaced");

        // Non-banned winner passes untouched.
        let picked = sample(cands(&[(A, 10.0), (X, 5.0)]), &opts, &mut state);
        assert_eq!(picked, A);

        // Every candidate banned → forced EOS (Deny semantics), so
        // generation terminates instead of deadlocking.
        let opts = SamplerConfig {
            modes: vec![SamplingMode::Greedy],
            repetition: None,
            deferred_grammar: None,
            lazy_grammar: false,
            banned_specials: vec![A, B, C, X],
        };
        let mut state = state_for(&opts);
        let picked = sample(cands(&[(A, 10.0), (X, 5.0)]), &opts, &mut state);
        assert_eq!(picked, EOS);
    }

    /// Ban vs grammar conflict: banning a token the grammar requires
    /// resolves to the grammar's forced-EOS termination, never a
    /// banned emission and never a deadlock. (Session never bans
    /// dialect-marker tokens, so this is a mechanism guarantee, not a
    /// steady-state path.)
    #[test]
    fn banned_specials_with_grammar_terminates() {
        let mut opts = opts_with_grammar(false);
        opts.banned_specials = vec![A];
        let mut state = state_for(&opts);
        // Grammar "ab" requires A first; A is banned → masked rerun
        // has no legal candidate → force-EOS.
        let picked =
            sample(cands(&[(A, 10.0), (B, 5.0), (X, 1.0)]), &opts, &mut state);
        assert_eq!(picked, EOS);
    }

    /// Legal unconstrained winner: fast path must keep it and advance the
    /// matcher exactly as the masked path would, token for token.
    ///
    /// Deliberately NOT asserted: RNG-stream alignment across paths.
    /// The masked filter drops empty-piece tokens (in any state — see
    /// `grammar_filter`), so its kept set can collapse to a single
    /// candidate, and `choose_candidate` consumes no draw for a
    /// singleton — draw *counts* legitimately differ between the lazy
    /// and masked paths. Cross-path bit-exactness only ever held for
    /// peaked distributions anyway (one uniform draw over different
    /// softmax sets can pick different tokens); the hard guarantee is
    /// the fallback replay, pinned by
    /// `lazy_fallback_matches_masked_path_exactly`.
    #[test]
    fn lazy_fast_path_matches_masked_on_legal_picks() {
        let mut results: Vec<Vec<Token>> = Vec::new();
        for lazy in [false, true] {
            let opts = opts_with_grammar(lazy);
            let mut state = state_for(&opts);
            let mut toks = Vec::new();
            // "a" then "b" dominate in turn; both legal under the grammar
            // at their step.
            toks.push(sample(
                cands(&[(A, 20.0), (X, 0.0), (EOS, -20.0)]),
                &opts,
                &mut state,
            ));
            toks.push(sample(
                cands(&[(B, 20.0), (X, 0.0), (EOS, -20.0)]),
                &opts,
                &mut state,
            ));
            results.push(toks);
        }
        assert_eq!(results[0], vec![A, B]);
        assert_eq!(results[0], results[1]);
    }

    /// Illegal unconstrained winner: the fallback must restore RNG + mu
    /// and rerun the masked path bit-exactly — same chosen token, same
    /// post-call RNG state as a `lazy = false` run.
    #[test]
    fn lazy_fallback_matches_masked_path_exactly() {
        let mut results: Vec<(Token, u64)> = Vec::new();
        for lazy in [false, true] {
            let opts = opts_with_grammar(lazy);
            let mut state = state_for(&opts);
            // "x" dominates but is illegal; "a" is the legal pick.
            let tok = sample(
                cands(&[(X, 20.0), (A, 10.0), (EOS, -20.0)]),
                &opts,
                &mut state,
            );
            results.push((tok, state.rng.next_u64()));
        }
        assert_eq!(results[0].0, A);
        assert_eq!(results[0], results[1]);
    }

    /// After the grammar completes, a lazy sample whose candidates hold
    /// no legal (or empty) piece must fall back into the filter's
    /// forced-EOS termination. The matcher stays complete — the old
    /// auto-reset-for-the-next-generation died with the config/state
    /// split (fresh state per call via `init_state` replaces it).
    #[test]
    fn lazy_completion_forces_eos_and_stays_complete() {
        let opts = opts_with_grammar(true);
        let mut state = state_for(&opts);
        assert_eq!(sample(cands(&[(A, 20.0), (X, 0.0)]), &opts, &mut state), A);
        assert_eq!(sample(cands(&[(B, 20.0), (X, 0.0)]), &opts, &mut state), B);
        // Grammar complete; only non-empty illegal pieces on offer.
        assert_eq!(
            sample(cands(&[(C, 20.0), (X, 0.0)]), &opts, &mut state),
            EOS
        );
        // No auto-reset: the matcher remains at accept.
        assert!(state.grammar_complete());
        // A fresh state from the same config starts at root again.
        let fresh = state_for(&opts);
        assert!(grammar_accepts(&opts, &fresh, b"a"));
    }

    /// Empty pieces are rejected mid-grammar on both paths: an active
    /// constraint owns termination, so a dominant empty-piece token
    /// (EOS variant or reserved slot) must lose to a byte-legal
    /// candidate rather than bail out of — or livelock inside — the
    /// forced structure. Regression for the mid-call stall observed
    /// on Qwen3.6: the repetition penalty crushed the grammar-forced
    /// structural tokens until an unpenalized empty-piece reserved
    /// token dominated, and generation burned to `max_tokens` with no
    /// visible output.
    #[test]
    fn lazy_rejects_empty_piece_mid_grammar() {
        for lazy in [true, false] {
            let opts = opts_with_grammar(lazy);
            let mut state = state_for(&opts);
            // EOS (empty piece) dominates mid-grammar; "a" is the
            // byte-legal pick and must win on both paths.
            let tok =
                sample(cands(&[(EOS, 20.0), (A, 0.0)]), &opts, &mut state);
            assert_eq!(tok, A, "lazy={lazy}");
        }
    }

    /// An end-of-generation token whose piece is byte-LEGAL under the
    /// grammar must still be rejected mid-parse, on both paths: EOG
    /// is an end-of-generation *signal*, not content, and inside
    /// permissive regions (raw until() values, JSON strings) its
    /// literal piece bytes always pass the matcher. Regression for
    /// the Qwen3.6 mid-call bail: the model sampled `<|im_end|>`
    /// inside a `<parameter=…>` value — legal bytes to the until-DFA
    /// — and the predictor's EOG stop killed the call at the second
    /// parameter.
    #[test]
    fn eog_with_byte_legal_piece_rejected_mid_grammar() {
        for lazy in [true, false] {
            let opts = opts_with_grammar(lazy);
            let mut state = state_for(&opts);
            // EOG_A's piece is "a" — exactly what the "ab" grammar
            // wants next — but it must lose to the real "a" token.
            let tok =
                sample(cands(&[(EOG_A, 20.0), (A, 0.0)]), &opts, &mut state);
            assert_eq!(tok, A, "lazy={lazy}");
            // The matcher advanced by "a" exactly once: "b" completes.
            assert!(grammar_accepts(&opts, &state, b"b"));
        }
    }

    /// An EOG token whose own piece bytes COMPLETE the constraint is
    /// legal mid-parse — the dialect-exit-marker case: Gemma 4's
    /// `<|tool_response>` is both the grammar-required exit bytes and
    /// a vocab EOG (libllama marks it), and rejecting it by id left
    /// the model with no trained stop, looping identical calls to the
    /// context limit (plan Phase G postmortem). Both sampling paths.
    #[test]
    fn eog_kept_when_piece_completes_constraint() {
        for lazy in [false, true] {
            let opts = opts_with_grammar(lazy);
            let mut state = state_for(&opts);
            assert_eq!(
                sample(cands(&[(A, 20.0), (X, 0.0)]), &opts, &mut state),
                A,
                "lazy={lazy}"
            );
            // Mid-parse (after "a"): EOG_B's piece "b" finishes
            // `root ::= "ab"` — it must survive the EOG-by-id rule.
            assert_eq!(
                sample(
                    cands(&[(EOG_B, 20.0), (B, 0.0), (X, -20.0)]),
                    &opts,
                    &mut state
                ),
                EOG_B,
                "lazy={lazy}"
            );
        }
    }

    /// Mid-grammar with ONLY empty/illegal pieces on offer, the
    /// filter's force-EOS branch terminates generation (a grammar
    /// violation the Session layer surfaces) instead of emitting an
    /// invisible token and spinning.
    #[test]
    fn empty_piece_only_candidates_force_eos_mid_grammar() {
        let opts = opts_with_grammar(false);
        let mut state = state_for(&opts);
        let tok = sample(cands(&[(RSV, 20.0), (EOS, 0.0)]), &opts, &mut state);
        assert_eq!(tok, EOS);
    }

    /// Post-complete, an empty-piece NON-EOS pick must not be
    /// accepted: the fallback's force-EOS terminates generation
    /// instead. Regression for the invisible reserved-token loop
    /// observed on Qwen3.6 (every constrained run consumed exactly
    /// `max_tokens` after the document completed).
    #[test]
    fn lazy_rejects_empty_piece_after_completion() {
        let opts = opts_with_grammar(true);
        let mut state = state_for(&opts);
        assert_eq!(sample(cands(&[(A, 20.0), (X, 0.0)]), &opts, &mut state), A);
        assert_eq!(sample(cands(&[(B, 20.0), (X, 0.0)]), &opts, &mut state), B);
        // Grammar complete. The dominant pick is a reserved-style
        // empty-piece token; pre-fix it was accepted and emitted
        // forever. Now: rejected, fallback drops it too, EOS forced.
        assert_eq!(
            sample(cands(&[(RSV, 20.0), (X, 0.0)]), &opts, &mut state),
            EOS
        );
    }

    /// Same shape through the masked path: the filter itself must
    /// drop post-complete empty pieces and force EOS.
    #[test]
    fn masked_rejects_empty_piece_after_completion() {
        let opts = opts_with_grammar(false);
        let mut state = state_for(&opts);
        assert_eq!(sample(cands(&[(A, 20.0), (X, 0.0)]), &opts, &mut state), A);
        assert_eq!(sample(cands(&[(B, 20.0), (X, 0.0)]), &opts, &mut state), B);
        assert_eq!(
            sample(cands(&[(RSV, 20.0), (X, 0.0)]), &opts, &mut state),
            EOS
        );
    }

    /// mu snapshot/restore: with mirostat in the chain, a lazy fallback
    /// must leave `mu` exactly as the masked run computes it.
    #[test]
    fn lazy_fallback_restores_mu_for_mirostat() {
        let mut results: Vec<(Token, Option<f32>, u64)> = Vec::new();
        for lazy in [false, true] {
            let mut opts = opts_with_grammar(lazy);
            opts.modes.push(SamplingMode::MirostatV2 {
                tau: 5.0,
                eta: 0.1,
                max_keep: None,
            });
            let mut state = state_for(&opts);
            let tok = sample(
                cands(&[(X, 20.0), (A, 10.0), (EOS, -20.0)]),
                &opts,
                &mut state,
            );
            results.push((tok, state.mu, state.rng.next_u64()));
        }
        assert_eq!(results[0].0, A);
        assert_eq!(results[0], results[1]);
    }

    /// The headline invariant of the config/state split: serialize a
    /// mid-generation `SamplerState`, restore it, and both the restored
    /// and original states must produce the identical continuation and
    /// remain equal (derived `PartialEq`) after every subsequent step.
    /// Covers the grammar matcher position, the working RNG, mirostat
    /// `mu`, and the n-gram stats.
    #[cfg(feature = "serde")]
    #[test]
    fn sampler_state_serde_round_trip_bit_exact() {
        let mut opts = opts_with_grammar(false);
        // Empty categories: the defaults resolve against a real
        // tokenizer, which MockModel doesn't have.
        opts.repetition = Some(
            RepetitionOptions::default()
                .set_ignored_categories(std::iter::empty()),
        );
        let opts = opts;
        let mut state = state_for(&opts);

        // Advance mid-grammar (one step into "ab").
        let mut tokens: Vec<Token> = Vec::new();
        let tok = sample_token(
            &tokens,
            cands(&[(A, 20.0), (X, 0.0), (EOS, -20.0)]),
            &opts,
            &mut state,
            &MockModel,
        )
        .unwrap();
        state.advance(&opts, tok, &MockModel);
        tokens.push(tok);

        // Snapshot → restore.
        let blob = serde_json::to_string(&state).unwrap();
        let mut restored: SamplerState = serde_json::from_str(&blob).unwrap();
        assert_eq!(state, restored);
        // Canonical bytes: a second serialization is identical.
        assert_eq!(blob, serde_json::to_string(&restored).unwrap());

        // Continue both in lockstep; streams and states must match
        // exactly at every step.
        let mut restored_tokens = tokens.clone();
        for step in 0..8 {
            let c = cands(&[(B, 4.0), (A, 3.9), (X, 3.8), (EOS, -20.0)]);
            let a =
                sample_token(&tokens, c.clone(), &opts, &mut state, &MockModel)
                    .unwrap();
            state.advance(&opts, a, &MockModel);
            let b = sample_token(
                &restored_tokens,
                c,
                &opts,
                &mut restored,
                &MockModel,
            )
            .unwrap();
            restored.advance(&opts, b, &MockModel);
            assert_eq!(a, b, "diverged at step {step}");
            assert_eq!(state, restored, "state diverged at step {step}");
            tokens.push(a);
            restored_tokens.push(b);
        }
    }
}
