use crate::{ngram::NGramStats, Candidates, Probability, Token};
// `is_protected` — the region-exit walk shared with the constrained
// repetition penalty, reused by the region-scoped emit ban (#37).
use crate::sample::region::RegionGuard as _;

use rand::RngExt as _;

use std::num::NonZeroUsize;

pub(crate) mod grammar;
mod json;
pub(crate) mod region;
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
#[non_exhaustive]
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
    /// Region-scoped special-token ban (sorted ids), applied *only*
    /// inside a **permissive** constraint region — a JSON string body
    /// or an `until()` raw value, where the grammar accepts nearly any
    /// byte and the model owns the content (issue #37).
    ///
    /// [`banned_specials`] must exempt the dialect's own markers, since
    /// the session emits `<tool_call>` legitimately as the *frame*. But
    /// a frame marker is only ever legitimate at a frame position, and
    /// those are grammar *literals* — so inside a free region the
    /// exemption buys nothing and costs everything: `<tool_call>` is
    /// byte-legal string content, ban-exempt, and therefore committed as
    /// the real special id inside an argument value. Relaying that text
    /// into another session's prompt trips the ingest injection guard
    /// and kills the receiving loop.
    ///
    /// So this set carries **no marker exemption** (every special except
    /// the EOG family) and is consulted only where frames are never
    /// legal. Two guards keep it from breaking legitimate emission:
    ///
    /// * It applies only when *every* active constraint is in a
    ///   permissive state. At a structural position the standing
    ///   [`banned_specials`] applies instead — banning marker tokens
    ///   there would force the frame to be spelled as multiple byte
    ///   tokens, which destabilizes the prefix cache.
    /// * A special whose bytes *leave* the region is never banned. That
    ///   is the region-exit walk the constrained-repetition guard
    ///   already implements, so a dialect whose exit delimiter is itself
    ///   a special (Harmony's `<|end|>`) stays completable.
    ///
    /// Empty (the default) disables the region-scoped check, leaving
    /// only the standing ban. Runtime wiring set by `Session`; like
    /// [`banned_specials`] it is unreachable from the wire.
    ///
    /// [`banned_specials`]: SamplerConfig::banned_specials
    #[cfg_attr(feature = "serde", serde(default))]
    pub banned_specials_constrained: Vec<Token>,
}

/// True for modes that constrain *what may be emitted* rather than
/// shape the distribution — grammar, JSON, and token-range denial.
/// These survive a request-driven chain rebuild; the truncation
/// samplers do not.
fn is_constraint(mode: &SamplingMode) -> bool {
    matches!(
        mode,
        SamplingMode::Json
            | SamplingMode::Grammar(_)
            | SamplingMode::Deny { .. }
    )
}

/// Fold a request's sampling knobs into an existing mode chain.
///
/// # The rule
///
/// Evaluated over the **whole requested set**, never per-parameter:
///
/// * If every knob the request names appears in `modes` **exactly
///   once**, each is patched in place and the rest of the chain is
///   left alone. So a request setting only `top_k` against the
///   default chain retunes the top-k and keeps `LocallyTypical`.
/// * Otherwise the chain is rebuilt: constraint modes (grammar,
///   JSON, `Deny`) are kept as a prefix, every distribution-shaping
///   mode is discarded,
///   and a canonical `TopK → TopP → MinP → Temperature` chain is
///   built from `requested` layered over `fallback` (the model's
///   [`recommended_sampling`](crate::backend::Model::recommended_sampling)).
///
/// # Why not per-parameter
///
/// Mixing would produce a chain that is neither behavior. Given
/// `[TopK, TopP, LocallyTypical]` and a request setting `top_p` and
/// `temperature`, patching the `TopP` while appending a `Temperature`
/// leaves `LocallyTypical` sitting at an arbitrary position between
/// them — unexplainable in docs and dependent on sidecar contents the
/// client cannot see.
///
/// # Why not always rebuild
///
/// Because the sidecar is *seeded* from the same metadata used as
/// `fallback`, patch and rebuild agree for any model that advertises
/// `general.sampling.*`. They diverge only on a hand-edited sidecar —
/// and there, discarding the operator's deliberate choice the moment
/// any client sets any knob is the worse surprise.
///
/// # What is never touched
///
/// Only `modes`. Repetition penalties, `lazy_grammar`, and
/// `banned_specials` live on [`SamplerConfig`] and are out of reach —
/// `banned_specials` in particular is emission-side protocol
/// integrity, which a remote client must not be able to switch off.
///
/// # Mirostat
///
/// A `requested` mirostat always forces a rebuild: it is terminal, so
/// patching it into an existing chain would leave the surrounding
/// modes as silent no-ops. In the other direction, a `fallback`
/// mirostat is dropped whenever the request asks for anything else,
/// since it would otherwise swallow the very knobs the client set.
pub fn apply_request_sampling(
    modes: Vec<SamplingMode>,
    requested: SamplingParams,
    fallback: SamplingParams,
) -> Vec<SamplingMode> {
    if requested.is_empty() {
        return modes;
    }

    // A knob is patchable when the request didn't name it, or named
    // it and the chain has exactly one mode to put it in. Zero is
    // "nowhere to put it"; two or more is "no way to tell which the
    // client meant" — both mean rebuild.
    let unambiguous = |asked: bool, pred: fn(&SamplingMode) -> bool| {
        !asked || modes.iter().filter(|m| pred(m)).count() == 1
    };
    let can_patch = requested.mirostat.is_none()
        && unambiguous(requested.temp.is_some(), |m| {
            matches!(m, SamplingMode::Temperature { .. })
        })
        && unambiguous(requested.top_p.is_some(), |m| {
            matches!(m, SamplingMode::TopP { .. })
        })
        && unambiguous(requested.top_k.is_some(), |m| {
            matches!(m, SamplingMode::TopK { .. })
        })
        && unambiguous(requested.min_p.is_some(), |m| {
            matches!(m, SamplingMode::MinP { .. })
        });

    if can_patch {
        return modes
            .into_iter()
            .map(|mode| match mode {
                SamplingMode::Temperature { t } => SamplingMode::Temperature {
                    t: requested.temp.unwrap_or(t),
                },
                SamplingMode::TopP { p, min_keep } => SamplingMode::TopP {
                    p: requested.top_p.unwrap_or(p),
                    min_keep,
                },
                SamplingMode::TopK { k } => SamplingMode::TopK {
                    k: requested.top_k.unwrap_or(k),
                },
                SamplingMode::MinP { p, min_keep } => SamplingMode::MinP {
                    p: requested.min_p.unwrap_or(p),
                    min_keep,
                },
                other => other,
            })
            .collect();
    }

    let mut merged = requested.or(fallback);
    // `requested` is non-empty and named no mirostat, so it named a
    // truncation knob — which a fallback mirostat would silently
    // swallow, since mirostat compiles to a chain of one.
    if requested.mirostat.is_none() {
        merged.mirostat = None;
    }

    let mut rebuilt: Vec<SamplingMode> =
        modes.into_iter().filter(is_constraint).collect();
    rebuilt.extend(Vec::<SamplingMode>::from(merged));
    rebuilt
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

/// The scalar sampling knobs that a *model* recommends and that the
/// Anthropic wire format can carry — the common vocabulary shared by
/// llama.cpp's `general.sampling.*` GGUF metadata, OpenAI's and
/// Anthropic's request bodies, and every `--temp`-style CLI on earth.
///
/// Deliberately **not** a [`SamplerConfig`]: this is a bag of
/// independent values, not an ordered pipeline. Converting to a
/// pipeline is [`From<SamplingParams> for Vec<SamplingMode>`], which
/// imposes the canonical `llama.cpp` order.
///
/// Every field is `Option` because "the model didn't say" and "the
/// client didn't ask" are the same shape and must stay
/// distinguishable from "the model said 1.0". Members are stored
/// **already validated** ([`Probability`] rather than raw `f32`) so
/// that conversion into modes cannot fail — validation happens once,
/// at the edge where the value enters (metadata parse or wire
/// deserialization).
///
/// Two producers:
/// * [`Model::recommended_sampling`](crate::backend::Model::recommended_sampling)
///   — what the model's own metadata asks for. Seeds a fresh sampling
///   sidecar (see the [`sidecar`](crate::sidecar) module docs).
/// * The request body's `temperature` / `top_p` / `top_k`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct SamplingParams {
    /// Temperature. Not a [`Probability`] — values above `1.0` are
    /// legal and useful (`llama.cpp` allows them), and `t <= 0.0` is
    /// defined as greedy. Nothing to validate.
    pub temp: Option<f32>,
    /// Nucleus / top-p threshold.
    pub top_p: Option<Probability<f64>>,
    /// Top-k cutoff.
    pub top_k: Option<NonZeroUsize>,
    /// Min-p threshold. Readable from model metadata; not on the
    /// Anthropic wire.
    pub min_p: Option<Probability<f32>>,
    /// Mirostat, if the model asks for it. Terminal — see
    /// [`From<SamplingParams> for Vec<SamplingMode>`].
    pub mirostat: Option<Mirostat>,
}

/// Which mirostat algorithm a model recommends, plus its parameters.
/// Mirostat is *terminal* — it yields a single token — so it never
/// composes with the truncation knobs in [`SamplingParams`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mirostat {
    /// Mirostat v1. See [`SamplingMode::Mirostat`].
    V1 {
        /// Target entropy.
        tau: f32,
        /// Learning rate.
        eta: f32,
    },
    /// Mirostat v2. See [`SamplingMode::MirostatV2`].
    V2 {
        /// Target entropy.
        tau: f32,
        /// Learning rate.
        eta: f32,
    },
}

impl SamplingParams {
    /// True when nothing at all was specified — the model carries no
    /// `general.sampling.*` metadata (gpt-oss), or the request set no
    /// sampling fields. Callers use this to decide whether to fall
    /// back to [`SamplerConfig::default`].
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    /// Field-by-field [`Option::or`]: `self`'s values win, `other`
    /// fills the gaps. Used to layer a request's explicit knobs over
    /// the model's recommendation, so a client that sets only
    /// `temperature` still gets the model's own top-k and top-p
    /// rather than the crate's.
    pub fn or(self, other: Self) -> Self {
        Self {
            temp: self.temp.or(other.temp),
            top_p: self.top_p.or(other.top_p),
            top_k: self.top_k.or(other.top_k),
            min_p: self.min_p.or(other.min_p),
            mirostat: self.mirostat.or(other.mirostat),
        }
    }
}

impl From<SamplingParams> for Vec<SamplingMode> {
    /// Compile a bag of scalars into an ordered chain.
    ///
    /// Order is `llama.cpp`'s conventional one — top-k, then top-p,
    /// then min-p, then temperature — so that truncation happens
    /// before rescaling and a chain built here behaves the way every
    /// other inference stack's does with the same numbers.
    ///
    /// Mirostat, when present, is emitted **alone**: it returns a
    /// single token, and every mode after a single-candidate set is a
    /// silent no-op (`Candidates::top_k` and friends open with
    /// `if len == 1 { return self }`). Emitting `[Mirostat, TopK]`
    /// would look like a composed chain while the `TopK` did nothing
    /// — the misleading shape is worse than the dropped knob.
    ///
    /// `None` members are skipped, so an empty `SamplingParams`
    /// yields an empty chain (which the caller should treat as
    /// "use the default", not "sample from the raw distribution").
    fn from(params: SamplingParams) -> Self {
        // `min_keep` has no metadata or wire equivalent; 1 matches
        // the crate's own `SamplingMode::top_p()` / `min_p()`
        // constructors.
        const MIN_KEEP: NonZeroUsize = NonZeroUsize::new(1).unwrap();

        if let Some(mirostat) = params.mirostat {
            return match mirostat {
                Mirostat::V1 { tau, eta } => vec![SamplingMode::Mirostat {
                    tau,
                    eta,
                    max_keep: None,
                }],
                Mirostat::V2 { tau, eta } => vec![SamplingMode::MirostatV2 {
                    tau,
                    eta,
                    max_keep: None,
                }],
            };
        }

        let mut modes = Vec::with_capacity(4);
        if let Some(k) = params.top_k {
            modes.push(SamplingMode::TopK { k });
        }
        if let Some(p) = params.top_p {
            modes.push(SamplingMode::TopP {
                p,
                min_keep: MIN_KEEP,
            });
        }
        if let Some(p) = params.min_p {
            modes.push(SamplingMode::MinP {
                p,
                min_keep: MIN_KEEP,
            });
        }
        if let Some(t) = params.temp {
            modes.push(SamplingMode::Temperature { t });
        }
        modes
    }
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
            banned_specials_constrained: Vec::new(),
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
            // Top-K 1024 is a pre-cut, not a sampler: typical mass
            // lives well inside the top few hundred tokens, so the
            // narrowing is behaviorally invisible — but it bounds the
            // locally-typical pass (softmax + entropy over candidates)
            // at 1024 entries instead of the full vocab.
            modes: vec![
                SamplingMode::TopK {
                    k: std::num::NonZeroUsize::new(1024).unwrap(),
                },
                SamplingMode::locally_typical(),
            ],
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
            banned_specials_constrained: Vec::new(),
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
#[non_exhaustive]
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
                    SamplingMode::Grammar(compiled) => MatcherState::Grammar {
                        grammar: compiled.source_hash(),
                        stack: compiled.root_state(),
                    },
                    SamplingMode::Json => MatcherState::Json(JsonState::new()),
                    _ => MatcherState::Stateless,
                })
                .collect(),
            deferred: self.deferred_grammar.as_ref().map(|d| DeferredMatcher {
                active: false,
                grammar: d.grammar.source_hash(),
                matcher: d.grammar.root_state(),
            }),
            mu: None,
            rng: rand_pcg::Pcg64Mcg::new(seed),
            ngram_stats: NGramStats::new(),
            step: 0,
            resolved_ignored: self
                .repetition
                .as_ref()
                .map(|r| r.resolved_ignored(model))
                .unwrap_or_default(),
            constrained_ngram_stats: NGramStats::new(),
            constrained_step: 0,
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
    // Repetition penalty, in one of three regimes per token:
    //
    // (a) UNCONSTRAINED — the pre-existing pass over the persistent
    //     prose corpus, unchanged.
    // (b) CONSTRAINED, all incomplete constraints in a *permissive*
    //     free region (JSON string body, until() value — see
    //     `sample::region`) — the penalty runs against a CALL-LOCAL
    //     accumulator with the region guard exempting every token
    //     whose bytes exit the region. This is what breaks
    //     small-model loops inside always-on tool-call grammars: the
    //     loop tokens accrue penalty while the closing delimiter's
    //     logit is untouched (and thus relatively boosted).
    // (c) CONSTRAINED at a *structural* state — full suspension, the
    //     original rule. Constrained output is format-bound there:
    //     the grammar necessarily repeats structural tokens (`\n`,
    //     `</`, tag words), and exiting requires an exact delimiter
    //     built from exactly those tokens. With the penalty live,
    //     each repetition crushes the delimiter's logits further,
    //     systematically steering sampled generation away from the
    //     only exit — observed on Qwen3.6 as tool calls thrashing
    //     inside the second parameter value until `max_tokens`
    //     (greedy was immune; its margins dwarf the penalty).
    //
    // Stats ingestion lives inside each pass: constrained-span tokens
    // never enter the PERSISTENT stats (structural markers must not
    // seed penalties against later prose, and Session's cold fold
    // could not re-derive them — seeding excludes tool args). The
    // call-local accumulator absorbs regime (b) and dies at the call
    // boundary. Each corpus advances its own step counter only when
    // its pass executes.
    if let Some(repetition) = &opts.repetition {
        let incomplete = state.constrained_incomplete();
        // Split borrow: the passes read the resolved ignore set and
        // the matcher positions, and mutate one stats accumulator —
        // disjoint state fields.
        let SamplerState {
            matchers,
            deferred,
            ngram_stats,
            resolved_ignored,
            step,
            constrained_ngram_stats,
            constrained_step,
            ..
        } = &mut *state;
        if !incomplete {
            candidates = apply_sample_repetition_ngram(
                candidates,
                tokens,
                *step,
                repetition,
                resolved_ignored,
                ngram_stats,
            )?;
            *step += 1;
        } else if repetition.constrained_regions() {
            // `build` returns None at structural states (regime (c))
            // and pre-checks every incomplete constraint permissive.
            // Single-threaded point of the step — `intern_base` safe.
            if let Some(guard) = region::ConstraintGuard::build(
                &opts.modes,
                &*matchers,
                deferred.as_ref(),
                opts.deferred_grammar.as_ref(),
                model,
            ) {
                candidates = repetition::apply_sample_repetition_ngram_guarded(
                    candidates,
                    tokens,
                    *constrained_step,
                    repetition,
                    resolved_ignored,
                    constrained_ngram_stats,
                    Some(&guard),
                )?;
                *constrained_step += 1;
            }
        }
    }

    let lazy = opts.lazy_grammar && state.has_active_constraint();
    let banned = opts.banned_specials.as_slice();
    let banned_in_region = opts.banned_specials_constrained.as_slice();

    // Fallback snapshots (lazy-grammar check and/or emit-side specials
    // ban): `Pcg64Mcg` is a single `u128` of state (Clone), `mu` is a
    // plain `Option<f32>`, and the pre-fold candidates clone is a
    // straight memcpy of the vector. Restoring these and replaying the
    // fold consumes the identical RNG draw sequence on either path, so
    // a fixed seed yields the same stream every run regardless of how
    // many checks fall back.
    let snapshot = if lazy || !banned.is_empty() || !banned_in_region.is_empty()
    {
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
    //
    // Two sets, selected by position (#37). At a structural position the
    // standing `banned_specials` applies — it exempts the dialect's own
    // markers so the frame can be emitted as single tokens (banning them
    // there would force a multi-token spelling and destabilize the prefix
    // cache). Inside a *permissive* free region — a JSON string body, an
    // `until()` raw value — no frame is ever legal, so the stricter
    // no-exemption `banned_specials_constrained` applies instead and
    // `<tool_call>` can no longer be committed as a real special id
    // inside an argument value.
    let banned = opts.banned_specials.as_slice();
    // Accept-then-check, same shape as everything else here: the region
    // query walks every active constraint, so it runs only once the
    // sampled id is known to be in the stricter set. Steady state pays
    // one binary search.
    let region_ban = !banned_in_region.is_empty()
        && banned_in_region.binary_search(&chosen).is_ok()
        && match region::ConstraintGuard::build(
            &opts.modes,
            &state.matchers,
            state.deferred.as_ref(),
            opts.deferred_grammar.as_ref(),
            model,
        ) {
            // `build` is `Some` only when every active constraint is
            // incomplete AND permissive — precisely where frames are
            // illegal. One exemption: a token whose bytes *leave* the
            // region is that region's exit delimiter, not content, so it
            // stays emittable even when it is a special (Harmony's
            // `<|end|>`). Banning it would make the constraint
            // uncompletable.
            Some(guard) => !guard.is_protected(chosen),
            // Structural position, or no live constraint at all: the
            // standing ban governs, unchanged.
            None => false,
        };
    let banned = if region_ban { banned_in_region } else { banned };
    if region_ban
        || (!banned.is_empty() && banned.binary_search(&chosen).is_ok())
    {
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
                    MatcherState::Grammar { stack, .. },
                ) => {
                    grammar::grammar_filter(candidates, compiled, stack, model)
                }
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

    // Ids 0–7 are stable (several tests hardcode them); new pieces are
    // append-only. 8+ serve the constrained-repetition battery: a bare
    // quote and the merged close `",` — the token shape the bare-char
    // ignore list can never cover.
    const PIECES: &[&str] =
        &["", "a", "b", "c", "x", "", "a", "b", "\"", "\","];
    const EOS: Token = 0;
    const A: Token = 1;
    const B: Token = 2;
    const C: Token = 3;
    const X: Token = 4;
    const QUOTE: Token = 8;
    const QUOTE_COMMA: Token = 9;
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
            2
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
        fn recommended_sampling(&self) -> crate::SamplingParams {
            crate::SamplingParams::default()
        }
    }

    /// `root ::= "ab"` — accepts exactly the string "ab".
    const AB_GRAMMAR: &str = r#"root ::= "ab""#;

    /// A quoted string closed by the merged token `",` — the free-region
    /// island shape for the constrained-repetition battery.
    const STR_GRAMMAR: &str = r#"root ::= "\"" [^"]* "\",""#;

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

    /// Build a `SamplingParams` from raw scalars, panicking on an
    /// out-of-range probability. Test-only sugar — production code
    /// goes through the fallible `Probability::from_f`.
    fn params(
        temp: Option<f32>,
        top_p: Option<f64>,
        top_k: Option<usize>,
    ) -> SamplingParams {
        SamplingParams {
            temp,
            top_p: top_p.map(|p| Probability::from_f(p).expect("in range")),
            top_k: top_k.map(|k| NonZeroUsize::new(k).expect("nonzero")),
            min_p: None,
            mirostat: None,
        }
    }

    /// `SamplingParams` compiles to the canonical `llama.cpp` order:
    /// truncation before rescaling, so temperature is last.
    #[test]
    fn params_into_modes_canonical_order() {
        let modes: Vec<SamplingMode> =
            params(Some(0.7), Some(0.9), Some(20)).into();
        assert_eq!(
            modes,
            vec![
                SamplingMode::TopK {
                    k: NonZeroUsize::new(20).unwrap()
                },
                SamplingMode::TopP {
                    p: Probability::from_f(0.9).unwrap(),
                    min_keep: NonZeroUsize::new(1).unwrap(),
                },
                SamplingMode::Temperature { t: 0.7 },
            ],
            "top-k then top-p then temperature"
        );
    }

    /// Absent members are skipped rather than defaulted — a request
    /// that sets only `temperature` must not silently acquire a
    /// top-k it never asked for.
    #[test]
    fn params_into_modes_skips_none() {
        let modes: Vec<SamplingMode> = params(Some(1.2), None, None).into();
        assert_eq!(modes, vec![SamplingMode::Temperature { t: 1.2 }]);

        let empty: Vec<SamplingMode> = SamplingParams::default().into();
        assert!(empty.is_empty(), "nothing specified => nothing emitted");
        assert!(SamplingParams::default().is_empty());
    }

    /// Mirostat is terminal — it yields a single token, after which
    /// every later mode is a silent no-op. Emitting it alongside the
    /// truncation knobs would look composed while doing nothing, so
    /// it is emitted alone.
    #[test]
    fn params_into_modes_mirostat_is_alone() {
        let mut p = params(Some(0.7), Some(0.9), Some(20));
        p.mirostat = Some(Mirostat::V2 { tau: 5.0, eta: 0.1 });
        let modes: Vec<SamplingMode> = p.into();
        assert_eq!(
            modes,
            vec![SamplingMode::MirostatV2 {
                tau: 5.0,
                eta: 0.1,
                max_keep: None
            }],
            "mirostat suppresses the truncation chain entirely"
        );
    }

    /// The crate default chain, the realistic patch target.
    fn default_chain() -> Vec<SamplingMode> {
        SamplerConfig::default().modes
    }

    /// The chain a metadata-seeded sidecar produces (Qwen3.6's).
    fn seeded_chain() -> Vec<SamplingMode> {
        params(Some(1.0), Some(0.95), Some(20)).into()
    }

    /// A knob that appears exactly once is retuned in place, and
    /// everything else in the chain survives. This is the case that
    /// makes `top_k` tweakable without blowing away `LocallyTypical`.
    #[test]
    fn request_patches_unambiguous_knob_in_place() {
        let got = apply_request_sampling(
            default_chain(),
            params(None, None, Some(7)),
            SamplingParams::default(),
        );
        assert_eq!(
            got,
            vec![
                SamplingMode::TopK {
                    k: NonZeroUsize::new(7).unwrap()
                },
                SamplingMode::locally_typical(),
            ],
            "top-k retuned, locally-typical untouched"
        );
    }

    /// A knob with no slot in the chain forces a rebuild — there is
    /// nowhere to put it, and inserting at a guessed position would
    /// make behavior depend on sidecar contents the client can't see.
    #[test]
    fn request_rebuilds_when_knob_absent() {
        // The default chain has no TopP.
        let got = apply_request_sampling(
            default_chain(),
            params(None, Some(0.8), None),
            SamplingParams::default(),
        );
        assert_eq!(
            got,
            vec![SamplingMode::TopP {
                p: Probability::from_f(0.8).unwrap(),
                min_keep: NonZeroUsize::new(1).unwrap(),
            }],
            "LocallyTypical is discarded on rebuild"
        );
    }

    /// Unspecified knobs fall back to the model's recommendation, not
    /// to the crate default — a client setting only `temperature`
    /// still gets the model's own top-k / top-p.
    #[test]
    fn request_rebuild_falls_back_to_model_metadata() {
        let got = apply_request_sampling(
            default_chain(),
            params(None, Some(0.8), None),
            params(Some(1.0), Some(0.95), Some(20)),
        );
        assert_eq!(
            got,
            vec![
                SamplingMode::TopK {
                    k: NonZeroUsize::new(20).unwrap()
                },
                SamplingMode::TopP {
                    p: Probability::from_f(0.8).unwrap(),
                    min_keep: NonZeroUsize::new(1).unwrap(),
                },
                SamplingMode::Temperature { t: 1.0 },
            ],
            "requested top_p wins; top_k and temp come from the model"
        );
    }

    /// Mixed presence is resolved over the whole requested set, not
    /// per-parameter: one absent knob rebuilds the entire chain.
    /// Patching `TopP` while appending `Temperature` would leave
    /// `LocallyTypical` at an arbitrary position — neither behavior.
    #[test]
    fn request_mixed_presence_rebuilds_whole_chain() {
        let mut chain = default_chain();
        chain.insert(
            1,
            SamplingMode::TopP {
                p: Probability::from_f(0.9).unwrap(),
                min_keep: NonZeroUsize::new(1).unwrap(),
            },
        );
        // top_p is present once, temperature is absent.
        let got = apply_request_sampling(
            chain,
            params(Some(0.5), Some(0.8), None),
            SamplingParams::default(),
        );
        assert!(
            !got.contains(&SamplingMode::locally_typical()),
            "one absent knob rebuilds everything: {got:?}"
        );
        assert_eq!(
            got,
            vec![
                SamplingMode::TopP {
                    p: Probability::from_f(0.8).unwrap(),
                    min_keep: NonZeroUsize::new(1).unwrap(),
                },
                SamplingMode::Temperature { t: 0.5 },
            ]
        );
    }

    /// A duplicated knob is ambiguous — there is no way to tell which
    /// slot the client meant — so it rebuilds rather than guessing.
    #[test]
    fn request_duplicate_knob_rebuilds() {
        let chain = vec![
            SamplingMode::TopK {
                k: NonZeroUsize::new(100).unwrap(),
            },
            SamplingMode::locally_typical(),
            SamplingMode::TopK {
                k: NonZeroUsize::new(40).unwrap(),
            },
        ];
        let got = apply_request_sampling(
            chain,
            params(None, None, Some(7)),
            SamplingParams::default(),
        );
        assert_eq!(
            got,
            vec![SamplingMode::TopK {
                k: NonZeroUsize::new(7).unwrap()
            }],
            "two top-k slots => rebuild, not a guess"
        );
    }

    /// Constraint modes are not sampling knobs: a grammar survives a
    /// rebuild, and stays at the front where it prunes candidates
    /// before the truncation samplers run. Losing this would let a
    /// client's `temperature` silently disable a tool-call grammar.
    #[test]
    fn request_rebuild_preserves_constraints() {
        let grammar = SamplingMode::grammar("root ::= .+").unwrap();
        let mut chain = vec![grammar.clone()];
        chain.extend(default_chain());

        let got = apply_request_sampling(
            chain,
            params(None, Some(0.8), None),
            SamplingParams::default(),
        );
        assert_eq!(got.first(), Some(&grammar), "grammar must lead: {got:?}");
        assert_eq!(got.len(), 2, "grammar + rebuilt top-p only");
    }

    /// An empty request is a no-op. Most calls set no sampling fields
    /// at all, and they must get exactly the configured chain.
    #[test]
    fn request_empty_leaves_chain_alone() {
        let got = apply_request_sampling(
            seeded_chain(),
            SamplingParams::default(),
            params(Some(1.0), Some(0.95), Some(20)),
        );
        assert_eq!(got, seeded_chain());
    }

    /// Against a metadata-seeded sidecar every knob has exactly one
    /// slot, so the common case is a pure in-place retune — patch and
    /// rebuild agree here, which is why patch-first is safe.
    #[test]
    fn request_against_seeded_chain_patches_all() {
        let got = apply_request_sampling(
            seeded_chain(),
            params(Some(0.0), Some(0.5), Some(3)),
            SamplingParams::default(),
        );
        assert_eq!(
            got,
            vec![
                SamplingMode::TopK {
                    k: NonZeroUsize::new(3).unwrap()
                },
                SamplingMode::TopP {
                    p: Probability::from_f(0.5).unwrap(),
                    min_keep: NonZeroUsize::new(1).unwrap(),
                },
                // 0.0 reaches the mode intact; `Candidates::temperature`
                // is what turns it into argmax.
                SamplingMode::Temperature { t: 0.0 },
            ]
        );
    }

    /// A model that recommends mirostat must not have it swallow the
    /// knobs a client explicitly asked for — mirostat compiles to a
    /// chain of one, so everything after it would be a silent no-op.
    #[test]
    fn request_drops_fallback_mirostat() {
        let fallback = SamplingParams {
            mirostat: Some(Mirostat::V2 { tau: 5.0, eta: 0.1 }),
            ..params(Some(1.0), None, Some(20))
        };
        let got = apply_request_sampling(
            vec![SamplingMode::locally_typical()],
            params(None, Some(0.8), None),
            fallback,
        );
        assert!(
            !got.iter()
                .any(|m| matches!(m, SamplingMode::MirostatV2 { .. })),
            "fallback mirostat must not eat the requested top_p: {got:?}"
        );
        assert!(got.contains(&SamplingMode::TopP {
            p: Probability::from_f(0.8).unwrap(),
            min_keep: NonZeroUsize::new(1).unwrap(),
        }));
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
            (
                SamplingMode::Grammar(compiled),
                MatcherState::Grammar { stack, .. },
            ) => stack.accepts_bytes(&compiled.grammar, bytes),
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
            ..SamplerConfig::default()
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
            ..SamplerConfig::default()
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

    // ── Region-scoped emit ban (#37) ─────────────────────────────────

    /// Grammar + greedy, with a region-scoped ban set and (by default)
    /// an empty standing set — the shape `Session` produces for a frame
    /// marker: exempt from the standing ban so the frame can be emitted,
    /// present in the stricter set so it cannot become argument content.
    fn region_ban_opts(
        standing: Vec<Token>,
        in_region: Vec<Token>,
    ) -> SamplerConfig {
        SamplerConfig {
            modes: vec![
                SamplingMode::grammar(STR_GRAMMAR)
                    .expect("test grammar parses"),
                SamplingMode::Greedy,
            ],
            repetition: None,
            deferred_grammar: None,
            lazy_grammar: false,
            banned_specials: standing,
            banned_specials_constrained: in_region,
        }
    }

    /// A state parked mid-string-body — the permissive free region.
    fn in_string_body(opts: &SamplerConfig) -> SamplerState {
        let mut state = state_for(opts);
        state.advance(opts, QUOTE, &MockModel);
        state.advance(opts, A, &MockModel);
        state
    }

    /// The headline (#37): a token exempt from the standing ban — a
    /// dialect frame marker — is masked inside a permissive region,
    /// where it would be argument *content* rather than framing.
    #[test]
    fn region_ban_masks_frame_special_inside_free_region() {
        let opts = region_ban_opts(vec![], vec![X]);
        let mut state = in_string_body(&opts);
        let picked = sample_token(
            &[QUOTE, A],
            cands(&[(X, 10.0), (B, 5.0)]),
            &opts,
            &mut state,
            &MockModel,
        )
        .unwrap();
        assert_eq!(
            picked, B,
            "frame special must be masked inside the free region"
        );
    }

    /// The load-bearing negative: with no live constraint there is no
    /// region, so the stricter set is inert and the standing ban alone
    /// governs. This is what keeps frame literals emittable as single
    /// tokens — banning them at frame positions would force a
    /// multi-token spelling and destabilize the prefix cache.
    #[test]
    fn region_ban_inert_outside_a_constraint() {
        let opts = SamplerConfig {
            modes: vec![SamplingMode::Greedy],
            repetition: None,
            deferred_grammar: None,
            lazy_grammar: false,
            banned_specials: vec![],
            banned_specials_constrained: vec![X],
        };
        let mut state = state_for(&opts);
        let picked = sample_token(
            &[],
            cands(&[(X, 10.0), (A, 5.0)]),
            &opts,
            &mut state,
            &MockModel,
        )
        .unwrap();
        assert_eq!(
            picked, X,
            "the stricter set must not apply without a permissive region"
        );
    }

    /// Structural positions inside a live grammar are not free regions
    /// either: at the root only `"` is legal, so `ConstraintGuard::build`
    /// vetoes and the stricter set stays inert.
    #[test]
    fn region_ban_inert_at_structural_position() {
        let opts = region_ban_opts(vec![], vec![QUOTE]);
        let mut state = state_for(&opts);
        let picked = sample_token(
            &[],
            cands(&[(QUOTE, 10.0), (A, 5.0)]),
            &opts,
            &mut state,
            &MockModel,
        )
        .unwrap();
        assert_eq!(
            picked, QUOTE,
            "structural position must fall back to the standing ban"
        );
    }

    /// The exit exemption: a token whose bytes *leave* the region is
    /// that region's delimiter, not content, so it stays emittable even
    /// when listed. Without this a dialect whose exit marker is itself a
    /// special (Harmony's `<|end|>`) would have an uncompletable
    /// constraint. Covers the merged form (`",`) the walk exists for.
    #[test]
    fn region_ban_exempts_region_exit_tokens() {
        let opts = region_ban_opts(vec![], vec![QUOTE_COMMA]);
        let mut state = in_string_body(&opts);
        let picked = sample_token(
            &[QUOTE, A],
            cands(&[(QUOTE_COMMA, 10.0), (A, 5.0)]),
            &opts,
            &mut state,
            &MockModel,
        )
        .unwrap();
        assert_eq!(
            picked, QUOTE_COMMA,
            "the region's own exit delimiter must stay emittable"
        );
    }

    /// Selecting the stricter set must not drop a standing ban: a token
    /// banned everywhere stays banned inside the region. (`Session`
    /// unions the two for exactly this reason; here the standing branch
    /// carries it.)
    #[test]
    fn region_ban_does_not_weaken_the_standing_ban() {
        let opts = region_ban_opts(vec![B], vec![X]);
        let mut state = in_string_body(&opts);
        let picked = sample_token(
            &[QUOTE, A],
            cands(&[(B, 10.0), (C, 5.0)]),
            &opts,
            &mut state,
            &MockModel,
        )
        .unwrap();
        assert_eq!(picked, C, "standing ban must survive inside a region");
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

    /// The reconcile-by-grammar-identity load rule
    /// ([`SamplerState::resumed_from`]): same grammar carries the
    /// matcher position; a different grammar resets ONLY the matcher
    /// (to the new grammar's root); `mu`/rng/ngram stats carry
    /// unconditionally; a longer modes vec cannot OOB (fresh
    /// index-aligned build).
    #[test]
    fn resumed_from_reconciles_by_grammar_identity() {
        let opts = opts_with_grammar(false);
        let mut cached = state_for(&opts);
        // Advance mid-grammar (one step into "ab") and give the
        // stream distinguishable content.
        let tok = sample(
            cands(&[(A, 20.0), (X, 0.0), (EOS, -20.0)]),
            &opts,
            &mut cached,
        );
        assert_eq!(tok, A);
        cached.mu = Some(3.5);
        cached.ngram_stats.add(crate::NGram::from(A), 1);

        // Same grammar: position carries — "b" completes, "a" no
        // longer accepted.
        let resumed = SamplerState::resumed_from(&cached, &opts, &MockModel);
        assert_eq!(resumed.matchers, cached.matchers);
        assert_eq!(resumed.mu, cached.mu);
        assert_eq!(resumed.rng, cached.rng);
        assert_eq!(resumed.ngram_stats, cached.ngram_stats);
        assert!(grammar_accepts(&opts, &resumed, b"b"));
        assert!(!grammar_accepts(&opts, &resumed, b"a"));

        // Different grammar: matcher at the NEW grammar's root; the
        // stream still carries.
        let other = SamplerConfig {
            modes: vec![SamplingMode::grammar(r#"root ::= "ba""#).unwrap()],
            ..opts.clone()
        };
        let resumed = SamplerState::resumed_from(&cached, &other, &MockModel);
        assert!(grammar_accepts(&other, &resumed, b"b"), "root of \"ba\"");
        assert!(!grammar_accepts(&other, &resumed, b"a"));
        assert_eq!(resumed.rng, cached.rng);
        assert_eq!(resumed.ngram_stats, cached.ngram_stats);

        // Longer modes vec (run_call prepends call-derived modes):
        // fresh index-aligned build, matched grammar still carries.
        let longer = SamplerConfig {
            modes: vec![opts.modes[0].clone(), SamplingMode::Greedy],
            ..opts.clone()
        };
        let resumed = SamplerState::resumed_from(&cached, &longer, &MockModel);
        assert_eq!(resumed.matchers.len(), 2);
        assert!(grammar_accepts(&longer, &resumed, b"b"), "carried position");
        assert!(matches!(
            resumed.matchers[1],
            crate::sample::state::MatcherState::Stateless
        ));

        // Shrunk to no constraints at all: nothing to reconcile, no
        // panic, stream carries.
        let bare = SamplerConfig {
            modes: vec![SamplingMode::Greedy],
            ..opts.clone()
        };
        let resumed = SamplerState::resumed_from(&cached, &bare, &MockModel);
        assert_eq!(resumed.matchers.len(), 1);
        assert_eq!(resumed.ngram_stats, cached.ngram_stats);
    }

    /// `resumed_from` deferred-matcher reconcile: same spec carries the
    /// activation flag + position; a different deferred grammar gets a
    /// fresh inactive root. `resolved_ignored` is recomputed from the
    /// effective config, not carried.
    #[test]
    fn resumed_from_deferred_and_resolved_ignored() {
        let deferred = crate::DeferredGrammar {
            grammar: CompiledGrammar::parse(AB_GRAMMAR).unwrap(),
            activate_after: vec![b"!".to_vec()],
            feed_trigger: false,
        };
        let opts = SamplerConfig {
            modes: vec![SamplingMode::Greedy],
            deferred_grammar: Some(deferred.clone()),
            repetition: None,
            ..SamplerConfig::default()
        };
        let mut cached = state_for(&opts);
        cached.activate_deferred(&deferred, b"a").unwrap();
        // Stale memo that a recompute must drop (repetition is None ⇒
        // resolved_ignored must come back empty).
        cached.resolved_ignored.insert(crate::NGram::from(X));

        // Same spec: activation + position carry.
        let resumed = SamplerState::resumed_from(&cached, &opts, &MockModel);
        assert_eq!(resumed.deferred, cached.deferred);
        assert!(resumed.deferred.as_ref().unwrap().active);
        assert!(
            resumed.resolved_ignored.is_empty(),
            "resolved_ignored is a config memo — recomputed, not carried",
        );

        // Different deferred grammar: fresh inactive root.
        let other = SamplerConfig {
            deferred_grammar: Some(crate::DeferredGrammar {
                grammar: CompiledGrammar::parse(r#"root ::= "ba""#).unwrap(),
                ..deferred
            }),
            ..opts.clone()
        };
        let resumed = SamplerState::resumed_from(&cached, &other, &MockModel);
        assert!(!resumed.deferred.as_ref().unwrap().active);

        // No deferred in the new config: none in the state.
        let none = SamplerConfig {
            deferred_grammar: None,
            ..opts.clone()
        };
        let resumed = SamplerState::resumed_from(&cached, &none, &MockModel);
        assert!(resumed.deferred.is_none());
    }

    /// #38 defect 1: an **activated** deferred grammar left
    /// mid-structure at end of generation is a violation, exactly like
    /// an incomplete eager one. One that never triggered is not —
    /// declining to call a tool is legal on the Auto path.
    ///
    /// Before the fix this returned `false` for the mid-structure case,
    /// so a truncated Auto call came back as `Ok` and its `<tool_call>`
    /// frame marker was seated as plain `Block::Text` — poison for the
    /// next ingest.
    #[test]
    fn constraint_incomplete_at_end_sees_activated_deferred() {
        let deferred = crate::DeferredGrammar {
            grammar: CompiledGrammar::parse(AB_GRAMMAR).unwrap(),
            activate_after: vec![b"!".to_vec()],
            feed_trigger: false,
        };
        let opts = SamplerConfig {
            modes: vec![SamplingMode::Greedy],
            deferred_grammar: Some(deferred.clone()),
            repetition: None,
            ..SamplerConfig::default()
        };

        // Configured but never triggered: the model just didn't call.
        assert!(
            !state_for(&opts).constraint_incomplete_at_end(),
            "a deferred grammar that never triggered is not a violation",
        );

        // Activated and mid-structure (`root ::= \"ab\"` fed only
        // \"a\") — the truncated-call shape.
        let mut mid = state_for(&opts);
        mid.activate_deferred(&deferred, b"a").unwrap();
        assert!(
            mid.constraint_incomplete_at_end(),
            "an activated deferred grammar left mid-structure must flag",
        );

        // Activated and complete: a clean turn.
        let mut done = state_for(&opts);
        done.activate_deferred(&deferred, b"ab").unwrap();
        assert!(
            !done.constraint_incomplete_at_end(),
            "a completed deferred grammar is not a violation",
        );

        // The activation flag survives a cache resume, so a turn that
        // resumes onto a live matcher is judged the same way.
        let resumed = SamplerState::resumed_from(&mid, &opts, &MockModel);
        assert!(resumed.deferred.as_ref().unwrap().active);
        assert!(
            resumed.constraint_incomplete_at_end(),
            "resumed activated matcher must be judged like a fresh one",
        );
    }

    /// The eager half is unchanged by the deferred clause: an eager
    /// grammar that never reached accept still flags, and a config with
    /// no byte-constraint at all never does (plain prose generation
    /// must not be reported as a violation).
    #[test]
    fn constraint_incomplete_at_end_eager_and_unconstrained() {
        let eager = SamplerConfig {
            modes: vec![SamplingMode::Grammar(
                CompiledGrammar::parse(AB_GRAMMAR).unwrap(),
            )],
            repetition: None,
            ..SamplerConfig::default()
        };
        assert!(
            state_for(&eager).constraint_incomplete_at_end(),
            "a fresh eager grammar has not reached accept",
        );

        let bare = SamplerConfig {
            modes: vec![SamplingMode::Greedy],
            repetition: None,
            ..SamplerConfig::default()
        };
        assert!(
            !state_for(&bare).constraint_incomplete_at_end(),
            "no constraint, no violation",
        );
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

    /// The constrained-region accumulator is call-local: `resumed_from`
    /// starts it empty (NOT cloned from the cached state) while the
    /// stream fields — rng, mu, persistent stats, step — carry. This is
    /// the determinism linchpin for cold-prefill ≡ resume equivalence.
    #[test]
    fn resumed_from_resets_constrained_ephemera() {
        let opts = opts_with_grammar(false);
        let mut cached = state_for(&opts);
        cached.ngram_stats.add(crate::NGram::from(A), 3);
        cached.step = 7;
        cached.constrained_ngram_stats.add(crate::NGram::from(B), 5);
        cached.constrained_step = 5;

        let resumed = SamplerState::resumed_from(&cached, &opts, &MockModel);
        assert_eq!(resumed.ngram_stats, cached.ngram_stats, "stream carries");
        assert_eq!(resumed.step, 7);
        assert_eq!(resumed.rng, cached.rng);
        assert_eq!(
            resumed.constrained_ngram_stats,
            crate::NGramStats::default(),
            "call-local accumulator must start empty on resume"
        );
        assert_eq!(resumed.constrained_step, 0);
    }

    /// A populated mid-call snapshot round-trips the constrained
    /// fields bit-exactly with canonical bytes. (Blobs from a binary
    /// predating the fields deserialize via `serde(default)` — worth
    /// the attribute for upgrade-within-cache-TTL skew, not worth a
    /// brittle canonical-tail test.)
    #[cfg(feature = "serde")]
    #[test]
    fn constrained_fields_serde_round_trip() {
        let opts = opts_with_grammar(false);
        let mut state = state_for(&opts);
        state.constrained_ngram_stats.add(crate::NGram::from(A), 2);
        state.constrained_step = 3;

        let blob = serde_json::to_string(&state).unwrap();
        let restored: SamplerState = serde_json::from_str(&blob).unwrap();
        assert_eq!(state, restored);
        assert_eq!(blob, serde_json::to_string(&restored).unwrap());
    }

    // ── Constrained-region repetition battery ────────────────────────

    /// Grammar + greedy, repetition tuned so a unigram loop flips the
    /// order within a few steps. `on` is the `constrained_regions`
    /// knob; `lazy` selects the fast path.
    fn str_opts(on: bool, lazy: bool) -> SamplerConfig {
        SamplerConfig {
            modes: vec![
                SamplingMode::grammar(STR_GRAMMAR)
                    .expect("test grammar parses"),
                SamplingMode::Greedy,
            ],
            repetition: Some(
                RepetitionOptions::default()
                    .set_ignored_categories(std::iter::empty())
                    .set_penalty_repeat(1.1)
                    .set_penalty_freq(0.5)
                    .set_penalty_present(0.5)
                    .set_constrained_regions(on),
            ),
            deferred_grammar: None,
            lazy_grammar: lazy,
            ..SamplerConfig::default()
        }
    }

    /// Dense candidates over the whole mock vocab (the penalty pass
    /// indexes `data[token_id]`, so every id must be present), with
    /// per-id overrides.
    fn dense(overrides: &[(Token, f32)]) -> Candidates {
        let mut logits = vec![-20.0f32; PIECES.len()];
        for &(id, logit) in overrides {
            logits[id as usize] = logit;
        }
        cands(
            &logits
                .iter()
                .enumerate()
                .map(|(id, &l)| (id as Token, l))
                .collect::<Vec<_>>(),
        )
    }

    /// Drive generation from just past the opening quote, greedy, with
    /// the content token `A` slightly above the merged close `",`.
    /// Returns (state, generated tokens, completed?).
    fn drive_string_island(
        opts: &SamplerConfig,
        max_steps: usize,
    ) -> (SamplerState, Vec<Token>, bool) {
        let mut state = state_for(opts);
        let mut tokens: Vec<Token> = vec![QUOTE];
        state.advance(opts, QUOTE, &MockModel);
        for _ in 0..max_steps {
            let c = dense(&[(A, 4.0), (QUOTE_COMMA, 3.9)]);
            let tok =
                sample_token(&tokens, c, opts, &mut state, &MockModel).unwrap();
            state.advance(opts, tok, &MockModel);
            tokens.push(tok);
            if state.grammar_complete() {
                return (state, tokens, true);
            }
        }
        (state, tokens, false)
    }

    /// The headline: with the feature ON, in-region repetition pressure
    /// flips the greedy order and the merged close `",` is emitted —
    /// the grammar completes. The call-local accumulator did the work;
    /// the persistent prose corpus is untouched. Post-completion, the
    /// persistent pass resumes.
    #[test]
    fn constrained_loop_breaks_and_closes() {
        let opts = str_opts(true, false);
        let (mut state, tokens, completed) = drive_string_island(&opts, 8);
        assert!(
            completed,
            "penalty must flip the order and close: {tokens:?}"
        );
        assert_eq!(*tokens.last().unwrap(), QUOTE_COMMA);
        assert!(
            state.constrained_step() > 0,
            "guarded pass must have executed"
        );
        assert_eq!(state.step(), 0, "prose corpus must not advance");
        assert_eq!(
            state.ngram_stats(),
            &crate::NGramStats::default(),
            "persistent stats must not see constrained tokens"
        );

        // Post-completion the constraint is complete ⇒ regime (a): the
        // persistent pass runs again (the complete grammar force-EOSes
        // the pick, which is irrelevant to the corpus bookkeeping).
        let _ = sample_token(
            &tokens,
            dense(&[(A, 4.0)]),
            &opts,
            &mut state,
            &MockModel,
        )
        .unwrap();
        assert_eq!(state.step(), 1, "prose pass resumes after completion");
        assert_ne!(state.ngram_stats(), &crate::NGramStats::default());
    }

    /// Counterfactual pinning the old failure: feature OFF restores the
    /// blanket suspension — the model loops on `A` for the whole budget
    /// and the grammar never completes.
    #[test]
    fn constrained_loop_off_never_closes() {
        let opts = str_opts(false, false);
        let (state, tokens, completed) = drive_string_island(&opts, 8);
        assert!(!completed, "without the feature the loop must persist");
        assert!(tokens[1..].iter().all(|&t| t == A), "{tokens:?}");
        assert_eq!(state.constrained_step(), 0);
        assert_eq!(
            state.constrained_ngram_stats(),
            &crate::NGramStats::default()
        );
    }

    /// Lazy/masked parity: greedy streams are identical and both
    /// complete. (Cross-path bit-equality holds here because greedy
    /// consumes no RNG.)
    #[test]
    fn constrained_loop_lazy_masked_parity() {
        let (_, masked_tokens, masked_done) =
            drive_string_island(&str_opts(true, false), 8);
        let (_, lazy_tokens, lazy_done) =
            drive_string_island(&str_opts(true, true), 8);
        assert!(masked_done && lazy_done);
        assert_eq!(masked_tokens, lazy_tokens);
    }

    /// The exit token is never penalized: even with heavy hand-seeded
    /// repetition of `",` in the call-local map, the guard protects it
    /// and it stays the greedy pick. A content token seeded the same
    /// way IS penalized — the contrast that proves the guard, not the
    /// tuning, made the difference.
    #[test]
    fn constrained_exit_token_never_penalized() {
        let opts = str_opts(true, false);

        let seeded = |seed_tok: Token| -> Token {
            let mut state = state_for(&opts);
            state.advance(&opts, QUOTE, &MockModel);
            for s in 0..5 {
                state
                    .constrained_ngram_stats
                    .add(crate::NGram::from(seed_tok), s);
            }
            state.constrained_step = 5;
            // The seeded token on top; a clean rival just below.
            let c = dense(&[(seed_tok, 4.0), (B, 3.9)]);
            sample_token(&[QUOTE], c, &opts, &mut state, &MockModel).unwrap()
        };

        // Protected exit: penalty skipped, stays on top.
        assert_eq!(seeded(QUOTE_COMMA), QUOTE_COMMA);
        // Unprotected content: penalized off the top.
        assert_eq!(seeded(A), B);
    }

    /// Structural states skip the guarded pass entirely (build returns
    /// None): counters stay zero and the pick matches the feature-off
    /// run — exactly the pre-feature suspension.
    #[test]
    fn constrained_structural_state_skips() {
        let run = |on: bool| -> (Token, SamplerState) {
            let opts = SamplerConfig {
                modes: vec![
                    SamplingMode::grammar(AB_GRAMMAR)
                        .expect("test grammar parses"),
                    SamplingMode::Greedy,
                ],
                repetition: Some(
                    RepetitionOptions::default()
                        .set_ignored_categories(std::iter::empty())
                        .set_constrained_regions(on),
                ),
                deferred_grammar: None,
                lazy_grammar: false,
                ..SamplerConfig::default()
            };
            let mut state = state_for(&opts);
            let tok = sample_token(
                &[],
                dense(&[(A, 4.0), (B, 3.0)]),
                &opts,
                &mut state,
                &MockModel,
            )
            .unwrap();
            (tok, state)
        };
        let (on_tok, on_state) = run(true);
        let (off_tok, _) = run(false);
        assert_eq!(on_tok, off_tok);
        assert_eq!(on_tok, A, "grammar wants 'a' first");
        assert_eq!(on_state.constrained_step(), 0);
        assert_eq!(
            on_state.constrained_ngram_stats(),
            &crate::NGramStats::default()
        );
        assert_eq!(on_state.step(), 0);
    }

    /// Unconstrained generation is bit-identical with the knob on or
    /// off — the feature must be invisible outside grammars.
    #[test]
    fn unconstrained_stream_unchanged_by_knob() {
        let run = |on: bool| -> (Vec<Token>, SamplerState) {
            let opts = SamplerConfig {
                modes: vec![],
                repetition: Some(
                    RepetitionOptions::default()
                        .set_ignored_categories(std::iter::empty())
                        .set_constrained_regions(on),
                ),
                deferred_grammar: None,
                lazy_grammar: true,
                ..SamplerConfig::default()
            };
            let mut state = state_for(&opts);
            let mut tokens: Vec<Token> = Vec::new();
            for _ in 0..16 {
                let c = dense(&[(A, 2.0), (B, 1.9), (C, 1.8), (X, 1.7)]);
                let tok =
                    sample_token(&tokens, c, &opts, &mut state, &MockModel)
                        .unwrap();
                state.advance(&opts, tok, &MockModel);
                tokens.push(tok);
            }
            (tokens, state)
        };
        let (on_tokens, on_state) = run(true);
        let (off_tokens, off_state) = run(false);
        assert_eq!(on_tokens, off_tokens);
        assert_eq!(on_state, off_state);
    }
}
