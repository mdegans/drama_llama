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

#[cfg(feature = "egui")]
const DELETE_ICON: egui::ImageSource<'static> =
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
#[derive(Clone, Debug, PartialEq)]
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

    /// The name of the sampling mode.
    pub const fn name(&self) -> &'static str {
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
        }
    }

    /// A help message for the sampling mode (but not it's parameters)
    pub const fn help(&self) -> &'static str {
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

#[derive(Debug, thiserror::Error, derive_more::From)]
pub enum SampleError {
    #[error("Sampling failed because of a repetition error: {err}")]
    RepetitionError { err: RepetitionError },
}

static_assertions::assert_impl_all!(SampleError: Send, Sync);

#[cfg_attr(
    feature = "serde",
    derive(rocket::serde::Deserialize, rocket::serde::Serialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "rocket::serde"))]
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
/// This is mostly a translation of the C++ code in `llama.cpp` with support for
/// n-grams. [`NGramStats`] is used to store the n-gram statistics.
pub fn apply_sample_repetition_ngram(
    candidates: Candidates,
    tokens: &[llama_token],
    opts: &mut RepetitionOptions,
    freq_map: &mut NGramStats,
    model: &crate::Model,
) -> Result<Candidates, RepetitionError> {
    let k = candidates.len();
    let mut candidates = candidates.sort(Sorted::ById { k });

    let RepetitionOptions {
        ignored_stopwords,
        ignored,
        ngram_max_size,
        ngram_min_size,
        penalty_freq,
        penalty_max_count,
        penalty_present,
        // NOTE: This makes a copy of the value, so we can modify it. It doesn't
        // mutate the value on the struct. RepetitionOptions does need to be
        // mutable, but only for the stopwords and there might be a better way
        // to handle that.
        mut penalty_repeat,
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

        candidate.logit -= (data.count() as f32) * *penalty_freq
            + (data.count() as f32) * *penalty_present;

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
