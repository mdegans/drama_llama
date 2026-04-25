use std::num::{NonZeroU128, NonZeroUsize};

use xorshift::{SeedableRng, Xoroshiro128};

use crate::{
    backend::{Backend, Decoder, Model},
    ngram::NGramStats,
    sample::SampleOptions,
    Candidates, Engine, NGram, Token,
};

#[cfg(feature = "serde")]
fn deserialize_regex_vec<'de, D>(
    deserializer: D,
) -> Result<Vec<regex::Regex>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Deserialize;

    let strings = Vec::<String>::deserialize(deserializer)?;
    strings
        .into_iter()
        .map(|s| regex::Regex::new(&s).map_err(serde::de::Error::custom))
        .collect()
}

#[cfg(feature = "serde")]
fn serialize_regex_vec<S>(
    regexes: &Vec<regex::Regex>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use serde::ser::SerializeSeq;

    let mut seq = serializer.serialize_seq(Some(regexes.len()))?;
    for regex in regexes {
        seq.serialize_element(&regex.as_str())?;
    }
    seq.end()
}

/// Options for prediction.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct PredictOptions {
    /// Maximum number of tokens to predict.
    pub n: NonZeroUsize,
    /// Random seed. If this is `Some`, the prediction will be deterministic.
    /// Otherwise the seed will be based on the current time.
    pub seed: Option<NonZeroU128>,
    /// Stop sequences by token. When any of these are reached, the prediction
    /// will stop.
    pub stop_sequences: Vec<Vec<Token>>,
    /// Stop sequences by string. When any of these are reached, the prediction
    /// will stop.
    pub stop_strings: Vec<String>,
    /// Regex stop sequences. When any of these are reached, the prediction will
    /// stop.
    #[cfg_attr(
        feature = "serde",
        serde(deserialize_with = "deserialize_regex_vec")
    )]
    #[cfg_attr(
        feature = "serde",
        serde(serialize_with = "serialize_regex_vec")
    )]
    pub regex_stop_sequences: Vec<regex::Regex>,
    /// Sampling options.
    pub sample_options: SampleOptions,
}

impl Default for PredictOptions {
    fn default() -> Self {
        Self {
            n: NonZeroUsize::new(512).unwrap(),
            seed: Some(Self::DEFAULT_SEED),
            stop_sequences: Vec::new(),
            stop_strings: Vec::new(),
            regex_stop_sequences: Vec::new(),
            sample_options: SampleOptions::default(),
        }
    }
}

impl PredictOptions {
    pub const DEFAULT_SEED: NonZeroU128 = match NonZeroU128::new(1337) {
        Some(seed) => seed,
        None => panic!("Bad seed."),
    };

    /// Shortcut for greedy sampling/
    pub fn greedy() -> Self {
        Self {
            sample_options: SampleOptions::greedy(),
            ..Self::default()
        }
    }

    /// Add the model's end-of-sequence and end-of-turn tokens as stop
    /// sequences.
    ///
    /// Both EOS and EOT are added when they're distinct so chat-tuned
    /// models that emit `<|eot_id|>` between turns terminate cleanly.
    /// Repetition penalty ignores these — otherwise a strong penalty can
    /// keep the model from ever closing a turn.
    pub fn add_model_stops<M: Model>(mut self, model: &M) -> Self {
        let eos = model.eos();
        self.stop_sequences.push(vec![eos]);
        if let Some(opts) = &mut self.sample_options.repetition {
            opts.ignored.insert(eos.into());
        }
        let eot = model.eot();
        if eot != eos && eot >= 0 {
            self.stop_sequences.push(vec![eot]);
            if let Some(opts) = &mut self.sample_options.repetition {
                opts.ignored.insert(eot.into());
            }
        }
        self
    }

    /// Add a stop sequence of tokens. If the [`Predictor`] reaches any of these
    /// sequences, it will stop predicting. The stop sequence will be included
    /// in the tokens.
    pub fn add_stop_sequence(mut self, sequence: Vec<Token>) -> Self {
        self.stop_sequences.push(sequence);

        self
    }

    /// Add a stop sequence by string. If the [`Predictor`] reaches any of these
    /// sequences, it will stop predicting. The stop sequence will be included
    /// in the text.
    pub fn add_stop(mut self, s: String) -> Self {
        self.stop_strings.push(s);

        self
    }

    /// Add a stop sequence by regex. If the [`Predictor`] reaches any of these
    /// sequences, it will stop predicting once the regex matches the text.
    pub fn add_stop_regex(mut self, regex: regex::Regex) -> Self {
        self.regex_stop_sequences.push(regex);

        self
    }

    /// Draw [`egui::Ui`] for the options.
    ///
    /// `max_context_size` caps the ui widget for the number of tokens to
    /// predict. It should be set to the maximum context size of the model
    /// minus the number of tokens in the prompt.
    #[cfg(feature = "egui")]
    pub fn draw(
        &mut self,
        ui: &mut egui::Ui,
        max_context_size: usize,
    ) -> egui::Response {
        let resp = egui::CollapsingHeader::new("Predict Options")
            .default_open(true)
            .show(ui, |ui| self.draw_inner(ui, max_context_size));

        let header_response = resp
            .header_response
            .on_hover_text_at_pointer("Options for `drama_llama` prediction.");

        resp.body_response.unwrap_or(header_response)
    }

    /// Draw [`egui::Ui`] for the options, but without the
    /// [`egui::CollapsingHeader`].
    ///
    /// `max_context_size` caps the ui widget for the number of tokens to
    /// predict. It should be set to the maximum context size of the model
    /// minus the number of tokens in the prompt.
    #[cfg(feature = "egui")]
    pub fn draw_inner(
        &mut self,
        ui: &mut egui::Ui,
        max_context_size: usize,
    ) -> egui::Response {
        egui_extras::install_image_loaders(ui.ctx());

        let mut resp = ui.label("Number of tokens to predict");
        let mut n = self.n.get();
        resp |= ui.add(
            egui::DragValue::new(&mut n)
                .speed(1.0)
                .clamp_range(1..=max_context_size),
        );
        // The max is because it's possible to drag the value to 0 even
        // though it's supposed to clamp. This may be a bug in egui or
        // I am holding it wrong - mdegans
        self.n = NonZeroUsize::new(n.max(1)).unwrap();

        resp |= ui.label("Random seed");
        let mut is_random = self.seed.is_none();
        resp |= ui.horizontal(|ui| {
                    ui.checkbox(&mut is_random, "Random")
                        .on_hover_text_at_pointer("If checked, the seed will be based on the current time. This is recommended unless you want deterministic results. Note that truly deterministic results are not guaranteed, especially across platforms.");
                    if !is_random {
                        // This isn't good, but egui doesn't support u128 yet.
                        let mut seed = self.seed.map(|s| s.get()).unwrap_or(1337) as usize;
                        ui.add(
                            egui::DragValue::new(&mut seed)
                                .speed(1.0)
                                .clamp_range(1..=usize::MAX),
                        );
                        self.seed = NonZeroU128::new(seed.max(1) as u128);
                    } else {
                        self.seed = None;
                    }
                }).response;

        if !self.stop_sequences.is_empty() {
            resp |= ui.label("Stop token sequences").on_hover_text_at_pointer("Note that these are not currently directly editable via the UI, however the JSON storage for egui does support editing this. A UI will be added in the future.");
            resp |= ui
                .vertical(|ui| {
                    for sequence in self.stop_sequences.iter() {
                        ui.label(format!("{:?}", sequence));
                    }
                })
                .response;
        }

        // FIXME: there is necessarily a way to do this in egui, but I
        // can't find it right now. This is a temporary solution.
        if !self.stop_strings.is_empty() {
            resp |= ui.label("Stop strings").on_hover_text_at_pointer("When any of these strings are found in the text, the prediction will stop. Note that `egui` escapes special characters, so you may need to edit the JSON directly to add a string with special characters. This will be fixed in the future.");
            resp |= ui
                .horizontal(|ui| {
                    for string in self.stop_strings.iter() {
                        ui.label(string);
                    }
                })
                .response;
        }

        if !self.regex_stop_sequences.is_empty() {
            resp |= ui.label("Regex stop sequences").on_hover_text_at_pointer("When any of these regexes match the text, the prediction will stop. Note that `egui` escapes special characters, so you may need to edit the JSON directly to add a regex with special characters. This will be fixed in the future.");
            resp |= ui
                .horizontal(|ui| {
                    ui.label("Regexes");
                    for regex in self.regex_stop_sequences.iter() {
                        ui.label(regex.as_str());
                    }
                })
                .response;
        }

        resp |= self.sample_options.draw(ui);

        resp
    }
}

/// An iterator that predicts a sequence of candidate distributions.
///
/// Generic over the decoder `D` and model `M`. Construction (`new` or
/// `new_resuming`) runs the initial prefill via [`Decoder::prefill`];
/// subsequent steps use [`Decoder::step`] after each
/// [`CandidatePredictor::record_choice`].
pub struct CandidatePredictor<'engine, B: Backend> {
    /// The inference engine.
    pub engine: &'engine mut Engine<B>,
    /// The tokens seen so far (prompt + any recorded choices).
    pub tokens: Vec<Token>,
    /// First-step candidates captured from the initial prefill —
    /// yielded on the first `next()` call, then taken.
    first_candidates: Option<Candidates>,
    /// Token that [`Self::record_choice`] stashed and `next()` will
    /// decode via [`Decoder::step`]. `None` means "no choice recorded
    /// since last yield" — in that state, iteration stops.
    pending_advance: Option<Token>,
    /// Next position to decode at. After prefill, equals
    /// `start_pos + prompt.len()`; each successful step bumps it by 1.
    pub n_cur: usize,
    /// The number of tokens that have been decoded.
    pub n_decode: usize,
    /// The number of tokens to generate.
    pub n: NonZeroUsize,
}

impl<'engine, B: Backend> CandidatePredictor<'engine, B> {
    /// Create a new `CandidatePredictor` that predicts `n` [`Candidates`]
    /// containers. Clears the KV cache and prefills `tokens` starting
    /// at position 0 on sequence 0.
    pub fn new(
        engine: &'engine mut Engine<B>,
        tokens: Vec<Token>,
        n: NonZeroUsize,
    ) -> Self {
        engine.decoder.memory_clear();
        let first_candidates = {
            let logits = engine
                .decoder
                .prefill(&tokens, 0, 0)
                .expect("prefill failed in CandidatePredictor::new");
            Candidates::from_logits(logits.iter().cloned())
        };
        let n_cur = tokens.len();
        Self {
            tokens,
            engine,
            first_candidates: Some(first_candidates),
            pending_advance: None,
            n_cur,
            n_decode: 0,
            n,
        }
    }

    /// Create a `CandidatePredictor` that resumes generation from a KV
    /// cache the caller has already populated for positions
    /// `[0, start_pos)` on `seq_id`.
    ///
    /// `tokens` is the suffix: positions `[start_pos, start_pos +
    /// tokens.len())` are prefilled here. The first `next()` yields
    /// candidates from those prefill logits; subsequent steps follow
    /// the usual decode loop.
    ///
    /// # Panics
    /// * If `tokens` is empty — there's nothing to resume from.
    pub fn new_resuming(
        engine: &'engine mut Engine<B>,
        tokens: Vec<Token>,
        start_pos: usize,
        seq_id: i32,
        n: NonZeroUsize,
    ) -> Self {
        assert!(
            !tokens.is_empty(),
            "CandidatePredictor::new_resuming requires non-empty tokens",
        );
        let first_candidates = {
            let logits = engine
                .decoder
                .prefill(&tokens, start_pos, seq_id)
                .expect("prefill failed in CandidatePredictor::new_resuming");
            Candidates::from_logits(logits.iter().cloned())
        };
        let n_cur = start_pos + tokens.len();
        Self {
            tokens,
            engine,
            first_candidates: Some(first_candidates),
            pending_advance: None,
            n_cur,
            n_decode: 0,
            n,
        }
    }

    /// Record the choice of a token. The token is pushed to `tokens`
    /// and stashed as the next step's input; the actual decode runs
    /// lazily on the next `next()` call. If `record_choice` is not
    /// called between two `next()` calls, iteration ends (no pending
    /// advance means nothing to decode).
    pub fn record_choice(&mut self, token: Token) {
        self.tokens.push(token);
        self.pending_advance = Some(token);
    }
}

impl<'engine, B: Backend> Iterator
    for CandidatePredictor<'engine, B>
{
    type Item = Candidates;

    fn next(&mut self) -> Option<Self::Item> {
        if self.n_decode == self.n.get()
            || self.n_cur >= self.engine.decoder.n_ctx() as usize
        {
            return None;
        }

        // First yield: logits from the constructor's prefill.
        if let Some(candidates) = self.first_candidates.take() {
            self.n_decode += 1;
            return Some(candidates);
        }

        // Subsequent yields: decode the token recorded via
        // `record_choice`. No recorded token → nothing to advance,
        // iteration stops.
        let token = self.pending_advance.take()?;
        let logits = self
            .engine
            .decoder
            .step(token, self.n_cur, 0)
            .expect("decoder.step failed");
        let candidates = Candidates::from_logits(logits.iter().cloned());
        self.n_cur += 1;
        self.n_decode += 1;
        Some(candidates)
    }
}

impl<'engine, B: Backend> From<CandidatePredictor<'engine, B>>
    for Vec<Token>
{
    fn from(predictor: CandidatePredictor<'engine, B>) -> Self {
        predictor.tokens
    }
}

pub struct TokenPredictor<'engine, B: Backend> {
    rng: Xoroshiro128,
    ngram_stats: NGramStats,
    options: PredictOptions,
    pub text: String,
    pub(crate) max_stop_len: usize,
    /// Mu value for Mirostat sampling
    mu: Option<f32>,
    pub(crate) inner: CandidatePredictor<'engine, B>,
}

impl<'engine, B: Backend> TokenPredictor<'engine, B> {
    pub fn new(
        engine: &'engine mut Engine<B>,
        tokens: Vec<Token>,
        options: PredictOptions,
    ) -> Self {
        let (rng, ngram_stats, options, max_stop_len) =
            Self::prepare(engine, &tokens, options);
        let inner = CandidatePredictor::new(engine, tokens, options.n);
        Self {
            rng,
            ngram_stats,
            options,
            text: String::new(),
            max_stop_len,
            mu: None,
            inner,
        }
    }

    /// Create a `TokenPredictor` that resumes generation from a
    /// pre-populated KV cache.
    pub fn new_resuming(
        engine: &'engine mut Engine<B>,
        tokens: Vec<Token>,
        start_pos: usize,
        seq_id: i32,
        options: PredictOptions,
    ) -> Self {
        let (rng, ngram_stats, options, max_stop_len) =
            Self::prepare(engine, &tokens, options);
        let inner = CandidatePredictor::new_resuming(
            engine, tokens, start_pos, seq_id, options.n,
        );
        Self {
            rng,
            ngram_stats,
            options,
            text: String::new(),
            max_stop_len,
            mu: None,
            inner,
        }
    }

    /// Shared setup for [`Self::new`] and [`Self::new_resuming`]: seed
    /// normalization, RNG construction, n-gram stats seeding, and the
    /// max stop-sequence length.
    fn prepare(
        engine: &Engine<B>,
        tokens: &[Token],
        mut options: PredictOptions,
    ) -> (Xoroshiro128, NGramStats, PredictOptions, usize) {
        // convert seed from a u128 to [u64; 2] to seed the rng
        let seed = match options.seed {
            Some(seed) => seed,
            None => match std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
            {
                0 => {
                    panic!("System clock is broken. Can't get a seed.")
                }
                seed => NonZeroU128::new(seed).unwrap(),
            },
        };
        options.seed = Some(seed);

        // I can't think of a cleaner way to do this without unsafe right now.
        let seed = seed.get().to_le_bytes();
        let seed = [
            u64::from_le_bytes([
                seed[0], seed[1], seed[2], seed[3], seed[4], seed[5], seed[6],
                seed[7],
            ]),
            u64::from_le_bytes([
                seed[8], seed[9], seed[10], seed[11], seed[12], seed[13],
                seed[14], seed[15],
            ]),
        ];

        let mut ngram_stats = NGramStats::new();
        // Dummy candidates to start. TODO: Rethink this. Ngram stats are
        // supposed to include data like cumulative probabilities, but we don't
        // have that here, although we could calculate them from the tokens.
        let candidates =
            Candidates::new(engine.model.n_vocab() as usize).unwrap();
        // Init ngram stats
        if let Some(opts) = &mut options.sample_options.repetition {
            for win in tokens.windows(opts.ngram_max_size.get().into()) {
                for slice in (opts.ngram_min_size.get()
                    ..opts.ngram_max_size.get())
                    .filter_map(|n| win.get((win.len() - n as usize)..))
                {
                    let ngram = NGram::try_from_tokens(slice).unwrap();
                    let _ = ngram_stats.add(ngram, &candidates);
                }
            }
        }

        let max_stop_len = options
            .stop_sequences
            .iter()
            .map(|s| s.len())
            .max()
            .unwrap_or(0);

        (
            Xoroshiro128::from_seed(&seed),
            ngram_stats,
            options,
            max_stop_len,
        )
    }
}

impl<'engine, B: Backend> From<TokenPredictor<'engine, B>>
    for Vec<Token>
{
    fn from(predictor: TokenPredictor<'engine, B>) -> Self {
        predictor.inner.into()
    }
}

// `B::Model: Sync` is required because the grammar filter fans
// candidate validation out across rayon's pool and borrows the model
// across threads. Backend's bound on Model satisfies this implicitly.
impl<'engine, B: Backend> Iterator
    for TokenPredictor<'engine, B>
{
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        let decoded_tokens =
            &self.inner.tokens[self.inner.tokens.len() - self.inner.n_decode..];

        for sequence in self.options.stop_sequences.iter() {
            if decoded_tokens.ends_with(sequence) {
                return None;
            }
        }
        // Beginning of the end to check for stop strings. We don't want to
        // check the entire text because context lengths are getting long and
        // users might use many stop strings.
        let end = self.inner.tokens.len().saturating_sub(
            self.max_stop_len + self.inner.engine.model.max_token_len(),
        );
        for s in self.options.stop_strings.iter() {
            if let Some(slice) = self.text.get(end..) {
                if slice.contains(s) {
                    return None;
                }
            }
        }
        for regex in self.options.regex_stop_sequences.iter() {
            if regex.is_match(&self.text) {
                return None;
            }
        }

        let candidates = self.inner.next()?;

        let next_token = candidates
            .sample_token(
                &self.inner.tokens,
                &mut self.options.sample_options,
                &mut self.ngram_stats,
                &mut self.rng,
                &mut self.mu,
                &self.inner.engine.model,
            )
            .unwrap();

        let piece = self.inner.engine.model.token_to_piece(next_token);
        self.text.push_str(&piece);

        // Deferred-grammar promotion: if the accumulated text now contains
        // the trigger bytes, move the grammar into `modes` and feed any
        // post-trigger tail bytes to its state so the matcher lines up with
        // the model. A matcher-level rejection on the tail collapses the
        // iterator — the caller sees generation end rather than an ungated
        // JSON phase. See `DeferredGrammar`.
        if let Some(trigger_end) = self
            .options
            .sample_options
            .deferred_grammar
            .as_ref()
            .and_then(|d| {
                find_deferred_trigger_end(
                    self.text.as_bytes(),
                    &d.activate_after,
                    self.max_stop_len
                        + self.inner.engine.model.max_token_len(),
                )
            })
        {
            let promoted = self
                .options
                .sample_options
                .deferred_grammar
                .take()
                .expect("deferred_grammar presence checked above");
            let tail = &self.text.as_bytes()[trigger_end..];
            if !tail.is_empty() {
                if let crate::SamplingMode::Grammar(state_arc) =
                    &promoted.grammar
                {
                    let mut locked = state_arc.lock().expect(
                        "deferred grammar mutex poisoned at promotion",
                    );
                    if locked.advance_bytes(tail).is_err() {
                        return None;
                    }
                }
            }
            self.options.sample_options.modes.push(promoted.grammar);
        }

        self.inner.record_choice(next_token);

        Some(next_token)
    }
}

/// Window-bounded search: returns the byte offset one past the last byte of
/// the first occurrence of `trigger` within the trailing `window` bytes of
/// `haystack`. Mirrors the window sizing used for stop-strings so the
/// per-step cost stays bounded even as `text` grows.
fn find_deferred_trigger_end(
    haystack: &[u8],
    trigger: &[u8],
    window: usize,
) -> Option<usize> {
    if trigger.is_empty() || trigger.len() > haystack.len() {
        return None;
    }
    let search_start = haystack
        .len()
        .saturating_sub(window.saturating_add(trigger.len()));
    haystack[search_start..]
        .windows(trigger.len())
        .position(|w| w == trigger)
        .map(|rel| search_start + rel + trigger.len())
}

/// A predictor that predicts pieces of text.
///
/// If the predictor stops predicting because of a stop sequence, the text will
/// be truncated at the stop sequence.
pub struct PiecePredictor<'engine, B: Backend> {
    inner: TokenPredictor<'engine, B>,
}

impl<'engine, B: Backend> PiecePredictor<'engine, B> {
    pub fn new(
        engine: &'engine mut Engine<B>,
        tokens: Vec<Token>,
        options: PredictOptions,
    ) -> Self {
        let token_predictor = TokenPredictor::new(engine, tokens, options);
        Self { inner: token_predictor }
    }

    /// Create a `PiecePredictor` that resumes generation from a
    /// pre-populated KV cache.
    pub fn new_resuming(
        engine: &'engine mut Engine<B>,
        tokens: Vec<Token>,
        start_pos: usize,
        seq_id: i32,
        options: PredictOptions,
    ) -> Self {
        let token_predictor = TokenPredictor::new_resuming(
            engine, tokens, start_pos, seq_id, options,
        );
        Self { inner: token_predictor }
    }

    /// Convert into the tokens and text that have been predicted so far.
    pub fn into_tokens_and_text(self) -> (Vec<Token>, String) {
        let token_predictor = self.inner;
        (token_predictor.inner.tokens, token_predictor.text)
    }

    /// Convert into the text that has been predicted so far.
    pub fn into_text(self) -> String {
        self.inner.text
    }

    /// Get the last token that was predicted.
    pub fn last_token(&self) -> Option<Token> {
        self.inner.inner.tokens.last().copied()
    }
}

impl<'engine, B: Backend> PiecePredictor<'engine, B> {
    /// Predict and collect all the pieces, truncating at stop sequences.
    pub fn collect_text(mut self) -> String {
        while let Some(_) = self.next() {}
        self.into_text()
    }

    /// Predict and collect the tokens and text, truncating at stop sequences.
    pub fn collect_tokens_and_text(mut self) -> (Vec<Token>, String) {
        while let Some(_) = self.next() {}
        self.into_tokens_and_text()
    }

    /// Predict and collect pieces, tokens, and text, truncating at stop
    /// sequences.
    pub fn collect_pieces_tokens_text(
        mut self,
    ) -> (Vec<String>, Vec<Token>, String) {
        let mut pieces = Vec::new();
        // We can't collect because it consumes the predictor.
        while let Some(piece) = self.next() {
            pieces.push(piece);
        }
        let (tokens, text) = self.into_tokens_and_text();
        (pieces, tokens, text)
    }
}

impl<'engine, B: Backend> Iterator
    for PiecePredictor<'engine, B>
{
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        let token = match self.inner.next() {
            Some(token) => token,
            None => {
                // We have to check the text for stop strings and truncate the
                // text if we find one. This matters in cases where a user is
                // using a while let loop and might convert the predictor into a
                // string at the end of the loop. If we don't truncate the text,
                // anything that follows the stop string will be included in the
                // text.
                let mut end = self.inner.inner.tokens.len().saturating_sub(
                    self.inner.max_stop_len
                        + self.inner.inner.engine.model.max_token_len(),
                );

                for s in self.inner.options.stop_strings.iter() {
                    // It's possible at this point that `end` does not lie on a
                    // character boundary, so we move backwards until we find a
                    // character boundary.
                    while !self.inner.text.is_char_boundary(end) {
                        if end == 0 {
                            break;
                        }
                        end -= 1;
                    }

                    if let Some(idx) = self.inner.text[end..].find(s) {
                        self.inner.text.truncate(
                            (end + idx + s.len()).min(self.inner.text.len()),
                        );
                    }
                }

                return None;
            }
        };
        let piece = self.inner.inner.engine.model.token_to_piece(token);
        Some(piece)
    }
}

impl<'engine, B: Backend> From<PiecePredictor<'engine, B>>
    for String
{
    fn from(predictor: PiecePredictor<'engine, B>) -> Self {
        predictor.into_text()
    }
}

impl<'engine, B: Backend> From<PiecePredictor<'engine, B>>
    for Vec<Token>
{
    fn from(predictor: PiecePredictor<'engine, B>) -> Self {
        predictor.inner.inner.tokens
    }
}

/// Contains a token and the associated piece. This is a convenience struct to
/// avoid ackward iterator usage when both the token and piece are needed.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Predicted {
    pub token: Token,
    pub piece: String,
}

pub struct Predictor<'engine, B: Backend> {
    inner: PiecePredictor<'engine, B>,
}

impl<'engine, B: Backend> Predictor<'engine, B> {
    pub fn new(
        engine: &'engine mut Engine<B>,
        tokens: Vec<Token>,
        options: PredictOptions,
    ) -> Self {
        let piece_predictor = PiecePredictor::new(engine, tokens, options);
        Self { inner: piece_predictor }
    }

    /// Convert into the tokens and text that have been predicted so far.
    pub fn into_tokens_and_text(self) -> (Vec<Token>, String) {
        self.inner.into_tokens_and_text()
    }
}

impl<'engine, B: Backend> Iterator
    for Predictor<'engine, B>
{
    type Item = Predicted;

    fn next(&mut self) -> Option<Predicted> {
        let piece = self.inner.next()?;
        let token = self.inner.last_token().unwrap();
        Some(Predicted { token, piece })
    }
}

#[cfg(all(test, feature = "llama-cpp"))]
mod tests {
    use crate::{LlamaCppEngine, PredictOptions, RepetitionOptions, SampleOptions, Token};
    use std::{num::NonZeroUsize, path::PathBuf};

    const PROMPT: &str = "The quick brown fox jumps over the lazy dog.";

    #[test]
    fn test_default_options() {
        let opts = PredictOptions::default();
        assert_eq!(opts.sample_options, SampleOptions::default());
        assert_eq!(
            opts.sample_options.repetition,
            Some(RepetitionOptions::default())
        );
    }

    #[test]
    #[ignore = "long running"]
    /// Test prediction with greedy sampling and a well-known sequence.
    fn test_token_predictor() {
        let mut engine = LlamaCppEngine::from_path(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf"),
        )
        .unwrap();

        let tokenized = engine.model.tokenize(PROMPT, false);
        let prefix = tokenized[..6].to_vec();
        let expected = tokenized[6..].to_vec();

        let mut opts = PredictOptions::greedy().add_stop(".".to_owned());
        opts.n = NonZeroUsize::new(2 + expected.len()).unwrap();

        let actual: Vec<Token> =
            engine.predict_tokens(prefix, opts).collect();

        assert_eq!(actual, expected);
    }

    #[test]
    /// Test candidate prediction with greedy sampling and a well-known sequence.
    #[ignore = "long running"]
    fn test_candidate_predictor() {
        let mut engine = LlamaCppEngine::from_path(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf"),
        )
        .unwrap();

        let tokenized = engine.model.tokenize(PROMPT, false);
        let prefix = tokenized[..6].to_vec();
        let expected_completion = &tokenized[6..];

        let mut predictor =
            engine.predict_candidates(prefix, 6.try_into().unwrap());

        // We can't use a for loop here because we need to record the choice in
        // the predictor, and a for loop *consumes* the `predictor`, so to use
        // the candidate predictor we need to use a while let loop (because
        // ownership issues). For an example of how to use it in a wrapper to
        // make your use more ergonomic, see the TokenPredictor struct.
        while let Some(candidates) = predictor.next() {
            let token = candidates.sample_token_greedy().is_one().unwrap();

            // This must be called or iteration will end.
            predictor.record_choice(token.id);

            // This is for the test only. In a real application, you would
            // probably want to use the PredictOptions to stop the prediction.
            if predictor.n_decode == expected_completion.len() {
                break;
            }
        }

        assert_eq!(predictor.tokens, tokenized);
    }

    #[test]
    /// Test candidate prediction with greedy sampling and a well-known sequence.
    #[ignore = "long running"]
    fn test_piece_predictor() {
        let mut engine = LlamaCppEngine::from_path(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf"),
        )
        .unwrap();

        let tokenized = engine.model.tokenize(PROMPT, false);
        let prefix = tokenized[..6].to_vec();
        let expected: Vec<String> = tokenized[6..]
            .iter()
            .map(|&t| engine.model.token_to_piece(t))
            .collect();

        let mut opts = PredictOptions::greedy().add_stop(".".to_owned());
        opts.n = NonZeroUsize::new(2 + expected.len()).unwrap();

        let actual: Vec<String> = engine.predict_pieces(prefix, opts).collect();

        assert_eq!(actual, expected);
    }

    #[test]
    fn find_deferred_trigger_end_at_end() {
        let hay = b"hello <think>bla</think>";
        let got = super::find_deferred_trigger_end(hay, b"</think>", 64);
        assert_eq!(got, Some(hay.len()));
    }

    #[test]
    fn find_deferred_trigger_end_mid_tail() {
        let hay = b"<think>bla</think>\n  ";
        let got = super::find_deferred_trigger_end(hay, b"</think>", 64);
        assert_eq!(got, Some(b"<think>bla</think>".len()));
    }

    #[test]
    fn find_deferred_trigger_end_none() {
        let hay = b"<think>unclosed body still growing";
        let got = super::find_deferred_trigger_end(hay, b"</think>", 64);
        assert_eq!(got, None);
    }

    #[test]
    fn find_deferred_trigger_end_empty_trigger_is_none() {
        let hay = b"anything";
        let got = super::find_deferred_trigger_end(hay, b"", 64);
        assert_eq!(got, None);
    }

    #[test]
    fn find_deferred_trigger_end_respects_window() {
        // Place the trigger way before the tail window; should miss.
        let mut hay = Vec::new();
        hay.extend_from_slice(b"</think>");
        hay.extend_from_slice(&vec![b'.'; 200]);
        let got = super::find_deferred_trigger_end(&hay, b"</think>", 16);
        assert_eq!(got, None);
    }

    #[test]
    fn find_deferred_trigger_end_window_includes_trigger_boundary() {
        // Trigger ends exactly at the start of the window — must still hit.
        let mut hay = Vec::new();
        hay.extend_from_slice(&vec![b'.'; 200]);
        hay.extend_from_slice(b"</think>");
        hay.extend_from_slice(&vec![b'x'; 8]);
        let got = super::find_deferred_trigger_end(&hay, b"</think>", 16);
        assert_eq!(got, Some(208));
    }
}
