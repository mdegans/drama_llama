use std::num::{NonZeroU128, NonZeroUsize};

use crate::{
    backend::{Backend, Decoder, Model},
    sample::SamplerConfig,
    Candidates, Engine, Token,
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
    /// Random seed. `Some` = deterministic prediction (a fork, under
    /// `Session`'s resume/fork/fresh trichotomy). `None` (the default)
    /// = a fresh random seed per call — unless the caller resumes a
    /// cached [`SamplerState`](crate::SamplerState), whose working rng
    /// continues the prior stream. [`Self::DEFAULT_SEED`] remains
    /// available for reproducible runs.
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
    pub sample_options: SamplerConfig,
}

impl Default for PredictOptions {
    fn default() -> Self {
        Self {
            n: NonZeroUsize::new(512).unwrap(),
            seed: None,
            stop_sequences: Vec::new(),
            stop_strings: Vec::new(),
            regex_stop_sequences: Vec::new(),
            sample_options: SamplerConfig::default(),
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
            sample_options: SamplerConfig::greedy(),
            ..Self::default()
        }
    }

    /// Add the model's end-of-generation tokens ([`Model::eog_tokens`])
    /// as stop sequences.
    ///
    /// Every one of them, and *only* them: EOG is the model's own
    /// answer to "does emitting this end the turn", and a vocab can
    /// have an `eot()` that is not in it (gpt-oss's `<|end|>` is the
    /// in-stream channel separator — stopping there truncates a
    /// Harmony turn at the end of its reasoning block). Multi-EOS
    /// vocabs are covered by the same set: Qwen3 declares
    /// `eos_token_id` as `[<|im_end|>, <|endoftext|>]`, and the
    /// secondary decodes to an empty piece, so missing it means the
    /// model loops invisibly until `max_tokens`.
    ///
    /// The repetition penalty ignores the EOG set *and* eos/eot even
    /// when eot isn't EOG: a penalized `<|end|>` is a Harmony model
    /// that can't close a channel. Structure is never repetition.
    ///
    /// [`Model::eog_tokens`]: crate::backend::Model::eog_tokens
    pub fn add_model_stops<M: Model>(mut self, model: &M) -> Self {
        let eog = model.eog_tokens();
        for &token in &eog {
            if token < 0 {
                continue;
            }
            self.stop_sequences.push(vec![token]);
        }
        if let Some(opts) = &mut self.sample_options.repetition {
            for token in eog
                .iter()
                .copied()
                .chain([model.eos(), model.eot()])
                .filter(|&t| t >= 0)
            {
                opts.ignored.insert(token.into());
            }
        }
        // Reserved / unused vocab slots (decode to empty strings,
        // sit outside `special_tokens()`) are NOT handled here —
        // they're masked at sample time via `SamplingMode::Deny`
        // prepended by `Session::prepare_call`, so they never
        // reach the candidate set in the first place.
        self
    }

    /// Push every token in `tokens` as its own single-token stop
    /// sequence and add to the repetition-ignored set. Used by
    /// `Session` to splice in the cached reserved-vocab list once
    /// `add_model_stops` has set the eos/eot anchors.
    pub fn add_token_stops<I>(mut self, tokens: I) -> Self
    where
        I: IntoIterator<Item = Token>,
    {
        for t in tokens {
            if t < 0 {
                continue;
            }
            self.stop_sequences.push(vec![t]);
            if let Some(opts) = &mut self.sample_options.repetition {
                opts.ignored.insert(t.into());
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
                .range(1..=max_context_size),
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
                                .range(1..=usize::MAX),
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
    /// The KV sequence generation decodes on — the constructor's
    /// `seq_id` (0 for [`Self::new`]). Every `step` must target it:
    /// a hardcoded 0 here once sent agent B's tokens into agent A's
    /// sequence under the multi-slot prefix cache (caught by
    /// llama.cpp's M-RoPE position-continuity check; on non-M-RoPE
    /// models it would have been silent cross-agent corruption).
    seq_id: i32,
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
            seq_id: 0,
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
            seq_id,
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

impl<'engine, B: Backend> Iterator for CandidatePredictor<'engine, B> {
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
            .step(token, self.n_cur, self.seq_id)
            .expect("decoder.step failed");
        let candidates = Candidates::from_logits(logits.iter().cloned());
        self.n_cur += 1;
        self.n_decode += 1;
        Some(candidates)
    }
}

impl<'engine, B: Backend> From<CandidatePredictor<'engine, B>> for Vec<Token> {
    fn from(predictor: CandidatePredictor<'engine, B>) -> Self {
        predictor.tokens
    }
}

/// Reassembles UTF-8 across token boundaries.
///
/// Byte-level BPE vocabularies (the `gpt2` family, which is what our
/// models use) have *bytes* for an alphabet, so a multi-byte codepoint
/// routinely splits across consecutive tokens. Each half is
/// **incomplete**, not invalid — rendering it in isolation destroys the
/// character outright. That was issue #55: `Model::token_to_piece`
/// substituted a sentinel string, and every streaming consumer then
/// string-matched the sentinel away, silently deleting the character.
///
/// So: hold the incomplete tail between tokens and emit it once the
/// codepoint closes. Bytes that can *never* close (a genuinely
/// malformed sequence) become U+FFFD immediately and are stepped over
/// — buffering those instead would grow `carry` without bound on a
/// malformed stream and stall every later emission.
#[derive(Debug, Default)]
struct Utf8Reassembler {
    /// Trailing bytes of a not-yet-complete codepoint. Never holds a
    /// complete one; those leave the moment they close. Bounded by the
    /// 3 bytes a truncated UTF-8 sequence can carry.
    carry: Vec<u8>,
    /// Reused scratch for [`Model::token_to_piece_ref`]. Distinct from
    /// `carry` on purpose — scratch lives for one token, carry lives
    /// *across* tokens (the distinction issue #55 turns on).
    scratch: Vec<u8>,
}

impl Utf8Reassembler {
    /// Feed one token's raw piece bytes; returns whatever is complete
    /// as of this token. Empty mid-codepoint.
    fn push<M: Model + ?Sized>(&mut self, model: &M, token: Token) -> String {
        // `token_to_piece_ref`, never `Model::token_to_piece` — the
        // latter is lossy on exactly the bytes we are here to
        // reassemble.
        let mut scratch = std::mem::take(&mut self.scratch);
        model.token_to_piece_ref(token, &mut scratch);
        let piece = self.push_bytes(&scratch);
        self.scratch = scratch;
        piece
    }

    /// Byte-level half of [`Self::push`], split out so the carry logic
    /// is testable without a model.
    fn push_bytes(&mut self, bytes: &[u8]) -> String {
        self.carry.extend_from_slice(bytes);

        let mut out = String::with_capacity(self.carry.len());
        let mut emitted = 0;
        loop {
            match std::str::from_utf8(&self.carry[emitted..]) {
                Ok(valid) => {
                    out.push_str(valid);
                    emitted = self.carry.len();
                    break;
                }
                Err(e) => {
                    let good = e.valid_up_to();
                    out.push_str(
                        std::str::from_utf8(&self.carry[emitted..][..good])
                            .expect("valid_up_to() is a UTF-8 boundary"),
                    );
                    emitted += good;
                    match e.error_len() {
                        // A real invalid sequence — no later token can
                        // complete it. Replace and step over it.
                        Some(n) => {
                            out.push(char::REPLACEMENT_CHARACTER);
                            emitted += n;
                        }
                        // Truncated at the end of input: the next token
                        // carries the rest. Hold it.
                        None => break,
                    }
                }
            }
        }
        self.carry.drain(..emitted);

        out
    }

    /// Stream end: whatever is still held can never close, so it
    /// surfaces as U+FFFD rather than vanishing.
    fn flush(&mut self) -> String {
        let tail = String::from_utf8_lossy(&self.carry).into_owned();
        self.carry.clear();
        tail
    }
}

/// Model-free tests for the issue #55 carry logic. The byte splits
/// below are what a byte-level BPE vocab actually hands us: one
/// codepoint, several byte-fallback tokens.
#[cfg(test)]
mod utf8_reassembler_tests {
    use super::Utf8Reassembler;

    /// U+1F999 LLAMA — four bytes, the worst case for splitting.
    const LLAMA: [u8; 4] = [0xf0, 0x9f, 0xa6, 0x99];

    #[test]
    fn ascii_passes_straight_through() {
        let mut r = Utf8Reassembler::default();
        assert_eq!(r.push_bytes(b"The quick "), "The quick ");
        assert_eq!(r.push_bytes(b"brown fox"), "brown fox");
        assert!(r.carry.is_empty());
        assert_eq!(r.flush(), "");
    }

    #[test]
    fn emoji_split_one_three() {
        let mut r = Utf8Reassembler::default();
        assert_eq!(r.push_bytes(&LLAMA[..1]), "");
        assert_eq!(r.carry.len(), 1);
        assert_eq!(r.push_bytes(&LLAMA[1..]), "🦙");
        assert!(r.carry.is_empty());
    }

    #[test]
    fn emoji_split_two_two() {
        let mut r = Utf8Reassembler::default();
        assert_eq!(r.push_bytes(&LLAMA[..2]), "");
        assert_eq!(r.push_bytes(&LLAMA[2..]), "🦙");
        assert!(r.carry.is_empty());
    }

    /// Surrounding text must ride along with the reassembled codepoint,
    /// in order, in one emission.
    #[test]
    fn text_around_a_split_codepoint_keeps_its_order() {
        let mut r = Utf8Reassembler::default();
        let mut head = b"a ".to_vec();
        head.extend_from_slice(&LLAMA[..3]);
        assert_eq!(r.push_bytes(&head), "a ");

        let mut tail = vec![LLAMA[3]];
        tail.extend_from_slice(b" b");
        assert_eq!(r.push_bytes(&tail), "🦙 b");
    }

    /// A lone continuation byte can never complete. It must become
    /// U+FFFD immediately — buffering it would grow `carry` forever and
    /// stall every later emission (the trap called out in issue #55).
    #[test]
    fn lone_continuation_byte_is_replaced_not_buffered() {
        let mut r = Utf8Reassembler::default();
        assert_eq!(r.push_bytes(&[0x80]), "\u{FFFD}");
        assert!(r.carry.is_empty(), "invalid bytes must not be carried");
        // And the stream keeps flowing afterwards.
        assert_eq!(r.push_bytes(b"ok"), "ok");
    }

    /// Repeated garbage must not accumulate — `carry` stays bounded.
    #[test]
    fn malformed_stream_does_not_grow_carry() {
        let mut r = Utf8Reassembler::default();
        for _ in 0..64 {
            assert_eq!(r.push_bytes(&[0xff]), "\u{FFFD}");
            assert!(r.carry.is_empty());
        }
    }

    /// Invalid bytes ahead of a truncated tail: replace the first,
    /// hold the second.
    #[test]
    fn invalid_then_incomplete() {
        let mut r = Utf8Reassembler::default();
        let mut bytes = vec![0x80];
        bytes.extend_from_slice(&LLAMA[..2]);
        assert_eq!(r.push_bytes(&bytes), "\u{FFFD}");
        assert_eq!(r.carry, &LLAMA[..2]);
    }

    #[test]
    fn flush_emits_the_orphaned_tail() {
        let mut r = Utf8Reassembler::default();
        assert_eq!(r.push_bytes(&LLAMA[..2]), "");
        assert_eq!(r.flush(), "\u{FFFD}");
        assert!(r.carry.is_empty());
        // Idempotent: `PiecePredictor` guards against a second call,
        // but the reassembler does not rely on that guard.
        assert_eq!(r.flush(), "");
    }
}

pub struct TokenPredictor<'engine, B: Backend> {
    /// Per-call sampler run-state (matchers, RNG, mu, n-gram stats).
    /// Built by `prepare` from the effective config via
    /// `SamplerConfig::init_state` — the config in `options` stays
    /// immutable for the life of the predictor.
    state: crate::SamplerState,
    options: PredictOptions,
    pub text: String,
    pub(crate) max_stop_len: usize,
    /// Set when the just-sampled token satisfied a stop condition; the
    /// following `next()` returns `None` without decoding it (the
    /// recorded-but-uncommitted terminal token the auto-tip relies on).
    /// Starts `false`: all stop windows (generated-token tail,
    /// generated text) are empty before the first sample, so no stop
    /// can precede it.
    stopped: bool,
    /// Set alongside `stopped` when the terminal token's bytes would
    /// have completed a constraint (`SamplerState::completes_with_terminal`).
    /// The terminal token never advances the matchers (tip invariant),
    /// but generation VALIDITY must still see its effect — the
    /// dialect-exit-marker shape where the grammar's required exit
    /// bytes are also a vocab EOG (Gemma 4's `<|tool_response>`).
    /// Folded into [`Self::grammar_complete`] /
    /// [`Self::constraint_incomplete_at_end`]; the state itself stays
    /// pure.
    terminal_completed: bool,
    /// Carries the incomplete tail of a codepoint split across
    /// byte-fallback tokens (issue #55). Owned *here*, alongside
    /// `text`, on purpose: the stop-string, regex and deferred-trigger
    /// scans all read `text`, so reassembling into it makes the bytes
    /// they scan the same bytes the model actually emitted.
    reassembler: Utf8Reassembler,
    pub(crate) inner: CandidatePredictor<'engine, B>,
}

impl<'engine, B: Backend> TokenPredictor<'engine, B> {
    /// `initial_state`: `Some` resumes a caller-owned
    /// [`SamplerState`](crate::SamplerState) (no `init_state`, no
    /// prompt seeding — see `Self::prepare`); `None` builds a fresh
    /// state from `options`.
    pub fn new(
        engine: &'engine mut Engine<B>,
        tokens: Vec<Token>,
        options: PredictOptions,
        initial_state: Option<crate::SamplerState>,
    ) -> Self {
        let (state, options, max_stop_len) =
            Self::prepare(engine, options, initial_state);
        let inner = CandidatePredictor::new(engine, tokens, options.n);
        Self {
            state,
            options,
            text: String::new(),
            max_stop_len,
            stopped: false,
            terminal_completed: false,
            reassembler: Utf8Reassembler::default(),
            inner,
        }
    }

    /// Create a `TokenPredictor` that resumes generation from a
    /// pre-populated KV cache. `initial_state` as in [`Self::new`].
    pub fn new_resuming(
        engine: &'engine mut Engine<B>,
        tokens: Vec<Token>,
        start_pos: usize,
        seq_id: i32,
        options: PredictOptions,
        initial_state: Option<crate::SamplerState>,
    ) -> Self {
        let (state, options, max_stop_len) =
            Self::prepare(engine, options, initial_state);
        let inner = CandidatePredictor::new_resuming(
            engine, tokens, start_pos, seq_id, options.n,
        );
        Self {
            state,
            options,
            text: String::new(),
            max_stop_len,
            stopped: false,
            terminal_completed: false,
            reassembler: Utf8Reassembler::default(),
            inner,
        }
    }

    /// The live sampler run-state: matcher positions, working RNG,
    /// mirostat `mu`, n-gram stats. The replacement for observing
    /// matcher progress through shared `Arc<Mutex<…>>` handles.
    pub fn sampler_state(&self) -> &crate::SamplerState {
        &self.state
    }

    /// True iff any constraint matcher — including an activated
    /// deferred grammar — has reached its accept state, or the
    /// terminal token's bytes would have completed one (the state
    /// itself never carries the terminal token — tip invariant; see
    /// `terminal_completed`).
    pub fn grammar_complete(&self) -> bool {
        self.state.grammar_complete() || self.terminal_completed
    }

    /// True iff generation ended mid-constraint — the incomplete-at-end
    /// violation signal. Covers an eager constraint that never reached
    /// accept **and** a deferred grammar that activated and is
    /// incomplete (issue #38, defect 1). A deferred grammar that never
    /// triggered is exempt: never calling a tool is legal.
    ///
    /// A terminal token whose bytes complete the constraint counts as
    /// completion (see `terminal_completed`) even though it never
    /// advances the state — that is what keeps the Gemma 4 shape, where
    /// the required closing bytes *are* an EOG token, from reading as a
    /// violation.
    pub fn constraint_incomplete_at_end(&self) -> bool {
        self.state.constraint_incomplete_at_end() && !self.terminal_completed
    }

    /// Close out the UTF-8 reassembler at stream end (issue #55).
    ///
    /// Bytes still held are a codepoint the generation cut in half, so
    /// they land in [`Self::text`] as U+FFFD instead of disappearing.
    /// Returns the appended text — the piece-level iterator yields
    /// exactly what this appended, keeping the stream and `text`
    /// byte-identical. Idempotent: a second call appends nothing.
    pub(crate) fn flush_utf8(&mut self) -> String {
        let tail = self.reassembler.flush();
        self.text.push_str(&tail);
        tail
    }

    /// Shared setup for [`Self::new`] and [`Self::new_resuming`]: seed
    /// normalization, state construction (unless injected), and the
    /// max stop-sequence length.
    ///
    /// NOTE: raw `Engine::predict_*` paths do NOT prompt-seed the
    /// repetition stats (the old whole-prompt window loop lived here).
    /// Prompt seeding is block-gated and Session-owned now —
    /// `Session`'s prose fold seeds prose blocks only (tool results
    /// and other structured content excluded) and injects the state
    /// through `initial_state`. A raw caller who wants prompt seeding
    /// builds its own [`SamplerState`](crate::SamplerState) and
    /// injects it.
    fn prepare(
        engine: &Engine<B>,
        mut options: PredictOptions,
        initial_state: Option<crate::SamplerState>,
    ) -> (crate::SamplerState, PredictOptions, usize) {
        let max_stop_len = options
            .stop_sequences
            .iter()
            .map(|s| s.len())
            .max()
            .unwrap_or(0);

        // A caller-provided state is authoritative: it resumes (or
        // freshly seeds) a prior stream — rng mid-sequence, carried
        // n-gram stats, reconciled matchers (see
        // `SamplerState::resumed_from`) — so `init_state` is skipped.
        if let Some(state) = initial_state {
            return (state, options, max_stop_len);
        }

        let seed = options.seed.unwrap_or_else(|| {
            // Fresh entropy per call ("no seed + no cached state" in
            // the resume/fork/fresh trichotomy). `max(1)` for NonZero;
            // losing one value of the space is harmless.
            NonZeroU128::new(rand::random::<u128>().max(1)).unwrap()
        });
        options.seed = Some(seed);

        let state =
            options.sample_options.init_state(seed.get(), &engine.model);

        (state, options, max_stop_len)
    }
}

impl<'engine, B: Backend> From<TokenPredictor<'engine, B>> for Vec<Token> {
    fn from(predictor: TokenPredictor<'engine, B>) -> Self {
        predictor.inner.into()
    }
}

// `B::Model: Sync` is required because the grammar filter fans
// candidate validation out across rayon's pool and borrows the model
// across threads. Backend's bound on Model satisfies this implicitly.
impl<'engine, B: Backend> Iterator for TokenPredictor<'engine, B> {
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        // NOTE (auto-tip dependency): a terminal token sets `stopped`
        // on the iteration that sampled it, and this early-return fires
        // BEFORE `inner.next()` would call `decoder.step` on it — so
        // the recorded terminal token (e.g. EOS) lands in
        // `inner.tokens` but never in the engine's KV cache.
        // `Session`'s prefix-cache auto-tip (see
        // `session/mod.rs::compute_tip_extension` and
        // `PrefixCache::tip`) relies on this exact behavior:
        // `prev_tokens` is set to the engine's KV state (EOS-free)
        // while the recorded-but-uncommitted EOS in `inner.tokens` is
        // what makes the next call's LCP extend one token past KV,
        // letting the tip qualify under `compute_l_hit`'s lcp-1
        // safety. If you change this ordering — e.g. commit every
        // recorded token before checking `stopped` — update
        // `Session::compute_tip_extension` in lockstep or the tip will
        // desync from KV and silently corrupt restores.
        if self.stopped {
            return None;
        }

        let candidates = self.inner.next()?;

        // Snapshot only when an installed hook declares appetite. Cheap
        // probe of the trait method (default `None`) keeps the
        // production path free of per-token softmax/sort cost.
        let snapshot = self
            .inner
            .engine
            .probe_hook
            .as_ref()
            .and_then(|h| h.snapshot_opts())
            .map(|opts| candidates.capture_snapshot(&opts));

        let next_token = candidates
            .sample_token(
                &self.inner.tokens,
                &self.options.sample_options,
                &mut self.state,
                &self.inner.engine.model,
            )
            .unwrap();

        // Reassembled, not converted in isolation: a token that is
        // only part of a codepoint yields nothing here and its bytes
        // ride along in the reassembler until the codepoint closes
        // (issue #55). `text` is the sole record of what was emitted —
        // `PiecePredictor` yields the delta it grew by, so the stream
        // and `into_text()` cannot disagree.
        let piece = self.reassembler.push(&self.inner.engine.model, next_token);
        self.text.push_str(&piece);

        // Evaluate every stop condition against the just-sampled token,
        // BEFORE advancing the constraint matchers: a token that
        // terminates generation must never mutate the sampler state
        // (tip invariant — it is absent from the cache entries and the
        // KV alike, so the state must not carry its bytes either). The
        // rng (and possibly mirostat `mu`) already advanced to *sample*
        // it; that is a deliberate, documented exemption — no oracle
        // can observe it. Inputs are byte-identical to what the old
        // top-of-next-call checks saw: the token tail is checked as if
        // `next_token` were already recorded. The stop-string window
        // needs no such compensation — `self.text` already carries this
        // token's bytes (pushed above), and the window is sized in
        // bytes off that text, not in tokens (#65).
        let decoded_tokens =
            &self.inner.tokens[self.inner.tokens.len() - self.inner.n_decode..];
        let stopped_by_sequence =
            self.options.stop_sequences.iter().any(|sequence| {
                match sequence.split_last() {
                    Some((&last, rest)) => {
                        last == next_token && decoded_tokens.ends_with(rest)
                    }
                    None => false,
                }
            });
        // Beginning of the end to check for stop strings. We don't want to
        // check the entire text because context lengths are getting long and
        // users might use many stop strings.
        let end = stop_window_start(
            &self.text,
            self.max_stop_len,
            self.inner.engine.model.max_token_len(),
        );
        let stopped_by_string = self
            .options
            .stop_strings
            .iter()
            .any(|s| self.text[end..].contains(s));
        let stopped_by_regex = self
            .options
            .regex_stop_sequences
            .iter()
            .any(|regex| regex.is_match(&self.text));
        self.stopped =
            stopped_by_sequence || stopped_by_string || stopped_by_regex;
        if self.stopped {
            // Read-only validity check for the terminal token (which
            // will not advance the matchers below).
            self.terminal_completed = self.state.completes_with_terminal(
                &self.options.sample_options,
                next_token,
                &self.inner.engine.model,
            );
        }

        // Advance the constraint matchers only when generation
        // continues. Order is load-bearing: advance-before-scan — at
        // advance time a not-yet-triggered deferred matcher is inactive
        // and receives nothing; activation below then feeds the tail
        // (including this piece's post-trigger bytes) exactly once.
        if !self.stopped {
            self.state.advance(
                &self.options.sample_options,
                next_token,
                &self.inner.engine.model,
            );
        }

        // Deferred-grammar activation: if the accumulated text now
        // contains a trigger, flag the state's deferred matcher active
        // and feed any post-trigger tail bytes so it lines up with the
        // model. A matcher-level rejection on the tail collapses the
        // iterator — the caller sees generation end rather than an
        // ungated JSON phase. See `DeferredGrammar`. A trigger completed
        // by a terminal token deliberately does not activate (tip
        // invariant): `stopped` short-circuits the scan.
        if let (false, Some(spec), Some(true)) = (
            self.stopped,
            self.options.sample_options.deferred_grammar.as_ref(),
            self.state.deferred_inactive(),
        ) {
            if let Some((trigger_end, trigger_len)) =
                find_any_deferred_trigger_end(
                    self.text.as_bytes(),
                    &spec.activate_after,
                    self.max_stop_len + self.inner.engine.model.max_token_len(),
                )
            {
                // Lazy-pattern grammars start their root at the trigger
                // itself; feed from the trigger's first byte so the
                // matcher lines up (`find_deferred_trigger_end`
                // guarantees `trigger_end >= trigger.len()`).
                let feed_from = if spec.feed_trigger {
                    trigger_end - trigger_len
                } else {
                    trigger_end
                };
                let tail = &self.text.as_bytes()[feed_from..];
                if self.state.activate_deferred(spec, tail).is_err() {
                    return None;
                }
            }
        }

        if let Some(hook) = self.inner.engine.probe_hook.as_mut() {
            hook.on_token(crate::ProbeCtx {
                token: next_token,
                n_cur: self.inner.n_cur,
                config: &self.options.sample_options,
                state: &self.state,
                snapshot: snapshot.as_ref(),
                piece: &piece,
                generation_index: (self.inner.n_decode - 1) as u32,
            });
        }

        self.inner.record_choice(next_token);

        Some(next_token)
    }
}

/// Byte offset at which a stop-string search over `text` may begin: the
/// trailing `max_stop_len + max_token_len` bytes, walked back to a char
/// boundary. Keeps the per-step cost bounded as generation grows.
///
/// **Bytes, not tokens** (issue #65). The window used to be sized from
/// `CandidatePredictor::tokens.len()` — a *prompt-inclusive token
/// count* — and then used as a byte index into generated-bytes-only
/// text. For any prompt longer than the window that produced an offset
/// past the end of the text, `str::get` returned `None`, and early
/// stop-string termination silently never fired. On an Agora-scale
/// prompt it never fired at all.
///
/// The char-boundary walk-back is load-bearing on its own: `str::get`
/// and `str` indexing both reject a non-boundary offset, so without it
/// the same silent failure returns intermittently, mid-codepoint. It
/// always terminates — offset 0 is a boundary by definition.
fn stop_window_start(
    text: &str,
    max_stop_len: usize,
    max_token_len: usize,
) -> usize {
    let mut end = text.len().saturating_sub(max_stop_len + max_token_len);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    end
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

/// Any-of variant over a trigger set: the earliest match wins (ties
/// go to the longer trigger, so `<x> to=` beats ` to=`-style overlaps
/// feeding the right byte count). Returns `(trigger_end,
/// trigger_len)` for the winner.
fn find_any_deferred_trigger_end(
    haystack: &[u8],
    triggers: &[Vec<u8>],
    window: usize,
) -> Option<(usize, usize)> {
    triggers
        .iter()
        .filter_map(|t| {
            find_deferred_trigger_end(haystack, t, window)
                .map(|end| (end, t.len()))
        })
        .min_by_key(|&(end, len)| (end - len, std::cmp::Reverse(len)))
}

/// A predictor that predicts pieces of text.
///
/// If the predictor stops predicting because of a stop sequence, the text will
/// be truncated at the stop sequence.
///
/// Pieces are **reassembled**, not converted one token at a time: a
/// codepoint split across byte-fallback tokens is held until it closes
/// and then yielded whole, so a token can yield the empty string
/// (issue #55).
pub struct PiecePredictor<'engine, B: Backend> {
    inner: TokenPredictor<'engine, B>,
    /// Whether [`TokenPredictor::flush_utf8`] has run. The flush is a
    /// one-shot extra yield at stream end, so the terminal `None` arm
    /// has to know not to repeat it.
    flushed: bool,
}

impl<'engine, B: Backend> PiecePredictor<'engine, B> {
    /// `initial_state` as in [`TokenPredictor::new`].
    pub fn new(
        engine: &'engine mut Engine<B>,
        tokens: Vec<Token>,
        options: PredictOptions,
        initial_state: Option<crate::SamplerState>,
    ) -> Self {
        let token_predictor =
            TokenPredictor::new(engine, tokens, options, initial_state);
        Self {
            inner: token_predictor,
            flushed: false,
        }
    }

    /// Create a `PiecePredictor` that resumes generation from a
    /// pre-populated KV cache. `initial_state` as in
    /// [`TokenPredictor::new`].
    pub fn new_resuming(
        engine: &'engine mut Engine<B>,
        tokens: Vec<Token>,
        start_pos: usize,
        seq_id: i32,
        options: PredictOptions,
        initial_state: Option<crate::SamplerState>,
    ) -> Self {
        let token_predictor = TokenPredictor::new_resuming(
            engine,
            tokens,
            start_pos,
            seq_id,
            options,
            initial_state,
        );
        Self {
            inner: token_predictor,
            flushed: false,
        }
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

    /// The live sampler run-state. See [`TokenPredictor::sampler_state`].
    pub fn sampler_state(&self) -> &crate::SamplerState {
        self.inner.sampler_state()
    }

    /// See [`TokenPredictor::grammar_complete`].
    pub fn grammar_complete(&self) -> bool {
        self.inner.grammar_complete()
    }

    /// See [`TokenPredictor::constraint_incomplete_at_end`].
    pub fn constraint_incomplete_at_end(&self) -> bool {
        self.inner.constraint_incomplete_at_end()
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

impl<'engine, B: Backend> Iterator for PiecePredictor<'engine, B> {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        // The yielded piece is the delta `text` grew by, sliced rather
        // than re-derived from the token: the reassembler may hold this
        // token's bytes back (or release a codepoint the *previous*
        // token opened), so token → piece is no longer a function.
        // Slicing makes "what was yielded" == "what is in `text`" a
        // property of the code rather than a promise (issue #55).
        let emitted = self.inner.text.len();
        match self.inner.next() {
            Some(_) => Some(self.inner.text[emitted..].to_owned()),
            None => {
                // Stream end. Surface any codepoint the last token left
                // half-delivered before the text is finalized — one
                // extra yield, then the next call falls through here to
                // the stop-string truncation below.
                if !self.flushed {
                    self.flushed = true;
                    let tail = self.inner.flush_utf8();
                    if !tail.is_empty() {
                        return Some(tail);
                    }
                }

                // We have to check the text for stop strings and truncate the
                // text if we find one. This matters in cases where a user is
                // using a while let loop and might convert the predictor into a
                // string at the end of the loop. If we don't truncate the text,
                // anything that follows the stop string will be included in the
                // text.
                // Recomputed per stop string, not hoisted: a truncation
                // below shortens `text`, and a stale offset past the new
                // end would panic on the next iteration's slice. The old
                // code got away with hoisting only because its offset was
                // a token count that the char-boundary walk-back happened
                // to clamp back into range — which is also why this site
                // never truncated anything on a long prompt (#65).
                for s in self.inner.options.stop_strings.iter() {
                    let end = stop_window_start(
                        &self.inner.text,
                        self.inner.max_stop_len,
                        self.inner.inner.engine.model.max_token_len(),
                    );
                    if let Some(idx) = self.inner.text[end..].find(s) {
                        // In range by construction: `idx + s.len()` is
                        // bounded by the length of the slice it was found in.
                        self.inner.text.truncate(end + idx + s.len());
                    }
                }

                None
            }
        }
    }
}

impl<'engine, B: Backend> From<PiecePredictor<'engine, B>> for String {
    fn from(predictor: PiecePredictor<'engine, B>) -> Self {
        predictor.into_text()
    }
}

impl<'engine, B: Backend> From<PiecePredictor<'engine, B>> for Vec<Token> {
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
    /// Text that became renderable *at* `token` — not necessarily the
    /// text *of* `token`. A codepoint split across byte-fallback
    /// tokens is held until it closes, so `piece` may be empty, or may
    /// carry bytes an earlier token contributed (issue #55).
    pub piece: String,
}

pub struct Predictor<'engine, B: Backend> {
    inner: PiecePredictor<'engine, B>,
}

impl<'engine, B: Backend> Predictor<'engine, B> {
    /// `initial_state` as in [`TokenPredictor::new`].
    pub fn new(
        engine: &'engine mut Engine<B>,
        tokens: Vec<Token>,
        options: PredictOptions,
        initial_state: Option<crate::SamplerState>,
    ) -> Self {
        let piece_predictor =
            PiecePredictor::new(engine, tokens, options, initial_state);
        Self {
            inner: piece_predictor,
        }
    }

    /// Convert into the tokens and text that have been predicted so far.
    pub fn into_tokens_and_text(self) -> (Vec<Token>, String) {
        self.inner.into_tokens_and_text()
    }
}

impl<'engine, B: Backend> Iterator for Predictor<'engine, B> {
    type Item = Predicted;

    fn next(&mut self) -> Option<Predicted> {
        let piece = self.inner.next()?;
        let token = self.inner.last_token().unwrap();
        Some(Predicted { token, piece })
    }
}

#[cfg(all(test, feature = "llama-cpp"))]
mod tests {
    use crate::{
        LlamaCppEngine, PredictOptions, RepetitionOptions, SamplerConfig, Token,
    };
    use std::{num::NonZeroUsize, path::PathBuf};

    const PROMPT: &str = "The quick brown fox jumps over the lazy dog.";

    #[test]
    fn test_default_options() {
        let opts = PredictOptions::default();
        assert_eq!(opts.sample_options, SamplerConfig::default());
        // SamplerConfig::default() ships repetition on as of v0.8.0
        // (windowed decay removed the long-form degradation). Probes
        // that want the raw logit gradient pass `--no-penalty` /
        // construct `SamplerConfig::greedy()`.
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
            engine.predict_tokens(prefix, opts, None).collect();

        // Greedy continuation should reproduce the source text, but the
        // final token is the model's pick — Qwen3.6 ends the pangram with
        // ".\n" (one token) where Llama 3.1 ended with "." — so compare
        // detokenized text by prefix rather than exact token ids.
        let actual_text = engine.model.tokens_to_string(actual);
        let expected_text = engine.model.tokens_to_string(expected);
        assert!(
            actual_text.starts_with(&expected_text),
            "greedy continuation {actual_text:?} should start with {expected_text:?}"
        );
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

        // Prefix-compare detokenized text; see test_token_predictor for
        // why exact token ids are too strict across models. The predictor
        // mutably borrows the engine, so take the tokens and release it
        // before detokenizing.
        let actual_tokens = predictor.tokens.clone();
        drop(predictor);
        let actual_text = engine.model.tokens_to_string(actual_tokens);
        let expected_text = engine.model.tokens_to_string(tokenized);
        assert!(
            actual_text.starts_with(&expected_text),
            "greedy continuation {actual_text:?} should start with {expected_text:?}"
        );
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

        let actual: Vec<String> =
            engine.predict_pieces(prefix, opts, None).collect();

        // Prefix-compare joined text; see test_token_predictor for why
        // exact piece-by-piece equality is too strict across models.
        let actual_text: String = actual.concat();
        let expected_text: String = expected.concat();
        assert!(
            actual_text.starts_with(&expected_text),
            "greedy continuation {actual_text:?} should start with {expected_text:?}"
        );
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

    /// #65: the window is sized in **bytes off the generated text**. It
    /// used to be sized off a prompt-inclusive token count, so on any
    /// prompt longer than the window the offset ran past the end of the
    /// text and early stop-string termination silently never fired.
    ///
    /// The regression shape to keep in mind: the offset must never
    /// depend on how long the prompt was. These cases pin that by
    /// construction — nothing here knows about a prompt at all.
    #[test]
    fn stop_window_start_is_sized_in_bytes_not_tokens() {
        // Text shorter than the window: scan all of it.
        assert_eq!(super::stop_window_start("short", 8, 16), 0);

        // Text longer than the window: scan exactly the trailing
        // `max_stop_len + max_token_len` bytes.
        let text = "a".repeat(100);
        assert_eq!(super::stop_window_start(&text, 8, 16), 76);

        // A stop string ending at the very end of the text is always
        // inside the window, which is the property the sizing exists
        // for.
        let text = format!("{}STOP", "x".repeat(500));
        let end = super::stop_window_start(&text, 4, 16);
        assert!(text[end..].contains("STOP"));
    }

    /// The walk-back is not cosmetic: `str` indexing rejects a
    /// non-boundary offset, so landing mid-codepoint would reintroduce
    /// #65's silent failure intermittently. Terminates at 0, which is a
    /// boundary by definition.
    #[test]
    fn stop_window_start_lands_on_a_char_boundary() {
        // Multi-byte throughout, so a naive offset lands mid-codepoint.
        let text = "é".repeat(50); // 100 bytes, 2 bytes per char
        for max_stop_len in 0..12 {
            let end = super::stop_window_start(&text, max_stop_len, 5);
            assert!(
                text.is_char_boundary(end),
                "offset {end} splits a codepoint (max_stop_len={max_stop_len})",
            );
            // Must not panic, and must be usable as a slice start.
            let _ = &text[end..];
        }

        // Degenerate: empty text, huge window.
        assert_eq!(super::stop_window_start("", 1000, 1000), 0);
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
