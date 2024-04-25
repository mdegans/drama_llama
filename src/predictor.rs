use std::num::{NonZeroU128, NonZeroUsize};

use llama_cpp_sys_3::llama_token;
use xorshift::{SeedableRng, Xoroshiro128};

use crate::{
    batch::AddError, ngram::NGramStats, prompt::Format, sample::SampleOptions,
    Batch, Candidates, Engine, Model, NGram,
};

#[cfg(feature = "serde")]
fn deserialize_regex_vec<'de, D>(
    deserializer: D,
) -> Result<Vec<regex::Regex>, D::Error>
where
    D: rocket::serde::Deserializer<'de>,
{
    use rocket::serde::de::Deserialize;

    let strings = Vec::<String>::deserialize(deserializer)?;
    strings
        .into_iter()
        .map(|s| {
            regex::Regex::new(&s).map_err(rocket::serde::de::Error::custom)
        })
        .collect()
}

#[cfg(feature = "serde")]
fn serialize_regex_vec<S>(
    regexes: &Vec<regex::Regex>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: rocket::serde::Serializer,
{
    use rocket::serde::ser::SerializeSeq;

    let mut seq = serializer.serialize_seq(Some(regexes.len()))?;
    for regex in regexes {
        seq.serialize_element(&regex.as_str())?;
    }
    seq.end()
}

/// Options for prediction.
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "serde",
    derive(rocket::serde::Deserialize, rocket::serde::Serialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "rocket::serde"))]
pub struct PredictOptions {
    /// Maximum number of tokens to predict.
    pub n: NonZeroUsize,
    /// Random seed. If this is `Some`, the prediction will be deterministic.
    /// Otherwise the seed will be based on the current time.
    pub seed: Option<NonZeroU128>,
    /// Stop sequences. When any of these are reached, the prediction will stop.
    // FIXME: Tokens won't match, but this is fixable. For now we can use
    // strings, but we have to detokenize each token.
    pub stop_sequences: Vec<Vec<llama_token>>,
    /// Stop sequences by string. When any of these are reached, the prediction
    /// will stop. This is a temporary solution until we can fix the token
    /// matching.
    pub(crate) stop_strings: Vec<String>,
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
    const DEFAULT_SEED: NonZeroU128 = match NonZeroU128::new(1337) {
        Some(seed) => seed,
        None => panic!("Bad seed."),
    };

    /// Add any stop tokens from the model. If there is an associated
    /// [`prompt::Format`], the stop tokens will be added. *Otherwise*, the EOS
    /// from [`Model::eos`] will be added (not both).
    ///
    /// To force the inclusion of the EOS token, use [`add_stop_sequence`] with
    /// [`Model::eos`].
    pub fn add_model_stops(mut self, model: &Model) -> Self {
        if let Some(format) = Format::from_model(model) {
            self = self.add_stop_format(format);
        } else {
            // eos is already included if a stop format is found.
            self = self.add_stop_sequence(vec![model.eos()]);

            if let Some(opts) = &mut self.sample_options.repetition {
                opts.ignored.push(model.eos().into());
            }
        }

        self
    }

    /// Add stop sequences for a given [`prompt::Format`]
    ///
    /// [`prompt::Format`]: crate::prompt::Format
    pub fn add_stop_format(mut self, format: Format) -> Self {
        if let Some(stop_tokens) = format.stop_tokens() {
            for &stop_token in stop_tokens {
                self.stop_sequences.push(vec![stop_token]);
                if let Some(opts) = &mut self.sample_options.repetition {
                    opts.ignored.push(stop_token.into());
                }
            }
        }

        self
    }

    /// Add a stop sequence of tokens. If the [`Predictor`] reaches any of these
    /// sequences, it will stop predicting. The stop sequence will be included
    /// in the tokens.
    pub fn add_stop_sequence(mut self, sequence: Vec<llama_token>) -> Self {
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
}
/// An iterator that predicts a sequence of tokens until the end of the sequence
/// is reached.
pub struct CandidatePredictor<'engine> {
    /// The inference engine.
    pub engine: &'engine mut Engine,
    /// The tokens to predict.
    pub tokens: Vec<llama_token>,
    /// The batch of tokens.
    pub batch: Batch,
    /// The current index in the batch.
    pub n_cur: usize,
    /// The number of tokens that have been decoded.
    pub n_decode: usize,
    /// The number of tokens to generate
    pub n: NonZeroUsize,
}

impl<'engine> CandidatePredictor<'engine> {
    /// Create a new `CandidatePredictor` that predicts `n` [`Candidates`]
    /// containers.
    ///
    /// # Panics
    /// * If the model's n_vocab is not between 1 and i32::MAX.
    pub fn new(
        engine: &'engine mut Engine,
        tokens: Vec<llama_token>,
        n: NonZeroUsize,
    ) -> Self {
        // FIXME: We don't need a batch this large. We can do the first decode
        // here and then use the batch size of 1 from then on. This is
        // especially true now that the c++ api has async decoding and the
        // decode is not blocking. It's also possible the engine could allow
        // multiple predictions at once and batch them together, but that would
        // require a lot of work.
        // NOTE: There is work going on right now in llama.cpp to rework the
        // batch system to be more efficient. This will likely change in the
        // future.
        let batch_capacity =
            (n.get() + tokens.len()).min(engine.n_ctx() as usize);
        let batch = Batch::from_tokens(batch_capacity, &tokens).unwrap();
        let n_cur = batch.batch.n_tokens as usize;

        // TODO: async decode was just added to the c++ api. We should start the
        // decode here and lazily get the logits as we need them.
        Self {
            tokens,
            engine,
            batch,
            n_cur,
            n_decode: 0,
            n,
        }
    }

    /// Record the choice of a token. This adds the token to the batch and
    /// tokens. If it's not possible (because the batch is too small. etc.) an
    /// error is returned.
    pub fn record_choice(
        &mut self,
        token: llama_token,
    ) -> Result<(), AddError> {
        self.batch.add_token(token, self.n_cur, None, true)?;
        self.tokens.push(token);

        Ok(())
    }
}

impl<'engine> Iterator for CandidatePredictor<'engine> {
    type Item = Candidates;

    fn next(&mut self) -> Option<Self::Item> {
        if self.n_decode == self.n.get()
            || self.n_cur >= self.engine.n_ctx() as usize
        {
            return None;
        }

        let batch = &mut self.batch;

        // If the batch is empty, we will stop prediction because if we call
        // decode with an empty batch, it will return an error.
        if batch.is_empty() {
            return None;
        }

        // Decode the next token
        self.engine.decode(batch).unwrap();

        // Get the logits for the next token
        let logits = self.engine.logits(batch.len() - 1);

        // Create or fill existing candidates from the logits
        let candidates = Candidates::from_logits(logits.iter().cloned());

        batch.clear();

        self.n_cur += 1;
        self.n_decode += 1;

        Some(candidates)
    }
}

impl<'engine> Into<Vec<llama_token>> for CandidatePredictor<'engine> {
    fn into(self) -> Vec<llama_token> {
        self.tokens
    }
}

pub struct TokenPredictor<'engine> {
    rng: Xoroshiro128,
    ngram_stats: NGramStats,
    options: PredictOptions,
    pub text: String,
    pub(crate) max_stop_len: usize,
    /// Mu value for Mirostat sampling
    mu: Option<f32>,
    pub(crate) inner: CandidatePredictor<'engine>,
}

impl<'engine> TokenPredictor<'engine> {
    pub fn new(
        engine: &'engine mut Engine,
        tokens: Vec<llama_token>,
        mut options: PredictOptions,
    ) -> Self {
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

        let inner = CandidatePredictor::new(engine, tokens, options.n);
        Self {
            rng: Xoroshiro128::from_seed(&seed),
            ngram_stats,
            options,
            text: String::new(),
            max_stop_len,
            mu: None,
            inner,
        }
    }
}

impl Into<Vec<llama_token>> for TokenPredictor<'_> {
    fn into(self) -> Vec<llama_token> {
        self.inner.into()
    }
}

impl<'engine> Iterator for TokenPredictor<'engine> {
    type Item = llama_token;

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
            self.max_stop_len + self.inner.engine.vocab.max_token_len(),
        );
        for s in self.options.stop_strings.iter() {
            if self.text[end..].contains(s) {
                return None;
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
                &self.inner.engine.vocab,
                &self.options.sample_options,
                &mut self.ngram_stats,
                &mut self.rng,
                &mut self.mu,
            )
            .unwrap();

        let piece = self.inner.engine.model.token_to_piece(next_token);
        self.text.push_str(&piece);

        self.inner.record_choice(next_token).ok()?;

        Some(next_token)
    }
}

/// A predictor that predicts pieces of text.
///
/// If the predictor stops predicting because of a stop sequence, the text will
/// be truncated at the stop sequence.
pub struct PiecePredictor<'engine> {
    inner: TokenPredictor<'engine>,
}

impl<'engine> PiecePredictor<'engine> {
    pub fn new(
        engine: &'engine mut Engine,
        tokens: Vec<llama_token>,
        options: PredictOptions,
    ) -> Self {
        let token_predictor = TokenPredictor::new(engine, tokens, options);

        Self {
            inner: token_predictor,
        }
    }

    /// Convert into the tokens and text that have been predicted so far.
    pub fn into_tokens_and_text(self) -> (Vec<llama_token>, String) {
        let token_predictor = self.inner;
        (token_predictor.inner.tokens, token_predictor.text)
    }

    /// Convert into the text that has been predicted so far.
    pub fn into_text(self) -> String {
        self.inner.text
    }

    /// Predict and collect all the pieces, truncating at stop sequences.
    pub fn collect_text(mut self) -> String {
        while let Some(_) = self.next() {}
        self.into_text()
    }

    /// Predict and collect the tokens and text, truncating at stop sequences.
    pub fn collect_tokens_and_text(mut self) -> (Vec<llama_token>, String) {
        while let Some(_) = self.next() {}
        self.into_tokens_and_text()
    }

    /// Predict and collect pieces, tokens, and text, truncating at stop
    /// sequences.
    pub fn collect_pieces_tokens_text(
        mut self,
    ) -> (Vec<String>, Vec<llama_token>, String) {
        let mut pieces = Vec::new();
        // We can't collect because it consumes the predictor.
        while let Some(piece) = self.next() {
            pieces.push(piece);
        }
        let (tokens, text) = self.into_tokens_and_text();
        (pieces, tokens, text)
    }

    /// Get the last token that was predicted.
    pub fn last_token(&self) -> Option<llama_token> {
        self.inner.inner.tokens.last().copied()
    }
}

impl<'engine> Iterator for PiecePredictor<'engine> {
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
                let end = self.inner.inner.tokens.len().saturating_sub(
                    self.inner.max_stop_len
                        + self.inner.inner.engine.vocab.max_token_len(),
                );

                for s in self.inner.options.stop_strings.iter() {
                    if let Some(idx) = self.inner.text[end..].find(s) {
                        self.inner.text.truncate(
                            (end + idx + s.len()).min(self.inner.text.len()),
                        );
                        return None;
                    }
                }

                return None;
            }
        };
        let piece = self.inner.inner.engine.model.token_to_piece(token);
        Some(piece)
    }
}

impl<'engine> Into<String> for PiecePredictor<'engine> {
    fn into(self) -> String {
        self.into_text()
    }
}
impl<'engine> Into<Vec<llama_token>> for PiecePredictor<'engine> {
    fn into(self) -> Vec<llama_token> {
        self.inner.inner.tokens
    }
}

#[cfg(test)]
mod tests {
    use llama_cpp_sys_3::llama_token;

    use crate::{Engine, PredictOptions};
    use std::{num::NonZeroUsize, path::PathBuf};

    const PROMPT: &str = "The quick brown fox jumps over the lazy dog.";

    #[test]
    #[ignore = "long running"]
    /// Test prediction with greedy sampling and a well-known sequence.
    fn test_token_predictor() {
        let mut engine = Engine::from_path(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf"),
        )
        .unwrap();

        let tokenized = engine.model.tokenize(PROMPT, false);
        let prefix = tokenized[..5].to_vec();
        let expected = tokenized[5..].to_vec();

        let mut opts = PredictOptions::default().add_stop(".".to_owned());
        opts.n = NonZeroUsize::new(2 + expected.len()).unwrap();

        let actual: Vec<llama_token> =
            engine.predict_tokens(prefix, opts).collect();

        assert_eq!(actual, expected);
    }

    #[test]
    // #[ignore = "long running"]
    /// Test candidate prediction with greedy sampling and a well-known sequence.
    #[ignore = "long running"]
    fn test_candidate_predictor() {
        let mut engine = Engine::from_path(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf"),
        )
        .unwrap();

        let tokenized = engine.model.tokenize(PROMPT, false);
        let prefix = tokenized[..5].to_vec();
        let expected_completion = &tokenized[5..];

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
            if predictor.record_choice(token.id).is_err() {
                break;
            }

            // This is for the test only. In a real application, you would
            // probably want to use the PredictOptions to stop the prediction.
            if predictor.n_decode == expected_completion.len() {
                break;
            }
        }

        assert_eq!(predictor.tokens, tokenized);
    }

    #[test]
    // #[ignore = "long running"]
    /// Test candidate prediction with greedy sampling and a well-known sequence.
    #[ignore = "long running"]
    fn test_piece_predictor() {
        let mut engine = Engine::from_path(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf"),
        )
        .unwrap();

        let tokenized = engine.model.tokenize(PROMPT, false);
        let prefix = tokenized[..5].to_vec();
        let expected: Vec<String> = tokenized[5..]
            .iter()
            .map(|&t| engine.model.token_to_piece(t))
            .collect();

        let mut opts = PredictOptions::default().add_stop(".".to_owned());
        opts.n = NonZeroUsize::new(2 + expected.len()).unwrap();

        let actual: Vec<String> = engine.predict_pieces(prefix, opts).collect();

        assert_eq!(actual, expected);
    }
}
