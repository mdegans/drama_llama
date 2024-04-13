use std::num::{NonZeroU128, NonZeroUsize};

use llama_cpp_sys::llama_token;
use xorshift::SeedableRng;

use crate::{
    ngram::NGramStats, sample::SampleOptions, Batch, Candidates, Engine, Model,
    NGram,
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
        None => panic!("Failed to create a non-zero seed."),
    };

    /// Add the model's end-of-sequence token to the stop sequences and any
    /// ignored n-grams if repetition penalties are enabled.
    pub fn add_model_eos(mut self, model: &Model) -> Self {
        self = self.add_stop_sequence(vec![model.eos()]);

        if let Some(opts) = &mut self.sample_options.repetition {
            opts.ignored.push(model.eos().into());
        }

        self
    }

    /// Add a stop sequence of tokens. If the [`Predictor`] reaches any of these
    /// sequences, it will stop predicting.
    pub fn add_stop_sequence(mut self, sequence: Vec<llama_token>) -> Self {
        self.stop_sequences.push(sequence);

        self
    }

    /// Add a stop sequence by string. If the [`Predictor`] reaches any of these
    /// sequences, it will stop predicting. Special tokens are not allowed. One
    /// should keep in mind that the string representation of the special tokens
    /// may not exactly match the string representation of the token. As such,
    /// it is recommended to use [`Self::add_stop_sequence`] with the token
    /// values directly.
    ///
    /// # Notes
    /// * Pieces can be previewed with [`Model::token_to_piece`] and other
    /// `Model::token_to_*` functions.
    pub fn add_stop(mut self, sequence: &str, _model: &Model) -> Self {
        // let tokens = model.tokenize(sequence, false);
        // FIXME: This won't work because the tokens won't match. We need to
        // Somehow make sure the string representation of the tokens matches the
        // detokenized string. For now we can just use the string directly.

        // self.add_stop_sequence(tokens)
        self.stop_strings.push(sequence.to_owned());

        self
    }
}

/// A prediction.
#[derive(Debug, Clone)]
#[cfg_attr(test, derive(PartialEq))]
pub struct Predicted {
    /// The predicted token.
    pub token: llama_token,
    /// The piece of text that the token represents
    pub piece: String,
}

/// An iterator that predicts a sequence of tokens until the end of the sequence
/// is reached.
pub struct Predictor<'engine, 'tokens> {
    /// The inference engine.
    engine: &'engine mut Engine,
    /// The tokens to predict.
    tokens: &'tokens mut Vec<llama_token>,
    /// Decoded text (temporary)
    text: String,
    /// The random number generator.
    rng: xorshift::Xoroshiro128,
    /// Candidates for the next token.
    // TODO: We can have a Vec of candidates to implement beam search and more.
    candidates: Candidates,
    /// The batch of tokens.
    batch: Batch,
    /// Prediction options.
    options: PredictOptions,
    /// NGram store for statistics and penalties.
    ngram_stats: NGramStats,
    /// The current index in the batch.
    n_cur: usize,
    /// The number of tokens that have been decoded.
    n_decode: usize,
    /// The mu value for Mirostat sampling.
    mu: Option<f32>,
}

impl<'engine, 'tokens> Predictor<'engine, 'tokens> {
    /// Create a new `PredictIter`.
    ///
    /// # Panics
    /// * If the model's n_vocab is not between 1 and i32::MAX.
    pub fn new(
        engine: &'engine mut Engine,
        tokens: &'tokens mut Vec<llama_token>,
        mut options: PredictOptions,
    ) -> Self {
        // This can't panic unless the model is broken and has a vocab size of
        // 0 or > i32::MAX.
        let mut candidates =
            Candidates::new(engine.model.n_vocab() as usize).unwrap();

        // TODO: we can reduce memory usage for the batch here since the size
        // is always 1 after the initial decode.
        let batch_capacity = options.n.get() + tokens.len();
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
        // FIXME: We don't need a batch this large. We can do the first decode
        // here and then use the batch size of 1 from then on. This is
        // especially true now that the c++ api has async decoding and the
        // decode is not blocking. It's also possible the engine could allow
        // multiple predictions at once and batch them together, but that would
        // require a lot of work.
        // NOTE: There is work going on right now in llama.cpp to rework the
        // batch system to be more efficient. This will likely change in the
        // future.
        let batch = Batch::from_tokens(batch_capacity, tokens).unwrap();
        let n_cur = batch.batch.n_tokens as usize;

        let mut ngram_stats = NGramStats::new();
        // Init ngram stats
        if let Some(opts) = &mut options.sample_options.repetition {
            for win in tokens.windows(opts.ngram_max_size.get().into()) {
                for slice in (opts.ngram_min_size.get()
                    ..opts.ngram_max_size.get())
                    .filter_map(|n| win.get((win.len() - n as usize)..))
                {
                    let ngram = NGram::try_from_tokens(slice).unwrap();
                    _ = ngram_stats.add(ngram, &mut candidates)
                }
            }
        }

        // TODO: async decode was just added to the c++ api. We should start the
        // decode here and lazily get the logits as we need them.
        Self {
            tokens,
            text: String::new(),
            engine,
            batch,
            rng: xorshift::Xoroshiro128::from_seed(&seed),
            candidates,
            ngram_stats,
            options,
            n_cur,
            n_decode: 0,
            mu: None,
        }
    }

    /// Get the [`PredictOptions`] for this predictor. This will include any
    /// seed that was generated if the seed was not provided.
    pub fn options(&self) -> &PredictOptions {
        &self.options
    }
}

impl<'a, 'b> Iterator for Predictor<'a, 'b> {
    type Item = Predicted;

    fn next(&mut self) -> Option<Self::Item> {
        if self.n_decode == self.options.n.get()
            || self.n_cur >= self.engine.n_ctx() as usize
        {
            return None;
        }

        let decoded_tokens = &self.tokens[self.tokens.len() - self.n_decode..];
        for sequence in self.options.stop_sequences.iter() {
            if decoded_tokens.ends_with(sequence) {
                return None;
            }
        }
        for s in self.options.stop_strings.iter() {
            if self.text.ends_with(s) {
                return None;
            }
        }
        for regex in self.options.regex_stop_sequences.iter() {
            if regex.is_match(&self.text) {
                return None;
            }
        }

        let batch = &mut self.batch;

        // Decode the next token
        self.engine.decode(batch).unwrap();

        // Get the logits for the next token
        let logits = self.engine.logits(batch.len() - 1);

        // Update the candidates
        self.candidates
            .data
            .iter_mut()
            .enumerate()
            .zip(logits.iter())
            .for_each(|((i, token), &logit)| {
                token.id = i as llama_token;
                token.logit = logit;
                token.p = 0.0;
            });

        // Sample the next token
        let next_token = self
            .candidates
            .sample_token(
                self.tokens,
                &self.engine.vocab,
                &self.options.sample_options,
                &mut self.ngram_stats,
                &mut self.rng,
                &mut self.mu,
            )
            .ok()?;

        self.tokens.push(next_token);

        self.batch.clear();
        self.batch
            .add_token(next_token, self.n_cur, None, true)
            .unwrap();

        self.n_cur += 1;
        self.n_decode += 1;

        let piece = self.engine.model.token_to_piece(next_token);

        self.text.push_str(&piece);

        Some(Predicted {
            token: next_token,
            piece,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{Engine, PredictOptions};
    use std::path::PathBuf;

    #[test]
    #[ignore = "long running"]
    /// Test prediction with greedy sampling and a well-known sequence.
    fn predict_greedy() {
        const PROMPT: &str = "The quick brown fox jumps over the lazy dog.";

        let mut engine = Engine::from_path(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf"),
        )
        .unwrap();

        let tokenized = engine.model.tokenize(PROMPT, false);
        let mut actual = tokenized[..5].to_vec();
        let expected_completion = tokenized[5..].to_vec();

        let opts = PredictOptions::default()
            .add_model_eos(&engine.model)
            // Stop at period.
            .add_stop_sequence(vec![29889]);

        dbg!(&opts);

        let mut predictor = engine.predict(&mut actual, opts);

        assert_eq!(predictor.next().unwrap().piece, " j");
        assert_eq!(predictor.next().unwrap().piece, "umps");
        assert_eq!(predictor.next().unwrap().piece, " over");
        assert_eq!(predictor.next().unwrap().piece, " the");
        assert_eq!(predictor.next().unwrap().piece, " lazy");
        assert_eq!(predictor.next().unwrap().piece, " dog");
        let last = predictor.next().unwrap();
        assert_eq!(&last.token, &29889);
        assert_eq!(&last.piece, ".");
        assert_eq!(predictor.next(), None);

        assert_eq!(actual[5..], expected_completion);
    }
}
