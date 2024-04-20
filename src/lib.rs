#[cfg(feature = "cli")]
pub mod cli;

#[cfg_attr(test, macro_use)]
pub(crate) mod utils;

pub mod data;

mod sample;
pub use sample::{
    RepetitionError, RepetitionOptions, SampleOptions, SamplingMode,
};

mod batch;
pub(crate) use batch::Batch;

mod candidates;
pub(crate) use candidates::Candidates;

pub mod prompt;
pub use prompt::{Message, Prompt, Role};

mod model;
pub use model::{llama_quantize, Model, Vocab, VocabKind};

mod ngram;
pub use ngram::{NGram, NGramStats};

mod engine;
pub use engine::Engine;

mod predictor;
pub use predictor::{PredictOptions, Predicted, Predictor};

mod probability;
pub use probability::{InvalidProbability, Probability};

pub const TOS: &str = include_str!("../TERMS_OF_USE.md");
