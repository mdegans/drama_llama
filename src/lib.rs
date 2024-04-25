// TODO: Importing everything from the submodules is fine for small crates, but
// this is getting crowded. When we go to version 0.2.0, we should consider
// making modules public.

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
pub use candidates::{Candidates, Sorted, TokenDataArray};

pub mod prompt;
pub use prompt::{Message, Prompt, Role};

mod model;
pub use model::{llama_quantize, Model, Vocab, VocabKind};

mod ngram;
pub use ngram::{NGram, NGramStats};

mod engine;
pub use engine::Engine;

mod predictor;
pub use predictor::{CandidatePredictor, PredictOptions, TokenPredictor};

mod probability;
pub use probability::{InvalidProbability, Probability};

pub const TOS: &str = include_str!("../TERMS_OF_USE.md");
