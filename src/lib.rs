// TODO: Importing everything from the submodules is fine for small crates, but
// this is getting crowded. When we go to version 0.2.0, we should consider
// making modules public.

#[cfg(feature = "cli")]
pub mod cli;

#[cfg_attr(test, macro_use)]
pub(crate) mod utils;

pub mod data;
#[allow(deprecated)]
pub use data::{IgnoreCategory, StopWords};

mod sample;
pub use sample::{
    Grammar, GrammarError, GrammarState, JsonError, JsonState, RepetitionError,
    RepetitionOptions, SampleOptions, SamplingMode,
};

mod batch;
pub(crate) use batch::Batch;

mod candidates;
pub use candidates::{Candidates, Sorted, TokenDataArray};

pub mod prompt;
pub use prompt::{
    AssistantMessage, Block, CachedPrompt, Content, Message, Prompt, Role,
    Tool, ToolChoice, UserMessage,
};

mod chat_template;
pub use chat_template::{ChatTemplate, ChatTemplateError, RenderOptions};

/// Re-export of [`minijinja`] for callers who need to construct
/// [`minijinja::value::Value`]s for [`RenderOptions::with_extra`].
pub use minijinja;

mod tool_choice;
pub use tool_choice::{
    build_grammar_source_for_debug, grammar_for_prompt,
    grammar_for_tool_choice, ToolChoiceError, ToolChoiceOptions,
};

mod model;
pub use model::{llama_quantize, Model};

mod ngram;
pub use ngram::{NGram, NGramData, NGramStats};

mod engine;
pub use engine::{
    restore_default_logs, silence_logs, Engine, FlashAttention, NewError,
};

mod predictor;
pub use predictor::{
    CandidatePredictor, PiecePredictor, PredictOptions, Predicted, Predictor,
    TokenPredictor,
};

mod session;
pub use session::{
    parse_completion, BlockParser, BlockStream, Session, SessionError,
    TokenTrace, TopKEntry,
};

mod probability;
pub use probability::{InvalidProbability, Probability};

pub const TOS: &str = include_str!("../TERMS_OF_USE.md");
