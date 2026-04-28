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
    grammar_stats_enabled, grammar_stats_reset, grammar_stats_snapshot,
    DeferredGrammar, Grammar, GrammarError, GrammarState, GrammarStats,
    JsonError, JsonState, RepetitionError, RepetitionOptions, SampleOptions,
    SamplingMode,
};

pub mod backend;
pub use backend::{Backend, Decoder, Model, Token, TokenData};

#[cfg(feature = "llama-cpp")]
mod batch;
#[cfg(feature = "llama-cpp")]
pub(crate) use batch::Batch;

mod candidates;
pub use candidates::{Candidates, Sorted};
#[cfg(feature = "llama-cpp")]
pub use candidates::TokenDataArray;

pub mod prompt;
pub use prompt::{
    AssistantMessage, Block, CachedPrompt, Content, Message, Prompt, Role,
    Tool, ToolChoice, UserMessage,
};

mod chat_template;
pub use chat_template::{ChatTemplate, ChatTemplateError, RenderOptions};

/// Re-export of [`minijinja`] for callers who need to construct
/// [`minijinja::value::Value`]s for `RenderOptions::with_extra`.
pub use minijinja;

pub(crate) mod grammar_compile;

mod tool_choice;
pub use tool_choice::{
    build_grammar_source_for_debug, grammar_for_prompt,
    grammar_for_tool_choice, ToolChoiceError, ToolChoiceOptions,
};

pub mod output_config;
pub use output_config::{
    compile_output_config, compile_prompt_output_config,
    grammar_for_output_config, CompiledOutputConfig, OutputConfigError,
    OutputConfigOptions,
};

#[cfg(feature = "llama-cpp")]
mod llama_cpp;
#[cfg(feature = "llama-cpp")]
pub use llama_cpp::{
    llama_quantize, restore_default_logs, silence_logs, DecodeError,
    FlashAttention, LlamaCppBackend, LlamaCppDecoder, LlamaCppEngine,
    LlamaCppModel, NewError,
};

#[cfg(all(feature = "moeflux", target_os = "macos"))]
pub mod moeflux;
#[cfg(all(feature = "moeflux", target_os = "macos"))]
pub use moeflux::{
    MoefluxBackend, MoefluxDecoder, MoefluxEngine, MoefluxError,
    MoefluxModel, MoefluxModelError,
};

mod ngram;
pub use ngram::{NGram, NGramData, NGramStats};

#[cfg(feature = "toml")]
pub mod sidecar;
#[cfg(feature = "toml")]
pub use sidecar::{
    load_sample_options, write_default_sample_options, SidecarError,
};

mod engine;
pub use engine::Engine;

mod probe;
pub use probe::{ProbeCtx, ProbeHook};

mod predictor;
pub use predictor::{
    CandidatePredictor, PiecePredictor, PredictOptions, Predicted, Predictor,
    TokenPredictor,
};

#[cfg(any(
    feature = "llama-cpp",
    all(feature = "moeflux", target_os = "macos")
))]
mod session;
#[cfg(any(
    feature = "llama-cpp",
    all(feature = "moeflux", target_os = "macos")
))]
pub use session::{
    parse_completion, BlockParser, BlockStream, Session, SessionError,
    TokenTrace, TopKEntry,
};

mod probability;
pub use probability::{InvalidProbability, Probability};

pub const TOS: &str = include_str!("../TERMS_OF_USE.md");
