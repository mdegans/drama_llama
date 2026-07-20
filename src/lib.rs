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
    apply_request_sampling, grammar_stats_enabled, grammar_stats_reset,
    grammar_stats_snapshot, CompiledGrammar, DeferredGrammar, Grammar,
    GrammarError, GrammarState, GrammarStats, JsonError, JsonState, Mirostat,
    RepetitionError, RepetitionOptions, SamplerConfig, SamplerState,
    SamplingMode, SamplingParams,
};

pub mod backend;
#[cfg(feature = "media")]
pub use backend::ImageDecodeError;
pub use backend::{
    Backend, Decoder, Image, ImageInfo, ImageNewError, MediaChunk, MediaSpan,
    Model, NoVision, Token, TokenData, Vision,
};

#[cfg(feature = "llama-cpp")]
mod batch;
#[cfg(feature = "llama-cpp")]
pub(crate) use batch::Batch;

mod candidates;
#[cfg(feature = "llama-cpp")]
pub use candidates::TokenDataArray;
pub use candidates::{Candidates, Snapshot, SnapshotOpts, Sorted};

pub mod prompt;
pub use prompt::{
    AssistantMessage, Block, CachedPrompt, Content, Message, Prompt, Role,
    Tool, ToolChoice, UserMessage,
};

mod chat_template;
pub use chat_template::{
    tokenize_with_breakpoints, ChatTemplate, ChatTemplateError,
    PromptBreakpoint, RenderOptions, RenderedWithBreakpoints,
};

/// Re-export of [`minijinja`] for callers who need to construct
/// [`minijinja::value::Value`]s for `RenderOptions::with_extra`.
pub use minijinja;

pub(crate) mod grammar_compile;
#[doc(hidden)]
pub use grammar_compile::{emit_until_rules, schema_to_gbnf, JSON_GRAMMAR};

mod tool_choice;
pub use tool_choice::{
    build_grammar_source_for_debug, deferred_grammar_for_prompt,
    grammar_for_prompt, grammar_for_tool_choice, ToolChoiceError,
    ToolChoiceOptions,
};

pub mod output_config;
pub use output_config::{
    compile_output_config, compile_prompt_output_config,
    grammar_for_output_config, CompiledOutputConfig, OutputConfigError,
    OutputConfigOptions,
};

pub mod dialect;
pub use dialect::CallSyntax;

#[cfg(feature = "llama-cpp")]
mod llama_cpp;
#[cfg(feature = "llama-cpp")]
pub use llama_cpp::{
    llama_quantize, DecodeError, FlashAttention, LlamaCppBackend,
    LlamaCppDecoder, LlamaCppEngine, LlamaCppModel, NewError,
};
#[cfg(feature = "mtmd")]
pub use llama_cpp::{Mtmd, MtmdParams};

#[cfg(feature = "llama-cpp")]
pub mod log;
#[cfg(feature = "llama-cpp")]
pub use log::{
    clear_log_callback, restore_default_logs, set_log_callback, silence_logs,
    LogLevel,
};

#[cfg(all(feature = "moeflux", target_os = "macos"))]
pub mod moeflux;
#[cfg(all(feature = "moeflux", target_os = "macos"))]
pub use moeflux::{
    MoefluxBackend, MoefluxDecoder, MoefluxEngine, MoefluxError, MoefluxModel,
    MoefluxModelError,
};

#[cfg(any(
    feature = "llama-cpp",
    all(feature = "moeflux", target_os = "macos")
))]
pub(crate) mod snapshot_store;

mod ngram;
pub use ngram::{NGram, NGramData, NGramStats};

// The module itself is feature-free: the chat-template sidecar is
// raw Jinja and must work without `toml`. Only the TOML-encoded
// sidecars (sampling, dialect) are gated.
pub mod sidecar;
#[cfg(feature = "toml")]
pub use sidecar::{
    load_call_syntax, load_sample_options, seed_config_for, write_call_syntax,
    write_sample_options,
};
pub use sidecar::{load_template_source, SidecarError};

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
#[cfg(feature = "tokio")]
pub use session::FromPath;
#[cfg(feature = "llama-cpp")]
pub use session::LlamaCppSession;
#[cfg(all(
    feature = "tokio",
    any(feature = "llama-cpp", all(feature = "moeflux", target_os = "macos"))
))]
pub use session::SessionTransport;
#[cfg(any(
    feature = "llama-cpp",
    all(feature = "moeflux", target_os = "macos")
))]
pub use session::{
    BlockStream, PrefixCacheConfig, Session, SessionError, TokenTrace,
    TopKEntry,
};

mod probability;
pub use probability::{InvalidProbability, Probability};

pub const TOS: &str = include_str!("../TERMS_OF_USE.md");
