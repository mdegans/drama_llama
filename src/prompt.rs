//! Chat prompt primitives, re-exported wholesale from [`misanthropic`].
//!
//! drama_llama uses misanthropic's [`Prompt`] directly as its source of
//! truth for chat state. This is the same type downstream apps use when
//! talking to the Anthropic Messages API or to an OpenAI-compatible
//! endpoint, so there's no conversion layer between local inference and
//! cloud inference — the same builder, the same JSON on the wire.
//!
//! # Fields relevant to local inference
//!
//! misanthropic's `Prompt` carries more than a local engine needs. The
//! subset drama_llama reads is:
//!
//! | Field              | Read by                                  |
//! |--------------------|------------------------------------------|
//! | `system`           | [`ChatTemplate`] rendering               |
//! | `messages`         | [`ChatTemplate`] rendering               |
//! | `tools`            | [`ChatTemplate`] (tools) + tool_choice   |
//! | `tool_choice`      | [`grammar_for_prompt`] grammar compiler  |
//! | `stop_sequences`   | callers wire into [`PredictOptions`]     |
//! | `thinking`         | [`ChatTemplate`] — drives `enable_thinking` extra |
//!
//! Request-level fields (`model` id, `max_tokens`, `temperature`,
//! `stream`, `top_k`, `top_p`, `metadata`) are ignored locally — use
//! [`PredictOptions`] and [`SamplerConfig`] for the local equivalents.
//!
//! [`Prompt`]: misanthropic::Prompt
//! [`ChatTemplate`]: crate::ChatTemplate
//! [`grammar_for_prompt`]: crate::grammar_for_prompt
//! [`PredictOptions`]: crate::PredictOptions
//! [`SamplerConfig`]: crate::SamplerConfig

// misanthropic ≥1.0.0-alpha.2 dropped the pervasive `'a` lifetime from
// its public API — everything is owned (`Cow<'static, _>`) now, so these
// are plain re-exports rather than `<'static>`-pinning aliases.
pub use misanthropic::prompt::message::Role;
pub use misanthropic::tool::Choice as ToolChoice;

// Prompt types. Prefer CachedPrompt for append-only flow (cache friendly).
pub use misanthropic::prompt::cached::CachedPrompt;
pub use misanthropic::Prompt;

// Typed messages, useful for avoiding turn order errors
pub use misanthropic::prompt::{AssistantMessage, UserMessage};

// Message and content
pub use misanthropic::prompt::message::{Block, Content, Message};

// Tool stuff. `Tool` is the *custom* (client-executed) definition — the
// only kind drama_llama can enforce a grammar for. `Prompt::tools` holds
// `MethodDef`s, which may also be server tools; see `grammar_for_prompt`
// for the narrowing.
pub type Tool = misanthropic::tool::CustomMethodDef;
pub type ToolResult = misanthropic::tool::Result;
pub type ToolUse = misanthropic::tool::Use;

// Api types and Usage stats
pub type MessageResponse = misanthropic::response::Message;
pub use misanthropic::client::AnthropicError;
pub use misanthropic::response::Usage;
