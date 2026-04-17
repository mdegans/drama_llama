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
//! | `functions`        | [`ChatTemplate`] (tools) + tool_choice   |
//! | `tool_choice`      | [`grammar_for_prompt`] grammar compiler  |
//! | `stop_sequences`   | callers wire into [`PredictOptions`]     |
//!
//! Request-level fields (`model` id, `max_tokens`, `temperature`,
//! `stream`, `top_k`, `top_p`, `metadata`) are ignored locally — use
//! [`PredictOptions`] and [`SampleOptions`] for the local equivalents.
//!
//! [`Prompt`]: misanthropic::Prompt
//! [`ChatTemplate`]: crate::ChatTemplate
//! [`grammar_for_prompt`]: crate::grammar_for_prompt
//! [`PredictOptions`]: crate::PredictOptions
//! [`SampleOptions`]: crate::SampleOptions

pub use misanthropic::prompt::message::{Block, Content, Message, Role};
pub use misanthropic::tool::{
    Choice as ToolChoice, Method as Tool, Result as ToolResult, Use as ToolUse,
};
pub use misanthropic::Prompt;
