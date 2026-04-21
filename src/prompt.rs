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

// Types with a lifetime parameter are aliased to `'static` so the rest of
// the crate doesn't have to thread `<'_>` through every signature. The
// underlying misanthropic types still carry `Cow<'a, _>` fields — we just
// commit to owned data at the drama_llama boundary. If you genuinely need
// a borrowed variant, reach for the fully-qualified misanthropic path.
pub use misanthropic::prompt::message::Role;
pub use misanthropic::tool::Choice as ToolChoice;

// Prompt types. Prefer CachedPrompt for append-only flow (cache friendly).
pub type Prompt = misanthropic::Prompt<'static>;
pub type CachedPrompt = misanthropic::prompt::cached::CachedPrompt<'static>;

// Typed messages, useful for avoiding turn order errors
pub type AssistantMessage = misanthropic::prompt::AssistantMessage<'static>;
pub type UserMessage = misanthropic::prompt::UserMessage<'static>;

// Message and content
pub type Message = misanthropic::prompt::message::Message<'static>;
pub type Content = misanthropic::prompt::message::Content<'static>;
pub type Block = misanthropic::prompt::message::Block<'static>;

// Tool stuff
pub type Tool = misanthropic::tool::Method<'static>;
pub type ToolResult = misanthropic::tool::Result<'static>;
pub type ToolUse = misanthropic::tool::Use<'static>;

// Api types and Usage stats
pub type MessageResponse = misanthropic::response::Message<'static>;
pub type AnthropicError = misanthropic::client::AnthropicError;
pub type Usage = misanthropic::response::Usage;
