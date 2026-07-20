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
//! | `max_tokens`       | [`Session`] — the sole generation cap    |
//! | `temperature` / `top_p` / `top_k` | [`Session`] — folded into the sampling chain |
//!
//! The remaining request-level fields (`model` id, `stream`,
//! `metadata`) are ignored locally — use [`PredictOptions`] and
//! [`SamplerConfig`] for the local equivalents.
//!
//! # Sampling precedence
//!
//! `temperature` / `top_p` / `top_k` are honored, layered over the
//! per-model sidecar and the model's own metadata:
//!
//! ```text
//! request temperature/top_p/top_k     (per-call)
//!   └─ <model>.sampling.toml sidecar  (per-model, editable)
//!        └─ general.sampling.* GGUF metadata  (seeds the sidecar)
//!             └─ SamplerConfig::default()
//! ```
//!
//! A request that names knobs already present exactly once in the
//! chain retunes them in place; anything else rebuilds the chain in
//! canonical order, falling back to the model's recommendation for
//! whatever the request left unset. See
//! [`apply_request_sampling`](crate::apply_request_sampling) for the
//! full rule and [`sidecar`](crate::sidecar) for the seeding half.
//!
//! Each tier is optional and drops through to the next. A model that
//! advertises no `general.sampling.*` metadata (gpt-oss, and every
//! moeflux model without a per-variant constant) simply seeds its
//! sidecar from [`SamplerConfig::default`] — the behavior before
//! this tier existed. A request that sets no sampling fields leaves
//! the chain exactly as configured.
//!
//! `temperature: 0.0` collapses to argmax, matching `llama.cpp`,
//! OpenAI, and Anthropic. Note that repetition penalties still apply
//! at `0.0` — it is greedy-with-penalty, not raw greedy — which stays
//! deterministic but is not the same distribution as
//! [`SamplerConfig::greedy`].
//!
//! An out-of-range `top_p` is a typed error
//! ([`SessionError::RequestTopP`]), never a silent clamp.
//!
//! [`Prompt`]: misanthropic::Prompt
//! [`ChatTemplate`]: crate::ChatTemplate
//! [`grammar_for_prompt`]: crate::grammar_for_prompt
//! [`PredictOptions`]: crate::PredictOptions
//! [`SamplerConfig`]: crate::SamplerConfig
//! [`SamplerConfig::default`]: crate::SamplerConfig::default
//! [`SamplerConfig::greedy`]: crate::SamplerConfig::greedy
//! [`Session`]: crate::Session
//! [`SessionError::RequestTopP`]: crate::SessionError::RequestTopP

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
