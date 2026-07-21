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
//! | `signature` (on [`Block::Thought`]) | *repurposed* — see [`OPEN_THOUGHT_SIGNATURE`] |
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

/// Marks a [`Block::Thought`] as **open** — the reasoning block was
/// never closed, so no close marker (`</think>`, `<|end|>`, …) may be
/// rendered for it.
///
/// Two ways a thought ends up open:
///
/// * the model ran out of `max_tokens` mid-reasoning, and
/// * the caller *prefilled* one deliberately, to bootstrap or steer a
///   chain of thought ([`open_thought`]).
///
/// Either way the invariant is the same: an open thought is legal only
/// as the **sole block of the trailing assistant message**, because
/// that is exactly the shape expressible as a suffix of the generation
/// prompt. Anything else is rejected at ingest — see
/// [`prune_open_thoughts`] for the remedy.
///
/// # This field is hijacked, deliberately
///
/// `signature` is Anthropic's tamper-proofing for thoughts, and is
/// meaningless for local models. We overload it as a provider-scoped
/// flag rather than extend `misanthropic`'s schema (it models the
/// Anthropic wire format; this is not part of it) and rather than put
/// a marker in the thought *body*, which would be framing bytes inside
/// content — the sentinel-string-in-content bug class (issue #55), and
/// fatal to byte-stable re-rendering.
///
/// The overload is safe in both directions: we never hand-craft a
/// thought *to* Anthropic, and any thought that arrives *from*
/// Anthropic is closed and carries a real base64 signature, which can
/// never equal this sentinel.
///
/// # Polarity is inverted vs `misanthropic`'s own helpers
///
/// [`Block::is_complete_thought`] and
/// [`Message::remove_incomplete_thought`] read an **empty** signature
/// as "incomplete". We read empty as *closed* and this sentinel as
/// *open*, because that fails safe: mis-flagging a closed thought as
/// open makes the renderer omit a close the model really wrote, while
/// the converse merely loses a continuation. Neither upstream helper
/// is used by drama_llama; if you call one on a local prompt it will
/// do the opposite of what you want. Use [`prune_open_thoughts`].
///
/// [`Block::is_complete_thought`]: misanthropic::prompt::message::Block::is_complete_thought
/// [`Message::remove_incomplete_thought`]: misanthropic::prompt::message::Message::remove_incomplete_thought
pub const OPEN_THOUGHT_SIGNATURE: &str = "drama_llama:open-thought";

/// Build an **open** [`Block::Thought`] — a reasoning block the
/// renderer will emit *without* a close marker, so the model continues
/// from it.
///
/// Place it as the sole block of the trailing assistant message. See
/// [`OPEN_THOUGHT_SIGNATURE`] for the invariant and the rationale.
///
/// ```
/// use drama_llama::prompt::{is_open_thought, open_thought};
///
/// let seed = open_thought("Let me re-read the policy before answering.");
/// assert!(is_open_thought(&seed));
/// ```
pub fn open_thought(
    thought: impl Into<std::borrow::Cow<'static, str>>,
) -> Block {
    Block::Thought {
        thought: thought.into(),
        signature: std::borrow::Cow::Borrowed(OPEN_THOUGHT_SIGNATURE),
    }
}

/// Whether `block` is a [`Block::Thought`] flagged open by
/// [`OPEN_THOUGHT_SIGNATURE`].
pub fn is_open_thought(block: &Block) -> bool {
    matches!(
        block,
        Block::Thought { signature, .. }
            if signature == OPEN_THOUGHT_SIGNATURE
    )
}

/// Drop every open thought from `prompt`, returning how many were
/// removed. A message left with no blocks is removed too.
///
/// This is the remedy when a prompt is rejected for carrying an open
/// thought somewhere it cannot be rendered byte-exactly: prune and
/// resubmit. Cost is one turn's worth of prefix-cache miss, not a
/// corrupted transcript.
///
/// Note this is *not* `misanthropic`'s
/// [`Message::remove_incomplete_thought`] — that reads the opposite
/// polarity (see [`OPEN_THOUGHT_SIGNATURE`]) and would strip closed
/// thoughts while keeping open ones.
///
/// [`Message::remove_incomplete_thought`]: misanthropic::prompt::message::Message::remove_incomplete_thought
pub fn prune_open_thoughts(prompt: &mut Prompt) -> usize {
    let mut pruned = 0;
    for message in prompt.messages.iter_mut() {
        let before = message.content.0.len();
        message.content.0.retain(|b| !is_open_thought(b));
        pruned += before - message.content.0.len();
    }
    prompt.messages.retain(|m| !m.content.0.is_empty());
    pruned
}
