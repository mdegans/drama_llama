//! Chat prompt primitives, re-exported from [`misanthropic`].
//!
//! drama_llama uses Anthropic-style message primitives as the source of
//! truth for chat transcripts: [`Message`] / [`Content`] / [`Block`] /
//! [`Role`]. They handle text, thinking, images, and tool-use/result blocks
//! — the same shape downstream apps use when talking to the Anthropic API
//! or to an OpenAI-compatible endpoint. drama_llama itself currently
//! renders only text and thought blocks through chat templates (see
//! [`crate::ChatTemplate`]); other blocks can be added as the surrounding
//! infra needs them.
//!
//! # Why not OpenAI shape?
//!
//! The Anthropic shape is strictly more expressive: messages can carry
//! multi-part content with discriminated blocks (including signed thought
//! blocks), which OpenAI messages can't natively. Converting
//! Anthropic-shape → OpenAI-shape is trivial; the reverse is lossy.
//!
//! [`misanthropic`]: https://github.com/mdegans/misanthropic

use std::borrow::Cow;

pub use misanthropic::prompt::message::{Block, Content, Message, Role};
pub use misanthropic::tool::{Method as Tool, Use as ToolUse};

/// A chat prompt: an optional system prompt plus a message transcript.
///
/// This is drama_llama's thin wrapper around misanthropic's message
/// primitives. It intentionally omits the request-level fields of
/// [`misanthropic::Prompt`] (model id, max_tokens, temperature, etc.) —
/// those are either carried by [`crate::PredictOptions`] or irrelevant to
/// a local inference engine.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Prompt<'a> {
    /// System prompt. May be `None`, a single text blob, or a multi-part
    /// content block (for models that support structured system prompts).
    pub system: Option<Content<'a>>,
    /// Chat transcript. User/Assistant messages in chronological order.
    /// A typical live turn ends with a user message and the template is
    /// asked to append an empty assistant header (see
    /// [`ChatTemplate::render`]).
    ///
    /// [`ChatTemplate::render`]: crate::ChatTemplate::render
    pub messages: Vec<Message<'a>>,
    /// Function/tool definitions to advertise to the model. Templates
    /// that support tool calling (Llama 3.1, Qwen, etc.) JSON-serialize
    /// these into the rendered prompt. `None` means "don't mention
    /// tools" — equivalent to the template's `tools is none` branch.
    pub tools: Option<Vec<Tool<'a>>>,
}

impl<'a> Prompt<'a> {
    /// Build an empty prompt.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the system prompt from any text-like value.
    pub fn with_system<S>(mut self, system: S) -> Self
    where
        S: Into<Cow<'a, str>>,
    {
        self.system = Some(Content::SinglePart(system.into()));
        self
    }

    /// Append a user message with single-part text content.
    pub fn push_user<S>(mut self, text: S) -> Self
    where
        S: Into<Cow<'a, str>>,
    {
        self.messages.push(Message {
            role: Role::User,
            content: Content::SinglePart(text.into()),
        });
        self
    }

    /// Append an assistant message with single-part text content.
    pub fn push_assistant<S>(mut self, text: S) -> Self
    where
        S: Into<Cow<'a, str>>,
    {
        self.messages.push(Message {
            role: Role::Assistant,
            content: Content::SinglePart(text.into()),
        });
        self
    }

    /// Append an already-constructed message.
    pub fn push_message(mut self, message: Message<'a>) -> Self {
        self.messages.push(message);
        self
    }

    /// Attach a tool/function definition. Repeat to add more.
    pub fn push_tool(mut self, tool: Tool<'a>) -> Self {
        self.tools.get_or_insert_with(Vec::new).push(tool);
        self
    }

    /// Convert to a `'static` lifetime by cloning all borrowed data.
    pub fn into_static(self) -> Prompt<'static> {
        Prompt {
            system: self.system.map(|c| c.into_static()),
            messages: self
                .messages
                .into_iter()
                .map(|m| m.into_static())
                .collect(),
            tools: self.tools.map(|ts| {
                ts.into_iter().map(|t| tool_into_static(t)).collect()
            }),
        }
    }
}

/// Shift a [`Tool`] to `'static` by cloning its Cow fields.
fn tool_into_static(tool: Tool<'_>) -> Tool<'static> {
    Tool {
        name: Cow::Owned(tool.name.into_owned()),
        description: Cow::Owned(tool.description.into_owned()),
        schema: tool.schema,
        cache_control: tool.cache_control,
    }
}
