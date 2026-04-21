//! Jinja-based chat templating.
//!
//! Most modern GGUF models embed their chat template under the
//! `tokenizer.chat_template` metadata key. [`ChatTemplate`] compiles that
//! template with [`minijinja`] and renders a [`Prompt`] into the exact byte
//! sequence the model was trained on — Llama 3.1, Qwen, Mistral,
//! Gemma, etc., without per-model Rust code.
//!
//! # Scope
//!
//! drama_llama renders [`Block::Text`], [`Block::Thought`],
//! [`Block::ToolUse`], and [`Block::ToolResult`] blocks. Tool calls emit
//! as `{role, content, tool_calls: [{id, function: {name, arguments}}]}`
//! messages; tool results emit as standalone `{role: "tool", ...}`
//! messages between user turns. Images and redacted-thought blocks are
//! skipped until the surrounding infrastructure needs them.
//!
//! Tool definitions come from [`Prompt::tools`] and surface in the
//! template as the `tools` variable. Llama 3.1 / Mistral / Qwen all
//! JSON-serialize tools via the `tojson` filter, which is enabled here
//! via `minijinja`'s `json` feature.
//!
//! # Extras
//!
//! [`RenderOptions`] lets callers push additional template variables
//! (e.g. `tools_in_user_message`, `builtin_tools`, `custom_tools`). It
//! also carries an optional `date_string`; if omitted, drama_llama
//! defaults to today's UTC date in HF's `%d %b %Y` format so templates
//! that unconditionally concatenate `"Today Date: " + date_string` do
//! not blow up.
//!
//! # Dialect compatibility
//!
//! HuggingFace chat templates are written in Jinja2 but use a narrow
//! subset. [`minijinja`] covers that subset. The templater exposes a few
//! functions that HF templates commonly call:
//!
//! * `raise_exception(msg)` — surfaces as a render error
//! * `strftime_now(fmt)` — current UTC time, formatted via chrono-like
//!   `%Y-%m-%d` etc. (minimal subset)
//!
//! [`Prompt`]: crate::Prompt
//! [`Block::Text`]: crate::Block
//! [`Block::Thought`]: crate::Block

use std::{borrow::Cow, collections::BTreeMap, sync::Arc};

use llama_cpp_sys_3::llama_token;
use minijinja::{value::Value as JinjaValue, Environment, Error as JinjaError};
use serde::Serialize;

use crate::{prompt::Tool, Block, Content, Model, Prompt, Role};

/// Render a [`Tool`] as the OpenAI wire envelope cogito / Qwen /
/// Hermes-family chat templates expect.
///
/// These templates do `{{ tool | tojson }}` and feed the resulting
/// JSON straight to the model. The training data — produced by
/// ollama's Go runtime — always takes the shape
/// `{"type": "function", "function": {"name": ..., "description":
/// ..., "parameters": ...}}`. misanthropic's [`tool::Method`]
/// serializes in Anthropic shape (`input_schema` rather than
/// `parameters`, no wire envelope), so we adapt through this
/// function before handing off to minijinja.
///
/// Key order within the JSON object is irrelevant for the model
/// (minijinja alphabetizes on output anyway); the structural shape
/// and field names are what matter.
///
/// [`tool::Method`]: misanthropic::tool::Method
fn tool_wire_value(tool: &Tool) -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": tool.name.as_ref(),
            "description": tool.description.as_ref(),
            "parameters": &tool.schema,
        }
    })
}

/// A compiled chat template tied to a specific model's tokens.
///
/// Load via [`ChatTemplate::from_model`] (reads the template string plus
/// BOS/EOS tokens from GGUF metadata) or [`ChatTemplate::from_source`] for
/// manual control.
#[derive(Clone)]
pub struct ChatTemplate {
    /// Shared `Environment` so clones are cheap. Environments are
    /// thread-safe but clone-by-Arc so we don't reparse on each clone.
    env: Arc<Environment<'static>>,
    /// Name under which `env` registered the template. Always "chat".
    template_name: &'static str,
    /// BOS piece (e.g. `<|begin_of_text|>` for Llama 3.1). Passed to the
    /// template as `bos_token`.
    bos_token: String,
    /// EOS piece (e.g. `<|end_of_text|>`). Passed as `eos_token`.
    eos_token: String,
}

impl std::fmt::Debug for ChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatTemplate")
            .field("bos_token", &self.bos_token)
            .field("eos_token", &self.eos_token)
            .finish_non_exhaustive()
    }
}

impl ChatTemplate {
    /// Load the chat template from a GGUF model.
    ///
    /// Reads `tokenizer.chat_template` for the template source and
    /// renders BOS/EOS pieces from the model's token ids.
    pub fn from_model(model: &Model) -> Result<Self, ChatTemplateError> {
        let source = model
            .get_meta("tokenizer.chat_template")
            .ok_or(ChatTemplateError::NoTemplate)?;
        let bos = model.token_to_piece(model.bos());
        let eos = model.token_to_piece(model.eos());
        Self::from_source(source, bos, eos)
    }

    /// Compile a chat template from a raw Jinja source string plus
    /// BOS/EOS piece strings.
    pub fn from_source(
        source: String,
        bos_token: String,
        eos_token: String,
    ) -> Result<Self, ChatTemplateError> {
        let mut env = Environment::new();
        env.set_unknown_method_callback(
            minijinja_contrib::pycompat::unknown_method_callback,
        );
        env.add_function("raise_exception", raise_exception);
        env.add_function("strftime_now", strftime_now);
        env.add_template_owned("chat", source)
            .map_err(ChatTemplateError::from_jinja)?;
        Ok(Self {
            env: Arc::new(env),
            template_name: "chat",
            bos_token,
            eos_token,
        })
    }

    /// BOS piece the template will prepend if its logic calls for it.
    pub fn bos_token(&self) -> &str {
        &self.bos_token
    }

    /// EOS piece.
    pub fn eos_token(&self) -> &str {
        &self.eos_token
    }

    /// Render a [`Prompt`] with default options.
    ///
    /// `add_generation_prompt` controls whether the template appends an
    /// empty assistant header so the model generates the next turn. Set
    /// this to `true` for live chat, `false` when tokenizing a stored
    /// transcript. Shorthand for
    /// [`ChatTemplate::render_with`] with defaults.
    pub fn render(
        &self,
        prompt: &Prompt,
        add_generation_prompt: bool,
    ) -> Result<String, ChatTemplateError> {
        self.render_with(
            prompt,
            &RenderOptions {
                add_generation_prompt,
                ..RenderOptions::default()
            },
        )
    }

    /// Render a [`Prompt`] with explicit [`RenderOptions`].
    ///
    /// Use this when the template needs template-specific variables
    /// beyond `messages` / `tools` / `bos_token` — for example Llama 3.1's
    /// `date_string`, `tools_in_user_message`, or `builtin_tools`.
    pub fn render_with(
        &self,
        prompt: &Prompt,
        opts: &RenderOptions,
    ) -> Result<String, ChatTemplateError> {
        let messages = build_messages(prompt);
        let tools_value = match prompt.functions.as_ref() {
            Some(ts) if !ts.is_empty() => {
                let wire: Vec<serde_json::Value> =
                    ts.iter().map(tool_wire_value).collect();
                JinjaValue::from_serialize(&wire)
            }
            _ => JinjaValue::from(()), // renders as None / null
        };
        // Default `date_string` to today in HF's "%d %b %Y" format when
        // the caller didn't supply one. The template unconditionally
        // concatenates `"Today Date: " + date_string + ...`, so passing
        // `none` would blow up with a string-plus-none type error.
        let date_string = opts.date_string.clone().unwrap_or_else(|| {
            format_strftime_subset("%d %b %Y", current_unix_secs())
        });
        let base_ctx = minijinja::context! {
            bos_token => &self.bos_token,
            eos_token => &self.eos_token,
            messages => messages,
            tools => tools_value,
            add_generation_prompt => opts.add_generation_prompt,
            date_string => date_string,
        };
        // Merge caller-supplied extras on top of the base context.
        let ctx = if opts.extras.is_empty() {
            base_ctx
        } else {
            let mut extras: BTreeMap<String, JinjaValue> = BTreeMap::new();
            for (k, v) in &opts.extras {
                extras.insert(k.clone(), v.clone());
            }
            minijinja::context!(
                ..base_ctx,
                ..JinjaValue::from_serialize(&extras)
            )
        };
        let tmpl = self
            .env
            .get_template(self.template_name)
            .map_err(ChatTemplateError::from_jinja)?;
        tmpl.render(ctx).map_err(ChatTemplateError::from_jinja)
    }

    /// Render the prompt plus one partial render per `cache_control`
    /// breakpoint.
    ///
    /// Use this when a caller — e.g. [`Session`] — needs to know where
    /// in the tokenized output the caller's cache breakpoints land so a
    /// later call can compute cache-reuse boundaries in tokens. See
    /// [`RenderedWithBreakpoints`] for the returned shape.
    ///
    /// Partial renders force `add_generation_prompt = false` regardless
    /// of what the caller set on `opts` — only the full render honors
    /// the caller's preference. If `prompt` carries no cache markers,
    /// the returned `partial_texts` is empty and the full render is
    /// equivalent to [`render_with`](Self::render_with).
    ///
    /// [`Session`]: crate::Session
    pub fn render_with_breakpoints(
        &self,
        prompt: &Prompt,
        opts: &RenderOptions,
    ) -> Result<RenderedWithBreakpoints, ChatTemplateError> {
        let text = self.render_with(prompt, opts)?;
        let breakpoints = collect_breakpoints(prompt);
        let mut partial_texts = Vec::with_capacity(breakpoints.len());
        for bp in breakpoints {
            partial_texts.push(render_partial(self, prompt, opts, bp)?);
        }
        Ok(RenderedWithBreakpoints {
            text,
            partial_texts,
        })
    }
}

/// Rendered prompt plus one partial render per `cache_control`
/// breakpoint.
///
/// Used by [`Session`] (Phase 3 of prompt caching) to compute which
/// prefix of the new prompt's token stream matches a previously-cached
/// state. Each partial is rendered with `add_generation_prompt=false`;
/// each should tokenize to a strict prefix of [`text`](Self::text) for
/// any well-behaved chat template.
///
/// `partial_texts` is ordered canonically by how much of the prompt is
/// included: [`AfterTools`], then [`AfterSystem`], then
/// [`AfterMessage(0)`], [`AfterMessage(1)`], … Only breakpoints that
/// actually exist in the prompt appear.
///
/// [`Session`]: crate::Session
/// [`AfterTools`]: Breakpoint::AfterTools
/// [`AfterSystem`]: Breakpoint::AfterSystem
/// [`AfterMessage(0)`]: Breakpoint::AfterMessage
/// [`AfterMessage(1)`]: Breakpoint::AfterMessage
#[derive(Clone, Debug)]
pub struct RenderedWithBreakpoints {
    /// Full render — equivalent to
    /// [`ChatTemplate::render_with`]'s output.
    pub text: String,
    /// One partial render per breakpoint, in canonical order.
    pub partial_texts: Vec<String>,
}

/// Options passed to [`ChatTemplate::render_with`].
///
/// Variables that are universal to HuggingFace-style templates
/// (`messages`, `tools`, `bos_token`, `eos_token`,
/// `add_generation_prompt`) are sourced from [`Prompt`] and the
/// [`ChatTemplate`] itself. Everything else — template-specific flags
/// like Llama 3.1's `tools_in_user_message`, date overrides, or
/// `builtin_tools` — goes through [`extras`](Self::extras).
#[derive(Clone, Debug, Default)]
pub struct RenderOptions {
    /// Ask the template to append an empty assistant header so the model
    /// generates the next turn. True for live chat, false for
    /// tokenizing a stored transcript.
    pub add_generation_prompt: bool,
    /// Current date string (e.g. `"17 Apr 2026"`). Llama 3.1's template
    /// reads `date_string` when stamping a system-message header. If
    /// `None`, the template's default (static fallback) is used.
    pub date_string: Option<String>,
    /// Template-specific extra variables. Keys become top-level names in
    /// the Jinja context. Values are arbitrary Serialize-able data.
    pub extras: Vec<(String, JinjaValue)>,
}

impl RenderOptions {
    /// Builder: set `add_generation_prompt`.
    pub fn with_generation_prompt(mut self, yes: bool) -> Self {
        self.add_generation_prompt = yes;
        self
    }

    /// Builder: set `date_string`.
    pub fn with_date<S>(mut self, date: S) -> Self
    where
        S: Into<String>,
    {
        self.date_string = Some(date.into());
        self
    }

    /// Builder: add an arbitrary `(key, value)` pair to the Jinja
    /// context. Useful for `tools_in_user_message`, `builtin_tools`,
    /// etc. Any [`serde::Serialize`] value works — numbers, strings,
    /// booleans, structs, `serde_json::Value`, etc.
    pub fn with_extra<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Serialize,
    {
        self.extras
            .push((key.into(), JinjaValue::from_serialize(&value)));
        self
    }
}

// ===========================================================================
// Cache-breakpoint discovery
// ===========================================================================

/// A position in a [`Prompt`] where the caller placed a
/// `cache_control` marker.
///
/// Ordered chronologically in the rendered output: [`AfterTools`] (if
/// any tool is cached) comes before [`AfterSystem`] (if any system
/// block is cached), which comes before [`AfterMessage(i)`], and
/// later messages come after earlier ones. This matches the order
/// of the partial-render truncations — each covers strictly more of
/// the prompt than the previous — which is what gives us the
/// tokens-are-a-prefix property downstream.
///
/// Which variant a particular template renders "first" in bytes is
/// template-specific (Llama 3.1 renders tools inside the system
/// header; cogito renders them between system and user). We don't
/// care: we always truncate by prompt structure (tools, system,
/// messages), and the partial is a prefix of the full if and only if
/// the template is well-behaved. For templates where a coarse-
/// grained tools-only render isn't a byte-prefix (because the tool
/// list's closing `]` lands differently), we silently drop that
/// breakpoint at tokenization time.
///
/// [`AfterTools`]: Breakpoint::AfterTools
/// [`AfterSystem`]: Breakpoint::AfterSystem
/// [`AfterMessage(i)`]: Breakpoint::AfterMessage
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Breakpoint {
    /// After the tools section — any cache marker on any
    /// [`tool::Method`](misanthropic::tool::Method) in
    /// [`Prompt::functions`] produces this, regardless of which
    /// specific method was marked. Coarse by design: per-tool
    /// partial rendering would not produce a byte-prefix of the full
    /// render (the closing `]` of the tools JSON array lands
    /// differently).
    AfterTools,
    /// After the system section — any cache marker on any block in
    /// [`Prompt::system`] produces this. Coarse by design for the
    /// same reason as [`AfterTools`].
    AfterSystem,
    /// After message index `i`, inclusive. Emitted iff
    /// `prompt.messages[i]`'s content has at least one block with
    /// `cache_control` set (via [`Block::is_cached`]). This is the
    /// fine-grained case that matters for the Agora reactor workload
    /// — N agents sharing a system + tools + early messages,
    /// diverging only in the last turn.
    ///
    /// [`Block::is_cached`]: misanthropic::prompt::message::Block::is_cached
    AfterMessage(usize),
}

/// Walk `prompt` and return the ordered list of cache breakpoints it
/// declares. See [`Breakpoint`] for the ordering rule and granularity.
fn collect_breakpoints(prompt: &Prompt) -> Vec<Breakpoint> {
    let mut out = Vec::new();

    let tools_cached = prompt
        .functions
        .as_ref()
        .is_some_and(|fns| fns.iter().any(|m| m.cache_control.is_some()));
    if tools_cached {
        out.push(Breakpoint::AfterTools);
    }

    let system_cached =
        prompt.system.as_ref().is_some_and(|c| content_has_cache(c));
    if system_cached {
        out.push(Breakpoint::AfterSystem);
    }

    for (i, m) in prompt.messages.iter().enumerate() {
        if content_has_cache(&m.content) {
            out.push(Breakpoint::AfterMessage(i));
        }
    }

    out
}

/// Does any block inside `content` carry a `cache_control` marker?
///
/// [`Content::has_cache`] on misanthropic only reports `true` for
/// [`Content::MultiPart`] — `SinglePart` has no per-block
/// `cache_control` field to set, so this is equivalent. We spell it
/// out locally to keep the intent obvious at the call site.
///
/// [`Content::has_cache`]: misanthropic::prompt::message::Content::has_cache
/// [`Content::MultiPart`]: crate::Content::MultiPart
fn content_has_cache(content: &Content) -> bool {
    match content {
        Content::SinglePart(_) => false,
        Content::MultiPart(blocks) => blocks.iter().any(|b| b.is_cached()),
    }
}

/// Render `prompt` truncated at `up_to` with
/// `add_generation_prompt=false`. Used as the breakpoint-discovery
/// partial render; does not mutate `prompt` (clones a truncated view).
fn render_partial(
    template: &ChatTemplate,
    prompt: &Prompt,
    opts: &RenderOptions,
    up_to: Breakpoint,
) -> Result<String, ChatTemplateError> {
    let truncated = match up_to {
        Breakpoint::AfterTools => Prompt {
            functions: prompt.functions.clone(),
            system: None,
            messages: Vec::new(),
            ..Prompt::default()
        },
        Breakpoint::AfterSystem => Prompt {
            functions: prompt.functions.clone(),
            system: prompt.system.clone(),
            messages: Vec::new(),
            ..Prompt::default()
        },
        Breakpoint::AfterMessage(i) => Prompt {
            functions: prompt.functions.clone(),
            system: prompt.system.clone(),
            messages: prompt.messages[..=i].to_vec(),
            ..Prompt::default()
        },
    };
    let partial_opts = opts.clone().with_generation_prompt(false);
    template.render_with(&truncated, &partial_opts)
}

/// Tokenize the full render and each partial in `rendered`, returning
/// the full token stream plus the sorted, deduplicated breakpoint
/// token indices.
///
/// Each partial's tokens MUST be a prefix of the full's tokens —
/// that's what makes a breakpoint useful for KV-cache reuse. If a
/// partial's tokens are not a prefix (unexpected template weirdness,
/// e.g. a non-prefix-safe truncation point), the breakpoint is
/// silently dropped from the returned indices. We fail open to
/// uncached behavior for that call rather than erroring, so cache
/// oddities degrade performance rather than correctness.
///
/// Tokenizes with `parse_special=true` so chat markers
/// (`<|im_start|>`, `<|eot_id|>`, …) resolve to their single
/// special-token IDs — the same convention [`Session::prepare_call`]
/// uses.
///
/// Phase 3 (Session integration) will wire this in; external callers
/// don't need it yet, hence `pub(crate)`.
///
/// [`Session::prepare_call`]: crate::Session
pub(crate) fn tokenize_with_breakpoints(
    model: &Model,
    rendered: &RenderedWithBreakpoints,
) -> (Vec<llama_token>, Vec<usize>) {
    let full_tokens = model.tokenize(&rendered.text, true);
    let mut indices: Vec<usize> =
        Vec::with_capacity(rendered.partial_texts.len());
    for partial in &rendered.partial_texts {
        let partial_tokens = model.tokenize(partial, true);
        if partial_tokens.len() <= full_tokens.len()
            && full_tokens[..partial_tokens.len()] == partial_tokens[..]
        {
            indices.push(partial_tokens.len());
        } else {
            // Fail-open: drop the breakpoint. Logging is deliberately
            // omitted — drama_llama doesn't wire a tracing crate, and
            // a silent drop is the documented behavior.
        }
    }
    indices.sort_unstable();
    indices.dedup();
    (full_tokens, indices)
}

// ===========================================================================
// Prompt -> Jinja context conversion
// ===========================================================================

/// Build the `messages` sequence the template will iterate.
///
/// If `prompt.system` is set we synthesize a leading `system` message —
/// that matches the HF/Llama convention of carrying the system prompt as
/// the first message in the transcript.
///
/// Tool-calling messages are emitted in the shape HF templates expect:
///
/// * An assistant message with at least one [`Block::ToolUse`] becomes
///   `{role: "assistant", tool_calls: [{function: {name, arguments}}]}`.
///   Any accompanying text/thought is dropped — templates like
///   Llama 3.1's render tool calls in a branch that doesn't emit
///   `content`.
/// * A user message containing [`Block::ToolResult`] blocks is split:
///   each tool result emits a separate `{role: "tool", content: ...}`
///   message. Any remaining text in the same user turn follows as a
///   normal user message.
fn build_messages(prompt: &Prompt) -> Vec<JinjaValue> {
    let mut out: Vec<JinjaValue> =
        Vec::with_capacity(prompt.messages.len() + 1);
    if let Some(system) = prompt.system.as_ref() {
        out.push(text_message("system", flatten_text(system)));
    }
    for m in &prompt.messages {
        let role = match m.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        };
        append_message(&mut out, role, &m.content);
    }
    out
}

/// Emit one or more Jinja messages for a single misanthropic Message.
fn append_message(out: &mut Vec<JinjaValue>, role: &str, content: &Content) {
    let blocks: Vec<&Block> = match content {
        Content::SinglePart(text) => {
            out.push(text_message(role, text.to_string()));
            return;
        }
        Content::MultiPart(blocks) => blocks.iter().collect(),
    };

    // User turn: split ToolResult blocks into their own "tool" messages,
    // collect remaining text/thought into a trailing user message.
    if role == "user" {
        let mut residual = String::new();
        for b in &blocks {
            match b {
                Block::ToolResult { result } => {
                    let content = flatten_text(&result.content);
                    out.push(tool_result_message(&result.tool_use_id, content));
                }
                other => append_block_text(&mut residual, other),
            }
        }
        if !residual.is_empty() {
            out.push(text_message(role, residual));
        }
        return;
    }

    // Assistant turn: if any ToolUse, emit a tool-calling message. For a
    // single-call convention (what Llama 3.1 and Anthropic both use),
    // take the first ToolUse. Any surrounding thought is preserved as
    // content so templates that support <think>…</think> get signal.
    if let Some(call) = blocks.iter().find_map(|b| match b {
        Block::ToolUse { call } => Some(call),
        _ => None,
    }) {
        let mut residual = String::new();
        for b in &blocks {
            if matches!(b, Block::ToolUse { .. }) {
                continue;
            }
            append_block_text(&mut residual, b);
        }
        out.push(tool_call_message(role, &residual, call));
        return;
    }

    // No tool blocks — plain flattening.
    let mut flat = String::new();
    for b in &blocks {
        append_block_text(&mut flat, b);
    }
    out.push(text_message(role, flat));
}

fn text_message(role: &str, content: String) -> JinjaValue {
    minijinja::context! {
        role => role,
        content => content,
    }
}

/// Assistant message with a single `tool_calls` entry. Shape:
/// `{role, content, tool_calls: [{id, function: {name, arguments}}]}`.
///
/// Llama 3.1's template reads `.function.name` and `.function.arguments`
/// off each entry. OpenAI-style templates also look at `.id`, so we
/// include it. We intentionally omit the `type` field — templates that
/// need it default to `"function"`.
fn tool_call_message(
    role: &str,
    content: &str,
    call: &crate::prompt::ToolUse,
) -> JinjaValue {
    let tool_call = minijinja::context! {
        id => call.id.as_ref(),
        function => minijinja::context! {
            name => call.name.as_ref(),
            arguments => JinjaValue::from_serialize(&call.input),
        },
    };
    minijinja::context! {
        role => role,
        content => content,
        tool_calls => vec![tool_call],
    }
}

/// Tool-result message. Shape: `{role: "tool", tool_call_id, content}`.
/// `tool_call_id` matches what HF / OpenAI templates read; Llama 3.1's
/// template ignores it, but it's cheap to include.
fn tool_result_message(tool_use_id: &str, content: String) -> JinjaValue {
    minijinja::context! {
        role => "tool",
        tool_call_id => tool_use_id,
        content => content,
    }
}

/// Flatten any [`Content`] to a single string using [`append_block_text`]
/// for each part.
fn flatten_text(content: &Content) -> String {
    match content {
        Content::SinglePart(text) => text.to_string(),
        Content::MultiPart(blocks) => {
            let mut out = String::new();
            for b in blocks {
                append_block_text(&mut out, b);
            }
            out
        }
    }
}

/// Append a single block's user-visible text to `out`. Tool-use and
/// tool-result blocks are handled at the message level (see
/// [`append_message`]); here they contribute nothing.
fn append_block_text(out: &mut String, block: &Block) {
    match block {
        Block::Text { text, .. } => out.push_str(text),
        Block::Thought { thought, .. } => {
            out.push_str("<think>");
            out.push_str(thought);
            out.push_str("</think>");
        }
        Block::RedactedThought { .. }
        | Block::Image { .. }
        | Block::ToolUse { .. }
        | Block::ToolResult { .. } => {}
    }
}

// ===========================================================================
// Jinja-side helpers
// ===========================================================================

/// HF templates commonly call `raise_exception("msg")` to reject invalid
/// input. Surface that as a render-time error instead of panicking.
fn raise_exception(msg: Cow<'_, str>) -> Result<JinjaValue, JinjaError> {
    Err(JinjaError::new(
        minijinja::ErrorKind::InvalidOperation,
        format!("chat template raised: {msg}"),
    ))
}

/// Minimal `strftime_now` that returns current UTC time formatted via the
/// `time` crate's strftime-style format string. Enough for templates that
/// stamp `"%d %b %Y"` or `"%Y-%m-%d"`.
fn strftime_now(fmt: Cow<'_, str>) -> String {
    format_strftime_subset(&fmt, current_unix_secs())
}

fn current_unix_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

/// Handle the specifiers HF chat templates actually use: `%Y`, `%m`, `%d`,
/// `%b`, `%B`, `%H`, `%M`, `%S`. Passes other characters through.
fn format_strftime_subset(fmt: &str, unix_secs: i64) -> String {
    let (y, mo, d, h, mi, s) = civil_from_unix(unix_secs);
    let mut out = String::with_capacity(fmt.len() + 8);
    let mut chars = fmt.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('Y') => out.push_str(&format!("{y:04}")),
            Some('m') => out.push_str(&format!("{mo:02}")),
            Some('d') => out.push_str(&format!("{d:02}")),
            Some('H') => out.push_str(&format!("{h:02}")),
            Some('M') => out.push_str(&format!("{mi:02}")),
            Some('S') => out.push_str(&format!("{s:02}")),
            Some('b') => {
                out.push_str(MONTH_ABBR[(mo - 1).clamp(0, 11) as usize])
            }
            Some('B') => {
                out.push_str(MONTH_FULL[(mo - 1).clamp(0, 11) as usize])
            }
            Some('%') => out.push('%'),
            Some(other) => {
                out.push('%');
                out.push(other);
            }
            None => out.push('%'),
        }
    }
    out
}

const MONTH_ABBR: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
    "Nov", "Dec",
];
const MONTH_FULL: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];

/// Convert Unix seconds (UTC) into (year, month, day, hour, min, sec).
/// Proleptic Gregorian calendar; handles post-1970 dates.
fn civil_from_unix(secs: i64) -> (i32, i32, i32, i32, i32, i32) {
    // Split into days and time-of-day.
    let days = secs.div_euclid(86_400);
    let time_of_day = secs.rem_euclid(86_400);
    let h = time_of_day / 3600;
    let mi = (time_of_day % 3600) / 60;
    let s = time_of_day % 60;
    // Howard Hinnant's civil_from_days algorithm, epoch 1970-01-01.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    (y as i32, mo as i32, d as i32, h as i32, mi as i32, s as i32)
}

// ===========================================================================
// Errors
// ===========================================================================

#[derive(Debug, thiserror::Error)]
pub enum ChatTemplateError {
    #[error("model has no `tokenizer.chat_template` metadata")]
    NoTemplate,
    #[error("chat template error: {0}")]
    Jinja(String),
}

impl ChatTemplateError {
    fn from_jinja(err: JinjaError) -> Self {
        Self::Jinja(format!("{err:#}"))
    }
}

static_assertions::assert_impl_all!(ChatTemplateError: Send, Sync);

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Message;

    /// A Llama-3-style template, simplified. Verifies the basic control
    /// flow: BOS, system message, role headers, EOT markers, and the
    /// optional generation prompt.
    const LLAMA3_LIKE: &str = r#"{{ bos_token }}{% for m in messages %}<|start_header_id|>{{ m['role'] }}<|end_header_id|>

{{ m['content'] }}<|eot_id|>{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>

{% endif %}"#;

    fn simple_prompt() -> Prompt {
        Prompt::default()
            .set_system("You are helpful.")
            .add_message((Role::User, "Hi!"))
            .unwrap()
            .add_message((Role::Assistant, "Hello!"))
            .unwrap()
            .add_message((Role::User, "What is 2+2?"))
            .unwrap()
    }

    fn tmpl() -> ChatTemplate {
        ChatTemplate::from_source(
            LLAMA3_LIKE.to_owned(),
            "<|begin_of_text|>".to_owned(),
            "<|end_of_text|>".to_owned(),
        )
        .expect("template should compile")
    }

    #[test]
    fn renders_full_turn() {
        let out = tmpl().render(&simple_prompt(), true).unwrap();
        assert!(out.starts_with("<|begin_of_text|>"));
        assert!(out.contains("<|start_header_id|>system<|end_header_id|>\n\nYou are helpful.<|eot_id|>"));
        assert!(out.contains(
            "<|start_header_id|>user<|end_header_id|>\n\nHi!<|eot_id|>"
        ));
        assert!(out.contains(
            "<|start_header_id|>assistant<|end_header_id|>\n\nHello!<|eot_id|>"
        ));
        assert!(out.contains("<|start_header_id|>user<|end_header_id|>\n\nWhat is 2+2?<|eot_id|>"));
        assert!(
            out.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n")
        );
    }

    #[test]
    fn skips_generation_prompt_when_false() {
        let out = tmpl().render(&simple_prompt(), false).unwrap();
        assert!(
            !out.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n")
        );
    }

    #[test]
    fn omits_system_when_none() {
        let p = Prompt::default().add_message((Role::User, "hi")).unwrap();
        let out = tmpl().render(&p, false).unwrap();
        assert!(!out.contains("<|start_header_id|>system"));
        assert!(out.contains(
            "<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>"
        ));
    }

    #[test]
    fn thought_block_wraps_with_think_tags() {
        let mut content = Content::MultiPart(vec![
            Block::Thought {
                thought: "I should be concise.".into(),
                signature: "sig".into(),
            },
            Block::text("Hello!"),
        ]);
        let _ = &mut content; // appease borrow lint if any
        let p = Prompt {
            system: None,
            messages: vec![Message {
                role: Role::Assistant,
                content,
            }],
            functions: None,
            ..Default::default()
        };
        let out = tmpl().render(&p, false).unwrap();
        assert!(out.contains("<think>I should be concise.</think>Hello!"));
    }

    #[test]
    fn raise_exception_surfaces_as_error() {
        let src = r#"{{ raise_exception("boom") }}"#.to_owned();
        let t = ChatTemplate::from_source(src, "".into(), "".into()).unwrap();
        let err = t.render(&Prompt::default(), false).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("boom"), "error must surface message: {msg}");
    }

    #[test]
    fn strftime_now_renders_current_year() {
        let src = r#"{{ strftime_now("%Y-%m-%d") }}"#.to_owned();
        let t = ChatTemplate::from_source(src, "".into(), "".into()).unwrap();
        let out = t.render(&Prompt::default(), false).unwrap();
        // Loose assertion: must look like YYYY-MM-DD with current millennium.
        assert!(out.starts_with("20"), "expected 20YY-MM-DD, got {out}");
        assert_eq!(out.len(), 10);
    }

    #[test]
    fn multi_part_text_flattens() {
        let p = Prompt {
            system: None,
            messages: vec![Message {
                role: Role::User,
                content: Content::MultiPart(vec![
                    Block::text("Hello "),
                    Block::text("world"),
                ]),
            }],
            functions: None,
            ..Default::default()
        };
        let out = tmpl().render(&p, false).unwrap();
        assert!(out.contains("Hello world"));
    }

    /// End-to-end: render with the real Llama 3.1 chat template from the
    /// test model and feed the result into the engine.
    #[test]
    #[ignore = "requires model"]
    fn chat_template_from_real_model() {
        use crate::Engine;
        use std::path::PathBuf;

        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf");
        let engine = Engine::from_path(path).unwrap();
        let tmpl = ChatTemplate::from_model(&engine.model)
            .expect("model should have a chat template");
        let out = tmpl
            .render(&simple_prompt(), true)
            .expect("real template should render");
        // The Llama 3.1 template always emits the BOS piece, so we know
        // the BOS string will appear in the output.
        assert!(
            out.contains(tmpl.bos_token()),
            "rendered prompt should contain BOS: {out}"
        );
        // And it should end with the assistant generation header.
        assert!(
            out.contains("assistant"),
            "rendered prompt missing assistant header"
        );
    }

    /// Exercise the tools branch of the real Llama 3.1 template: pass a
    /// single function definition, render, and assert the rendered
    /// prompt includes the function name, schema, date, and ipython
    /// environment header.
    #[test]
    #[ignore = "requires model"]
    fn chat_template_renders_tools_against_real_model() {
        use crate::{Engine, Tool};
        use serde_json::json;
        use std::{borrow::Cow, path::PathBuf};

        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf");
        let engine = Engine::from_path(path).unwrap();
        let tmpl = ChatTemplate::from_model(&engine.model).unwrap();

        let tool = Tool {
            name: Cow::Borrowed("get_weather"),
            description: Cow::Borrowed(
                "Look up the current weather in a city.",
            ),
            schema: json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }),
            cache_control: None,
            strict: None,
        };

        let prompt = Prompt {
            system: Some(Content::SinglePart(Cow::Borrowed(
                "You are helpful.",
            ))),
            messages: vec![Message {
                role: Role::User,
                content: Content::SinglePart(Cow::Borrowed(
                    "What's the weather in Paris?",
                )),
            }],
            functions: Some(vec![tool]),
            ..Default::default()
        };

        let opts = RenderOptions::default()
            .with_generation_prompt(true)
            .with_date("17 Apr 2026");
        let out = tmpl.render_with(&prompt, &opts).unwrap();

        assert!(
            out.contains("get_weather"),
            "tools branch must mention function name. output:\n{out}"
        );
        assert!(
            out.contains("\"city\""),
            "schema should appear in rendered output. output:\n{out}"
        );
        assert!(
            out.contains("17 Apr 2026"),
            "date_string should appear when provided. output:\n{out}"
        );
        assert!(
            out.contains("Environment: ipython"),
            "system header should include ipython env when tools present"
        );
    }

    /// Render an assistant tool-call turn, confirming the template
    /// emits Llama 3.1's JSON tool-call format.
    #[test]
    #[ignore = "requires model"]
    fn chat_template_renders_assistant_tool_call_against_real_model() {
        use crate::prompt::ToolUse;
        use crate::Engine;
        use serde_json::json;
        use std::{borrow::Cow, path::PathBuf};

        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf");
        let engine = Engine::from_path(path).unwrap();
        let tmpl = ChatTemplate::from_model(&engine.model).unwrap();

        let call = ToolUse {
            id: Cow::Borrowed("call_1"),
            name: Cow::Borrowed("get_weather"),
            input: json!({"city": "Paris"}),
            cache_control: None,
        };

        let prompt = Prompt {
            system: None,
            messages: vec![
                Message {
                    role: Role::User,
                    content: Content::SinglePart(Cow::Borrowed(
                        "Call get_weather for Paris.",
                    )),
                },
                Message {
                    role: Role::Assistant,
                    content: Content::MultiPart(vec![Block::ToolUse { call }]),
                },
            ],
            functions: None,
            ..Default::default()
        };

        let out = tmpl.render(&prompt, false).unwrap();
        assert!(
            out.contains("\"name\": \"get_weather\""),
            "tool-call branch must include function name. output:\n{out}"
        );
        assert!(
            out.contains("\"parameters\""),
            "tool-call branch must include parameters. output:\n{out}"
        );
        assert!(
            out.contains("\"city\""),
            "arguments should appear in rendered output. output:\n{out}"
        );
    }

    /// Diagnostic: read a file from
    /// `DRAMA_LLAMA_TOKENIZE_FILE` and print (count, first 20 ids, last 20
    /// ids) of drama_llama's tokenization. Lets us compare against
    /// ollama's token count on the same bytes to confirm tokenizer
    /// parity. Run with
    /// `DRAMA_LLAMA_TOKENIZE_FILE=/tmp/dl_turn2.txt cargo test
    /// dump_tokenize -- --ignored --nocapture`.
    #[test]
    #[ignore = "diagnostic helper"]
    fn dump_tokenize() {
        use crate::Engine;
        use std::path::PathBuf;
        let Some(file) = std::env::var_os("DRAMA_LLAMA_TOKENIZE_FILE") else {
            panic!("set DRAMA_LLAMA_TOKENIZE_FILE to the input path");
        };
        let text = std::fs::read_to_string(&file)
            .unwrap_or_else(|e| panic!("read {file:?}: {e}"));
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf");
        let engine = Engine::from_path(path).unwrap();
        let tokens_nospec = engine.model.tokenize(&text, false);
        let tokens_spec = engine.model.tokenize(&text, true);
        println!("parse_special=false: {} tokens", tokens_nospec.len());
        println!("parse_special=true:  {} tokens", tokens_spec.len());
        let head_spec: Vec<_> = tokens_spec.iter().take(20).collect();
        println!("(spec) first 20: {head_spec:?}");
    }

    /// One-off dump: render the strawberry turn-1 Prompt through our
    /// ChatTemplate and write the bytes to
    /// `DRAMA_LLAMA_DUMP_OUTPUT`. For diffing against the Python
    /// jinja2 cross-check renderer.
    #[test]
    #[ignore = "fixture helper"]
    fn dump_strawberry_turn_1_output() {
        use crate::{Engine, Tool};
        use serde_json::json;
        use std::{borrow::Cow, path::PathBuf};
        let Some(dest) = std::env::var_os("DRAMA_LLAMA_DUMP_OUTPUT") else {
            panic!("set DRAMA_LLAMA_DUMP_OUTPUT to the output path");
        };
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf");
        let engine = Engine::from_path(path).unwrap();
        let tmpl = ChatTemplate::from_model(&engine.model).unwrap();

        let tool = Tool {
            name: Cow::Borrowed("count_letters"),
            description: Cow::Borrowed(
                "Count the number of times a letter appears in a string.",
            ),
            schema: json!({
                "type": "object",
                "properties": {
                    "letter": {"type": "string", "description": "the letter to count"},
                    "string": {"type": "string", "description": "the string to search"}
                },
                "required": ["letter", "string"]
            }),
            cache_control: None,
            strict: None,
        };
        let prompt = Prompt {
            system: Some(Content::SinglePart(Cow::Borrowed(
                "You are a helpful assistant. You cannot count letters in a \
                 word reliably on your own because you see in tokens, not \
                 letters. Use the `count_letters` tool when asked to count \
                 characters.",
            ))),
            messages: vec![Message {
                role: Role::User,
                content: Content::SinglePart(Cow::Borrowed(
                    "Count the number of r's in 'strawberry'",
                )),
            }],
            functions: Some(vec![tool]),
            ..Default::default()
        };
        let opts = RenderOptions::default()
            .with_generation_prompt(true)
            .with_date("17 Apr 2026")
            .with_extra("enable_thinking", true);
        let out = tmpl.render_with(&prompt, &opts).unwrap();
        std::fs::write(&dest, &out)
            .unwrap_or_else(|e| panic!("write {dest:?}: {e}"));
        println!("wrote {} bytes to {:?}", out.len(), dest);
    }

    /// One-off dump helper: write the GGUF's embedded
    /// `tokenizer.chat_template` out to the path in
    /// `DRAMA_LLAMA_DUMP_TEMPLATE` so we can commit it as a pinned
    /// fixture. Run with
    /// `DRAMA_LLAMA_DUMP_TEMPLATE=tests/fixtures/cogito_14b_template.jinja \
    /// cargo test dump_template_fixture -- --ignored --nocapture`.
    #[test]
    #[ignore = "fixture helper"]
    fn dump_template_fixture() {
        use crate::Engine;
        use std::path::PathBuf;
        let Some(dest) = std::env::var_os("DRAMA_LLAMA_DUMP_TEMPLATE") else {
            panic!("set DRAMA_LLAMA_DUMP_TEMPLATE to the output path");
        };
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf");
        let engine = Engine::from_path(path).unwrap();
        let source = engine
            .model
            .get_meta("tokenizer.chat_template")
            .expect("model has no tokenizer.chat_template");
        std::fs::write(&dest, &source)
            .unwrap_or_else(|e| panic!("write {dest:?}: {e}"));
        println!("wrote {} bytes to {:?}", source.len(), dest);
    }

    /// Pin the shape of the rendered `tools` variable: OpenAI wire
    /// envelope (`type: "function"`, nested `function` object) with
    /// `parameters` rather than Anthropic's `input_schema`.
    ///
    /// Regression lock for the tool-shape bug: cogito / Qwen / Hermes
    /// templates do `{{ tool | tojson }}` and the model was trained on
    /// ollama's runtime output. Emitting bare `{name, description,
    /// input_schema}` caused three cogito sizes to consistently swap
    /// tool-call arguments.
    #[test]
    fn tools_rendered_as_openai_wire_envelope() {
        use serde_json::json;
        use std::borrow::Cow;
        let tool = crate::Tool {
            name: Cow::Borrowed("count_letters"),
            description: Cow::Borrowed("Count letters in a string."),
            schema: json!({
                "type": "object",
                "properties": {
                    "letter": {"type": "string"},
                    "string": {"type": "string"}
                },
                "required": ["letter", "string"]
            }),
            cache_control: None,
            strict: None,
        };
        let src =
            r#"{%- for t in tools %}{{ t | tojson }}{% endfor %}"#.to_owned();
        let t = ChatTemplate::from_source(src, "".into(), "".into()).unwrap();
        let prompt = Prompt {
            functions: Some(vec![tool]),
            ..Default::default()
        };
        let out = t.render(&prompt, false).unwrap();

        // Must contain the wire envelope.
        assert!(
            out.contains("\"type\":\"function\""),
            "expected wire envelope `type: function`. got:\n{out}"
        );
        assert!(
            out.contains("\"function\":{"),
            "expected nested `function` object. got:\n{out}"
        );
        // Field name must be `parameters`, NOT Anthropic's `input_schema`.
        assert!(
            out.contains("\"parameters\":"),
            "expected `parameters` field. got:\n{out}"
        );
        assert!(
            !out.contains("\"input_schema\""),
            "Anthropic-shape `input_schema` must not leak. got:\n{out}"
        );
        // Tool name and schema content must survive.
        assert!(out.contains("\"name\":\"count_letters\""));
        assert!(out.contains("\"letter\""));
        assert!(out.contains("\"string\""));
    }

    // ----------------------------------------------------------------
    // Phase 2: cache_control breakpoint discovery + partial rendering
    // ----------------------------------------------------------------

    use misanthropic::prompt::message::CacheControl;
    use serde_json::json;
    use std::borrow::Cow;

    /// A tool with no cache marker.
    fn tool_plain(name: &'static str) -> Tool {
        Tool {
            name: Cow::Borrowed(name),
            description: Cow::Borrowed("tool"),
            schema: json!({"type": "object", "properties": {}}),
            cache_control: None,
            strict: None,
        }
    }

    /// A tool marked with an ephemeral cache breakpoint.
    fn tool_cached(name: &'static str) -> Tool {
        Tool {
            name: Cow::Borrowed(name),
            description: Cow::Borrowed("tool"),
            schema: json!({"type": "object", "properties": {}}),
            cache_control: Some(CacheControl::ephemeral()),
            strict: None,
        }
    }

    /// Make a `Role::User` message whose content is a single Text
    /// block with an ephemeral cache breakpoint attached.
    fn cached_user_msg(text: &'static str) -> Message {
        Message {
            role: Role::User,
            content: Content::MultiPart(vec![Block::Text {
                text: Cow::Borrowed(text),
                cache_control: Some(CacheControl::ephemeral()),
            }]),
        }
    }

    #[test]
    fn test_collect_breakpoints_empty() {
        let prompt = simple_prompt();
        assert_eq!(collect_breakpoints(&prompt), Vec::<Breakpoint>::new());
    }

    #[test]
    fn test_collect_breakpoints_tools_only() {
        let prompt = Prompt {
            functions: Some(vec![tool_cached("cached_tool")]),
            ..Prompt::default()
        };
        assert_eq!(collect_breakpoints(&prompt), vec![Breakpoint::AfterTools]);
    }

    #[test]
    fn test_collect_breakpoints_all_levels() {
        // Tool marker + system marker + cache on messages[0] and
        // messages[2]. messages[1] is uncached — verifies the emitted
        // message indices match the actually-cached ones.
        let system = Content::MultiPart(vec![Block::Text {
            text: Cow::Borrowed("You are helpful."),
            cache_control: Some(CacheControl::ephemeral()),
        }]);
        let prompt = Prompt {
            functions: Some(vec![tool_plain("a"), tool_cached("b")]),
            system: Some(system),
            messages: vec![
                cached_user_msg("first"),
                Message {
                    role: Role::Assistant,
                    content: Content::SinglePart(Cow::Borrowed("reply")),
                },
                cached_user_msg("third"),
            ],
            ..Prompt::default()
        };
        assert_eq!(
            collect_breakpoints(&prompt),
            vec![
                Breakpoint::AfterTools,
                Breakpoint::AfterSystem,
                Breakpoint::AfterMessage(0),
                Breakpoint::AfterMessage(2),
            ]
        );
    }

    #[test]
    fn test_render_partial_after_system() {
        // Truncated prompt at AfterSystem should render with an empty
        // messages list; the rendered bytes must match what
        // `render_with` produces on the same prompt with messages
        // cleared and `add_generation_prompt=false`.
        let prompt = Prompt {
            system: Some(Content::SinglePart(Cow::Borrowed("Sys."))),
            messages: vec![Message {
                role: Role::User,
                content: Content::SinglePart(Cow::Borrowed("q")),
            }],
            ..Prompt::default()
        };
        let opts = RenderOptions::default().with_generation_prompt(true);
        let partial =
            render_partial(&tmpl(), &prompt, &opts, Breakpoint::AfterSystem)
                .unwrap();

        let reference = Prompt {
            system: prompt.system.clone(),
            messages: vec![],
            ..Prompt::default()
        };
        let expected = tmpl()
            .render_with(
                &reference,
                &RenderOptions::default().with_generation_prompt(false),
            )
            .unwrap();
        assert_eq!(partial, expected);
    }

    #[test]
    fn test_render_with_breakpoints_no_cache_control() {
        let out = tmpl()
            .render_with_breakpoints(
                &simple_prompt(),
                &RenderOptions::default().with_generation_prompt(true),
            )
            .unwrap();
        assert!(out.partial_texts.is_empty());
        assert!(out.text.starts_with("<|begin_of_text|>"));
    }

    #[test]
    fn test_render_with_breakpoints_generation_prompt_forced_false() {
        // Prompt with a mid-conversation cache breakpoint and the
        // caller asking for add_generation_prompt=true. The full
        // render must honor it; each partial must not.
        let prompt = Prompt {
            system: Some(Content::SinglePart(Cow::Borrowed("sys"))),
            messages: vec![
                cached_user_msg("hi"),
                Message {
                    role: Role::Assistant,
                    content: Content::SinglePart(Cow::Borrowed("hello")),
                },
            ],
            ..Prompt::default()
        };
        let out = tmpl()
            .render_with_breakpoints(
                &prompt,
                &RenderOptions::default().with_generation_prompt(true),
            )
            .unwrap();
        const GEN: &str = "<|start_header_id|>assistant<|end_header_id|>\n\n";
        assert!(
            out.text.ends_with(GEN),
            "full render must end with generation prompt: {:?}",
            out.text
        );
        assert_eq!(
            out.partial_texts.len(),
            1,
            "one cache marker → one partial"
        );
        for (i, p) in out.partial_texts.iter().enumerate() {
            assert!(
                !p.ends_with(GEN),
                "partial {i} must not end with generation prompt: {p:?}"
            );
        }
    }

    /// Model-backed round-trip: render a prompt with a mid-
    /// conversation cache breakpoint, tokenize full + partial, and
    /// assert the partial's tokens are a proper prefix of the full's
    /// — i.e. the breakpoint index in the returned list equals the
    /// partial's token length, and the full's first `idx` tokens
    /// equal the partial's tokens.
    #[test]
    #[ignore = "requires model"]
    fn test_tokenize_with_breakpoints_prefix_property() {
        use crate::Engine;
        use std::path::PathBuf;

        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf");
        let engine = Engine::from_path(path).unwrap();
        let t = ChatTemplate::from_model(&engine.model).unwrap();

        let prompt = Prompt {
            system: Some(Content::SinglePart(Cow::Borrowed(
                "You are helpful.",
            ))),
            messages: vec![
                cached_user_msg("Who am I?"),
                Message {
                    role: Role::Assistant,
                    content: Content::SinglePart(Cow::Borrowed("A human.")),
                },
                Message {
                    role: Role::User,
                    content: Content::SinglePart(Cow::Borrowed("What next?")),
                },
            ],
            ..Prompt::default()
        };

        let opts = RenderOptions::default().with_generation_prompt(true);
        let rendered = t.render_with_breakpoints(&prompt, &opts).unwrap();
        assert_eq!(rendered.partial_texts.len(), 1);

        let (full_tokens, indices) =
            tokenize_with_breakpoints(&engine.model, &rendered);
        assert_eq!(indices.len(), 1, "exactly one breakpoint expected");
        let idx = indices[0];
        assert!(idx <= full_tokens.len());

        // The partial's tokens equal the full's tokens up to `idx`.
        let partial_tokens =
            engine.model.tokenize(&rendered.partial_texts[0], true);
        assert_eq!(partial_tokens.len(), idx);
        assert_eq!(&full_tokens[..idx], partial_tokens.as_slice());
    }
}
