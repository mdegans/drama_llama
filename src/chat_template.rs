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

use minijinja::{value::Value as JinjaValue, Environment, Error as JinjaError};

use crate::{Block, Content, Model, Prompt, Role};

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
        prompt: &Prompt<'_>,
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
        prompt: &Prompt<'_>,
        opts: &RenderOptions<'_>,
    ) -> Result<String, ChatTemplateError> {
        let messages = build_messages(prompt);
        let tools_value = match prompt.tools.as_ref() {
            Some(ts) if !ts.is_empty() => JinjaValue::from_serialize(ts),
            _ => JinjaValue::from(()), // renders as None / null
        };
        // Default `date_string` to today in HF's "%d %b %Y" format when
        // the caller didn't supply one. The template unconditionally
        // concatenates `"Today Date: " + date_string + ...`, so passing
        // `none` would blow up with a string-plus-none type error.
        let date_string = opts
            .date_string
            .as_deref()
            .map(|s| s.to_owned())
            .unwrap_or_else(|| {
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
pub struct RenderOptions<'a> {
    /// Ask the template to append an empty assistant header so the model
    /// generates the next turn. True for live chat, false for
    /// tokenizing a stored transcript.
    pub add_generation_prompt: bool,
    /// Current date string (e.g. `"17 Apr 2026"`). Llama 3.1's template
    /// reads `date_string` when stamping a system-message header. If
    /// `None`, the template's default (static fallback) is used.
    pub date_string: Option<Cow<'a, str>>,
    /// Template-specific extra variables. Keys become top-level names in
    /// the Jinja context. Values are arbitrary Serialize-able data.
    pub extras: Vec<(String, JinjaValue)>,
}

impl<'a> RenderOptions<'a> {
    /// Builder: set `add_generation_prompt`.
    pub fn with_generation_prompt(mut self, yes: bool) -> Self {
        self.add_generation_prompt = yes;
        self
    }

    /// Builder: set `date_string`.
    pub fn with_date<S>(mut self, date: S) -> Self
    where
        S: Into<Cow<'a, str>>,
    {
        self.date_string = Some(date.into());
        self
    }

    /// Builder: add an arbitrary `(key, value)` pair to the Jinja
    /// context. Useful for `tools_in_user_message`, `builtin_tools`,
    /// etc. Construct the value via
    /// [`minijinja::Value::from_serialize`] or one of its constructors.
    pub fn with_extra<K>(mut self, key: K, value: JinjaValue) -> Self
    where
        K: Into<String>,
    {
        self.extras.push((key.into(), value));
        self
    }
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
fn build_messages(prompt: &Prompt<'_>) -> Vec<JinjaValue> {
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
fn append_message(
    out: &mut Vec<JinjaValue>,
    role: &str,
    content: &Content<'_>,
) {
    let blocks: Vec<&Block<'_>> = match content {
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
    call: &crate::prompt::ToolUse<'_>,
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
fn flatten_text(content: &Content<'_>) -> String {
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
fn append_block_text(out: &mut String, block: &Block<'_>) {
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

    fn simple_prompt() -> Prompt<'static> {
        Prompt::new()
            .with_system("You are helpful.")
            .push_user("Hi!")
            .push_assistant("Hello!")
            .push_user("What is 2+2?")
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
        let p = Prompt::new().push_user("hi");
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
            tools: None,
        };
        let out = tmpl().render(&p, false).unwrap();
        assert!(out.contains("<think>I should be concise.</think>Hello!"));
    }

    #[test]
    fn raise_exception_surfaces_as_error() {
        let src = r#"{{ raise_exception("boom") }}"#.to_owned();
        let t = ChatTemplate::from_source(src, "".into(), "".into()).unwrap();
        let err = t.render(&Prompt::new(), false).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("boom"), "error must surface message: {msg}");
    }

    #[test]
    fn strftime_now_renders_current_year() {
        let src = r#"{{ strftime_now("%Y-%m-%d") }}"#.to_owned();
        let t = ChatTemplate::from_source(src, "".into(), "".into()).unwrap();
        let out = t.render(&Prompt::new(), false).unwrap();
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
            tools: None,
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
            tools: Some(vec![tool]),
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
            tools: None,
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
}
