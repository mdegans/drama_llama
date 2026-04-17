//! `tool_choice` → [`SamplingMode`] compiler.
//!
//! Takes a [`ToolChoice`] and the advertised [`Tool`] list, emits a GBNF
//! that forces the model's output to match the chosen shape. Rendering is
//! separate: the chat template (via `Prompt::tools`) advertises the tool
//! to the model; this module *enforces* that the model actually calls it.
//!
//! # Shapes currently supported
//!
//! The grammar targets the Llama 3.1 tool-call JSON shape, which is also
//! what HuggingFace's `transformers` tool-calling loop emits by default:
//!
//! ```json
//! {"name": "<function_name>", "parameters": {...}}
//! ```
//!
//! Some shapes (Anthropic, OpenAI) use `"arguments"` instead of
//! `"parameters"`. Configure that via
//! [`ToolChoiceOptions::arguments_field`].
//!
//! # Optional thought preamble
//!
//! Set [`ToolChoiceOptions::allow_thought`] to let the model emit a
//! `<think>…</think>` block *before* the JSON call. Useful for reasoning
//! models; a no-op for templates that don't render thought blocks on
//! re-ingest (Llama 3.1's tool-call branch ignores any surrounding
//! `content`). The grammar accepts a single thought block, then the
//! JSON call. Two balanced tags, no nesting.
//!
//! # Auto
//!
//! [`ToolChoice::Auto`] returns `None` — no sampling constraint is
//! needed because the model is free to call or not call. Pair with
//! `Prompt::tools` to advertise the tools.
//!
//! [`ToolChoice`]: misanthropic::tool::Choice
//! [`Tool`]: crate::Tool
//! [`SamplingMode`]: crate::SamplingMode

use std::fmt::Write;

use crate::{GrammarError, SamplingMode, Tool};

pub use misanthropic::tool::Choice as ToolChoice;

/// Options for [`grammar_for_tool_choice`].
#[derive(Clone, Debug)]
pub struct ToolChoiceOptions {
    /// Permit an optional `<think>…</think>` block before the JSON call.
    /// Generation-time only — survives serialization only if the caller
    /// persists the assistant content themselves.
    pub allow_thought: bool,
    /// Name of the tool-call arguments field. Llama 3.1 and HF use
    /// `"parameters"`. Anthropic / OpenAI function-calling use
    /// `"arguments"`.
    pub arguments_field: &'static str,
}

impl Default for ToolChoiceOptions {
    fn default() -> Self {
        Self {
            allow_thought: false,
            arguments_field: "parameters",
        }
    }
}

/// Build a [`SamplingMode::Grammar`] that forces the model's output to
/// match the chosen tool-call shape.
///
/// Returns `Ok(None)` for [`ToolChoice::Auto`] — no constraint is
/// appropriate there. For [`ToolChoice::Any`] the grammar accepts any
/// of the tools in `tools`; [`ToolChoice::Method`] pins a specific
/// tool. Returns an error if the chosen tool is not in `tools`, if
/// `tools` is empty (and choice is not `Auto`), or if the generated
/// GBNF fails to compile.
pub fn grammar_for_tool_choice(
    choice: &ToolChoice,
    tools: &[Tool<'_>],
    opts: &ToolChoiceOptions,
) -> Result<Option<SamplingMode>, ToolChoiceError> {
    let names: Vec<&str> = match choice {
        ToolChoice::Auto => return Ok(None),
        ToolChoice::Any => {
            if tools.is_empty() {
                return Err(ToolChoiceError::NoTools);
            }
            tools.iter().map(|t| t.name.as_ref()).collect()
        }
        ToolChoice::Method { name } => {
            if !tools.iter().any(|t| t.name.as_ref() == name.as_str()) {
                return Err(ToolChoiceError::UnknownTool(name.clone()));
            }
            vec![name.as_str()]
        }
    };

    let source = build_grammar_source(&names, opts);
    let mode = SamplingMode::grammar(&source)?;
    Ok(Some(mode))
}

/// Emit the GBNF source text for a tool-choice constraint.
///
/// Kept pub(crate) + exposed through tests so the generated grammar can
/// be inspected directly without constructing a full [`SamplingMode`].
pub(crate) fn build_grammar_source(
    names: &[&str],
    opts: &ToolChoiceOptions,
) -> String {
    let mut src = String::with_capacity(1024);

    // Root rule: optional thought block, then the JSON call.
    if opts.allow_thought {
        let _ = writeln!(src, r#"root ::= thought? ws call"#);
        // Thought: `<think>…</think>` with arbitrary content that does
        // not itself close the tag. We accept anything up to the
        // literal `</think>` — using a greedy-ish construction that
        // refuses `<` inside the body to avoid ambiguity with the
        // closing tag.
        let _ = writeln!(src, r#"thought ::= "<think>" think_body "</think>""#);
        let _ = writeln!(src, r#"think_body ::= think_char*"#);
        let _ = writeln!(src, r#"think_char ::= [^<]"#);
    } else {
        let _ = writeln!(src, r#"root ::= ws call"#);
    }

    // The tool call: `{"name": <chosen>, "<args>": <json_object>}`.
    // Whitespace is permissive between tokens but not inside the
    // keyword literals.
    let _ = writeln!(
        src,
        r#"call ::= "{{" ws "\"name\"" ws ":" ws name_choice ws "," ws "\"{arg}\"" ws ":" ws object ws "}}""#,
        arg = opts.arguments_field
    );

    // Alternation over the allowed function names, each quoted.
    let mut alt = String::new();
    for (i, n) in names.iter().enumerate() {
        if i > 0 {
            alt.push_str(" | ");
        }
        // Escape any `"` in the name, defensively.
        let escaped = n.replace('"', r#"\""#);
        write!(alt, r#""\"{escaped}\"""#).unwrap();
    }
    let _ = writeln!(src, r#"name_choice ::= {alt}"#);

    // Standard JSON grammar — RFC 8259-ish, enough for tool arguments.
    src.push_str(JSON_GRAMMAR);

    src
}

/// Shared JSON value grammar appended to every tool-choice GBNF.
///
/// Handles object / array / string / number / literal, with permissive
/// intra-structure whitespace. Not strict about number formatting edge
/// cases (e.g. `01` is rejected as JSON would); good enough for tool
/// arguments, which downstream parsers will validate.
const JSON_GRAMMAR: &str = r#"
value ::= object | array | string | number | "true" | "false" | "null"
object ::= "{" ws ( member ( ws "," ws member )* )? ws "}"
member ::= string ws ":" ws value
array ::= "[" ws ( value ( ws "," ws value )* )? ws "]"
string ::= "\"" char* "\""
char ::= unescaped | escape
unescaped ::= [^"\\] | [\x20-\x21] | [\x23-\x5B] | [\x5D-\x7F]
escape ::= "\\" ( ["\\/bfnrt] | "u" hex hex hex hex )
hex ::= [0-9a-fA-F]
number ::= int frac? exp?
int ::= "-"? ( "0" | [1-9] [0-9]* )
frac ::= "." [0-9]+
exp ::= [eE] [+\-]? [0-9]+
ws ::= [ \t\n\r]*
"#;

/// Errors from [`grammar_for_tool_choice`].
#[derive(Debug, thiserror::Error)]
pub enum ToolChoiceError {
    #[error("tool list is empty; cannot enforce tool_choice")]
    NoTools,
    #[error("tool_choice picked unknown tool `{0}`")]
    UnknownTool(String),
    #[error("compiled grammar is invalid: {0}")]
    Grammar(#[from] GrammarError),
}

static_assertions::assert_impl_all!(ToolChoiceError: Send, Sync);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Grammar, GrammarState};
    use serde_json::json;
    use std::{borrow::Cow, sync::Arc};

    fn tool(name: &'static str) -> Tool<'static> {
        Tool {
            name: Cow::Borrowed(name),
            description: Cow::Borrowed(""),
            schema: json!({"type": "object"}),
            cache_control: None,
        }
    }

    fn accepts(src: &str, input: &str) -> bool {
        let grammar = Grammar::parse(src).expect("grammar should parse");
        let mut state = GrammarState::new(Arc::new(grammar));
        state.advance_bytes(input.as_bytes()).is_ok() && state.is_complete()
    }

    #[test]
    fn auto_returns_none() {
        let got = grammar_for_tool_choice(
            &ToolChoice::Auto,
            &[],
            &ToolChoiceOptions::default(),
        )
        .unwrap();
        assert!(got.is_none());
    }

    #[test]
    fn any_rejects_empty_tool_list() {
        let err = grammar_for_tool_choice(
            &ToolChoice::Any,
            &[],
            &ToolChoiceOptions::default(),
        )
        .unwrap_err();
        assert!(matches!(err, ToolChoiceError::NoTools));
    }

    #[test]
    fn method_rejects_unknown_name() {
        let err = grammar_for_tool_choice(
            &ToolChoice::Method {
                name: "missing".into(),
            },
            &[tool("get_weather")],
            &ToolChoiceOptions::default(),
        )
        .unwrap_err();
        assert!(
            matches!(err, ToolChoiceError::UnknownTool(ref n) if n == "missing")
        );
    }

    #[test]
    fn method_grammar_accepts_forced_call() {
        let src = build_grammar_source(
            &["get_weather"],
            &ToolChoiceOptions::default(),
        );
        assert!(accepts(
            &src,
            r#"{"name": "get_weather", "parameters": {"city": "Paris"}}"#
        ));
        // Other tool names must be rejected.
        assert!(!accepts(
            &src,
            r#"{"name": "send_email", "parameters": {}}"#
        ));
    }

    #[test]
    fn any_grammar_accepts_any_listed_name() {
        let src =
            build_grammar_source(&["a", "b"], &ToolChoiceOptions::default());
        assert!(accepts(&src, r#"{"name": "a", "parameters": {}}"#));
        assert!(accepts(&src, r#"{"name": "b", "parameters": {}}"#));
        assert!(!accepts(&src, r#"{"name": "c", "parameters": {}}"#));
    }

    #[test]
    fn arguments_field_is_configurable() {
        let opts = ToolChoiceOptions {
            arguments_field: "arguments",
            ..ToolChoiceOptions::default()
        };
        let src = build_grammar_source(&["x"], &opts);
        assert!(accepts(&src, r#"{"name": "x", "arguments": {}}"#));
        // With `arguments`, `parameters` is rejected.
        assert!(!accepts(&src, r#"{"name": "x", "parameters": {}}"#));
    }

    #[test]
    fn thought_preamble_optional() {
        let opts = ToolChoiceOptions {
            allow_thought: true,
            ..ToolChoiceOptions::default()
        };
        let src = build_grammar_source(&["x"], &opts);
        // No thought.
        assert!(accepts(&src, r#"{"name": "x", "parameters": {}}"#));
        // With thought.
        assert!(accepts(
            &src,
            "<think>I should call x.</think>\n{\"name\": \"x\", \"parameters\": {}}"
        ));
    }

    #[test]
    fn deeply_nested_arguments_accepted() {
        let src = build_grammar_source(&["x"], &ToolChoiceOptions::default());
        assert!(accepts(
            &src,
            r#"{"name": "x", "parameters": {"a": {"b": [1, 2, {"c": "d"}]}}}"#
        ));
    }

    #[test]
    fn malformed_arguments_rejected() {
        let src = build_grammar_source(&["x"], &ToolChoiceOptions::default());
        // Trailing comma is invalid JSON.
        assert!(!accepts(&src, r#"{"name": "x", "parameters": {"a": 1,}}"#));
        // Single-quoted string is invalid JSON.
        assert!(!accepts(&src, r#"{"name": "x", "parameters": {'a': 1}}"#));
    }

    /// End-to-end: force Llama 3.1 to make a specific tool call. The
    /// grammar constraint should make the generated text parse as
    /// exactly the expected tool call shape.
    #[test]
    #[ignore = "requires model"]
    fn tool_choice_forces_call_against_real_model() {
        use crate::{
            ChatTemplate, Content, Engine, Message, PredictOptions, Prompt,
            RenderOptions, Role, SampleOptions, SamplingMode,
        };
        use std::{num::NonZeroUsize, path::PathBuf};

        let model_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf");
        let mut engine = Engine::from_path(model_path).unwrap();
        let tmpl = ChatTemplate::from_model(&engine.model).unwrap();

        let weather = tool("get_weather");
        let prompt = Prompt {
            system: Some(Content::SinglePart(Cow::Borrowed(
                "You are a helpful assistant.",
            ))),
            messages: vec![Message {
                role: Role::User,
                content: Content::SinglePart(Cow::Borrowed(
                    "What's the weather in Paris? Call the tool.",
                )),
            }],
            tools: Some(vec![weather.clone()]),
        };
        let rendered = tmpl
            .render_with(
                &prompt,
                &RenderOptions::default().with_generation_prompt(true),
            )
            .unwrap();

        // Build a tool-choice grammar that forces the get_weather call.
        let choice = ToolChoice::Method {
            name: "get_weather".into(),
        };
        let tools = [weather];
        let forced = grammar_for_tool_choice(
            &choice,
            &tools,
            &ToolChoiceOptions::default(),
        )
        .unwrap()
        .expect("Method choice should yield a grammar");

        let tokens = engine.model.tokenize(&rendered, false);
        let mut opts = PredictOptions::default().add_model_stops(&engine.model);
        opts.n = NonZeroUsize::new(256).unwrap();
        opts.sample_options = SampleOptions {
            modes: vec![forced, SamplingMode::locally_typical()],
            ..SampleOptions::default()
        };

        let eos_piece = engine.model.token_to_piece(engine.model.eos());
        let predictor = engine.predict_pieces(tokens, opts);
        let output: String = predictor.collect();

        println!(
            "=== forced tool call ===\n{output}\n========================"
        );
        let _ = eos_piece; // Some tokens render as [Invalid UTF-8] so
                           // trim_end_matches isn't reliable; slice to
                           // the grammar-forced closing brace instead.

        // Slice from the first `{` to the matching `}`. The grammar
        // guarantees a single balanced object.
        let start =
            output.find('{').expect("output must contain opening brace");
        let mut depth = 0;
        let mut end = 0;
        for (i, b) in output[start..].bytes().enumerate() {
            match b {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = start + i + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        assert!(end > start, "unbalanced braces in output: {output:?}");
        let trimmed = &output[start..end];

        let value: serde_json::Value =
            serde_json::from_str(trimmed).unwrap_or_else(|e| {
                panic!(
                    "forced output must parse as JSON: {e}\noutput: {output:?}\ntrimmed: {trimmed:?}"
                )
            });
        assert_eq!(
            value.get("name").and_then(|v| v.as_str()),
            Some("get_weather"),
            "forced output must have name == get_weather. output: {output:?}"
        );
        assert!(
            value.get("parameters").is_some(),
            "forced output must have a `parameters` field. output: {output:?}"
        );
    }
}
