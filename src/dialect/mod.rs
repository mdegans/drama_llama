//! Per-model tool-call dialects: `CallSyntax` and the template
//! analyzer that derives it.
//!
//! The tool-call *format* a model was trained on is implied by its
//! chat template. [`CallSyntax`] captures that format as data — one
//! value drives both the GBNF emitter (Phase D) and the generic
//! envelope parser (Phase D), so enforce/parse/re-ingest can't drift
//! apart. It is **derived from** the template by differential
//! probing ([`analyzer`]): render sentinel payloads, diff the
//! outputs, extract the markers. Baked constants cover the known
//! families; a `<model>.dialect.toml` sidecar overrides analysis
//! when a finetune's template misdetects.
//!
//! Design and field vocabulary follow llama.cpp's auto-parser
//! (`common/chat-auto-parser.h` @ b9754, MIT) — see
//! `.claude/memory/plan_tool_dialects.md` for the mapping and the
//! decisions (no PEG-arena port; grammar + parser are compiled from
//! this struct instead).

mod analyzer;
mod segment;

pub use analyzer::{analyze_template, AnalyzeError};

/// Tool-call format family, per llama.cpp's classification.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(rename_all = "snake_case")
)]
pub enum Family {
    /// Template renders no tool calls at all. The `Instructed`
    /// dialect (Phase F) owns this case.
    #[default]
    None,
    /// Pure JSON: `{"name": "X", "arguments": {...}}`, possibly
    /// inside section markers.
    JsonNative,
    /// Tag-carried function with JSON arguments:
    /// `<function=X>{...}</function>`.
    TagWithJson,
    /// Tag-carried function with individually tagged arguments:
    /// `<parameter=key>value</parameter>` (Qwen3-Coder/3.6 XML).
    TagWithTagged,
}

/// How the model's reasoning block is delimited.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(rename_all = "snake_case")
)]
pub enum ReasoningMode {
    /// No reasoning markers detected.
    #[default]
    None,
    /// Tag-based (`<think>…</think>`); `start` may be empty for
    /// delimiter-style templates that only close.
    TagBased,
    /// Reasoning renders only on tool-call turns.
    ToolsOnly,
}

/// How assistant content is wrapped.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(rename_all = "snake_case")
)]
pub enum ContentMode {
    /// Plain text, no markers.
    #[default]
    Plain,
    /// Always wrapped (`<response>…</response>`, Cohere-style).
    AlwaysWrapped,
    /// Wrapped only when reasoning precedes it (Granite-style).
    WrappedWithReasoning,
}

/// Where a call ID renders relative to the function name and args
/// (tagged formats only; JSON formats carry it as a field).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(rename_all = "snake_case")
)]
pub enum CallIdPosition {
    #[default]
    None,
    PreFuncName,
    BetweenFuncAndArgs,
    PostArgs,
}

#[cfg(feature = "serde")]
fn is_default<T: Default + PartialEq>(v: &T) -> bool {
    *v == T::default()
}

/// Reasoning-block markers.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct ReasoningSyntax {
    pub mode: ReasoningMode,
    /// e.g. `"<think>"`, `""` (delimiter-style templates only close).
    pub start: String,
    /// e.g. `"</think>"`.
    pub end: String,
}

/// Content-block markers.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct ContentSyntax {
    pub mode: ContentMode,
    pub start: String,
    pub end: String,
}

/// Function-name markers within a call.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct FunctionSyntax {
    /// e.g. `"<function="`, `"\"name\": \""`.
    pub name_prefix: String,
    /// e.g. `">\n"`, `"\""`.
    pub name_suffix: String,
    /// e.g. `"</function>\n"`, `""`.
    pub close: String,
}

/// Argument markers within a call.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct ArgumentsSyntax {
    /// Wrapper around the whole argument list, if any
    /// (e.g. `"<|tool_call_argument_begin|>"`).
    pub start: String,
    pub end: String,
    /// e.g. `"<parameter="`.
    pub name_prefix: String,
    /// e.g. `">\n"`.
    pub name_suffix: String,
    /// Marker before a value (rare; usually empty).
    pub value_prefix: String,
    /// e.g. `"\n</parameter>\n"`.
    pub value_suffix: String,
    /// Separator between arguments, if any.
    pub separator: String,
}

/// Call-ID markers (tagged formats).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct CallIdSyntax {
    pub position: CallIdPosition,
    pub prefix: String,
    pub suffix: String,
}

/// JSON field names for [`Family::JsonNative`] /
/// [`Family::TagWithJson`] calls.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct JsonFields {
    /// Wrapper object key when calls nest one level
    /// (OpenAI's `{"function": {...}}`); empty when flat.
    pub function_field: String,
    pub name_field: String,
    pub args_field: String,
    /// Field the template renders the *client-supplied* call ID into;
    /// empty when unsupported.
    pub id_field: String,
    /// Heuristic id-ish field distinct from `id_field`.
    pub gen_id_field: String,
    /// `{"<fun_name>": {...args...}}` — the function name IS the key.
    pub fun_name_is_key: bool,
    /// Parallel calls render inside one JSON array.
    pub tools_array_wrapped: bool,
    /// Field order the template renders (drives emitter ordering so
    /// re-render matches emission byte-for-byte).
    pub parameter_order: Vec<String>,
}

impl Default for JsonFields {
    fn default() -> Self {
        Self {
            function_field: String::new(),
            name_field: "name".into(),
            args_field: "arguments".into(),
            id_field: String::new(),
            gen_id_field: String::new(),
            fun_name_is_key: false,
            tools_array_wrapped: false,
            parameter_order: Vec::new(),
        }
    }
}

/// One model's tool-call dialect: every marker needed to *emit*
/// (GBNF) and *parse* (envelope parser) calls in the shape the chat
/// template re-renders — the single source of truth for the
/// round-trip byte-stability invariant.
///
/// Whitespace in markers is significant (`"<tool_call>\n"` — the
/// newline is part of the trained format).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct CallSyntax {
    pub family: Family,
    /// Wrapper around the whole tool-call section (all calls),
    /// e.g. `"<tool_call>\n"` for single-call-style templates.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "is_default"))]
    pub section_start: String,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "is_default"))]
    pub section_end: String,
    /// Per-call wrapper for templates with parallel-call support.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "is_default"))]
    pub per_call_start: String,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "is_default"))]
    pub per_call_end: String,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "is_default"))]
    pub function: FunctionSyntax,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "is_default"))]
    pub arguments: ArgumentsSyntax,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "is_default"))]
    pub call_id: CallIdSyntax,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "is_default"))]
    pub json: JsonFields,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "is_default"))]
    pub reasoning: ReasoningSyntax,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "is_default"))]
    pub content: ContentSyntax,
    /// Start-of-message markers, for parser anchoring / debugging.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "is_default"))]
    pub user_start: String,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "is_default"))]
    pub assistant_start: String,
    /// Union of all non-empty markers (whitespace-trimmed) — tokens
    /// the tokenizer must keep intact / the sampler may see whole.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "is_default"))]
    pub preserved_tokens: Vec<String>,
}

impl CallSyntax {
    /// The byte sequence whose appearance in generated text activates
    /// the lazy tool-call grammar: the outermost call-opening marker.
    pub fn trigger(&self) -> &str {
        if !self.section_start.is_empty() {
            &self.section_start
        } else {
            &self.per_call_start
        }
    }

    /// Hermes / Qwen-chat style: `<tool_call>\n{json}\n</tool_call>`
    /// — JSON-native calls inside section markers. (What today's
    /// `ToolChoiceOptions::default()` hardcodes.)
    pub fn hermes_json() -> Self {
        Self {
            family: Family::JsonNative,
            section_start: "<tool_call>\n".into(),
            section_end: "\n</tool_call>".into(),
            json: JsonFields::default(),
            ..Self::default()
        }
    }

    /// Llama-3.1 style bare JSON with `parameters` as the args field.
    pub fn llama31_json() -> Self {
        Self {
            family: Family::JsonNative,
            json: JsonFields {
                args_field: "parameters".into(),
                ..JsonFields::default()
            },
            ..Self::default()
        }
    }

    /// Qwen3-Coder / Qwen3.6 XML-ish tagged format — the expected
    /// analyzer output for those templates (llama.cpp pins the same
    /// markers in `tests/test-chat-auto-parser.cpp`). Newlines
    /// significant.
    pub fn qwen_xml() -> Self {
        Self {
            family: Family::TagWithTagged,
            per_call_start: "<tool_call>\n".into(),
            per_call_end: "</tool_call>".into(),
            function: FunctionSyntax {
                name_prefix: "<function=".into(),
                name_suffix: ">\n".into(),
                close: "</function>\n".into(),
            },
            arguments: ArgumentsSyntax {
                name_prefix: "<parameter=".into(),
                name_suffix: ">\n".into(),
                value_suffix: "\n</parameter>\n".into(),
                ..ArgumentsSyntax::default()
            },
            ..Self::default()
        }
    }
}
