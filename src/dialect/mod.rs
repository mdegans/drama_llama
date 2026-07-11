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
mod emit;
mod parse;
mod segment;

pub use analyzer::{analyze_template, vocab_cross_check, AnalyzeError};
pub use emit::{
    grammar_source, render_reference, validate_representable, Anchor,
    DialectError, EmitOptions,
};
pub use parse::{parse_text, Leniency, ParseStatus, Parsed, StreamParser};

/// Tool-call format family, per llama.cpp's classification.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(rename_all = "snake_case")
)]
pub enum Family {
    /// Template renders no tool calls at all. The `Instructed`
    /// dialect (deferred follow-up to Phase F) will own this case;
    /// today Session falls back to Hermes-JSON enforcement.
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
    /// Tag-carried function whose arguments render as one brace dict
    /// with *bare* keys and a dedicated string-quote marker instead
    /// of JSON quoting:
    /// `<|tool_call>call:name{city:<|"|>London<|"|>}<tool_call|>`
    /// (Gemma 4). The quote marker is a special token, so string
    /// values need no in-band escaping — but a value containing the
    /// marker's literal bytes is unrepresentable (typed error, like
    /// [`Family::TagWithTagged`] close delimiters).
    TagWithDict,
    /// OpenAI Harmony (gpt-oss): every message is a channel block
    /// (`<|channel|>analysis|commentary|final<|message|>…`), tool
    /// calls address a recipient (`to=functions.NAME`, legal in the
    /// role header or the channel header) with JSON arguments and end
    /// at the EOG token `<|call|>`. Too structural for the generic
    /// marker fields — the emitter and parser have hand-built paths
    /// keyed on this variant, like upstream's dedicated handler
    /// (`common_chat_params_init_gpt_oss`).
    Harmony,
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

/// How the assistant's prior thoughts feed back through the chat
/// template on re-ingest.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(rename_all = "snake_case")
)]
pub enum ReasoningReingest {
    /// Thought text is wrapped `<think>…</think>` inline in the
    /// message `content` (legacy Qwen convention — those templates
    /// reconstruct reasoning by splitting content on the tag).
    #[default]
    InlineThink,
    /// Thought text is passed as the message's `reasoning` /
    /// `reasoning_content` fields; the template owns the markers
    /// (Gemma 4, DeepSeek-style templates). Inlining `<think>` for
    /// these templates would pollute content instead.
    Field,
    /// [`Field`](Self::Field), plus the gpt-oss stock-template
    /// accommodations: tool-call messages additionally carry the
    /// `thinking` key (the field that template documents) and
    /// withhold the merged `content` (the template raises when both
    /// are set with `tool_calls`; upstream erases content the same
    /// way). `thinking` is NOT set on plain assistant messages — the
    /// stock template renders any past message carrying that key as
    /// an analysis channel block, swallowing its final content — so
    /// cache-stable sidecars read `reasoning_content` /
    /// `content_pre` / `content_post` instead.
    Thinking,
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
    /// Re-ingest convention (drives `RenderOptions` in `Session`).
    pub reingest: ReasoningReingest,
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
    /// String-quote marker for [`Family::TagWithDict`] values
    /// (e.g. `"<|\"|>"`); empty for other families.
    pub string_quote: String,
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
    /// Marker that opens tool responses *inside the same turn*
    /// (Gemma 4's `<|tool_response>`). Non-empty means the template
    /// keeps a call turn open awaiting results, and the model's
    /// trained continuation after its last call is this marker — so
    /// the tool grammar *requires* it as the turn exit (after which
    /// only EOG is legal: a deterministic stop that is also the
    /// canonical re-render byte), and the parser swallows it as
    /// envelope rather than surfacing it as text. Empty for dialects
    /// whose models end call turns with EOG directly (Qwen, Hermes,
    /// Llama).
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "is_default"))]
    pub tool_response_start: String,
    /// Union of all non-empty markers (whitespace-trimmed) — tokens
    /// the tokenizer must keep intact / the sampler may see whole.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "is_default"))]
    pub preserved_tokens: Vec<String>,
}

/// Literal markers of the Harmony ([`Family::Harmony`]) wire format,
/// shared by the hand-built emitter and parser so they cannot drift.
/// Verbatim from the gpt-oss chat template / upstream
/// `common_chat_params_init_gpt_oss`.
pub mod harmony {
    /// Opens every message after the first in a generation (the
    /// generation prompt itself ends with it, so the first block
    /// starts directly at its channel header).
    pub const START_ASSISTANT: &str = "<|start|>assistant";
    pub const CHANNEL: &str = "<|channel|>";
    pub const MESSAGE: &str = "<|message|>";
    /// Ends analysis / commentary blocks in-stream. NOT
    /// end-of-generation — libllama removes it from `special_eog_ids`
    /// for Harmony vocabs (see `LlamaCppModel::extra_eos_tokens`).
    pub const END: &str = "<|end|>";
    /// EOG: ends a tool call.
    pub const CALL: &str = "<|call|>";
    /// EOG: ends a final message. Re-ingest renders `<|end|>` in its
    /// place (upstream issue #15417); the divergence costs one LCP
    /// token per final turn.
    pub const RETURN: &str = "<|return|>";
    pub const CONSTRAIN: &str = "<|constrain|>";
    /// Recipient prefix for client tools; other recipients
    /// (`container.exec`, `python`, `browser.search`) are builtin
    /// tools we swallow, upstream parity.
    pub const TO_FUNCTIONS: &str = " to=functions.";
    pub const ANALYSIS_OPEN: &str = "<|channel|>analysis<|message|>";
    pub const COMMENTARY: &str = "<|channel|>commentary";
    pub const COMMENTARY_OPEN: &str = "<|channel|>commentary<|message|>";
    pub const FINAL_OPEN: &str = "<|channel|>final<|message|>";
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

    /// All lazy-activation byte sequences. Most dialects have exactly
    /// one ([`Self::trigger`]); Harmony's tool-call header has no
    /// single distinctive marker, so it triggers on any of the
    /// recipient-bearing header shapes. Deliberately conservative — a
    /// false activation derails generation (the grammar starts
    /// forcing call bytes mid-thought), while a miss only loses
    /// enforcement for that call: the parser still recognizes it and
    /// the canonicalization gate covers the bytes. Upstream uses
    /// anchored regexes for the same reason (`chat.cpp` gpt-oss
    /// `grammar_triggers`).
    pub fn triggers(&self) -> Vec<String> {
        match self.family {
            Family::Harmony => vec![
                // Recipient in the role header.
                format!(
                    "{}{}",
                    harmony::START_ASSISTANT,
                    harmony::TO_FUNCTIONS
                ),
                // Recipient in the channel header.
                format!(
                    "{}commentary{}",
                    harmony::CHANNEL,
                    harmony::TO_FUNCTIONS
                ),
                format!(
                    "{}analysis{}",
                    harmony::CHANNEL,
                    harmony::TO_FUNCTIONS
                ),
            ],
            _ => vec![self.trigger().to_string()],
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
            reasoning: ReasoningSyntax {
                mode: ReasoningMode::TagBased,
                start: "<think>".into(),
                end: "</think>".into(),
                reingest: ReasoningReingest::InlineThink,
            },
            ..Self::default()
        }
    }

    /// Gemma 4 native dialect — hand-built (llama.cpp hand-builds it
    /// too: `common_chat_params_init_gemma4`), selected by source
    /// sniff in the analyzer patches, not by differential probing.
    ///
    /// Calls: `<|tool_call>call:name{key:value,…}<tool_call|>` where
    /// the dict has bare keys and strings quoted by the special token
    /// `<|"|>`. Reasoning is channel-based
    /// (`<|channel>thought\n…<channel|>`) and renders only on
    /// tool-call turns. All markers are single vocab tokens.
    pub fn gemma4() -> Self {
        Self {
            family: Family::TagWithDict,
            per_call_start: "<|tool_call>".into(),
            per_call_end: "<tool_call|>".into(),
            function: FunctionSyntax {
                name_prefix: "call:".into(),
                name_suffix: String::new(),
                close: String::new(),
            },
            arguments: ArgumentsSyntax {
                start: "{".into(),
                end: "}".into(),
                name_suffix: ":".into(),
                separator: ",".into(),
                string_quote: "<|\"|>".into(),
                ..ArgumentsSyntax::default()
            },
            reasoning: ReasoningSyntax {
                mode: ReasoningMode::ToolsOnly,
                start: "<|channel>thought\n".into(),
                // Leading newline is canonical: the template re-renders
                // reasoning as `<|channel>thought\n{text}\n<channel|>`,
                // so the grammar must force the trailing newline too.
                // Parsing trims, so the model may omit it (upstream's
                // zero-budget case) at the cost of a one-turn
                // canonicalization repair.
                end: "\n<channel|>".into(),
                reingest: ReasoningReingest::Field,
            },
            user_start: "<|turn>user\n".into(),
            assistant_start: "<|turn>model\n".into(),
            tool_response_start: "<|tool_response>".into(),
            preserved_tokens: vec![
                "<|channel>".into(),
                "<channel|>".into(),
                "<|tool_call>".into(),
                "<tool_call|>".into(),
                "<|tool_response>".into(),
                "<tool_response|>".into(),
                "<|turn>".into(),
                "<turn|>".into(),
                "<|\"|>".into(),
            ],
            ..Self::default()
        }
    }

    /// gpt-oss / OpenAI Harmony dialect — hand-built (llama.cpp
    /// hand-builds it too: `common_chat_params_init_gpt_oss`),
    /// selected by source sniff (`<|channel|>` in the template).
    ///
    /// Every message is a channel block; a generation is
    /// `analysis*/commentary* blocks separated by <|start|>assistant`,
    /// ending in either a final message
    /// (`<|channel|>final<|message|>… <|return|>`) or a tool call.
    /// Canonical call shape (what the grammar forces and the
    /// cache-stable sidecar re-renders — the model's *trained*
    /// channel-header form):
    /// `<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{args}<|call|>`.
    /// The parser additionally accepts the role-header form
    /// (`<|start|>assistant to=functions.NAME<|channel|>…`), bare
    /// constraint types without `<|constrain|>`, and the stock
    /// template's re-render (` json<|message|>`), upstream parity.
    ///
    /// The call-marker fields stay empty: the shapes above don't
    /// decompose into the generic prefix/suffix vocabulary, so the
    /// emitter and parser use hand-built paths over the [`harmony`]
    /// literals instead. Empty `per_call_start` also keeps Session's
    /// parallel-call gate off — Harmony is structurally single-call
    /// (`<|call|>` is EOG).
    pub fn gpt_oss() -> Self {
        Self {
            family: Family::Harmony,
            reasoning: ReasoningSyntax {
                mode: ReasoningMode::TagBased,
                start: harmony::ANALYSIS_OPEN.into(),
                end: harmony::END.into(),
                reingest: ReasoningReingest::Thinking,
            },
            content: ContentSyntax {
                mode: ContentMode::AlwaysWrapped,
                start: harmony::FINAL_OPEN.into(),
                // No close marker: final content runs to EOG.
                end: String::new(),
            },
            user_start: "<|start|>user<|message|>".into(),
            assistant_start: harmony::START_ASSISTANT.into(),
            preserved_tokens: vec![
                "<|start|>".into(),
                "<|channel|>".into(),
                "<|message|>".into(),
                "<|end|>".into(),
                "<|constrain|>".into(),
                "<|call|>".into(),
                "<|return|>".into(),
            ],
            ..Self::default()
        }
    }
}
