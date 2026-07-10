//! Pin the dialect analyzer's output per vendored template fixture —
//! mirror of llama.cpp's `tests/test-chat-auto-parser.cpp`
//! expectations (b9754). Whitespace in markers is significant; the
//! newlines here are the trained format, not formatting accidents.

use drama_llama::dialect::{
    analyze_template, CallSyntax, Family, ReasoningMode,
};

fn analyze(fixture: &str, bos: &str, eos: &str) -> CallSyntax {
    let path = format!(
        "{}/tests/fixtures/templates/{fixture}",
        env!("CARGO_MANIFEST_DIR")
    );
    let source = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {path}: {e}"));
    analyze_template(&source, bos, eos)
        .unwrap_or_else(|e| panic!("analyze {fixture}: {e}"))
}

/// Upstream pins these exact markers for Qwen3-Coder
/// (`test-chat-auto-parser.cpp:1358-1376`). This is the load-bearing
/// case: Qwen3.6 tool dialect.
#[test]
fn qwen3_coder_xml() {
    let s = analyze("Qwen3-Coder.jinja", "", "<|im_end|>");
    assert_eq!(s.family, Family::TagWithTagged, "{s:#?}");
    assert_eq!(s.per_call_start, "<tool_call>\n", "{s:#?}");
    assert_eq!(s.per_call_end, "</tool_call>", "{s:#?}");
    assert_eq!(s.function.name_prefix, "<function=", "{s:#?}");
    assert_eq!(s.function.name_suffix, ">\n", "{s:#?}");
    assert_eq!(s.function.close, "</function>\n", "{s:#?}");
    assert_eq!(s.arguments.name_prefix, "<parameter=", "{s:#?}");
    assert_eq!(s.arguments.name_suffix, ">\n", "{s:#?}");
    assert_eq!(s.arguments.value_suffix, "\n</parameter>\n", "{s:#?}");
    assert_eq!(s.trigger(), "<tool_call>\n", "{s:#?}");
}

/// The template we actually serve: Qwen3.6 (Unsloth GGUF dump).
/// Same XML dialect as Qwen3-Coder.
#[test]
fn qwen36_gguf_xml() {
    let s = analyze("qwen3.6-gguf.jinja", "", "<|im_end|>");
    assert_eq!(s.family, Family::TagWithTagged, "{s:#?}");
    assert_eq!(s.per_call_start, "<tool_call>\n", "{s:#?}");
    assert_eq!(s.per_call_end, "</tool_call>", "{s:#?}");
    assert_eq!(s.function.name_prefix, "<function=", "{s:#?}");
    assert_eq!(s.function.name_suffix, ">\n", "{s:#?}");
    assert_eq!(s.function.close, "</function>\n", "{s:#?}");
    assert_eq!(s.arguments.name_prefix, "<parameter=", "{s:#?}");
    assert_eq!(s.arguments.name_suffix, ">\n", "{s:#?}");
    assert_eq!(s.arguments.value_suffix, "\n</parameter>\n", "{s:#?}");
}

/// Qwen3 chat (0.6B template): Hermes-style JSON inside <tool_call>
/// section markers — JSON_NATIVE in upstream's classification
/// (TAG_WITH_JSON is the Functionary `<function=X>{json}` shape).
/// Thinking via <think>.
#[test]
fn qwen3_chat_hermes_json() {
    let s = analyze("Qwen-Qwen3-0.6B.jinja", "", "<|im_end|>");
    assert_eq!(s.family, Family::JsonNative, "{s:#?}");
    assert!(
        s.trigger().starts_with("<tool_call>"),
        "trigger: {:?}\n{s:#?}",
        s.trigger()
    );
    assert_eq!(s.reasoning.mode, ReasoningMode::TagBased, "{s:#?}");
    assert!(s.reasoning.end.contains("</think>"), "{s:#?}");
}

/// Qwen3.5 moved to the XML dialect (same as Coder / 3.6) — this is
/// exactly why per-model dialects exist.
#[test]
fn qwen35_xml() {
    let s = analyze("Qwen3.5-4B.jinja", "", "<|im_end|>");
    assert_eq!(s.family, Family::TagWithTagged, "{s:#?}");
    assert_eq!(s.function.name_prefix, "<function=", "{s:#?}");
    assert_eq!(s.arguments.name_prefix, "<parameter=", "{s:#?}");
    assert!(
        s.trigger().starts_with("<tool_call>"),
        "trigger: {:?}\n{s:#?}",
        s.trigger()
    );
}

/// Hermes 3: the original <tool_call>{json}</tool_call> shape —
/// JSON_NATIVE with section markers.
#[test]
fn hermes3_json_in_section() {
    let s = analyze(
        "NousResearch-Hermes-3-Llama-3.1-8B-tool_use.jinja",
        "",
        "<|im_end|>",
    );
    assert_eq!(s.family, Family::JsonNative, "{s:#?}");
    assert_eq!(s.json.name_field, "name", "{s:#?}");
    assert_eq!(s.json.args_field, "arguments", "{s:#?}");
    assert!(
        s.trigger().starts_with("<tool_call>"),
        "trigger: {:?}\n{s:#?}",
        s.trigger()
    );
}

/// Llama 3.1: bare JSON with `parameters` as the args field.
#[test]
fn llama31_json_native() {
    let s = analyze(
        "meta-llama-Llama-3.1-8B-Instruct.jinja",
        "<|begin_of_text|>",
        "<|eot_id|>",
    );
    assert_eq!(s.family, Family::JsonNative, "{s:#?}");
    assert_eq!(s.json.name_field, "name", "{s:#?}");
    assert_eq!(s.json.args_field, "parameters", "{s:#?}");
}

/// A template with no tool support at all classifies as None
/// (Instructed-dialect territory). Minimal inline source.
#[test]
fn toolless_template_family_none() {
    let source = r#"{%- for m in messages -%}
<|{{ m.role }}|>{{ m.content }}<|end|>
{% endfor -%}
{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}"#;
    let s = analyze_template(source, "", "<|end|>").expect("analyze");
    assert_eq!(s.family, Family::None, "{s:#?}");
}
