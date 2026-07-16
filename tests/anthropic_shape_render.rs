//! Repro for the `system: [{...}]` 500 surfaced via blallama.
//!
//! Two equivalent prompts — one with `system` as plain text
//! ([`Content::SinglePart`]), one as a single Text block with
//! `cache_control` ([`Content::MultiPart`]) — should render to the
//! same Jinja-input shape and produce the same output bytes.
//!
#![cfg(feature = "llama-cpp")]
//! `#[ignore]` because it needs a Qwen-family chat template: either
//! `DRAMA_LLAMA_QWEN3_TEMPLATE` (a `chat_template.jinja` on disk, e.g.
//! from the moeflux artifact directory on the macOS box), or — the
//! fallback — the template embedded in `models/model.gguf`.

use std::{borrow::Cow, path::PathBuf};

use drama_llama::{
    ChatTemplate, Content, Message, Prompt, RenderOptions, Role,
};
use misanthropic::prompt::message::{Block, CacheControl};

fn load_template() -> ChatTemplate {
    if let Ok(path) = std::env::var("DRAMA_LLAMA_QWEN3_TEMPLATE") {
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("template {path} unreadable: {e}"));
        return ChatTemplate::from_source(
            source,
            String::new(),
            "<|im_end|>".to_string(),
        )
        .expect("template compiles");
    }
    let model_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf");
    let model = drama_llama::LlamaCppModel::from_file(model_path, None)
        .expect("set DRAMA_LLAMA_QWEN3_TEMPLATE or provide models/model.gguf");
    ChatTemplate::from_model(&model).expect("template compiles")
}

fn opts() -> RenderOptions {
    RenderOptions {
        add_generation_prompt: true,
        ..RenderOptions::default()
    }
}

#[test]
#[ignore = "needs qwen3 chat template fixture"]
fn singlepart_system_renders() {
    let t = load_template();
    let prompt = Prompt {
        system: Some(Content::text("You are a concise assistant.")),
        messages: vec![Message {
            role: Role::User,
            content: Content::text("Hello"),
        }],
        ..Prompt::default()
    };
    let rendered = t.render_with(&prompt, &opts()).expect("render ok");
    eprintln!("=== SinglePart system render ===\n{rendered}\n=== end ===");
    assert!(rendered.contains("system"));
    assert!(rendered.contains("Hello"));
}

#[test]
#[ignore = "needs qwen3 chat template fixture"]
fn multipart_system_with_cache_control_renders() {
    let t = load_template();
    let prompt = Prompt {
        system: Some(Content(vec![Block::Text {
            text: Cow::Borrowed("You are a concise assistant."),
            cache_control: Some(CacheControl::ephemeral()),
            citations: None,
        }])),
        messages: vec![Message {
            role: Role::User,
            content: Content(vec![Block::Text {
                text: Cow::Borrowed("Hello"),
                cache_control: Some(CacheControl::ephemeral()),
                citations: None,
            }]),
        }],
        ..Prompt::default()
    };
    let result = t.render_with(&prompt, &opts());
    match result {
        Ok(rendered) => {
            eprintln!(
                "=== MultiPart system render ===\n{rendered}\n=== end ==="
            );
            assert!(rendered.contains("system"));
            assert!(rendered.contains("Hello"));
        }
        Err(e) => {
            panic!("MultiPart system render failed: {e}");
        }
    }
}

/// Reproduces the exact JSON shape blallama received in the 500
/// repro: `system: [{"type":"text", ..., "cache_control":...}]` plus
/// MultiPart user content. Deserializes via misanthropic::Prompt to
/// confirm it lands in the expected `Content::MultiPart` shape, then
/// renders.
#[test]
#[ignore = "needs qwen3 chat template fixture"]
fn deserialize_and_render_anthropic_shape() {
    let body = serde_json::json!({
        "model": "qwen3-6-a3b",
        "max_tokens": 200,
        "system": [
            {
                "type": "text",
                "text": "You are a concise assistant. Answer in one short paragraph.",
                "cache_control": {"type": "ephemeral"}
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Tell me one interesting historical fact about 1969.",
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            }
        ]
    });

    let prompt: Prompt = serde_json::from_value(body).expect("deserialize ok");
    eprintln!(
        "=== deserialized prompt ===\nsystem: {:#?}\nmessages: {:#?}\n=== end ===",
        prompt.system, prompt.messages,
    );

    let t = load_template();
    let rendered = t
        .render_with(&prompt, &opts())
        .expect("Anthropic-shape render must succeed");
    eprintln!("=== rendered ===\n{rendered}\n=== end ===");
    assert!(rendered.contains("Tell me one interesting historical fact"));
}

/// **Regression**: `system` with `cache_control` triggers an
/// `AfterSystem` breakpoint, whose partial render has empty
/// `messages`. Templates like Qwen3 raise when no user message is
/// present. The breakpoint partial-render path must fail open
/// (drop the breakpoint) rather than propagating the template
/// error to the caller — caching is a perf hint, not a correctness
/// guarantee.
#[test]
#[ignore = "needs qwen3 chat template fixture"]
fn render_with_breakpoints_survives_after_system_breakpoint() {
    let t = load_template();
    let prompt = Prompt {
        system: Some(Content(vec![Block::Text {
            text: Cow::Borrowed("You are a concise assistant."),
            cache_control: Some(CacheControl::ephemeral()),
            citations: None,
        }])),
        messages: vec![Message {
            role: Role::User,
            content: Content(vec![Block::Text {
                text: Cow::Borrowed("Hello"),
                cache_control: Some(CacheControl::ephemeral()),
                citations: None,
            }]),
        }],
        ..Prompt::default()
    };
    let rendered = t
        .render_with_breakpoints(&prompt, &opts())
        .expect("render_with_breakpoints must succeed even when AfterSystem partial fails");
    eprintln!("=== full ===\n{}\n=== end ===", rendered.text);
    eprintln!("=== {} partials ===", rendered.partials.len());
    for (i, p) in rendered.partials.iter().map(|(_, p)| p).enumerate() {
        eprintln!("--- partial {i} ---\n{p}");
    }
    assert!(rendered.text.contains("Hello"));
}

/// Output equivalence: SinglePart and MultiPart with the same text
/// content must render to byte-identical output.
#[test]
#[ignore = "needs qwen3 chat template fixture"]
fn singlepart_and_multipart_render_equivalent() {
    let t = load_template();
    let single = Prompt {
        system: Some(Content::text("You are a concise assistant.")),
        messages: vec![Message {
            role: Role::User,
            content: Content::text("Hello"),
        }],
        ..Prompt::default()
    };
    let multi = Prompt {
        system: Some(Content(vec![Block::Text {
            text: Cow::Borrowed("You are a concise assistant."),
            cache_control: Some(CacheControl::ephemeral()),
            citations: None,
        }])),
        messages: vec![Message {
            role: Role::User,
            content: Content(vec![Block::Text {
                text: Cow::Borrowed("Hello"),
                cache_control: Some(CacheControl::ephemeral()),
                citations: None,
            }]),
        }],
        ..Prompt::default()
    };
    let single_out = t.render_with(&single, &opts()).expect("single render");
    let multi_out = t.render_with(&multi, &opts()).expect("multi render");
    assert_eq!(
        single_out, multi_out,
        "SinglePart and MultiPart shapes must render identically",
    );
}
