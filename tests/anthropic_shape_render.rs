//! Repro for the `system: [{...}]` 500 surfaced via blallama.
//!
//! Two equivalent prompts — one with `system` as plain text
//! ([`Content::SinglePart`]), one as a single Text block with
//! `cache_control` ([`Content::MultiPart`]) — should render to the
//! same Jinja-input shape and produce the same output bytes.
//!
//! `#[ignore]` because it reads the Qwen3 chat template from the
//! moeflux artifact directory rather than committing a copy.

use std::{borrow::Cow, path::PathBuf};

use drama_llama::{
    ChatTemplate, Content, Message, Prompt, RenderOptions, Role,
};
use misanthropic::prompt::message::{Block, CacheControl};

fn template_path() -> PathBuf {
    PathBuf::from(
        std::env::var("DRAMA_LLAMA_QWEN3_TEMPLATE").unwrap_or_else(|_| {
            "/Volumes/Temp Backup/models/blallama/qwen3-6-a3b/mlx/chat_template.jinja"
                .to_string()
        }),
    )
}

fn load_template() -> ChatTemplate {
    let source = std::fs::read_to_string(template_path())
        .expect("Qwen3 chat_template.jinja missing");
    ChatTemplate::from_source(
        source,
        String::new(),
        "<|im_end|>".to_string(),
    )
    .expect("template compiles")
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
        system: Some(Content::SinglePart(Cow::Borrowed(
            "You are a concise assistant.",
        ))),
        messages: vec![Message {
            role: Role::User,
            content: Content::SinglePart(Cow::Borrowed("Hello")),
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
        system: Some(Content::MultiPart(vec![Block::Text {
            text: Cow::Borrowed("You are a concise assistant."),
            cache_control: Some(CacheControl::ephemeral()),
        }])),
        messages: vec![Message {
            role: Role::User,
            content: Content::MultiPart(vec![Block::Text {
                text: Cow::Borrowed("Hello"),
                cache_control: Some(CacheControl::ephemeral()),
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

    let prompt: Prompt =
        serde_json::from_value(body).expect("deserialize ok");
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
        system: Some(Content::MultiPart(vec![Block::Text {
            text: Cow::Borrowed("You are a concise assistant."),
            cache_control: Some(CacheControl::ephemeral()),
        }])),
        messages: vec![Message {
            role: Role::User,
            content: Content::MultiPart(vec![Block::Text {
                text: Cow::Borrowed("Hello"),
                cache_control: Some(CacheControl::ephemeral()),
            }]),
        }],
        ..Prompt::default()
    };
    let rendered = t
        .render_with_breakpoints(&prompt, &opts())
        .expect("render_with_breakpoints must succeed even when AfterSystem partial fails");
    eprintln!("=== full ===\n{}\n=== end ===", rendered.text);
    eprintln!(
        "=== {} partials ===",
        rendered.partial_texts.len()
    );
    for (i, p) in rendered.partial_texts.iter().enumerate() {
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
        system: Some(Content::SinglePart(Cow::Borrowed(
            "You are a concise assistant.",
        ))),
        messages: vec![Message {
            role: Role::User,
            content: Content::SinglePart(Cow::Borrowed("Hello")),
        }],
        ..Prompt::default()
    };
    let multi = Prompt {
        system: Some(Content::MultiPart(vec![Block::Text {
            text: Cow::Borrowed("You are a concise assistant."),
            cache_control: Some(CacheControl::ephemeral()),
        }])),
        messages: vec![Message {
            role: Role::User,
            content: Content::MultiPart(vec![Block::Text {
                text: Cow::Borrowed("Hello"),
                cache_control: Some(CacheControl::ephemeral()),
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
