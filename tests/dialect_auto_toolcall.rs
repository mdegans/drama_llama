//! Regression repro for the council judge derail (2026-07-17, third
//! live run): with tool_choice **auto** (no forced grammar), the model
//! emitted its native XML dialect (`<tool_call>\n<function=...>`) but
//! no `ToolUse` block was parsed — the raw frame marker was seated as
//! assistant text, which poisons the transcript for the next turn
//! (ingest guard rejects the reserved special).
//!
//! The tool mirrors the council's `Bench::open_case`: namespaced name
//! with double underscores, a single OPTIONAL parameter — the shape
//! that derailed live.
//!
//! `cargo test --test dialect_auto_toolcall -- --ignored --nocapture`

#![cfg(feature = "llama-cpp")]

use std::{num::NonZeroUsize, path::PathBuf};

use drama_llama::{Block, LlamaCppSession, Prompt, Role, SamplingMode};

fn model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf")
}

fn open_case_tool() -> misanthropic::tool::CustomMethodDef {
    misanthropic::tool::CustomMethodDef::builder("toolbox__bench__open_case")
        .description(
            "Open a case: reset the docket and put the petitioner's \
             latest message — verbatim, attached by the docket itself \
             — to every advisor as round 1 (sealed, independent).",
        )
        .schema(serde_json::json!({
            "type": "object",
            "properties": {
                "note": {
                    "type": ["string", "null"],
                    "description": "Optional context from the bench, \
                        published alongside (never instead of) the \
                        petitioner's verbatim question. Leave it out \
                        unless the advisors need something the \
                        question doesn't say."
                }
            },
            "required": []
        }))
        .build_unchecked()
}

/// Control: the same tool with a REQUIRED plain-string parameter. If
/// this parses while the nullable-parameter variant dies, the bug is
/// the `Option<T>` (`"type": ["string","null"]`) schema shape in the
/// dialect grammar's parameter-value rule.
#[test]
#[ignore = "long running, requires models/model.gguf"]
fn auto_tool_call_required_string_control() {
    let tool = misanthropic::tool::CustomMethodDef::builder(
        "toolbox__bench__open_case",
    )
    .description("Open a case: put the question to every advisor as round 1.")
    .schema(serde_json::json!({
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question for the council."
            }
        },
        "required": ["question"]
    }))
    .build_unchecked();
    run_auto_toolcall(tool);
}

/// Auto tool choice on the model's native dialect: the completion must
/// either parse a `ToolUse` block or contain NO dialect frame markers
/// in its text — a frame marker seated as text is never acceptable
/// (it poisons the next turn's ingest).
#[test]
#[ignore = "long running, requires models/model.gguf"]
fn auto_tool_call_parses_or_text_is_clean() {
    run_auto_toolcall(open_case_tool());
}

fn run_auto_toolcall(tool: misanthropic::tool::CustomMethodDef) {
    let mut session = LlamaCppSession::from_path_sync(model_path())
        .expect("session load")
        .quiet()
        .without_repetition()
        .with_sampling([SamplingMode::Greedy])
        .with_max_tokens(NonZeroUsize::new(256).unwrap());

    let prompt = Prompt {
        system: Some(misanthropic::prompt::message::Content::text(
            "You are `judge`, the bench of a small council. You do not \
             answer questions yourself — open a case with the bench's \
             `open_case`. The docket attaches the petitioner's latest \
             message verbatim on its own; you may add a `note` for \
             context but you cannot rewrite the question.",
        )),
        messages: vec![drama_llama::Message {
            role: Role::User,
            content: misanthropic::prompt::message::Content::text(
                "The car wash is 100m away. Should I walk or drive?",
            ),
        }],
        tools: Some(vec![tool.into()]),
        ..Prompt::default()
    };

    let response = session.complete_response(&prompt).expect("completion");
    let blocks = &response.inner.content.0;
    eprintln!("=== blocks ===");
    for b in blocks {
        match b {
            Block::Text { text, .. } => eprintln!("TEXT: {text:?}"),
            Block::ToolUse { call } => {
                eprintln!("TOOL_USE: {} {:?}", call.name, call.input)
            }
            other => eprintln!("OTHER: {other:?}"),
        }
    }
    let has_tool_use =
        blocks.iter().any(|b| matches!(b, Block::ToolUse { .. }));
    let text_has_frame = blocks.iter().any(|b| match b {
        Block::Text { text, .. } => text.contains("<tool_call>"),
        _ => false,
    });
    assert!(
        !text_has_frame,
        "dialect frame marker seated as text — transcript poison \
         (the council-judge derail)",
    );
    assert!(
        has_tool_use,
        "judge-shaped prompt with auto tool choice should call the tool",
    );
}
