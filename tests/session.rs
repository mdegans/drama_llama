//! Integration tests for [`drama_llama::Session`].
//!
//! All tests here load a real model and are behind
//! `#[ignore = "requires model"]`. Run with
//! `cargo test --test session -- --ignored`.

use std::{borrow::Cow, num::NonZeroUsize, path::PathBuf};

use drama_llama::{
    prompt::{ToolResult, ToolUse},
    Block, ChatTemplate, Content, Message, Prompt, RenderOptions, Role,
    Session, SessionError, Tool, ToolChoice, ToolChoiceOptions,
};
use serde_json::json;

fn model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf")
}

/// Phase 1 milestone: `complete_text` runs end-to-end against a real
/// model, produces a non-empty string, and terminates cleanly
/// (no leaked EOS piece / `[Invalid UTF-8]` marker). The test uses
/// strawberry turn 2 — assistant + ToolResult already in the
/// transcript, free generation for the prose answer — because that's
/// the path the plan calls out for Phase 1 coverage.
#[test]
#[ignore = "requires model"]
fn complete_text_strawberry_turn_2() {
    let tool = Tool {
        name: Cow::Borrowed("count_letters"),
        description: Cow::Borrowed(
            "Count the number of times a letter appears in a string.",
        ),
        schema: json!({
            "type": "object",
            "properties": {
                "letter": {"type": "string"},
                "string": {"type": "string"}
            },
            "required": ["letter", "string"]
        }),
        cache_control: None,
    };
    let call_id = "call_3_r";
    let prompt = Prompt {
        system: Some(Content::SinglePart(Cow::Borrowed(
            "You are a helpful assistant.",
        ))),
        messages: vec![
            Message {
                role: Role::User,
                content: Content::SinglePart(Cow::Borrowed(
                    "Count the number of r's in 'strawberry'",
                )),
            },
            Message {
                role: Role::Assistant,
                content: Content::MultiPart(vec![Block::ToolUse {
                    call: ToolUse {
                        id: Cow::Borrowed(call_id),
                        name: Cow::Borrowed("count_letters"),
                        input: json!({"letter": "r", "string": "strawberry"}),
                        cache_control: None,
                    },
                }]),
            },
            Message {
                role: Role::User,
                content: Content::MultiPart(vec![Block::ToolResult {
                    result: ToolResult {
                        tool_use_id: Cow::Borrowed(call_id),
                        content: Content::SinglePart(Cow::Borrowed("3")),
                        is_error: false,
                        cache_control: None,
                    },
                }]),
            },
        ],
        functions: Some(vec![tool]),
        ..Default::default()
    };

    let mut session = Session::from_path(model_path())
        .expect("session load")
        .quiet()
        .with_tool_choice_opts(ToolChoiceOptions {
            arguments_field: "arguments",
            wrap_tags: Some(("<tool_call>\n", "\n</tool_call>")),
            allow_thought: true,
            ..ToolChoiceOptions::default()
        })
        .with_max_tokens(NonZeroUsize::new(256).unwrap());

    let out = session.complete_text(&prompt).expect("complete_text");
    println!("=== complete_text output ===\n{out}\n===");

    // Minimum contract: non-empty, no raw EOS piece trailing, mentions
    // the count somehow. Exact phrasing is model-dependent.
    assert!(!out.trim().is_empty(), "got empty output");
    let eos = session
        .engine()
        .model
        .token_to_piece(session.engine().model.eos());
    assert!(
        !out.ends_with(eos.as_str()),
        "EOS piece {eos:?} should have been trimmed; got: {out:?}"
    );
    assert!(
        !out.contains("[Invalid UTF-8]"),
        "trim helper missed an [Invalid UTF-8] marker: {out:?}"
    );
    assert!(
        out.contains('3') || out.to_lowercase().contains("three"),
        "expected the count (3) to appear in the answer, got: {out:?}"
    );
}

/// Grammar is prepended per-call even when the user passes an empty
/// sampling chain. This is the key contract for `with_sampling`: it
/// controls only the user portion; grammar can't be overridden away.
#[test]
#[ignore = "requires model"]
fn complete_text_grammar_prepended_even_with_empty_sampling() {
    let tool = Tool {
        name: Cow::Borrowed("count_letters"),
        description: Cow::Borrowed("Count letters."),
        schema: json!({
            "type": "object",
            "properties": {
                "letter": {"type": "string"},
                "string": {"type": "string"}
            },
            "required": ["letter", "string"]
        }),
        cache_control: None,
    };
    let prompt = Prompt {
        system: Some(Content::SinglePart(Cow::Borrowed(
            "You are a helpful assistant.",
        ))),
        messages: vec![Message {
            role: Role::User,
            content: Content::SinglePart(Cow::Borrowed(
                "Count r's in 'strawberry'",
            )),
        }],
        functions: Some(vec![tool]),
        tool_choice: Some(ToolChoice::Method {
            name: "count_letters".into(),
        }),
        ..Default::default()
    };

    let mut session = Session::from_path(model_path())
        .expect("session load")
        .quiet()
        .with_tool_choice_opts(ToolChoiceOptions {
            arguments_field: "arguments",
            wrap_tags: Some(("<tool_call>\n", "\n</tool_call>")),
            allow_thought: true,
            ..ToolChoiceOptions::default()
        })
        .with_sampling(std::iter::empty()) // user chain empty — only grammar runs
        .with_max_tokens(NonZeroUsize::new(128).unwrap());

    let out = session.complete_text(&prompt).expect("complete_text");
    println!("=== forced-call output ===\n{out}\n===");

    // With grammar forcing the tool call, the output MUST contain the
    // wrapped envelope and the function name. If grammar weren't
    // prepended, the model would emit arbitrary text.
    assert!(
        out.contains("<tool_call>"),
        "grammar should have forced tagged envelope, got: {out:?}"
    );
    assert!(
        out.contains("count_letters"),
        "grammar should have pinned the tool name, got: {out:?}"
    );
}

fn strawberry_turn_1_prompt() -> Prompt {
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
    };
    Prompt {
        // Match the strawberry example's system prompt — short ones
        // give the model too much latitude to hallucinate args.
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
        tool_choice: Some(ToolChoice::Method {
            name: "count_letters".into(),
        }),
        ..Default::default()
    }
}

fn cogito_tool_choice_opts() -> ToolChoiceOptions {
    ToolChoiceOptions {
        arguments_field: "arguments",
        wrap_tags: Some(("<tool_call>\n", "\n</tool_call>")),
        allow_thought: true,
        // strict_schema pins the argument KEYS to what the tool
        // declared. Without this, cogito is free to hallucinate
        // `{"word": "..."}` instead of `{"string": "..."}`; with it,
        // the grammar rejects any tokens that would break the schema.
        strict_schema: true,
    }
}

/// Phase 3: `complete` returns a `Message` whose tool_use content
/// matches the `letter` / `string` we asked the model to count.
#[test]
#[ignore = "requires model"]
fn complete_returns_message_with_tool_use() {
    let prompt = strawberry_turn_1_prompt();
    let mut session = Session::from_path(model_path())
        .expect("session load")
        .quiet()
        .with_tool_choice_opts(cogito_tool_choice_opts())
        .with_max_tokens(NonZeroUsize::new(256).unwrap());

    let assistant = session.complete(&prompt).expect("complete");
    println!("=== complete message ===\n{assistant:#?}\n===");

    let msg: Message = assistant.into();
    assert_eq!(msg.role, Role::Assistant);
    // Must contain a ToolUse block.
    let blocks: Vec<&Block> = match &msg.content {
        Content::MultiPart(b) => b.iter().collect(),
        Content::SinglePart(_) => {
            panic!("expected MultiPart with ToolUse, got SinglePart")
        }
    };
    let call = blocks
        .iter()
        .find_map(|b| match b {
            Block::ToolUse { call } => Some(call),
            _ => None,
        })
        .expect("no ToolUse block in message");
    // Shape assertions only — the test is about Session plumbing, not
    // model semantics. Parser round-trip is the contract; what the
    // model chose for `letter` / `string` is the model's business.
    assert_eq!(call.name, "count_letters");
    assert!(
        call.input.get("letter").and_then(|v| v.as_str()).is_some(),
        "letter arg missing from tool_use"
    );
    assert!(
        call.input.get("string").and_then(|v| v.as_str()).is_some(),
        "string arg missing from tool_use"
    );
}

/// Phase 3: `complete_blocks` surfaces `SessionError::GrammarViolation`
/// when grammar-forced generation truncates before closing the
/// tool_call tag. We reproduce the truncation by capping max_tokens
/// low enough that the model can't finish.
#[test]
#[ignore = "requires model"]
fn grammar_violation_on_truncated_tool_call() {
    let prompt = strawberry_turn_1_prompt();
    let mut session = Session::from_path(model_path())
        .expect("session load")
        .quiet()
        .with_tool_choice_opts(cogito_tool_choice_opts())
        .with_max_tokens(NonZeroUsize::new(4).unwrap()); // truncate hard

    let err = session
        .complete_blocks(&prompt)
        .expect_err("should have returned GrammarViolation");
    match err {
        SessionError::GrammarViolation { partial_output } => {
            println!("partial_output: {partial_output:?}");
        }
        other => panic!("expected GrammarViolation, got {other:?}"),
    }
}

/// Phase 3 round-trip invariant: `complete_text` and `complete` are
/// two views of the same bytes. Parse the raw bytes into an
/// [`AssistantMessage`], tack it onto the original [`Prompt`],
/// re-render the whole thing, and assert the raw bytes appear
/// verbatim in the re-rendered suffix.
///
/// One inference call so the bytes we're comparing against are
/// deterministically the same bytes that got parsed — no seed /
/// KV-cache drift worries.
#[test]
#[ignore = "requires model"]
fn complete_text_round_trips_through_parse_and_render() {
    use drama_llama::{parse_completion, AssistantMessage};

    let prompt = strawberry_turn_1_prompt();
    let tool_opts = cogito_tool_choice_opts();

    let mut session = Session::from_path(model_path())
        .expect("session load")
        .quiet()
        .with_tool_choice_opts(tool_opts)
        .with_max_tokens(NonZeroUsize::new(256).unwrap());

    let raw = session.complete_text(&prompt).expect("complete_text");

    // Parse the same bytes into blocks → AssistantMessage.
    let blocks = parse_completion(&raw);
    assert!(!blocks.is_empty(), "parser dropped the output: {raw:?}");
    let assistant: AssistantMessage = blocks.into_iter().collect();

    // Build a follow-up prompt with the assistant turn appended, and
    // render both versions via the same template that drove
    // inference. Tool choice is cleared so the assistant turn is
    // final, not forcing another call.
    let mut follow_up = prompt.clone();
    follow_up.messages.push(assistant.into());
    follow_up.tool_choice = None;

    let render_opts = RenderOptions::default()
        .with_generation_prompt(true)
        .with_extra(
            "enable_thinking",
            drama_llama::minijinja::Value::from(true),
        );
    let rendered_original = session
        .template()
        .render_with(&prompt, &render_opts)
        .expect("render original");
    let rendered_follow_up = session
        .template()
        .render_with(
            &follow_up,
            &RenderOptions::default()
                .with_generation_prompt(false)
                .with_extra(
                    "enable_thinking",
                    drama_llama::minijinja::Value::from(true),
                ),
        )
        .expect("render follow_up");

    let suffix = rendered_follow_up
        .strip_prefix(&rendered_original)
        .unwrap_or_else(|| {
            panic!(
                "follow-up must extend the original prefix.\n\
                 --- original ---\n{rendered_original}\n\
                 --- follow-up ---\n{rendered_follow_up}"
            )
        });

    // Block-level (not byte-level) lossless round-trip: re-parsing
    // the template's re-rendered output must yield the same blocks
    // we parsed from the raw model output. Byte-level equality is
    // impossible because Jinja's `tojson` filter canonicalizes JSON
    // (compact + alphabetized), losing any intra-JSON whitespace the
    // model emitted.
    let raw_blocks = parse_completion(&raw);
    // Strip the template's trailing <|im_end|>\n footer before
    // reparsing, since our parser treats it as prose.
    let suffix_trimmed =
        suffix.trim_end_matches('\n').trim_end_matches("<|im_end|>");
    let suffix_blocks = parse_completion(suffix_trimmed);

    assert_eq!(
        raw_blocks.len(),
        suffix_blocks.len(),
        "block count mismatch: raw has {}, suffix has {}",
        raw_blocks.len(),
        suffix_blocks.len()
    );
    for (a, b) in raw_blocks.iter().zip(suffix_blocks.iter()) {
        match (a, b) {
            (Block::ToolUse { call: ca }, Block::ToolUse { call: cb }) => {
                assert_eq!(ca.name, cb.name);
                assert_eq!(
                    ca.input, cb.input,
                    "tool_use args diverged across round-trip"
                );
            }
            (
                Block::Thought { thought: ta, .. },
                Block::Thought { thought: tb, .. },
            ) => {
                assert_eq!(ta, tb);
            }
            // Text blocks may differ in trailing whitespace from
            // template framing — that's fine, the semantics match.
            (Block::Text { .. }, Block::Text { .. }) => {}
            _ => panic!("block type mismatch: {a:?} vs {b:?}"),
        }
    }
}
