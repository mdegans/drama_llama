//! Integration tests for [`drama_llama::Session`].
//!
//! All tests here load a real model and are behind
//! `#[ignore = "requires model"]`. Run with
//! `cargo test --test session -- --ignored`.

use std::{borrow::Cow, num::NonZeroUsize, path::PathBuf};

use drama_llama::{
    prompt::{ToolResult, ToolUse},
    Block, Content, Message, Prompt, RenderOptions, Role, SessionError, Tool,
    ToolChoice,
};
use serde_json::json;

fn model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf")
}

/// Parse raw completion bytes through the session's own dialect —
/// what `Session::complete*` do internally. `pre_opened` mirrors the
/// generation prompt ending inside an open reasoning block.
fn parse_with_dialect(
    session: &drama_llama::LlamaCppSession,
    prompt: &Prompt,
    raw: &str,
    pre_opened: bool,
) -> Vec<Block> {
    let tool_refs: Vec<&Tool> = prompt
        .tools
        .iter()
        .flatten()
        .filter_map(|def| def.as_method())
        .collect();
    drama_llama::dialect::parse_text(
        session.dialect(),
        &tool_refs,
        raw,
        pre_opened,
        drama_llama::dialect::Leniency::Final,
    )
    .blocks
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
    let tool = Tool::builder("count_letters")
        .description("Count the number of times a letter appears in a string.")
        .schema(json!({
            "type": "object",
            "properties": {
                "letter": {"type": "string"},
                "string": {"type": "string"}
            },
            "required": ["letter", "string"]
        }))
        .build()
        .expect("valid test tool");
    let call_id = "call_3_r";
    let prompt = Prompt {
        system: Some(Content::text("You are a helpful assistant.")),
        messages: vec![
            Message {
                role: Role::User,
                content: Content::text(
                    "Count the number of r's in 'strawberry'",
                ),
            },
            Message {
                role: Role::Assistant,
                content: Content(vec![Block::ToolUse {
                    call: ToolUse {
                        id: Cow::Borrowed(call_id),
                        name: Cow::Borrowed("count_letters"),
                        input: json!({"letter": "r", "string": "strawberry"}),
                        cache_control: None,
                        caller: None,
                    },
                }]),
            },
            Message {
                role: Role::User,
                content: Content(vec![Block::ToolResult {
                    result: ToolResult {
                        tool_use_id: Cow::Borrowed(call_id),
                        content: Content::text("3"),
                        is_error: false,
                        cache_control: None,
                    },
                }]),
            },
        ],
        tools: Some(vec![tool.into()]),
        ..Default::default()
    };

    // No dialect override: the session derives the tool-call format
    // from the model's chat template at load (#30 Phase E).
    let mut session =
        drama_llama::LlamaCppSession::from_path_sync(model_path())
            .expect("session load")
            .quiet()
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
    let tool = Tool::builder("count_letters")
        .description("Count letters.")
        .schema(json!({
            "type": "object",
            "properties": {
                "letter": {"type": "string"},
                "string": {"type": "string"}
            },
            "required": ["letter", "string"]
        }))
        .build()
        .expect("valid test tool");
    let prompt = Prompt {
        system: Some(Content::text("You are a helpful assistant.")),
        messages: vec![Message {
            role: Role::User,
            content: Content::text("Count r's in 'strawberry'"),
        }],
        tools: Some(vec![tool.into()]),
        tool_choice: Some(ToolChoice::method("count_letters")),
        ..Default::default()
    };

    let mut session =
        drama_llama::LlamaCppSession::from_path_sync(model_path())
            .expect("session load")
            .quiet()
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
    let tool = Tool::builder("count_letters")
        .description(
            "Count the number of times a letter appears in a string.",
        )
        .schema(json!({
            "type": "object",
            "properties": {
                "letter": {"type": "string", "description": "the letter to count"},
                "string": {"type": "string", "description": "the string to search"}
            },
            "required": ["letter", "string"]
        }))
        .build()
        .expect("valid test tool");
    Prompt {
        // Match the strawberry example's system prompt — short ones
        // give the model too much latitude to hallucinate args.
        system: Some(Content::text(
            "You are a helpful assistant. You cannot count letters in a \
             word reliably on your own because you see in tokens, not \
             letters. Use the `count_letters` tool when asked to count \
             characters.",
        )),
        messages: vec![Message {
            role: Role::User,
            content: Content::text("Count the number of r's in 'strawberry'"),
        }],
        tools: Some(vec![tool.into()]),
        tool_choice: Some(ToolChoice::method("count_letters")),
        ..Default::default()
    }
}

/// Phase 3: `complete` returns a `Message` whose tool_use content
/// matches the `letter` / `string` we asked the model to count.
#[test]
#[ignore = "requires model"]
fn complete_returns_message_with_tool_use() {
    let prompt = strawberry_turn_1_prompt();
    let mut session =
        drama_llama::LlamaCppSession::from_path_sync(model_path())
            .expect("session load")
            .quiet()
            .with_max_tokens(NonZeroUsize::new(256).unwrap());

    let assistant = session.complete(&prompt).expect("complete");
    println!("=== complete message ===\n{assistant:#?}\n===");

    let msg: Message = assistant.into();
    assert_eq!(msg.role, Role::Assistant);
    // Must contain a ToolUse block.
    let blocks: Vec<&Block> = msg.content.0.iter().collect();
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
    let mut session =
        drama_llama::LlamaCppSession::from_path_sync(model_path())
            .expect("session load")
            .quiet()
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

/// Round-trip byte-stability — the #30 cache-correctness invariant,
/// asserted end-to-end: `render(parse(emission))` must reproduce the
/// emission byte-for-byte within the assistant span. Parse the raw
/// bytes through the session's template-derived dialect into an
/// [`AssistantMessage`], tack it onto the original [`Prompt`],
/// re-render, and assert the raw bytes are a byte prefix of the
/// re-rendered suffix. This is what keeps the prefix cache (and its
/// hash-keyed auto-tip) valid across tool turns — including the
/// Qwen3.6 XML-ish shape the pre-dialect parser couldn't re-ingest
/// (the old `<function=` skip is exactly what #29/#30 Phase E
/// removed).
///
/// One inference call so the bytes we're comparing against are
/// deterministically the same bytes that got parsed — no seed /
/// KV-cache drift worries.
#[test]
#[ignore = "requires model"]
fn complete_text_round_trips_through_parse_and_render() {
    use drama_llama::AssistantMessage;

    let prompt = strawberry_turn_1_prompt();
    let mut session =
        drama_llama::LlamaCppSession::from_path_sync(model_path())
            .expect("session load")
            .quiet()
            .with_max_tokens(NonZeroUsize::new(256).unwrap());
    println!("=== dialect ===\n{:#?}\n===", session.dialect());

    // Mirror the session's own render defaults (`from_engine`):
    // generation prompt on, `preserve_thinking` on, and NO explicit
    // `enable_thinking` — the template derives it from
    // `prompt.thinking`, and this prompt doesn't enable it. Using
    // different opts here would compute the wrong `pre_opened` flag
    // and desync the byte-prefix comparison from what `complete_text`
    // actually rendered.
    let render_opts = RenderOptions::default()
        .with_generation_prompt(true)
        .with_extra("preserve_thinking", true);
    let rendered_original = session
        .template()
        .render_with(&prompt, &render_opts)
        .expect("render original");
    let reasoning_open = session.dialect().reasoning.start.trim().to_owned();
    let pre_opened = !reasoning_open.is_empty()
        && rendered_original
            .trim_end()
            .ends_with(reasoning_open.as_str());

    let raw = session.complete_text(&prompt).expect("complete_text");
    println!("=== raw emission ===\n{raw}\n===");

    // Parse the same bytes the way `Session::complete*` do.
    let blocks = parse_with_dialect(&session, &prompt, &raw, pre_opened);
    assert!(!blocks.is_empty(), "parser dropped the output: {raw:?}");
    assert!(
        blocks.iter().any(|b| matches!(b, Block::ToolUse { .. })),
        "Method tool_choice must parse to a ToolUse block; got {blocks:?}"
    );
    let assistant: AssistantMessage = blocks.into_iter().collect();

    // Build a follow-up prompt with the assistant turn appended, and
    // render via the same template that drove inference. Tool choice
    // is cleared so the assistant turn is final, not forcing another
    // call.
    let mut follow_up = prompt.clone();
    follow_up.messages.push(assistant.into());
    follow_up.tool_choice = None;
    let rendered_follow_up = session
        .template()
        .render_with(
            &follow_up,
            &RenderOptions::default()
                .with_generation_prompt(false)
                .with_extra("preserve_thinking", true),
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

    // The invariant itself. `complete_text` trims the EOS piece and
    // trailing whitespace, so `raw` is a (possibly shortened) prefix
    // of the true emission — `starts_with` is exactly the right
    // comparison. If this fails, emitter and template have drifted:
    // the canonicalization gate in `Session::run_call` will keep the
    // cache safe (by skipping the auto-tip), but every tool turn
    // pays a re-prefill — fix the dialect, don't relax the assert.
    assert!(
        suffix.starts_with(&raw),
        "emission is not a byte prefix of the canonical re-render.\n\
         --- emission ---\n{raw}\n--- re-rendered suffix ---\n{suffix}"
    );
}

/// #30 Phase E: under `Auto` (unforced) tool choice the model calls
/// the tool in its **native** dialect — no system-prompt retcon, no
/// forced-JSON off-distribution emission — via the lazy
/// trigger-activated grammar, and the dialect parser re-ingests it.
/// This is the unforced path issue #27 reported broken (Qwen XML
/// calls came back as `Block::Text`).
#[test]
#[ignore = "requires model"]
fn auto_tool_choice_parses_native_dialect_call() {
    let mut prompt = strawberry_turn_1_prompt();
    prompt.tool_choice = Some(ToolChoice::auto());
    let mut session =
        drama_llama::LlamaCppSession::from_path_sync(model_path())
            .expect("session load")
            .quiet()
            .with_max_tokens(NonZeroUsize::new(1024).unwrap());

    let blocks = session.complete_blocks(&prompt).expect("complete_blocks");
    println!("=== auto blocks ===\n{blocks:#?}\n===");

    let call = blocks
        .iter()
        .find_map(|b| match b {
            Block::ToolUse { call } => Some(call),
            _ => None,
        })
        .expect("unforced path must still parse the native tool call");
    assert_eq!(call.name, "count_letters");
    assert!(
        call.input.get("letter").is_some()
            && call.input.get("string").is_some(),
        "arguments must coerce to typed JSON, got: {:?}",
        call.input
    );
}

/// #30 Phase E: reasoning works *under* an eager (Method-forced)
/// grammar. Qwen-style templates pre-open `<think>\n` in the
/// generation prompt; the grammar anchors on that tail
/// (`Anchor::EagerThoughtPreOpened`) instead of demanding a fresh
/// open tag, and the parser attributes the pre-close bytes to a
/// `Thought` block. Before Phase A/E this combination force-EOS'd or
/// mislabeled the reasoning.
#[test]
#[ignore = "requires model"]
fn thinking_works_under_forced_tool_grammar() {
    use misanthropic::prompt::thinking::Thinking;
    // Thinking must be *enabled on the prompt* for the template to
    // pre-open `<think>` in the generation prompt — that's the
    // combination this test exists to exercise.
    let prompt = strawberry_turn_1_prompt().thinking(Thinking::Enabled {
        budget_tokens: std::num::NonZeroU32::new(512).unwrap(),
        display: None,
    });
    // Explicit n_ctx: `from_path` inherits llama.cpp's default 512,
    // and prefill (~330) + a stream-dependent think phase overflows
    // it — the context ceiling silently ends iteration mid-call and
    // surfaces as GrammarViolation (bit us when the 0.8.0 RNG swap
    // produced a longer-thinking trajectory). Room for prompt +
    // max_tokens is the requirement.
    let mut session = drama_llama::LlamaCppSession::from_path_with_n_ctx(
        model_path(),
        4096,
    )
    .expect("session load")
    .quiet()
    .with_max_tokens(NonZeroUsize::new(1024).unwrap());

    let blocks = session.complete_blocks(&prompt).expect("complete_blocks");
    println!("=== forced blocks ===\n{blocks:#?}\n===");

    assert!(
        blocks.iter().any(|b| matches!(b, Block::ToolUse { .. })),
        "Method tool_choice must produce a ToolUse block"
    );
    // Thought is model-dependent in principle, but a pre-opened
    // `<think>` template makes some reasoning bytes all but
    // guaranteed; the point is they parse as Thought, not Text.
    if let Some(Block::Thought { thought, .. }) =
        blocks.iter().find(|b| matches!(b, Block::Thought { .. }))
    {
        assert!(
            !thought.trim().is_empty(),
            "pre-opened reasoning parsed as an empty Thought"
        );
    }
    assert!(
        !blocks.iter().any(|b| matches!(
            b,
            Block::Text { text, .. } if text.contains("</think>")
        )),
        "reasoning close marker leaked into a Text block: {blocks:#?}"
    );
}

/// #30 Phase E cache-stability: across a tool turn (call → result →
/// follow-up) the prefix cache reuses KV state instead of
/// re-prefilling from scratch. This only holds when emission,
/// re-render, and parse agree byte-for-byte — i.e. the whole dialect
/// pipeline is consistent. GPU wall-clock comparisons are done
/// elsewhere; here we assert the accounting: turn 2 must report
/// nonzero `cache_read_input_tokens`.
#[test]
#[ignore = "requires model"]
fn prefix_cache_survives_tool_turn() {
    use misanthropic::prompt::message::AssistantMessage;

    let mut prompt = strawberry_turn_1_prompt();
    // Breakpoint after the tools block anchors the front of the
    // prompt; the auto-tip covers the generated turn.
    if let Some(tools) = prompt.tools.as_mut() {
        if let Some(def) = tools.first_mut() {
            if let Some(tool) = def.as_method_mut() {
                tool.cache_control = Some(
                    misanthropic::prompt::message::CacheControl::ephemeral(),
                );
            }
        }
    }
    let mut session =
        drama_llama::LlamaCppSession::from_path_sync(model_path())
            .expect("session load")
            .quiet()
            .with_prefix_cache(true)
            .with_max_tokens(NonZeroUsize::new(1024).unwrap());

    // Turn 1: forced call.
    let blocks = session.complete_blocks(&prompt).expect("turn 1");
    let call = blocks
        .iter()
        .find_map(|b| match b {
            Block::ToolUse { call } => Some(call.clone()),
            _ => None,
        })
        .expect("turn 1 must produce a ToolUse block");

    // Turn 2: append assistant turn + tool result, ask for prose.
    let assistant: AssistantMessage = blocks.iter().cloned().collect();
    prompt.messages.push(assistant.into());
    prompt.messages.push(Message {
        role: Role::User,
        content: Content(vec![Block::ToolResult {
            result: ToolResult {
                tool_use_id: call.id.clone(),
                content: Content::text("3"),
                is_error: false,
                cache_control: None,
            },
        }]),
    });
    prompt.tool_choice = None;

    let out = session.complete_text(&prompt).expect("turn 2");
    println!("=== turn 2 ===\n{out}\n===");
    let read = session
        .last_usage()
        .cache_read_input_tokens
        .unwrap_or_default();
    assert!(
        read > 0,
        "turn 2 reused no prefix — emission/re-render drift broke the \
         cache across the tool turn (usage: {:?})",
        session.last_usage()
    );
}

/// `complete_response_id` threads the caller-supplied UUID through to
/// `Message::id`. Used by blallama to correlate the sync response with
/// the per-token probe stream — both sides need to share the same id.
#[test]
#[ignore = "requires model"]
fn complete_response_id_uses_supplied_uuid() {
    let prompt = Prompt {
        messages: vec![Message {
            role: Role::User,
            content: Content::text("Say hi."),
        }],
        ..Default::default()
    };

    let mut session =
        drama_llama::LlamaCppSession::from_path_sync(model_path())
            .expect("session load")
            .quiet()
            .with_max_tokens(NonZeroUsize::new(4).unwrap());

    let id = uuid::Uuid::from_u128(0x0123_4567_89AB_CDEF_FEDC_BA98_7654_3210);
    let response = session
        .complete_response_id(&prompt, id)
        .expect("complete_response_id");

    assert_eq!(
        response.id.as_ref(),
        id.to_string(),
        "response.id must match the supplied UUID",
    );
}
