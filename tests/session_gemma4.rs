//! Gemma 4 e2e (#30 Phase F): the `TagWithDict` dialect against the
//! real model. Mirrors the Phase E Qwen suite in `session.rs` —
//! forced/auto calls, thinking under grammar, round-trip
//! byte-stability, prefix-cache survival — for the dict-encoded
//! native format (`<|tool_call>call:name{k:<|"|>v<|"|>}<tool_call|>`).
//!
//! All tests load `models/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf` and are
//! `#[ignore]`d. Run with
//! `cargo test --features serde --test session_gemma4 -- --ignored`.

use std::{borrow::Cow, num::NonZeroUsize, path::PathBuf};

use drama_llama::{
    prompt::{ToolResult, ToolUse},
    Block, CallSyntax, Content, Message, Prompt, RenderOptions, Role, Tool,
    ToolChoice,
};
use serde_json::json;

fn model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf")
}

fn load_session(max_tokens: usize) -> drama_llama::LlamaCppSession {
    drama_llama::LlamaCppSession::from_path_sync(model_path())
        .expect("session load")
        .quiet()
        .with_max_tokens(NonZeroUsize::new(max_tokens).unwrap())
}

fn count_letters_prompt() -> Prompt {
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

/// The sniff patch must fire on the real GGUF's template at load —
/// everything else in this suite builds on that.
#[test]
#[ignore = "requires Gemma 4 model"]
fn dialect_resolves_to_gemma4_at_load() {
    let session = load_session(16);
    assert_eq!(
        session.dialect(),
        &CallSyntax::gemma4(),
        "template sniff must resolve the baked Gemma 4 dialect"
    );
}

/// Method-forced call: the grammar forces the dict envelope and the
/// parser returns a typed ToolUse.
#[test]
#[ignore = "requires Gemma 4 model"]
fn forced_call_parses_to_tool_use() {
    let prompt = count_letters_prompt();
    let mut session = load_session(256);

    let blocks = session.complete_blocks(&prompt).expect("complete_blocks");
    println!("=== forced blocks ===\n{blocks:#?}\n===");

    let call = blocks
        .iter()
        .find_map(|b| match b {
            Block::ToolUse { call } => Some(call),
            _ => None,
        })
        .expect("Method tool_choice must produce a ToolUse block");
    assert_eq!(call.name, "count_letters");
    assert!(
        call.input.get("letter").and_then(|v| v.as_str()).is_some(),
        "letter arg missing/untyped: {:?}",
        call.input
    );
    assert!(
        call.input.get("string").and_then(|v| v.as_str()).is_some(),
        "string arg missing/untyped: {:?}",
        call.input
    );
}

/// Auto tool choice: lazy trigger-activated grammar on
/// `<|tool_call>`, native dict emission, dialect parse — the
/// unforced path.
#[test]
#[ignore = "requires Gemma 4 model"]
fn auto_tool_choice_parses_native_dict_call() {
    let mut prompt = count_letters_prompt();
    prompt.tool_choice = Some(ToolChoice::auto());
    let mut session = load_session(1024);

    let blocks = session.complete_blocks(&prompt).expect("complete_blocks");
    println!("=== auto blocks ===\n{blocks:#?}\n===");

    let call = blocks
        .iter()
        .find_map(|b| match b {
            Block::ToolUse { call } => Some(call),
            _ => None,
        })
        .expect("unforced path must still parse the native dict call");
    assert_eq!(call.name, "count_letters");
    assert!(
        call.input.get("letter").is_some()
            && call.input.get("string").is_some(),
        "arguments must coerce to typed JSON, got: {:?}",
        call.input
    );
}

/// Thinking under the eager (Method-forced) grammar. Gemma renders a
/// bare `<|turn>model\n` generation prompt when thinking is enabled
/// (no pre-open — the *opposite* of Qwen); the model emits its own
/// `<|channel>thought\n…\n<channel|>` which the optional-reasoning
/// grammar prefix must admit and the parser must label `Thought`.
#[test]
#[ignore = "requires Gemma 4 model"]
fn thinking_works_under_forced_tool_grammar() {
    use misanthropic::prompt::thinking::Thinking;
    let prompt = count_letters_prompt().thinking(Thinking::Enabled {
        budget_tokens: std::num::NonZeroU32::new(512).unwrap(),
        display: None,
    });
    let mut session = load_session(1024);

    let blocks = session.complete_blocks(&prompt).expect("complete_blocks");
    println!("=== thinking blocks ===\n{blocks:#?}\n===");

    assert!(
        blocks.iter().any(|b| matches!(b, Block::ToolUse { .. })),
        "Method tool_choice must produce a ToolUse block"
    );
    if let Some(Block::Thought { thought, .. }) =
        blocks.iter().find(|b| matches!(b, Block::Thought { .. }))
    {
        assert!(
            !thought.trim().is_empty(),
            "reasoning parsed as an empty Thought"
        );
    }
    assert!(
        !blocks.iter().any(|b| matches!(
            b,
            Block::Text { text, .. } if text.contains("<channel|>")
                || text.contains("<|channel>")
        )),
        "channel markers leaked into a Text block: {blocks:#?}"
    );
}

/// Round-trip byte-stability e2e — the #30 cache-correctness
/// invariant for the dict dialect: the raw emission must be a byte
/// prefix of the canonical template re-render of the parsed blocks.
#[test]
#[ignore = "requires Gemma 4 model"]
fn emission_round_trips_through_parse_and_render() {
    use drama_llama::AssistantMessage;

    let prompt = count_letters_prompt();
    let mut session = load_session(256);
    println!("=== dialect ===\n{:#?}\n===", session.dialect());

    // Mirror the session's own render defaults, including the
    // dialect-driven thought re-ingest convention.
    let render_opts = RenderOptions::default()
        .with_generation_prompt(true)
        .with_extra("preserve_thinking", true)
        .with_thought_reingest(session.dialect().reasoning.reingest);
    let rendered_original = session
        .template()
        .render_with(&prompt, &render_opts)
        .expect("render original");

    let raw = session.complete_text(&prompt).expect("complete_text");
    println!("=== raw emission ===\n{raw}\n===");

    let tool_refs: Vec<&Tool> = prompt
        .tools
        .iter()
        .flatten()
        .filter_map(|def| def.as_method())
        .collect();
    let blocks = drama_llama::dialect::parse_text(
        session.dialect(),
        &tool_refs,
        &raw,
        false, // Gemma never pre-opens reasoning in the prompt tail
        drama_llama::dialect::Leniency::Final,
    )
    .blocks;
    assert!(
        blocks.iter().any(|b| matches!(b, Block::ToolUse { .. })),
        "Method tool_choice must parse to a ToolUse block; got {blocks:?}"
    );
    let assistant: AssistantMessage = blocks.into_iter().collect();

    let mut follow_up = prompt.clone();
    follow_up.messages.push(assistant.into());
    follow_up.tool_choice = None;
    let rendered_follow_up = session
        .template()
        .render_with(
            &follow_up,
            &RenderOptions::default()
                .with_generation_prompt(false)
                .with_extra("preserve_thinking", true)
                .with_thought_reingest(session.dialect().reasoning.reingest),
        )
        .expect("render follow_up");

    // Gemma template quirk (accepted): the NON-thinking generation
    // prompt ends with a pre-closed empty thought scaffold
    // (`<|channel>thought\n<channel|>`) that a re-ingested assistant
    // turn does not reproduce — the same family of quirk upstream
    // works around at chat.cpp:1223. Production pays a one-turn
    // re-prefill via the canonicalization LCP fallback; the shared
    // prefix must therefore extend through `<|turn>model\n`, and the
    // emission must follow it byte-for-byte in the re-render. (In
    // thinking mode there is no scaffold and stability is full —
    // pinned by the reconstruction harness.)
    let scaffold = "<|channel>thought\n<channel|>";
    let base = rendered_original
        .strip_suffix(scaffold)
        .unwrap_or(&rendered_original);
    let suffix = rendered_follow_up.strip_prefix(base).unwrap_or_else(|| {
        panic!(
            "follow-up must extend the original prefix (modulo the \
                 empty-thought scaffold).\n\
                 --- original ---\n{rendered_original}\n\
                 --- follow-up ---\n{rendered_follow_up}"
        )
    });
    assert!(
        suffix.starts_with(&raw),
        "emission is not a byte prefix of the canonical re-render.\n\
         --- emission ---\n{raw}\n--- re-rendered suffix ---\n{suffix}"
    );
}

/// Tool turn 2: assistant call + tool result re-ingest through the
/// dict dialect (the template's forward-scan of `role: tool`
/// messages), then free prose. The answer should surface the result.
#[test]
#[ignore = "requires Gemma 4 model"]
fn tool_result_turn_produces_prose_answer() {
    let call_id = "call_0_count_letters";
    let mut prompt = count_letters_prompt();
    prompt.tool_choice = None;
    prompt.messages.push(Message {
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
    });
    prompt.messages.push(Message {
        role: Role::User,
        content: Content(vec![Block::ToolResult {
            result: ToolResult {
                tool_use_id: Cow::Borrowed(call_id),
                content: Content::text("3"),
                is_error: false,
                cache_control: None,
            },
        }]),
    });

    let mut session = load_session(256);
    let out = session.complete_text(&prompt).expect("complete_text");
    println!("=== turn 2 ===\n{out}\n===");
    assert!(!out.trim().is_empty(), "got empty output");
    assert!(
        out.contains('3') || out.to_lowercase().contains("three"),
        "expected the count (3) in the answer, got: {out:?}"
    );
}

/// Prefix cache across a tool turn: emission == re-render byte-for-
/// byte means turn 2 reuses KV state. Nonzero cache_read is the
/// accounting proof the whole dict pipeline is byte-stable.
#[test]
#[ignore = "requires Gemma 4 model"]
fn prefix_cache_survives_tool_turn() {
    use misanthropic::prompt::message::AssistantMessage;

    let mut prompt = count_letters_prompt();
    if let Some(tools) = prompt.tools.as_mut() {
        if let Some(def) = tools.first_mut() {
            if let Some(tool) = def.as_method_mut() {
                tool.cache_control = Some(
                    misanthropic::prompt::message::CacheControl::ephemeral(),
                );
            }
        }
    }
    let mut session = load_session(1024).with_prefix_cache(true);

    let blocks = session.complete_blocks(&prompt).expect("turn 1");
    let call = blocks
        .iter()
        .find_map(|b| match b {
            Block::ToolUse { call } => Some(call.clone()),
            _ => None,
        })
        .expect("turn 1 must produce a ToolUse block");

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
