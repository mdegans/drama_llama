//! gpt-oss e2e (#30 Phase G): the Harmony dialect against the real
//! model. Mirrors the Phase E/F suites — forced/auto calls, thinking
//! under grammar, round-trip byte-stability, prefix-cache survival —
//! for the channel-structured Harmony format
//! (`<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{args}<|call|>`).
//!
//! All tests load `models/gpt-oss-20b-UD-Q8_K_XL.gguf` and are
//! `#[ignore]`d. Run with
//! `cargo test --features serde,cuda --test session_gptoss -- --ignored`.

use std::{borrow::Cow, num::NonZeroUsize, path::PathBuf};

use drama_llama::{
    prompt::{ToolResult, ToolUse},
    Block, CallSyntax, Content, Message, Prompt, RenderOptions, Role, Tool,
    ToolChoice,
};
use serde_json::json;

fn model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models/gpt-oss-20b-UD-Q8_K_XL.gguf")
}

/// Install the cache-stability template sidecar next to the model —
/// the deployment configuration this suite validates (blallama ships
/// the same file). Idempotent; sourced from the versioned fixture.
fn install_template_sidecar() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/templates/gptoss-cache-stable.jinja");
    let sidecar = model_path().with_extension("template.jinja");
    std::fs::copy(&fixture, &sidecar).expect("install template sidecar");
}

fn load_session(max_tokens: usize) -> drama_llama::LlamaCppSession {
    install_template_sidecar();
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

/// The sniff must fire on the real GGUF's template at load —
/// everything else in this suite builds on that.
#[test]
#[ignore = "requires gpt-oss model"]
fn dialect_resolves_to_gpt_oss_at_load() {
    let session = load_session(16);
    assert_eq!(
        session.dialect(),
        &CallSyntax::gpt_oss(),
        "template sniff must resolve the baked Harmony dialect"
    );
}

/// Method-forced call: the eager grammar admits analysis/preamble
/// blocks and then forces the canonical channel-header call; the
/// parser returns a typed ToolUse and no channel markers leak into
/// prose.
#[test]
#[ignore = "requires gpt-oss model"]
fn forced_call_parses_to_tool_use() {
    let prompt = count_letters_prompt();
    let mut session = load_session(1024);

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
    // gpt-oss reasons by default; if a Thought came back it must be
    // non-empty, and Harmony envelope must never leak into Text.
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
            Block::Text { text, .. } if text.contains("<|channel|>")
                || text.contains("<|message|>")
                || text.contains("<|start|>")
        )),
        "Harmony markers leaked into a Text block: {blocks:#?}"
    );
}

/// Auto tool choice: lazy any-of trigger grammar on the recipient
/// headers, native emission, dialect parse — the unforced path.
#[test]
#[ignore = "requires gpt-oss model"]
fn auto_tool_choice_parses_native_call() {
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
        .expect("unforced path must still parse the native Harmony call");
    assert_eq!(call.name, "count_letters");
    assert!(
        call.input.get("letter").is_some()
            && call.input.get("string").is_some(),
        "arguments must parse to typed JSON, got: {:?}",
        call.input
    );
}

/// Round-trip byte-stability e2e — the #30 cache-correctness
/// invariant for Harmony: the raw emission must be a byte prefix of
/// the canonical template re-render of the parsed blocks (sidecar
/// installed: analysis kept, channel-header call shape).
#[test]
#[ignore = "requires gpt-oss model"]
fn emission_round_trips_through_parse_and_render() {
    use drama_llama::AssistantMessage;

    let prompt = count_letters_prompt();
    let mut session = load_session(1024);
    println!("=== dialect ===\n{:#?}\n===", session.dialect());

    let render_opts = RenderOptions::default()
        .with_generation_prompt(true)
        .with_extra("preserve_thinking", true)
        .with_thought_reingest(session.dialect().reasoning.reingest);
    let rendered_original = session
        .template()
        .render_with(&prompt, &render_opts)
        .expect("render original");

    let raw = session.complete_text(&prompt).expect("complete_text");
    println!("=== raw emission ===\n{raw:?}\n===");

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
        false, // gpt-oss never pre-opens reasoning in the prompt tail
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

    // STRICT byte-prefix: with the cache-stability sidecar installed
    // (`gptoss-cache-stable.jinja`), the re-ingested turn keeps the
    // analysis block and re-renders the call in the trained
    // channel-header shape, so the follow-up render extends the
    // generation prompt exactly — no LCP fallback.
    let suffix = rendered_follow_up
        .strip_prefix(&rendered_original)
        .unwrap_or_else(|| {
            panic!(
                "follow-up must extend the original prefix exactly \
                 (is the template sidecar installed?).\n\
                 --- original ---\n{rendered_original}\n\
                 --- follow-up ---\n{rendered_follow_up}"
            )
        });
    assert!(
        suffix.starts_with(&raw),
        "emission is not a byte prefix of the canonical re-render.\n\
         --- emission ---\n{raw:?}\n--- re-rendered suffix ---\n{suffix:?}"
    );
}

/// Announce-then-call (causality): nudged under Auto, prose the model
/// emits before its call comes back in emission order (commentary
/// preamble block) and the re-render reproduces the emission bytes.
#[test]
#[ignore = "requires gpt-oss model"]
fn announce_then_call_round_trips_in_emission_order() {
    use drama_llama::AssistantMessage;

    let mut prompt = count_letters_prompt();
    // Harmony-idiomatic preamble nudge: the first e2e rounds showed
    // the model *planning* an announcement in analysis but skipping
    // the commentary preamble unless the channel is named explicitly.
    prompt.system = Some(Content::text(
        "You are a helpful assistant. Use the `count_letters` tool when \
         asked to count characters. IMPORTANT: before every tool call, \
         first output a preamble message on the commentary channel (no \
         recipient) telling the user what you are about to do — e.g. \
         `<|channel|>commentary<|message|>I'll count that now.<|end|>` \
         — and only then make the tool call.",
    ));
    prompt.tool_choice = Some(ToolChoice::auto());
    let mut session = load_session(1024);

    let render_opts = RenderOptions::default()
        .with_generation_prompt(true)
        .with_extra("preserve_thinking", true)
        .with_thought_reingest(session.dialect().reasoning.reingest);
    let rendered_original = session
        .template()
        .render_with(&prompt, &render_opts)
        .expect("render original");

    let raw = session.complete_text(&prompt).expect("complete_text");
    println!("=== raw emission ===\n{raw:?}\n===");

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
        false,
        drama_llama::dialect::Leniency::Final,
    )
    .blocks;
    println!("=== blocks ===\n{blocks:#?}\n===");
    let first_text =
        blocks.iter().position(|b| matches!(b, Block::Text { .. }));
    let first_call = blocks
        .iter()
        .position(|b| matches!(b, Block::ToolUse { .. }))
        .expect("nudged auto should still call the tool");
    let announced = first_text.is_some_and(|t| t < first_call);
    assert!(
        announced,
        "nudge did not produce prose before the call this run — \
         blocks: {blocks:#?}"
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
    let suffix = rendered_follow_up
        .strip_prefix(&rendered_original)
        .unwrap_or_else(|| {
            panic!(
                "follow-up must extend the original prefix exactly.\n\
                 --- original ---\n{rendered_original}\n\
                 --- follow-up ---\n{rendered_follow_up}"
            )
        });
    assert!(
        suffix.starts_with(&raw),
        "announce-then-call emission is not a byte prefix of the \
         re-render — prose got reordered or altered.\n\
         --- emission ---\n{raw:?}\n--- suffix ---\n{suffix:?}"
    );
}

/// Tool turn 2: assistant call + tool result re-ingest (the sidecar's
/// forward-scan of `role: tool` messages), then free prose. The
/// answer should surface the result.
#[test]
#[ignore = "requires gpt-oss model"]
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

    let mut session = load_session(1024);
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
/// accounting proof the whole Harmony pipeline is byte-stable.
#[test]
#[ignore = "requires gpt-oss model"]
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

/// The EOG contract Phase G's grammar and stop logic rely on, pinned
/// against the real vocab: `<|return|>` and `<|call|>` end
/// generation, while `<|end|>` (the in-stream channel separator) must
/// NOT — libllama's o200k_harmony workaround removes it from
/// `special_eog_ids`, and `extra_eos_tokens` must surface exactly
/// that set. CPU-only load: vocab introspection needs no GPU.
#[test]
#[ignore = "long running - requires gpt-oss model"]
fn gptoss_eog_token_set() {
    use drama_llama::Model as _;

    let mut params = unsafe { llama_cpp_sys_3::llama_model_default_params() };
    params.n_gpu_layers = 0;
    let model =
        drama_llama::LlamaCppModel::from_file(model_path(), Some(params))
            .expect("model load");

    let piece_of = |t| drama_llama::Model::token_to_piece(&model, t);
    let by_piece = |s: &str| {
        let toks = model.tokenize(s, true);
        assert_eq!(toks.len(), 1, "{s:?} must be a single token: {toks:?}");
        toks[0]
    };

    // GGUF metadata: eos = <|return|>, eot = <|endoftext|>.
    assert_eq!(piece_of(model.eos()), "<|return|>");

    let call = by_piece("<|call|>");
    let end = by_piece("<|end|>");
    let ret = by_piece("<|return|>");

    let extra = model.extra_eos_tokens();
    assert!(
        extra.contains(&call),
        "<|call|> must stop generation (tool-call turn exit); extra = \
         {:?}",
        extra.iter().map(|&t| piece_of(t)).collect::<Vec<_>>()
    );
    assert!(
        !extra.contains(&end) && model.eos() != end && model.eot() != end,
        "<|end|> is the in-stream channel separator and must stay \
         generatable"
    );
    assert!(model.eos() == ret || extra.contains(&ret));
}
