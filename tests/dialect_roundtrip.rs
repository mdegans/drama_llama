//! Reconstruction harness (tool-dialects plan, Phase D): for each
//! tool-capable fixture template, assert the cache-correctness
//! invariant `render(parse(emission)) ⊇ emission` — the canonical
//! bytes [`render_reference`] produces (and the emitted grammar
//! forces) must reappear byte-for-byte when the parsed calls are
//! re-rendered through the *real* chat template. Mirrors llama.cpp's
//! `expect_reconstruction()` (`tests/test-chat.cpp`).

use std::borrow::Cow;

use drama_llama::dialect::{
    analyze_template, parse_text, render_reference, CallSyntax, Leniency,
    ParseStatus,
};
use drama_llama::prompt::{Content, Message, Role, ToolUse};
use drama_llama::{Block, ChatTemplate, Prompt, RenderOptions, Tool};
use serde_json::json;

fn fixture_source(name: &str) -> String {
    let path = format!(
        "{}/tests/fixtures/templates/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

fn test_tool() -> Tool {
    Tool::builder("get_weather")
        .description("Get the weather for a city.")
        .schema(json!({
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "city name"},
                "days": {"type": "integer", "description": "forecast days"},
            },
            "required": ["city", "days"],
        }))
        .build()
        .expect("valid test tool")
}

/// The harness: analyze the template, produce a canonical emission,
/// parse it back, feed the parsed calls through the real template,
/// and assert the emission bytes survive intact in the re-render.
fn assert_reconstruction(fixture: &str, bos: &str, eos: &str) {
    let source = fixture_source(fixture);
    let syntax: CallSyntax =
        analyze_template(&source, bos, eos).expect("analyze");
    let tool = test_tool();

    let input = json!({"city": "Paris", "days": 3});
    let emission = render_reference(&syntax, &[("get_weather", &input)])
        .expect("representable");

    // Emission → blocks.
    let parsed =
        parse_text(&syntax, &[&tool], &emission, false, Leniency::Final);
    assert_eq!(
        parsed.status,
        ParseStatus::Complete,
        "{fixture}: {emission:?} → {:#?}",
        parsed.blocks
    );
    let calls: Vec<&ToolUse> = parsed
        .blocks
        .iter()
        .filter_map(|b| match b {
            Block::ToolUse { call } => Some(call),
            _ => None,
        })
        .collect();
    assert_eq!(
        calls.len(),
        1,
        "{fixture}: emission {emission:?} → {:#?}",
        parsed.blocks
    );
    assert_eq!(calls[0].name.as_ref(), "get_weather", "{fixture}");
    assert_eq!(calls[0].input, input, "{fixture}");
    // A clean emission parses to calls ONLY — a stray Text block
    // means the parser left marker crumbs behind.
    assert!(
        parsed
            .blocks
            .iter()
            .all(|b| matches!(b, Block::ToolUse { .. })),
        "{fixture}: stray non-call blocks: {:#?}",
        parsed.blocks
    );

    // Blocks → template re-render.
    let prompt = Prompt {
        messages: vec![
            Message {
                role: Role::User,
                content: Content::text("What's the weather in Paris?"),
            },
            Message {
                role: Role::Assistant,
                content: Content(vec![Block::ToolUse {
                    call: ToolUse {
                        id: Cow::Borrowed("call00001"),
                        name: Cow::Borrowed("get_weather"),
                        input: input.clone(),
                        cache_control: None,
                        caller: None,
                    },
                }]),
            },
        ],
        tools: Some(vec![tool.clone().into()]),
        ..Default::default()
    };
    let template =
        ChatTemplate::from_source(source, bos.to_string(), eos.to_string())
            .expect("template compiles");
    let opts = RenderOptions::default()
        .with_generation_prompt(false)
        .with_extra("enable_thinking", true);
    let rendered = template.render_with(&prompt, &opts).expect("render");

    // The invariant: canonical emission bytes appear verbatim in the
    // re-render. Any drift here is a future prefix-cache invalidation
    // on every tool turn.
    assert!(
        rendered.contains(&emission),
        "{fixture}: reconstruction drift.\n--- canonical emission ---\n\
         {emission:?}\n--- template re-render ---\n{rendered:?}"
    );
}

#[test]
fn reconstruct_qwen3_coder() {
    assert_reconstruction("Qwen3-Coder.jinja", "", "<|im_end|>");
}

#[test]
fn reconstruct_qwen36_gguf() {
    assert_reconstruction("qwen3.6-gguf.jinja", "", "<|im_end|>");
}

#[test]
fn reconstruct_qwen35() {
    assert_reconstruction("Qwen3.5-4B.jinja", "", "<|im_end|>");
}

#[test]
fn reconstruct_hermes3() {
    assert_reconstruction(
        "NousResearch-Hermes-3-Llama-3.1-8B-tool_use.jinja",
        "",
        "<|im_end|>",
    );
}

#[test]
fn reconstruct_qwen3_chat() {
    assert_reconstruction("Qwen-Qwen3-0.6B.jinja", "", "<|im_end|>");
}

#[test]
fn reconstruct_llama31() {
    assert_reconstruction(
        "meta-llama-Llama-3.1-8B-Instruct.jinja",
        "<|begin_of_text|>",
        "<|eot_id|>",
    );
}

#[test]
fn reconstruct_gemma4_gguf() {
    assert_reconstruction("gemma4-gguf.jinja", "<bos>", "<turn|>");
}

#[test]
fn reconstruct_gemma4_cache_stable() {
    // The patch must not disturb the call-rendering path.
    assert_reconstruction("gemma4-cache-stable.jinja", "<bos>", "<turn|>");
}

/// The cache property itself, pinned FFI-free against the patched
/// template: the generation-prompt render is a byte PREFIX of the
/// follow-up render (turn appended, no generation prompt), so the KV
/// laid down during generation is reusable verbatim. Three cases the
/// stock template breaks:
/// 1. non-thinking — the empty-thought scaffold must reappear on
///    re-ingest;
/// 2. thinking — the model's own thought block must re-render
///    byte-identically;
/// 3. aged thinking — `preserve_thinking` keeps the thought bytes
///    once a later user message lands.
#[test]
fn gemma4_cache_stable_prefix_continuity() {
    use drama_llama::dialect::ReasoningReingest;

    let source = fixture_source("gemma4-cache-stable.jinja");
    let syntax = CallSyntax::gemma4();
    let tool = test_tool();
    let input = json!({"city": "Paris", "days": 3});
    let calls_ref = render_reference(&syntax, &[("get_weather", &input)])
        .expect("representable");
    let template = ChatTemplate::from_source(
        source,
        "<bos>".to_string(),
        "<turn|>".to_string(),
    )
    .expect("template compiles");

    let opts = |gen: bool, thinking: bool| {
        RenderOptions::default()
            .with_generation_prompt(gen)
            .with_extra("enable_thinking", thinking)
            .with_extra("preserve_thinking", true)
            .with_thought_reingest(ReasoningReingest::Field)
    };
    let user = Message {
        role: Role::User,
        content: Content::text("What's the weather in Paris?"),
    };
    let call_block = Block::ToolUse {
        call: ToolUse {
            id: Cow::Borrowed("call00001"),
            name: Cow::Borrowed("get_weather"),
            input: input.clone(),
            cache_control: None,
            caller: None,
        },
    };
    let base = Prompt {
        messages: vec![user.clone()],
        tools: Some(vec![tool.clone().into()]),
        ..Default::default()
    };

    // Case 1: non-thinking. The generation prompt ends with the
    // empty-thought scaffold; the re-ingested turn must reproduce it.
    let p = template
        .render_with(&base, &opts(true, false))
        .expect("render");
    assert!(
        p.ends_with("<|turn>model\n<|channel>thought\n<channel|>"),
        "generation prompt tail changed: {p:?}"
    );
    let mut with_turn = base.clone();
    with_turn.messages.push(Message {
        role: Role::Assistant,
        content: Content(vec![call_block.clone()]),
    });
    let f = template
        .render_with(&with_turn, &opts(false, false))
        .expect("render");
    let suffix = f.strip_prefix(&p).unwrap_or_else(|| {
        panic!(
            "non-thinking: follow-up must extend the generation \
             prompt.\n--- gen ---\n{p:?}\n--- follow-up ---\n{f:?}"
        )
    });
    assert!(
        suffix.starts_with(&calls_ref),
        "non-thinking: emission bytes must follow.\n{suffix:?}"
    );
    assert!(
        suffix[calls_ref.len()..].starts_with("<|tool_response>"),
        "turn-exit marker must follow the calls.\n{suffix:?}"
    );

    // Case 2: thinking. Bare model header; the model's thought block
    // plus calls must re-render byte-identically.
    let p = template
        .render_with(&base, &opts(true, true))
        .expect("render");
    assert!(
        p.ends_with("<|turn>model\n"),
        "thinking generation prompt tail changed: {p:?}"
    );
    let thought = "weighing the options";
    let mut with_turn = base.clone();
    with_turn.messages.push(Message {
        role: Role::Assistant,
        content: Content(vec![
            Block::Thought {
                thought: Cow::Borrowed(thought),
                signature: Cow::Borrowed(""),
            },
            call_block.clone(),
        ]),
    });
    let f = template
        .render_with(&with_turn, &opts(false, true))
        .expect("render");
    let emission =
        format!("<|channel>thought\n{thought}\n<channel|>{calls_ref}");
    let suffix = f.strip_prefix(&p).unwrap_or_else(|| {
        panic!(
            "thinking: follow-up must extend the generation prompt.\n\
             --- gen ---\n{p:?}\n--- follow-up ---\n{f:?}"
        )
    });
    assert!(
        suffix.starts_with(&emission),
        "thinking: thought + calls must re-render byte-exact.\n\
         --- want ---\n{emission:?}\n--- got ---\n{suffix:?}"
    );

    // Case 4: announce-then-call (causality patch). Prose the model
    // emitted before its call re-renders in emission order and
    // VERBATIM (trailing whitespace intact), not reordered into the
    // after-responses slot; post-call prose keeps the native slot.
    let announce = "I'll count the r's for you.\n\n";
    let p = template
        .render_with(&base, &opts(true, false))
        .expect("render");
    let mut with_turn = base.clone();
    with_turn.messages.push(Message {
        role: Role::Assistant,
        content: Content(vec![
            Block::Text {
                text: Cow::Borrowed(announce),
                citations: None,
                cache_control: None,
            },
            call_block.clone(),
        ]),
    });
    let f = template
        .render_with(&with_turn, &opts(false, false))
        .expect("render");
    let suffix = f.strip_prefix(&p).unwrap_or_else(|| {
        panic!(
            "announce-then-call: follow-up must extend the generation \
             prompt.\n--- gen ---\n{p:?}\n--- follow-up ---\n{f:?}"
        )
    });
    let emission = format!("{announce}{calls_ref}");
    assert!(
        suffix.starts_with(&emission),
        "announce-then-call must re-render in emission order, \
         verbatim.\n--- want ---\n{emission:?}\n--- got ---\n{suffix:?}"
    );

    // Case 3: aging. A later user message must not strip the thought
    // bytes (preserve_thinking) — the prior render stays a prefix.
    let mut aged = with_turn.clone();
    aged.messages.push(Message {
        role: Role::User,
        content: Content(vec![Block::ToolResult {
            result: drama_llama::prompt::ToolResult {
                tool_use_id: Cow::Borrowed("call00001"),
                content: Content::text("22C, sunny"),
                is_error: false,
                cache_control: None,
            },
        }]),
    });
    let f_aged = template
        .render_with(&aged, &opts(false, true))
        .expect("render");
    assert!(
        f_aged.contains(&emission),
        "aged turn lost its thought bytes despite preserve_thinking.\n\
         --- want contained ---\n{emission:?}\n--- render ---\n{f_aged:?}"
    );
}

#[test]
fn reconstruct_gemma4_upstream() {
    assert_reconstruction("google-gemma-4-31B-it.jinja", "<bos>", "<turn|>");
}

/// Gemma 4, the full-fidelity version: thought routed through the
/// `reasoning` field (`ReasoningReingest::Field`), parallel calls,
/// and the value-type corners probed against minijinja (null renders
/// `none`, floats in ryu-shortest form, nested dicts re-sorted).
/// The canonical emission — reasoning block plus both calls — must
/// appear byte-for-byte in the template re-render.
#[test]
fn reconstruct_gemma4_thought_and_values() {
    use drama_llama::dialect::ReasoningReingest;

    let source = fixture_source("gemma4-gguf.jinja");
    let syntax: CallSyntax =
        analyze_template(&source, "<bos>", "<turn|>").expect("analyze");
    assert_eq!(syntax, CallSyntax::gemma4(), "sniff patch must fire");

    let tool = Tool::builder("configure")
        .description("Configure a thing.")
        .schema(json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
                "ratio": {"type": "number"},
                "flag": {"type": "boolean"},
                "maybe": {"type": ["string", "null"]},
                "tags": {"type": "array", "items": {"type": "string"}},
                "nested": {"type": "object", "properties": {
                    "z": {"type": "integer"}, "a": {"type": "string"}}},
            },
            "required": ["name", "count"],
        }))
        .build()
        .expect("valid test tool");

    let input_a = json!({
        "name": "unit 🍓",
        "count": 3,
        "ratio": 1.5e10,
        "flag": false,
        "maybe": null,
        "tags": ["x", "y"],
        "nested": {"z": 2, "a": "v"},
    });
    let input_b = json!({"name": "second", "count": 1});
    let thought = "weighing the options";

    let calls_ref = render_reference(
        &syntax,
        &[("configure", &input_a), ("configure", &input_b)],
    )
    .expect("representable");
    // What the template renders for a thinking tool-call turn:
    // reasoning block immediately followed by the calls.
    let emission =
        format!("<|channel>thought\n{thought}\n<channel|>{calls_ref}");

    // Emission → blocks (the parser half).
    let parsed =
        parse_text(&syntax, &[&tool], &emission, false, Leniency::Final);
    assert_eq!(parsed.status, ParseStatus::Complete, "{parsed:#?}");
    let blocks = &parsed.blocks;
    assert!(
        matches!(&blocks[0], Block::Thought { thought: t, .. }
            if t.as_ref() == thought),
        "{blocks:#?}"
    );
    let calls: Vec<&ToolUse> = blocks
        .iter()
        .filter_map(|b| match b {
            Block::ToolUse { call } => Some(call),
            _ => None,
        })
        .collect();
    assert_eq!(calls.len(), 2, "{blocks:#?}");
    assert_eq!(calls[0].input, input_a, "{blocks:#?}");
    assert_eq!(calls[1].input, input_b, "{blocks:#?}");

    // Blocks → template re-render (the render half).
    let prompt = Prompt {
        messages: vec![
            Message {
                role: Role::User,
                content: Content::text("Configure it."),
            },
            Message {
                role: Role::Assistant,
                content: Content(parsed.blocks),
            },
        ],
        tools: Some(vec![tool.into()]),
        ..Default::default()
    };
    let template = ChatTemplate::from_source(
        source,
        "<bos>".to_string(),
        "<turn|>".to_string(),
    )
    .expect("template compiles");
    let opts = RenderOptions::default()
        .with_generation_prompt(false)
        .with_extra("enable_thinking", true)
        .with_thought_reingest(ReasoningReingest::Field);
    let rendered = template.render_with(&prompt, &opts).expect("render");
    assert!(
        rendered.contains(&emission),
        "reconstruction drift.\n--- canonical emission ---\n{emission:?}\n\
         --- template re-render ---\n{rendered:?}"
    );
}
