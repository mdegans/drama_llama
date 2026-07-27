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

/// The payloads every fixture is swept with. `clean` is the original
/// harness payload; `adversarial` exists because #85 hid behind clean
/// ASCII for two weeks — it carries the characters serializers
/// disagree about (apostrophe, `&`, `<`, `>`, a double quote, a
/// multi-byte arrow, an embedded newline) plus a `", "` inside the
/// string, which is byte-identical to the JSON envelope's field
/// separator and probes parser greediness.
fn payloads() -> [(&'static str, serde_json::Value); 2] {
    [
        ("clean", json!({"city": "Paris", "days": 3})),
        (
            "adversarial",
            json!({
                "city": "José's \"B&B\", floor <2> → east\nannex",
                "days": 3,
            }),
        ),
    ]
}

/// The harness: analyze the template, produce a canonical emission,
/// parse it back, feed the parsed calls through the real template,
/// and assert the emission bytes survive intact in the re-render.
fn assert_reconstruction(fixture: &str, bos: &str, eos: &str) {
    let source = fixture_source(fixture);
    assert_reconstruction_source(fixture, &source, bos, eos);
}

/// Source-taking variant for fixtures that don't live under
/// `tests/fixtures/templates/` (cogito). Sweeps every payload.
fn assert_reconstruction_source(
    fixture: &str,
    source: &str,
    bos: &str,
    eos: &str,
) {
    let syntax: CallSyntax =
        analyze_template(source, bos, eos).expect("analyze");
    let tool = test_tool();

    for (payload, input) in payloads() {
        let fixture = &format!("{fixture}/{payload}");
        assert_call_round_trips(
            fixture, source, &syntax, &tool, &input, bos, eos,
        );
    }
}

/// One payload through one template: emission → parse → re-render.
fn assert_call_round_trips(
    fixture: &str,
    source: &str,
    syntax: &CallSyntax,
    tool: &Tool,
    input: &serde_json::Value,
    bos: &str,
    eos: &str,
) {
    let emission = render_reference(syntax, &[("get_weather", input)])
        .expect("representable");

    // Emission → blocks.
    let parsed = parse_text(syntax, &[tool], &emission, false, Leniency::Final);
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
    assert_eq!(&calls[0].input, input, "{fixture}");
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
    let template = ChatTemplate::from_source(
        source.to_string(),
        bos.to_string(),
        eos.to_string(),
    )
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

/// Cogito (#85's template): `JsonNative` with hardcoded spaced
/// envelope literals (`{"name": "` / `", "arguments": `) around a
/// `tojson` argument interior. Lives at the fixtures root because
/// `template_rendering.rs` pins byte-exact renders against the same
/// file; byte-identical to the 32B GGUF's embedded template.
#[test]
fn reconstruct_cogito() {
    assert_reconstruction_source(
        "cogito_14b",
        &cogito_source(),
        "",
        "<|im_end|>",
    );
}

fn cogito_source() -> String {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/cogito_14b_template.jinja"
    );
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

/// The #85 cache property itself, pinned FFI-free against the STOCK
/// cogito template: the generation-prompt render is a byte PREFIX of
/// the follow-up render, and the delta begins with the canonical
/// emission — so the KV laid down during generation stays reusable
/// and the auto-tip survives the turn. Uses the adversarial payload
/// deliberately: before `368d11e` (tojson HTML-escaping) and
/// `86c9fe4` (canonical `ws`) this failed on exactly such bytes while
/// clean-ASCII payloads passed.
#[test]
fn cogito_prefix_continuity() {
    let source = cogito_source();
    let syntax = analyze_template(&source, "", "<|im_end|>").expect("analyze");
    let tool = test_tool();
    let (_, input) = &payloads()[1];
    let calls_ref = render_reference(&syntax, &[("get_weather", input)])
        .expect("representable");
    let template = ChatTemplate::from_source(
        source,
        String::new(),
        "<|im_end|>".to_string(),
    )
    .expect("template compiles");
    // enable_thinking=true rewrites the FRONT of cogito's prompt, so
    // it can't change between renders here; #86 tracks the partial-
    // render half of that. Continuity is pinned in non-thinking mode.
    let opts = |gen: bool| {
        RenderOptions::default()
            .with_generation_prompt(gen)
            .with_extra("enable_thinking", false)
    };
    let base = Prompt {
        messages: vec![Message {
            role: Role::User,
            content: Content::text("What's the weather in Paris?"),
        }],
        tools: Some(vec![tool.clone().into()]),
        ..Default::default()
    };

    // Case 1: a bare tool-call turn extends the generation prompt,
    // starting with the canonical emission and closing with eos.
    let p = template.render_with(&base, &opts(true)).expect("render");
    let mut with_turn = base.clone();
    with_turn.messages.push(Message {
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
    });
    let f = template
        .render_with(&with_turn, &opts(false))
        .expect("render");
    let suffix = f.strip_prefix(&p).unwrap_or_else(|| {
        panic!(
            "tool-call turn must extend the generation prompt.\n\
             --- gen ---\n{p:?}\n--- follow-up ---\n{f:?}"
        )
    });
    assert!(
        suffix.starts_with(&calls_ref),
        "emission bytes must lead the turn delta.\n\
         --- want ---\n{calls_ref:?}\n--- got ---\n{suffix:?}"
    );
    assert!(
        suffix[calls_ref.len()..].starts_with("<|im_end|>"),
        "canonical close must follow the calls.\n{suffix:?}"
    );

    // Case 2: aging. The tool response and the next generation prompt
    // must keep the whole prior render as a byte prefix — this is the
    // LCP walk that has to cross the tool turn to reach the tip.
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
    let f_aged = template.render_with(&aged, &opts(true)).expect("render");
    assert!(
        f_aged.starts_with(&f),
        "aged render must extend the prior render byte-for-byte.\n\
         --- prior ---\n{f:?}\n--- aged ---\n{f_aged:?}"
    );
}

/// Prefix continuity for the Qwen3.6 GGUF template — the stock
/// template of the model CI's runner actually holds, and the origin
/// of the XML dialect the whole tool-dialects arc was built for.
/// Same property as `cogito_prefix_continuity`; non-thinking mode
/// (aged-thinking continuity is a Phase 4 owned-template decision).
#[test]
fn qwen36_prefix_continuity() {
    let source = fixture_source("qwen3.6-gguf.jinja");
    let syntax = analyze_template(&source, "", "<|im_end|>").expect("analyze");
    let tool = test_tool();
    let (_, input) = &payloads()[1];
    let calls_ref = render_reference(&syntax, &[("get_weather", input)])
        .expect("representable");
    let template = ChatTemplate::from_source(
        source,
        String::new(),
        "<|im_end|>".to_string(),
    )
    .expect("template compiles");
    let opts = |gen: bool| {
        RenderOptions::default()
            .with_generation_prompt(gen)
            .with_extra("enable_thinking", false)
    };
    let base = Prompt {
        messages: vec![Message {
            role: Role::User,
            content: Content::text("What's the weather in Paris?"),
        }],
        tools: Some(vec![tool.clone().into()]),
        ..Default::default()
    };

    // A bare tool-call turn extends the generation prompt, leading
    // with the canonical emission and closing with eos.
    let p = template.render_with(&base, &opts(true)).expect("render");
    let mut with_turn = base.clone();
    with_turn.messages.push(Message {
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
    });
    let f = template
        .render_with(&with_turn, &opts(false))
        .expect("render");
    let suffix = f.strip_prefix(&p).unwrap_or_else(|| {
        panic!(
            "tool-call turn must extend the generation prompt.\n\
             --- gen ---\n{p:?}\n--- follow-up ---\n{f:?}"
        )
    });
    assert!(
        suffix.starts_with(&calls_ref),
        "emission bytes must lead the turn delta.\n\
         --- want ---\n{calls_ref:?}\n--- got ---\n{suffix:?}"
    );
    assert!(
        suffix[calls_ref.len()..].starts_with("<|im_end|>"),
        "canonical close must follow the calls.\n{suffix:?}"
    );

    // Aging: tool response plus the next generation prompt keeps the
    // prior render as a byte prefix.
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
    let f_aged = template.render_with(&aged, &opts(true)).expect("render");
    assert!(
        f_aged.starts_with(&f),
        "aged render must extend the prior render byte-for-byte.\n\
         --- prior ---\n{f:?}\n--- aged ---\n{f_aged:?}"
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

/// gpt-oss, cache-stable sidecar: the canonical channel-header
/// emission must reappear byte-for-byte (the generic harness).
#[test]
fn reconstruct_gptoss_cache_stable() {
    assert_reconstruction(
        "gptoss-cache-stable.jinja",
        "<|startoftext|>",
        "<|return|>",
    );
}

/// gpt-oss, stock (Unsloth GGUF) template: re-renders the ROLE-header
/// re-ingest shape (` to=functions.NAME<|channel|>commentary json`)
/// rather than the trained channel-header emission shape, so byte
/// stability requires the sidecar. Pin the *semantic* round-trip:
/// emission → blocks → stock re-render → parse again → same call.
#[test]
fn reconstruct_gptoss_gguf_semantic() {
    let source = fixture_source("gptoss-gguf.jinja");
    let syntax: CallSyntax =
        analyze_template(&source, "<|startoftext|>", "<|return|>")
            .expect("analyze");
    assert_eq!(syntax, CallSyntax::gpt_oss(), "sniff must fire");
    let tool = test_tool();
    let input = json!({"city": "Paris", "days": 3});

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
    let template = ChatTemplate::from_source(
        source,
        "<|startoftext|>".to_string(),
        "<|return|>".to_string(),
    )
    .expect("template compiles");
    let opts = RenderOptions::default().with_generation_prompt(false);
    let rendered = template.render_with(&prompt, &opts).expect("render");

    // Locate the re-rendered assistant call turn and parse it back.
    let call_at = rendered
        .find("<|start|>assistant to=functions.")
        .unwrap_or_else(|| panic!("no role-header call in:\n{rendered:?}"));
    let parsed = parse_text(
        &syntax,
        &[&tool],
        &rendered[call_at..],
        false,
        Leniency::Final,
    );
    let calls: Vec<&ToolUse> = parsed
        .blocks
        .iter()
        .filter_map(|b| match b {
            Block::ToolUse { call } => Some(call),
            _ => None,
        })
        .collect();
    assert_eq!(calls.len(), 1, "{:#?}", parsed.blocks);
    assert_eq!(calls[0].name.as_ref(), "get_weather");
    assert_eq!(calls[0].input, input, "{:#?}", parsed.blocks);
}

/// The cache property itself, pinned FFI-free against the gpt-oss
/// sidecar: the generation-prompt render is a byte PREFIX of the
/// follow-up render, through thinking, announce-then-call, tool
/// responses, aged reasoning, and a final turn. The stock template
/// breaks every one of these (CoT dropped on past turns, role-header
/// call shape, first-call-only).
#[test]
fn gptoss_cache_stable_prefix_continuity() {
    use drama_llama::dialect::ReasoningReingest;

    let source = fixture_source("gptoss-cache-stable.jinja");
    let syntax = CallSyntax::gpt_oss();
    let tool = test_tool();
    let input = json!({"city": "Paris", "days": 3});
    let calls_ref = render_reference(&syntax, &[("get_weather", &input)])
        .expect("representable");
    let template = ChatTemplate::from_source(
        source,
        "<|startoftext|>".to_string(),
        "<|return|>".to_string(),
    )
    .expect("template compiles");

    let opts = |gen: bool| {
        RenderOptions::default()
            .with_generation_prompt(gen)
            .with_extra("preserve_thinking", true)
            .with_thought_reingest(ReasoningReingest::Thinking)
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
    let thought = "user wants weather; get_weather fits";
    let announce = "Let me check that for you.\n\n";

    // Generation prompt tail: bare `<|start|>assistant`, the block
    // opener every emission continues from.
    let p = template.render_with(&base, &opts(true)).expect("render");
    assert!(
        p.ends_with("<|start|>assistant"),
        "generation prompt tail changed: {p:?}"
    );

    // Announce-then-call with thinking: the emission byte order —
    // analysis block, commentary preamble (VERBATIM, trailing
    // whitespace intact), canonical call — must re-render as a strict
    // extension of the generation prompt.
    let mut with_turn = base.clone();
    with_turn.messages.push(Message {
        role: Role::Assistant,
        content: Content(vec![
            Block::Thought {
                thought: Cow::Borrowed(thought),
                signature: Cow::Borrowed(""),
            },
            Block::Text {
                text: Cow::Borrowed(announce),
                citations: None,
                cache_control: None,
            },
            call_block.clone(),
        ]),
    });
    let f = template
        .render_with(&with_turn, &opts(false))
        .expect("render");
    let suffix = f.strip_prefix(&p).unwrap_or_else(|| {
        panic!(
            "follow-up must extend the generation prompt.\n--- gen ---\n\
             {p:?}\n--- follow-up ---\n{f:?}"
        )
    });
    let emission = format!(
        "<|channel|>analysis<|message|>{thought}<|end|>\
         <|start|>assistant<|channel|>commentary<|message|>{announce}<|end|>\
         <|start|>assistant{calls_ref}<|call|>"
    );
    assert!(
        suffix.starts_with(&emission),
        "emission bytes must re-render verbatim.\n--- want ---\n\
         {emission:?}\n--- got ---\n{suffix:?}"
    );

    // Tool response renders by forward-scan with the id-resolved
    // name, and the aged turn keeps its thought bytes
    // (preserve_thinking).
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
    let f_aged = template.render_with(&aged, &opts(false)).expect("render");
    let with_response = format!(
        "{emission}<|start|>functions.get_weather to=assistant\
         <|channel|>commentary<|message|>22C, sunny<|end|>"
    );
    assert!(
        f_aged.contains(&with_response),
        "aged call turn + response must render continuously.\n\
         --- want contained ---\n{with_response:?}\n--- render ---\n\
         {f_aged:?}"
    );

    // Final turn: thinking then content. The re-render closes with
    // <|end|> where the emission ended with the (uncommitted)
    // <|return|> EOG — the emission TEXT is still a strict prefix.
    let mut final_turn = base.clone();
    final_turn.messages.push(Message {
        role: Role::Assistant,
        content: Content(vec![
            Block::Thought {
                thought: Cow::Borrowed(thought),
                signature: Cow::Borrowed(""),
            },
            Block::Text {
                text: Cow::Borrowed("It's 22C and sunny."),
                citations: None,
                cache_control: None,
            },
        ]),
    });
    let f_final = template
        .render_with(&final_turn, &opts(false))
        .expect("render");
    let suffix = f_final.strip_prefix(&p).unwrap_or_else(|| {
        panic!(
            "final turn must extend the generation prompt.\n--- gen ---\n\
             {p:?}\n--- follow-up ---\n{f_final:?}"
        )
    });
    let emission = format!(
        "<|channel|>analysis<|message|>{thought}<|end|>\
         <|start|>assistant<|channel|>final<|message|>It's 22C and sunny."
    );
    assert!(
        suffix.starts_with(&emission),
        "final-turn emission must be a byte prefix of the re-render.\n\
         --- want ---\n{emission:?}\n--- got ---\n{suffix:?}"
    );
    assert!(
        suffix[emission.len()..].starts_with("<|end|>"),
        "re-ingest close must be <|end|> (issue #15417 rewrite).\n{suffix:?}"
    );
}

#[test]
fn reconstruct_gemma4_upstream() {
    assert_reconstruction("google-gemma-4-31B-it.jinja", "<bos>", "<turn|>");
}

/// Gemma 4, the full-fidelity version: thought routed through the
/// `reasoning` field (`ReasoningReingest::Field`), parallel calls,
/// and the value-type corners probed against minijinja (null renders
/// `none`, floats in ryu-shortest form, nested dicts explicitly
/// re-sorted to match the template's `dictsort`).
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

// ===========================================================================
// #60 — schema declaration order
// ===========================================================================

/// A tool whose field declaration order (`zulu`, `alpha`, `mike`)
/// disagrees with alphabetical, with an optional (`alpha`)
/// interleaved between the two required fields.
fn zam_tool() -> (Tool, serde_json::Value) {
    let tool = Tool::builder("plan_route")
        .description("Plan a route.")
        .schema(json!({
            "type": "object",
            "properties": {
                "zulu": {"type": "string"},
                "alpha": {"type": "string"},
                "mike": {"type": "integer"},
            },
            "required": ["zulu", "mike"],
        }))
        .build()
        .expect("valid test tool");
    let input = json!({"zulu": "first", "alpha": "between", "mike": 3});
    (tool, input)
}

/// Positions of the three arg names in `text`, for order assertions.
fn zam_positions(text: &str, ctx: &str) -> (usize, usize, usize) {
    let find = |k: &str| {
        text.find(k)
            .unwrap_or_else(|| panic!("{ctx}: `{k}` missing:\n{text}"))
    };
    (find("zulu"), find("alpha"), find("mike"))
}

/// #60, the point of `preserve_order`: for the tojson/serde-driven
/// families, args must flow in schema DECLARATION order — grammar,
/// canonical emission, parse, and template re-render all agreeing —
/// even when declaration order contradicts alphabetical. If this
/// trips, `preserve_order` regressed somewhere in the chain and the
/// prefix cache is next.
fn assert_declaration_order(fixture: &str, bos: &str, eos: &str) {
    use drama_llama::dialect::{grammar_source, EmitOptions};

    let source = fixture_source(fixture);
    let syntax: CallSyntax =
        analyze_template(&source, bos, eos).expect("analyze");
    let (tool, input) = zam_tool();

    // Grammar constrains generation to declaration order.
    let grammar = grammar_source(&syntax, &[&tool], &EmitOptions::default())
        .expect("grammar");
    let (z, a, m) = zam_positions(&grammar, fixture);
    assert!(
        z < a && a < m,
        "{fixture}: grammar not in declaration order:\n{grammar}"
    );

    // Canonical emission renders the input Map's declaration order.
    let emission = render_reference(&syntax, &[("plan_route", &input)])
        .expect("representable");
    let (z, a, m) = zam_positions(&emission, fixture);
    assert!(
        z < a && a < m,
        "{fixture}: emission not in declaration order: {emission:?}"
    );

    // Emission → blocks, byte-order preserved through the parse.
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
    assert_eq!(calls.len(), 1, "{fixture}: {:#?}", parsed.blocks);
    assert_eq!(calls[0].input, input, "{fixture}");
    // Map equality ignores order — pin the serialized bytes too.
    assert_eq!(
        serde_json::to_string(&calls[0].input).unwrap(),
        serde_json::to_string(&input).unwrap(),
        "{fixture}: parse re-ordered the args"
    );

    // Blocks → template re-render: emission bytes survive verbatim.
    let prompt = Prompt {
        messages: vec![
            Message {
                role: Role::User,
                content: Content::text("Plan the route."),
            },
            Message {
                role: Role::Assistant,
                content: Content(vec![Block::ToolUse {
                    call: ToolUse {
                        id: Cow::Borrowed("call00001"),
                        name: Cow::Borrowed("plan_route"),
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
    assert!(
        rendered.contains(&emission),
        "{fixture}: reconstruction drift.\n--- canonical emission ---\n\
         {emission:?}\n--- template re-render ---\n{rendered:?}"
    );
}

#[test]
fn declaration_order_qwen3_coder() {
    assert_declaration_order("Qwen3-Coder.jinja", "", "<|im_end|>");
}

#[test]
fn declaration_order_hermes3() {
    assert_declaration_order(
        "NousResearch-Hermes-3-Llama-3.1-8B-tool_use.jinja",
        "",
        "<|im_end|>",
    );
}

#[test]
fn declaration_order_llama31() {
    assert_declaration_order(
        "meta-llama-Llama-3.1-8B-Instruct.jinja",
        "<|begin_of_text|>",
        "<|eot_id|>",
    );
}

#[test]
fn declaration_order_gptoss_cache_stable() {
    assert_declaration_order(
        "gptoss-cache-stable.jinja",
        "<|startoftext|>",
        "<|return|>",
    );
}

/// The dict family is the deliberate exception to #60: Gemma 4's
/// model-shipped templates pipe arguments through `| dictsort`, which
/// alphabetizes regardless of `preserve_order` — so grammar, encoder,
/// and re-render stay explicitly alphabetical there.
#[test]
fn dict_family_stays_dictsort_alphabetical() {
    use drama_llama::dialect::{grammar_source, EmitOptions};

    let source = fixture_source("gemma4-gguf.jinja");
    let syntax: CallSyntax =
        analyze_template(&source, "<bos>", "<turn|>").expect("analyze");
    assert_eq!(syntax, CallSyntax::gemma4(), "sniff patch must fire");
    let (tool, input) = zam_tool();

    let grammar = grammar_source(&syntax, &[&tool], &EmitOptions::default())
        .expect("grammar");
    let (z, a, m) = zam_positions(&grammar, "gemma4 grammar");
    assert!(
        a < m && m < z,
        "gemma4: grammar not alphabetical:\n{grammar}"
    );

    let emission = render_reference(&syntax, &[("plan_route", &input)])
        .expect("representable");
    let (z, a, m) = zam_positions(&emission, "gemma4 emission");
    assert!(
        a < m && m < z,
        "gemma4: emission not alphabetical: {emission:?}"
    );

    // And the template agrees — dictsort re-renders alphabetically.
    let parsed =
        parse_text(&syntax, &[&tool], &emission, false, Leniency::Final);
    assert_eq!(parsed.status, ParseStatus::Complete, "{:#?}", parsed.blocks);
    let prompt = Prompt {
        messages: vec![
            Message {
                role: Role::User,
                content: Content::text("Plan the route."),
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
        .with_extra("enable_thinking", true);
    let rendered = template.render_with(&prompt, &opts).expect("render");
    assert!(
        rendered.contains(&emission),
        "gemma4: reconstruction drift.\n--- canonical emission ---\n\
         {emission:?}\n--- template re-render ---\n{rendered:?}"
    );
}
