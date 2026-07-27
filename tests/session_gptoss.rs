//! gpt-oss e2e (#30 Phase G): the Harmony dialect against the real
//! model. Mirrors the Phase E/F suites — forced/auto calls, thinking
//! under grammar, round-trip byte-stability, prefix-cache survival —
//! for the channel-structured Harmony format
//! (`<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{args}<|call|>`).
//!
//! All tests load `models/gpt-oss-20b-UD-Q8_K_XL.gguf` and are
//! `#[ignore]`d. Run with
//! `cargo test --features serde,cuda --test session_gptoss -- --ignored`.

use std::{borrow::Cow, num::NonZeroU32, path::PathBuf};

use drama_llama::{
    prompt::{ToolResult, ToolUse},
    Block, CallSyntax, Content, FromPath, Message, Prompt, RenderOptions, Role,
    Tool, ToolChoice,
};
use serde_json::json;

fn model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models/gpt-oss-20b-UD-Q8_K_XL.gguf")
}

/// Install the cache-stability template sidecar next to the model.
/// The same bytes are baked into the crate (`baked::GPTOSS`, #88) and
/// would apply without any sidecar; installing one anyway makes this
/// suite exercise rung 1 of the loading ladder over rung 2.
/// Idempotent; sourced from the shipped template.
fn install_template_sidecar() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("templates/gptoss-cache-stable.jinja");
    let sidecar = model_path().with_extension("template.jinja");
    std::fs::copy(&fixture, &sidecar).expect("install template sidecar");
}

fn load_session() -> drama_llama::LlamaCppSession {
    install_template_sidecar();
    drama_llama::LlamaCppSession::from_path(model_path())
        .expect("session load")
        .quiet()
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
    let session = load_session();
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
    let prompt =
        count_letters_prompt().max_tokens(NonZeroU32::new(1024).unwrap());
    let mut session = load_session();

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
    let mut prompt =
        count_letters_prompt().max_tokens(NonZeroU32::new(1024).unwrap());
    prompt.tool_choice = Some(ToolChoice::auto());
    let mut session = load_session();

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

    let prompt =
        count_letters_prompt().max_tokens(NonZeroU32::new(1024).unwrap());
    let mut session = load_session();
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

/// Announce-then-call (causality), DETERMINISTIC. The invariant worth
/// regression-testing is *ours*: that an announce-then-call assistant
/// turn — reasoning, then a commentary preamble, then the tool call —
/// renders, parses back in the same emission order, and round-trips
/// byte-for-byte (the cache-stability contract). It does NOT depend on
/// live generation: gpt-oss won't reliably emit the preamble without
/// model-specific nudging (it plans it in the analysis channel, then
/// skips to the call), and its output isn't reproducible across runs or
/// hardware (Metal especially). So we construct the turn, drive it
/// through the real gpt-oss template + dialect, and assert on our own
/// render/parse. Live "does Auto still call the tool" coverage lives in
/// `auto_tool_choice_parses_native_call`.
#[test]
#[ignore = "requires gpt-oss model (for its real template + dialect)"]
fn announce_then_call_round_trips_in_emission_order() {
    use drama_llama::AssistantMessage;

    let session = load_session();
    let prompt = count_letters_prompt();

    // Shared render options; only the generation-prompt flag differs
    // between the prompt prefix and the completed turn.
    let opts = |gen_prompt: bool| {
        RenderOptions::default()
            .with_generation_prompt(gen_prompt)
            .with_extra("preserve_thinking", true)
            .with_thought_reingest(session.dialect().reasoning.reingest)
    };

    // An announce-then-call turn: analysis (Thought), a commentary
    // preamble (Text), then the call (ToolUse) — in that order.
    let turn = vec![
        Block::Thought {
            thought: "The user wants the count of 'r' in 'strawberry'. \
                      I'll announce what I'm doing, then call \
                      count_letters."
                .into(),
            signature: Cow::Borrowed(""),
        },
        "I'll count that now.".into(),
        Block::ToolUse {
            call: ToolUse {
                id: Cow::Borrowed("call_0_count_letters"),
                name: Cow::Borrowed("count_letters"),
                input: json!({"letter": "r", "string": "strawberry"}),
                cache_control: None,
                caller: None,
            },
        },
    ];

    // Render the turn through the real template, then isolate the
    // emission (the suffix past the generation-prompt prefix). This is
    // the canonical byte form our pipeline produces.
    let prefix = session
        .template()
        .render_with(&prompt, &opts(true))
        .expect("render prompt prefix");
    let render_turn = |turn: Vec<Block>| -> String {
        let assistant: AssistantMessage = turn.into_iter().collect();
        let mut with_turn = prompt.clone();
        with_turn.messages.push(assistant.into());
        let rendered = session
            .template()
            .render_with(&with_turn, &opts(false))
            .expect("render turn");
        rendered
            .strip_prefix(&prefix)
            .expect("turn must extend the prompt prefix exactly")
            .to_string()
    };
    let emission = render_turn(turn);
    println!("=== emission ===\n{emission:?}\n===");

    // Parse the emission back: the commentary preamble (Text) must
    // survive and still precede the tool call in emission order.
    let tool_refs: Vec<&Tool> = prompt
        .tools
        .iter()
        .flatten()
        .filter_map(|def| def.as_method())
        .collect();
    let blocks = drama_llama::dialect::parse_text(
        session.dialect(),
        &tool_refs,
        &emission,
        false, // gpt-oss never pre-opens reasoning in the prompt tail
        drama_llama::dialect::Leniency::Final,
    )
    .blocks;
    println!("=== parsed blocks ===\n{blocks:#?}\n===");
    let first_text =
        blocks.iter().position(|b| matches!(b, Block::Text { .. }));
    let first_call = blocks
        .iter()
        .position(|b| matches!(b, Block::ToolUse { .. }))
        .expect("round-trip must preserve the tool call");
    assert!(
        first_text.is_some_and(|t| t < first_call),
        "commentary preamble must precede the call in emission order — \
         blocks: {blocks:#?}"
    );

    // Byte-stability: re-rendering the parsed blocks reproduces the
    // emission exactly (the prefix-cache contract).
    let emission2 = render_turn(blocks);
    assert_eq!(
        emission, emission2,
        "announce-then-call is not byte-stable across parse -> render"
    );
}

/// Tool turn 2: assistant call + tool result re-ingest (the sidecar's
/// forward-scan of `role: tool` messages), then free prose. The
/// answer should surface the result.
#[test]
#[ignore = "requires gpt-oss model"]
fn tool_result_turn_produces_prose_answer() {
    let call_id = "call_0_count_letters";
    let mut prompt =
        count_letters_prompt().max_tokens(NonZeroU32::new(1024).unwrap());
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

    let mut session = load_session();
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

    let mut prompt =
        count_letters_prompt().max_tokens(NonZeroU32::new(1024).unwrap());
    if let Some(tools) = prompt.tools.as_mut() {
        if let Some(def) = tools.first_mut() {
            if let Some(tool) = def.as_method_mut() {
                tool.cache_control = Some(
                    misanthropic::prompt::message::CacheControl::ephemeral(),
                );
            }
        }
    }
    let mut session = load_session().with_prefix_cache(true);

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
/// against the real vocab: `<|return|>` and `<|call|>` end generation,
/// while `<|end|>` (the in-stream channel separator) must NOT.
///
/// The trap this test exists to catch: `<|end|>` **is this vocab's
/// EOT**. libllama auto-detects EOT by token text and `"<|end|>"` is on
/// that list, so `special_eot_id` lands on it; the o200k_harmony
/// workaround then removes `<|end|>` from `special_eog_ids` — and
/// *doesn't* touch `special_eot_id`. Upstream stays consistent because
/// its generation loop only ever asks `llama_vocab_is_eog`. So must
/// ours: a stop set built as `{eos} ∪ {eot} ∪ extras` drags `<|end|>`
/// back in, and a Harmony turn then dies right after its analysis block
/// (unconstrained) or can never close the channel at all (under a tool
/// grammar, where the same set is masked while the grammar is
/// incomplete). Hence `Model::eog_tokens` is the *whole* set and the
/// single authority. CPU-only load: vocab introspection needs no GPU.
#[test]
#[ignore = "long running - requires gpt-oss model"]
fn gptoss_eog_token_set() {
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

    let call = by_piece("<|call|>");
    let end = by_piece("<|end|>");
    let ret = by_piece("<|return|>");

    // EOS is the final-message stop. Deterministic: read from KV
    // metadata, not guessed.
    assert_eq!(piece_of(model.eos()), "<|return|>");

    // The eog set is the contract, so it is asserted FIRST — the eot
    // pin below is the flakier claim and must not be able to abort the
    // run before these have executed. (It did exactly that on Linux:
    // the carve-out went unverified there until this reorder.)
    let eog = model.eog_tokens();
    let pieces = || eog.iter().map(|&t| piece_of(t)).collect::<Vec<_>>();
    assert!(
        eog.contains(&call),
        "<|call|> must stop generation (tool-call turn exit); eog = {:?}",
        pieces()
    );
    assert!(
        eog.contains(&ret),
        "<|return|> must stop generation (final-message exit); eog = {:?}",
        pieces()
    );
    assert!(
        !eog.contains(&end),
        "<|end|> is the in-stream channel separator and must stay \
         generatable — it IS this vocab's eot(), which is exactly why \
         the stop set must be eog_tokens() and never a union with \
         eot(); eog = {:?}",
        pieces()
    );

    // Now the eot pin — deliberately weaker than it looks, because
    // upstream does not actually guarantee a value here.
    //
    // `llama_vocab::impl::load` auto-detects EOT by iterating
    // `token_to_id` and taking the FIRST entry whose text is on a
    // candidate list — and `token_to_id` is a
    // `std::unordered_map<std::string, llama_token>`
    // (llama-vocab.cpp). Iteration order of an unordered container is
    // unspecified, so for any vocab holding two or more candidates the
    // winner is whatever the standard library happened to hash first.
    // This vocab holds both `<|end|>` and `<|endoftext|>`, and the two
    // platforms disagree: libc++ (macOS) yields `<|end|>`, libstdc++
    // (Linux/CI) yields `<|endoftext|>`, on byte-identical weights —
    // sha256-verified, after the difference was first misdiagnosed as
    // two different downloads.
    //
    // So this asserts only what upstream can actually deliver: that
    // auto-detection still fires and lands on a plausible turn-ender.
    // The change that would genuinely hurt — upstream giving up and
    // leaving eot NULL, or picking something absurd — still trips it.
    // What it must NOT do is pin one platform's hash order and call
    // that a contract.
    //
    // None of this reaches generation, and that is the point of
    // `.claude/memory/eog_is_not_eos_plus_eot.md`: we stop on
    // `eog_tokens()`, never on a set built from `eot()`. The label
    // wobbles across platforms; the predicate does not.
    let eot = piece_of(model.eot());
    assert!(
        eot == "<|end|>" || eot == "<|endoftext|>",
        "eot auto-detection landed somewhere unexpected: {eot:?}. \
         Upstream picks the first candidate out of an unordered_map, so \
         either value is legal here — but a third one means the \
         candidate list or the detection changed, and the carve-out \
         above needs re-reading. Never assume eot == stop."
    );
}

/// Prefix cache across a FINAL turn — the `<|return|>` → `<|end|>`
/// re-ingest rewrite (upstream #15417). The model ends its answer
/// with the EOG `<|return|>`, but every later render of that turn
/// closes with `<|end|>`; recording the sampled stop as the tip
/// prediction made the next call's LCP die exactly at the tip, and
/// since restore targets are only checkpointed positions, reuse fell
/// back to the last explicit `cache_control` breakpoint — here there
/// are NONE, so pre-fix this test reads zero cache. The tip now
/// records the CANONICAL close token from the byte-stable re-render,
/// so turn 2 splices at the tip and reuses the whole first turn.
#[test]
#[ignore = "requires gpt-oss model"]
fn prefix_cache_survives_final_turn() {
    use drama_llama::AssistantMessage;

    let mut session = load_session().with_prefix_cache(true);
    let mut prompt = Prompt {
        system: Some(Content::text("You are a helpful assistant.")),
        messages: vec![Message {
            role: Role::User,
            content: Content::text(
                "What is the capital of France? Answer briefly.",
            ),
        }],
        ..Default::default()
    }
    .max_tokens(NonZeroU32::new(1024).unwrap());

    let blocks = session.complete_blocks(&prompt).expect("turn 1");
    assert!(
        blocks.iter().any(|b| matches!(b, Block::Text { .. })),
        "turn 1 must produce a final answer: {blocks:#?}"
    );
    let turn1_prompt = session.last_usage().input_tokens;

    let assistant: AssistantMessage = blocks.into_iter().collect();
    prompt.messages.push(assistant.into());
    prompt.messages.push(Message {
        role: Role::User,
        content: Content::text("And of Italy?"),
    });

    let out = session.complete_text(&prompt).expect("turn 2");
    println!("=== turn 2 ===\n{out}\n===");
    let read = session
        .last_usage()
        .cache_read_input_tokens
        .unwrap_or_default();
    // The tip is the ONLY eligible anchor (no cache_control set), and
    // it sits past the whole first turn (prompt + generation). Zero
    // or prompt-sized reuse means the canonical-close substitution
    // regressed and the <|return|>/<|end|> mismatch disqualified it.
    assert!(
        read as u64 > turn1_prompt as u64,
        "turn 2 must splice at the tip (past the whole first turn); \
         cache_read={read}, turn-1 prompt={turn1_prompt} (usage: {:?})",
        session.last_usage()
    );
}
