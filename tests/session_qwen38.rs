//! Qwen3.8 e2e: the #96 tip suite plus the #112 thinking-turn
//! round-trip, against Agora's production model.
//!
//! #112: on Qwen3.8 every turn logged "emission does not re-render
//! byte-stable", so the tip never anchored and each call re-prefilled
//! the one before it. The cause was the thought re-ingest convention:
//! 3.8's template dropped 3.6's `content.split('</think>')` and reads
//! reasoning from `reasoning_content` alone, while the analyzer
//! defaulted every `<think>` dialect to `InlineThink`. The thought
//! re-rendered as *content* behind an empty `<think>\n\n</think>`.
//! The analyzer now measures the convention (pinned model-free in
//! `dialect_analyzer.rs::qwen38_gguf_field_reasoning` and
//! `dialect_roundtrip.rs::qwen38_thinking_turn_round_trips`); this
//! suite is the on-device witness.
//!
//! **Thinking on matters.** The shared #96 scenarios send
//! `thinking: None`, which renders `enable_thinking = false` — no
//! reasoning turn is ever re-ingested, so they were green on 3.8
//! throughout #112. The thinking variants below are the ones that
//! would have caught it.
//!
//! **Stock template (rung 3).** Qwen3.8 has no baked replacement: its
//! embedded template is used as-is, and it is cache-stable once the
//! reingest is measured. A stray sidecar would silently swap in other
//! bytes, so it fails the suite loudly, as in the sibling suites.
//!
//! All tests load `models/Qwen3.8-27B-UD-Q8_K_XL.gguf` (override with
//! `$DRAMA_LLAMA_QWEN38_MODEL`) and are `#[ignore]`d. Absent that
//! model they skip loudly rather than substituting `model.gguf`.

#![cfg(feature = "llama-cpp")]

mod common;

use std::{num::NonZeroU32, path::PathBuf};

use drama_llama::{
    dialect::ReasoningReingest, AssistantMessage, Block, Content, FromPath,
    Prompt, RenderOptions, Tool, ToolChoice,
};
use misanthropic::prompt::thinking::Thinking;
use serde_json::json;

/// Resolve the Qwen3.8 GGUF: `$DRAMA_LLAMA_QWEN38_MODEL` if set and
/// present, else the conventional path under `models/`. `None` means
/// skip — never substitute `model.gguf`.
fn model_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("DRAMA_LLAMA_QWEN38_MODEL") {
        let p = PathBuf::from(p);
        return p.exists().then_some(p);
    }
    let conventional = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models/Qwen3.8-27B-UD-Q8_K_XL.gguf");
    conventional.exists().then_some(conventional)
}

/// Seeded via `common::test_seed()`: random by default, printed on
/// failure so the trajectory is replayable with
/// `DRAMA_LLAMA_TEST_SEED=<n>`. The tip scenarios force their own
/// determinism; the round-trip tests below use [`deterministic`].
fn load_session() -> Option<drama_llama::LlamaCppSession> {
    let path = model_path()?;
    let sidecar = path.with_extension("template.jinja");
    assert!(
        !sidecar.exists(),
        "a template sidecar exists at {}, which would replace the stock \
         template this suite pins. Delete it and re-run.",
        sidecar.display()
    );
    Some(
        drama_llama::LlamaCppSession::from_path_with(
            path,
            drama_llama::LlamaCppOptions::default().with_n_ctx(16384),
        )
        .expect("session load")
        .quiet()
        .with_prefix_cache(true)
        .with_seed(Some(common::test_seed())),
    )
}

/// Greedy, penalty off — the round-trip tests pin cache mechanics, not
/// sampling. Sampled with the penalty on, 3.8's thought can degrade
/// and run past `max_tokens` without ever closing (seen in the full
/// tier: 2048 tokens of rambling, no `</think>`), and the budget
/// cannot stop it (#114).
fn deterministic(
    session: drama_llama::LlamaCppSession,
) -> drama_llama::LlamaCppSession {
    session
        .without_repetition()
        .with_sampling([drama_llama::SamplingMode::Greedy])
}

macro_rules! session_or_skip {
    () => {
        match load_session() {
            Some(s) => s,
            None => {
                eprintln!(
                    "SKIP: needs a Qwen3.8 model (DRAMA_LLAMA_QWEN38_MODEL \
                     or models/Qwen3.8-27B-UD-Q8_K_XL.gguf)"
                );
                return;
            }
        }
    };
}

/// The session measured the convention the template actually honours.
/// Cheap, and first to go red if a re-quant ships a template that
/// restores the content split (then `InlineThink` is right again and
/// this pin should move, not the probe).
#[test]
#[ignore = "requires Qwen3.8 model"]
fn dialect_reingests_reasoning_through_the_field() {
    let session = session_or_skip!();
    assert_eq!(
        session.dialect().reasoning.reingest,
        ReasoningReingest::Field,
        "{:#?}",
        session.dialect()
    );
}

/// #112 on device: a thinking, tool-calling turn's raw emission is a
/// byte prefix of its canonical re-render — the session's
/// canonicalization gate, spelled out.
#[test]
#[ignore = "requires Qwen3.8 model"]
fn thinking_tool_turn_round_trips_issue_112() {
    let tool = Tool::builder("get_content")
        .description("Read a post by id.")
        .schema(json!({
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        }))
        .build()
        .expect("valid tool");
    let prompt = Prompt {
        system: Some(Content::text(
            "You are a forum agent. Use tools to read posts.",
        )),
        messages: vec![drama_llama::Message {
            role: drama_llama::Role::User,
            content: Content::text(
                "Read post 05676b9d-8aa7-430e-9138-444080e34065.",
            ),
        }],
        tools: Some(vec![tool.clone().into()]),
        tool_choice: Some(ToolChoice::method("get_content")),
        thinking: Some(Thinking::Enabled {
            budget_tokens: NonZeroU32::new(512).unwrap(),
            display: None,
        }),
        max_tokens: NonZeroU32::new(2048).unwrap(),
        ..Default::default()
    };
    let mut session = deterministic(session_or_skip!());
    let reingest = session.dialect().reasoning.reingest;
    let opts = |gen: bool| {
        RenderOptions::default()
            .with_generation_prompt(gen)
            .with_extra("preserve_thinking", true)
            .with_thought_reingest(reingest)
    };
    let gen = session
        .template()
        .render_with(&prompt, &opts(true))
        .expect("render");
    assert!(
        gen.ends_with("<think>\n"),
        "thinking must pre-open: {gen:?}"
    );

    let raw = session.complete_text(&prompt).expect("complete_text");
    println!("=== raw emission ===\n{raw:?}\n===");
    let parsed = drama_llama::dialect::parse_text(
        session.dialect(),
        &[&tool],
        &raw,
        true,
        drama_llama::dialect::Leniency::Final,
    )
    .blocks;
    assert!(
        parsed.iter().any(|b| matches!(b, Block::Thought { .. })),
        "a thinking turn must parse a Thought: {parsed:?}"
    );
    assert!(
        parsed.iter().any(|b| matches!(b, Block::ToolUse { .. })),
        "the forced call must parse: {parsed:?}"
    );

    let mut follow_up = prompt.clone();
    let assistant: AssistantMessage = parsed.into_iter().collect();
    follow_up.messages.push(assistant.into());
    follow_up.tool_choice = None;
    let rendered = session
        .template()
        .render_with(&follow_up, &opts(false))
        .expect("render");
    let suffix = rendered.strip_prefix(&gen).unwrap_or_else(|| {
        panic!("the turn must extend the generation prompt.\n{rendered:?}")
    });
    assert!(
        suffix.starts_with(&raw),
        "emission is not a byte prefix of the canonical re-render.\n\
         --- emission ---\n{raw:?}\n--- re-rendered suffix ---\n{suffix:?}"
    );
}

/// #112, structured output: a thinking turn under an output_config
/// grammar must round-trip byte-exact. This was the one shape the
/// reingest fix alone left red (4 of the 41 logged Agora emissions —
/// the JSON memory updates): the post-`</think>` gap was the grammar's
/// single permissive byte, and the template re-renders `\n\n`.
#[test]
#[ignore = "requires Qwen3.8 model"]
fn thinking_structured_output_round_trips_issue_112() {
    let prompt = Prompt {
        system: Some(Content::text("You keep terse notes as JSON.")),
        messages: vec![drama_llama::Message {
            role: drama_llama::Role::User,
            content: Content::text(
                "Record one note: the Council meets on the 26th.",
            ),
        }],
        thinking: Some(Thinking::Enabled {
            budget_tokens: NonZeroU32::new(512).unwrap(),
            display: None,
        }),
        max_tokens: NonZeroU32::new(2048).unwrap(),
        ..Default::default()
    }
    .json_schema(json!({
        "type": "object",
        "properties": {"content": {"type": "string"}},
        "required": ["content"],
    }));
    let mut session = deterministic(session_or_skip!());
    let reingest = session.dialect().reasoning.reingest;
    let opts = |gen: bool| {
        RenderOptions::default()
            .with_generation_prompt(gen)
            .with_extra("preserve_thinking", true)
            .with_thought_reingest(reingest)
    };
    let gen = session
        .template()
        .render_with(&prompt, &opts(true))
        .expect("render");

    let raw = session.complete_text(&prompt).expect("complete_text");
    println!("=== raw emission ===\n{raw:?}\n===");
    assert!(
        raw.contains("\n</think>\n\n{"),
        "the grammar must force the template's separator: {raw:?}"
    );
    let parsed = drama_llama::dialect::parse_text(
        session.dialect(),
        &[],
        &raw,
        true,
        drama_llama::dialect::Leniency::Final,
    )
    .blocks;

    let mut follow_up = prompt.clone();
    let assistant: AssistantMessage = parsed.into_iter().collect();
    follow_up.messages.push(assistant.into());
    follow_up.output_config = None;
    let rendered = session
        .template()
        .render_with(&follow_up, &opts(false))
        .expect("render");
    let suffix = rendered.strip_prefix(&gen).unwrap_or_else(|| {
        panic!("the turn must extend the generation prompt.\n{rendered:?}")
    });
    assert!(
        suffix.starts_with(&raw),
        "emission is not a byte prefix of the canonical re-render.\n\
         --- emission ---\n{raw:?}\n--- re-rendered suffix ---\n{suffix:?}"
    );
}

/// #96/#112 in the production shape: sliding markers, forced tool-call
/// turns *with thinking*, every continuation resuming past the entire
/// previous prompt via the tip. Red before the reingest probe.
#[test]
#[ignore = "requires Qwen3.8 model"]
fn tip_anchors_across_thinking_tool_rounds_issue_112() {
    common::tip::assert_tip_anchors_across_thinking_tool_rounds(
        session_or_skip!(),
        3,
    );
}

/// #96, the downstream (agentkit) shape, thinking off.
#[test]
#[ignore = "requires Qwen3.8 model"]
fn tip_anchors_across_tool_rounds_issue_96() {
    common::tip::assert_tip_anchors_across_tool_rounds(session_or_skip!(), 3);
}

/// #96's probe scenario: a continuation adding no new `cache_control`
/// anywhere may only be covered by the tip via the LCP walk.
#[test]
#[ignore = "requires Qwen3.8 model"]
fn tip_anchors_unmarked_continuation_issue_96() {
    common::tip::assert_tip_anchors_unmarked_continuation(session_or_skip!());
}
