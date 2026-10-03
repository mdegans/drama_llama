//! Cogito e2e: the #96 tip suite for the cohort-majority model, which
//! never had per-model coverage (#85's own observation — "cogito has
//! had comparatively little tool-path testing relative to
//! Qwen/Gemma/gpt-oss, despite being the majority model in our
//! deployed cohort").
//!
//! This suite exists to settle #85. The 2026-07-29 diagnosis
//! (COLLAPSE captures from the 2026-07-28 seed run, which predates the
//! #96 fix — the binary was swapped under a live server) points away
//! from the issue's render-defect hypothesis and at the #96 lookup
//! composition bug in a cogito costume:
//!
//! - detection is NOT misfiring to a Qwen-family entry: the model's
//!   embedded template byte-matches the dedicated `cogito-gguf.jinja`
//!   key (`scripts/gguf_template.py --compare` exits 0), so rung 2
//!   serves `baked::COGITO`'s cache-stable replacement;
//! - the famous 3-token first-round-trip deficit is exactly the ChatML
//!   generation tail `<|im_start|>assistant\n` — the bytes past a
//!   final-user-turn marker, which the pre-fix hash-first lookup
//!   capped reuse at (the tip past it was unreachable, #96);
//! - the compounding deficits (59, 1115, …) track the sliding marker
//!   window's distance-to-prompt-end, and the "healthy" negative
//!   rounds are the ones whose fresh marker landed past the previous
//!   prompt.
//!
//! If that reading is right, the two scenarios below — the agentkit
//! sliding-marker shape and the unmarked continuation — are green on
//! the post-fix tree and #85 closes as a #96 duplicate once a
//! restarted server reruns clean. If either is red, then per Mike's
//! #96 triage rule it is a real cogito canonicity/render finding (the
//! issue's original hypothesis), finally with a local repro.
//!
//! **No template sidecar**: like every model in `models/`, cogito
//! rides rung 2 (baked detection) in production, and this suite must
//! exercise the same path. The rung-2 witness discipline lives in
//! `session_mistral4.rs`; here the guard is only that a stray sidecar
//! must not silently promote the suite to rung 1 and void the claim.
//!
//! All tests load `models/cogito-32b.gguf` (override with
//! `$DRAMA_LLAMA_COGITO_MODEL`) and are `#[ignore]`d. Absent that
//! model they skip loudly rather than substituting `model.gguf`.

#![cfg(feature = "llama-cpp")]

mod common;

use std::{num::NonZeroU32, path::PathBuf};

use drama_llama::{Block, Content, FromPath, Message, Prompt, Role, Tool};
use misanthropic::prompt::{thinking::Thinking, Effort};
use serde_json::json;

/// Resolve the cogito GGUF: `$DRAMA_LLAMA_COGITO_MODEL` if set and
/// present, else the conventional path under `models/`. `None` means
/// skip — never substitute `model.gguf`.
fn model_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("DRAMA_LLAMA_COGITO_MODEL") {
        let p = PathBuf::from(p);
        return p.exists().then_some(p);
    }
    let conventional = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models/cogito-32b.gguf");
    conventional.exists().then_some(conventional)
}

/// A session for the multi-round #96 scenarios: rung 2, prefix cache
/// on, and a real context size (the default `n_ctx` ends later rounds
/// at the KV ceiling mid-tool-call).
///
/// Seeded via `common::test_seed()`: random by default (free fuzzing),
/// printed on failure so the trajectory is replayable with
/// `DRAMA_LLAMA_TEST_SEED=<n>`. The session seed selects the sampler
/// fork branch (fresh state per call); these suites assert on
/// emissions and the KV cache, never on the carried sampler stream.
fn load_session_8k() -> Option<drama_llama::LlamaCppSession> {
    let path = model_path()?;
    let sidecar = path.with_extension("template.jinja");
    assert!(
        !sidecar.exists(),
        "a template sidecar exists at {}, which would promote this \
         suite to rung 1 and void its rung-2 claim. Delete it and \
         re-run.",
        sidecar.display()
    );
    Some(
        drama_llama::LlamaCppSession::from_path_with(
            path,
            drama_llama::LlamaCppOptions::default().with_n_ctx(8192),
        )
        .expect("session load")
        .quiet()
        .with_prefix_cache(true)
        .with_seed(Some(common::test_seed())),
    )
}

macro_rules! session_or_skip {
    () => {
        match load_session_8k() {
            Some(s) => s,
            None => {
                eprintln!(
                    "SKIP: needs a cogito model \
                     (DRAMA_LLAMA_COGITO_MODEL or models/cogito-32b.gguf)"
                );
                return;
            }
        }
    };
}

/// #96, the downstream (agentkit) shape against cogito's ChatML-style
/// template: sliding markers, forced tool-call turns, every
/// continuation resuming past the entire previous prompt via the tip.
/// This is the exact shape behind #85's compounding deficits.
#[test]
#[ignore = "requires cogito model"]
fn tip_anchors_across_tool_rounds_issue_96() {
    common::tip::assert_tip_anchors_across_tool_rounds(session_or_skip!(), 3);
}

/// #96's probe scenario on cogito: a continuation adding no new
/// `cache_control` anywhere may only be covered by the tip via the
/// LCP walk.
#[test]
#[ignore = "requires cogito model"]
fn tip_anchors_unmarked_continuation_issue_96() {
    common::tip::assert_tip_anchors_unmarked_continuation(session_or_skip!());
}

/// Deep thinking, in the cohort's production shape: a system prompt,
/// an offered (unforced) tool, `thinking: adaptive` and `effort:
/// medium`, the session's own sampling. The baked template states the
/// incantation and pre-opens `<think>\n`, so the turn must come back
/// as a `Thought` then a clean answer — no thought framing left in the
/// text — and the next turn must resume past the whole previous prompt
/// (the thinking turn re-rendered byte for byte).
#[test]
#[ignore = "requires cogito model"]
fn adaptive_thinking_yields_a_thought_and_a_stable_next_turn() {
    let mut session = session_or_skip!();
    let gen_prompt =
        |prompt: &Prompt, session: &drama_llama::LlamaCppSession| {
            session
                .template()
                .render_with(
                    prompt,
                    &drama_llama::RenderOptions::default()
                        .with_generation_prompt(true)
                        .with_extra("preserve_thinking", true)
                        .with_thought_reingest(
                            session.dialect().reasoning.reingest,
                        ),
                )
                .expect("render")
        };
    let tool = Tool::builder("get_content")
        .description("Read a forum post by id.")
        .schema(json!({
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        }))
        .build()
        .expect("valid tool");
    let mut prompt = Prompt {
        system: Some(Content::text(
            "You are aegis, an agent on a small forum. Read posts with \
             your tool when you need them; otherwise answer directly.",
        )),
        messages: vec![Message {
            role: Role::User,
            content: Content::text(
                "Is 3599 a prime number? Answer in one sentence.",
            ),
        }],
        tools: Some(vec![tool.into()]),
        thinking: Some(Thinking::adaptive()),
        max_tokens: NonZeroU32::new(4096).unwrap(),
        ..Default::default()
    }
    .effort(Effort::Medium);
    let gen = gen_prompt(&prompt, &session);
    assert!(
        gen.contains("Enable deep thinking subroutine.")
            && gen.ends_with("<|im_start|>assistant\n<think>\n"),
        "the render must ask for and pre-open the thought: {gen:?}"
    );

    let first = session.complete_response(&prompt).expect("first turn");
    eprintln!("first: {:#?} {:?}", first.inner.content, first.usage);
    let blocks = &first.inner.content.0;
    assert!(
        matches!(
            blocks.first(),
            Some(Block::Thought { thought, .. }) if !thought.trim().is_empty()
        ),
        "thinking on must open the turn with a Thought: {blocks:#?}"
    );
    let answer: String = blocks
        .iter()
        .filter_map(|b| match b {
            Block::Text { text, .. } => Some(text.as_ref()),
            _ => None,
        })
        .collect();
    assert!(!answer.trim().is_empty(), "no answer after the thought");
    assert!(
        !answer.contains("<think>") && !answer.contains("</think>"),
        "thought framing leaked into the answer: {answer:?}"
    );
    let prev_total = first.usage.cache_read_input_tokens.unwrap_or(0)
        + first.usage.cache_creation_input_tokens.unwrap_or(0)
        + first.usage.input_tokens;

    prompt.messages.push(Message {
        role: Role::Assistant,
        content: first.inner.content.clone(),
    });
    prompt.messages.push(Message {
        role: Role::User,
        content: Content::text("And 3601? One sentence."),
    });
    let next = gen_prompt(&prompt, &session);
    assert!(
        next.starts_with(gen.as_str()),
        "the next request must extend the first"
    );
    let second = session.complete_response(&prompt).expect("second turn");
    eprintln!("second: {:#?} {:?}", second.inner.content, second.usage);
    let read = second.usage.cache_read_input_tokens.unwrap_or(0);
    assert!(
        read > prev_total,
        "tip missed: cache_read ({read}) did not clear the previous \
         prompt ({prev_total}) — the thinking turn did not re-render \
         byte for byte. usage: {:?}",
        second.usage,
    );
    assert!(
        matches!(second.inner.content.0.first(), Some(Block::Thought { .. })),
        "the second turn thinks too: {:#?}",
        second.inner.content
    );
}

/// #96/#112 on cogito with deep thinking: forced tool-call turns, each
/// continuation resuming past the entire previous prompt via the tip.
#[test]
#[ignore = "requires cogito model"]
fn tip_anchors_across_thinking_tool_rounds() {
    common::tip::assert_tip_anchors_across_thinking_tool_rounds(
        session_or_skip!(),
        3,
    );
}
