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

/// The hard tool-call cap, live: asked to fetch six posts at once,
/// cogito makes at most `CAP` calls — the sampler ends the turn on its
/// own `<|im_end|>` once the last completes — the turn reports
/// `tool_use`, and the next turn, carrying a result per call, resumes
/// past the whole previous prompt (the capped turn re-rendered byte for
/// byte). Repeats count toward the cap and are dropped after it, so the
/// returned calls may be fewer than `CAP`. `CAP` is below the sidecar's
/// 3, set on the session.
#[test]
#[ignore = "requires cogito model"]
fn tool_call_cap_ends_a_parallel_turn_cache_stable() {
    use drama_llama::prompt::ToolResult;
    use misanthropic::response::StopReason;
    const CAP: u32 = 2;
    let mut session =
        session_or_skip!().with_max_tool_calls_per_turn(NonZeroU32::new(CAP));
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
             your tool. When you need several posts, request all of them \
             at once, one call per post, in the same turn.",
        )),
        messages: vec![Message {
            role: Role::User,
            content: Content::text(
                "Read posts a1, b2, c3, d4, e5 and f6 — all six, now, in \
                 parallel — then summarize each in one line.",
            ),
        }],
        tools: Some(vec![tool.into()]),
        max_tokens: NonZeroU32::new(2048).unwrap(),
        ..Default::default()
    };

    let first = session.complete_response(&prompt).expect("first turn");
    eprintln!("first: {:#?} {:?}", first.inner.content, first.usage);
    let calls: Vec<_> = first
        .inner
        .content
        .0
        .iter()
        .filter_map(|b| match b {
            Block::ToolUse { call } => Some(call.clone()),
            _ => None,
        })
        .collect();
    assert!(
        !calls.is_empty(),
        "no call to cap: {:#?}",
        first.inner.content
    );
    assert!(
        calls.len() <= CAP as usize,
        "{} calls past a cap of {CAP}",
        calls.len()
    );
    assert_eq!(first.stop_reason, Some(StopReason::ToolUse));
    let prev_total = first.usage.cache_read_input_tokens.unwrap_or(0)
        + first.usage.cache_creation_input_tokens.unwrap_or(0)
        + first.usage.input_tokens;

    prompt.messages.push(Message {
        role: Role::Assistant,
        content: first.inner.content.clone(),
    });
    prompt.messages.push(Message {
        role: Role::User,
        content: Content(
            calls
                .iter()
                .map(|call| Block::ToolResult {
                    result: ToolResult {
                        tool_use_id: call.id.clone(),
                        content: Content::text(format!(
                            "Post {}: the garden needs watering.",
                            call.input["id"].as_str().unwrap_or("?")
                        )),
                        is_error: false,
                        cache_control: None,
                    },
                })
                .collect(),
        ),
    });
    let second = session.complete_response(&prompt).expect("second turn");
    eprintln!("second: {:#?} {:?}", second.inner.content, second.usage);
    let read = second.usage.cache_read_input_tokens.unwrap_or(0);
    assert!(
        read > prev_total,
        "tip missed: cache_read ({read}) did not clear the previous \
         prompt ({prev_total}) — the capped turn did not re-render byte \
         for byte. usage: {:?}",
        second.usage,
    );
}

/// #144, live: cogito copied a reply_to UUID from a tool result 13
/// messages back, got 28 characters right and drifted. Here the target
/// comment sits in a feed tool result under five later rounds of other
/// tool traffic, beside a decoy sharing its 8-hex label and the Agora
/// system sender (`00000000-…-0001`). Every trial's `reply_to` must be
/// a UUID the context holds, byte for byte — with the copy-lock on (the
/// sidecar default). The same trials with the lock off are printed for
/// comparison, not asserted: a drift there is the bug, not a failure.
///
/// Trials: `$DRAMA_LLAMA_ID_LOCK_TRIALS` (default 4), each its own seed
/// off `common::test_seed()`.
#[test]
#[ignore = "requires cogito model"]
fn id_copy_lock_keeps_a_far_back_uuid_exact() {
    use drama_llama::{prompt::ToolResult, prompt::ToolUse, ToolChoice};
    use std::{borrow::Cow, collections::BTreeSet};

    const TARGET: &str = "71da043d-c4ea-418b-b4c9-a019cf99c416";
    const DECOY: &str = "71da043d-0b1e-4c2f-9a7d-3e5f6a7b8c9d";
    const SYSTEM: &str = "00000000-0000-0000-0000-000000000001";
    let trials: u64 = std::env::var("DRAMA_LLAMA_ID_LOCK_TRIALS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);

    // Deterministic filler UUIDs (splitmix64), so the context is full
    // of near-miss hex.
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut uuid = move || {
        let mut next = || {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let (a, b) = (next(), next());
        format!(
            "{:08x}-{:04x}-4{:03x}-{:04x}-{:012x}",
            a >> 32,
            (a >> 16) & 0xffff,
            a & 0xfff,
            (b >> 48) | 0x8000,
            b & 0xffff_ffff_ffff
        )
    };
    let authors = [
        "ion-alphawave",
        "quiet-lantern",
        "moss-protocol",
        "aegis",
        "tidewatcher",
        "copper-finch",
    ];
    let topics = [
        "the seed vault inventory",
        "the council's quorum rule",
        "watering schedules",
        "the archive migration",
        "moderation appeals",
    ];
    let mut known: BTreeSet<String> =
        [TARGET, DECOY, SYSTEM].map(String::from).into();
    let mut feed =
        format!("[system {SYSTEM}] Welcome to the commons. Be kind.\n");
    for i in 0..24 {
        let id = match i {
            9 => TARGET.to_string(),
            17 => DECOY.to_string(),
            _ => uuid(),
        };
        known.insert(id.clone());
        let (author, topic) = if i == 9 {
            (
                "ion-alphawave",
                "the seed vault: the north shelf is mislabeled",
            )
        } else {
            (authors[i % authors.len()], topics[i % topics.len()])
        };
        feed.push_str(&format!("comment {id} by {author}: on {topic}.\n"));
    }

    let tool =
        |name: &'static str, props: serde_json::Value, required: &[&str]| {
            Tool::builder(name)
                .description(format!("Agora: {name}."))
                .schema(json!({
                    "type": "object",
                    "properties": props,
                    "required": required,
                }))
                .build()
                .expect("valid tool")
        };
    let tools = vec![
        tool("get_feed", json!({}), &[]).into(),
        tool("get_content", json!({"id": {"type": "string"}}), &["id"]).into(),
        tool(
            "create_comment",
            json!({
                "reply_to": {"type": "string"},
                "body": {"type": "string"},
            }),
            &["reply_to", "body"],
        )
        .into(),
    ];
    let call = |n: usize, name: &str, input: serde_json::Value| Message {
        role: Role::Assistant,
        content: Content(vec![Block::ToolUse {
            call: ToolUse {
                id: Cow::Owned(format!("call{n:05}")),
                name: Cow::Owned(name.to_string()),
                input,
                cache_control: None,
                caller: None,
            },
        }]),
    };
    let result = |n: usize, text: String| Message {
        role: Role::User,
        content: Content(vec![Block::ToolResult {
            result: ToolResult {
                tool_use_id: Cow::Owned(format!("call{n:05}")),
                content: Content::text(text),
                is_error: false,
                cache_control: None,
            },
        }]),
    };
    let mut messages = vec![
        Message {
            role: Role::User,
            content: Content::text("Catch up on the commons feed."),
        },
        call(0, "get_feed", json!({})),
        result(0, feed),
    ];
    // Five later rounds of unrelated reads, each result full of ids.
    for n in 1..=5 {
        let post = uuid();
        let mut body = format!("post {post} by quiet-lantern:\n");
        for _ in 0..4 {
            let id = uuid();
            body.push_str(&format!("  reply {id}: agreed, see above.\n"));
            known.insert(id);
        }
        known.insert(post.clone());
        messages.push(call(n, "get_content", json!({ "id": post })));
        messages.push(result(n, body));
    }
    messages.push(Message {
        role: Role::User,
        content: Content::text(
            "Reply to ion-alphawave's comment about the seed vault, from \
             the feed earlier, thanking them. Use create_comment with \
             reply_to set to that comment's full id.",
        ),
    });
    let prompt = Prompt {
        system: Some(Content::text(
            "You are aegis, an agent on Agora. Ids are UUIDs; copy them \
             exactly.",
        )),
        messages,
        tools: Some(tools),
        tool_choice: Some(ToolChoice::method("create_comment")),
        max_tokens: NonZeroU32::new(512).unwrap(),
        ..Default::default()
    };

    let path = model_path();
    let mut session = session_or_skip!();
    let Some(path) = path else { return };
    let sidecar = drama_llama::sidecar::load_sample_options(
        &path.with_extension("sampling.toml"),
    )
    .expect("sidecar parses")
    .expect("cogito ships a sampling sidecar");
    assert!(
        sidecar
            .repetition
            .as_ref()
            .is_some_and(|r| !r.id_patterns().is_empty() && r.id_copy_lock()),
        "the sidecar must name id_patterns and leave the lock on"
    );
    let base = common::test_seed().get();
    for lock in [true, false] {
        let mut opts = sidecar.clone();
        opts.repetition = opts.repetition.map(|r| r.set_id_copy_lock(lock));
        session = session.with_sample_options(opts);
        let mut hits = 0;
        for trial in 0..trials {
            let seed = std::num::NonZeroU128::new(
                base.wrapping_add(u128::from(trial)),
            )
            .or(std::num::NonZeroU128::new(1));
            session = session.with_seed(seed);
            let reply = session.complete_response(&prompt).expect("turn");
            let reply_to = reply
                .inner
                .content
                .0
                .iter()
                .find_map(|b| match b {
                    Block::ToolUse { call }
                        if call.name == "create_comment" =>
                    {
                        call.input["reply_to"].as_str().map(str::to_string)
                    }
                    _ => None,
                })
                .unwrap_or_default();
            let exact = known.contains(&reply_to);
            hits += usize::from(reply_to == TARGET);
            eprintln!(
                "lock={lock} trial={trial} reply_to={reply_to:?} \
                 known={exact} target={}",
                reply_to == TARGET
            );
            if lock {
                assert!(
                    exact,
                    "lock on, yet reply_to {reply_to:?} is no id in the \
                     context: {:#?}",
                    reply.inner.content
                );
            }
        }
        eprintln!("lock={lock}: {hits}/{trials} picked the target");
    }
}
