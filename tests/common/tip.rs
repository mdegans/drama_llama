//! Shared #96 regression scenarios: the post-generation tip must
//! anchor prefix reuse for a *continuation* — the downstream shape
//! where only the newest messages are ever prefilled.
//!
//! Two scenarios, both parameterized by a per-model session so every
//! per-model suite runs the same assertions against its own template
//! and dialect:
//!
//! - [`assert_tip_anchors_across_tool_rounds`] — the agentkit shape:
//!   `tools+system | user… | assistant |` with the last two markers
//!   *sliding* forward each round and every assistant turn a tool
//!   call. Sliding matters: markers are metadata, not rendered bytes,
//!   so moving one must never invalidate the cache (a historical bug
//!   did exactly that).
//! - [`assert_tip_anchors_unmarked_continuation`] — issue #96's probe:
//!   a continuation that adds **no** new `cache_control` anywhere. The
//!   only anchor past the old markers is the tip via the LCP walk;
//!   before the #96 fix this scenario fell back to the last explicit
//!   marker on every model.
//!
//! Both assert `cache_read` strictly exceeds the *previous* call's
//! entire `input_tokens`: a resume past the whole previous prompt can
//! only come from the tip. No golden text anywhere — the models differ
//! per machine (CI runs smaller family members) and the tip contract
//! doesn't care what was said, only that it round-trips. Sessions are
//! forced greedy with repetition penalties off: this suite pins cache
//! mechanics, and a penalty steering the model into a non-canonical
//! re-spelling would fail the LCP for sampler reasons, not cache
//! reasons.
//!
//! A red here triages per Mike's rule for #96: if the emission
//! re-tokenizes non-canonically under a grammar, that is a *grammar
//! bug* to track (the grammar should constrain generation to
//! canonical tokenizations), not a flake to paper over.

use std::{borrow::Cow, num::NonZeroU32};

use drama_llama::{
    prompt::ToolResult, Block, Content, LlamaCppSession, Message, Prompt, Role,
    SamplingMode, Tool, ToolChoice,
};
use misanthropic::prompt::message::CacheControl;
use serde_json::json;

/// The tool every scenario calls: small schema, deterministic ask.
fn count_letters_tool() -> Tool {
    Tool::builder("count_letters")
        .description("Count the number of times a letter appears in a string.")
        .schema(json!({
            "type": "object",
            "properties": {
                "letter": {
                    "type": "string",
                    "description": "the letter to count"
                },
                "string": {
                    "type": "string",
                    "description": "the string to search"
                }
            },
            "required": ["letter", "string"]
        }))
        .build()
        .expect("valid test tool")
}

/// Force the determinism this suite needs regardless of what the
/// per-model constructor configured.
fn deterministic(session: LlamaCppSession) -> LlamaCppSession {
    session
        .without_repetition()
        .with_sampling([SamplingMode::Greedy])
        .with_prefix_cache(true)
}

/// Strip every `cache_control` from every message block — the "old"
/// half of sliding the marker window forward.
fn strip_message_markers(prompt: &mut Prompt) {
    for message in &mut prompt.messages {
        for block in &mut message.content.0 {
            match block {
                Block::Text { cache_control, .. } => *cache_control = None,
                Block::ToolUse { call } => call.cache_control = None,
                Block::ToolResult { result } => result.cache_control = None,
                _ => {}
            }
        }
    }
}

/// Set an ephemeral marker on the last markable block of `content` —
/// how a caching client marks the turn it just appended. The marker is
/// metadata, never rendered, so this must not perturb the render.
fn mark_last_block(content: &mut Content) {
    if let Some(block) = content.0.last_mut() {
        match block {
            Block::Text { cache_control, .. } => {
                *cache_control = Some(CacheControl::ephemeral());
            }
            Block::ToolUse { call } => {
                call.cache_control = Some(CacheControl::ephemeral());
            }
            Block::ToolResult { result } => {
                result.cache_control = Some(CacheControl::ephemeral());
            }
            _ => {}
        }
    }
}

/// One-word tag per block kind, for the per-round trace below.
fn block_kind(block: &Block) -> &'static str {
    match block {
        Block::Text { .. } => "text",
        Block::Thought { .. } => "thinking",
        Block::ToolUse { .. } => "tool_use",
        Block::ToolResult { .. } => "tool_result",
        _ => "other",
    }
}

/// Trace a round's shape to stderr: which block kinds the model
/// produced, and the usage that the assertions run against. Quiet on
/// green captured runs (cargo swallows stderr), decisive on red: the
/// first triage question for a tip miss is "did this turn carry
/// reasoning, and did it survive re-ingest?" — this answers half of
/// it from the failure output alone.
fn trace_round(label: &str, resp: &drama_llama::prompt::MessageResponse) {
    let kinds: Vec<&str> =
        resp.inner.content.0.iter().map(block_kind).collect();
    eprintln!(
        "{label}: blocks={kinds:?} input={} read={:?} output={}",
        resp.usage.input_tokens,
        resp.usage.cache_read_input_tokens,
        resp.usage.output_tokens,
    );
}

/// A user message whose single text block carries an ephemeral marker.
fn marked_user(text: String) -> Message {
    Message {
        role: Role::User,
        content: Content(vec![Block::Text {
            text: Cow::Owned(text),
            cache_control: Some(CacheControl::ephemeral()),
            citations: None,
        }]),
    }
}

/// The downstream (agentkit) shape:
/// `tools+system | user… | assistant |` with the last two markers
/// sliding forward each round, every turn a forced tool call, plus a
/// final unforced turn. Each continuation must resume past the entire
/// previous prompt — a fallback to the tools+system marker or to the
/// last user's marker is exactly the #96 regression and fails here.
pub fn assert_tip_anchors_across_tool_rounds(
    session: LlamaCppSession,
    tool_rounds: usize,
) {
    const WORDS: [&str; 6] = [
        "strawberry",
        "raspberry",
        "blueberry",
        "boysenberry",
        "huckleberry",
        "elderberry",
    ];
    assert!(tool_rounds < WORDS.len(), "not enough prompts for that");
    let mut session = deterministic(session);

    let mut prompt = Prompt {
        system: Some(Content::text(
            "You are a helpful assistant. You cannot count letters in a \
             word reliably on your own because you see in tokens, not \
             letters. Use the `count_letters` tool when asked to count \
             characters.",
        )),
        messages: vec![marked_user(format!(
            "Count the number of r's in '{}'",
            WORDS[0]
        ))],
        tools: Some(vec![count_letters_tool().into()]),
        tool_choice: Some(ToolChoice::method("count_letters")),
        // Thinking models reason before every call; a tight budget
        // ends generation mid-call as a constraint violation (seen on
        // Qwen3.6 at 1024 in round 3).
        max_tokens: NonZeroU32::new(4096).unwrap(),
        ..Default::default()
    };
    // The static marker: the tool definitions. Never slides.
    if let Some(def) = prompt.tools.as_mut().and_then(|t| t.first_mut()) {
        if let Some(tool) = def.as_method_mut() {
            tool.cache_control = Some(CacheControl::ephemeral());
        }
    }

    let mut prev_input: u64 = 0;
    for round in 0..tool_rounds {
        let resp = session
            .complete_response(&prompt)
            .unwrap_or_else(|e| panic!("tool round {round}: {e}"));
        trace_round(&format!("tool round {round}"), &resp);
        if round > 0 {
            let read =
                resp.usage.cache_read_input_tokens.unwrap_or_default() as u64;
            assert!(
                read > prev_input,
                "tool round {round}: tip missed — cache_read ({read}) \
                 did not clear the previous prompt ({prev_input}); the \
                 call fell back to an explicit marker (#96). \
                 usage: {:?}",
                resp.usage,
            );
        }
        prev_input = resp.usage.input_tokens as u64;

        let call = resp
            .inner
            .tool_use()
            .unwrap_or_else(|| {
                panic!(
                    "tool round {round}: no ToolUse block — the forced \
                     call did not round-trip"
                )
            })
            .clone();

        // Slide the window: strip the previous round's message markers,
        // then mark the turn pair being appended.
        strip_message_markers(&mut prompt);
        let mut assistant = resp.inner.content.clone();
        mark_last_block(&mut assistant);
        prompt.messages.push(Message {
            role: Role::Assistant,
            content: assistant,
        });
        prompt.messages.push(Message {
            role: Role::User,
            content: Content(vec![
                Block::ToolResult {
                    result: ToolResult {
                        tool_use_id: call.id.clone(),
                        // Honest count: a wrong result invites the
                        // model to second-guess the tool at length,
                        // burning the output budget on re-derivation.
                        content: Content::text(
                            WORDS[round].matches('r').count().to_string(),
                        ),
                        is_error: false,
                        cache_control: None,
                    },
                },
                Block::Text {
                    text: Cow::Owned(format!(
                        "Now count the r's in '{}'",
                        WORDS[round + 1]
                    )),
                    cache_control: Some(CacheControl::ephemeral()),
                    citations: None,
                },
            ]),
        });
    }

    // Final, unforced prose turn: the tip must anchor coming off a
    // tool-call ending too (grammar-complete is the normal ending for
    // a call — #88 phase 5's lesson).
    prompt.tool_choice = None;
    let resp = session.complete_response(&prompt).expect("final round");
    trace_round("final round", &resp);
    let read = resp.usage.cache_read_input_tokens.unwrap_or_default() as u64;
    assert!(
        read > prev_input,
        "final round: tip missed — cache_read ({read}) did not clear \
         the previous prompt ({prev_input}) (#96). usage: {:?}",
        resp.usage,
    );
}

/// Issue #96's probe scenario, verbatim: a continuation that appends
/// the assistant's actual reply plus a new user turn with **no new
/// `cache_control` anywhere**. The old markers still hash-match, so
/// before the fix the pick snapped to the last marker on every model;
/// only the tip, via the plain LCP walk, can cover the region past it.
pub fn assert_tip_anchors_unmarked_continuation(session: LlamaCppSession) {
    let mut session = deterministic(session);
    let mut prompt = Prompt {
        system: Some(Content::text(
            "You are a concise assistant. Answer in one short sentence.",
        )),
        messages: vec![marked_user("Name a primary color.".to_string())],
        max_tokens: NonZeroU32::new(96).unwrap(),
        ..Default::default()
    };

    let r1 = session.complete_response(&prompt).expect("round 1");
    trace_round("round 1", &r1);
    let input1 = r1.usage.input_tokens as u64;

    prompt.messages.push(Message {
        role: Role::Assistant,
        content: r1.inner.content.clone(),
    });
    prompt.messages.push(Message {
        role: Role::User,
        content: Content::text("Now name a shape."),
    });

    let r2 = session.complete_response(&prompt).expect("round 2");
    trace_round("round 2", &r2);
    let read2 = r2.usage.cache_read_input_tokens.unwrap_or_default() as u64;
    assert!(
        read2 > input1,
        "unmarked continuation: tip missed — cache_read ({read2}) did \
         not clear round 1's whole prompt ({input1}); the call fell \
         back to the marker inside it (#96). usage: {:?}",
        r2.usage,
    );
    assert!(
        (read2 as usize) < r2.usage.input_tokens as usize,
        "cache_read must remain a strict prefix of the new prompt",
    );
}
