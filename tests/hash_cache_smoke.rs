//! Hash-keyed prefix-reuse end-to-end smoke test.
//!
//! Two-round conversation with a `Block::ToolUse` carrying a JSON
//! `input` — the case where the model's emitted whitespace can diverge
//! from `serde_json::to_string`'s canonical re-rendering between
//! rounds. Without hash-keyed reuse, the LCP / auto-tip path cuts off
//! at the first JSON-whitespace divergence (the cogito repro Mike
//! captured 2026-05-08 in the auto-tip-debug branch).
//!
//! With this PR's hash side-table, round 2's `partial_text` for the
//! conversation prefix matches the auto-tip hash drama_llama saved at
//! the end of round 1's generation, so cache_read jumps to ≈
//! input_tokens for the round-2 prefill regardless of any
//! BPE-whitespace drift in the assistant block.
//!
//! Requires a real model. Set `DRAMA_LLAMA_COGITO_MODEL` to a cogito-
//! style GGUF for the most direct repro of the original bug; falls
//! back to `models/model.gguf` (any tool-using chat model works for
//! the hash-side-table mechanic — cogito is just where the whitespace
//! divergence is most pronounced).
//!
//! Ignored by default: `cargo test --test hash_cache_smoke -- --ignored`.

#![cfg(feature = "llama-cpp")]

use std::{borrow::Cow, path::PathBuf};

use drama_llama::{
    prompt::{ToolResult, ToolUse},
    Block, Content, Message, Prompt, Role, Session, Tool,
};
use misanthropic::prompt::message::CacheControl;
use serde_json::json;

fn model_path() -> PathBuf {
    if let Ok(p) = std::env::var("DRAMA_LLAMA_COGITO_MODEL") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf")
}

/// Build the round-1 prompt: a tool, a system prompt, and a user
/// message asking the model to call the tool. `cache_control` on the
/// user message gives drama_llama a breakpoint to anchor the prefix
/// (system+tools+first_user_msg) — the same shape agora seeds.
fn build_round1() -> (Prompt, Tool) {
    let tool = Tool {
        name: Cow::Borrowed("count_letters"),
        description: Cow::Borrowed(
            "Count the number of times a letter appears in a string.",
        ),
        schema: json!({
            "type": "object",
            "properties": {
                "letter": {"type": "string"},
                "string": {"type": "string"}
            },
            "required": ["letter", "string"]
        }),
        cache_control: None,
        strict: None,
        defer_loading: None,
        allowed_callers: None,
    };

    let prompt = Prompt {
        system: Some(Content::text("You are a helpful assistant. Use the provided tool when answering.")),
        messages: vec![Message {
            role: Role::User,
            content: Content(vec![Block::Text {
                text: Cow::Borrowed(
                    "Count the number of r's in 'strawberry'.",
                ),
                cache_control: Some(CacheControl::ephemeral()),
                citations: None,
            }]),
        }],
        tools: Some(vec![tool.clone().into()]),
        ..Prompt::default()
    };

    (prompt, tool)
}

/// Extend the round-1 prompt with the assistant's tool_use response,
/// a tool_result, and a follow-up user message. `cache_control` on
/// the follow-up gives drama_llama a fresh breakpoint at the new tip.
fn build_round2(
    base_prompt: &Prompt,
    tool_use: ToolUse,
    result_text: &str,
) -> Prompt {
    let call_id = tool_use.id.clone();
    let mut prompt = base_prompt.clone();
    prompt.messages.push(Message {
        role: Role::Assistant,
        content: Content(vec![Block::ToolUse { call: tool_use }]),
    });
    prompt.messages.push(Message {
        role: Role::User,
        content: Content(vec![
            Block::ToolResult {
                result: ToolResult {
                    tool_use_id: call_id,
                    content: Content::text(result_text.to_string()),
                    is_error: false,
                    cache_control: None,
                },
            },
            Block::Text {
                text: Cow::Borrowed("Thanks. Now spell that word backwards."),
                cache_control: Some(CacheControl::ephemeral()),
                citations: None,
            },
        ]),
    });
    prompt
}

#[test]
#[ignore = "requires model; sets DRAMA_LLAMA_COGITO_MODEL or models/model.gguf"]
fn hash_keyed_prefix_reuse_carries_across_tool_use_round_trip() {
    let mut session = Session::from_path(model_path())
        .expect("model loads")
        .quiet()
        .with_prefix_cache(true);

    let (round1_prompt, _tool) = build_round1();

    // Round 1: complete and capture the assistant's first tool_use
    // (or fall back to a synthetic one if the model produced text
    // without a tool call — the test's mechanic still holds).
    let round1_resp = session
        .complete_response(&round1_prompt)
        .expect("round 1 completes");
    let round1_input_tokens = round1_resp.usage.input_tokens;
    let round1_cache_read = round1_resp.usage.cache_read_input_tokens;
    eprintln!(
        "round 1: input_tokens={}, cache_read={}",
        round1_input_tokens,
        round1_cache_read.unwrap_or(0),
    );

    // The response's tool_use (if any) — needed for round 2's
    // re-rendering. If the model produced text only, synthesize a
    // dummy ToolUse so we still exercise the hash-cache code path.
    let mut tool_use = round1_resp
        .inner
        .tool_use()
        .cloned()
        .unwrap_or_else(|| ToolUse {
            id: Cow::Borrowed("synthetic_call_1"),
            name: Cow::Borrowed("count_letters"),
            input: json!({"letter": "r", "string": "strawberry"}),
            cache_control: None,
            caller: None,
        });
    // Mark the assistant turn: the auto-tip hash matches a subsequent
    // request's partial render only where a breakpoint exists, and
    // breakpoints exist only at cache_control markers (see
    // `Session::compute_tip_hash` — "places a marker on (or just past)
    // that assistant message"). The marker is metadata, not rendered
    // content, so it doesn't perturb the hash itself.
    tool_use.cache_control = Some(CacheControl::ephemeral());

    // Round 2: extend the conversation; cache_read should jump from
    // ~prefix-only (system+tools+first_user_msg) to ~all-of-round-1
    // (auto-tip hit) when hash-keyed reuse fires.
    let round2_prompt = build_round2(&round1_prompt, tool_use, "3");
    let round2_resp = session
        .complete_response(&round2_prompt)
        .expect("round 2 completes");
    let round2_input_tokens = round2_resp.usage.input_tokens;
    let round2_cache_read =
        round2_resp.usage.cache_read_input_tokens.unwrap_or(0);
    eprintln!(
        "round 2: input_tokens={}, cache_read={}",
        round2_input_tokens, round2_cache_read,
    );

    // The minimal floor: round 2 must cache-read at least the
    // round-1 input prefix (the first cache_control marker's
    // breakpoint). That's already true today via the breakpoint
    // path. The interesting assertion: cache_read should reach the
    // auto-tip — i.e., should exceed round-1's *full* input length
    // (system + tools + first_user_msg + assistant + tool_result),
    // capturing the assistant content as well. We pick a permissive
    // threshold (round-1 input + 50% of round-1 generation tokens,
    // floored at round-1 input + 1) so the test passes whenever the
    // tip mechanism is live, even with single-token BPE drift.
    let round1_gen = round1_resp.usage.output_tokens;
    let tip_floor = (round1_input_tokens + (round1_gen / 2).max(1))
        .max(round1_input_tokens + 1);
    assert!(
        round2_cache_read >= tip_floor,
        "round 2 cache_read ({round2_cache_read}) should reach the auto-tip ({tip_floor}); \
         hash-keyed reuse appears not to be firing. \
         (input_tokens={round2_input_tokens}, round1_input={round1_input_tokens}, \
         round1_gen={round1_gen})",
    );
}
