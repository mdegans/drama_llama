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
    prompt::ToolResult, Block, Content, FromPath, Message, Prompt,
    RenderOptions, Role, Tool,
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
    let tool = Tool::builder("count_letters")
        .description("Count the number of times a letter appears in a string.")
        .schema(json!({
            "type": "object",
            "properties": {
                "letter": {"type": "string"},
                "string": {"type": "string"}
            },
            "required": ["letter", "string"]
        }))
        .build()
        .expect("valid test tool");

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

/// Extend the round-1 prompt with the assistant's response content
/// (verbatim, as parsed) and a follow-up user turn. `cache_control` on
/// the follow-up gives drama_llama a fresh breakpoint at the new tip.
fn build_round2(
    base_prompt: &Prompt,
    assistant_content: Content,
    user_blocks: Vec<Block>,
) -> Prompt {
    let mut prompt = base_prompt.clone();
    prompt.messages.push(Message {
        role: Role::Assistant,
        content: assistant_content,
    });
    prompt.messages.push(Message {
        role: Role::User,
        content: Content(user_blocks),
    });
    prompt
}

/// Set an ephemeral cache marker on the last block of an assistant
/// turn that supports one. The auto-tip hash matches a subsequent
/// request's partial render only where a breakpoint exists, and
/// breakpoints exist only at `cache_control` markers (see
/// `Session::compute_tip_hash`). The marker is metadata, not rendered
/// content, so it doesn't perturb the hash itself.
fn mark_last_block(content: &mut Content) {
    if let Some(last) = content.last_mut() {
        match last {
            Block::Text { cache_control, .. } => {
                *cache_control = Some(CacheControl::ephemeral());
            }
            Block::ToolUse { call } => {
                call.cache_control = Some(CacheControl::ephemeral());
            }
            _ => {}
        }
    }
}

#[test]
#[ignore = "requires model; sets DRAMA_LLAMA_COGITO_MODEL or models/model.gguf"]
fn hash_keyed_prefix_reuse_carries_across_tool_use_round_trip() {
    // `preserve_thinking` keeps prior-turn reasoning in re-renders.
    // Without it, think-stripping templates (Qwen3.5/3.6) render round
    // 2's transcript with different bytes than round 1 generated, the
    // transcripts genuinely diverge at the assistant turn, and the
    // auto-tip correctly cannot fire — reuse stops at the divergence.
    // Byte-stable rendering is the contract the tip mechanic needs.
    let mut session = drama_llama::LlamaCppSession::from_path(model_path())
        .expect("model loads")
        .quiet()
        .with_render_opts(
            RenderOptions::default()
                .with_generation_prompt(true)
                .with_extra("preserve_thinking", true),
        )
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

    // Round 2's assistant turn is the content round 1 *actually
    // produced*, verbatim — the auto-tip hash was computed over those
    // parsed blocks, so substituting anything else (e.g. a synthetic
    // tool call) guarantees a miss. A tool_result rides along only if
    // the model really called the tool; otherwise the follow-up is a
    // plain user turn.
    let mut assistant_content = round1_resp.inner.content.clone();
    mark_last_block(&mut assistant_content);

    let mut user_blocks: Vec<Block> = Vec::new();
    if let Some(call) = round1_resp.inner.tool_use() {
        user_blocks.push(Block::ToolResult {
            result: ToolResult {
                tool_use_id: call.id.clone(),
                content: Content::text("3"),
                is_error: false,
                cache_control: None,
            },
        });
    }
    user_blocks.push(Block::Text {
        text: Cow::Borrowed("Thanks. Now spell that word backwards."),
        cache_control: Some(CacheControl::ephemeral()),
        citations: None,
    });

    // Round 2: extend the conversation; cache_read should jump from
    // ~prefix-only (system+tools+first_user_msg) to ~all-of-round-1
    // (auto-tip hit) when hash-keyed reuse fires.
    let round2_prompt =
        build_round2(&round1_prompt, assistant_content, user_blocks);
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

    // The assertion this test shipped without (2026-07-24): a tip
    // resume must actually GENERATE. The stale-matcher-carry bug made
    // every round 2 after a tool round-trip emit 0 tokens (the tip's
    // grammar matcher, parked at tool-call-complete, carried into the
    // fresh assistant turn where only the turn terminator was legal) —
    // while this test's cache_read threshold kept passing. Cache stats
    // are not a proxy for output.
    assert!(
        round2_resp.usage.output_tokens > 0,
        "round 2 generated no tokens after the tip resume \
         (stale constraint-matcher carry?)",
    );
    assert!(
        !round2_resp.inner.content.0.is_empty(),
        "round 2 returned an empty content array",
    );
}
