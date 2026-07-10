//! Integration tests for `Session` prompt caching: breakpoint reuse
//! across a growing conversation, and correctness of the cache path
//! (cached generation must equal uncached generation).
//!
//! Written against the llama.cpp backend (the only one buildable on
//! Linux); the same assertions are intended to hold for every
//! [`Backend`](drama_llama::Backend) — port to moeflux when a macOS
//! runner is available (see `tests/moeflux_session_pollution.rs` for
//! the existing moeflux-side parity test).
//!
//! All tests load a real model: `cargo test --test session_cache --
//! --ignored`.

#![cfg(feature = "llama-cpp")]

use std::{borrow::Cow, num::NonZeroUsize, path::PathBuf};

use drama_llama::{
    Block, Content, LlamaCppSession, Message, Prompt, Role, SamplingMode,
};
use misanthropic::prompt::message::CacheControl;

fn model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf")
}

/// Deterministic session: greedy, no repetition penalty, short cap.
fn session(prefix_cache: bool) -> LlamaCppSession {
    LlamaCppSession::from_path_sync(model_path())
        .expect("session load")
        .quiet()
        .without_repetition()
        .with_sampling([SamplingMode::Greedy])
        .with_max_tokens(NonZeroUsize::new(48).unwrap())
        .with_prefix_cache(prefix_cache)
}

/// A user message whose single text block carries an ephemeral cache
/// breakpoint — the shape a caching client sends for its latest turn.
fn cached_user(text: &'static str) -> Message {
    Message {
        role: Role::User,
        content: Content(vec![Block::Text {
            text: Cow::Borrowed(text),
            cache_control: Some(CacheControl::ephemeral()),
            citations: None,
        }]),
    }
}

fn base_prompt() -> Prompt {
    Prompt {
        system: Some(Content::text(
            "You are a concise assistant. Answer in one short sentence.",
        )),
        messages: vec![cached_user("Name a primary color.")],
        ..Prompt::default()
    }
}

/// Push the assistant's reply and a fresh cache-marked user turn —
/// the standard multi-round caching shape.
fn extend(prompt: &mut Prompt, reply: Content, next_user: &'static str) {
    prompt.messages.push(Message {
        role: Role::Assistant,
        content: reply,
    });
    prompt.messages.push(cached_user(next_user));
}

/// Across a growing conversation, `cache_read_input_tokens` must
/// start at zero, become nonzero once a breakpoint exists, and keep
/// growing as the reusable prefix grows — while always staying below
/// the request's `input_tokens`.
#[test]
#[ignore = "long running, requires models/model.gguf"]
fn cache_read_grows_across_rounds() {
    let mut session = session(true);
    let mut prompt = base_prompt();

    let r1 = session.complete_response(&prompt).expect("round 1");
    assert_eq!(
        r1.usage.cache_read_input_tokens,
        Some(0),
        "first round has nothing to reuse"
    );

    extend(&mut prompt, r1.inner.content.clone(), "Now name a shape.");
    let r2 = session.complete_response(&prompt).expect("round 2");
    let read2 = r2.usage.cache_read_input_tokens.unwrap_or(0);
    assert!(
        read2 > 0,
        "round 2 must reuse the round-1 prefix, got cache_read={read2}"
    );
    assert!(
        read2 < r2.usage.input_tokens,
        "cache_read ({read2}) must be a strict prefix of input_tokens ({})",
        r2.usage.input_tokens
    );

    extend(&mut prompt, r2.inner.content.clone(), "And name an animal.");
    let r3 = session.complete_response(&prompt).expect("round 3");
    let read3 = r3.usage.cache_read_input_tokens.unwrap_or(0);
    assert!(
        read3 > read2,
        "round 3 should reuse a longer prefix than round 2 \
         (read2={read2}, read3={read3})"
    );
    assert!(read3 < r3.usage.input_tokens);
}

/// A conversation that returns to a previously-cached prefix after a
/// diverging turn still reuses the shared breakpoint (the
/// system+first-turn anchor must survive the divergence).
#[test]
#[ignore = "long running, requires models/model.gguf"]
fn shared_prefix_survives_divergent_turn() {
    let mut session = session(true);

    let shared = base_prompt();
    let r1 = session.complete_response(&shared).expect("prime");
    let primed_read = r1.usage.input_tokens;

    // Divergent continuation A.
    let mut branch_a = shared.clone();
    extend(&mut branch_a, r1.inner.content.clone(), "Why that one?");
    session.complete_response(&branch_a).expect("branch a");

    // Divergent continuation B — same shared prefix, different tail.
    let mut branch_b = shared.clone();
    extend(&mut branch_b, r1.inner.content.clone(), "Name another.");
    let rb = session.complete_response(&branch_b).expect("branch b");
    let read_b = rb.usage.cache_read_input_tokens.unwrap_or(0);
    assert!(
        read_b > 0,
        "branch B shares the primed prefix and must reuse it, \
         got cache_read={read_b} (primed input was {primed_read})"
    );
}

/// The cache is an optimization, not a semantic: with deterministic
/// (greedy, penalty-free) sampling, a session reusing cached prefixes
/// must produce byte-identical output to a fresh session with the
/// cache disabled. This is the llama.cpp analog of moeflux's
/// `partial_hit_output_matches_fresh_session`.
#[test]
#[ignore = "long running, requires models/model.gguf"]
fn cached_output_matches_uncached_output() {
    // The sessions run sequentially, not side by side: two fully
    // offloaded models don't fit a 24 GB card together, and the
    // comparison doesn't need them concurrent.
    let prompt = base_prompt();
    let mut prompt2 = prompt.clone();

    let (cached_text_1, cached_text_2, cached_read_2) = {
        let mut cached = session(true);
        let r1 = cached.complete_response(&prompt).expect("cached r1");
        // Round 2 extends with the cached session's assistant turn;
        // the fresh session will replay the identical transcript, so
        // the cached session's reuse path is the only variable.
        extend(&mut prompt2, r1.inner.content.clone(), "Now name a shape.");
        let r2 = cached.complete_response(&prompt2).expect("cached r2");
        (
            r1.inner.content.to_string(),
            r2.inner.content.to_string(),
            r2.usage.cache_read_input_tokens.unwrap_or(0),
        )
    };

    let mut fresh = session(false);
    let out_fresh_1 = fresh.complete_response(&prompt).expect("fresh r1");
    assert_eq!(
        cached_text_1,
        out_fresh_1.inner.content.to_string(),
        "round 1 outputs diverged before any cache reuse — \
         nondeterminism, not a cache bug"
    );

    let out_fresh_2 = fresh.complete_response(&prompt2).expect("fresh r2");
    assert!(
        cached_read_2 > 0,
        "cached session failed to reuse — test would vacuously pass"
    );
    assert_eq!(
        cached_text_2,
        out_fresh_2.inner.content.to_string(),
        "cache-path output diverged from fresh-session output"
    );
}

/// `clear_prefix_cache` invalidates reuse: the next identical request
/// re-prefills from scratch.
#[test]
#[ignore = "long running, requires models/model.gguf"]
fn clear_prefix_cache_forces_full_reprefill() {
    let mut session = session(true);
    let prompt = base_prompt();

    session.complete_response(&prompt).expect("prime");
    session.clear_prefix_cache();
    let r = session.complete_response(&prompt).expect("after clear");
    assert_eq!(
        r.usage.cache_read_input_tokens,
        Some(0),
        "cleared cache must not reuse"
    );
}
