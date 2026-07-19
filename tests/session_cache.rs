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

use std::{borrow::Cow, num::NonZeroU32, path::PathBuf};

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
        max_tokens: NonZeroU32::new(48).unwrap(),
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

/// A per-agent prompt: distinct cached system persona + a cached user
/// turn — the swarm/council shape in miniature.
fn agent_prompt(persona: &'static str, user: &'static str) -> Prompt {
    Prompt {
        system: Some(Content(vec![Block::Text {
            text: Cow::Borrowed(persona),
            cache_control: Some(CacheControl::ephemeral()),
            citations: None,
        }])),
        messages: vec![cached_user(user)],
        ..Prompt::default()
    }
}

/// Three agents with distinct histories round-robining through ONE
/// session must each get prefix reuse on their second round — the
/// multi-slot cache's reason to exist. The tripwire is armed for the
/// whole test: any unexpected miss past an agent's first turn panics
/// with a cache-state dump instead of silently re-prefilling.
#[test]
#[ignore = "long running, requires models/model.gguf"]
fn multi_agent_round_robin_hits() {
    // OnceLock reads this on the first miss check; nextest's
    // process-per-test isolation keeps it scoped to this test.
    std::env::set_var("DRAMA_LLAMA_CACHE_TRIPWIRE", "1");
    let mut session =
        LlamaCppSession::from_path_with_cache_slots(model_path(), 4096, 3)
            .expect("session load")
            .quiet()
            .without_repetition()
            .with_sampling([SamplingMode::Greedy])
            .with_prefix_cache(true);

    let personas = [
        "You are `ant`, a terse architect. One short sentence.",
        "You are `bee`, a cheerful builder. One short sentence.",
        "You are `moth`, a skeptical tester. One short sentence.",
    ];
    let mut prompts: Vec<Prompt> = personas
        .iter()
        .map(|p| {
            agent_prompt(p, "Introduce yourself.")
                .max_tokens(NonZeroU32::new(32).unwrap())
        })
        .collect();

    // Round 1: every agent's first turn is a legitimate miss.
    for prompt in prompts.iter_mut() {
        let r = session.complete_response(prompt).expect("round 1");
        extend(prompt, r.inner.content, "Now name your favorite tool.");
    }

    // Round 2, interleaved: every call must reuse its own slot.
    for (i, prompt) in prompts.iter().enumerate() {
        let r = session.complete_response(prompt).expect("round 2");
        let read = r.usage.cache_read_input_tokens.unwrap_or(0);
        assert!(
            read > 0,
            "agent {i} round 2 must hit its slot; got read={read}"
        );
    }
}

/// With a cell budget that fits only ~2 of 3 agents, the
/// least-recently-used agent's slot is evicted; its next call is a
/// *working* miss (coherent output, zero reuse), and an immediate
/// repeat re-establishes the slot and hits.
#[test]
#[ignore = "long running, requires models/model.gguf"]
fn capacity_eviction_recovers() {
    let mut session =
        LlamaCppSession::from_path_with_cache_slots(model_path(), 2048, 3)
            .expect("session load")
            .quiet()
            .without_repetition()
            .with_sampling([SamplingMode::Greedy])
            .with_prefix_cache_config(drama_llama::PrefixCacheConfig {
                max_slots: 3,
                // Each agent's history is ~33 cells and an incoming call
                // budgets ~54 (30-cell prompt + 24-token headroom): two
                // resident histories + one incoming call exceed 100, so the
                // third arrival must evict the LRU slot; one resident + one
                // incoming fits.
                capacity_cells: Some(100),
            });

    let a = agent_prompt("You are agent A. One short sentence.", "Say 'A'.")
        .max_tokens(NonZeroU32::new(24).unwrap());
    let b = agent_prompt("You are agent B. One short sentence.", "Say 'B'.")
        .max_tokens(NonZeroU32::new(24).unwrap());
    let c = agent_prompt("You are agent C. One short sentence.", "Say 'C'.")
        .max_tokens(NonZeroU32::new(24).unwrap());

    session.complete_response(&a).expect("A primes");
    session.complete_response(&b).expect("B primes");
    session.complete_response(&c).expect("C evicts A (LRU)");

    // A was evicted: working miss.
    let r = session.complete_response(&a).expect("A after eviction");
    assert_eq!(
        r.usage.cache_read_input_tokens,
        Some(0),
        "evicted agent must re-prefill"
    );
    assert!(
        !r.inner.content.to_string().is_empty(),
        "post-eviction output must be coherent"
    );

    // ...and the re-established slot hits on repeat.
    let r = session.complete_response(&a).expect("A re-established");
    assert!(
        r.usage.cache_read_input_tokens.unwrap_or(0) > 0,
        "re-established slot must hit"
    );
}
