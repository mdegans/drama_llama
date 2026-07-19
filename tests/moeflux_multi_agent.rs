//! moeflux twin of `session_cache::multi_agent_round_robin_hits`:
//! two agents with distinct histories round-robining through one
//! `Session<MoefluxBackend>`.
//!
//! On moeflux a cache slot's "sequence" is realized as whole-state
//! blobs (`state_save`/`state_load` — the decoder wrapper's
//! `(seq, pos)` snapshot store), and switching agents *is* a state
//! load. This test therefore exercises exactly the paths the llama.cpp
//! twin can't: cross-sequence restore via blob, the `active_seq`
//! stream-reset on a fresh agent's first prefill, and the
//! inactive-sequence `memory_seq_rm` guard (slot eviction while the
//! other agent's stream is live must not corrupt it — the corruption
//! would surface as gibberish in the next assertion).
//!
//! Run (assets on the external drive, override via
//! `DRAMA_LLAMA_MOEFLUX_*` like the pollution test):
//!
//! ```sh
//! cargo test --features "moeflux,moeflux-model-qwen3-6-35b-a3b" \
//!     --test moeflux_multi_agent --release -- --ignored --nocapture
//! ```

#![cfg(all(feature = "moeflux", target_os = "macos"))]

use std::{borrow::Cow, num::NonZeroU32, path::PathBuf};

use drama_llama::{
    Block, Content, Message, MoefluxEngine, Prompt, Role, Session,
};
use misanthropic::prompt::message::CacheControl;

fn env_path(var: &str, default: &str) -> PathBuf {
    PathBuf::from(std::env::var(var).unwrap_or_else(|_| default.to_string()))
}

fn build_session() -> Session<drama_llama::MoefluxBackend> {
    let mlx_dir = env_path(
        "DRAMA_LLAMA_MOEFLUX_MLX_DIR",
        "/Volumes/Temp Backup/models/moeflux/qwen3-6-35b-a3b-mlx-4bit",
    );
    let artifacts_dir = env_path(
        "DRAMA_LLAMA_MOEFLUX_ARTIFACTS_DIR",
        "/Volumes/Temp Backup/models/moeflux/qwen3-6-35b-a3b-artifacts",
    );
    let experts_dir = env_path(
        "DRAMA_LLAMA_MOEFLUX_EXPERTS_DIR",
        "/Volumes/Temp Backup/models/moeflux/qwen3-6-35b-a3b-root",
    );

    let engine = MoefluxEngine::from_paths(
        &mlx_dir,
        &artifacts_dir,
        &experts_dir,
        false,
    )
    .expect("MoefluxEngine::from_paths");

    Session::from_engine(engine)
        .expect("Session::from_engine")
        .with_prefix_cache(true)
}

/// A cache-marked user turn.
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

/// Distinct cached system persona + cached user turn.
fn agent_prompt(persona: &'static str, user: &'static str) -> Prompt {
    Prompt {
        system: Some(Content(vec![Block::Text {
            text: Cow::Borrowed(persona),
            cache_control: Some(CacheControl::ephemeral()),
            citations: None,
        }])),
        messages: vec![cached_user(user)],
        max_tokens: NonZeroU32::new(48).unwrap(),
        ..Prompt::default()
    }
}

#[test]
#[ignore = "long running; requires moeflux model assets"]
fn multi_agent_round_robin_moeflux() {
    std::env::set_var("DRAMA_LLAMA_CACHE_TRIPWIRE", "1");
    let mut session = build_session();

    let mut prompts: Vec<Prompt> = [
        "You are `ant`, a terse architect. Answer in one short sentence.",
        "You are `bee`, a cheerful builder. Answer in one short sentence.",
    ]
    .iter()
    .map(|p| agent_prompt(p, "Introduce yourself."))
    .collect();

    // Round 1: legitimate first-turn misses, interleaved.
    for prompt in prompts.iter_mut() {
        let r = session.complete_response(prompt).expect("round 1");
        let text = r.inner.content.to_string();
        assert!(!text.is_empty(), "round 1 output empty");
        prompt.messages.push(Message {
            role: Role::Assistant,
            content: r.inner.content,
        });
        prompt
            .messages
            .push(cached_user("Now name your favorite tool."));
    }

    // Round 2: every agent switch is a whole-state blob load; each
    // call must reuse its own slot and still produce coherent text.
    for (i, prompt) in prompts.iter().enumerate() {
        let r = session.complete_response(prompt).expect("round 2");
        let read = r.usage.cache_read_input_tokens.unwrap_or(0);
        assert!(
            read > 0,
            "agent {i} round 2 must hit its slot; got read={read}"
        );
        assert!(
            !r.inner.content.to_string().is_empty(),
            "agent {i} round 2 output empty — state-load corruption?"
        );
    }
}
