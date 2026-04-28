//! Regression test: `Session<MoefluxBackend>` must not produce
//! degenerate output on consecutive completions.
//!
//! Tracks the blallama symptom catalog from
//! `.claude/memory/blallama_session_state_pollution.md`. Sends three
//! distinct chat prompts back-to-back on a single Session with the
//! prefix cache enabled, then asserts none of the outputs degenerated:
//!
//! - **Symptom B**: model gives up after ~24 tokens. Caught by the
//!   per-completion length floor.
//! - **Symptom C**: long verbatim substring shared across two
//!   completions (or repeated within one). Caught by the cross-output
//!   substring check.
//!
//! The bug lived in moeflux: `mf_memory_seq_rm` partial-truncate
//! reset linear-attention layer state to zeros, losing the recurrence
//! across cache reuse. Phase 7 (this branch) replaces the
//! `memory_seq_rm` cache path with snapshot/restore at breakpoint
//! boundaries — `Session::kv_setup_and_chunk_prefill` checkpoints
//! after each cache-breakpoint chunk during prefill and rewinds via
//! `Engine::restore_to` on partial hit. Tests in this file should
//! pass post-fix.
//!
//! `#[ignore]`'d — needs the moeflux artifacts mounted. Run with:
//!
//! ```bash
//! cargo test --features "moeflux,moeflux-model-qwen3-6-35b-a3b" \
//!     --test moeflux_session_pollution --release \
//!     -- --ignored --nocapture
//! ```

#![cfg(all(feature = "moeflux", target_os = "macos"))]

use std::borrow::Cow;
use std::num::NonZeroUsize;
use std::path::PathBuf;

use drama_llama::{
    Content, MoefluxEngine, Message, Prompt, Role, Session,
};

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
        8,     // experts_per_tok
        false, // use_2bit
    )
    .expect("MoefluxEngine::from_paths");

    Session::from_engine(engine)
        .expect("Session::from_engine")
        .with_prefix_cache(true)
        .with_max_tokens(NonZeroUsize::new(MAX_TOKENS).unwrap())
}

fn user_prompt(system: &'static str, user: &'static str) -> Prompt {
    Prompt {
        system: Some(Content::SinglePart(Cow::Borrowed(system))),
        messages: vec![Message {
            role: Role::User,
            content: Content::SinglePart(Cow::Borrowed(user)),
        }],
        ..Prompt::default()
    }
}

/// Length floor per completion. Symptom B emits ~24 tokens (a few
/// dozen characters); 600 chars is comfortably above the floor and
/// well below what a healthy 600-token essay produces.
const MIN_CHARS_PER_COMPLETION: usize = 600;

/// Window for the cross-output substring check. Symptom C emits
/// 40-token verbatim paragraphs (typically ~200 chars); a 100-character
/// window catches the repetition without false-positiving on shared
/// phrasing across distinct essays.
const VERBATIM_WINDOW: usize = 100;

/// Generation cap. Original blallama repro used 600-token outputs;
/// shorter caps don't reliably reproduce symptoms B or C — a model
/// that's about to derail still emits coherent text for the first
/// couple hundred tokens.
const MAX_TOKENS: usize = 600;

/// True if `a` and `b` share any substring of `window` consecutive
/// chars. O(|a| * |b|) but inputs are small (a few KB).
fn share_long_substring(a: &str, b: &str, window: usize) -> bool {
    if a.len() < window || b.len() < window {
        return false;
    }
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    for i in 0..=a_bytes.len() - window {
        let needle = &a_bytes[i..i + window];
        if b_bytes
            .windows(window)
            .any(|w: &[u8]| w == needle)
        {
            return true;
        }
    }
    false
}

#[test]
#[ignore = "long running; needs moeflux artifacts"]
fn three_consecutive_completions_do_not_degenerate() {
    let mut session = build_session();
    eprintln!(
        "[pollution] session built, prefix_cache=on max_tokens={MAX_TOKENS}"
    );

    // Same shape as the original blallama symptom-B/C repro: long
    // essay prompts that share a chat-template prefix and diverge in
    // the user message. The 600-token cap forces the kind of extended
    // generation where state pollution surfaces.
    let prompts = [
        user_prompt(
            "You are a helpful assistant.",
            "Write a detailed essay (around 600 words) about the Apollo 11 mission \
             and its historical significance.",
        ),
        user_prompt(
            "You are a helpful assistant.",
            "Write a detailed essay (around 600 words) about the early history of \
             the internet, from ARPANET to the World Wide Web.",
        ),
        user_prompt(
            "You are a helpful assistant.",
            "Write a detailed essay (around 600 words) about the origins and \
             cultural significance of jazz music.",
        ),
    ];

    let mut outputs = Vec::with_capacity(prompts.len());
    for (i, prompt) in prompts.iter().enumerate() {
        let text = session
            .complete_text(prompt)
            .unwrap_or_else(|e| panic!("complete_text #{i}: {e:?}"));
        eprintln!(
            "[pollution] completion #{i} len={} chars first=`{}…`",
            text.len(),
            text.chars().take(80).collect::<String>(),
        );
        outputs.push(text);
    }

    // Symptom B: per-completion length floor.
    for (i, text) in outputs.iter().enumerate() {
        assert!(
            text.len() >= MIN_CHARS_PER_COMPLETION,
            "completion #{i} only {} chars (< {MIN_CHARS_PER_COMPLETION}); \
             likely symptom B (model gave up early). text: {text:?}",
            text.len()
        );
    }

    // Symptom C: cross-output verbatim substring.
    for i in 0..outputs.len() {
        for j in (i + 1)..outputs.len() {
            assert!(
                !share_long_substring(&outputs[i], &outputs[j], VERBATIM_WINDOW),
                "completions #{i} and #{j} share a {VERBATIM_WINDOW}-char \
                 verbatim substring; likely symptom C (state pollution).\n\
                 #{i}: {:?}\n#{j}: {:?}",
                outputs[i],
                outputs[j],
            );
        }
    }

    eprintln!("[pollution] PASS — three consecutive completions stayed clean");
}

/// Phase 7 correctness assertion: with prefix cache enabled, a
/// multi-turn chat must produce the SAME output a fresh-session
/// run produces for the equivalent prompt. The cache path is only
/// permitted to skip work, never to change outputs.
///
/// Procedure:
/// 1. Open Session A with `with_prefix_cache(true)`. Run turn 1
///    (system + user1) — primes the cache.
/// 2. On A, run turn 2 (system + user1 + assistant1 + user2). This
///    exercises partial-hit on `[system + user1]`, restores to that
///    breakpoint, and chunked-prefills the new tail. Capture output.
/// 3. Open a fresh Session B with `with_prefix_cache(false)`. Run
///    turn 2 directly (no prior cache). Capture output.
/// 4. Assert outputs match exactly.
///
/// Pre-Phase-7 this test would have failed: the lossy partial
/// truncate would corrupt session A's recurrent state, and turn 2's
/// generation would diverge from B's fresh-session ground truth.
#[test]
#[ignore = "long running; needs moeflux artifacts"]
fn partial_hit_output_matches_fresh_session() {
    use drama_llama::Block;
    use misanthropic::prompt::message::CacheControl;

    fn cached_user(text: &'static str) -> Message {
        Message {
            role: Role::User,
            content: Content::MultiPart(vec![Block::Text {
                text: Cow::Borrowed(text),
                cache_control: Some(CacheControl::ephemeral()),
            }]),
        }
    }

    let system_text =
        "You are a concise assistant. Answer in one short paragraph.";
    let user1_text =
        "Tell me one interesting historical fact about the year 1969.";
    let assistant1_text =
        "Apollo 11 landed on the Moon on July 20, 1969.";
    let user2_text =
        "Now tell me one interesting historical fact about the year 1989.";

    let turn1 = Prompt {
        system: Some(Content::MultiPart(vec![Block::Text {
            text: Cow::Borrowed(system_text),
            cache_control: Some(CacheControl::ephemeral()),
        }])),
        messages: vec![cached_user(user1_text)],
        ..Prompt::default()
    };

    let turn2 = Prompt {
        system: Some(Content::MultiPart(vec![Block::Text {
            text: Cow::Borrowed(system_text),
            cache_control: Some(CacheControl::ephemeral()),
        }])),
        messages: vec![
            cached_user(user1_text),
            Message {
                role: Role::Assistant,
                content: Content::SinglePart(Cow::Borrowed(assistant1_text)),
            },
            cached_user(user2_text),
        ],
        ..Prompt::default()
    };

    // Cached run: A primes cache with turn1, then runs turn2 hitting
    // the partial-hit code path.
    let mut session_a = build_session();
    let _warm = session_a
        .complete_text(&turn1)
        .expect("complete_text turn1 (cached session)");
    let cached_turn2 = session_a
        .complete_text(&turn2)
        .expect("complete_text turn2 (cached session)");

    // Fresh ground-truth run: B has no cache, runs turn2 directly.
    let mut session_b = build_session().with_prefix_cache(false);
    let fresh_turn2 = session_b
        .complete_text(&turn2)
        .expect("complete_text turn2 (fresh session)");

    eprintln!(
        "[partial-hit] cached_turn2 ({} chars) first 80: {}",
        cached_turn2.len(),
        cached_turn2.chars().take(80).collect::<String>(),
    );
    eprintln!(
        "[partial-hit] fresh_turn2 ({} chars) first 80: {}",
        fresh_turn2.len(),
        fresh_turn2.chars().take(80).collect::<String>(),
    );

    assert_eq!(
        cached_turn2, fresh_turn2,
        "partial-hit output diverged from fresh-session ground truth",
    );
}
