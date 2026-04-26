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
//! The bug lives in moeflux: per `consecutive_eval_prompt.rs` in the
//! moeflux crate, `mf_memory_seq_rm` partial-truncate does not reset
//! linear-attention layer state. drama_llama's `Session` calls
//! `memory_seq_rm(0, l_hit, -1)` + `eval_prompt(suffix, l_hit)` on
//! every cached call, hitting the bug. This test stays red until the
//! upstream fix lands; once green, it locks in the contract.
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
