//! Smoke test for the moeflux backend.
//!
//! Loads `MoefluxEngine` against the 35B-A3B artifacts on Mike's
//! `/Volumes/Temp Backup` mount, runs a single prefill + 4 greedy
//! steps, and asserts the decoder produces logits of the right shape
//! and the tokenizer round-trips. This is the Phase 4 gate-4 entry
//! point — if this passes the cross-backend regression test can run.
//!
//! Skipped (#[ignore]) by default since it needs ~17 GB of expert
//! shards mounted. Enable with:
//!
//! ```bash
//! cargo test --features "moeflux,moeflux-model-qwen3-6-35b-a3b" \
//!     --test moeflux_smoke -- --ignored --nocapture
//! ```
//!
//! Override default paths via env:
//! - `DRAMA_LLAMA_MOEFLUX_MLX_DIR`
//! - `DRAMA_LLAMA_MOEFLUX_ARTIFACTS_DIR`
//! - `DRAMA_LLAMA_MOEFLUX_EXPERTS_DIR`

#![cfg(all(feature = "moeflux", target_os = "macos"))]

use std::num::NonZeroUsize;
use std::path::PathBuf;

use drama_llama::backend::Model;
use drama_llama::MoefluxEngine;

fn env_path(var: &str, default: &str) -> PathBuf {
    PathBuf::from(std::env::var(var).unwrap_or_else(|_| default.to_string()))
}

#[test]
#[ignore = "long running; requires moeflux artifacts on /Volumes/Temp Backup"]
fn moeflux_loads_and_predicts() {
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

    eprintln!("loading moeflux engine...");
    let mut engine = MoefluxEngine::from_paths(
        &mlx_dir,
        &artifacts_dir,
        &experts_dir,
        8,     // experts_per_tok — Qwen3 MoE default top-K
        false, // use_2bit — stick with 4-bit
    )
    .expect("MoefluxEngine::from_paths failed");

    let n_vocab = engine.model.n_vocab();
    eprintln!("loaded: n_vocab={n_vocab}, n_ctx={}", engine.n_ctx());
    assert_eq!(
        n_vocab, 248320,
        "Qwen3.6-35B-A3B should report vocab_size=248320"
    );
    assert!(engine.n_ctx() > 0);

    // Tokenization round-trip via the HF tokenizer.
    let prompt = "The quick brown fox";
    let tokens = engine.model.tokenize(prompt, false);
    eprintln!("tokenized {prompt:?} -> {tokens:?}");
    assert!(!tokens.is_empty());
    let detok: String =
        tokens.iter().map(|&t| engine.model.token_to_piece(t)).collect();
    assert!(
        detok.contains("quick brown fox"),
        "detokenized text should contain the input substring (got {detok:?})"
    );

    // Single prefill + a few greedy steps. Logits must be exactly
    // n_vocab long; argmax must be a valid token id.
    let mut iter =
        engine.predict_candidates(tokens, NonZeroUsize::new(4).unwrap());

    let mut decoded: Vec<i32> = Vec::new();
    while let Some(candidates) = iter.next() {
        assert_eq!(
            candidates.as_slice().len(),
            n_vocab as usize,
            "candidates buffer size should match n_vocab"
        );
        let chosen = candidates.sample_token_greedy().is_one().unwrap();
        assert!(chosen.id >= 0 && chosen.id < n_vocab);
        decoded.push(chosen.id);
        iter.record_choice(chosen.id);
    }

    eprintln!("decoded 4 greedy tokens: {decoded:?}");
    let pieces: Vec<String> = decoded
        .iter()
        .map(|&t| engine.model.token_to_piece(t))
        .collect();
    eprintln!("pieces: {pieces:?}");
    assert_eq!(decoded.len(), 4);
}
