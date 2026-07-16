//! Integration tests for llama.cpp state save/load and the
//! per-sequence snapshot path behind [`Decoder::checkpoint_pos`] /
//! [`Decoder::restore_to`].
//!
//! The snapshot path exists for recurrent / hybrid models (whose layer
//! state cannot be rewound by KV truncation) but is testable on any
//! model by forcing it on via `set_seq_snapshots(true)` — snapshots
//! are architecture-agnostic serialized sequence state.
//!
//! All tests load a real model: `cargo test --test state_snapshot --
//! --ignored`.
//!
//! [`Decoder::checkpoint_pos`]: drama_llama::Decoder::checkpoint_pos
//! [`Decoder::restore_to`]: drama_llama::Decoder::restore_to

#![cfg(feature = "llama-cpp")]

use std::{num::NonZeroUsize, path::PathBuf};

use drama_llama::{LlamaCppEngine, PredictOptions, Token};

fn model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf")
}

fn engine() -> LlamaCppEngine {
    LlamaCppEngine::from_path(model_path()).expect("model loads")
}

/// Greedy, seeded, short — identical inputs must yield identical
/// outputs, which is what every test here leans on.
fn greedy_opts(n: usize) -> PredictOptions {
    PredictOptions {
        n: NonZeroUsize::new(n).unwrap(),
        ..PredictOptions::greedy()
    }
}

/// Prefill everything but the last prompt token on seq 0, so a
/// subsequent `predict_tokens_resuming(&[last], len - 1, ...)` decodes
/// that token for fresh logits and samples from there. Returns
/// `(head_len, tail)`.
fn prefill_head(
    engine: &mut LlamaCppEngine,
    tokens: &[Token],
) -> (usize, Vec<Token>) {
    let head_len = tokens.len() - 1;
    engine.memory_clear();
    engine
        .prefill_chunk(&tokens[..head_len], 0, 0)
        .expect("prefill");
    (head_len, vec![tokens[tokens.len() - 1]])
}

fn greedy_continuation(
    engine: &mut LlamaCppEngine,
    tail: Vec<Token>,
    start_pos: usize,
    n: usize,
) -> Vec<Token> {
    engine
        .predict_tokens_resuming(tail, start_pos, 0, greedy_opts(n), None)
        .collect()
}

/// `get_state_seq` bytes restore a sequence to the exact boundary they
/// were taken at: generation after the restore reproduces generation
/// before it, token for token.
#[test]
#[ignore = "long running, requires models/model.gguf"]
fn state_seq_roundtrip_resumes_generation_identically() {
    let mut engine = engine();
    let tokens = engine
        .model
        .tokenize("Once upon a time, in a land far away,", true);
    assert!(tokens.len() >= 2, "prompt must tokenize to >= 2 tokens");

    let (head_len, tail) = prefill_head(&mut engine, &tokens);
    let saved = engine.get_state_seq(0);
    assert_eq!(
        saved.len(),
        engine.state_seq_size(0),
        "get_state_seq length must match state_seq_size"
    );

    let a = greedy_continuation(&mut engine, tail.clone(), head_len, 24);
    assert!(!a.is_empty(), "greedy continuation produced no tokens");

    // KV now holds prompt + generation. Wipe the sequence and restore
    // the boundary snapshot.
    engine.memory_seq_rm(0, -1, -1);
    assert!(
        engine.set_state_seq(&saved, 0),
        "set_state_seq rejected bytes produced by get_state_seq"
    );

    let b = greedy_continuation(&mut engine, tail, head_len, 24);
    assert_eq!(a, b, "generation after state restore diverged");
}

/// With snapshots forced on, `checkpoint_pos` stores a restorable
/// snapshot: after the sequence is wiped outright (simulating exactly
/// the case where a KV truncate cannot rewind — recurrent memory, or
/// unexpected KV loss), `restore_to` falls back to the snapshot and
/// generation resumes identically.
#[test]
#[ignore = "long running, requires models/model.gguf"]
fn forced_snapshot_restores_after_kv_wipe() {
    let mut engine = engine();
    engine.set_seq_snapshots(true);
    assert!(engine.seq_snapshots_enabled());

    let tokens = engine
        .model
        .tokenize("The quick brown fox jumps over the lazy dog. Then", true);
    let (head_len, tail) = prefill_head(&mut engine, &tokens);

    engine.checkpoint_pos(0, head_len as i32);
    assert_eq!(engine.seq_snapshot_count(), 1);

    let a = greedy_continuation(&mut engine, tail.clone(), head_len, 24);

    // Wipe the whole sequence. A plain truncate to head_len can no
    // longer produce a valid prefix — the pos_max guard in restore_to
    // must detect that and reload the snapshot instead.
    engine.memory_seq_rm(0, -1, -1);
    engine
        .restore_to(0, head_len as i32)
        .expect("restore_to should succeed via the stored snapshot");
    assert_eq!(
        engine.memory_seq_pos_max(0),
        head_len as i32 - 1,
        "sequence must hold [0, head_len) after snapshot restore"
    );

    let b = greedy_continuation(&mut engine, tail, head_len, 24);
    assert_eq!(a, b, "generation after snapshot restore diverged");

    // The snapshot survives its own restore — a breakpoint stays
    // restorable across repeated rewinds.
    assert_eq!(engine.seq_snapshot_count(), 1);
}

/// Without a stored snapshot, a rewind onto missing KV must report
/// `NoCheckpoint` (routing Session to its full-reprefill fallback) —
/// not silently "succeed" over an empty sequence.
#[test]
#[ignore = "long running, requires models/model.gguf"]
fn restore_without_snapshot_after_wipe_is_no_checkpoint() {
    use drama_llama::backend::MemoryRmError;

    let mut engine = engine();
    engine.set_seq_snapshots(true);

    let tokens = engine.model.tokenize("Hello there, general", true);
    let (head_len, _tail) = prefill_head(&mut engine, &tokens);

    // No checkpoint_pos call. Wipe and attempt a rewind.
    engine.memory_seq_rm(0, -1, -1);
    match engine.restore_to(0, head_len as i32) {
        Err(MemoryRmError::NoCheckpoint { pos }) => {
            assert_eq!(pos, head_len as i32);
        }
        other => panic!(
            "expected NoCheckpoint for a wiped sequence with no \
             snapshot, got {other:?}"
        ),
    }
}

/// `forget_pos` drops exactly the named snapshot; `memory_clear`
/// drops them all.
#[test]
#[ignore = "long running, requires models/model.gguf"]
fn forget_pos_and_memory_clear_free_snapshots() {
    let mut engine = engine();
    engine.set_seq_snapshots(true);

    let tokens = engine.model.tokenize("Counting: one two three four", true);
    let mid = tokens.len() / 2;
    engine.memory_clear();
    engine.prefill_chunk(&tokens[..mid], 0, 0).expect("prefill");
    engine.checkpoint_pos(0, mid as i32);
    engine
        .prefill_chunk(&tokens[mid..], mid, 0)
        .expect("prefill");
    engine.checkpoint_pos(0, tokens.len() as i32);
    assert_eq!(engine.seq_snapshot_count(), 2);

    engine.forget_pos(0, mid as i32).expect("forget_pos");
    assert_eq!(engine.seq_snapshot_count(), 1);
    // Idempotent on an already-forgotten position.
    engine.forget_pos(0, mid as i32).expect("forget_pos twice");
    assert_eq!(engine.seq_snapshot_count(), 1);

    engine.memory_clear();
    assert_eq!(
        engine.seq_snapshot_count(),
        0,
        "memory_clear must free all snapshots"
    );
}

/// Global (whole-context) state roundtrip — the pre-existing
/// `get_state` / `set_state` surface: restoring resumes generation
/// identically, and sizes agree.
#[test]
#[ignore = "long running, requires models/model.gguf"]
fn global_state_roundtrip_resumes_generation_identically() {
    let mut engine = engine();
    let tokens = engine
        .model
        .tokenize("List three colors of the rainbow:", true);

    let (head_len, tail) = prefill_head(&mut engine, &tokens);
    let saved = engine.get_state();
    assert_eq!(saved.len(), engine.state_size());

    let a = greedy_continuation(&mut engine, tail.clone(), head_len, 16);

    engine.memory_clear();
    engine.set_state(&saved);

    let b = greedy_continuation(&mut engine, tail, head_len, 16);
    assert_eq!(a, b, "generation after global state restore diverged");
}
