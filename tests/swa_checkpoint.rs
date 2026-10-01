//! Checkpoint rewinds on the models a KV truncate cannot rewind alone:
//! sliding-window attention (gpt-oss, Gemma 4) and hybrid recurrent
//! (Qwen3.6, Qwen3.8). The unit tests in `llama_cpp::checkpoint` pin the
//! rules against a simulated cache; these check llama.cpp agrees, on
//! real weights.
//!
//! Every test loads a model onto the GPU, so all are `#[ignore]`d:
//! `just test swa_checkpoint`. Models, each skipped when absent:
//!
//! - sliding window: `$DRAMA_LLAMA_SWA_MODEL`, else
//!   `models/gpt-oss-120b-MXFP4.gguf` (try Gemma 4 too:
//!   `models/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf`);
//! - hybrid: `$DRAMA_LLAMA_HYBRID_MODEL`, else `models/model.gguf`.

#![cfg(feature = "llama-cpp")]

use std::{num::NonZeroUsize, path::PathBuf};

use drama_llama::{
    backend::MemoryRmError, Checkpointing, LlamaCppEngine, LlamaCppOptions,
    PredictOptions, Token,
};

fn resolve(var: &str, default: &str) -> Option<PathBuf> {
    let path = std::env::var_os(var).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(default)
    });
    if path.exists() {
        Some(path)
    } else {
        eprintln!("SKIP: no model at {} (set {var})", path.display());
        None
    }
}

/// Two sequences over one unified pool — the serving shape
/// (`--cache-slots`), where neighbours recycle each other's masked
/// window cells.
fn engine(path: PathBuf) -> LlamaCppEngine {
    LlamaCppEngine::from_path_with(
        path,
        LlamaCppOptions::default()
            .with_n_ctx(8192)
            .with_cache_slots(2),
    )
    .expect("engine loads")
}

fn swa_engine() -> Option<LlamaCppEngine> {
    let engine = engine(resolve(
        "DRAMA_LLAMA_SWA_MODEL",
        "models/gpt-oss-120b-MXFP4.gguf",
    )?);
    assert!(engine.model().n_swa() > 0, "not a sliding-window model");
    assert_eq!(engine.checkpointing(), Checkpointing::Partial);
    Some(engine)
}

fn hybrid_engine() -> Option<LlamaCppEngine> {
    let engine =
        engine(resolve("DRAMA_LLAMA_HYBRID_MODEL", "models/model.gguf")?);
    assert!(engine.model().is_hybrid(), "not a hybrid model");
    assert_eq!(engine.checkpointing(), Checkpointing::Partial);
    Some(engine)
}

/// A prompt several windows long: the anchor's window then lies well
/// below where the sequence ends up.
fn long_prompt(engine: &LlamaCppEngine, min_len: usize) -> Vec<Token> {
    let line = "The quick brown fox jumps over the lazy dog, and then \
                counts the stars one by one until the morning comes. ";
    let mut text = String::from("Here is a story. ");
    while engine.model().tokenize(&text, true).len() < min_len {
        text.push_str(line);
    }
    text.push_str("What did the fox count?");
    engine.model().tokenize(&text, true)
}

fn greedy(
    engine: &mut LlamaCppEngine,
    tail: Token,
    pos: usize,
    seq: i32,
) -> Vec<Token> {
    let opts = PredictOptions {
        n: NonZeroUsize::new(24).unwrap(),
        ..PredictOptions::greedy()
    };
    engine
        .predict_tokens_resuming(vec![tail], pos, seq, opts, None)
        .collect()
}

/// Prefill all but the last prompt token on `seq` and checkpoint there:
/// the anchor sits at `prompt.len() - 1`.
fn prefill_and_checkpoint(
    engine: &mut LlamaCppEngine,
    prompt: &[Token],
    seq: i32,
) -> usize {
    let anchor = prompt.len() - 1;
    engine
        .prefill_chunk(&prompt[..anchor], 0, seq)
        .expect("prefill");
    engine.checkpoint_pos(seq, anchor as i32);
    anchor
}

/// The live gpt-oss miss: a slot generates past its anchor, a
/// neighbour decodes enough to recycle the window below it, and the
/// slot rewinds to the anchor. The truncate alone cannot restore that
/// window; the checkpoint must, and generation must match the run
/// before the rewind token for token.
#[test]
#[ignore = "requires a sliding-window model on the GPU"]
fn swa_rewind_survives_a_neighbour_recycling_the_window() {
    let Some(mut engine) = swa_engine() else {
        return;
    };
    let n_swa = engine.model().n_swa() as usize;
    let prompt = long_prompt(&engine, 3 * n_swa);
    let anchor = prefill_and_checkpoint(&mut engine, &prompt, 0);
    let first = greedy(&mut engine, prompt[anchor], anchor, 0);
    assert!(!first.is_empty());

    let neighbour = long_prompt(&engine, 6 * n_swa + 1024);
    engine.memory_seq_rm(1, -1, -1);
    engine.prefill_chunk(&neighbour, 0, 1).expect("neighbour");

    engine
        .restore_to(0, anchor as i32)
        .expect("the checkpoint restores the anchor");
    assert_eq!(engine.memory_seq_pos_max(0), anchor as i32 - 1);
    let again = greedy(&mut engine, prompt[anchor], anchor, 0);
    assert_eq!(first, again, "generation after the rewind diverged");
}

/// The control, and the silent half of the bug: without a checkpoint,
/// the same rewind must be *refused* — not reported as a lossless
/// truncate over a window with a hole in it.
#[test]
#[ignore = "requires a sliding-window model on the GPU"]
fn swa_truncate_never_claims_a_partial_window() {
    let Some(mut engine) = swa_engine() else {
        return;
    };
    engine.set_seq_snapshots(false);
    let n_swa = engine.model().n_swa() as usize;
    let prompt = long_prompt(&engine, 3 * n_swa);
    let anchor = prefill_and_checkpoint(&mut engine, &prompt, 0);
    assert_eq!(engine.seq_snapshot_count(), 0);
    greedy(&mut engine, prompt[anchor], anchor, 0);
    let neighbour = long_prompt(&engine, 6 * n_swa + 1024);
    engine.memory_seq_rm(1, -1, -1);
    engine.prefill_chunk(&neighbour, 0, 1).expect("neighbour");

    assert_eq!(
        engine.restore_to(0, anchor as i32),
        Err(MemoryRmError::NoCheckpoint { pos: anchor as i32 }),
    );
}

/// A rewind to the head needs no checkpoint and costs no load: the
/// window there is whole.
#[test]
#[ignore = "requires a sliding-window model on the GPU"]
fn swa_rewind_to_the_head_is_a_plain_truncate() {
    let Some(mut engine) = swa_engine() else {
        return;
    };
    engine.set_seq_snapshots(false);
    let prompt = long_prompt(&engine, 512);
    let anchor = prompt.len() - 1;
    engine
        .prefill_chunk(&prompt[..anchor], 0, 0)
        .expect("prefill");
    engine
        .restore_to(0, anchor as i32)
        .expect("the head is intact");
}

/// Hybrid: the recurrent state comes back from a partial checkpoint
/// (the recurrent layers only), the attention KV from the truncate —
/// and generation matches. Two anchors, restored top-down, as the
/// session's ladder walks them.
#[test]
#[ignore = "requires a hybrid model on the GPU"]
fn hybrid_partial_checkpoints_rewind_each_anchor() {
    let Some(mut engine) = hybrid_engine() else {
        return;
    };
    let prompt = long_prompt(&engine, 600);
    let low = prompt.len() / 2;
    engine.prefill_chunk(&prompt[..low], 0, 0).expect("prefill");
    engine.checkpoint_pos(0, low as i32);
    let high = prefill_rest(&mut engine, &prompt, low);
    let at_high = greedy(&mut engine, prompt[high], high, 0);

    engine.restore_to(0, high as i32).expect("high anchor");
    assert_eq!(greedy(&mut engine, prompt[high], high, 0), at_high);

    engine.restore_to(0, low as i32).expect("low anchor");
    assert_eq!(
        engine.restore_to(0, high as i32),
        Err(MemoryRmError::NoCheckpoint { pos: high as i32 }),
        "the high anchor's future is gone after rewinding below it",
    );
    let again = prefill_rest(&mut engine, &prompt, low);
    assert_eq!(again, high);
    assert_eq!(greedy(&mut engine, prompt[high], high, 0), at_high);
}

/// Prefill from `from` to the last prompt token and checkpoint there.
fn prefill_rest(
    engine: &mut LlamaCppEngine,
    prompt: &[Token],
    from: usize,
) -> usize {
    let high = prompt.len() - 1;
    engine
        .prefill_chunk(&prompt[from..high], from, 0)
        .expect("prefill");
    engine.checkpoint_pos(0, high as i32);
    high
}

/// A partial checkpoint dies with the KV under it: wipe the sequence,
/// and the anchor must be unrestorable rather than graft the old
/// recurrent state onto whatever is decoded next.
#[test]
#[ignore = "requires a hybrid model on the GPU"]
fn hybrid_checkpoint_dies_with_its_sequence() {
    let Some(mut engine) = hybrid_engine() else {
        return;
    };
    let prompt = long_prompt(&engine, 200);
    let anchor = prefill_and_checkpoint(&mut engine, &prompt, 0);
    assert_eq!(engine.seq_snapshot_count(), 1);
    engine.memory_seq_rm(0, -1, -1);
    assert_eq!(engine.seq_snapshot_count(), 0);
    assert_eq!(
        engine.restore_to(0, anchor as i32),
        Err(MemoryRmError::NoCheckpoint { pos: anchor as i32 }),
    );
}
