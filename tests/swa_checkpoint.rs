//! Checkpoint rewinds on the models a KV truncate cannot rewind alone:
//! sliding-window attention (gpt-oss, Gemma 4) and hybrid recurrent
//! (Qwen3.6, Qwen3.8). The unit tests in `llama_cpp::checkpoint` pin the
//! rules against a simulated cache; these check llama.cpp agrees, on
//! real weights.
//!
//! Every test loads a model's weights, so all are `#[ignore]`d:
//! `just test swa_checkpoint`. Models are found under
//! `$DRAMA_LLAMA_MODEL_DIR`, else `models/`, each skipped when absent:
//!
//! - sliding window: `$DRAMA_LLAMA_SWA_MODEL`, else
//!   `gpt-oss-120b-MXFP4.gguf` (try Gemma 4 too:
//!   `gemma-4-31B-it-qat-UD-Q4_K_XL.gguf`);
//! - hybrid: `$DRAMA_LLAMA_HYBRID_MODEL`, else `model.gguf`;
//! - dense: `cogito-32b.gguf` and
//!   `Mistral-Small-4-119B-2603-UD-Q4_K_XL.gguf`.

#![cfg(feature = "llama-cpp")]

use std::{num::NonZeroUsize, path::PathBuf};

use drama_llama::{
    backend::MemoryRmError, CheckpointBudget, Checkpointing, LlamaCppEngine,
    LlamaCppModel, LlamaCppOptions, PredictOptions, Token,
};

fn model_dir() -> PathBuf {
    std::env::var_os("DRAMA_LLAMA_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models")
        })
}

fn resolve(var: &str, default: &str) -> Option<PathBuf> {
    let path = std::env::var_os(var)
        .map(PathBuf::from)
        .unwrap_or_else(|| model_dir().join(default));
    if path.exists() {
        Some(path)
    } else {
        eprintln!("SKIP: no model at {} (set {var})", path.display());
        None
    }
}

/// The sliding window a GGUF declares, from its metadata. A
/// `vocab_only` load (CPU, no tensors) is enough for the metadata, but
/// skips hyperparameters, so `n_swa()` reads `0` on it.
fn gguf_window(path: &std::path::Path) -> usize {
    let mut params = LlamaCppOptions::default().model_params();
    params.vocab_only = true;
    params.n_gpu_layers = 0;
    let model = LlamaCppModel::from_file(path.to_path_buf(), Some(params))
        .expect("vocab-only load");
    let arch = model.get_meta("general.architecture").expect("an arch");
    let key = format!("{arch}.attention.sliding_window");
    model
        .get_meta(key.as_str())
        .map_or(0, |w| w.parse().expect("an integer"))
}

/// How many windows long the sliding-window tests' prompt is: the
/// anchor's window then lies well below where the sequence ends up.
const PROMPT_WINDOWS: usize = 3;

/// The neighbour that recycles the window below the anchor.
fn neighbour_len(n_swa: usize) -> usize {
    6 * n_swa + 1024
}

/// Two sequences over one unified pool of `n_ctx` cells — the serving
/// shape (`--cache-slots`), where neighbours recycle each other's
/// masked window cells.
fn engine(path: PathBuf, n_ctx: u32) -> LlamaCppEngine {
    LlamaCppEngine::from_path_with(
        path,
        LlamaCppOptions::default()
            .with_n_ctx(n_ctx)
            .with_cache_slots(2),
    )
    .expect("engine loads")
}

/// Sized from the window: the pool must hold the prompt, its
/// generation and the neighbour at once (Gemma 4's 1024-token window
/// needs ≈ 10.3k cells, where a flat 8192 failed the neighbour).
fn swa_engine() -> Option<LlamaCppEngine> {
    let path = resolve("DRAMA_LLAMA_SWA_MODEL", "gpt-oss-120b-MXFP4.gguf")?;
    let n_swa = gguf_window(&path);
    assert!(n_swa > 0, "not a sliding-window model");
    let cells = PROMPT_WINDOWS * n_swa + neighbour_len(n_swa) + 2048;
    let n_ctx = cells.next_multiple_of(1024).max(8192) as u32;
    let engine = engine(path, n_ctx);
    assert_eq!(engine.model().n_swa() as usize, n_swa);
    assert_eq!(engine.checkpointing(), Checkpointing::Partial);
    Some(engine)
}

fn hybrid_engine() -> Option<LlamaCppEngine> {
    let engine =
        engine(resolve("DRAMA_LLAMA_HYBRID_MODEL", "model.gguf")?, 8192);
    assert!(engine.model().is_hybrid(), "not a hybrid model");
    assert_eq!(engine.checkpointing(), Checkpointing::Partial);
    Some(engine)
}

/// A prompt of at least `min_len` tokens.
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
    let prompt = long_prompt(&engine, PROMPT_WINDOWS * n_swa);
    let anchor = prefill_and_checkpoint(&mut engine, &prompt, 0);
    let first = greedy(&mut engine, prompt[anchor], anchor, 0);
    assert!(!first.is_empty());

    let neighbour = long_prompt(&engine, neighbour_len(n_swa));
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
    let prompt = long_prompt(&engine, PROMPT_WINDOWS * n_swa);
    let anchor = prefill_and_checkpoint(&mut engine, &prompt, 0);
    assert_eq!(engine.seq_snapshot_count(), 0);
    greedy(&mut engine, prompt[anchor], anchor, 0);
    let neighbour = long_prompt(&engine, neighbour_len(n_swa));
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

/// An anchor restored by truncation, with no checkpoint stored, gets
/// one there and then: after the neighbour recycles its window, the
/// same anchor restores again — from that checkpoint — and generation
/// matches.
#[test]
#[ignore = "requires a sliding-window model on the GPU"]
fn swa_truncate_rewind_takes_the_missing_checkpoint() {
    let Some(mut engine) = swa_engine() else {
        return;
    };
    let n_swa = engine.model().n_swa() as usize;
    let prompt = long_prompt(&engine, PROMPT_WINDOWS * n_swa);
    let anchor = prompt.len() - 1;
    engine
        .prefill_chunk(&prompt[..anchor], 0, 0)
        .expect("prefill");
    assert_eq!(engine.seq_snapshot_count(), 0);
    engine
        .restore_to(0, anchor as i32)
        .expect("the head is intact");
    assert_eq!(engine.seq_snapshot_count(), 1, "taken on the way");

    let first = greedy(&mut engine, prompt[anchor], anchor, 0);
    let neighbour = long_prompt(&engine, neighbour_len(n_swa));
    engine.memory_seq_rm(1, -1, -1);
    engine.prefill_chunk(&neighbour, 0, 1).expect("neighbour");
    engine
        .restore_to(0, anchor as i32)
        .expect("the checkpoint taken by the truncate restores it");
    assert_eq!(greedy(&mut engine, prompt[anchor], anchor, 0), first);
}

/// A checkpoint is the window (or recurrent state), not the prefix,
/// and the byte budget bounds what is held: a budget below one
/// checkpoint holds none. Prints the size — Gemma 4's is the one the
/// default budget is sized around (≈ 800 MiB).
#[test]
#[ignore = "requires a sliding-window model on the GPU"]
fn swa_checkpoints_are_window_sized_and_budgeted() {
    let Some(mut engine) = swa_engine() else {
        return;
    };
    let n_swa = engine.model().n_swa() as usize;
    let prompt = long_prompt(&engine, PROMPT_WINDOWS * n_swa);
    let started = std::time::Instant::now();
    let anchor = prefill_and_checkpoint(&mut engine, &prompt, 0);
    let bytes = engine.seq_snapshot_bytes();
    eprintln!(
        "n_swa {n_swa}: checkpoint at {anchor} is {bytes} bytes \
         ({:.1} MiB); prefill + checkpoint {:?}",
        bytes as f64 / (1 << 20) as f64,
        started.elapsed(),
    );
    assert_eq!(engine.seq_snapshot_count(), 1);
    // The window's cells, not the prefix's dense KV as well.
    let whole = engine.state_seq_size(0);
    assert!(bytes < whole, "{bytes} bytes, the whole sequence {whole}");

    engine.set_checkpoint_budget(CheckpointBudget::new(bytes - 1, bytes - 1));
    assert_eq!(engine.seq_snapshot_count(), 0, "evicted to the budget");
    engine.checkpoint_pos(0, anchor as i32);
    assert_eq!(engine.seq_snapshot_count(), 0, "too large to keep");
    engine.set_checkpoint_budget(CheckpointBudget::default());
    engine.checkpoint_pos(0, anchor as i32);
    assert_eq!(engine.seq_snapshot_bytes(), bytes);
}

/// Dense attention keeps every position, so the dense fleet models
/// take no checkpoints on a real load — the decoder's own `n_swa`,
/// `is_hybrid` and `is_recurrent`, not GGUF metadata, decide that.
/// CPU-only, so the weights stay memory-mapped rather than resident on
/// Metal: Mistral 4 alone is 74 GB, and nothing here runs the model.
#[test]
#[ignore = "loads the dense fleet models (cogito, Mistral 4), CPU-only"]
fn dense_models_stay_off() {
    let mut checked = 0;
    for file in [
        "cogito-32b.gguf",
        "Mistral-Small-4-119B-2603-UD-Q4_K_XL.gguf",
    ] {
        let path = model_dir().join(file);
        if !path.exists() {
            eprintln!("SKIP: no model at {}", path.display());
            continue;
        }
        let engine = LlamaCppEngine::from_path_with(
            path,
            LlamaCppOptions::default().cpu_only().with_n_ctx(512),
        )
        .expect("engine loads");
        assert_eq!(engine.model().n_swa(), 0, "{file}");
        assert_eq!(engine.checkpointing(), Checkpointing::Off, "{file}");
        assert!(!engine.seq_snapshots_enabled(), "{file}");
        checked += 1;
    }
    assert!(
        checked > 0,
        "no dense model under {}",
        model_dir().display()
    );
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
