//! Phase 4 gate-4: cross-backend regression check.
//!
//! Runs `LlamaCppEngine` and `MoefluxEngine` against the same prompt
//! and the same model (Qwen3.6-35B-A3B; GGUF for llama.cpp, MLX +
//! moeflux artifacts for moeflux), then compares:
//!
//! - **Token-level argmax agreement** ≥ `ARGMAX_AGREEMENT_MIN` of
//!   greedy steps.
//! - **Top-K logit set overlap** ≥ `TOPK_OVERLAP_MIN` averaged across
//!   step 0 and step N (set membership only — order and exact
//!   magnitudes drift between backends).
//!
//! Tolerances come from `plan_v0.8.0_backend_split.md` Phase 4 exit
//! criterion #4. Identical output across backends is unrealistic
//! (numerical noise, different attention kernels, GPU vs Metal MoE);
//! the bar is "same shape, same heads of the distribution."
//!
//! Skipped (#[ignore]) by default — needs ~40 GB of artifacts mounted
//! and both backends linked. Run with:
//!
//! ```bash
//! cargo test --features "llama-cpp,moeflux,moeflux-model-qwen3-6-35b-a3b" \
//!     --test cross_backend -- --ignored --nocapture
//! ```
//!
//! Override default paths via env (matches `moeflux_smoke.rs`):
//! - `DRAMA_LLAMA_MOEFLUX_MLX_DIR`
//! - `DRAMA_LLAMA_MOEFLUX_ARTIFACTS_DIR`
//! - `DRAMA_LLAMA_MOEFLUX_EXPERTS_DIR`
//! - `DRAMA_LLAMA_GGUF_PATH`

#![cfg(all(feature = "llama-cpp", feature = "moeflux", target_os = "macos"))]

use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::path::PathBuf;

use drama_llama::backend::{Decoder, Model};
use drama_llama::{
    Candidates, Engine, LlamaCppEngine, MoefluxEngine, Token, TokenData,
};

/// Same prompt as `regression.rs`; chosen because greedy under it is
/// stable across llama.cpp point releases.
const PROMPT: &str = "The quick brown fox";

/// 32 steps matches the llama.cpp baseline; long enough to catch
/// trajectory drift, short enough that two ~20 GB models in one
/// process don't time out.
const N_STEPS: usize = 32;

/// Top-K window for set-overlap comparison. Wider than 20 dilutes the
/// signal (long tail is essentially noise); narrower than 20 makes a
/// single tied logit swap a step.
const TOP_K: usize = 20;

/// Minimum fraction of `N_STEPS` for which both backends must pick
/// the same argmax token. The plan's bar is ≥0.95.
const ARGMAX_AGREEMENT_MIN: f32 = 0.95;

/// Minimum mean Jaccard overlap of top-K id-sets across the captured
/// step 0 and step N. The plan's bar is ≥0.80.
const TOPK_OVERLAP_MIN: f32 = 0.80;

/// Captured signals from one backend's run. Token stream is the
/// greedy argmax-per-step trajectory of *that* backend (so
/// trajectories diverge after the first disagreement). top-K snapshots
/// are taken from the *prefill logits* (step 0) and the logits
/// returned by the final step decode (step N).
struct Capture {
    n_vocab: i32,
    tokens: Vec<Token>,
    pieces: Vec<String>,
    topk_step_0: Vec<TokenData>,
    topk_step_n: Vec<TokenData>,
}

fn env_path(var: &str, default: &str) -> PathBuf {
    PathBuf::from(std::env::var(var).unwrap_or_else(|_| default.to_string()))
}

/// Sort descending by logit, break ties by ascending id; return the
/// top-K. Mirrors `regression.rs::partial_top_k` for consistency.
fn partial_top_k(candidates: &Candidates, k: usize) -> Vec<TokenData> {
    let mut buf: Vec<TokenData> = candidates.as_slice().to_vec();
    let k = k.min(buf.len());
    buf.sort_by(|a, b| {
        b.logit
            .partial_cmp(&a.logit)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });
    buf.truncate(k);
    buf
}

/// Generic capture loop: works for any `Engine<D, M>`. Tokenization
/// uses `add_special=false` so both backends see identical prompt
/// tokens (BOS handling differs — moeflux's HF tokenizer would prefix
/// nothing for Qwen, llama.cpp's gguf might add BOS). The returned
/// trajectory is each backend's own greedy continuation.
fn capture<D, M>(engine: &mut Engine<D, M>) -> Capture
where
    D: Decoder,
    M: Model + Sync,
{
    let prompt_tokens = engine.model.tokenize(PROMPT, false);
    let n_vocab = engine.model.n_vocab();

    let mut topk_step_0: Option<Vec<TokenData>> = None;
    let mut topk_step_n: Option<Vec<TokenData>> = None;
    let mut tokens: Vec<Token> = Vec::with_capacity(N_STEPS);

    {
        let mut iter = engine.predict_candidates(
            prompt_tokens.clone(),
            NonZeroUsize::new(N_STEPS).unwrap(),
        );

        let mut step = 0usize;
        while let Some(candidates) = iter.next() {
            let top = partial_top_k(&candidates, TOP_K);
            let chosen = top[0].id;
            if step == 0 {
                topk_step_0 = Some(top.clone());
            }
            if step + 1 == N_STEPS {
                topk_step_n = Some(top.clone());
            }
            tokens.push(chosen);
            iter.record_choice(chosen);
            step += 1;
        }
    }

    let pieces: Vec<String> = tokens
        .iter()
        .map(|&t| engine.model.token_to_piece(t))
        .collect();

    Capture {
        n_vocab,
        tokens,
        pieces,
        topk_step_0: topk_step_0.expect("step 0 never captured"),
        topk_step_n: topk_step_n.expect("step N never captured"),
    }
}

fn jaccard(a: &[TokenData], b: &[TokenData]) -> f32 {
    let sa: HashSet<Token> = a.iter().map(|t| t.id).collect();
    let sb: HashSet<Token> = b.iter().map(|t| t.id).collect();
    let inter = sa.intersection(&sb).count() as f32;
    let union = sa.union(&sb).count() as f32;
    if union == 0.0 {
        1.0
    } else {
        inter / union
    }
}

fn dump_topk(label: &str, top: &[TokenData], n: usize) {
    eprintln!("  {label} (top {}):", n.min(top.len()));
    for (i, td) in top.iter().take(n).enumerate() {
        eprintln!("    {i:>2}. id={:>6} logit={:>9.4}", td.id, td.logit);
    }
}

#[test]
#[ignore = "long running; needs both backends + 35B-A3B artifacts"]
fn moeflux_matches_llama_cpp() {
    let gguf = env_path(
        "DRAMA_LLAMA_GGUF_PATH",
        "/Volumes/Temp Backup/models/gguf/qwen3-6-35b-a3b-q4_k_m.gguf",
    );
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

    eprintln!("== llama.cpp side ==");
    eprintln!("loading {gguf:?}");
    let llama_capture = {
        let mut engine =
            LlamaCppEngine::from_path(gguf.clone()).unwrap().quiet();
        eprintln!("  n_vocab={}", engine.model.n_vocab());
        eprintln!("  n_ctx={}", engine.n_ctx());
        capture(&mut engine)
    };

    eprintln!("\n== moeflux side ==");
    eprintln!("loading from {mlx_dir:?} + {artifacts_dir:?}");
    let moeflux_capture = {
        let mut engine = MoefluxEngine::from_paths(
            &mlx_dir,
            &artifacts_dir,
            &experts_dir,
            8,
            false,
        )
        .expect("MoefluxEngine::from_paths");
        eprintln!("  n_vocab={}", engine.model.n_vocab());
        eprintln!("  n_ctx={}", engine.n_ctx());
        capture(&mut engine)
    };

    eprintln!("\n== comparison ==");
    assert_eq!(
        llama_capture.n_vocab, moeflux_capture.n_vocab,
        "n_vocab differs ({} vs {}) — backends loaded incompatible models",
        llama_capture.n_vocab, moeflux_capture.n_vocab
    );

    // Argmax agreement: pairwise across the trajectory. Both arrays
    // are length N_STEPS; matches even when the tokens chosen drive
    // the two backends down different paths after step 0.
    let argmax_matches = llama_capture
        .tokens
        .iter()
        .zip(moeflux_capture.tokens.iter())
        .filter(|(a, b)| a == b)
        .count();
    let argmax_frac = argmax_matches as f32 / N_STEPS as f32;
    eprintln!(
        "argmax agreement: {argmax_matches}/{N_STEPS} = {argmax_frac:.3} \
         (min {ARGMAX_AGREEMENT_MIN:.2})"
    );

    // Top-K set overlap (Jaccard). Average step 0 and step N — step
    // 0 measures the prefill agreement; step N measures whether
    // accumulated divergence wrecks the head of the distribution.
    let overlap_0 =
        jaccard(&llama_capture.topk_step_0, &moeflux_capture.topk_step_0);
    let overlap_n =
        jaccard(&llama_capture.topk_step_n, &moeflux_capture.topk_step_n);
    let overlap_mean = (overlap_0 + overlap_n) / 2.0;
    eprintln!(
        "top-{TOP_K} jaccard overlap: step0={overlap_0:.3}, \
         stepN={overlap_n:.3}, mean={overlap_mean:.3} \
         (min {TOPK_OVERLAP_MIN:.2})"
    );

    // Side-by-side diagnostic dump. Always print on success too — the
    // numbers are useful for tracking drift across moeflux changes.
    eprintln!("\n  llama.cpp tokens: {:?}", llama_capture.tokens);
    eprintln!("  llama.cpp pieces: {:?}", llama_capture.pieces);
    eprintln!("  moeflux   tokens: {:?}", moeflux_capture.tokens);
    eprintln!("  moeflux   pieces: {:?}", moeflux_capture.pieces);
    dump_topk("llama.cpp step 0", &llama_capture.topk_step_0, 10);
    dump_topk("moeflux   step 0", &moeflux_capture.topk_step_0, 10);

    assert!(
        argmax_frac >= ARGMAX_AGREEMENT_MIN,
        "argmax agreement {argmax_frac:.3} below threshold \
         {ARGMAX_AGREEMENT_MIN:.2}"
    );
    assert!(
        overlap_mean >= TOPK_OVERLAP_MIN,
        "top-{TOP_K} mean Jaccard overlap {overlap_mean:.3} below \
         threshold {TOPK_OVERLAP_MIN:.2}"
    );
}
