//! Golden-output regression harness for the `llama-cpp` backend.
//!
//! Captures a fixed set of deterministic signals from a known model on a
//! known prompt and compares them against a checked-in golden file. The
//! goal is to catch silent regressions when the trait layer or sampling
//! loop is refactored, and to give Phase 4 a shape to compare flash-moe
//! output against.
//!
//! Signals captured per run:
//!
//! - `prompt_tokens` — BPE tokenization of the probe prompt.
//! - `tokens` — token stream under greedy sampling for `N_STEPS`.
//! - `logits_step_0` — top-K vocab ids + logits after the prefill.
//! - `logits_step_n` — top-K after the final step.
//!
//! Token streams must match byte-for-byte (greedy is deterministic).
//! Top-K entries must match by id + order (stable across builds) with
//! logit magnitudes within `LOGIT_TOL` (rebuilds with different SIMD /
//! backend flags shift magnitudes but not argmax).
//!
//! Running:
//!
//! ```bash
//! # Compare against committed golden.
//! cargo test --features "cli" --test regression -- --ignored
//!
//! # Regenerate the golden after an intentional change.
//! DRAMA_LLAMA_UPDATE_GOLDEN=1 cargo test --features "cli" \
//!     --test regression -- --ignored
//! ```

#![cfg(feature = "llama-cpp")]

use std::fs;
use std::num::NonZeroUsize;
use std::path::PathBuf;

use drama_llama::{Candidates, LlamaCppEngine, Token, TokenData};
use serde::{Deserialize, Serialize};

/// Prompt chosen to (a) exercise BOS-prepending tokenization and (b)
/// give the model a clear greedy trajectory so the token stream is
/// stable across llama.cpp point releases.
const PROMPT: &str = "The quick brown fox";

/// Number of sampling steps. 32 is enough to expose drift in the
/// decoder / sampler loop without making the test slow on a CPU-only
/// build.
const N_STEPS: usize = 32;

/// Top-K logits captured per step. 20 is wide enough that argmax
/// changes show up in the upper entries; narrow enough that float
/// tolerance noise on tail entries doesn't dominate failures.
const TOP_K: usize = 20;

/// Absolute tolerance on logit magnitudes when comparing against the
/// golden. Order-of-magnitude for llama.cpp's Metal path across
/// rebuilds; tighten later if we pin a build.
const LOGIT_TOL: f32 = 1e-2;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct TopK {
    token: Token,
    piece: String,
    logit: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelMeta {
    n_vocab: i32,
    architecture: Option<String>,
    context_size: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Golden {
    backend: String,
    model_meta: ModelMeta,
    prompt: String,
    prompt_tokens: Vec<Token>,
    n_steps: usize,
    top_k: usize,
    tokens: Vec<Token>,
    pieces: Vec<String>,
    logits_step_0: Vec<TopK>,
    logits_step_n: Vec<TopK>,
}

/// Sort descending by logit, break ties by ascending id. Returns the
/// top-K as a `Vec<TokenData>` so the caller can read both argmax and
/// the full top-K from the same snapshot.
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

fn capture() -> Golden {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf");
    let mut engine = LlamaCppEngine::from_path(path).unwrap().quiet();

    let prompt_tokens = engine.model.tokenize(PROMPT, true);
    let meta = ModelMeta {
        n_vocab: engine.model.n_vocab(),
        architecture: engine.model.get_meta("general.architecture"),
        context_size: engine.model.context_size(),
    };

    let mut step_0_raw: Option<Vec<TokenData>> = None;
    let mut step_n_raw: Option<Vec<TokenData>> = None;
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
                step_0_raw = Some(top.clone());
            }
            if step + 1 == N_STEPS {
                step_n_raw = Some(top.clone());
            }
            tokens.push(chosen);
            iter.record_choice(chosen);
            step += 1;
        }
    }

    // With the iterator dropped, the engine is available again — decode
    // pieces for the captured token ids via the model.
    let topk_to_entries = |raw: Vec<TokenData>| -> Vec<TopK> {
        raw.into_iter()
            .map(|td| TopK {
                token: td.id,
                piece: engine.model.token_to_piece(td.id),
                logit: td.logit,
            })
            .collect()
    };

    let logits_step_0 = topk_to_entries(step_0_raw.expect("step 0 never captured"));
    let logits_step_n = topk_to_entries(step_n_raw.expect("step N never captured"));
    let pieces: Vec<String> = tokens
        .iter()
        .map(|&t| engine.model.token_to_piece(t))
        .collect();

    Golden {
        backend: "llama-cpp".to_string(),
        model_meta: meta,
        prompt: PROMPT.to_string(),
        prompt_tokens,
        n_steps: N_STEPS,
        top_k: TOP_K,
        tokens,
        pieces,
        logits_step_0,
        logits_step_n,
    }
}

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/regression/llama_cpp_baseline.json")
}

fn assert_topk_close(label: &str, got: &[TopK], want: &[TopK]) {
    assert_eq!(
        got.len(),
        want.len(),
        "{label}: top-K length mismatch ({} vs {})",
        got.len(),
        want.len()
    );
    for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            a.token, b.token,
            "{label}[{i}]: token id mismatch (got {} `{}`, want {} `{}`)",
            a.token, a.piece, b.token, b.piece
        );
        let diff = (a.logit - b.logit).abs();
        assert!(
            diff <= LOGIT_TOL,
            "{label}[{i}] (token {}): logit drift {} > tol {} (got {}, want {})",
            a.token, diff, LOGIT_TOL, a.logit, b.logit
        );
    }
}

#[test]
#[ignore = "long running; requires models/model.gguf"]
fn regression_llama_cpp_baseline() {
    let current = capture();
    let path = golden_path();

    let update = std::env::var("DRAMA_LLAMA_UPDATE_GOLDEN")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    if update || !path.exists() {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create fixture dir");
        }
        let serialized =
            serde_json::to_string_pretty(&current).expect("serialize golden");
        fs::write(&path, serialized).expect("write golden");
        eprintln!(
            "wrote golden at {} ({} prompt tokens, {} steps)",
            path.display(),
            current.prompt_tokens.len(),
            current.tokens.len()
        );
        return;
    }

    let raw = fs::read_to_string(&path).expect("read golden");
    let golden: Golden = serde_json::from_str(&raw).expect("parse golden");

    assert_eq!(current.backend, golden.backend, "backend name changed");
    assert_eq!(
        current.model_meta.n_vocab, golden.model_meta.n_vocab,
        "model identity changed (n_vocab mismatch) — regenerate with \
         DRAMA_LLAMA_UPDATE_GOLDEN=1 if this is intentional"
    );
    assert_eq!(
        current.prompt_tokens, golden.prompt_tokens,
        "prompt tokenization drifted"
    );
    assert_eq!(
        current.tokens, golden.tokens,
        "greedy token stream drifted"
    );
    assert_eq!(current.pieces, golden.pieces, "decoded pieces drifted");
    assert_topk_close(
        "logits_step_0",
        &current.logits_step_0,
        &golden.logits_step_0,
    );
    assert_topk_close(
        "logits_step_n",
        &current.logits_step_n,
        &golden.logits_step_n,
    );
}
