//! Coherence-test worker for the moeflux-backed engine.
//!
//! Invoked once per `(prompt_id, MOEFLUX_MOE_GATHER_ID)` pair by the
//! orchestrator at `tests/moeflux_coherence.rs`. Loads the moeflux
//! engine, tokenizes the prompt, greedy-decodes a fixed number of
//! steps, and emits per-step `(chosen_id, top-50 (id, logit) pairs)`
//! on stdout as JSONL.
//!
//! Each subprocess has its own process-local `OnceLock<bool>` cache of
//! `MOEFLUX_MOE_GATHER_ID` (set inside `moeflux` at
//! `crates/moeflux/src/riir/attn/linear_attn_forward.rs:46-55`), so
//! the orchestrator can A/B by spawning twice with different envs —
//! something a single in-process run cannot do.
//!
//! Run directly (mostly for debugging):
//! ```bash
//! MOEFLUX_MOE_GATHER_ID=1 \
//!     cargo run --release --example moeflux_coherence_decode \
//!         --features "moeflux,moeflux-model-qwen3-6-35b-a3b" -- hobbit
//! ```
//!
//! Override model paths via env (matches `tests/moeflux_smoke.rs`):
//! - `DRAMA_LLAMA_MOEFLUX_MLX_DIR`
//! - `DRAMA_LLAMA_MOEFLUX_ARTIFACTS_DIR`
//! - `DRAMA_LLAMA_MOEFLUX_EXPERTS_DIR`
//!
//! ## Output format (stdout, JSONL)
//!
//! ```text
//! {"header":{"prompt_id":"hobbit","gather_id":"0","n_prompt_tokens":12,
//!            "n_vocab":248320,"steps":30,
//!            "stop_set":[151643, 151645, ...]}}
//! {"step":0,"chosen":12345,"top":[[12345, 4.21],[12346, 3.98],...]}
//! ... 30 step lines ...
//! {"trailer":{"done":true,"chosen_seq":[12345, 9876, ...]}}
//! ```
//!
//! `stop_set` is `Model::eog_tokens()` (the actual stop set, NOT
//! `special_tokens()` — which includes chat-template controls that can
//! appear in legitimate continuations — and NOT a union with `eot()`,
//! which on some vocabs is in-stream structure rather than a
//! terminator).
//!
//! Exit codes:
//! - `0` — success (all 30 steps produced + trailer emitted)
//! - `2` — controlled error (model not found, bad prompt-id, decode
//!         error). A single `{"error":"..."}` line precedes the exit.
//! - `1` — Rust panic (default).

#![cfg(all(feature = "moeflux", target_os = "macos"))]

use std::io::Write;
use std::num::NonZeroUsize;
use std::path::PathBuf;

use drama_llama::backend::Model;
use drama_llama::{Candidates, MoefluxEngine, Token, TokenData};

const N_STEPS: usize = 30;
const TOP_K_DUMP: usize = 50;

fn env_path(var: &str, default: &str) -> PathBuf {
    PathBuf::from(std::env::var(var).unwrap_or_else(|_| default.to_string()))
}

/// Sort candidates by logit (desc) with id-ascending tiebreak; truncate
/// to `k`. Mirrors `tests/regression.rs:92-103`.
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

fn prompt_for(id: &str) -> Option<&'static str> {
    match id {
        // Completion-style prompts — pick continuations every well-trained
        // LLM can produce. Reusable across moeflux model variants.
        "tobe" => Some("To be, or not to be, that is"),
        "constitution" => Some(
            "We the People of the United States, in Order to form a more perfect Union,",
        ),
        "hobbit" => Some("In a hole in the ground there lived a hobbit."),
        _ => None,
    }
}

fn emit_error(msg: impl std::fmt::Display) -> ! {
    let _ = writeln!(
        std::io::stdout(),
        "{}",
        serde_json::json!({ "error": format!("{msg}") }),
    );
    std::process::exit(2);
}

fn main() {
    // -- argv: <prompt-id> --
    let mut args = std::env::args();
    let _progname = args.next();
    let prompt_id = match args.next() {
        Some(s) => s,
        None => emit_error(
            "missing prompt-id arg (one of: tobe, constitution, hobbit)",
        ),
    };
    let prompt_str = match prompt_for(&prompt_id) {
        Some(p) => p,
        None => emit_error(format!("unknown prompt-id: {prompt_id:?}")),
    };
    let gather_id = std::env::var("MOEFLUX_MOE_GATHER_ID")
        .unwrap_or_else(|_| "0".to_string());

    // -- model paths --
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

    // -- engine --
    let mut engine = match MoefluxEngine::from_paths(
        &mlx_dir,
        &artifacts_dir,
        &experts_dir,
        false,
    ) {
        Ok(e) => e,
        Err(e) => emit_error(format!("MoefluxEngine::from_paths failed: {e}")),
    };

    let n_vocab = engine.model().n_vocab();
    let stop_set: Vec<Token> = engine.model().eog_tokens();

    // -- tokenize --
    let prompt_tokens: Vec<Token> = engine.model().tokenize(prompt_str, false);
    if prompt_tokens.is_empty() {
        emit_error(format!(
            "tokenization produced 0 tokens for {prompt_str:?}"
        ));
    }

    // -- header --
    let header = serde_json::json!({
        "header": {
            "prompt_id":       prompt_id,
            "gather_id":       gather_id,
            "n_prompt_tokens": prompt_tokens.len(),
            "n_vocab":         n_vocab,
            "steps":           N_STEPS,
            "stop_set":        stop_set,
        }
    });
    println!("{header}");

    // -- decode --
    let mut chosen_seq: Vec<Token> = Vec::with_capacity(N_STEPS);
    {
        let mut iter = engine.predict_candidates(
            prompt_tokens,
            NonZeroUsize::new(N_STEPS).unwrap(),
        );
        let mut step = 0usize;
        while let Some(candidates) = iter.next() {
            let top = partial_top_k(&candidates, TOP_K_DUMP);
            let chosen = top[0].id;
            let top_pairs: Vec<(Token, f32)> =
                top.iter().map(|td| (td.id, td.logit)).collect();
            let line = serde_json::json!({
                "step":   step,
                "chosen": chosen,
                "top":    top_pairs,
            });
            println!("{line}");
            chosen_seq.push(chosen);
            iter.record_choice(chosen);
            step += 1;
        }
    }

    if chosen_seq.len() != N_STEPS {
        emit_error(format!(
            "decoded {} steps, expected {N_STEPS} — iterator cut short",
            chosen_seq.len(),
        ));
    }

    // -- trailer --
    let trailer = serde_json::json!({
        "trailer": {
            "done":        true,
            "chosen_seq":  chosen_seq,
        }
    });
    println!("{trailer}");
}
