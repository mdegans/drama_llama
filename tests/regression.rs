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
//! # What is pinned, and what is merely recorded
//!
//! The golden is checked in once and compared on every machine, so it
//! may only assert things that are actually invariant across backends,
//! GPU drivers, and llama.cpp point releases. Measured on this model
//! (Qwen3.6-35B-A3B, IQ4_XS), that line falls in a specific place:
//!
//! - **`prompt_tokens`, `tokens`, `pieces` — asserted exactly.** Greedy
//!   is deterministic, and empirically it is deterministic *across
//!   backends* too: this golden was captured on CUDA and the full
//!   32-token stream reproduces byte-for-byte on Metal. Note this is
//!   already 32 argmax assertions over a 248k vocab — it is the
//!   harness's real teeth.
//! - **`logits_step_0` — asserted BY ID, within `LOGIT_TOL`.** Prefill
//!   is a single forward pass: no accumulated drift, no compounding.
//!   CUDA and Metal agree here. Rank, however, is *not* pinned and never
//!   was an invariant — the golden's ranks 2 and 3 sit 0.11 apart and
//!   ranks 10/11 sit 0.0001 apart, so a thousandth of a nat reorders
//!   them. Argmax exact, membership exact away from the K-th cutoff,
//!   magnitudes within tolerance, order ignored.
//! - **`logits_step_n` — only the argmax is asserted.** Everything below
//!   it is recorded and reported, not enforced. 31 decode steps in, CUDA
//!   and Metal share just 10 of the top 20 tokens and drift up to 1.86
//!   nats on the ones they do share, while the argmax stays put (it
//!   leads by 7.18 nats). The cause is almost certainly **MoE routing**:
//!   8 of 256 experts are active, a hair of numeric difference flips
//!   which ones, and that is a *discrete* divergence rather than smooth
//!   float noise — it moves the tail wholesale while leaving a confident
//!   top token untouched. Pinning that tail would turn a GPU driver
//!   update into a red suite, and a test that cries wolf gets ignored.
//!
//! The rule the split encodes: assert invariants, record everything
//! else. A drift summary (max per-id delta, shared-id count, id-aligned
//! cosine) is printed for the unasserted tail, so a genuine shift is
//! still visible to a human reading the log.
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

use std::collections::{BTreeMap, BTreeSet};
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

/// Absolute tolerance on logit magnitudes when comparing `logits_step_0`
/// against the golden. Comfortably covers the CUDA↔Metal prefill drift
/// we measure, and is the same order as the drift a rebuild with
/// different SIMD / backend flags produces.
///
/// Deliberately NOT stretched to cover `logits_step_n`: that divergence
/// is structural, not numeric (see module docs). Half of its top-20 is
/// *different tokens*, not the same tokens with different values, and no
/// tolerance can paper over a membership change.
const LOGIT_TOL: f32 = 0.5;

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
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf");
    let mut engine = LlamaCppEngine::from_path(path).unwrap().quiet();

    let prompt_tokens = engine.model().tokenize(PROMPT, true);
    let meta = ModelMeta {
        n_vocab: engine.model().n_vocab(),
        architecture: engine.model().get_meta("general.architecture"),
        context_size: engine.model().context_size(),
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
                piece: engine.model().token_to_piece(td.id),
                logit: td.logit,
            })
            .collect()
    };

    let logits_step_0 =
        topk_to_entries(step_0_raw.expect("step 0 never captured"));
    let logits_step_n =
        topk_to_entries(step_n_raw.expect("step N never captured"));
    let pieces: Vec<String> = tokens
        .iter()
        .map(|&t| engine.model().token_to_piece(t))
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

/// Assert only the argmax, and *report* the rest.
///
/// For the deep-context snapshot (`logits_step_n`), which is 31
/// accumulated decode steps deep. The argmax there is 7.18 nats clear
/// of rank 1 and is genuinely invariant; the tail below it is not, and
/// pinning it would make this suite go red for reasons that are nobody's
/// bug. See the module docs for why (MoE routing).
///
/// This is a smaller loss than it looks: `tokens` already compares 32
/// greedy argmaxes over the full 248k vocab, one per step. The tail of a
/// single snapshot adds little detection power on top of that, and adds
/// a lot of false-failure surface. The diagnostics are printed either
/// way, so a genuine shift is still visible to a human reading the log —
/// it just doesn't fail the build.
fn assert_argmax_only(label: &str, got: &[TopK], want: &[TopK]) {
    assert!(!want.is_empty() && !got.is_empty(), "{label}: empty top-K");
    let by_id = |v: &[TopK]| -> BTreeMap<Token, TopK> {
        v.iter().map(|e| (e.token, e.clone())).collect()
    };
    let diag = diagnostics(&by_id(got), &by_id(want));

    // Informational: visible under `just test regression` (which runs
    // uncaptured) and on any failure. Not an assertion — see above.
    eprintln!("{label}: tail not asserted.{diag}");

    assert_eq!(
        got[0].token,
        want[0].token,
        "{label}: ARGMAX changed (got {} `{}` @ {}, want {} `{}` @ {}). \
         This one IS an invariant — it is the sampled token.{diag}",
        got[0].token,
        got[0].piece,
        got[0].logit,
        want[0].token,
        want[0].piece,
        want[0].logit,
    );
}

/// Compare a captured top-K against the golden BY TOKEN ID.
///
/// For the prefill snapshot (`logits_step_0`) — a single forward pass,
/// so no accumulated drift and no compounding routing flips. It agrees
/// across backends within `LOGIT_TOL`, which makes it the sensitive
/// signal worth holding to a tight contract.
///
/// Three assertions, in descending order of what they'd cost us to
/// miss:
///
/// 1. **Argmax matches.** The only rank that is itself an invariant —
///    it *is* the sampled token under greedy, and it's 2.6 nats clear
///    of rank 1 here. (The token-stream comparison already covers this;
///    keeping it is belt and braces, and it fails with a better
///    message.)
/// 2. **Membership matches, away from the cutoff.** Every golden entry
///    that sits at least `LOGIT_TOL` above the golden's K-th (last)
///    logit must still be present. Entries *within* tolerance of that
///    cutoff may churn in and out — they are only on the list because K
///    got chopped at 20, and a hair of noise reshuffles who made it.
///    Symmetrically, no *new* id may appear above that line: a token
///    surging into the top group is a real distribution change.
/// 3. **Magnitudes match**, within `LOGIT_TOL`, per id.
///
/// Order is deliberately not compared — see the module docs.
fn assert_topk_close(label: &str, got: &[TopK], want: &[TopK]) {
    assert_eq!(
        got.len(),
        want.len(),
        "{label}: top-K length mismatch ({} vs {})",
        got.len(),
        want.len()
    );
    assert!(!want.is_empty(), "{label}: empty golden top-K");

    let by_id = |v: &[TopK]| -> BTreeMap<Token, TopK> {
        v.iter().map(|e| (e.token, e.clone())).collect()
    };
    let (got_by_id, want_by_id) = (by_id(got), by_id(want));

    // Both lists arrive sorted descending by logit (`partial_top_k`).
    assert_eq!(
        got[0].token,
        want[0].token,
        "{label}: ARGMAX changed (got {} `{}` @ {}, want {} `{}` @ {}){}",
        got[0].token,
        got[0].piece,
        got[0].logit,
        want[0].token,
        want[0].piece,
        want[0].logit,
        diagnostics(&got_by_id, &want_by_id),
    );

    // Below this line, membership is noise-dependent: an entry only
    // this close to the K-th logit could have been displaced by any
    // token just outside the captured window.
    let cutoff = want[want.len() - 1].logit + LOGIT_TOL;

    for (id, w) in &want_by_id {
        match got_by_id.get(id) {
            Some(g) => {
                let drift = (g.logit - w.logit).abs();
                assert!(
                    drift <= LOGIT_TOL,
                    "{label}: token {id} `{}` drifted {drift:.4} > tol \
                     {LOGIT_TOL} (got {}, want {}){}",
                    w.piece,
                    g.logit,
                    w.logit,
                    diagnostics(&got_by_id, &want_by_id),
                );
            }
            None => assert!(
                w.logit < cutoff,
                "{label}: token {id} `{}` (logit {}) vanished from the \
                 top-{} — it sat clear of the {:.4} cutoff, so this is a \
                 distribution change, not window churn{}",
                w.piece,
                w.logit,
                want.len(),
                cutoff,
                diagnostics(&got_by_id, &want_by_id),
            ),
        }
    }

    for (id, g) in &got_by_id {
        if !want_by_id.contains_key(id) {
            assert!(
                g.logit < cutoff,
                "{label}: token {id} `{}` surged into the top-{} at logit \
                 {} — clear of the {:.4} cutoff, so it displaced a golden \
                 entry rather than churning at the boundary{}",
                g.piece,
                want.len(),
                g.logit,
                cutoff,
                diagnostics(&got_by_id, &want_by_id),
            );
        }
    }
}

/// Appended to every failure message: is this noise or a real shift?
/// `max_drift` is the verdict-grade number (a bound, per id); cosine is
/// a shape summary over the id-aligned union — near-1.0 for float
/// noise, and it drops once magnitudes genuinely move. Cosine is NOT an
/// assertion: it averages, so one token exploding by 5 nats is diluted
/// across the other nineteen, and it says nothing an id-keyed delta
/// doesn't say better.
fn diagnostics(
    got: &BTreeMap<Token, TopK>,
    want: &BTreeMap<Token, TopK>,
) -> String {
    let ids: BTreeSet<Token> = got.keys().chain(want.keys()).copied().collect();
    let logit = |m: &BTreeMap<Token, TopK>, id: &Token| {
        m.get(id).map(|e| e.logit).unwrap_or(0.0)
    };
    let (mut dot, mut ga, mut wa, mut max_drift) =
        (0.0f64, 0.0f64, 0.0f64, 0.0f32);
    for id in &ids {
        let (g, w) = (logit(got, id), logit(want, id));
        dot += (g * w) as f64;
        ga += (g * g) as f64;
        wa += (w * w) as f64;
        if got.contains_key(id) && want.contains_key(id) {
            max_drift = max_drift.max((g - w).abs());
        }
    }
    let cos = dot / (ga.sqrt() * wa.sqrt()).max(f64::MIN_POSITIVE);
    let shared = got.keys().filter(|id| want.contains_key(id)).count();
    format!(
        "\n  diagnostics: max_drift={max_drift:.4} (tol {LOGIT_TOL}), \
         shared_ids={shared}/{}, id-aligned cosine={cos:.8}",
        want.len(),
    )
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
    assert_eq!(current.tokens, golden.tokens, "greedy token stream drifted");
    assert_eq!(current.pieces, golden.pieces, "decoded pieces drifted");
    assert_topk_close(
        "logits_step_0",
        &current.logits_step_0,
        &golden.logits_step_0,
    );
    assert_argmax_only(
        "logits_step_n",
        &current.logits_step_n,
        &golden.logits_step_n,
    );
}
