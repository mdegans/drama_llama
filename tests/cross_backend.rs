//! Phase 4 gate-4: cross-backend equivalence check.
//!
//! Runs `LlamaCppEngine` and `MoefluxEngine` against the same prompt and
//! the same model (Qwen3.6-35B-A3B; GGUF for llama.cpp, MLX + moeflux
//! artifacts for moeflux) and asks: **given identical context, do the
//! two backends compute the same distribution?**
//!
//! # Teacher forcing, and why
//!
//! Both engines are driven down ONE reference trajectory — llama.cpp's
//! greedy path — via `record_choice`. moeflux is *forced* onto it rather
//! than following its own argmax.
//!
//! This matters, and the original design got it wrong. Letting each
//! backend follow its own greedy path measures whether their
//! trajectories stay *locked*, which is a chaotic property, not a
//! fidelity one: the two engines are different quantizations of the same
//! weights (GGUF IQ4_XS vs MLX 4-bit), so their logits sit a few tenths
//! of a nat apart, and the moment that flips one near-tie the contexts
//! fork and every later "step i vs step i" compares two different
//! sentences. Observed, before this was fixed: the two agreed perfectly
//! for 7 steps, then hit the paragraph break after
//! `"The quick brown fox jumps over the lazy dog.\n\n"` — a genuine
//! coin-flip — where llama.cpp repeated the pangram and moeflux opened a
//! fenced Python block. Both continuations are fine. Agreement scored
//! 0.219 and top-K overlap at the final step scored 0.000, neither of
//! which said anything about whether moeflux is correct.
//!
//! Forced onto one trajectory, every position is a like-for-like
//! comparison and a single ambiguous branch costs one step instead of
//! decorrelating the run.
//!
//! # What is asserted
//!
//! - **Argmax agreement on DECISIVE steps** ≥ `ARGMAX_AGREEMENT_MIN`.
//!   A step is decisive when llama.cpp's top-1 leads its top-2 by at
//!   least `DECISIVE_MARGIN` nats. Where the reference model itself is
//!   torn between two continuations, a fraction-of-a-nat quantization
//!   difference decides the tie, and which way it falls is not evidence
//!   about either backend. Those steps are reported, not asserted.
//! - **Probability-mass recall** ≥ `MASS_RECALL_MIN` at the *worst*
//!   step: of the mass llama.cpp puts on its top-K, how much sits on
//!   tokens moeflux also ranks in its top-K. Not raw set overlap —
//!   Jaccard treats every window member as equally important, but the
//!   more confident the model is the more of its top-K is junk, so set
//!   overlap scores *worst* exactly where the backends agree most (0.48
//!   at the confident steps vs 0.82 at the uncertain ones, measured).
//!   Weighting by mass puts the sensitivity where the distribution
//!   actually lives. Jaccard is still printed, as a diagnostic.
//! - Enough decisive steps to be worth measuring, so the argmax
//!   assertion cannot go vacuously green.
//!
//! These descend from `plan_v0.8.0_backend_split.md` Phase 4 exit
//! criterion #4, whose intent — "same shape, same heads of the
//! distribution" — is kept exactly; only the measurement changed, to one
//! that tracks the heads instead of the noise. Identical output across
//! backends is unrealistic (different quants, different attention
//! kernels, CUDA vs Metal MoE), and demanding it was what made the
//! original version fragile.
//!
//! Skipped (#[ignore]) by default — needs ~40 GB of artifacts mounted
//! and both backends linked. Run with:
//!
//! ```bash
//! just test moeflux cross_backend
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

use drama_llama::backend::{Backend, Model};
use drama_llama::{
    Candidates, Engine, LlamaCppEngine, MoefluxEngine, Token, TokenData,
};

/// Same prompt as `regression.rs`; chosen because greedy under it is
/// stable across llama.cpp point releases.
const PROMPT: &str = "The quick brown fox";

/// 32 steps matches the llama.cpp baseline; long enough to catch
/// accumulated drift, short enough that two ~20 GB models in one
/// process don't time out.
const N_STEPS: usize = 32;

/// Top-K window for set-overlap comparison. Wider than 20 dilutes the
/// signal (the long tail is essentially noise); narrower than 20 makes a
/// single tied logit swap a step.
const TOP_K: usize = 20;

/// A step counts as *decisive* when the reference backend's top-1 leads
/// its top-2 by at least this many nats — i.e. the model actually has an
/// opinion, and a backend that computes the same function must land on
/// the same token.
///
/// Below this margin the reference is genuinely torn, the two backends'
/// quantization offsets (~0.2–0.4 nats, measured) are the same order as
/// the gap, and the tie is decided by rounding. Which way it falls says
/// nothing about correctness, so those steps are reported and excluded
/// from the argmax assertion.
const DECISIVE_MARGIN: f32 = 1.0;

/// Minimum fraction of DECISIVE steps on which the two backends must
/// pick the same argmax. The plan's bar is ≥0.95.
const ARGMAX_AGREEMENT_MIN: f32 = 0.95;

/// Minimum share of steps that must be decisive. Guards the argmax
/// assertion against going vacuous: if almost nothing is decisive, the
/// prompt (or the model) has stopped being a useful probe and we want to
/// hear about it rather than pass on an empty set.
const DECISIVE_STEPS_MIN: f32 = 0.5;

/// Minimum [`mass_recall`] at the WORST step — the share of the
/// reference's top-K probability mass that the other backend also ranks
/// in its top-K.
///
/// Asserted per-step rather than on the mean: a mean over 32 steps
/// dilutes one catastrophic position into invisibility, and one position
/// where the backends disagree about where the probability lives is
/// exactly the thing we want to hear about.
///
/// This replaces the plan's "top-K set overlap ≥ 0.80". That criterion
/// was calibrated against a step-0-shaped number and then applied to
/// every step, where it measures the churn of the near-zero-mass tail
/// (see [`mass_recall`]) rather than agreement. Same intent — "same
/// shape, same heads of the distribution" — measured in a way that
/// tracks the heads instead of the noise.
const MASS_RECALL_MIN: f32 = 0.95;

/// One backend's view of the reference trajectory: the top-K it computed
/// at each step, having been fed the reference tokens.
struct Capture {
    n_vocab: i32,
    /// `topk[step]` — descending by logit. `topk[step][0]` is the token
    /// this backend *would* have picked at that step.
    topk: Vec<Vec<TokenData>>,
    /// The tokens actually fed. For the reference backend this is its
    /// own greedy path; for the other it is the same path, forced.
    fed: Vec<Token>,
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

/// Drive `engine` for `N_STEPS`, letting `pick` choose the token fed at
/// each step. Pass a greedy `pick` to *produce* a reference trajectory;
/// pass a lookup into that trajectory to *force* a second backend down
/// it.
///
/// Tokenization uses `add_special=false` so both backends see identical
/// prompt tokens (BOS handling differs — moeflux's HF tokenizer prefixes
/// nothing for Qwen, llama.cpp's GGUF might add BOS).
fn capture<B, F>(engine: &mut Engine<B>, mut pick: F) -> Capture
where
    B: Backend,
    F: FnMut(usize, &[TokenData]) -> Token,
{
    let prompt_tokens = engine.model().tokenize(PROMPT, false);
    let n_vocab = engine.model().n_vocab();

    let mut topk: Vec<Vec<TokenData>> = Vec::with_capacity(N_STEPS);
    let mut fed: Vec<Token> = Vec::with_capacity(N_STEPS);

    let mut iter = engine
        .predict_candidates(prompt_tokens, NonZeroUsize::new(N_STEPS).unwrap());

    let mut step = 0usize;
    while let Some(candidates) = iter.next() {
        let top = partial_top_k(&candidates, TOP_K);
        let chosen = pick(step, &top);
        topk.push(top);
        fed.push(chosen);
        iter.record_choice(chosen);
        step += 1;
    }

    Capture { n_vocab, topk, fed }
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

/// Of the probability mass the REFERENCE puts on its top-K, how much
/// sits on tokens the other backend also ranks in its top-K?
///
/// This is the assertable one, and raw Jaccard is not. Set overlap
/// treats every member of the window as equally important, which is
/// backwards: the more confident the model is, the more of its top-K is
/// junk. At a 7-nat margin the argmax holds essentially all the mass and
/// ranks 2..20 are noise whose ordering a fraction-of-a-nat quant offset
/// reshuffles at will — so Jaccard scores *worst* exactly where the two
/// backends agree most (measured: 0.48 at the confident steps, 0.82 at
/// the uncertain ones).
///
/// Weighting by mass inverts that, correctly. A token 7 nats down
/// contributes ~0.001 and its churn is ignored; but at step 6, where
/// top-1 and top-2 are 0.09 nats apart and hold ~30% of the mass each,
/// dropping either one craters the score. Sensitivity lands where the
/// distribution actually lives.
///
/// Softmax is taken WITHIN the window (renormalized over the K entries),
/// so this reads as "of the mass the reference places up here, how much
/// does the other backend agree belongs up here."
fn mass_recall(reference: &[TokenData], other: &[TokenData]) -> f32 {
    let ids: HashSet<Token> = other.iter().map(|t| t.id).collect();
    // Shift by the max before exp — these are raw logits (~18), and
    // exp(18) is fine in f64 but the shift is free and keeps it exact.
    let max = reference[0].logit;
    let (mut total, mut shared) = (0.0f64, 0.0f64);
    for td in reference {
        let p = f64::from(td.logit - max).exp();
        total += p;
        if ids.contains(&td.id) {
            shared += p;
        }
    }
    if total == 0.0 {
        1.0
    } else {
        (shared / total) as f32
    }
}

/// Top-1 minus top-2, in nats: how strong an opinion the model has here.
fn margin(top: &[TokenData]) -> f32 {
    if top.len() < 2 {
        f32::INFINITY
    } else {
        top[0].logit - top[1].logit
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

    // -- reference: llama.cpp, free-running greedy --------------------
    eprintln!("== llama.cpp side (reference trajectory) ==");
    eprintln!("loading {gguf:?}");
    let mut llama_engine =
        LlamaCppEngine::from_path(gguf.clone()).unwrap().quiet();
    eprintln!("  n_vocab={}", llama_engine.model().n_vocab());
    eprintln!("  n_ctx={}", llama_engine.n_ctx());
    let llama = capture(&mut llama_engine, |_step, top| top[0].id);
    let reference: Vec<Token> = llama.fed.clone();

    // -- moeflux, FORCED down the same trajectory ---------------------
    eprintln!("\n== moeflux side (forced onto the reference) ==");
    eprintln!("loading from {mlx_dir:?} + {artifacts_dir:?}");
    let moeflux = {
        let mut engine = MoefluxEngine::from_paths(
            &mlx_dir,
            &artifacts_dir,
            &experts_dir,
            false,
        )
        .expect("MoefluxEngine::from_paths");
        eprintln!("  n_vocab={}", engine.model().n_vocab());
        eprintln!("  n_ctx={}", engine.n_ctx());
        capture(&mut engine, |step, _top| reference[step])
    };

    eprintln!("\n== comparison ==");
    assert_eq!(
        llama.n_vocab, moeflux.n_vocab,
        "n_vocab differs ({} vs {}) — backends loaded incompatible models",
        llama.n_vocab, moeflux.n_vocab
    );
    assert_eq!(
        moeflux.fed, reference,
        "moeflux was not driven down the reference trajectory"
    );

    // Every step is now a like-for-like comparison: same context, two
    // backends, two distributions.
    let piece = |t: Token| llama_engine.model().token_to_piece(t);
    let steps: Vec<Step> = (0..N_STEPS)
        .map(|i| {
            let (l, m) = (&llama.topk[i], &moeflux.topk[i]);
            Step {
                fed: reference[i],
                llama_argmax: l[0].id,
                moeflux_argmax: m[0].id,
                margin: margin(l),
                jaccard: jaccard(l, m),
                mass: mass_recall(l, m),
            }
        })
        .collect();

    let decisive: Vec<&Step> = steps
        .iter()
        .filter(|s| s.margin >= DECISIVE_MARGIN)
        .collect();
    let agreed = decisive.iter().filter(|s| s.agrees()).count();
    let argmax_frac = if decisive.is_empty() {
        0.0
    } else {
        agreed as f32 / decisive.len() as f32
    };
    let decisive_frac = decisive.len() as f32 / N_STEPS as f32;
    let overlap_mean =
        steps.iter().map(|s| s.jaccard).sum::<f32>() / N_STEPS as f32;
    let mass_mean = steps.iter().map(|s| s.mass).sum::<f32>() / N_STEPS as f32;
    let worst = steps
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.mass.total_cmp(&b.mass))
        .expect("N_STEPS > 0");
    let overall_agreed = steps.iter().filter(|s| s.agrees()).count();

    eprintln!(
        "argmax agreement (decisive steps): {agreed}/{} = {argmax_frac:.3} \
         (min {ARGMAX_AGREEMENT_MIN:.2})",
        decisive.len()
    );
    eprintln!(
        "decisive steps: {}/{N_STEPS} = {decisive_frac:.3} (min \
         {DECISIVE_STEPS_MIN:.2}, margin >= {DECISIVE_MARGIN} nats)",
        decisive.len()
    );
    eprintln!(
        "argmax agreement (all steps, informational): \
         {overall_agreed}/{N_STEPS}"
    );
    eprintln!(
        "top-{TOP_K} mass recall: worst={:.4} @ step {} (min \
         {MASS_RECALL_MIN:.2}), mean={mass_mean:.4}",
        worst.1.mass, worst.0,
    );
    eprintln!(
        "top-{TOP_K} mean jaccard overlap: {overlap_mean:.3} \
         (informational — see `mass_recall`)"
    );

    // Per-step table. Printed on success too — the numbers track drift
    // across moeflux changes, and the ambiguous steps are exactly where
    // to look first when something regresses.
    eprintln!(
        "\n     step  fed                llama argmax         \
         moeflux argmax       margin  jaccard    mass"
    );
    for (i, s) in steps.iter().enumerate() {
        let flag = match (s.agrees(), s.margin >= DECISIVE_MARGIN) {
            (true, _) => "   ",
            (false, true) => "!! ", // decisive disagreement: a real fault
            (false, false) => " ~ ", // coin-flip: reported, not asserted
        };
        eprintln!(
            "{flag}{i:>3}  {:<18} {:>6} {:<12} {:>6} {:<12} {:>6.2}  {:>6.3}  \
             {:>6.4}",
            format!("{:?}", piece(s.fed)),
            s.llama_argmax,
            format!("{:?}", piece(s.llama_argmax)),
            s.moeflux_argmax,
            format!("{:?}", piece(s.moeflux_argmax)),
            s.margin,
            s.jaccard,
            s.mass,
        );
    }
    dump_topk("llama.cpp step 0", &llama.topk[0], 10);
    dump_topk("moeflux   step 0", &moeflux.topk[0], 10);

    assert!(
        decisive_frac >= DECISIVE_STEPS_MIN,
        "only {decisive_frac:.3} of steps were decisive (min \
         {DECISIVE_STEPS_MIN:.2}) — the argmax check would be near-vacuous, \
         so the probe prompt is no longer doing its job"
    );
    assert!(
        argmax_frac >= ARGMAX_AGREEMENT_MIN,
        "argmax agreement on decisive steps {argmax_frac:.3} below threshold \
         {ARGMAX_AGREEMENT_MIN:.2} — the backends disagree where the model \
         is NOT torn (marked `!!` above), which is a real divergence"
    );
    assert!(
        worst.1.mass >= MASS_RECALL_MIN,
        "step {} carries only {:.4} of the reference's top-{TOP_K} probability \
         mass in moeflux's top-{TOP_K} (min {MASS_RECALL_MIN:.2}) — the two \
         backends disagree about where the probability actually IS, which no \
         amount of tail churn can explain",
        worst.0,
        worst.1.mass,
    );
}

/// One position along the reference trajectory, seen by both backends.
struct Step {
    fed: Token,
    llama_argmax: Token,
    moeflux_argmax: Token,
    margin: f32,
    /// Raw top-K set overlap. Informational only — see [`mass_recall`].
    jaccard: f32,
    mass: f32,
}

impl Step {
    fn agrees(&self) -> bool {
        self.llama_argmax == self.moeflux_argmax
    }
}
