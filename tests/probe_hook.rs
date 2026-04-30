//! Smoke test for the per-token probe-mode hook.
//!
//! Verifies that an installed [`ProbeHook`] is invoked exactly once
//! per yielded token, with monotonically increasing positions and
//! the right tokens, and that uninstalling clears the hook.
//!
//! Requires `models/model.gguf`; ignored by default.

#![cfg(feature = "llama-cpp")]

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use drama_llama::{
    LlamaCppEngine, PredictOptions, ProbeCtx, ProbeHook, SnapshotOpts, Token,
};

const PROMPT: &str = "The quick brown fox jumps over the lazy dog.";
const N_STEPS: usize = 5;

/// Captures every token + position the hook is called with.
struct CountingHook {
    log: Arc<Mutex<Vec<(Token, usize)>>>,
}

impl ProbeHook for CountingHook {
    fn on_token(&mut self, ctx: ProbeCtx<'_>) {
        self.log.lock().unwrap().push((ctx.token, ctx.n_cur));
    }
}

#[test]
#[ignore = "long running"]
fn probe_hook_fires_once_per_yielded_token() {
    let log: Arc<Mutex<Vec<(Token, usize)>>> = Arc::new(Mutex::new(Vec::new()));

    let mut engine = LlamaCppEngine::from_path(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf"),
    )
    .unwrap();

    engine.set_probe_hook(Some(Box::new(CountingHook {
        log: log.clone(),
    })));

    let prompt_tokens = engine.model.tokenize(PROMPT, false);
    let prompt_len = prompt_tokens.len();

    let mut opts = PredictOptions::greedy();
    opts.n = NonZeroUsize::new(N_STEPS).unwrap();

    let yielded: Vec<Token> = engine.predict_tokens(prompt_tokens, opts).collect();

    let recorded = log.lock().unwrap().clone();

    // Hook fires once per yielded token (no extra fire on the prefill /
    // first-yield path; CandidatePredictor's first_candidates branch
    // doesn't sample — it just exposes the prefill logits).
    assert_eq!(
        recorded.len(),
        yielded.len(),
        "hook should fire exactly once per yielded token"
    );

    // Tokens recorded by the hook match the iterator's yields.
    let recorded_tokens: Vec<Token> = recorded.iter().map(|&(t, _)| t).collect();
    assert_eq!(recorded_tokens, yielded);

    // n_cur is monotonically increasing and starts at the prefill end.
    let positions: Vec<usize> = recorded.iter().map(|&(_, p)| p).collect();
    for window in positions.windows(2) {
        assert!(window[1] > window[0], "positions must be strictly increasing: {positions:?}");
    }
    assert_eq!(
        positions[0], prompt_len,
        "first hook position should be the prefill length"
    );
}

#[test]
#[ignore = "long running"]
fn probe_hook_can_be_cleared() {
    let log: Arc<Mutex<Vec<(Token, usize)>>> = Arc::new(Mutex::new(Vec::new()));

    let mut engine = LlamaCppEngine::from_path(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf"),
    )
    .unwrap();

    engine.set_probe_hook(Some(Box::new(CountingHook {
        log: log.clone(),
    })));
    engine.set_probe_hook(None);

    let prompt_tokens = engine.model.tokenize(PROMPT, false);
    let mut opts = PredictOptions::greedy();
    opts.n = NonZeroUsize::new(N_STEPS).unwrap();

    let _: Vec<Token> = engine.predict_tokens(prompt_tokens, opts).collect();

    assert!(
        log.lock().unwrap().is_empty(),
        "cleared hook must not be invoked"
    );
}

/// Captures snapshot presence + sampled-token lookup results per token.
/// Used to verify the rich-capture path is wired correctly when a hook
/// declares appetite via `snapshot_opts`.
struct SnapshotHook {
    log: Arc<Mutex<Vec<(Token, bool, f32, Option<u32>, usize)>>>,
    opts: SnapshotOpts,
}

impl ProbeHook for SnapshotHook {
    fn on_token(&mut self, ctx: ProbeCtx<'_>) {
        let (has_snap, sampled_p, sampled_rank, top_k_len) = match ctx.snapshot
        {
            Some(s) => (
                true,
                s.lookup_p(ctx.token),
                s.lookup_rank(ctx.token),
                s.top_k.len(),
            ),
            None => (false, 0.0, None, 0),
        };
        self.log.lock().unwrap().push((
            ctx.token,
            has_snap,
            sampled_p,
            sampled_rank,
            top_k_len,
        ));
    }
    fn snapshot_opts(&self) -> Option<SnapshotOpts> {
        Some(self.opts.clone())
    }
}

#[test]
#[ignore = "long running"]
fn probe_hook_snapshot_populated_when_requested() {
    let log: Arc<Mutex<Vec<(Token, bool, f32, Option<u32>, usize)>>> =
        Arc::new(Mutex::new(Vec::new()));

    let mut engine = LlamaCppEngine::from_path(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf"),
    )
    .unwrap();

    engine.set_probe_hook(Some(Box::new(SnapshotHook {
        log: log.clone(),
        opts: SnapshotOpts {
            top_k: NonZeroUsize::new(20).unwrap(),
            p_threshold: 0.0,
            compute_entropy: true,
        },
    })));

    let prompt_tokens = engine.model.tokenize(PROMPT, false);
    let mut opts = PredictOptions::greedy();
    opts.n = NonZeroUsize::new(N_STEPS).unwrap();

    let _: Vec<Token> = engine.predict_tokens(prompt_tokens, opts).collect();

    let recorded = log.lock().unwrap().clone();
    assert!(!recorded.is_empty(), "hook must fire at least once");

    for (token, has_snap, sampled_p, sampled_rank, top_k_len) in recorded {
        assert!(has_snap, "snapshot must be Some when appetite declared");
        assert!(top_k_len > 0, "top_k must be non-empty");
        assert!(
            (0.0..=1.0).contains(&sampled_p),
            "p out of range: {sampled_p}",
        );
        // Greedy sampling picks argmax of the chain output. With no
        // chain modes installed (`PredictOptions::greedy` uses
        // `SamplingMode::Greedy`), the chain output equals the
        // pre-everything argmax, so sampled_rank should be Some(1).
        assert_eq!(
            sampled_rank,
            Some(1),
            "greedy + no chain ⇒ rank 1 for token {token}",
        );
    }
}

#[test]
#[ignore = "long running"]
fn probe_hook_snapshot_does_not_perturb_sampling() {
    // Same prompt, same seed, two runs: one with no-snapshot hook,
    // one with rich-snapshot hook. Token sequences must be identical —
    // capture is `&self`-pure, so this is structurally guaranteed,
    // but the regression test catches accidents in future refactors.
    let prompt = PROMPT;

    let run = |with_snapshot: bool| -> Vec<Token> {
        let mut engine = LlamaCppEngine::from_path(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf"),
        )
        .unwrap();

        if with_snapshot {
            let log = Arc::new(Mutex::new(Vec::new()));
            engine.set_probe_hook(Some(Box::new(SnapshotHook {
                log,
                opts: SnapshotOpts::default(),
            })));
        } else {
            let log = Arc::new(Mutex::new(Vec::new()));
            engine.set_probe_hook(Some(Box::new(CountingHook { log })));
        }

        let prompt_tokens = engine.model.tokenize(prompt, false);
        let mut opts = PredictOptions::greedy();
        opts.n = NonZeroUsize::new(N_STEPS).unwrap();
        engine.predict_tokens(prompt_tokens, opts).collect()
    };

    let no_snap = run(false);
    let with_snap = run(true);

    assert_eq!(
        no_snap, with_snap,
        "snapshot capture must not perturb greedy sampling",
    );
}
