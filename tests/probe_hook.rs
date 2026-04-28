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
    LlamaCppEngine, PredictOptions, ProbeCtx, ProbeHook, Token,
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
