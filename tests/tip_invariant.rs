//! Tip-invariant regression tests: a token that terminates generation
//! (stop sequence / stop string / EOG) must never mutate the sampler
//! state — entries, KV, and `SamplerState` all describe the same
//! stream position. See `SamplerState::advance` and
//! `TokenPredictor::next`.
//!
//! All tests load a real model: `cargo test --test tip_invariant --
//! --ignored`.

#![cfg(feature = "llama-cpp")]

use std::{num::NonZeroUsize, path::PathBuf};

use drama_llama::{LlamaCppEngine, PredictOptions, SamplingMode};

fn model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf")
}

fn engine() -> LlamaCppEngine {
    LlamaCppEngine::from_path(model_path()).expect("model loads")
}

/// A stop string that fires on the exact bytes completing the grammar:
/// the terminal token's bytes must NOT have advanced the matcher, so
/// the grammar reads as incomplete even though the text completed it.
///
/// Pre-hoist (advance inside `sample_token`) this asserted the
/// opposite: the matcher had eaten the terminal piece and
/// `grammar_complete()` returned `true` — the exact state pollution a
/// cached tip would have inherited.
#[test]
#[ignore = "long running, requires models/model.gguf"]
fn terminal_token_does_not_advance_matchers() {
    let mut engine = engine();
    let tokens = engine.model.tokenize("Say hello: ", true);

    let mut options = PredictOptions {
        n: NonZeroUsize::new(8).unwrap(),
        ..Default::default()
    };
    options.sample_options.modes = vec![
        SamplingMode::grammar("root ::= \"hello\"").expect("grammar compiles"),
        SamplingMode::Greedy,
    ];
    options.stop_strings.push("hello".to_string());

    let mut predictor = engine.predict_tokens(tokens, options);
    let mut generated = 0usize;
    for _ in predictor.by_ref() {
        generated += 1;
    }

    assert!(generated >= 1, "grammar-forced generation produced nothing");
    assert!(
        predictor.text.contains("hello"),
        "grammar should have forced the text 'hello', got {:?}",
        predictor.text
    );
    assert!(
        !predictor.grammar_complete(),
        "tip invariant violated: the terminal token's bytes advanced the \
         grammar matcher (state describes a position past the KV head)"
    );
}
