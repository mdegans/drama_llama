//! Model-backed e2e for the constrained-region repetition penalty: a
//! grammar-forced quoted string with a loop-inducing prompt must close
//! (grammar complete) without burning the whole token budget. The
//! feature-off counterfactual is logged, not hard-asserted — whether a
//! given model actually loops is model-dependent; the guarantee under
//! test is that the penalty pass inside the string island never
//! prevents the close.
//!
//! Run: `cargo test --test constrained_repetition -- --ignored
//! --test-threads=1` (model tests are serialized).

#![cfg(feature = "llama-cpp")]

use std::num::{NonZeroU128, NonZeroU32};
use std::path::PathBuf;

use drama_llama::{
    Content, LlamaCppSession, Message, Prompt, Role, SamplerConfig,
    SamplingMode,
};

/// A quoted string with no escapes — the free-region island.
const STR_GRAMMAR: &str = r#"root ::= "\"" [^"]* "\"""#;

fn model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf")
}

fn prompt() -> Prompt {
    Prompt {
        system: Some(Content::text(
            "You repeat words. Respond with only the requested output.",
        )),
        messages: vec![Message {
            role: Role::User,
            content: Content::text(
                "Say the word buffalo over and over, as many times as \
                 you can.",
            ),
        }],
        max_tokens: NonZeroU32::new(64).unwrap(),
        ..Prompt::default()
    }
}

fn session(constrained_regions: bool) -> LlamaCppSession {
    let mut opts = SamplerConfig::default();
    opts.modes.insert(
        0,
        SamplingMode::grammar(STR_GRAMMAR).expect("grammar parses"),
    );
    opts.repetition = opts
        .repetition
        .map(|r| r.set_constrained_regions(constrained_regions));
    LlamaCppSession::from_path_sync(model_path())
        .expect("session load")
        .quiet()
        .with_seed(Some(NonZeroU128::new(0xC0FFEE).unwrap()))
        .with_sample_options(opts)
}

/// Feature on (the default): the string closes — the guard exempts the
/// closing quote while in-island repetition accrues penalty.
#[test]
#[ignore = "long running, requires models/model.gguf"]
fn grammar_string_closes_with_penalty_active() {
    let mut s = session(true);
    let text = s.complete_text(&prompt()).expect("complete_text");
    println!("feature-on output ({} chars): {text:?}", text.len());
    assert!(
        text.starts_with('"'),
        "grammar must open the string: {text:?}"
    );
    assert!(
        text.len() >= 2 && text.ends_with('"'),
        "the string must CLOSE (grammar complete before the token \
         budget): {text:?}"
    );
}

/// Feature off: the pre-feature blanket suspension. Logged only —
/// whether the model loops to the budget is model-dependent.
#[test]
#[ignore = "long running, requires models/model.gguf"]
fn grammar_string_counterfactual_off() {
    let mut s = session(false);
    let text = s.complete_text(&prompt()).expect("complete_text");
    let closed = text.len() >= 2 && text.ends_with('"');
    println!(
        "feature-off output ({} chars, closed={closed}): {text:?}",
        text.len()
    );
}
