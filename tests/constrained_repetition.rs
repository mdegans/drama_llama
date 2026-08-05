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
use std::sync::{Arc, Mutex};

use drama_llama::{
    Block, Content, FromPath, LlamaCppSession, Message, NGramStats, ProbeCtx,
    ProbeHook, Prompt, RepetitionOptions, Role, SamplerConfig, SamplingMode,
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
    LlamaCppSession::from_path(model_path())
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

// ── #106: seeded history pressure ────────────────────────────────────

/// A prompt whose history is tool traffic: the assistant called a
/// tool, the result carried the thread phrase, and the user asks for
/// output. The phrase is long (≥4 tokens) and padded so it sits well
/// past the block start — shorter or block-leading phrases seed
/// nothing under the fold's windows(max) shape.
fn tool_history_prompt() -> Prompt {
    Prompt {
        system: Some(Content::text(
            "You repeat words. Respond with only the requested output.",
        )),
        messages: vec![
            Message {
                role: Role::User,
                content: Content::text("Read the thread, then reply."),
            },
            Message {
                role: Role::Assistant,
                content: Content(vec![drama_llama::prompt::ToolUse::new(
                    "read_thread",
                    serde_json::json!({"thread": "buffalo-appreciation"}),
                )
                .with_id("call_1")
                .into()]),
            },
            Message {
                role: Role::User,
                content: Content(vec![
                    Block::ToolResult {
                        result: misanthropic::tool::Result {
                            tool_use_id: "call_1".into(),
                            // TWO posts share the phrase — the actual
                            // Agora symptom, and what pushes the
                            // surgical gate past effective > 1 (a
                            // once-seen phrase exerts zero pressure
                            // under the surgical default).
                            content: "Thread posts. First post: the \
                                      magnificent buffalo herd thundered \
                                      across the golden prairie grasslands \
                                      at dawn. Second post: once more the \
                                      magnificent buffalo herd thundered \
                                      across the golden prairie grasslands \
                                      at dusk."
                                .into(),
                            is_error: false,
                            cache_control: None,
                        },
                    },
                    Block::Text {
                        text: "Describe the thread's latest post in one \
                               sentence."
                            .into(),
                        cache_control: None,
                        citations: None,
                    },
                ]),
            },
        ],
        max_tokens: NonZeroU32::new(64).unwrap(),
        ..Prompt::default()
    }
}

fn seeded_session(
    configure: impl FnOnce(RepetitionOptions) -> RepetitionOptions,
) -> LlamaCppSession {
    let mut opts = SamplerConfig::default();
    opts.modes.insert(
        0,
        SamplingMode::grammar(STR_GRAMMAR).expect("grammar parses"),
    );
    opts.repetition = opts.repetition.map(configure);
    LlamaCppSession::from_path(model_path())
        .expect("session load")
        .quiet()
        .with_seed(Some(NonZeroU128::new(0xC0FFEE).unwrap()))
        .with_sample_options(opts)
}

/// Grabs the constrained accumulator at the first sampled token.
struct ConstrainedCapture {
    out: Arc<Mutex<Option<(NGramStats, u64)>>>,
}

impl ProbeHook for ConstrainedCapture {
    fn on_token(&mut self, ctx: ProbeCtx<'_>) {
        let mut slot = self.out.lock().unwrap();
        if slot.is_none() {
            *slot = Some((
                ctx.state.constrained_ngram_stats().clone(),
                ctx.state.constrained_step(),
            ));
        }
    }
}

/// #106 e2e: with the seeding defaults, tool-borne thread context is
/// present in the constrained accumulator at the first token (hard
/// assert — the seed survived the real tokenizer and the full Session
/// path), and the guard still closes the string under that seeded
/// pressure. Output divergence vs. seeding-off is logged, not
/// asserted — whether the pressure changes this model's pick is
/// logit-dependent; the contract is that it is *applied* and never
/// breaks the grammar.
#[test]
#[ignore = "long running, requires models/model.gguf"]
fn seeded_history_pressures_constrained_region() {
    let capture = |s: &mut LlamaCppSession| -> (NGramStats, u64, String) {
        let out = Arc::new(Mutex::new(None));
        s.engine_mut()
            .set_probe_hook(Some(Box::new(ConstrainedCapture {
                out: out.clone(),
            })));
        let text = s
            .complete_text(&tool_history_prompt())
            .expect("complete_text");
        s.engine_mut().set_probe_hook(None);
        let (stats, step) =
            out.lock().unwrap().take().expect("probe captured no token");
        (stats, step, text)
    };

    // Defaults: everything on.
    let mut on = seeded_session(|r| r);
    let (stats, step, text_on) = capture(&mut on);
    assert!(
        stats.total_ngram_count() > 0,
        "the tool-borne phrase must be seeded into the constrained \
         accumulator at generation start",
    );
    assert!(
        step > 0,
        "constrained_step must be rebased into the prose step-space",
    );
    assert!(
        text_on.starts_with('"')
            && text_on.len() >= 2
            && text_on.ends_with('"'),
        "the string must still close under seeded pressure: {text_on:?}",
    );

    // Counterfactual: seeding off (constrained_regions stays on).
    let mut off = seeded_session(|r| {
        r.set_seed_constrained_regions(false)
            .set_seed_tool_results(false)
            .set_seed_tool_args(false)
    });
    let (stats_off, step_off, text_off) = capture(&mut off);
    assert_eq!(stats_off.total_ngram_count(), 0, "off means off");
    assert_eq!(step_off, 0);
    println!("seeded-on  output: {text_on:?}");
    println!("seeded-off output: {text_off:?}");
}
