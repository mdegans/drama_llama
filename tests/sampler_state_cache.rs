//! Phase-2 integration matrix: SamplerState homed in the prefix
//! cache — the incremental-vs-cold seeding equality invariant, the
//! resume/fork/fresh trichotomy, and grammar-change-across-resume
//! robustness. Companion to the model-free reconcile-matrix unit
//! tests in `src/sample.rs` and the predictor-level tip-invariant
//! breaker in `tests/tip_invariant.rs`.
//!
//! All tests load a real model: `cargo test --test sampler_state_cache
//! -- --ignored --test-threads=1` (model tests are serialized — see
//! .config/nextest.toml).

#![cfg(feature = "llama-cpp")]

use std::{
    borrow::Cow,
    num::{NonZeroU128, NonZeroU32},
    path::PathBuf,
    sync::{Arc, Mutex},
};

use drama_llama::{
    Block, Content, FromPath, LlamaCppSession, Message, NGramStats, ProbeCtx,
    ProbeHook, Prompt, Role, SamplingMode,
};
use misanthropic::prompt::message::CacheControl;

fn model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf")
}

fn session() -> LlamaCppSession {
    LlamaCppSession::from_path(model_path())
        .expect("session load")
        .quiet()
        .with_prefix_cache(true)
}

/// A user message whose single text block carries an ephemeral cache
/// breakpoint.
fn cached_user(text: &'static str) -> Message {
    Message {
        role: Role::User,
        content: Content(vec![Block::Text {
            text: Cow::Borrowed(text),
            cache_control: Some(CacheControl::ephemeral()),
            citations: None,
        }]),
    }
}

const SYSTEM: &str = "You are a concise assistant. Answer in one short \
                      sentence.";
const USER_1: &str = "Name a primary color.";
const ASSISTANT_1: &str = "Blue is a primary color.";
const USER_2: &str = "Name another one, and a fruit of that color.";

/// Round-1 prompt: system + cache-marked user turn.
fn prompt_1() -> Prompt {
    Prompt {
        system: Some(Content::text(SYSTEM)),
        messages: vec![cached_user(USER_1)],
        max_tokens: NonZeroU32::new(8).unwrap(),
        ..Prompt::default()
    }
}

/// Round-2 prompt extending round 1 with a FIXED assistant reply
/// (deliberately not the model's actual reply, so the auto-tip can
/// never match and the cache hit lands on `USER_1`'s breakpoint) and
/// a fresh cache-marked user turn.
fn prompt_2() -> Prompt {
    let mut p = prompt_1();
    p.messages.push(Message {
        role: Role::Assistant,
        content: Content::text(ASSISTANT_1),
    });
    p.messages.push(cached_user(USER_2));
    p
}

/// Captures `(ngram_stats, step)` from the FIRST sampled token's
/// state. rng is deliberately not captured — cold and resumed states
/// legitimately differ there (fresh entropy vs the breakpoint
/// snapshot's rng); the fold invariant is about the stats.
struct FoldCapture {
    out: Arc<Mutex<Option<(NGramStats, u64)>>>,
}

impl ProbeHook for FoldCapture {
    fn on_token(&mut self, ctx: ProbeCtx<'_>) {
        let mut slot = self.out.lock().unwrap();
        if slot.is_none() {
            *slot = Some((ctx.state.ngram_stats().clone(), ctx.state.step()));
        }
    }
}

fn capture_first_token_fold(
    session: &mut LlamaCppSession,
    prompt: &Prompt,
) -> (NGramStats, u64) {
    let out = Arc::new(Mutex::new(None));
    session
        .engine_mut()
        .set_probe_hook(Some(Box::new(FoldCapture { out: out.clone() })));
    let _ = session.complete_text(prompt).expect("complete_text");
    session.engine_mut().set_probe_hook(None);
    let captured = out.lock().unwrap().take();
    captured.expect("probe hook captured no token")
}

/// THE Phase-2 invariant: resuming at a cache_control breakpoint and
/// folding only the suffix produces bit-identical n-gram stats (and
/// prose-step counter) to a cold whole-prompt fold. `NGramStats`
/// derives `PartialEq` over a `BTreeMap`, so this is exact, not
/// approximate.
#[test]
#[ignore = "long running, requires models/model.gguf"]
fn incremental_fold_matches_cold_fold() {
    // Cold: a fresh session folds prompt_2 from the top.
    let mut cold = session();
    let (cold_stats, cold_step) =
        capture_first_token_fold(&mut cold, &prompt_2());
    drop(cold);

    // Incremental: prime with prompt_1 (snapshotting fold state at
    // USER_1's breakpoint), then resume prompt_2 at that breakpoint
    // and fold only the assistant + USER_2 suffix.
    let mut warm = session();
    let _ = warm.complete_text(&prompt_1()).expect("priming call");
    assert!(
        warm.last_usage().cache_creation_input_tokens.unwrap_or(0) > 0
            || warm.last_usage().input_tokens > 0,
        "priming call should have processed the prompt",
    );
    let (warm_stats, warm_step) =
        capture_first_token_fold(&mut warm, &prompt_2());
    assert!(
        warm.last_usage().cache_read_input_tokens.unwrap_or(0) > 0,
        "second call must actually hit the breakpoint \
         (cache_read_input_tokens > 0) for this test to test anything",
    );

    assert_eq!(
        cold_step, warm_step,
        "prose-step counter diverged between cold and incremental folds",
    );
    assert_eq!(
        cold_stats, warm_stats,
        "n-gram stats diverged between cold and incremental folds",
    );
}

/// Round-2 prompt for the #106 oracle: extends round 1 with an
/// assistant tool call and a user turn carrying its tool result plus
/// a fresh cache-marked text block. With the seeding flags at their
/// defaults, the fold must ingest the call's argument string values
/// and the result's text identically on both arms.
fn prompt_2_tools() -> Prompt {
    let mut p = prompt_1();
    p.messages.push(Message {
        role: Role::Assistant,
        content: Content(vec![drama_llama::prompt::ToolUse::new(
            "log_color",
            serde_json::json!({
                "color": "blue is a primary color pigment",
                "confidence": 3,
            }),
        )
        .with_id("call_1")
        .into()]),
    });
    p.messages.push(Message {
        role: Role::User,
        content: Content(vec![
            Block::ToolResult {
                result: misanthropic::tool::Result {
                    tool_use_id: "call_1".into(),
                    content: "logged the color blue as a primary color".into(),
                    is_error: false,
                    cache_control: None,
                },
            },
            Block::Text {
                text: Cow::Borrowed(USER_2),
                cache_control: Some(CacheControl::ephemeral()),
                citations: None,
            },
        ]),
    });
    p
}

/// [`FoldCapture`] plus the #106 constrained fields.
struct ConstrainedFoldCapture {
    #[allow(clippy::type_complexity)]
    out: Arc<Mutex<Option<(NGramStats, u64, NGramStats, u64)>>>,
}

impl ProbeHook for ConstrainedFoldCapture {
    fn on_token(&mut self, ctx: ProbeCtx<'_>) {
        let mut slot = self.out.lock().unwrap();
        if slot.is_none() {
            *slot = Some((
                ctx.state.ngram_stats().clone(),
                ctx.state.step(),
                ctx.state.constrained_ngram_stats().clone(),
                ctx.state.constrained_step(),
            ));
        }
    }
}

fn capture_first_token_fold_constrained(
    session: &mut LlamaCppSession,
    prompt: &Prompt,
) -> (NGramStats, u64, NGramStats, u64) {
    let out = Arc::new(Mutex::new(None));
    session.engine_mut().set_probe_hook(Some(Box::new(
        ConstrainedFoldCapture { out: out.clone() },
    )));
    let _ = session.complete_text(prompt).expect("complete_text");
    session.engine_mut().set_probe_hook(None);
    let captured = out.lock().unwrap().take();
    captured.expect("probe hook captured no token")
}

/// The #106 oracle: with tool blocks in the turn-2 suffix and every
/// seeding flag at its default, cold and incremental folds still agree
/// bit-exactly — now including the seeded constrained accumulator.
/// The session carries a grammar so the capability gate passes; at the
/// first token the seed has run and nothing has diverged, so the
/// constrained fields must equal the persistent ones on both arms.
#[test]
#[ignore = "long running, requires models/model.gguf"]
fn incremental_fold_matches_cold_fold_with_tools() {
    let grammar =
        SamplingMode::grammar("root ::= \"red\" | \"green\" | \"blue\"")
            .expect("grammar compiles");

    let mut cold =
        session().with_sampling([grammar.clone(), SamplingMode::Greedy]);
    let (cold_stats, cold_step, cold_cstats, cold_cstep) =
        capture_first_token_fold_constrained(&mut cold, &prompt_2_tools());
    drop(cold);

    let mut warm = session().with_sampling([grammar, SamplingMode::Greedy]);
    let _ = warm.complete_text(&prompt_1()).expect("priming call");
    let (warm_stats, warm_step, warm_cstats, warm_cstep) =
        capture_first_token_fold_constrained(&mut warm, &prompt_2_tools());
    assert!(
        warm.last_usage().cache_read_input_tokens.unwrap_or(0) > 0,
        "second call must actually hit the breakpoint for this test to \
         test anything",
    );

    assert_eq!(cold_step, warm_step, "prose-step counter diverged");
    assert_eq!(cold_stats, warm_stats, "n-gram stats diverged");
    assert_eq!(
        cold_cstep, warm_cstep,
        "constrained step diverged between cold and incremental folds",
    );
    assert_eq!(
        cold_cstats, warm_cstats,
        "constrained stats diverged between cold and incremental folds",
    );

    // Sanity on the seed itself: capability gate passed, defaults on,
    // and the first token's structural state mutated neither corpus —
    // the constrained fields still equal the persistent ones.
    assert!(cold_step > 0, "the fold folded something");
    assert_eq!(cold_cstep, cold_step, "seed rebases into prose steps");
    assert_eq!(cold_cstats, cold_stats, "seed clones the corpus");
}

/// Fork: a fixed session seed makes identical calls bit-reproducible
/// even with the prefix cache on — the cached stream is ignored by
/// design (`Some(seed)` arm of the trichotomy).
#[test]
#[ignore = "long running, requires models/model.gguf"]
fn fork_with_seed_is_reproducible() {
    let mut s = session().with_seed(NonZeroU128::new(1337));
    let a = s.complete_text(&prompt_1()).expect("call 1");
    let b = s.complete_text(&prompt_1()).expect("call 2");
    assert_eq!(a, b, "seeded fork must reproduce exactly");
}

/// Resume determinism at a prompt breakpoint: with no seed, an
/// identical repeated prompt resumes the state snapshotted at the
/// breakpoint — whose rng is the first call's *initial* rng (the fold
/// never draws) — so the continuation is identical too. This pins the
/// resume arm without relying on stochastic divergence.
#[test]
#[ignore = "long running, requires models/model.gguf"]
fn resume_at_breakpoint_is_deterministic() {
    let mut s = session();
    let a = s.complete_text(&prompt_1()).expect("call 1");
    let b = s.complete_text(&prompt_1()).expect("call 2");
    assert!(
        s.last_usage().cache_read_input_tokens.unwrap_or(0) > 0,
        "second call must hit the breakpoint",
    );
    assert_eq!(
        a, b,
        "breakpoint resume replays the same rng position — identical \
         prompts must produce identical continuations",
    );
}

/// Changing the session grammar between resumed calls must reconcile
/// (matcher reset to the new grammar's root; stream carries) — never
/// panic or index out of bounds. The reconcile matrix is unit-tested
/// against MockModel in src/sample.rs; this is the end-to-end pin.
#[test]
#[ignore = "long running, requires models/model.gguf"]
fn changed_grammar_across_resume_reconciles() {
    let g1 = SamplingMode::grammar("root ::= \"red\" | \"green\" | \"blue\"")
        .expect("g1 compiles");
    let g2 =
        SamplingMode::grammar("root ::= \"apple\" | \"banana\" | \"cherry\"")
            .expect("g2 compiles");

    let mut s = session().with_sampling([g1, SamplingMode::Greedy]);
    let a = s.complete_text(&prompt_1()).expect("call 1 (grammar 1)");
    assert!(
        ["red", "green", "blue"].iter().any(|w| a.contains(w)),
        "grammar 1 output should be one of its words: {a:?}",
    );

    s = s.with_sampling([g2, SamplingMode::Greedy]);
    let b = s.complete_text(&prompt_2()).expect("call 2 (grammar 2)");
    assert!(
        ["apple", "banana", "cherry"].iter().any(|w| b.contains(w)),
        "grammar 2 output should be one of its words after a resumed \
         call with a different grammar: {b:?}",
    );
}
