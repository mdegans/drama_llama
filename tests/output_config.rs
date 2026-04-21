//! Integration test for `Prompt::output_config` → Session structured
//! output.
//!
//! Loads a real model (cogito-family recommended) and exercises the
//! full path: optional `<think>...</think>` prefix, nested `$ref`
//! structs, `anyOf` enum, arrays, boolean fields, and round-trip via
//! misanthropic's typed `Message::json()`.
//!
//! All tests here are `#[ignore = "requires model"]`. Run with:
//! `cargo test --test output_config --features json-schema -- --ignored`.
//!
//! The scenario is a tight micro-whodunit: three suspects, five
//! evidence lines, exactly one of the three is the answer. The model
//! reasons inside the thought block and commits to the structured
//! verdict. Thinking is where the reasoning actually lives — the JSON
//! only holds the conclusion — so this is a real stress test of the
//! thought-prefix behavior rather than a vanity demo.
#![cfg(feature = "json-schema")]

use std::{borrow::Cow, num::NonZeroUsize, path::PathBuf};

use drama_llama::{Block, Content, Prompt, RenderOptions, Role, Session};

fn model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf")
}

#[derive(schemars::JsonSchema, serde::Deserialize, Debug)]
#[allow(dead_code)] // fields appear in schema + Debug output, not all asserted on
struct Suspect {
    name: String,
    motive: String,
    had_opportunity: bool,
}

#[derive(schemars::JsonSchema, serde::Deserialize, Debug, PartialEq)]
enum Confidence {
    /// Evidence is thin; a jury would not convict.
    Low,
    /// The case is plausible but not airtight.
    Medium,
    /// The evidence conclusively identifies the culprit.
    High,
}

#[derive(schemars::JsonSchema, serde::Deserialize, Debug)]
#[allow(dead_code)]
struct CaseFile {
    /// Every suspect considered, with their motive and whether they
    /// had physical access to the scene.
    suspects_considered: Vec<Suspect>,
    /// The evidence items the detective weighed, in the order
    /// considered.
    key_evidence: Vec<String>,
    /// The suspect name (must match one of `suspects_considered`).
    culprit: String,
    /// How certain the detective is of the verdict.
    confidence: Confidence,
    /// One-sentence summary of the deductive chain.
    reasoning_summary: String,
}

const SCENARIO: &str = "\
Scenario: Sir Harold was found dead in his locked study at 11 PM. \
The door key was still inside, in the lock. The window was wide open.

Suspects and known facts:
- BUTLER (Mr. Finch): served Sir Harold a nightcap at 9 PM. He was \
  owed six months' wages and had been threatened with dismissal.
- NIECE (Lady Elsie): the sole heir to the estate. A footman saw her \
  entering the rose garden outside the study window at 10:45 PM.
- BUSINESS PARTNER (Mr. Crane): arrived at the mansion at 11:15 PM, \
  AFTER the body was discovered. No known motive.

Evidence found at the scene:
1. The study door was locked from inside; only Sir Harold's key was on it.
2. The window faces the rose garden. Fresh footprints in the mulch \
   match a woman's boot.
3. Sir Harold was poisoned. Toxicology says the poison was in the \
   nightcap glass.
4. A half-finished letter was on the desk: Sir Harold was going to \
   disinherit the niece in the morning.
5. Mr. Finch prepared the nightcap personally, but the glass sat on \
   the sideboard for an hour before Sir Harold drank it.

Weigh motive AND opportunity for all three suspects. Consider whether \
the butler was framed (someone else could reach the glass during the \
hour it sat unattended). Do not commit to a verdict until you have \
considered all three suspects in <think>...</think>.";

#[test]
#[ignore = "requires model"]
fn whodunit_verdict() {
    // Default n_ctx (512) truncates long before the thought block
    // finishes. Bump to 8192 so scenario + thinking + verdict fit.
    let mut session = Session::from_path_with_n_ctx(model_path(), 8192)
        .expect("session load")
        .quiet()
        .with_render_opts(
            RenderOptions::default().with_extra("enable_thinking", true),
        )
        .with_max_tokens(NonZeroUsize::new(4096).unwrap());

    let prompt = Prompt::default()
        .structured_output::<CaseFile>()
        .set_system(Cow::Borrowed(
            "You are a careful detective. Reason step by step inside \
             <think>...</think>, weighing motive and opportunity for \
             every suspect before committing. Then output a structured \
             verdict as JSON matching the given schema.",
        ))
        .add_message((Role::User, Content::text(SCENARIO)))
        .expect("add_message");

    let response = session
        .complete_response(&prompt)
        .expect("complete_response");

    // (1) Response is multipart and contains at least one non-empty
    // Thought block — the model reasoned before committing.
    let blocks = match &response.inner.content {
        Content::SinglePart(_) => {
            panic!(
                "expected multipart response with thought + text blocks, \
                 got SinglePart"
            );
        }
        Content::MultiPart(blocks) => blocks,
    };
    let has_thought = blocks.iter().any(|b| {
        matches!(b, Block::Thought { thought, .. } if !thought.trim().is_empty())
    });
    assert!(
        has_thought,
        "expected at least one non-empty Block::Thought, got: {blocks:#?}"
    );

    // (2) Typed round-trip: the final Block::Text is valid JSON that
    // deserializes into the CaseFile schema via misanthropic's
    // Message::json() helper.
    let verdict: CaseFile = response
        .json()
        .expect("structured output should deserialize into CaseFile");

    println!("=== verdict ===\n{verdict:#?}\n===");

    // (3) The array of suspects deserialized correctly through the
    // $ref → Suspect path. We asked for three; at least two should
    // appear.
    assert!(
        verdict.suspects_considered.len() >= 2,
        "expected ≥2 suspects considered (tests $ref array), got {}: {:#?}",
        verdict.suspects_considered.len(),
        verdict.suspects_considered,
    );

    // (4) Culprit matches one of the considered suspects.
    let names: Vec<&str> = verdict
        .suspects_considered
        .iter()
        .map(|s| s.name.as_str())
        .collect();
    let matched = names.iter().any(|n| {
        n.eq_ignore_ascii_case(&verdict.culprit)
            || n.contains(&verdict.culprit)
            || verdict.culprit.contains(n)
    });
    assert!(
        matched,
        "culprit {:?} not present in suspects_considered {:?}",
        verdict.culprit, names,
    );

    // (5) Evidence array is populated.
    assert!(
        !verdict.key_evidence.is_empty(),
        "expected non-empty key_evidence array"
    );

    // (6) Confidence is a valid enum variant (tests anyOf
    // alternation compiled from schemars-emitted enum-with-docs).
    assert!(
        matches!(
            verdict.confidence,
            Confidence::Low | Confidence::Medium | Confidence::High
        ),
        "confidence deserialized to an unexpected variant: {:?}",
        verdict.confidence,
    );
}
