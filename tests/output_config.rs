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

use std::{num::NonZeroUsize, path::PathBuf};

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
Scenario: Sir Harold was found dead in his study at 11 PM, poisoned.

Suspects and verified facts:
- BUTLER (Mr. Finch): disliked Sir Harold. Served the nightcap at 9 PM, \
  but as a precaution (Sir Harold was paranoid) he took a sip from the \
  same glass in front of the house physician. He is alive and unharmed, \
  so the glass was not yet poisoned when it left his hands.
- NIECE (Lady Elsie): stood to inherit if Sir Harold died. She attended \
  the village charity gala from 8 PM to midnight; twenty named guests \
  place her there continuously. She cannot have been at the mansion.
- BUSINESS PARTNER (Mr. Crane): Sir Harold's ledger, found open on the \
  desk, showed Mr. Crane had been embezzling for two years and Sir \
  Harold intended to report him in the morning. Mr. Crane has a copy of \
  the study key (Sir Harold gave him one years ago). Two staff saw him \
  alone in the study from 10:30 to 10:50 PM. The poison is one Mr. \
  Crane keeps for his prize rose bushes.

All three had a motive. Only one had both opportunity (access to the \
glass after the butler's safe sip) AND means (possession of the \
specific poison used). Identify that suspect.";

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
        .set_system(
            "Enable deep thinking subroutine. You are a brief, \
             decisive detective. Reason inside <think>...</think> in \
             under 300 tokens: note which suspects are ruled out by \
             their alibis, identify the one remaining with motive, \
             means, and opportunity, then CLOSE the think tag. Output \
             the structured verdict as JSON matching the given schema.",
        )
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
