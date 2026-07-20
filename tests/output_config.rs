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

use std::{
    num::{NonZeroU128, NonZeroU32},
    path::PathBuf,
};

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
    /// Every suspect considered — one entry per named suspect in the
    /// scenario, with their motive and whether they had physical
    /// access to the scene. Never empty: even exonerated suspects
    /// were considered and belong here. (`length(min = 1)` survives
    /// the Anthropic schema sanitizer and compiles into the grammar,
    /// so the empty-array exit is closed at the sampling level.)
    #[schemars(length(min = 1))]
    suspects_considered: Vec<Suspect>,
    /// The evidence items the detective weighed, in the order
    /// considered. At least one — a verdict without evidence is not a
    /// verdict.
    #[schemars(length(min = 1))]
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
        // Seeded deliberately. The assertions below are about the
        // *shape* of the structured output — thought block present,
        // `$ref` array populated, typed round-trip — but they are
        // read off stochastic generation, so unseeded they measure
        // the model's mood as much as the code. Seeding makes a
        // failure mean "the structured-output path changed", which is
        // the only thing this test can honestly report on.
        //
        // A seed is safe here in a way it would not be in the cache
        // suites: those rely on unseeded runs to exercise the
        // sampler-state resume path (a forced seed forks every call
        // and discards the KV-paired snapshot), whereas this is a
        // single completion with no reuse.
        .with_seed(NonZeroU128::new(42))
        .with_render_opts(
            RenderOptions::default().with_extra("enable_thinking", true),
        );

    let prompt = Prompt::default()
        .max_tokens(NonZeroU32::new(4096).unwrap())
        .structured_output::<CaseFile>()
        .system(
            "You are a brief, decisive detective. Think before \
             answering, in under 300 tokens: note which suspects are \
             ruled out by their alibis, identify the one remaining with \
             motive, means, and opportunity. Then output the structured \
             verdict as JSON matching the given schema, listing ALL \
             named suspects in `suspects_considered`.",
        )
        .add_message((Role::User, Content::text(SCENARIO)))
        .expect("add_message");

    let response = session
        .complete_response(&prompt)
        .expect("complete_response");

    // (1) Response is multipart and contains at least one non-empty
    // Thought block — the model reasoned before committing.
    let blocks = &response.inner.content.0;
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

    // (3) The array of suspects deserialized through the $ref →
    // Suspect path, and is non-empty.
    //
    // Non-emptiness is the strongest claim this assertion can
    // honestly make: `length(min = 1)` is enforced in the grammar,
    // but counts beyond 1 are deliberately NOT — misanthropic's
    // sanitizer strips `minItems >= 2` (Anthropic only enforces
    // non-emptiness server-side) and `schema_to_gbnf` ignores larger
    // counts on purpose, because forcing N items makes the model
    // manufacture filler entries. See
    // `.claude/memory/schema_constraint_keywords_decision.md` — whose
    // trigger was *this test*, answering `suspects_considered: []`.
    //
    // So asserting ≥2 here, as this used to, claimed more than the
    // grammar guarantees and left the difference resting on the
    // model's mood; it duly broke when the sidecar's default chain
    // got hotter. Multi-element `$ref`-array coverage lives where it
    // can be deterministic: `compiles_ref_array_accepts_populated` in
    // `src/grammar_compile.rs` accepts a two-element array with no
    // model in the loop.
    assert!(
        !verdict.suspects_considered.is_empty(),
        "suspects_considered must be non-empty (grammar enforces \
         minItems=1): {:#?}",
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
