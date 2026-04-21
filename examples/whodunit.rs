//! Manual smoke for structured output via `Prompt::output_config`.
//!
//! Mirrors the integration test in `tests/output_config.rs` without
//! the `#[ignore]` gate, so you can iterate on the grammar / prompt
//! against a live model. The detective must reason inside
//! `<think>...</think>` then commit to a structured verdict.
//!
//! Run with:
//! ```text
//! cargo run --example whodunit --features json-schema --release -- \
//!     [path/to/model.gguf]
//! ```
//!
//! Defaults to `$PWD/models/model.gguf` if no path is given.
//!
//! Prints the thought blocks, the raw JSON, and the parsed [`CaseFile`]
//! struct. Exits non-zero on grammar violation or deserialization
//! failure so you notice regressions.

use std::{borrow::Cow, num::NonZeroUsize, path::PathBuf};

use drama_llama::{Block, Content, Prompt, RenderOptions, Role, Session};

#[derive(schemars::JsonSchema, serde::Deserialize, Debug)]
#[allow(dead_code)]
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
    suspects_considered: Vec<Suspect>,
    key_evidence: Vec<String>,
    culprit: String,
    confidence: Confidence,
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
the butler was framed. Do not commit to a verdict until you have \
considered all three suspects in <think>...</think>.";

fn main() {
    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/model.gguf"));
    if !path.exists() {
        eprintln!("model not found at {path:?}");
        std::process::exit(2);
    }

    let mut session = Session::from_path(path)
        .expect("session load")
        .quiet()
        .with_render_opts(
            RenderOptions::default().with_extra("enable_thinking", true),
        )
        .with_max_tokens(NonZeroUsize::new(2048).unwrap());

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

    let response = match session.complete_response(&prompt) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("generation failed: {e}");
            std::process::exit(1);
        }
    };

    if let Content::MultiPart(blocks) = &response.inner.content {
        for block in blocks {
            match block {
                Block::Thought { thought, .. } => {
                    println!("--- thought ---\n{thought}\n");
                }
                Block::Text { text, .. } => {
                    println!("--- raw verdict JSON ---\n{text}\n");
                }
                other => {
                    println!("--- unexpected block ---\n{other:#?}\n");
                }
            }
        }
    }

    match response.json::<CaseFile>() {
        Ok(verdict) => {
            println!("--- parsed CaseFile ---\n{verdict:#?}");
        }
        Err(e) => {
            eprintln!("deserialize failed: {e}");
            std::process::exit(1);
        }
    }
}
