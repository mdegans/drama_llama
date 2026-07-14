//! Structured output via [`Prompt::structured_output`] against local
//! inference — solve a murder mystery into a typed `CaseFile`.
//!
//! The schema is derived from the `CaseFile` struct below (via
//! `schemars`); the session compiles it into a sampling grammar, so the
//! JSON body *cannot* deviate from the schema. Reasoning models think
//! first — [`Session::complete_stream`] yields each [`Block`] as it
//! parses, so the thought arrives as soon as its envelope closes rather
//! than after the whole run.
//!
//! ```sh
//! cargo run --example whodunit --features json-schema --release -- \
//!     [path/to/model.gguf]
//! ```
//!
//! Defaults to `models/model.gguf` if no path is given.
//!
//! [`Prompt::structured_output`]: misanthropic::Prompt::structured_output
//! [`Session::complete_stream`]: drama_llama::Session::complete_stream
//! [`Block`]: drama_llama::Block

use std::path::PathBuf;

use drama_llama::{Block, LlamaCppSession, Prompt, Role};
use schemars::JsonSchema;
use serde::Deserialize;

/// One considered suspect. Field docs become schema descriptions the
/// model sees.
#[derive(JsonSchema, Deserialize, Debug)]
#[allow(dead_code)]
struct Suspect {
    name: String,
    motive: String,
    had_opportunity: bool,
}

#[derive(JsonSchema, Deserialize, Debug, PartialEq)]
enum Confidence {
    /// Evidence is thin; a jury would not convict.
    Low,
    /// The case is plausible but not airtight.
    Medium,
    /// The evidence conclusively identifies the culprit.
    High,
}

/// The verdict. Field order is generation order — each field is
/// context for the next, so the survey of suspects and evidence comes
/// before the culprit, and the summary last.
#[derive(JsonSchema, Deserialize, Debug)]
#[allow(dead_code)]
struct CaseFile {
    suspects_considered: Vec<Suspect>,
    key_evidence: Vec<String>,
    culprit: String,
    confidence: Confidence,
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Debugging aid: built with `--features json-schema,axum`, the
    // library's tracing instrumentation (grammar resolution, per-token
    // grammar filter counts) is compiled in — surface it with e.g.
    // `RUST_LOG=drama_llama::session=debug`.
    #[cfg(feature = "axum")]
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/model.gguf"));

    let mut session =
        LlamaCppSession::from_path_with_n_ctx(path, 8192)?.quiet();

    // No literal `<think>` / `</think>` in the prompt text: those are
    // control tokens in reasoning vocabs, and `Session` rejects blocks
    // that tokenize to specials (`SessionError::InjectedSpecialToken`) —
    // the reasoning envelope is the dialect's to emit, not content's to
    // spell. Asking for the *behaviour* is dialect-neutral and works on
    // models whose tags aren't spelled `<think>` at all.
    let prompt = Prompt::default()
        .structured_output::<CaseFile>()
        .system(
            "Enable deep thinking subroutine. You are a brief, \
             decisive detective. Think first, in under 300 tokens: note \
             which suspects are ruled out by their alibis, then identify \
             the one remaining with motive, means, and opportunity. Then \
             stop reasoning and output the structured verdict as JSON \
             matching the given schema.",
        )
        .add_message((Role::User, SCENARIO))?;

    // Stream blocks as they parse. Thought blocks buffer until their
    // closing tag, but prose streams *incrementally* — one `Block::Text`
    // per decoded piece — so the JSON body must be accumulated, not
    // taken from the last Text block. (The batch methods merge adjacent
    // prose for you; the stream deliberately does not.)
    let mut json_text: Option<String> = None;
    for block in session.complete_stream(&prompt)? {
        match block {
            Block::Thought { thought, .. } => {
                eprintln!("--- thought ---\n{thought}\n");
            }
            Block::Text { text, .. } => {
                json_text.get_or_insert_with(String::new).push_str(&text)
            }
            _ => {}
        }
    }

    // The grammar guarantees the text block matches the schema, so
    // this deserialize only fails if generation was cut off early
    // (e.g. by max_tokens).
    let text = json_text.ok_or("model never reached the JSON body")?;
    let verdict: CaseFile = serde_json::from_str(&text)?;
    println!("--- parsed CaseFile ---\n{verdict:#?}");

    Ok(())
}
