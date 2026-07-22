//! Example: classify a unified diff into a structured commit message using
//! (Ported from `misanthropic`, driven through a local `SessionTransport` —
//! same prompt/output-config types, no API key.)
//! [`Prompt::structured_output`] / [`Message::json`]. Field order is generation
//! order (schemars preserves source order): `summary` precedes `category` so
//! the model describes the diff before labeling it — otherwise `category` is
//! picked first and `summary` becomes post-hoc justification.
//!
//! ```sh
//! # Against the last commit
//! git diff HEAD~1 | cargo run --features "tokio,cli,json-schema" \
//!     --example structured_commit_classifier
//!
//! # Against a file
//! cargo run --features "tokio,cli,json-schema" \
//!     --example structured_commit_classifier -- --diff changes.patch
//! ```
//!
//! [`Prompt::structured_output`]: misanthropic::Prompt::structured_output
//! [`Message::json`]: misanthropic::response::Message::json

mod utils;

use std::io::Read;

use clap::Parser;
use misanthropic::{prompt::message::Role, Prompt, Transport};
use schemars::JsonSchema;
use serde::Deserialize;

/// Conventional-commit category for a diff.
/// Schemars emits string enums as `anyOf` of `const` variants after
/// [`OutputConfig::for_type`] sanitizes the schema.
///
/// [`OutputConfig::for_type`]: misanthropic::prompt::OutputConfig::for_type
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(rename_all = "snake_case")]
enum Category {
    /// New feature or a new capability added to an existing feature.
    Feat,
    /// Bug fix — restores intended behavior.
    Fix,
    /// Internal rework with no behavioral change.
    Refactor,
    /// Documentation-only change.
    Docs,
    /// Test-only change.
    Test,
    /// Build system, tooling, or dependency change.
    Build,
    /// CI configuration change.
    Ci,
    /// Performance improvement with no behavioral change.
    Perf,
    /// Whitespace, formatting, or lint-only change.
    Style,
    /// Catch-all for anything that doesn't fit above.
    Chore,
}

/// Structured classification of a diff into commit-message components.
/// Field order is generation order — `summary` before `category`.
#[derive(Debug, Deserialize, JsonSchema)]
struct CommitClassification {
    /// Imperative one-line summary of the change, 70 characters or
    /// fewer, with no trailing period. Example: "Add cache_1h variant
    /// for 1-hour TTL". Do NOT include the category prefix. Generated
    /// first so the model describes the change before labeling it.
    summary: String,
    /// Conventional-commit category that best describes this diff,
    /// chosen after articulating the summary above.
    category: Category,
    /// True if this change likely alters the public API in a
    /// backwards-incompatible way (removed/renamed public items,
    /// changed signatures, new required fields on public structs).
    breaking: bool,
    /// Optional body paragraph explaining the "why" of the change.
    /// Wrap at roughly 72 columns. Empty string if no body needed.
    body: String,
}

#[derive(Parser, Debug)]
#[command(
    version,
    about = "Classify a unified diff into a structured commit message using \
             a local model."
)]
struct Args {
    #[command(flatten)]
    common: utils::CommonArgs,

    /// Path to a diff file. If omitted, reads the diff from stdin.
    #[arg(short, long)]
    diff: Option<std::path::PathBuf>,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = Args::parse();
    utils::log_init(args.common.verbose);

    let diff = match args.diff {
        Some(path) => std::fs::read_to_string(&path)?,
        None => {
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf)?;
            buf
        }
    };

    if diff.trim().is_empty() {
        return Err(
            "No diff provided. Pipe a diff to stdin or pass --diff PATH."
                .into(),
        );
    }

    let transport = args.common.transport().build()?;

    let system = "You are an experienced code reviewer classifying a diff \
        into a conventional-commit-style message. Base the category on \
        what the diff actually does, not on the filenames. Keep the \
        summary short, imperative, and specific. Only mark `breaking` \
        true if the public API is altered incompatibly.";

    let prompt = args
        .common
        .configure(
            Prompt::default()
                .structured_output::<CommitClassification>()
                .system(system),
        )
        .add_message((
            Role::User,
            format!("Classify this diff:\n\n```diff\n{diff}\n```"),
        ))?;

    let response = transport.send(&prompt).await?;
    let classification: CommitClassification = response.json()?;

    let prefix = match classification.category {
        Category::Feat => "feat",
        Category::Fix => "fix",
        Category::Refactor => "refactor",
        Category::Docs => "docs",
        Category::Test => "test",
        Category::Build => "build",
        Category::Ci => "ci",
        Category::Perf => "perf",
        Category::Style => "style",
        Category::Chore => "chore",
    };
    let bang = if classification.breaking { "!" } else { "" };
    println!("{prefix}{bang}: {}", classification.summary);
    if !classification.body.is_empty() {
        println!();
        println!("{}", classification.body);
    }

    Ok(())
}
