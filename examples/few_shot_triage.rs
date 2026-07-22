//! Example: *few-shot* structured output via [`Prompt::add_examples`]. Each
//! `(input, output)` pair becomes a user/assistant exchange *and* seeds the
//! [`output_config`] schema — no separate [`Prompt::structured_output`] call
//! needed, and the constraint can't drift from the examples. Exemplars teach
//! the model the depth of field population you want (`repro_steps` concrete and
//! non-empty; `is_regression` inferred from phrasing), which zero-shot misses
//! on smaller models.
//!
//! Ported from `misanthropic` — driven through a local
//! [`SessionTransport`](drama_llama::SessionTransport); same prompt and
//! output-config types, no API key.
//!
//! ```sh
//! cargo run --example few_shot_triage --features "tokio,cli,json-schema" -- \
//!     "Search returns no results for any query since this morning."
//! ```
//!
//! [`Prompt::add_examples`]: misanthropic::Prompt::add_examples
//! [`Prompt::structured_output`]: misanthropic::Prompt::structured_output
//! [`output_config`]: misanthropic::Prompt::output_config

mod utils;

use clap::Parser;
use misanthropic::{prompt::message::Role, Prompt, Transport};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Triage severity. Generated first so it anchors the rest of the triage.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(rename_all = "snake_case")]
enum Severity {
    /// Cosmetic or low-impact.
    Low,
    /// Degraded but workable.
    Medium,
    /// Major feature broken for many users.
    High,
    /// Outage, data loss, or security issue.
    Critical,
}

/// Structured triage of a free-text bug report.
/// Field order is generation order — each field is context for the next.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Triage {
    /// One-line, imperative summary of the underlying problem.
    summary: String,
    /// How bad it is, chosen after summarizing.
    severity: Severity,
    /// The component most likely at fault (e.g. "auth-ui", "checkout-api").
    component: String,
    /// Concrete, ordered steps to reproduce. Never leave empty when the report
    /// implies them — infer the obvious steps a tester would follow.
    repro_steps: Vec<String>,
    /// True when the report indicates the behavior regressed (e.g. "worked
    /// last week", "started after the update").
    is_regression: bool,
}

/// Triage a free-text bug report into a structured form.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    #[command(flatten)]
    common: utils::CommonArgs,

    /// The bug report to triage. A default report is used if omitted.
    report: Option<String>,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let cli = Cli::parse();
    utils::log_init(cli.common.verbose);
    let transport = cli.common.transport().build()?;

    let report = cli.report.unwrap_or_else(|| {
        "Checkout total shows $0.00 even though the cart has items. Only \
         happens on mobile Safari. A few customers reported it today."
            .to_string()
    });

    let base = cli.common.configure(Prompt::default().system(
        "You triage incoming bug reports into a structured form. \
                 Infer concrete reproduction steps and whether the issue is \
                 a regression from the wording of the report.",
    ));
    let prompt = base
        .add_examples([
            (
                "Safari users say the login button does nothing when clicked. \
                 Started after last week's release.",
                Triage {
                    summary: "Login button unresponsive on Safari".into(),
                    severity: Severity::High,
                    component: "auth-ui".into(),
                    repro_steps: vec![
                        "Open the app in Safari 17".into(),
                        "Click the 'Log in' button".into(),
                        "Observe: nothing happens; no network request fires"
                            .into(),
                    ],
                    is_regression: true,
                },
            ),
            (
                "Uploads over about 5 MB silently fail on mobile, no error \
                 is shown to the user.",
                Triage {
                    summary: "Large uploads fail silently on mobile".into(),
                    severity: Severity::Medium,
                    component: "upload-service".into(),
                    repro_steps: vec![
                        "On a mobile browser, open the upload dialog".into(),
                        "Select a file larger than 5 MB".into(),
                        "Submit and observe: the upload fails with no error"
                            .into(),
                    ],
                    is_regression: false,
                },
            ),
        ])?
        .add_message((Role::User, report))?;

    let triage: Triage = transport.send(&prompt).await?.json()?;

    println!("{triage:#?}");
    Ok(())
}
