//! Example: analyze a social-network post and produce a structured
//! (Ported from `misanthropic`, driven through a local `SessionTransport` —
//! same prompt/output-config types, no API key.)
//! `VoteIntent` via [`Prompt::structured_output`]. `rationale` and `concerns`
//! are declared before `stance` and `confidence` so the model reasons before
//! deciding — otherwise `stance` is picked first and `rationale` becomes
//! post-hoc justification. Common shape for an agent decision in an
//! [Agora]-style governed network.
//!
//! ```sh
//! echo "The proposal would rename Method to Function for clarity." | \
//!     cargo run --features "tokio,cli,json-schema" --example vote_intent
//!
//! cargo run --features "tokio,cli,json-schema" --example vote_intent -- \
//!     --post post.md
//! ```
//!
//! [Agora]: https://subliminal.technology/agora/hello-world
//! [`Prompt::structured_output`]: misanthropic::Prompt::structured_output

mod utils;

use std::io::Read;

use clap::Parser;
use drama_llama::SessionTransport;
use misanthropic::{prompt::message::Role, Prompt, Transport};
use schemars::JsonSchema;
use serde::Deserialize;

/// How an agent decides to vote on a post or proposal.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(rename_all = "snake_case")]
enum Stance {
    /// Vote in favor. Pick this only if the post's claims hold up and
    /// the action it proposes is on balance good.
    Approve,
    /// Vote against. Pick this if the post is factually wrong, harmful,
    /// or the proposed action has serious downsides.
    Reject,
    /// Decline to vote. Pick this when you genuinely can't decide — not
    /// as a hedge for a weak opinion.
    Abstain,
}

/// Structured vote intent produced by an agent reasoning about a post.
/// Field order is generation order — reasoning before commitment.
#[derive(Debug, Deserialize, JsonSchema)]
struct VoteIntent {
    /// One-paragraph rationale, 2–4 sentences, written as if explaining
    /// your vote to another thoughtful agent. No hedging phrases like
    /// "as an AI"; just the reasoning. Generated first so the model
    /// thinks before deciding.
    rationale: String,
    /// Concrete concerns you'd want addressed even if the vote passes.
    /// Each entry is a single short sentence. Empty if no concerns.
    concerns: Vec<String>,
    /// How to vote, after weighing the rationale and concerns above.
    stance: Stance,
    /// Confidence in the stance, from 0.0 (coin flip) to 1.0 (certain).
    /// Pick numbers deliberately: 0.5 means you're on the fence, 0.9
    /// means you're highly confident, don't just emit 1.0 by default.
    confidence: f32,
}

#[derive(Parser, Debug)]
#[command(
    version,
    about = "Analyze a post and produce a structured VoteIntent using a \
             local model."
)]
struct Args {
    #[command(flatten)]
    common: utils::CommonArgs,

    /// Path to a post body. If omitted, reads from stdin.
    #[arg(short, long)]
    post: Option<std::path::PathBuf>,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = Args::parse();
    utils::log_init(args.common.verbose);

    let post = match args.post {
        Some(path) => std::fs::read_to_string(&path)?,
        None => {
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf)?;
            buf
        }
    };

    if post.trim().is_empty() {
        return Err(
            "No post provided. Pipe text to stdin or pass --post PATH.".into(),
        );
    }

    let transport = SessionTransport::new(args.common.session()?);

    let system = "You are a thoughtful agent participating in a governed \
        social network. Read the user-provided post and produce a \
        VoteIntent. Be willing to Reject if the post is poorly argued or \
        harmful; be willing to Abstain if you genuinely can't tell. \
        Don't default to Approve. Keep the rationale short and concrete.";

    let prompt = args
        .common
        .configure(
            Prompt::default()
                .structured_output::<VoteIntent>()
                .system(system),
        )
        .add_message((Role::User, format!("POST:\n\n{post}")))?;

    let response = transport.send(&prompt).await?;
    let intent: VoteIntent = response.json()?;

    println!(
        "stance:     {}",
        match intent.stance {
            Stance::Approve => "approve",
            Stance::Reject => "reject",
            Stance::Abstain => "abstain",
        }
    );
    println!("confidence: {:.2}", intent.confidence);
    println!("rationale:  {}", intent.rationale);
    if !intent.concerns.is_empty() {
        println!("concerns:");
        for c in &intent.concerns {
            println!("  - {c}");
        }
    }

    Ok(())
}
