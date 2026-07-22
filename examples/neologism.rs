//! Port of `misanthropic`'s `neologism` example — the canonical
//! non-streaming completion — driven through a local
//! [`SessionTransport`] instead of the API [`Client`]: same prompt types,
//! same [`Transport::send`] call shape, no API key.
//!
//! ```sh
//! cargo run --example neologism --features "tokio,cli"
//! ```
//!
//! [`Client`]: misanthropic::Client

mod utils;

use clap::Parser;
use misanthropic::{prompt::message::Role, Prompt, Transport};

/// Invent new words and provide their definitions based on user-provided
/// concepts or ideas.
///
/// https://docs.anthropic.com/en/prompt-library/neologism-creator
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    #[command(flatten)]
    common: utils::CommonArgs,

    /// Prompt
    #[arg(
        short,
        long,
        default_value = "Can you help me create a new word for the act of pretending to understand something in order to avoid looking ignorant or uninformed?"
    )]
    prompt: String,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    utils::log_init(args.common.verbose);

    // Load the local model and wrap it as a `Transport` — the local
    // stand-in for `Client::new(key)`.
    let transport = args.common.transport().build()?;

    // Request a completion. The prompt-building patterns are misanthropic's
    // own — messages from tuples of `Role` and `String`, etc.
    let prompt = args
        .common
        .configure(Prompt::default().messages([(Role::User, args.prompt)])?);
    let message = transport.send(&prompt).await?;

    println!("{}", message);

    Ok(())
}
