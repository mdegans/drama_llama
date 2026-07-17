//! Example: **prompt caching** with [`CachedPrompt`] — pay the prefill for a
//! large prefix once, then re-use it on every following turn. Ported from
//! `misanthropic`'s example of the same name, driven through a local
//! [`SessionTransport`] — and arguably a better demo here, because locally
//! the cache win is *wall-clock*: turn one prefills the whole embedded-docs
//! prefix, later turns skip straight past it via the session's hash-keyed
//! prefix reuse.
//!
//! The cache is keyed on the request prefix (`tools` → `system` →
//! `messages`), so mutating any prefix field after the first request turns
//! every later request back into a full re-prefill. [`CachedPrompt`] makes
//! that mistake a compile error: [`CachedPrompt::cached`] adds a breakpoint
//! after the system prompt and freezes the prefix behind the wrapper,
//! leaving only cache-safe operations like [`CachedPrompt::push_message`].
//!
//! The demo is self-referential: the system prompt embeds the crate README
//! *and the `SessionTransport` source*, then asks two questions about them.
//! Watch the usage line and the timing flip between turns: turn one reports
//! zero `cache read` and a long prefill; turn two reports the shared prefix
//! as `cache read` and returns far sooner.
//!
//! ```sh
//! cargo run --example prompt_caching --features "tokio,cli"
//!
//! # Or ask your own questions about the crate:
//! cargo run --example prompt_caching --features "tokio,cli" -- \
//!     "What does SessionTransport::session return?" "Why Send and not Sync?"
//! ```
//!
//! [`CachedPrompt`]: misanthropic::CachedPrompt
//! [`CachedPrompt::cached`]: misanthropic::CachedPrompt::cached
//! [`CachedPrompt::push_message`]: misanthropic::CachedPrompt::push_message
//! [`SessionTransport`]: drama_llama::SessionTransport

mod utils;

use clap::Parser;
use drama_llama::SessionTransport;
use misanthropic::{prompt::message::Role, CachedPrompt, Prompt, Transport};

/// The large, stable prefix worth caching: instructions plus the documents
/// they refer to. `concat!` + `include_str!` bakes the crate's own docs in
/// at compile time — comfortably past a trivially-reusable prefix length.
const SYSTEM: &str = concat!(
    "You are an expert on the `drama_llama` Rust crate. Answer questions \
     using the crate README and the `SessionTransport` module source below. \
     Be concise and refer to items by name.\n\n# README.md\n\n",
    include_str!("../README.md"),
    "\n\n# src/session/transport.rs\n\n```rust\n",
    include_str!("../src/session/transport.rs"),
    "\n```",
);

#[derive(Parser, Debug)]
#[command(
    version,
    about = "Multi-turn Q&A over the crate's own docs with the large prefix \
             cached: full prefill on turn one, hash-keyed prefix reuse after."
)]
struct Args {
    #[command(flatten)]
    common: utils::CommonArgs,

    /// Questions to ask, one turn each. At least two turns are needed to
    /// see a cache read.
    #[arg(default_values_t = [
        "What does CachedPrompt freeze, and why does that matter for the \
         session's prefix cache?"
            .to_string(),
        "Why is SessionTransport's max_concurrency 1, and what does cloning \
         the transport share?"
            .to_string(),
    ])]
    questions: Vec<String>,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = Args::parse();
    utils::log_init(args.common.verbose);

    let transport = SessionTransport::new(args.common.session()?);

    // Wrap and drop the default breakpoint after the system prompt. From
    // here on the prefix is immutable — `prompt.system(…)` or
    // `prompt.model(…)` simply don't exist on the wrapper.
    let mut prompt = CachedPrompt::cached(
        args.common.configure(Prompt::default().system(SYSTEM)),
    );

    for (i, question) in args.questions.iter().enumerate() {
        println!("\n## Q{}: {question}\n", i + 1);

        prompt.push_message((Role::User, question.clone()))?;
        let start = std::time::Instant::now();
        let message = transport.send(&prompt).await?;
        let elapsed = start.elapsed();
        println!("{message}");

        // On turn one, `cache read` is zero and the whole embedded-docs
        // prefix is prefilled; on later turns the shared prefix shows up as
        // `cache read` and the turn returns far sooner.
        let usage = &message.usage;
        println!(
            "\n(input: {} | cache read: {} | output: {} | {:.1}s)",
            usage.input_tokens,
            usage.cache_read_input_tokens.unwrap_or(0),
            usage.output_tokens,
            elapsed.as_secs_f32(),
        );

        // The response converts straight back into a prompt message.
        prompt.push_message(message)?;
    }

    Ok(())
}
