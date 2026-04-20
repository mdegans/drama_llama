//! Interactive chat REPL demonstrating [`Session`]'s prefix cache.
//!
//! Each turn prints the assistant reply followed by a usage line:
//! `[input: N | cached: M (pp%) | out: O]`. The `cached` number
//! should grow turn-over-turn as [`CachedPrompt::cache`] registers
//! new breakpoints after each exchange.
//!
//! Run:
//! ```text
//! cargo run --release --example chat_repl --features "repl" -- path/to/model.gguf
//! ```
//!
//! Ctrl-D or Ctrl-C exits and prints session totals.
//!
//! [`Session`]: drama_llama::Session
//! [`CachedPrompt::cache`]: drama_llama::CachedPrompt

use std::{num::NonZeroUsize, path::PathBuf};

use clap::Parser;
use drama_llama::{
    AssistantMessage, Block, CachedPrompt, Content, Prompt, Session,
    UserMessage,
};
use rustyline::{error::ReadlineError, DefaultEditor};

#[derive(Parser, Debug)]
#[command(about = "Chat REPL demonstrating drama_llama's prefix cache")]
struct Args {
    /// Path to the GGUF model file.
    model: PathBuf,
    /// Maximum tokens per reply.
    #[arg(short = 'n', long, default_value_t = 256)]
    max_tokens: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let n =
        NonZeroUsize::new(args.max_tokens).ok_or("--max-tokens must be > 0")?;

    let mut session = Session::from_path(args.model)?
        .quiet()
        .with_prefix_cache(true)
        .with_max_tokens(n);

    // Seed with a small, cacheable system prompt. `CachedPrompt::cached`
    // adds a 5-minute breakpoint on the last cacheable block (the
    // system here), so turn 1 establishes the cache and turn 2+ read
    // from it.
    let mut prompt = CachedPrompt::cached(
        Prompt::default()
            .set_system("You are a helpful assistant. Keep replies concise."),
    );

    let mut rl = DefaultEditor::new()?;
    println!("chat_repl — prefix cache on. Ctrl-D or Ctrl-C to exit.");

    loop {
        let line = match rl.readline(">>> ") {
            Ok(l) if l.trim().is_empty() => continue,
            Ok(l) => l,
            Err(ReadlineError::Interrupted | ReadlineError::Eof) => break,
            Err(e) => return Err(e.into()),
        };
        rl.add_history_entry(&line)?;

        prompt.push_message(UserMessage::from(line))?;

        let response = session.complete_response(&prompt)?;
        println!("{}", response.inner.content);
        print_usage(&response.usage);

        prompt.push_message(response.inner)?;
        // Register a new breakpoint on the last cacheable block (the
        // assistant turn we just appended) so the NEXT call can reuse
        // up to this point.
        prompt.cache();
    }

    print_totals(session.total_usage());
    Ok(())
}

fn print_usage(u: &misanthropic::response::Usage) {
    let input = u.input_tokens;
    let cached = u.cache_read_input_tokens.unwrap_or(0);
    let pct = if input > 0 {
        (cached as f64 / input as f64) * 100.0
    } else {
        0.0
    };
    eprintln!(
        "  [input: {input} | cached: {cached} ({pct:.1}%) | out: {}]",
        u.output_tokens,
    );
}

fn print_totals(t: &misanthropic::response::Usage) {
    eprintln!(
        "\nsession totals — input: {} | cached: {} | out: {}",
        t.input_tokens,
        t.cache_read_input_tokens.unwrap_or(0),
        t.output_tokens,
    );
}
