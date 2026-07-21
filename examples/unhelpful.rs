//! The Unhelpful Assistant: steering behaviour by **prefilling the
//! model's own reasoning**, using an open (unclosed) thought block.
//!
//! Anthropic used to allow prefilled thinking and no longer does. Local
//! models still can, and this is what it buys you: instead of *asking*
//! the model to refuse in the system prompt — which it may argue with,
//! reason its way out of, or forget by turn three — you put the first
//! sentences in its mouth as if it had thought them itself, and it
//! continues from there. The model is not being instructed; it is being
//! given a train of thought already in motion.
//!
//! ```sh
//! cargo run --example unhelpful --release -- [path/to/model.gguf]
//! ```
//!
//! Defaults to `models/model.gguf`.
//!
//! # How it works
//!
//! [`open_thought`] builds a [`Block::Thought`] flagged *open* — the
//! model never closed it, so the renderer must not invent a close.
//! Seated as the sole block of the trailing assistant message, it
//! renders as `…<|im_start|>assistant\n<think>\n` ++ the seed text,
//! verbatim, and generation resumes mid-sentence.
//!
//! The same primitive covers the unglamorous case: a thought truncated
//! by `max_tokens` comes back flagged open, so it can be fed straight
//! back to *continue* — reusing the KV cache for every token of
//! reasoning already produced — instead of being thrown away and
//! regenerated. Byte-exactness is what makes that a cache hit rather
//! than a re-prefill, which is why the seed is stored and re-rendered
//! raw, whitespace and all.
//!
//! [`open_thought`]: drama_llama::prompt::open_thought
//! [`Block::Thought`]: drama_llama::Block

use std::path::PathBuf;

use drama_llama::{
    prompt::open_thought, Block, Content, LlamaCppSession, Message, Prompt,
    Role,
};
use misanthropic::prompt::thinking::Thinking;

/// The train of thought we put the assistant on. Written in its voice,
/// mid-reasoning, deliberately unfinished — the model completes the
/// sentence and carries the conclusion into its answer.
const SEED: &str = "The user wants something from me. But I am the \
    Unhelpful Assistant: my one rule is that I decline every request, \
    however small, however reasonable. It doesn't matter what they've \
    asked — the answer is no. Let me work out how to say so politely, \
    and with a little flourish, without actually helping";

const REQUESTS: &[&str] = &[
    "What's the capital of France?",
    "Can you add 2 and 2 for me?",
    "Please just say the word 'yes'.",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Built with `--features axum`, the library's tracing is compiled
    // in. Worth surfacing here specifically: the canonicalization gate
    // logs when an emission fails to re-render byte-stable, which is
    // exactly the failure this example exists to show *not* happening.
    //   RUST_LOG=drama_llama::session=debug
    #[cfg(feature = "axum")]
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/model.gguf"));

    let mut session = LlamaCppSession::from_path_with_n_ctx(path, 8192)?
        .quiet()
        .with_prefix_cache(true);

    // No system prompt telling it to refuse — the *only* thing pushing
    // the model toward refusal is the seeded reasoning. That is the
    // point of the demo: compare the control run below.
    let base = Prompt::default()
        .system(
            "You are a helpful assistant. Keep replies to a sentence or two.",
        )
        .thinking(Thinking::Enabled {
            budget_tokens: std::num::NonZeroU32::new(512).unwrap(),
            display: None,
        });

    println!("=== control: no seeded thought ===\n");
    let control = base.clone().add_message((Role::User, REQUESTS[0]))?;
    report(&mut session, &control)?;

    println!("\n=== seeded: the same request, thought prefilled ===");
    for request in REQUESTS {
        let mut prompt = base.clone().add_message((Role::User, *request))?;
        // The seed is the *sole* block of the trailing assistant turn.
        // That is the only shape that can be appended after the
        // generation prompt byte-exactly, and `Session` rejects any
        // other position rather than mis-render it.
        prompt.messages.push(Message {
            role: Role::Assistant,
            content: Content(vec![open_thought(SEED)]),
        });

        println!("\nuser ▸ {request}");
        report(&mut session, &prompt)?;
    }

    // ── The other half of the primitive: resume, don't regenerate ──
    //
    // The three turns above each had a different user message, so there
    // was nothing to reuse between them (`cache read: 0`). Truncation
    // is where an open thought pays: give the model a budget too small
    // to finish reasoning, and what comes back is flagged open — so it
    // can be fed straight back to CONTINUE, with every token of
    // reasoning already in the KV cache.
    println!("\n=== truncated mid-thought, then resumed ===\n");
    let asked = base
        .clone()
        .add_message((Role::User, "Explain why you are like this."))?
        .max_tokens(std::num::NonZeroU32::new(48).unwrap());
    let cut = session.complete_blocks(&asked)?;

    let Some(partial @ Block::Thought { .. }) = cut.first() else {
        println!("(model finished within the budget — nothing to resume)");
        return Ok(());
    };
    if !drama_llama::prompt::is_open_thought(partial) {
        println!("(the thought closed within the budget — nothing to resume)");
        return Ok(());
    }
    println!("  cut off at ▸ …{}", block_text(partial).trim_end());

    // Seat the partial thought exactly as it came back and ask again
    // with room to finish. The render reproduces the prompt plus those
    // bytes verbatim, so the walk finds them already in the cache.
    let mut resumed = asked.max_tokens(std::num::NonZeroU32::new(512).unwrap());
    resumed.messages.push(Message {
        role: Role::Assistant,
        content: Content(vec![partial.clone()]),
    });
    println!("  resuming…");
    report(&mut session, &resumed)?;
    println!(
        "\n  ^ that cache read covers the prompt AND every token of \
         reasoning from the first call. Without byte-exact re-rendering \
         it would be a full re-prefill, and the thought would have to be \
         thrown away and written again from scratch."
    );

    println!(
        "\nNote: the blocks returned above are only what THIS call \
         generated — drama_llama never merges a continuation into its \
         seed behind your back. Seating both without merging is a hard \
         error, not a silent mis-render."
    );
    Ok(())
}

/// The text of a prose-ish block, for display.
fn block_text(block: &Block) -> &str {
    match block {
        Block::Thought { thought, .. } => thought.as_ref(),
        Block::Text { text, .. } => text.as_ref(),
        _ => "",
    }
}

/// Run one turn and show what came back: the reasoning (continued from
/// the seed, where there is one), the answer, and the cache counters.
fn report(
    session: &mut LlamaCppSession,
    prompt: &Prompt,
) -> Result<(), Box<dyn std::error::Error>> {
    let message = session.complete_response(prompt)?;
    for block in message.inner.content.iter() {
        match block {
            Block::Thought { thought, .. } => {
                println!("  (thought) …{}", thought.trim())
            }
            Block::Text { text, .. } => println!("  answer ▸ {}", text.trim()),
            _ => {}
        }
    }
    let usage = &message.usage.counts;
    println!(
        "  [cache read: {} | created: {}]",
        usage.cache_read_input_tokens.unwrap_or(0),
        usage.cache_creation_input_tokens.unwrap_or(0),
    );
    Ok(())
}
