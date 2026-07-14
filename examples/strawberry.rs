//! Port of misanthropic's `strawberry.rs` to `drama_llama::Session` —
//! *typed* tool use against local inference.
//!
//! Same shape as the original: the model can't count letters within
//! tokens, so it gets a `count_letters` tool built with the
//! [`tool`] macro. The macro derives the wire schema from
//! `CountLetters` and gives `Strawberry` a concrete
//! [`Tool`] impl, so dispatch is typed — no hand-parsing of
//! `serde_json::Value`.
//!
//! The one thing the local port adds: [`ToolChoice::method`] doesn't
//! just *ask* for the call like the API does — the session compiles it
//! into a grammar that constrains sampling, so the call is guaranteed.
//!
//! ```sh
//! cargo run --example strawberry --features "cli,json-schema" --release
//! ```
//!
//! [`tool`]: misanthropic::tool::tool
//! [`Tool`]: misanthropic::tool::Tool
//! [`ToolChoice::method`]: drama_llama::ToolChoice::method

use std::{num::NonZeroUsize, path::PathBuf};

use clap::Parser;
use drama_llama::{
    Content, LlamaCppSession, Prompt, RenderOptions, Role, ToolChoice,
};
use misanthropic::tool::{tool, Tool};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// User prompt.
    #[arg(
        short,
        long,
        default_value = "Count the number of r's in 'strawberry'"
    )]
    prompt: String,
    /// Path to a GGUF model. Defaults to `models/model.gguf`.
    #[arg(long)]
    model: Option<PathBuf>,
}

/// Arguments for the `count_letters` method. The field docs become the
/// schema property descriptions the model sees (via `schemars`).
#[derive(Debug, Deserialize, JsonSchema)]
struct CountLetters {
    /// The letter to count.
    letter: String,
    /// The string to count letters in.
    string: String,
}

/// A stateless tool that counts letters. The [`tool`] macro generates
/// the schema wiring *and* a concrete `impl Tool` from the annotated
/// method below.
///
/// [`tool`]: misanthropic::tool::tool
struct Strawberry;

#[tool]
impl Strawberry {
    /// Count the occurrences of a letter in a string.
    #[method]
    async fn count_letters(
        &mut self,
        args: CountLetters,
    ) -> Result<Content, Content> {
        // BPE tokenizers love a leading space; trim before picking the
        // letter so `" r"` still counts r's.
        let Some(letter) = args.letter.trim().chars().next() else {
            return Err("`letter` must contain a letter".into());
        };
        let letter = letter.to_ascii_lowercase();
        let count = args
            .string
            .chars()
            .filter(|c| c.to_ascii_lowercase() == letter)
            .count();

        Ok(count.to_string().into())
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let model_path = args.model.unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf")
    });

    let mut strawberry = Strawberry;

    let mut session = LlamaCppSession::from_path_sync(model_path)?
        .quiet()
        // A repetition penalty can talk the model out of repeating the
        // tool result's digit in its answer; the grammar already keeps
        // the call itself well-formed.
        .without_repetition()
        .with_render_opts(
            RenderOptions::default()
                .with_generation_prompt(true)
                // Templates that gate a <think> block on this variable
                // (Qwen3, Cogito) pair it with the grammar's optional
                // thought preamble.
                .with_extra("enable_thinking", true),
        )
        .with_max_tokens(NonZeroUsize::new(256).unwrap());

    let mut chat = Prompt::default()
        .system(
            "You are a helpful assistant. You cannot count letters in a \
             word reliably on your own because you see in tokens, not \
             letters. Use the `count_letters` tool when asked to count \
             characters.",
        )
        // The generated definitions — schemas derived from
        // `CountLetters`, namespaced `Strawberry__count_letters`.
        .add_tools(strawberry.definitions())
        .add_message((Role::User, args.prompt))?;

    // Force the call. Locally this compiles to a sampling grammar, so
    // unlike the API's `tool_choice` it is a guarantee, not a request.
    let name = strawberry.definitions()[0].name().to_string();
    chat.tool_choice = Some(ToolChoice::method(name));

    // Turn 1: the grammar-forced tool call.
    let message = session.complete_response(&chat)?;
    let Some(call) = message.tool_use() else {
        return Err("tool was not called".into());
    };
    let call = call.clone();
    println!("[tool] {}({})", call.name, call.input);
    chat.push_message(message)?;

    // Typed dispatch: `call.input` is deserialized into `CountLetters`
    // and validated; bad arguments become a model-facing error
    // automatically. Returns a `tool::Result` ready to push.
    let result = strawberry.call(call).await;
    chat.push_message(result)?;

    // Turn 2: free generation for the prose answer.
    chat.tool_choice = None;
    let answer = session.complete_text(&chat)?;
    println!("{}", answer.trim());

    Ok(())
}
