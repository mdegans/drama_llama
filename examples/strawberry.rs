//! Port of misanthropic's `strawberry.rs` example to `drama_llama`.
//!
//! Demonstrates tool use end-to-end with local inference:
//!
//! 1. Advertise a `count_letters` tool via [`Prompt::add_tool`].
//! 2. Force the model to call it with [`ToolChoice::Method`] +
//!    [`grammar_for_prompt`], which compiles to a
//!    [`SamplingMode::Grammar`] that constrains the output to the exact
//!    tool-call JSON shape.
//! 3. Parse the tool call, invoke the Rust handler, append the result
//!    as a [`Block::ToolResult`].
//! 4. Re-render the prompt (now with `tool_choice: Auto` so the model
//!    is free to answer in prose), and generate the final answer.
//!
//! Run with (pointing `models/model.gguf` at a Llama-3.1-Instruct GGUF):
//!
//! ```sh
//! cargo run --example strawberry --release -- \
//!     --prompt "Count the number of r's in 'strawberry'"
//! ```

use std::{num::NonZeroUsize, path::PathBuf};

use clap::Parser;
use drama_llama::{
    grammar_for_prompt, minijinja::Value as JinjaValue, silence_logs,
    ChatTemplate, Content, Engine, PredictOptions, Prompt, RenderOptions, Role,
    SampleOptions, SamplingMode, Tool, ToolChoice, ToolChoiceOptions,
};
use serde_json::json;

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
    /// Path to the GGUF model. Defaults to `models/model.gguf` relative
    /// to the drama_llama crate root.
    #[arg(long)]
    model: Option<PathBuf>,
    /// Print the raw rendered prompts at each turn.
    #[arg(long)]
    verbose: bool,
    /// Leave llama.cpp + ggml log spew enabled (they're silenced by
    /// default for readability).
    #[arg(long)]
    loud: bool,
}

/// The actual tool. Counts occurrences of `letter` in `string`,
/// case-insensitively.
fn count_letters(letter: char, string: &str) -> usize {
    let letter = letter.to_ascii_lowercase();
    string
        .chars()
        .filter(|c| c.to_ascii_lowercase() == letter)
        .count()
}

/// Extract the balanced `{…}` starting at the first `{` in `s`.
/// The grammar constraint guarantees a single balanced object, so
/// this is a cheap, byte-level slice rather than a full JSON parse.
fn slice_json_object(s: &str) -> Option<&str> {
    let start = s.find('{')?;
    let mut depth = 0i32;
    for (i, b) in s[start..].bytes().enumerate() {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&s[start..start + i + 1]);
                }
            }
            _ => {}
        }
    }
    None
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if !args.loud {
        // Hush llama.cpp + ggml before any context is created so even
        // model-load chatter stays quiet.
        silence_logs();
    }
    let model_path = args.model.unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf")
    });

    let mut engine = Engine::from_path(model_path)?;
    let tmpl = ChatTemplate::from_model(&engine.model)?;

    // Build the tool definition. The schema drives how the Jinja
    // template advertises the tool to the model; the tool_choice +
    // grammar enforcement happens separately below.
    let count_letters_tool = Tool {
        name: "count_letters".into(),
        description: "Count the number of times a letter appears in a string."
            .into(),
        schema: json!({
            "type": "object",
            "properties": {
                "letter": {"type": "string", "description": "the letter to count"},
                "string": {"type": "string", "description": "the string to search"}
            },
            "required": ["letter", "string"]
        }),
        cache_control: None,
    };

    // First turn: force the model to call count_letters.
    let mut prompt: Prompt<'static> = Prompt::default()
        .set_system(
            "You are a helpful assistant. You cannot count letters in a \
             word reliably on your own because you see in tokens, not \
             letters. Use the `count_letters` tool when asked to count \
             characters.",
        )
        .add_tool(count_letters_tool.clone())
        .add_message((Role::User, args.prompt.clone()))?;
    prompt.tool_choice = Some(ToolChoice::Method {
        name: "count_letters".into(),
    });

    // Turn on cogito's thinking mode via the template's `enable_thinking`
    // variable. The template injects an instruction into the system
    // prompt that unlocks the model's reasoning pass, which pairs well
    // with `allow_thought` on the grammar side.
    let render_opts = RenderOptions::default()
        .with_generation_prompt(true)
        .with_extra("enable_thinking", JinjaValue::from(true));
    let rendered = tmpl.render_with(&prompt, &render_opts)?;
    if args.verbose {
        println!("=== rendered prompt (turn 1) ===\n{rendered}\n===");
    }

    // Force BOTH the outer tool-call shape AND the tool's input_schema.
    // Cogito / Qwen / Hermes templates want tool calls wrapped in
    // `<tool_call>…</tool_call>` and use `"arguments"` rather than
    // Llama 3.1's bare `{"name": ..., "parameters": ...}`.
    //
    // `allow_thought` lets the model emit a `<think>…</think>` preamble
    // before the JSON — useful for reasoning models that need to work
    // out which argument goes where before committing.
    let opts = ToolChoiceOptions {
        strict_schema: true,
        arguments_field: "arguments",
        wrap_tags: Some(("<tool_call>\n", "\n</tool_call>")),
        allow_thought: true,
        ..ToolChoiceOptions::default()
    };
    if args.verbose {
        let src = drama_llama::build_grammar_source_for_debug(
            &["count_letters"],
            Some(&count_letters_tool),
            &opts,
        );
        println!("=== compiled GBNF ===\n{src}\n===");
    }
    let forced = grammar_for_prompt(&prompt, &opts)?
        .expect("Method tool_choice must yield a grammar");

    let tokens = engine.model.tokenize(&rendered, false);
    let mut opts = PredictOptions::default().add_model_stops(&engine.model);
    opts.n = NonZeroUsize::new(256).unwrap();
    opts.sample_options = SampleOptions {
        modes: vec![forced, SamplingMode::locally_typical()],
        ..SampleOptions::default()
    };

    let call_output: String = engine.predict_pieces(tokens, opts).collect();
    if args.verbose {
        println!("=== raw tool-call output ===\n{call_output}\n===");
    }

    // Slice the grammar-forced JSON object out of the output.
    let call_json = slice_json_object(&call_output)
        .ok_or("model output did not contain a JSON object")?;
    let call: serde_json::Value = serde_json::from_str(call_json)?;

    let name = call["name"].as_str().ok_or("missing name")?;
    if name != "count_letters" {
        return Err(format!("unexpected tool: {name}").into());
    }
    // Match the arguments_field we forced in the grammar.
    let params = &call["arguments"];
    let letter = params["letter"]
        .as_str()
        .and_then(|s| s.chars().next())
        .ok_or("missing 'letter' argument")?;
    let string = params["string"]
        .as_str()
        .ok_or("missing 'string' argument")?;
    let count = count_letters(letter, string);
    println!(
        "[tool] count_letters(letter={letter:?}, string={string:?}) = {count}"
    );

    // Second turn: append the assistant's tool-call message, then the
    // tool result. `tool::Use` and `tool::Result` both implement
    // `Into<Message>` — no manual Block / Content construction needed.
    let call_id = format!("call_{count}_{letter}");
    prompt = prompt.add_message(drama_llama::prompt::ToolUse {
        id: call_id.clone().into(),
        name: "count_letters".into(),
        input: params.clone(),
        cache_control: None,
    })?;
    prompt = prompt.add_message(drama_llama::prompt::ToolResult {
        tool_use_id: call_id.into(),
        content: Content::SinglePart(count.to_string().into()),
        is_error: false,
        cache_control: None,
    })?;
    prompt.tool_choice = None; // free generation for the prose answer

    let rendered2 = tmpl.render_with(
        &prompt,
        &RenderOptions::default()
            .with_generation_prompt(true)
            .with_extra("enable_thinking", JinjaValue::from(true)),
    )?;
    if args.verbose {
        println!("=== rendered prompt (turn 2) ===\n{rendered2}\n===");
    }
    let tokens2 = engine.model.tokenize(&rendered2, false);
    let mut opts2 = PredictOptions::default().add_model_stops(&engine.model);
    opts2.n = NonZeroUsize::new(256).unwrap();
    let answer: String = engine.predict_pieces(tokens2, opts2).collect();
    // Trim trailing EOS / `[Invalid UTF-8]` junk.
    let eos_piece = engine.model.token_to_piece(engine.model.eos());
    let trimmed = answer
        .trim_end_matches(eos_piece.as_str())
        .trim_end_matches("[Invalid UTF-8]")
        .trim_end();
    println!("\n=== answer ===\n{trimmed}");
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
