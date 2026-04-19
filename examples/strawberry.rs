//! Port of misanthropic's `strawberry.rs` to `drama_llama::Session`.
//!
//! Same end-to-end tool-calling flow as the original misanthropic
//! example, but against local inference via
//! [`Session::complete`][complete] — which mirrors
//! [`misanthropic::Client::message`][cm] in shape, so the two
//! examples read line-for-line the same.
//!
//! 1. Advertise a `count_letters` tool on the [`Prompt`].
//! 2. Force the model to call it via
//!    [`ToolChoice::Method`][tcm] + the session's grammar compiler.
//! 3. [`Session::complete`][complete] returns an
//!    [`AssistantMessage`][am]; extract the [`ToolUse`][tu].
//! 4. Invoke the Rust handler, append
//!    [`ToolUse`][tu] and [`ToolResult`][tr] to the prompt, clear
//!    `tool_choice`, call [`Session::complete`][complete] again for
//!    the prose answer.
//!
//! ```sh
//! cargo run --example strawberry --release
//! ```
//!
//! [complete]: drama_llama::Session::complete
//! [cm]: https://docs.rs/misanthropic/latest/misanthropic/struct.Client.html#method.message
//! [am]: drama_llama::AssistantMessage
//! [tcm]: drama_llama::ToolChoice::Method
//! [tu]: drama_llama::prompt::ToolUse
//! [tr]: drama_llama::prompt::ToolResult

use std::{borrow::Cow, num::NonZeroUsize, path::PathBuf};

use clap::Parser;
use drama_llama::{
    minijinja::Value as JinjaValue,
    prompt::{ToolResult, ToolUse},
    Block, Content, FlashAttention, Prompt, RenderOptions, Role, SamplingMode,
    Session, Tool, ToolChoice, ToolChoiceOptions,
};
use serde_json::json;

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    #[arg(
        short,
        long,
        default_value = "Count the number of r's in 'strawberry'"
    )]
    prompt: String,
    #[arg(long)]
    model: Option<PathBuf>,
    /// Diagnostic mode: do turn 1 (tool call + result) locally, then
    /// serialize the prompt as-is and POST it to ollama's
    /// `/v1/messages` Anthropic-compatible endpoint instead of
    /// running turn 2 through [`Session`]. Useful for A/B-ing
    /// drama_llama vs ollama with the exact same `Prompt` JSON so
    /// any divergence in the prose answer can be pinned on one side
    /// or the other. Requires a running `ollama serve` on
    /// `localhost:11434`.
    #[arg(long)]
    ollama: bool,
    /// Pure-baseline mode: skip tool_choice forcing and repetition
    /// penalty entirely. Runs one `complete_text` against the raw
    /// user prompt, just to see what cogito produces with zero
    /// drama_llama constraints on top of the model.
    #[arg(long)]
    free: bool,
    /// Diagnostic: force Flash Attention off when building the
    /// [`Session`]'s underlying [`Engine`]. FA's fused softmax can
    /// flip the argmax on close-race token distributions, and
    /// toggling it off rules it out as the cause of divergence
    /// against other runners.
    #[arg(long)]
    fa_off: bool,
    /// Diagnostic: run CPU-only (zero GPU layers). Expensive, but
    /// rules out Metal/CUDA kernel divergence.
    #[arg(long)]
    cpu: bool,
    /// Diagnostic: instead of running the full tool loop, dump the
    /// top-k candidates + logits + pieces at every generated
    /// position for turn 1, one position per line. Intended for
    /// diffing against ollama's `/v1/chat/completions` logprobs
    /// response on the same prompt. When > 0, this flag overrides
    /// normal generation.
    #[arg(long, default_value_t = 0)]
    topk: usize,
}

/// Count occurrences of `letter` in `string`, case-insensitively.
fn count_letters(letter: char, string: &str) -> usize {
    let letter = letter.to_ascii_lowercase();
    string
        .chars()
        .filter(|c| c.to_ascii_lowercase() == letter)
        .count()
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let model_path = args.model.unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf")
    });

    let count_letters_tool = Tool {
        name: Cow::Borrowed("count_letters"),
        description: Cow::Borrowed(
            "Count the number of times a letter appears in a string.",
        ),
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

    let load_session = |path: PathBuf| -> Result<Session, _> {
        if args.cpu {
            Session::from_path_cpu_only(path)
        } else if args.fa_off {
            Session::from_path_with_flash_attention(
                path,
                FlashAttention::Disabled,
            )
        } else {
            Session::from_path(path)
        }
    };

    // Early branch: `--free` runs a single unconstrained turn so
    // we can see the model's raw behavior with no grammar, no tool
    // forcing, no repetition penalty.
    if args.free {
        let mut session = load_session(model_path)?
            .quiet()
            .without_repetition()
            .with_max_tokens(NonZeroUsize::new(256).unwrap());
        let prompt = Prompt::default()
            .set_system(
                "You are a helpful assistant. You cannot count letters in a \
                 word reliably on your own because you see in tokens, not \
                 letters. Use the `count_letters` tool when asked to count \
                 characters.",
            )
            .add_tool(Tool {
                name: Cow::Borrowed("count_letters"),
                description: Cow::Borrowed(
                    "Count the number of times a letter appears in a string.",
                ),
                schema: json!({
                    "type": "object",
                    "properties": {
                        "letter": {"type": "string"},
                        "string": {"type": "string"}
                    },
                    "required": ["letter", "string"]
                }),
                cache_control: None,
            })
            .add_message((Role::User, args.prompt.clone()))?;
        let out = session.complete_text(&prompt)?;
        println!("=== free-mode output ===\n{out}");
        return Ok(());
    }

    let mut session = load_session(model_path)?
        .quiet()
        .with_tool_choice_opts(ToolChoiceOptions {
            arguments_field: "arguments",
            wrap_tags: Some(("<tool_call>\n", "\n</tool_call>")),
            allow_thought: true,
            // strict_schema pins the JSON arg keys (`letter`,
            // `string`) as exact literal terminals. Without it the
            // BPE tokenizer emits keys like `" string"` (leading
            // space) that then don't match lookups. Phase 0.5.3
            // (minLength/maxLength/pattern) would further tighten
            // the VALUE side; for this demo strict_schema keys +
            // post-hoc value trimming is enough.
            strict_schema: true,
        })
        // Mild repetition penalty: 1.06 dampens the infinite-tool-
        // call loop some chat models fall into on the turn after a
        // tool result, without strangling single-digit answers the
        // way the old 1.15 default did. `penalty_max_count: 2`
        // means a token only gets penalized after its *second*
        // appearance, so the "3" from the tool_result doesn't
        // prevent the assistant from saying "3".
        .without_repetition()
        // Diagnostic: greedy sampling (argmax) — simplest possible
        // picker. If this produces coherent output, the bug was the
        // more exotic samplers (locally_typical / top_p edge cases).
        // If greedy is ALSO bad, something deeper is wrong.
        .with_sampling([SamplingMode::Greedy])
        .with_render_opts(
            // Cogito's template injects an extra reasoning-pass
            // instruction when `enable_thinking` is true. Pairs well
            // with `allow_thought` on the grammar side.
            RenderOptions::default()
                .with_generation_prompt(true)
                .with_extra("enable_thinking", JinjaValue::from(true)),
        )
        .with_max_tokens(NonZeroUsize::new(256).unwrap());

    let mut prompt: Prompt<'static> = Prompt::default()
        .set_system(
            "You are a helpful assistant. You cannot count letters in a word \
             reliably on your own because you see in tokens, not letters. \
             Use the `count_letters` tool when asked to count characters.",
        )
        .add_tool(count_letters_tool)
        .add_message((Role::User, args.prompt.clone()))?;
    prompt.tool_choice = Some(ToolChoice::Method {
        name: "count_letters".into(),
    });

    // Diagnostic: dump the turn-1 rendered prompt so we can feed it
    // raw into ollama and compare logits.
    if let Ok(path) = std::env::var("DUMP_TURN1") {
        let rendered = session
            .template()
            .render_with(
                &prompt,
                &RenderOptions::default()
                    .with_generation_prompt(true)
                    .with_extra("enable_thinking", JinjaValue::from(true)),
            )
            .expect("render turn1");
        std::fs::write(&path, &rendered)?;
        eprintln!(
            "[debug] wrote turn-1 render ({} bytes) to {path}",
            rendered.len()
        );
    }

    // Diagnostic: dump top-k logits and exit before generation.
    // Used to diff against ollama's /v1/chat/completions logprobs
    // on the same prompt.
    if args.topk > 0 {
        let trace = session.top_k_trace(&prompt, args.topk)?;
        println!("pos  piece                  logit       top-alts");
        for entry in &trace {
            let pick = &entry.top_k[0];
            let alts: Vec<String> = entry
                .top_k
                .iter()
                .skip(1)
                .take(3)
                .map(|e| format!("{:?}={:.2}", e.piece, e.logit))
                .collect();
            println!(
                "{:>3}  {:<22} {:>10.4}  {}",
                entry.position,
                format!("{:?}", pick.piece),
                pick.logit,
                alts.join(" | ")
            );
        }
        return Ok(());
    }

    // Turn 1: grammar-forced tool call.
    let assistant = session.complete(&prompt)?;
    let call = first_tool_use(&assistant).ok_or("no tool_use in response")?;
    let params = &call.input;
    // BPE tokenizers merge common words with leading spaces — trim
    // before picking the first char. See examples/ notes on why.
    let letter = params["letter"]
        .as_str()
        .and_then(|s| s.trim().chars().next())
        .ok_or("missing 'letter'")?;
    let string = params["string"].as_str().ok_or("missing 'string'")?;
    let count = count_letters(letter, string);
    println!(
        "[tool] count_letters(letter={letter:?}, string={string:?}) = {count}"
    );

    // Turn 2: append the assistant tool-use + the tool result, then
    // free generation for the prose answer.
    let call_id = format!("call_{count}_{letter}");
    prompt.push_message(assistant)?;
    prompt.push_message(ToolResult {
        tool_use_id: Cow::Owned(call_id),
        content: Content::SinglePart(count.to_string().into()),
        is_error: false,
        cache_control: None,
    })?;
    prompt.tool_choice = None;

    if args.ollama {
        // Ensure the misanthropic Prompt serializes cleanly to the
        // Anthropic API shape ollama's /v1/messages accepts. Pin
        // `model` so ollama routes to cogito:14b, and set a small
        // `max_tokens` cap so the A/B is fast.
        prompt.model = "cogito:14b".into();
        prompt.max_tokens = std::num::NonZeroU32::new(256).unwrap();
        let body = serde_json::to_string(&prompt)?;
        let out = std::process::Command::new("curl")
            .args([
                "-s",
                "-X",
                "POST",
                "-H",
                "Content-Type: application/json",
                "--data-binary",
                "@-",
                "http://localhost:11434/v1/messages",
            ])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()?;
        {
            use std::io::Write;
            out.stdin
                .as_ref()
                .ok_or("stdin")?
                .write_all(body.as_bytes())?;
        }
        let response = out.wait_with_output()?;
        let text = String::from_utf8_lossy(&response.stdout);
        println!("\n=== ollama /v1/messages answer ===\n{text}");
    } else {
        let answer = session.complete_text(&prompt)?;
        println!("\n=== answer ===\n{}", answer.trim());
    }
    Ok(())
}

/// Find the first `ToolUse` block inside an assistant message,
/// regardless of whether the content is SinglePart or MultiPart.
fn first_tool_use<'a>(
    assistant: &'a drama_llama::AssistantMessage<'a>,
) -> Option<&'a ToolUse<'a>> {
    match assistant.content() {
        Content::SinglePart(_) => None,
        Content::MultiPart(blocks) => blocks.iter().find_map(|b| match b {
            Block::ToolUse { call } => Some(call),
            _ => None,
        }),
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
