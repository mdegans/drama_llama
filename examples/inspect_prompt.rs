//! Inspect a saved misanthropic `Prompt` JSON: render via the chat
//! template, tokenize with breakpoints, and print enough structure to
//! attribute prefix-cache mismatches (re-tokenization shifts at
//! assistant-content boundaries, breakpoint placement, etc.).
//!
//! Usage:
//!   cargo run --example inspect_prompt --features cli -- \
//!     <path-to-prompt.json>
//!
//! The model path defaults to `models/model.gguf` (the same symlink
//! `dump_template` and the integration tests use). Override via the
//! `DRAMA_LLAMA_MODEL` env var if needed.
//!
//! Output highlights for cache debugging:
//! - Total tokens + breakpoint positions (cache_control markers).
//! - For each assistant message: the token position range and the
//!   first/last 5 tokens (decoded as pieces) — useful for spotting
//!   thought-stripping or JSON re-serialization between calls when
//!   comparing two consecutive prompt logs.
//! - The `add_generation_prompt=true` render's tail tokens (what the
//!   model would see queued up before its first generated token).
//!
//! Deliberately backend-concrete: renders and tokenizes through
//! `LlamaCppModel` directly, below the `Transport` erasure the other
//! examples use. Keep it that way — prefix-cache attribution is a
//! tokenizer-level question, not a transport-level one.

use std::path::PathBuf;

use drama_llama::{
    tokenize_with_breakpoints, ChatTemplate, LlamaCppModel, RenderOptions,
    Token,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let prompt_path: PathBuf = std::env::args()
        .nth(1)
        .ok_or("usage: inspect_prompt <prompt.json>")?
        .into();
    let model_path: PathBuf = std::env::var_os("DRAMA_LLAMA_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf")
        });

    let json = std::fs::read_to_string(&prompt_path)?;
    let prompt: misanthropic::Prompt = serde_json::from_str(&json)?;
    let model = LlamaCppModel::from_file(model_path.clone(), None).ok_or_else(
        || format!("could not load model: {}", model_path.display()),
    )?;
    let tmpl = ChatTemplate::from_model(&model)?;

    println!("=== prompt: {} ===", prompt_path.display());
    println!("model         : {}", prompt.model);
    println!("messages      : {}", prompt.messages.len());
    println!(
        "tool_choice   : {:?}",
        prompt.tool_choice.as_ref().map(|c| format!("{c:?}"))
    );
    println!("output_config : {}", prompt.output_config.is_some());
    println!();

    let opts = RenderOptions::default().with_generation_prompt(true);
    let rendered = tmpl.render_with_breakpoints(&prompt, &opts)?;
    let (tokens, breakpoints) = tokenize_with_breakpoints(&model, &rendered);

    println!("=== full render ===");
    println!("rendered bytes : {}", rendered.text.len());
    println!("total tokens   : {}", tokens.len());
    println!("breakpoints    : {:?}", breakpoints);
    println!();

    // Per-message render inspection: render successive partials
    // (system, system+msg0, system+msg0+msg1, ...) and use their
    // tokenized lengths to attribute each message's token range.
    println!("=== per-message token ranges ===");
    let mut cursor = 0usize;
    for (i, msg) in prompt.messages.iter().enumerate() {
        // Render up to and including this message via render_with on a
        // truncated prompt clone (cheap: minijinja's hot path).
        let mut partial = prompt.clone();
        partial.messages.truncate(i + 1);
        // No assistant header on partial renders; we want the close of
        // each message, not "ready for next".
        let partial_opts =
            RenderOptions::default().with_generation_prompt(false);
        let partial_text = tmpl.render_with(&partial, &partial_opts)?;
        let partial_tokens = model.tokenize(&partial_text, true);
        let end = partial_tokens.len();
        let role = msg.role.as_str();
        let preview = match peek_first_text(&msg.content) {
            Some(s) => {
                let trimmed: String =
                    s.chars().take(60).collect::<String>().replace('\n', "⏎");
                format!("{trimmed}…")
            }
            None => "<no text block>".to_string(),
        };
        println!(
            "  msg {i:2} {role:9} tokens [{cursor:5}..{end:5}) ({} tok) {preview}",
            end - cursor,
        );
        if msg.role.as_lowercase() == "assistant" {
            // For asst messages: print first 5 + last 5 token ids of the
            // message body. Useful for diff'ing the same message across
            // two consecutive prompt logs to spot re-render shifts.
            let slice = &tokens[cursor..end.min(tokens.len())];
            let head = slice.iter().take(5).copied().collect::<Vec<Token>>();
            let tail = slice
                .iter()
                .rev()
                .take(5)
                .rev()
                .copied()
                .collect::<Vec<Token>>();
            let head_p: Vec<String> =
                head.iter().map(|&t| model.token_to_piece(t)).collect();
            let tail_p: Vec<String> =
                tail.iter().map(|&t| model.token_to_piece(t)).collect();
            println!(
                "                head 5: {head:?} = {head_p:?}\n                tail 5: {tail:?} = {tail_p:?}"
            );
        }
        cursor = end;
    }
    println!();

    // Tail of the full render with add_generation_prompt=true: shows
    // the assistant header the model is queued up to continue from.
    let tail_n = 12usize;
    let tail_start = tokens.len().saturating_sub(tail_n);
    let tail_pieces: Vec<String> = tokens[tail_start..]
        .iter()
        .map(|&t| model.token_to_piece(t))
        .collect();
    println!("=== last {tail_n} tokens (with generation_prompt=true) ===");
    println!("ids    : {:?}", &tokens[tail_start..]);
    println!("pieces : {:?}", tail_pieces);

    Ok(())
}

fn peek_first_text(
    content: &misanthropic::prompt::message::Content,
) -> Option<String> {
    use misanthropic::prompt::message::Block;
    for b in &content.0 {
        if let Block::Text { text, .. } = b {
            return Some(text.to_string());
        }
    }
    None
}
