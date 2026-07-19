//! The interview room: a chat REPL over [`Session`] that can also reseat a
//! prompt dumped by another example and let you talk to the agent it
//! belonged to.
//!
//! Fresh chat (the classic prefix-cache demo — watch `cache r` grow
//! turn-over-turn):
//!
//! ```text
//! cargo run --release --example chat --features "tokio,repl"
//! ```
//!
//! Exit interview — run the council with `--dump`, then reseat a seat and
//! ask it *why*: what was clear, what wasn't, why it filed what it filed.
//! (Downstream in Agora, agent feedback drives development; this closes
//! the same loop here.)
//!
//! ```text
//! cargo run --release --example chat --features "tokio,repl" -- \
//!     --load council_prompts/jester.json
//! ```
//!
//! ## Tool policy
//!
//! A loaded prompt's tools are **kept**, schemas and all: pruning them
//! would prune exactly the context an interview is for ("is this
//! description confusing?") and would shift the rendered prefix away from
//! what the agent actually saw. Every tool call the model makes is
//! printed. Only bash (below) executes; anything else is *caught* —
//! answered with a stub receipt naming this as an interview — so the
//! transcript stays wire-legal and you can say "go ahead, try the tool"
//! and debug interactively. A forced `tool_choice` (council seats) is
//! honored as one call per beat: a forced choice can never go quiet on
//! its own, so after the filing is receipted, control returns to you.
//!
//! - `--clear-tools` strips tools and any forced choice — a prose-only
//!   interview.
//! - `--tool-choice <auto|any|none|method:NAME>` overrides the loaded
//!   choice while keeping the defs rendered, so the prefix is unchanged
//!   (unlike `--clear-tools`). `auto` un-forces a council seat so it may
//!   answer in prose *or* file; `none` forbids any call while the agent
//!   still sees its tools — the prefix-preserving "don't call" mode.
//! - `--add-bash` appends misanthropic's [`RichBash`] in a Docker sandbox
//!   (needs Docker; the published misan-bashd image is pulled on first
//!   run) and dispatches its calls for real — no [`ToolBox`], the loop
//!   drives the [`Tool`] trait by hand.
//!
//! [`Session`]: drama_llama::Session
//! [`ToolBox`]: misanthropic::tool::ToolBox

mod utils;

use std::{collections::HashSet, path::PathBuf};

use clap::{CommandFactory, FromArgMatches, Parser};
use misanthropic::{
    prompt::message::{Block, CacheControl, Content, Role},
    response::TokenCounts,
    tool::{
        self,
        bash::{DockerSandbox, RichBash},
        Tool, Use,
    },
    Prompt,
};
use rustyline::{error::ReadlineError, DefaultEditor};
use utils::BoxError;

/// A diagnostic chat REPL: fresh chat, or reseat a dumped prompt and
/// interview the agent it belonged to.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    #[command(flatten)]
    common: utils::CommonArgs,
    #[command(flatten)]
    chat: utils::ChatArgs,
    /// Reseat a prompt dumped to JSON (e.g. by `council --dump`).
    #[arg(long)]
    load: Option<PathBuf>,
    /// Strip the loaded prompt's tools and forced tool choice — a
    /// prose-only interview.
    #[arg(long)]
    clear_tools: bool,
    /// Override the loaded prompt's tool choice, keeping the defs
    /// rendered (so the prefix is unchanged — unlike `--clear-tools`):
    /// `auto` (call or answer in prose), `any` (must call some tool),
    /// `none` (may not call — the prefix-preserving "don't call" mode),
    /// or `method:NAME` (must call NAME). Un-force a council seat with
    /// `auto`; forbid its filing with `none`.
    #[arg(
        long,
        value_parser = parse_tool_choice,
        value_name = "auto|any|none|method:NAME"
    )]
    tool_choice: Option<tool::Choice>,
    /// Add misanthropic's rich bash tool in a Docker sandbox (needs
    /// Docker; the published misan-bashd image is pulled on first run).
    #[arg(long)]
    add_bash: bool,
}

/// The receipt seated for a caught (non-bash) tool call — including the
/// dangling call a reseated transcript may end in. Short and byte-stable
/// on purpose: a forced seat files once per beat, so this lands in the
/// transcript repeatedly and should never thrash the cache.
const CAUGHT: &str =
    "Received. This transcript has been reseated for a post-run \
     interview: the tool's original backend is detached, and your calls \
     are read directly by the human interviewer.";

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), BoxError> {
    // Interview subjects arrive with council-sized transcripts (the
    // judge's record alone can pass 9k tokens), so mirror the council's
    // `--n-ctx` default rather than the shared 8192. `mut_arg` so
    // `--help` prints the default that will actually be used.
    let cli = Cli::from_arg_matches(
        &Cli::command()
            .mut_arg("n_ctx", |a| a.default_value("32768"))
            .get_matches(),
    )?;
    utils::log_init(cli.common.verbose);

    // ── The prompt: fresh, or a reseated transcript ──────────────
    // `tokio::fs` rather than `std::fs`: main is async (the bash tool's
    // `call` is), and example code gets copied into contexts where
    // blocking the runtime matters.
    let mut prompt: Prompt = match &cli.load {
        Some(path) => {
            serde_json::from_str(&tokio::fs::read_to_string(path).await?)?
        }
        None => Prompt::default()
            .system("You are a helpful assistant. Keep replies concise."),
    };
    prompt = cli.common.configure(prompt);
    if cli.clear_tools {
        prompt.tools = None;
        prompt.tool_choice = None;
    }
    // Override last: `--tool-choice` reshapes the choice while leaving
    // the defs in place (prefix intact). `any`/`method:` on a
    // defs-stripped prompt (`--clear-tools`) will surface as a typed
    // "no tools" error at completion, which is the honest outcome of
    // contradictory flags.
    if let Some(choice) = &cli.tool_choice {
        prompt.tool_choice = Some(choice.clone());
    }

    // ── Bash, manually driven ────────────────────────────────────
    // The defs APPEND to whatever the prompt already carries (see the
    // module docs on why loaded tools are kept). `on_init` boots the
    // container; teardown below stops it.
    let mut bash = match cli.add_bash {
        true => {
            println!("(booting the bash sandbox…)");
            let mut bash = RichBash::new(DockerSandbox::default());
            bash.on_init(&mut prompt).await?;
            Some(bash)
        }
        false => None,
    };
    let bash_methods: HashSet<String> = bash
        .iter()
        .flat_map(|bash| bash.definitions())
        .map(|def| def.name().to_string())
        .collect();
    if let Some(bash) = &bash {
        prompt = prompt.add_tools(bash.definitions());
    }

    let mut session = cli.common.session_with_cache_slots(1)?;
    if cli.common.max_tokens.is_none() {
        // Council-sized filings need council-sized budgets: a forced
        // seat that hits the cap mid-call is a typed GrammarViolation,
        // not a graceful truncation (the council learned this at 1024
        // and 2048; its default is what interviews inherit here).
        session = session.with_max_tokens(
            std::num::NonZeroUsize::new(4096).expect("nonzero"),
        );
    }

    match &cli.load {
        Some(path) => println!(
            "chat — reseated {} ({} message(s), {} tool method(s){}). \
             Ctrl-D exits.",
            path.display(),
            prompt.messages.len(),
            prompt.tools.as_deref().unwrap_or_default().len(),
            if forced(&prompt) {
                ", tool choice forced"
            } else {
                ""
            },
        ),
        None => println!("chat — prefix cache on. Ctrl-D exits."),
    }
    if let Some(choice) = &cli.tool_choice {
        println!("  tool choice → {}", describe_choice(choice));
    }

    // Drive the interview, then tear the sandbox down even on the error
    // path — it is a running container, not memory.
    let mut total = TokenCounts::default();
    let outcome = drive(
        &cli,
        &mut session,
        &mut prompt,
        bash.as_mut(),
        &bash_methods,
        &mut total,
    )
    .await;
    if let Some(bash) = &mut bash {
        if let Err(error) = bash.on_teardown(&mut prompt).await {
            log::warn!("bash teardown failed: {error}");
        }
    }
    println!("\nsession totals — {}", pay_line(&total));
    outcome
}

/// Parse `--tool-choice`: `auto`, `any`, `none`, or `method:NAME`.
fn parse_tool_choice(s: &str) -> Result<tool::Choice, String> {
    match s {
        "auto" => Ok(tool::Choice::auto()),
        "any" => Ok(tool::Choice::any()),
        "none" => Ok(tool::Choice::none()),
        other => other
            .strip_prefix("method:")
            .filter(|name| !name.is_empty())
            .map(tool::Choice::method)
            .ok_or_else(|| {
                format!("expected auto|any|none|method:NAME, got {other:?}")
            }),
    }
}

/// A short human label for an effective [`tool::Choice`], for the
/// startup banner.
fn describe_choice(choice: &tool::Choice) -> String {
    match choice {
        tool::Choice::Auto { .. } => "auto (call or prose)".into(),
        tool::Choice::Any { .. } => "any (must call a tool)".into(),
        tool::Choice::None => "none (calls forbidden, defs kept)".into(),
        tool::Choice::Method { name, .. } => format!("must call {name}"),
    }
}

/// Whether `prompt` forces a tool call on every completion.
fn forced(prompt: &Prompt) -> bool {
    matches!(
        prompt.tool_choice,
        Some(tool::Choice::Any { .. } | tool::Choice::Method { .. })
    )
}

/// The interview loop: read a line, seat it, run the model to quiescence
/// — printing every block and answering every tool call — until Ctrl-D.
async fn drive(
    cli: &Cli,
    session: &mut drama_llama::LlamaCppSession,
    prompt: &mut Prompt,
    mut bash: Option<&mut RichBash>,
    bash_methods: &HashSet<String>,
    total: &mut TokenCounts,
) -> Result<(), BoxError> {
    let max_rounds = cli.chat.max_tool_calls.unwrap_or(8);
    let forced = forced(prompt);
    // `Prompt::seat`'s pending-system buffer. Nothing here seats system
    // content, but the buffer keeps the call sites uniform.
    let mut pending = None;

    // A reseated transcript can end in an unanswered tool call — the
    // council answers filings out-of-band (as mail), but the wire
    // demands leading `tool_result`s in the next user turn and
    // `Prompt::seat` enforces it. Receipt the tail once so the first
    // interview line has a legal seat to merge into.
    let dangling: Vec<Block> = prompt
        .messages
        .last()
        .filter(|message| message.role == Role::Assistant)
        .map(|message| {
            message
                .content
                .iter()
                .filter_map(|block| block.tool_use())
                .map(|call| {
                    Block::from(tool::Result::new(call.id.clone(), CAUGHT))
                })
                .collect()
        })
        .unwrap_or_default();
    if !dangling.is_empty() {
        println!("(receipted the transcript's unanswered tool call)");
        prompt.seat((Role::User, Content(dangling)), &mut pending)?;
    }

    let mut editor = DefaultEditor::new()?;

    loop {
        let line = match editor.readline("you ▸ ") {
            Ok(line) if line.trim().is_empty() => continue,
            Ok(line) => {
                editor.add_history_entry(&line).ok();
                line
            }
            Err(ReadlineError::Interrupted) => continue,
            Err(ReadlineError::Eof) => return Ok(()),
            Err(error) => return Err(error.into()),
        };
        // Same ingest guard as the council: framing bytes in a human
        // line would be rejected at ingest anyway, so catch them as a
        // rephrase instead of an error.
        if let Some((id, _)) = session.scan_text_for_specials(&line) {
            println!(
                "✗ your message contains a reserved framing sequence \
                 (token #{id}) and cannot be relayed — please rephrase"
            );
            continue;
        }
        prompt.seat((Role::User, line), &mut pending)?;

        let mut rounds = 0usize;
        loop {
            let message = session.complete_response(prompt)?;
            *total += message.usage.counts;

            // Echo every block in order: prose as prose, tool calls as
            // pretty-printed JSON.
            let mut calls: Vec<Use> = Vec::new();
            for block in message.inner.content.iter() {
                match block.tool_use() {
                    Some(call) => {
                        println!(
                            "⚙ {} {}",
                            call.name,
                            serde_json::to_string_pretty(&call.input)?,
                        );
                        calls.push(call.clone());
                    }
                    None => println!("claude ▸ {block}"),
                }
            }
            println!("  [{}]", pay_line(&message.usage.counts));

            prompt.seat(message.inner, &mut pending)?;
            // Rolling assistant-tail breakpoint — the render the
            // session's prefix cache keys on. One-hour TTL: an
            // interviewer thinks longer than the 5-minute sweep.
            prompt.cache_windowed_with(2, CacheControl::one_hour());

            if calls.is_empty() {
                break;
            }

            // Answer every call: bash runs for real, the rest are
            // caught — printed above, receipted here — so the
            // transcript stays wire-legal.
            let mut results = Vec::with_capacity(calls.len());
            for call in calls {
                let result = match &mut bash {
                    Some(bash) if bash_methods.contains(call.name.as_ref()) => {
                        let result = bash.call(call).await;
                        println!(
                            "⚙ bash{} ▸ {}",
                            if result.is_error { " ✗" } else { "" },
                            result.content,
                        );
                        result
                    }
                    _ => {
                        println!("  (caught — receipt seated)");
                        tool::Result::new(call.id, CAUGHT)
                    }
                };
                results.push(Block::from(result));
            }
            prompt.seat((Role::User, Content(results)), &mut pending)?;

            rounds += 1;
            if forced {
                // A forced choice never goes quiet on its own: one
                // filing per beat, then back to the interviewer.
                break;
            }
            if rounds >= max_rounds {
                println!("(tool budget exhausted — back to you)");
                break;
            }
        }
    }
}

/// One usage row: the four token counters that price a turn.
fn pay_line(counts: &TokenCounts) -> String {
    format!(
        "in: {} | cache w: {} | cache r: {} | out: {}",
        counts.input_tokens,
        counts.cache_creation_input_tokens.unwrap_or(0),
        counts.cache_read_input_tokens.unwrap_or(0),
        counts.output_tokens,
    )
}
