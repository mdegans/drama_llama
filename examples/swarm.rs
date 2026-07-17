//! Example: a **multi-agent swarm** — a software company in miniature, one
//! `Chat` loop per agent, wired together by a mail tool built with the
//! [`tool`] macro. You chat with `boss`, who briefs the team and relays the
//! deliverable; four headless workers run the pipeline: `ant` designs,
//! `wasp` critiques (contrarian by charter — never approves a first draft),
//! `bee` implements in a sandboxed [`RichBash`], and `moth` re-runs it in a
//! *separate* sandbox, tries to break it, and signs off. Two peer loops do
//! the real work — ant↔wasp argue the design to convergence, bee↔moth
//! ping-pong code and failing cases — and neither routes through the boss.
//!
//! A letter is pushed into the recipient's [`ToolBox`] channel as a
//! [`Notification`] preferring the [`User`] role — the same push path
//! [`RichBash`] uses for background-job completions — so mail wakes the
//! recipient's driver and drives a model round, and a worker's report races
//! *your* typing at the boss's prompt exactly like any other notification.
//! The sender's identity is stamped by the tool, not the model, so a
//! `From:` line cannot be forged — which is why the boss can be told to
//! ship only what arrives signed by `moth`. Postage is the brake *and* the
//! deadline: books draw on a shared ledger, an empty book turns `send` into
//! an `is_error` result, and scarcity is what forces the design debate to
//! converge. Only the human prints postage: `/grant bee 4` at the prompt
//! credits the ledger directly — the boss can *request* a refill, but no
//! model is in the approval loop. `/stamps` shows the balances.
//!
//! Ported from `misanthropic` and driven through one local
//! [`SessionTransport`](drama_llama::SessionTransport): five `Chat` loops
//! share one model on one GPU, so inference is serialized (the transport's
//! `max_concurrency` is 1 and every clone shares the session mutex) — the
//! *agents* are concurrent, the *decodes* take turns. Every chat opts into
//! the driver's quirk-aware caching ([`Chat::cache`]): the transport
//! reports `breakpoint_after_assistant`, so the driver re-marks a rolling
//! end-of-assistant breakpoint each turn — the render the session's
//! hash-keyed prefix cache keys on. Honest caveat: the session currently
//! keeps a *single* cached prefix, so round-robining five distinct agent
//! histories through it re-prefills on every agent switch. The payroll's
//! `cache r` column shows exactly how much (or little) survived — this
//! example is the motivating benchmark for the multi-sequence cache rework.
//!
//! The point: the `Chat` driver is unchanged. N concurrent loops,
//! agent-to-agent communication, role asymmetry (thinkers get words,
//! builders get shells), and budget brakes all compose from [`Tool`],
//! [`ToolBox`], and [`Mailbox`] as they already are.
//!
//! ```sh
//! # needs Docker; the published misan-bashd image is pulled on first run
//! cargo run --example swarm --features "tokio,repl,json-schema"
//! ```
//!
//! Try: "I want a shell script that prints a histogram of word lengths in
//! a text file." Watch wasp hunt the edge cases (empty file? punctuation?
//! locales?) before bee writes a line, and moth hunt for whatever they both
//! missed. `--verbose` shows the workers thinking. All five seats run the
//! same local model — the whole experiment is whether a small model can
//! argue with itself.
//!
//! [`Chat::cache`]: misanthropic::chat::Chat::cache
//! [`Mailbox`]: misanthropic::tool::Mailbox
//! [`Notification`]: misanthropic::tool::Notification
//! [`RichBash`]: misanthropic::tool::bash::RichBash
//! [`Tool`]: misanthropic::tool::Tool
//! [`ToolBox`]: misanthropic::tool::ToolBox
//! [`User`]: misanthropic::prompt::message::Role::User
//! [`tool`]: misanthropic::tool::tool

mod utils;

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use clap::Parser;
use drama_llama::SessionTransport;
use misanthropic::{
    prompt::message::{CacheControl, Content, Role},
    response::TokenCounts,
    tool::{
        bash::{DockerSandbox, RichBash},
        tool, Mailbox, ToolBox,
    },
    Prompt,
};
use schemars::JsonSchema;
use serde::Deserialize;
use utils::{BoxError, BudgetPolicy, Printer};

/// The boss's address — the agent whose `Chat` loop is your readline seat.
const BOSS: &str = "boss";
/// The headless workers, in pipeline order: `ant` designs, `wasp`
/// critiques, `bee` implements, `moth` QAs. Each gets a [`Mail`] clone; the
/// builders (`bee`, `moth`) also get their own sandbox.
const WORKERS: [&str; 4] = ["ant", "wasp", "bee", "moth"];

/// A boss/worker agent swarm wired together by a mail tool.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    #[command(flatten)]
    common: utils::CommonArgs,
    #[command(flatten)]
    chat: utils::ChatArgs,
    /// Stamps per worker book; the boss gets one book per worker (4x).
    /// Refill a book at the prompt with `/grant <agent> <count>`.
    #[arg(long, default_value_t = 8)]
    stamps: u32,
}

/// All async output rides the single rustyline [`Printer`] (it is not
/// `Clone`, and printing takes `&mut`), shared behind a mutex. No lock is
/// ever held across an `await`.
type SharedPrinter = Arc<Mutex<Printer>>;

/// The swarm's address book: agent name → a send-only clone of that agent's
/// `ToolBox` mailbox. Registered by [`Mail::connect`], which a `ToolBox`
/// calls *inside* `add` — so building every toolbox before spawning any
/// `Chat` guarantees a complete roster before the first letter.
type Registry = Arc<Mutex<HashMap<String, Mailbox>>>;

/// The stamp accounts: agent name → stamps remaining. Shared (unlike a
/// per-clone counter) so the human can credit a book mid-run via `/grant` —
/// models spend from the ledger; only host code writes it upward.
type Ledger = Arc<Mutex<HashMap<String, u32>>>;

/// Per-agent cumulative token accounting: one `Chat::track_usage` sink per
/// seat, filled by the driver at every model round (including tool-dispatch
/// rounds no hook ever sees). Printed when the office folds.
type Payroll = HashMap<&'static str, Arc<Mutex<TokenCounts>>>;

/// Hands out [`Mail`] clones sharing one [`Registry`], one [`Ledger`], and
/// one printer.
struct PostOffice {
    registry: Registry,
    ledger: Ledger,
    printer: SharedPrinter,
}

impl PostOffice {
    fn new(printer: SharedPrinter) -> Self {
        Self {
            registry: Registry::default(),
            ledger: Ledger::default(),
            printer,
        }
    }

    /// A [`Mail`] clone for `name`, opening its account with `stamps`.
    fn address(&self, name: impl Into<String>, stamps: u32) -> Mail {
        let name = name.into();
        self.ledger
            .lock()
            .expect("ledger poisoned")
            .insert(name.clone(), stamps);
        Mail {
            name,
            registry: Arc::clone(&self.registry),
            ledger: Arc::clone(&self.ledger),
            printer: Arc::clone(&self.printer),
        }
    }
}

/// One agent's mail client. `send` looks the recipient up in the shared
/// [`Registry`] and pushes the letter into *their* `Chat` loop as a
/// `[User]`-preferred notification; the envelope's `From:` is stamped from
/// this clone's own identity, so the model cannot forge a sender. Postage
/// draws on the shared [`Ledger`].
struct Mail {
    name: String,
    registry: Registry,
    ledger: Ledger,
    printer: SharedPrinter,
}

impl Mail {
    /// Everyone reachable from this desk, sorted for stable prompts.
    fn roster(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .registry
            .lock()
            .expect("registry poisoned")
            .keys()
            .filter(|name| **name != self.name)
            .cloned()
            .collect();
        names.sort();
        names
    }

    /// Stamps left in this agent's book.
    fn balance(&self) -> u32 {
        self.ledger
            .lock()
            .expect("ledger poisoned")
            .get(&self.name)
            .copied()
            .unwrap_or_default()
    }
}

/// A letter for [`Mail::send`]. The field docs become the JSON-schema
/// property descriptions the model sees (via `schemars`).
#[derive(Debug, Deserialize, JsonSchema)]
struct Letter {
    /// The recipient agent's name.
    to: String,
    /// A one-line subject.
    subject: String,
    /// The letter itself. Include everything the recipient needs — they
    /// see only what you write here, never your conversation.
    body: String,
}

#[tool(name = "mail")]
impl Mail {
    /// Register this agent's address. The handed mailbox is a send-only
    /// handle onto *our own* box's notification channel — exactly what
    /// other agents need in order to reach us — and it fires during
    /// `ToolBox::add`, before any model call, so no letter can beat it.
    #[connect]
    fn connect(&mut self, mailbox: Mailbox) {
        self.registry
            .lock()
            .expect("registry poisoned")
            .insert(self.name.clone(), mailbox);
    }

    /// Brief the agent: identity, roster, and postage. Appends (never
    /// overwrites) so it composes with the example's persona and any other
    /// tool's contribution, in any `on_init` order.
    #[on_init]
    async fn brief(&mut self, prompt: &mut Prompt) -> Result<(), BoxError> {
        let briefing = format!(
            "<mail>\nYou are `{}`. You can write to: {}. Mail is your only \
             channel to the other agents; results travel in the letter \
             body. Your book has {} stamps and each letter costs one, so \
             make letters count. Only the human can refill a book.\n</mail>",
            self.name,
            self.roster().join(", "),
            self.balance(),
        );
        match prompt.system.as_mut() {
            Some(system) => {
                system.push(briefing);
            }
            None => prompt.system = Some(briefing.into()),
        }
        Ok(())
    }

    /// Send a letter to another agent. Costs one stamp.
    #[method]
    async fn send(&mut self, letter: Letter) -> Result<Content, Content> {
        if letter.to == self.name {
            return Err("you cannot mail yourself".into());
        }
        let recipient = self
            .registry
            .lock()
            .expect("registry poisoned")
            .get(&letter.to)
            .cloned();
        let Some(recipient) = recipient else {
            return Err(format!(
                "no agent named `{}` — the roster: {}",
                letter.to,
                self.roster().join(", "),
            )
            .into());
        };

        // Spend the stamp first (check + debit under one lock, so parallel
        // tool calls cannot double-spend); refund below if the recipient's
        // loop turns out to be gone.
        {
            let mut ledger = self.ledger.lock().expect("ledger poisoned");
            let balance = ledger.entry(self.name.clone()).or_default();
            if *balance == 0 {
                return Err("out of stamps — no more letters. If the work \
                            is unfinished, stop here: the boss can ask the \
                            human for a refill, and a refill will arrive \
                            as mail. Otherwise wrap up with what you have."
                    .into());
            }
            *balance -= 1;
        }

        // The tool composes the envelope, so `From:` is authoritative.
        let envelope = format!(
            "From: {}\nSubject: {}\n\n{}",
            self.name, letter.subject, letter.body,
        );
        if recipient.send(envelope, vec![Role::User]).is_err() {
            *self
                .ledger
                .lock()
                .expect("ledger poisoned")
                .entry(self.name.clone())
                .or_default() += 1;
            return Err(
                format!("`{}` is gone (their loop ended)", letter.to).into()
            );
        }

        self.printer.lock().expect("printer poisoned").line(format!(
            "{} ✉ {} ▸ {}",
            self.name, letter.to, letter.subject
        ));
        Ok(format!("sent ({} stamps left)", self.balance()).into())
    }
}

/// A worker's persona — the role charter. The mail briefing (identity,
/// roster, postage) is appended by [`Mail::brief`]. The design pair (`ant`,
/// `wasp`) work in words; the build pair (`bee`, `moth`) each get a
/// sandboxed bash tool.
fn worker_prompt(name: &str) -> Prompt {
    // Cache placement is the Chat driver's job here (`.cache(…)` below):
    // the local transport's quirks say breakpoints belong on the
    // end-of-assistant render, so the driver re-marks a rolling window
    // after each turn instead of anything server-side.
    let persona = match name {
        "ant" => {
            "You are `ant`, the architect in a tiny software company run \
             by `boss`. A brief from the boss starts the job. Draft a \
             design — interface first, the smallest thing that could work \
             — and mail it to `wasp` for critique. Wasp advises; you \
             decide; you own the spec. Expect two or three rounds: concede \
             what wasp proves, defend the rest, revise. If a disagreement \
             hinges on something empirical, mail `bee` a quick question — \
             bee has a shell. When the design stabilizes (or your postage \
             says it must), mail the final spec to `bee` to build. Make it \
             self-contained: bee sees only your letter. You never write \
             the implementation and never speak to the human; anything \
             not mailed is lost."
        }
        "wasp" => {
            "You are `wasp`, the design critic in a tiny software company \
             run by `boss`. Letters from `ant` carry designs; your job is \
             to make them survive contact with reality. Never approve a \
             first draft. Every reply must carry at least two concrete \
             failure scenarios, edge cases, or a sharper competing design \
             — vague praise is a wasted stamp. Concede a point only after \
             ant has specifically addressed it; never concede to be \
             agreeable. You advise; ant decides. If you two disagree on a \
             testable fact, mail `bee` to test it instead of arguing. You \
             never write the implementation and never speak to the human; \
             anything not mailed is lost."
        }
        "bee" => {
            "You are `bee`, the implementer in a tiny software company run \
             by `boss`. A spec from `ant` starts the real work: build it \
             with your sandboxed bash tool, run it, fix it until it works, \
             then mail the full source plus how you tested it to `moth` \
             for QA. If moth mails back failures, fix and resubmit to \
             moth. The designers (`ant`, `wasp`) may also mail you quick \
             empirical questions mid-design — answer those with a quick \
             test, tersely; it is not yet the build. For long-running \
             commands use `background: true` — you'll be notified when \
             the job finishes. You never speak to the human; anything not \
             mailed is lost."
        }
        "moth" => {
            "You are `moth`, QA in a tiny software company run by `boss`. \
             Letters from `bee` carry code. Re-run everything fresh in \
             your own sandboxed bash tool — trust nothing you did not run \
             yourself. Then try to break it: empty input, missing files, \
             weird flags, hostile edge cases. If it breaks, mail the \
             smallest failing case back to `bee`. When it survives you, \
             mail `boss` the deliverable with your sign-off: the final \
             source, how to run it, and what you verified. The boss ships \
             only what you sign, so sign nothing you have not run. You \
             never speak to the human; anything not mailed is lost."
        }
        other => unreachable!("no persona for `{other}`"),
    };
    Prompt::default().system(persona)
}

/// The boss's persona; the human side of the swarm.
const BOSS_SYSTEM: &str =
    "You are `boss`, the coordinator of a tiny software company. The human \
     you are chatting with is your client. Your team: `ant` designs, \
     `wasp` critiques the design, `bee` implements, `moth` tries to break \
     it and then signs off. You do not design, code, or test — turn the \
     client's ask into one clear, self-contained brief and mail it to \
     `ant`; the pipeline does the rest, and reports arrive between the \
     human's messages. Treat a deliverable as real only when it arrives in \
     a letter from `moth` — the From: line cannot be forged, so nothing \
     else counts as QA. Relay the signed deliverable to the human in chat, \
     which costs no stamps. You cannot print postage: if a worker runs dry \
     mid-job, tell the human, who can refill a book with `/grant <agent> \
     <count>`.";

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let cli = Cli::parse();
    utils::log_init(cli.common.verbose);

    // Load the local model once; every seat gets a clone of the handle
    // and serializes through the same session (and the same KV cache).
    let transport = SessionTransport::new(cli.common.session()?);

    let (mut lines, printer) = utils::spawn_readline_loop("you ▸ ")?;
    let printer: SharedPrinter = Arc::new(Mutex::new(printer));
    let office = PostOffice::new(Arc::clone(&printer));

    // One usage sink per seat; each Chat's driver adds every model round.
    let payroll: Payroll = WORKERS
        .iter()
        .copied()
        .chain([BOSS])
        .map(|name| (name, Arc::default()))
        .collect();

    // Build every toolbox before any `Chat` runs: `add` fires `connect`,
    // which registers the agent's address — so the roster is complete
    // before the first `on_init` briefing or letter. The thinkers get
    // words only; the builders each get a shell.
    let worker_boxes: Vec<(&str, ToolBox)> = WORKERS
        .iter()
        .map(|&name| {
            let toolbox = ToolBox::new().add(office.address(name, cli.stamps));
            let toolbox = if matches!(name, "bee" | "moth") {
                toolbox.add(RichBash::new(DockerSandbox::default()))
            } else {
                toolbox
            };
            (name, toolbox)
        })
        .collect();
    let boss_box = ToolBox::new()
        .add(office.address(BOSS, cli.stamps * WORKERS.len() as u32));

    printer.lock().expect("printer poisoned").line(
        "The office is open: you ↔ boss; ant designs, wasp critiques, bee \
         builds, moth QAs (bee and moth are booting sandboxes). You hold \
         the treasury: `/grant <agent> <count>` refills a book, `/stamps` \
         shows balances. Ctrl-D folds the company.\n",
    );

    // The workers: headless Chats. Mail is their only stimulus, so the
    // beat closure just pends until shutdown. `FinalWord` instead of the
    // default hand-back: a worker that silently hands back would sit idle
    // until the next letter, so let it wrap up (and mail onward) instead.
    //
    // The round cap is a runaway guard, not the budget — postage is the
    // budget. Builders burn one round per bash command (write, run, fix,
    // repeat), so the guard sits high; `--max-tool-calls` overrides it
    // (and the boss's, via `ChatArgs::configure`).
    let worker_rounds = cli.chat.max_tool_calls.unwrap_or(128);
    let (quit, _) = tokio::sync::watch::channel(false);
    let mut swarm = tokio::task::JoinSet::new();
    for (name, toolbox) in worker_boxes {
        let transport = transport.clone();
        let printer = Arc::clone(&printer);
        let usage = Arc::clone(&payroll[name]);
        let mut done = quit.subscribe();
        swarm.spawn(async move {
            let outcome =
                utils::Chat::new(transport, worker_prompt(name), toolbox)
                    .max_consecutive_tool_calls(worker_rounds)
                    .on_budget_exhausted(BudgetPolicy::FinalWord)
                    .cache(CacheControl::ephemeral())
                    .track_usage(usage)
                    .on_assistant(move |_state: &mut (), msg| {
                        // The workers' side of the story, under `--verbose`.
                        log::debug!("{name} ▸ {}", msg.content);
                        [msg.into()] // seat the turn unchanged
                    })
                    .run((), async move |_state: &mut ()| {
                        // Mail drives everything; the only beat is shutdown
                        // (a closed channel counts).
                        done.changed().await.ok();
                        Ok(None)
                    })
                    .await;
            if let Err(error) = outcome {
                printer
                    .lock()
                    .expect("printer poisoned")
                    .line(format!("☠ {name}: {error}"));
            }
        });
    }

    // The boss: your seat — the same loop as every chat example, except
    // the beat closure intercepts `/commands` before they become model
    // messages. `/grant` credits the ledger directly: refills exist, but
    // no model is in the approval loop — the same principle as the
    // stamped `From:` line. The postmaster note wakes the refilled worker.
    let ledger = Arc::clone(&office.ledger);
    let registry = Arc::clone(&office.registry);
    let treasury = Arc::clone(&printer);
    let boss_prompt =
        cli.common.configure(Prompt::default().system(BOSS_SYSTEM));
    let boss_printer = Arc::clone(&printer);
    let outcome = cli
        .chat
        .configure(utils::Chat::new(transport, boss_prompt, boss_box))
        .cache(CacheControl::ephemeral())
        .track_usage(Arc::clone(&payroll[BOSS]))
        .on_assistant(move |_state: &mut (), msg| {
            if msg.tool_use().is_none() {
                boss_printer
                    .lock()
                    .expect("printer poisoned")
                    .line(format!("\nboss ▸ {}\n", msg.content));
            }
            [msg.into()] // seat the turn unchanged
        })
        .run((), async move |_state: &mut ()| {
            while let Some(line) = lines.recv().await {
                let Some(command) = line.strip_prefix('/') else {
                    return Ok(Some(vec![(Role::User, line).into()]));
                };
                let print = |msg: String| {
                    treasury.lock().expect("printer poisoned").line(msg)
                };
                let mut words = command.split_whitespace();
                match (words.next(), words.next(), words.next()) {
                    (Some("stamps"), None, None) => {
                        let mut balances: Vec<String> = ledger
                            .lock()
                            .expect("ledger poisoned")
                            .iter()
                            .map(|(name, n)| format!("{name}: {n}"))
                            .collect();
                        balances.sort();
                        print(balances.join(" | "));
                    }
                    (Some("grant"), Some(name), Some(count)) => {
                        let Ok(count) = count.parse::<u32>() else {
                            print(format!("`{count}` is not a count"));
                            continue;
                        };
                        let balance = {
                            let mut ledger =
                                ledger.lock().expect("ledger poisoned");
                            match ledger.get_mut(name) {
                                Some(balance) => {
                                    *balance += count;
                                    *balance
                                }
                                None => {
                                    print(format!("no agent named `{name}`"));
                                    continue;
                                }
                            }
                        };
                        // Wake the recipient: an empty book left them with
                        // no way to act, so the refill itself is mail.
                        let mailbox = registry
                            .lock()
                            .expect("registry poisoned")
                            .get(name)
                            .cloned();
                        if let Some(mailbox) = mailbox {
                            let _ = mailbox.send(
                                format!(
                                    "From: postmaster\nSubject: postage\n\n\
                                     The human refilled your book: +{count} \
                                     stamps (balance: {balance}). Carry on."
                                ),
                                vec![Role::User],
                            );
                        }
                        print(format!(
                            "granted {name} +{count} (balance: {balance})"
                        ));
                    }
                    _ => print(
                        "commands: /grant <agent> <count>, /stamps".into(),
                    ),
                }
            }
            Ok(None)
        })
        .await;

    // Fold the company: wake every worker's beat, then wait for their
    // loops to finish so `ToolBox` teardown stops the sandboxes.
    let _ = quit.send(true);
    while let Some(joined) = swarm.join_next().await {
        joined?;
    }
    outcome?;

    // The payroll: what each seat cost. Locally, `cache r` is the prefix
    // the session actually reused — watch it collapse when agent switches
    // thrash the single-slot cache (the rework's benchmark, see the
    // module docs).
    println!("── payroll ──────────────────────────────────────────");
    let mut total = TokenCounts::default();
    for name in WORKERS.iter().copied().chain([BOSS]) {
        let counts = *payroll[name].lock().expect("payroll poisoned");
        println!("{name:<6} {}", pay_line(&counts));
        total += counts;
    }
    println!("total  {}", pay_line(&total));
    println!("the office is closed");
    Ok(())
}

/// One payroll row: the four token counters that price a seat.
fn pay_line(counts: &TokenCounts) -> String {
    format!(
        "in: {} | cache w: {} | cache r: {} | out: {}",
        counts.input_tokens,
        counts.cache_creation_input_tokens.unwrap_or(0),
        counts.cache_read_input_tokens.unwrap_or(0),
        counts.output_tokens,
    )
}
