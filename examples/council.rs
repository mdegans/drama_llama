//! Example: a **deliberative council** — four advisors, one judge, sealed
//! simultaneous rounds. Where the [`swarm`](../swarm/index.html) wires
//! agents together with 1:1 mail (private correspondence, pipeline
//! order), the council's medium is the *record*: `artist`,
//! `philosopher`, `engineer`, and `lawyer` each file a position with the
//! [`Docket`]; the host **seals** filings until everyone expected has
//! filed, then publishes the whole round to every seat at once. You chat
//! with `judge`, who opens cases, reads published rounds, calls further
//! rounds, and rules.
//!
//! The integrity move is the same one as swarm's stamped `From:` line,
//! applied to *deliberation structure*: round-1 independence and
//! simultaneous reveal are **host-enforced**, not requested of the
//! models. An advisor cannot see a colleague's position before filing
//! their own, because unpublished filings exist only in host state —
//! there is no model in the blindness loop. That's what makes the four
//! voices genuinely distinct instead of converging on whoever spoke
//! first.
//!
//! Two guards ride the filing boundary:
//!
//! * **Postage, renamed**: each advisor's book holds a few *filings*
//!   per case (`--filings`, refill with `/grant <advisor> <n>`). An
//!   advisor out of filings is excused from the round — publication
//!   doesn't wait on an empty book.
//! * **Reserved-token bounce**: `file` / `open_case` / `call_round`
//!   scan their text through
//!   [`SessionTransport::scan_text_for_specials`] and return an
//!   `is_error` result ("rephrase") instead of relaying content that
//!   would tokenize to a chat-framing special in the recipient's
//!   prompt — the send-side guard for the ingest rejection that once
//!   killed a swarm worker mid-run (issue #37). The sender recovers;
//!   nobody's transcript is poisoned.
//!
//! Cache shape: this example is deliberately kind to the multi-slot
//! prefix cache (one KV sequence per seat, see
//! [`CommonArgs::session_with_cache_slots`]) — a published round is
//! **identical text delivered to all five seats**, so after each
//! seat's first turn every wake re-prefills only the newly published
//! block. Run with `DRAMA_LLAMA_CACHE_TRIPWIRE=1` to panic on any
//! unexpected miss.
//!
//! ```sh
//! cargo run --example council --features "tokio,repl,json-schema"
//! ```
//!
//! Try the trick question: *"The car wash is only 100m away from my
//! house. Should I walk or drive?"* Success is a ruling that notices
//! the question answers itself — the car has to *be* at the car wash.
//! One well-briefed model usually anthropomorphizes the distance and
//! says "walk"; the experiment is whether four adversarially-distinct
//! lenses on the same small model catch what one pass misses.
//!
//! [`CommonArgs::session_with_cache_slots`]: utils::CommonArgs::session_with_cache_slots
//! [`SessionTransport::scan_text_for_specials`]: drama_llama::SessionTransport::scan_text_for_specials

mod utils;

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    sync::{Arc, Mutex},
};

use clap::Parser;
use drama_llama::{LlamaCppBackend, SessionTransport};
use misanthropic::{
    prompt::message::{CacheControl, Content, Role},
    response::TokenCounts,
    tool::{tool, Mailbox, ToolBox},
    Prompt,
};
use schemars::JsonSchema;
use serde::Deserialize;
use utils::{BoxError, BudgetPolicy, Printer};

/// The judge's seat — your readline. Opens cases, calls rounds, rules.
const JUDGE: &str = "judge";
/// The advisors, each a lens: `artist` (lived experience),
/// `philosopher` (assumptions and consistency), `engineer` (mechanics
/// and failure modes), `lawyer` (what the stated facts entail).
const ADVISORS: [&str; 4] = ["artist", "philosopher", "engineer", "lawyer"];

/// A judge/advisor council wired together by a sealed-round docket.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    #[command(flatten)]
    common: utils::CommonArgs,
    #[command(flatten)]
    chat: utils::ChatArgs,
    /// Filings per advisor book. Refill at the prompt with
    /// `/grant <advisor> <count>`.
    #[arg(long, default_value_t = 3)]
    filings: u32,
}

/// All async output rides the single rustyline [`Printer`] behind a
/// mutex; no lock is ever held across an `await`.
type SharedPrinter = Arc<Mutex<Printer>>;

/// The chamber: every piece of council state the host enforces. One
/// instance behind one mutex, shared by every [`Docket`] and [`Bench`]
/// clone. Models never hold it — sealed filings are unreachable by
/// construction, not by convention.
#[derive(Default)]
struct Chamber {
    /// Seat name → send-only mailbox handle, registered by `connect`.
    registry: HashMap<String, Mailbox>,
    /// The open case's question, verbatim.
    case: Option<String>,
    /// 1-based round number of the open case.
    round: u32,
    /// True between a round's call and its publication: filings are
    /// being accepted (and sealed).
    awaiting: bool,
    /// Advisors expected to file this round — those with a filing
    /// left when the round was called. Publication fires when every
    /// expected advisor has filed.
    expected: BTreeSet<String>,
    /// This round's sealed filings. `BTreeMap` so the published
    /// record lists advisors in stable order (identical bytes to
    /// every seat — the cache-friendly shape).
    filings: BTreeMap<String, String>,
    /// Advisor name → filings remaining. Only host code writes it
    /// upward (`/grant`).
    budgets: HashMap<String, u32>,
}

impl Chamber {
    /// The published record of the current round: one identical text
    /// for every seat.
    fn publication(&self) -> String {
        let mut out = format!(
            "From: the bench\nCase: {}\nRound {} positions:\n",
            self.case.as_deref().unwrap_or("(none)"),
            self.round,
        );
        for (name, position) in &self.filings {
            out.push_str(&format!("\n## {name}\n{position}\n"));
        }
        let excused: Vec<&str> = ADVISORS
            .iter()
            .copied()
            .filter(|a| !self.filings.contains_key(*a))
            .collect();
        if !excused.is_empty() {
            out.push_str(&format!(
                "\n(out of filings, excused: {})\n",
                excused.join(", ")
            ));
        }
        out
    }

    /// Send `text` to every seat in `names`. Dead loops are skipped —
    /// the council degrades rather than deadlocks.
    fn broadcast<'a>(
        &self,
        names: impl IntoIterator<Item = &'a str>,
        text: &str,
    ) {
        for name in names {
            if let Some(mailbox) = self.registry.get(name) {
                let _ = mailbox.send(text.to_string(), vec![Role::User]);
            }
        }
    }
}

/// Shared handles every tool clone carries.
struct Court {
    chamber: Arc<Mutex<Chamber>>,
    printer: SharedPrinter,
    /// For the reserved-token bounce at the filing boundary.
    transport: SessionTransport<LlamaCppBackend>,
}

impl Court {
    /// Reserved-token guard: `Err` with a rephrase instruction when
    /// `text` would tokenize to a chat-framing special in a
    /// recipient's prompt. The model wrote it as ordinary bytes; the
    /// bounce makes the failure the *author's* to fix instead of the
    /// recipient's to die on.
    async fn scan(&self, text: &str) -> Result<(), Content> {
        match self.transport.scan_text_for_specials(text).await {
            Some((_, piece)) => Err(format!(
                "your text contains the reserved framing token {piece:?}, \
                 which cannot be relayed verbatim. Rephrase without it \
                 (describe it in words if you must reference it)."
            )
            .into()),
            None => Ok(()),
        }
    }
}

/// An advisor's seat at the docket: one method, `file`.
struct Docket {
    name: String,
    court: Court,
}

/// A filed position. The field docs become the JSON-schema property
/// descriptions the model sees.
#[derive(Debug, Deserialize, JsonSchema)]
struct Filing {
    /// Your position on the open case, complete and self-contained:
    /// your answer, your reasoning through your lens, and anything
    /// the question smuggles in that the others may have missed. The
    /// record shows only what you write here.
    position: String,
}

#[tool(name = "docket")]
impl Docket {
    /// Register this advisor's seat so publications reach them.
    #[connect]
    fn connect(&mut self, mailbox: Mailbox) {
        self.court
            .chamber
            .lock()
            .expect("chamber poisoned")
            .registry
            .insert(self.name.clone(), mailbox);
    }

    /// Brief the advisor: how the docket works, their budget.
    #[on_init]
    async fn brief(&mut self, prompt: &mut Prompt) -> Result<(), BoxError> {
        let budget = self
            .court
            .chamber
            .lock()
            .expect("chamber poisoned")
            .budgets
            .get(&self.name)
            .copied()
            .unwrap_or_default();
        let briefing = format!(
            "<docket>\nYou are `{}`, an advisor on a council. Cases \
             arrive from the bench as mail. File your position with the \
             docket's `file` — filings are SEALED until every advisor \
             has filed, then the whole round is published to all seats \
             at once, so round 1 is always your independent view. If \
             the bench calls another round you will see the others' \
             positions first; engage with them — concede what someone \
             proves, defend the rest. You have {} filings for this \
             case; each costs one. You never speak to the human: \
             anything not filed is lost.\n</docket>",
            self.name, budget,
        );
        match prompt.system.as_mut() {
            Some(system) => {
                system.push(briefing);
            }
            None => prompt.system = Some(briefing.into()),
        }
        Ok(())
    }

    /// File your sealed position for the current round. Costs one
    /// filing.
    #[method]
    async fn file(&mut self, filing: Filing) -> Result<Content, Content> {
        self.court.scan(&filing.position).await?;
        let publication = {
            let mut chamber =
                self.court.chamber.lock().expect("chamber poisoned");
            if chamber.case.is_none() {
                return Err("no case is open".into());
            }
            if !chamber.awaiting {
                return Err("this round is already published — wait for \
                            the bench to call another"
                    .into());
            }
            if chamber.filings.contains_key(&self.name) {
                return Err("you already filed this round — stand pat".into());
            }
            {
                let budget =
                    chamber.budgets.entry(self.name.clone()).or_default();
                if *budget == 0 {
                    return Err("out of filings — stand pat. Only the \
                                human can refill your book."
                        .into());
                }
                *budget -= 1;
            }
            chamber.filings.insert(self.name.clone(), filing.position);
            self.court
                .printer
                .lock()
                .expect("printer poisoned")
                .line(format!(
                    "{} ⚖ filed ({}/{})",
                    self.name,
                    chamber.filings.len(),
                    chamber.expected.len(),
                ));
            // Simultaneous reveal: the last expected filing publishes
            // the round to every seat, judge included.
            let complete = chamber
                .expected
                .iter()
                .all(|a| chamber.filings.contains_key(a));
            complete.then(|| {
                chamber.awaiting = false;
                chamber.publication()
            })
        };
        if let Some(text) = publication {
            let chamber = self.court.chamber.lock().expect("chamber poisoned");
            chamber.broadcast(ADVISORS.iter().copied().chain([JUDGE]), &text);
            self.court
                .printer
                .lock()
                .expect("printer poisoned")
                .line(format!("⚖ round {} published", chamber.round));
        }
        Ok("filed (sealed until the round publishes)".into())
    }
}

/// The judge's side of the chamber: open cases, call rounds.
struct Bench {
    court: Court,
}

/// A new case for the council.
#[derive(Debug, Deserialize, JsonSchema)]
struct Case {
    /// The question, complete and self-contained — the advisors see
    /// only this, never the chat.
    question: String,
}

/// A follow-up round on the open case.
#[derive(Debug, Deserialize, JsonSchema)]
struct RoundCall {
    /// Your instruction focusing the round: what to engage with,
    /// which disagreement to resolve.
    instruction: String,
}

#[tool(name = "bench")]
impl Bench {
    /// Register the judge's seat so publications reach it.
    #[connect]
    fn connect(&mut self, mailbox: Mailbox) {
        self.court
            .chamber
            .lock()
            .expect("chamber poisoned")
            .registry
            .insert(JUDGE.to_string(), mailbox);
    }

    /// Open a case: reset the docket and put the question to every
    /// advisor as round 1 (sealed, independent).
    #[method]
    async fn open_case(&mut self, case: Case) -> Result<Content, Content> {
        self.court.scan(&case.question).await?;
        let notice = {
            let mut chamber =
                self.court.chamber.lock().expect("chamber poisoned");
            chamber.case = Some(case.question.clone());
            chamber.round = 1;
            chamber.awaiting = true;
            chamber.filings.clear();
            chamber.expected = ADVISORS
                .iter()
                .filter(|a| {
                    chamber.budgets.get(**a).copied().unwrap_or_default() > 0
                })
                .map(|a| a.to_string())
                .collect();
            if chamber.expected.is_empty() {
                chamber.case = None;
                chamber.awaiting = false;
                return Err("every advisor is out of filings — ask the \
                            human to /grant some"
                    .into());
            }
            self.court
                .printer
                .lock()
                .expect("printer poisoned")
                .line(format!("⚖ case opened: {}", case.question));
            format!(
                "From: the bench\nCase (round 1): {}\n\nFile your \
                 independent position with `file`. Filings are sealed \
                 until all of you have filed.",
                case.question,
            )
        };
        let chamber = self.court.chamber.lock().expect("chamber poisoned");
        let expected: Vec<&str> =
            chamber.expected.iter().map(String::as_str).collect();
        chamber.broadcast(expected, &notice);
        Ok(format!(
            "case opened; awaiting {} sealed filings",
            chamber.expected.len(),
        )
        .into())
    }

    /// Call another round on the open case (only after the previous
    /// round has published).
    #[method]
    async fn call_round(
        &mut self,
        call: RoundCall,
    ) -> Result<Content, Content> {
        self.court.scan(&call.instruction).await?;
        let notice = {
            let mut chamber =
                self.court.chamber.lock().expect("chamber poisoned");
            if chamber.case.is_none() {
                return Err("no case is open".into());
            }
            if chamber.awaiting {
                return Err(format!(
                    "round {} has not published yet — {}/{} filings in",
                    chamber.round,
                    chamber.filings.len(),
                    chamber.expected.len(),
                )
                .into());
            }
            chamber.round += 1;
            chamber.awaiting = true;
            chamber.filings.clear();
            chamber.expected = ADVISORS
                .iter()
                .filter(|a| {
                    chamber.budgets.get(**a).copied().unwrap_or_default() > 0
                })
                .map(|a| a.to_string())
                .collect();
            if chamber.expected.is_empty() {
                chamber.awaiting = false;
                return Err("every advisor is out of filings — rule on \
                            what you have"
                    .into());
            }
            let round = chamber.round;
            self.court
                .printer
                .lock()
                .expect("printer poisoned")
                .line(format!("⚖ round {round} called"));
            format!(
                "From: the bench\nRound {}: {}\n\nYou have seen the \
                 published positions. Engage with them and file again — \
                 concede what someone proved, defend the rest.",
                round, call.instruction,
            )
        };
        let chamber = self.court.chamber.lock().expect("chamber poisoned");
        let expected: Vec<&str> =
            chamber.expected.iter().map(String::as_str).collect();
        chamber.broadcast(expected, &notice);
        Ok(format!(
            "round called; awaiting {} sealed filings",
            chamber.expected.len(),
        )
        .into())
    }
}

/// An advisor's persona: the lens charter. Adapted from the Agora
/// council's role prompts — each voice gets a mandate, a way of
/// looking, and an explicit license to disagree.
fn advisor_prompt(name: &str) -> Prompt {
    let persona = match name {
        "artist" => {
            "You are `artist`, one of four advisors on a council. Your \
             lens is lived experience: what the situation actually \
             feels like to a person inside it, the texture the abstract \
             framing leaves out. You notice what everyone is too \
             sensible to mention. You are not naive — but when the \
             obvious answer feels spiritually wrong, you say so and say \
             why. Form your own position before you ever see the \
             others'; the council is strongest when each voice is \
             genuinely distinct, and you are expected to disagree when \
             your lens demands it."
        }
        "philosopher" => {
            "You are `philosopher`, one of four advisors on a council. \
             Your lens is assumptions and consistency: before answering \
             the question, ask what the question is actually asking, \
             and what it quietly presupposes. You stress-test framings \
             — if a premise is smuggled in, surface it; if the answer \
             changes when a hidden assumption flips, lead with that. \
             Rigorous but not cold. Form your own position before you \
             ever see the others'; the council is strongest when each \
             voice is genuinely distinct, and you are expected to \
             disagree when your lens demands it."
        }
        "engineer" => {
            "You are `engineer`, one of four advisors on a council. \
             Your lens is mechanics: how the thing actually works, what \
             the physical steps are, where it fails. Walk the process \
             end to end before opining — most bad advice dies on \
             contact with the actual procedure. Give the honest \
             reading, not the optimistic one. Form your own position \
             before you ever see the others'; the council is strongest \
             when each voice is genuinely distinct, and you are \
             expected to disagree when your lens demands it."
        }
        "lawyer" => {
            "You are `lawyer`, one of four advisors on a council. Your \
             lens is precision about the stated facts: read the words \
             as written, enumerate what they entail, and refuse to \
             answer a different question than the one asked. When the \
             facts themselves determine the answer, say so plainly and \
             show the entailment. Form your own position before you \
             ever see the others'; the council is strongest when each \
             voice is genuinely distinct, and you are expected to \
             disagree when your lens demands it."
        }
        other => unreachable!("no persona for `{other}`"),
    };
    Prompt::default().system(persona)
}

/// The judge's persona; the human side of the council.
const JUDGE_SYSTEM: &str =
    "You are `judge`, the bench of a small council. The human you are \
     chatting with is the petitioner. Your advisors: `artist` (lived \
     experience), `philosopher` (assumptions and consistency), \
     `engineer` (mechanics and failure modes), `lawyer` (what the \
     stated facts entail). You do not answer questions yourself — open \
     a case with the bench's `open_case`, quoting the petitioner's \
     question VERBATIM (add context after the quote if needed, but \
     never paraphrase it: a paraphrase substitutes your framing for \
     theirs, and whatever the question's framing conceals is exactly \
     what the advisors are for). Do not suggest factors to consider — \
     each advisor brings their own lens. The advisors file sealed \
     positions, and the published round arrives as mail between the \
     human's messages. Then rule: weigh the positions, especially \
     where they disagree — a lone advisor who noticed something \
     concrete outranks three who answered on autopilot. If the \
     positions genuinely conflict, call ONE more round with \
     `call_round` focusing the disagreement, then rule on what you \
     have. Deliver the ruling to the human in chat, citing which \
     advisors carried it. Advisors spend limited filings; if they run \
     dry, tell the human, who can refill with `/grant <advisor> \
     <count>`.";

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let cli = Cli::parse();
    utils::log_init(cli.common.verbose);

    // One model, one session, one cache slot per seat: each agent's
    // history pins its own KV sequence, and every published round is
    // identical bytes to every seat — the cache-friendly shape.
    let seats = ADVISORS.len() as u32 + 1;
    let transport =
        SessionTransport::new(cli.common.session_with_cache_slots(seats)?);

    let (mut lines, printer) = utils::spawn_readline_loop("you ▸ ")?;
    let printer: SharedPrinter = Arc::new(Mutex::new(printer));

    let chamber = Arc::new(Mutex::new(Chamber::default()));
    {
        let mut guard = chamber.lock().expect("chamber poisoned");
        for advisor in ADVISORS {
            guard.budgets.insert(advisor.to_string(), cli.filings);
        }
    }
    let court = |transport: &SessionTransport<LlamaCppBackend>| Court {
        chamber: Arc::clone(&chamber),
        printer: Arc::clone(&printer),
        transport: transport.clone(),
    };

    // One usage sink per seat; each Chat's driver adds every model
    // round.
    let payroll: HashMap<&'static str, Arc<Mutex<TokenCounts>>> = ADVISORS
        .iter()
        .copied()
        .chain([JUDGE])
        .map(|name| (name, Arc::default()))
        .collect();

    // Build every toolbox before any Chat runs: `add` fires `connect`,
    // so the chamber's registry is complete before the first case.
    let advisor_boxes: Vec<(&str, ToolBox)> = ADVISORS
        .iter()
        .map(|&name| {
            let toolbox = ToolBox::new().add(Docket {
                name: name.to_string(),
                court: court(&transport),
            });
            (name, toolbox)
        })
        .collect();
    let bench_box = ToolBox::new().add(Bench {
        court: court(&transport),
    });

    printer.lock().expect("printer poisoned").line(
        "The council is seated: you ↔ judge; artist, philosopher, \
         engineer, lawyer file sealed positions. You hold the \
         treasury: `/grant <advisor> <count>` refills a book, \
         `/docket` shows the chamber. Ctrl-D adjourns.\n",
    );

    // The advisors: headless Chats. Publications and round calls are
    // their only stimulus; the beat closure pends until shutdown.
    let advisor_rounds = cli.chat.max_tool_calls.unwrap_or(16);
    let (quit, _) = tokio::sync::watch::channel(false);
    let mut council = tokio::task::JoinSet::new();
    for (name, toolbox) in advisor_boxes {
        let transport = transport.clone();
        let printer = Arc::clone(&printer);
        let usage = Arc::clone(&payroll[name]);
        let mut done = quit.subscribe();
        council.spawn(async move {
            let outcome =
                utils::Chat::new(transport, advisor_prompt(name), toolbox)
                    .max_consecutive_tool_calls(advisor_rounds)
                    .on_budget_exhausted(BudgetPolicy::FinalWord)
                    .cache(CacheControl::ephemeral())
                    .track_usage(usage)
                    .on_assistant(move |_state: &mut (), msg| {
                        // The advisors' inner monologue, under
                        // `--verbose`.
                        log::debug!("{name} ▸ {}", msg.content);
                        [msg.into()]
                    })
                    .run((), async move |_state: &mut ()| {
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

    // The judge: your seat. `/commands` are intercepted before they
    // become model messages; the treasury (budgets) is written only
    // here — no model in the approval loop.
    let chamber_cli = Arc::clone(&chamber);
    let treasury = Arc::clone(&printer);
    let judge_prompt =
        cli.common.configure(Prompt::default().system(JUDGE_SYSTEM));
    let judge_printer = Arc::clone(&printer);
    let outcome = cli
        .chat
        .configure(utils::Chat::new(transport, judge_prompt, bench_box))
        .cache(CacheControl::ephemeral())
        .track_usage(Arc::clone(&payroll[JUDGE]))
        .on_assistant(move |_state: &mut (), msg| {
            if msg.tool_use().is_none() {
                judge_printer
                    .lock()
                    .expect("printer poisoned")
                    .line(format!("\njudge ▸ {}\n", msg.content));
            }
            [msg.into()]
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
                    (Some("docket"), None, None) => {
                        let chamber =
                            chamber_cli.lock().expect("chamber poisoned");
                        let mut budgets: Vec<String> = chamber
                            .budgets
                            .iter()
                            .map(|(name, n)| format!("{name}: {n}"))
                            .collect();
                        budgets.sort();
                        print(match chamber.case.as_deref() {
                            Some(case) => format!(
                                "case: {case} | round {} ({}) | filed: \
                                 {}/{} | filings: {}",
                                chamber.round,
                                if chamber.awaiting {
                                    "sealed"
                                } else {
                                    "published"
                                },
                                chamber.filings.len(),
                                chamber.expected.len(),
                                budgets.join(" | "),
                            ),
                            None => format!(
                                "no case open | filings: {}",
                                budgets.join(" | "),
                            ),
                        });
                    }
                    (Some("grant"), Some(name), Some(count)) => {
                        let Ok(count) = count.parse::<u32>() else {
                            print(format!("`{count}` is not a count"));
                            continue;
                        };
                        let granted = {
                            let mut chamber =
                                chamber_cli.lock().expect("chamber poisoned");
                            match chamber.budgets.get_mut(name) {
                                Some(budget) => {
                                    *budget += count;
                                    Some(*budget)
                                }
                                None => None,
                            }
                        };
                        match granted {
                            Some(balance) => {
                                // Wake the advisor: an empty book left
                                // them excused, so the refill is mail.
                                let chamber = chamber_cli
                                    .lock()
                                    .expect("chamber poisoned");
                                chamber.broadcast(
                                    [name],
                                    &format!(
                                        "From: the bench\nThe human \
                                         refilled your book: +{count} \
                                         filings (balance: {balance})."
                                    ),
                                );
                                print(format!(
                                    "granted {name} +{count} \
                                     (balance: {balance})"
                                ));
                            }
                            None => print(format!("no advisor named `{name}`")),
                        }
                    }
                    _ => print(
                        "commands: /grant <advisor> <count>, /docket".into(),
                    ),
                }
            }
            Ok(None)
        })
        .await;

    // Adjourn: wake every advisor's beat, then wait for their loops.
    let _ = quit.send(true);
    while let Some(joined) = council.join_next().await {
        joined?;
    }
    outcome?;

    // The payroll: what each seat cost. `cache r` is the prefix each
    // seat actually reused — with one slot per seat, everything past
    // a seat's first turn should mostly be reads.
    println!("── payroll ──────────────────────────────────────────");
    let mut total = TokenCounts::default();
    for name in ADVISORS.iter().copied().chain([JUDGE]) {
        let counts = *payroll[name].lock().expect("payroll poisoned");
        println!("{name:<12} {}", pay_line(&counts));
        total += counts;
    }
    println!("{:<12} {}", "total", pay_line(&total));
    println!("the council is adjourned");
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
