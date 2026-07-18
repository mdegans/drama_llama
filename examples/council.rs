//! Example: a **deliberative council** — four advisors, one judge, sealed
//! simultaneous rounds. Where the [`swarm`](../swarm/index.html) wires
//! agents together with 1:1 mail (private correspondence, pipeline
//! order), the council's medium is the *record*: `artist`,
//! `philosopher`, `engineer`, and `lawyer` each file a position with the
//! [`Docket`]; the host **seals** filings until everyone expected has
//! filed. The sealed round then goes to the jester for the mandatory
//! rebuttal, returns to the advisors for one **reaction** each, and only
//! then — positions, rebuttal, reactions, complete — reaches the bench.
//! You chat with `judge`, who opens cases, reads published records,
//! calls further rounds, and rules.
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
//! The sixth seat is the **jester** — the licensed contrarian, a role
//! borrowed from Agora where the human plays it. The jester files no
//! blind position: when the last advisor files, the sealed round goes
//! to the jester *alone*, whose mandatory rebuttal (attack the
//! consensus, premises first, even if you agree) joins the record.
//! Structural, not prompted, because two live runs showed the
//! failure mode: a unanimous-wrong round 1 produces no disagreement,
//! the judge rules on agreement, and the engagement phase where trick
//! questions crack never happens. The jester guarantees every round
//! one adversarial pass — a consensus that survives it deserved to
//! win. The same principle covers the question itself: `open_case`
//! has no question parameter — the docket attaches the petitioner's
//! latest chat message **verbatim** (host-captured), because the first
//! live run's judge paraphrased the trick question into a generic
//! brief twice, once per prompting fix. Never leave to prompting what
//! you can force.
//!
//! The **reaction phase** and **judge-rules-last** apply that same
//! move to the engagement itself. A third live run (2026-07-18)
//! showed the rebuttal *working* — two advisors flipped their
//! verdicts after reading it — and the work being discarded: the
//! published round used to go to every seat at once as mail, the
//! judge's short ruling completion always out-raced the advisors'
//! long reactions through the session mutex, and the late reactions
//! bounced off an already-published round. A ruling is chat prose,
//! not a tool call, so it cannot be gated directly; what *can* be
//! host-enforced is information ordering. So the sealed round now
//! returns to the advisors after the rebuttal for one mandatory
//! reaction each (free — the book charges positions, not
//! engagement), and the bench's copy of the record is the **last to
//! arrive**. The corollary: **no seat is ever woken without an
//! action to take**. Advisors wake to file and to react, the jester
//! wakes to rebut, the judge wakes to rule — and each seat receives
//! any record it is missing (e.g. the reactions) with its next
//! duty, not as an idle notification. Deliberating into the void is
//! structurally impossible, and a round costs *fewer* model turns
//! than the broadcast design it replaces.
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
//!   nobody's transcript is poisoned. The bounce echoes the offender
//!   **defanged** (`‹tool_call›`, never the verbatim spelling): the
//!   bounce itself seats as a tool result in the author's prompt, so
//!   a verbatim quote would be the guard poisoning the very
//!   transcript it protects.
//! * **Lost advisors are excused, not awaited — a lost jester is
//!   fatal**: an advisor whose `Chat` loop dies (issues #37/#38 are
//!   live ways to die) is excused the moment its loop exits — the
//!   round advances as if the seat had answered, which matters most
//!   when the round is WITH the dead seat (run two's dead seat left
//!   the round hung with no ruling; nothing would ever have
//!   delivered to it again). Failed deliveries (`Mailbox::send`) and
//!   `/nudge <seat>` excuse as fallbacks. The jester is the one seat
//!   the council cannot lose and still be itself — deliberation
//!   without the adversarial pass is the failure mode this example
//!   exists to prevent — so a dead jester adjourns the council on
//!   the spot ([`jester_is_dead`]) rather than quietly degrading it.
//!
//! Cache shape: this example is deliberately kind to the multi-slot
//! prefix cache (one KV sequence per seat, see
//! [`CommonArgs::session_with_cache_slots`]) — every broadcast within
//! a phase is **identical text to its recipients** (the case notice
//! and the reaction request to all four advisors), so after each
//! seat's first turn every wake re-prefills only the newly delivered
//! block onto that seat's own slot. Run with
//! `DRAMA_LLAMA_CACHE_TRIPWIRE=1` to panic on any unexpected miss.
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
    sync::{Arc, Mutex, MutexGuard},
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
/// The licensed contrarian. Files no blind position: when the last
/// advisor files, the sealed round goes to the jester ALONE, whose
/// mandatory rebuttal sends it back to the advisors for reactions.
/// Structural, not prompted — two live runs showed a unanimous-wrong
/// round 1 never reaches the engagement phase where trick questions
/// crack (the judge rules on agreement), so every round is guaranteed
/// one adversarial pass — and one round of engagement with it.
/// The role is Agora-canon: the human plays it downstream.
const JESTER: &str = "jester";

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

/// Where the open case's current round stands.
#[derive(Debug, Default, PartialEq)]
enum Phase {
    /// No round in flight: no case yet, or the last round published.
    #[default]
    Idle,
    /// Advisors are filing; filings are sealed.
    Filing,
    /// Every expected advisor has filed; the sealed round is with the
    /// jester for the mandatory rebuttal. Nobody else has seen it.
    Rebuttal,
    /// The rebuttal is in; the sealed round (positions + rebuttal) is
    /// back with the advisors for one reaction each. The judge has
    /// STILL seen nothing — its copy of the record is the last to
    /// arrive, which is what makes "the judge rules last"
    /// host-enforced rather than prompted.
    Reaction,
}

/// The chamber: every piece of council state the host enforces. One
/// instance behind one mutex, shared by every [`Docket`] and [`Bench`]
/// clone. Models never hold it — sealed filings are unreachable by
/// construction, not by convention.
#[derive(Default)]
struct Chamber {
    /// Seat name → send-only mailbox handle, registered by `connect`.
    registry: HashMap<String, Mailbox>,
    /// The human's most recent chat message, captured host-side by
    /// the judge's beat closure. `open_case` attaches THIS as the
    /// case — the judge never authors the question, so it cannot be
    /// paraphrased. ("Never leave to prompting what you can force":
    /// the first live run's judge rewrote the trick question into a
    /// generic transport brief twice, once per prompting fix.)
    petition: Option<String>,
    /// The open case: the petition, verbatim, frozen at `open_case`.
    case: Option<String>,
    /// 1-based round number of the open case.
    round: u32,
    /// Where the open case's current round stands.
    phase: Phase,
    /// The jester's rebuttal for the current round, once filed.
    /// Cleared when a new round is called.
    rebuttal: Option<String>,
    /// This round's reactions to the rebuttal, one per expected
    /// advisor. The last one in publishes the record to the bench.
    /// Cleared when a new round is called.
    reactions: BTreeMap<String, String>,
    /// The rendered reactions of the last *published* round. Two
    /// consumers, each a seat's next duty-bearing wake: the next
    /// round-call notice (the advisors' only missing piece of the
    /// record) and the next sealed-round mail to the jester (whose
    /// rebuttal they answered). Set at publish; `None` when a round
    /// publishes without a rebuttal; NOT cleared by `call_round` —
    /// the jester needs it at mid-round seal time.
    last_reactions: Option<String>,
    /// Advisors expected this round — those with a filing left and a
    /// live seat when the round was called. The phase advances when
    /// every expected advisor has filed (and later reacted); a seat
    /// discovered dead is excused from the set rather than awaited.
    expected: BTreeSet<String>,
    /// This round's sealed filings. `BTreeMap` so the published
    /// record lists advisors in stable order (identical bytes to
    /// every recipient of a broadcast — the cache-friendly shape).
    filings: BTreeMap<String, String>,
    /// Advisor name → filings remaining. Only host code writes it
    /// upward (`/grant`).
    budgets: HashMap<String, u32>,
    /// Seats already auto-nudged this round. A seat whose turn ends
    /// with no tool call while its filing is due gets exactly ONE
    /// host reminder per round (a derailed model that ignores the
    /// nudge is the human's to `/nudge` — unbounded auto-nudging
    /// would burn model rounds forever).
    nudged: BTreeSet<String>,
}

/// A dead jester is fatal, not excusable: every later round would
/// publish un-rebutted, silently stripping the council of the exact
/// adversarial guarantee it exists to demonstrate. Better a loud
/// corpse than a quiet sham. (`std::process::exit`, not `panic!` — a
/// panic in a worker task is captured by the `JoinSet` and would not
/// end the run until adjourn, which is the hang this replaces.)
fn jester_is_dead(printer: &SharedPrinter) -> ! {
    printer.lock().expect("printer poisoned").line(
        "the jester is dead; the council cannot deliberate \
         un-rebutted — adjourning",
    );
    std::process::exit(1);
}

impl Chamber {
    /// The sealed round so far: positions, plus the rebuttal once it
    /// is in. One identical text for every recipient of a broadcast.
    fn publication(&self) -> String {
        let mut out = format!(
            "From: the bench\nCase: {}\nRound {} positions:\n",
            self.case.as_deref().unwrap_or("(none)"),
            self.round,
        );
        for (name, position) in &self.filings {
            out.push_str(&format!("\n## {name}\n{position}\n"));
        }
        if let Some(rebuttal) = &self.rebuttal {
            out.push_str(&format!("\n## {JESTER} (rebuttal)\n{rebuttal}\n"));
        }
        let excused: Vec<&str> = ADVISORS
            .iter()
            .copied()
            .filter(|a| !self.filings.contains_key(*a))
            .collect();
        if !excused.is_empty() {
            out.push_str(&format!(
                "\n(no filing from: {})\n",
                excused.join(", ")
            ));
        }
        out
    }

    /// The reactions of the current round, rendered bare (no intro
    /// line — each consumer supplies its own). Advisors who filed but
    /// never reacted (seat lost mid-round) are listed so their
    /// silence is on the record, not invisible.
    fn reactions_section(&self) -> String {
        let mut out = String::new();
        for (name, reaction) in &self.reactions {
            out.push_str(&format!("\n## {name} (reaction)\n{reaction}\n"));
        }
        let missing: Vec<&str> = self
            .filings
            .keys()
            .map(String::as_str)
            .filter(|a| !self.reactions.contains_key(*a))
            .collect();
        if !missing.is_empty() {
            out.push_str(&format!(
                "\n(no reaction from: {})\n",
                missing.join(", ")
            ));
        }
        out
    }

    /// Does the chamber currently owe a filing from `name`? True for
    /// an expected advisor who hasn't filed while the round is in
    /// Filing, for the jester while the round is in Rebuttal, and for
    /// an expected advisor who hasn't reacted while the round is in
    /// Reaction.
    fn filing_due(&self, name: &str) -> bool {
        match self.phase {
            Phase::Filing => {
                self.expected.contains(name) && !self.filings.contains_key(name)
            }
            Phase::Rebuttal => name == JESTER && self.rebuttal.is_none(),
            Phase::Reaction => {
                self.expected.contains(name)
                    && !self.reactions.contains_key(name)
            }
            Phase::Idle => false,
        }
    }

    /// Send `text` to every seat in `names`, returning the names that
    /// could not be delivered — a failed [`Mailbox::send`] means the
    /// seat's `Chat` loop is gone, so its mailbox is dropped from the
    /// registry too. Callers excuse the returned seats from whatever
    /// they were owed; the council degrades rather than deadlocks.
    fn broadcast<'a>(
        &mut self,
        names: impl IntoIterator<Item = &'a str>,
        text: &str,
    ) -> Vec<String> {
        let mut lost = Vec::new();
        for name in names {
            let delivered = self.registry.get(name).is_some_and(|mailbox| {
                mailbox.send(text.to_string(), vec![Role::User]).is_ok()
            });
            if !delivered {
                self.registry.remove(name);
                lost.push(name.to_string());
            }
        }
        lost
    }

    /// Excuse a lost **advisor** from the round — mailbox dropped, no
    /// longer expected — and advance the phase when that completes
    /// it: the docket never waits on a seat that cannot answer.
    /// Excusing the last awaited advisor seals or publishes as if
    /// they had filed. The jester is not excusable: reaching this
    /// with a dead jester is [`jester_is_dead`] — fatal.
    fn excuse(&mut self, name: &str, printer: &SharedPrinter) {
        self.registry.remove(name);
        self.expected.remove(name);
        match self.phase {
            Phase::Filing => {
                if self.expected.is_empty() && self.filings.is_empty() {
                    // Nobody filed and nobody remains: the round
                    // collapses. Idle so the bench can call another.
                    self.phase = Phase::Idle;
                    self.broadcast(
                        [JUDGE],
                        "From: the docket\nThe round collapsed — every \
                         expected seat was lost before filing. Tell \
                         the human.",
                    );
                } else if self
                    .expected
                    .iter()
                    .all(|a| self.filings.contains_key(a))
                {
                    self.seal_or_publish(printer);
                }
            }
            Phase::Reaction => {
                if self.expected.iter().all(|a| self.reactions.contains_key(a))
                {
                    self.publish_to_bench(printer);
                }
            }
            Phase::Rebuttal if name == JESTER => jester_is_dead(printer),
            _ => {}
        }
    }

    /// Every expected position is in: seal the round to the jester
    /// for the mandatory rebuttal. An empty jester book publishes
    /// directly to the bench with the gap on the record (the human
    /// declined to fund a rebuttal — recoverable via `/grant`); a
    /// jester whose seat is *dead* is [`jester_is_dead`] — fatal.
    fn seal_or_publish(&mut self, printer: &SharedPrinter) {
        let jester_ready = self.registry.contains_key(JESTER)
            && self.budgets.get(JESTER).copied().unwrap_or_default() > 0;
        if jester_ready {
            let mut text = self.publication();
            if let Some(reactions) = &self.last_reactions {
                text.push_str(&format!(
                    "\nThe advisors' reactions to your previous \
                     rebuttal, for the record:\n{reactions}",
                ));
            }
            text.push_str(
                "\nThe round is sealed with you before anyone else \
                 sees it. File your rebuttal with `file` now — attack \
                 the consensus (or the strongest position if they \
                 split), premises first, even if you privately agree. \
                 The round then returns to the advisors for reactions \
                 and publishes to the bench.",
            );
            self.phase = Phase::Rebuttal;
            self.nudged.remove(JESTER);
            if !self.broadcast([JESTER], &text).is_empty() {
                jester_is_dead(printer);
            }
            printer
                .lock()
                .expect("printer poisoned")
                .line(format!("⚖ round {} sealed to the jester", self.round));
            return;
        }
        // Empty book (never death — that is fatal above): the human
        // declined to fund a rebuttal, so the round publishes without
        // one, the gap on the record.
        self.publish_to_bench(printer);
    }

    /// Publish the completed round to the judge — and ONLY the judge.
    /// Every other seat already holds its share of the record (or
    /// receives the missing piece with its next duty), so no seat is
    /// woken without an action to take: the bench structurally cannot
    /// be out-raced, and nobody deliberates into the void.
    fn publish_to_bench(&mut self, printer: &SharedPrinter) {
        self.phase = Phase::Idle;
        let mut text = self.publication();
        if self.rebuttal.is_none() {
            text.push_str("\n(the jester is unavailable — no rebuttal)\n");
            self.last_reactions = None;
        } else {
            let section = self.reactions_section();
            text.push_str(&format!("\nReactions to the rebuttal:\n{section}"));
            self.last_reactions = Some(section);
        }
        self.broadcast([JUDGE], &text);
        printer
            .lock()
            .expect("printer poisoned")
            .line(format!("⚖ round {} published to the bench", self.round));
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
    ///
    /// The echo is **defanged**: the bounce seats as a tool result in
    /// the author's own prompt, so quoting the piece verbatim would
    /// re-tokenize to the special at the next call's ingest and kill
    /// the very loop the scan protects — the guard as injection
    /// vector. Angle brackets swap for lookalikes; a marker with no
    /// angles gets a zero-width space after its first character.
    async fn scan(&self, text: &str) -> Result<(), Content> {
        match self.transport.scan_text_for_specials(text).await {
            Some((_, piece)) => {
                let swapped: String = piece
                    .chars()
                    .map(|c| match c {
                        '<' => '‹',
                        '>' => '›',
                        _ => c,
                    })
                    .collect();
                let defanged = if swapped == piece {
                    let mut chars = piece.chars();
                    match chars.next() {
                        Some(first) => {
                            format!("{first}\u{200B}{}", chars.as_str())
                        }
                        None => swapped,
                    }
                } else {
                    swapped
                };
                Err(format!(
                    "your text contains the reserved framing sequence \
                     {defanged} (shown defanged), which cannot be \
                     relayed verbatim. Rephrase without it (describe \
                     it in words if you must reference it)."
                )
                .into())
            }
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
/// descriptions the model sees. Field order is load-bearing: the
/// tagged-dialect grammar emits parameters alphabetically, so
/// `analysis` is physically written before `verdict` — the model must
/// work the problem before it is allowed to conclude (reason-first,
/// enforced by structure rather than asked for in prose).
#[derive(Debug, Deserialize, JsonSchema)]
struct Filing {
    /// Work the case through your lens FIRST, in full: premises, what
    /// the words entail, concrete scenarios, anything the question
    /// smuggles in that the others may miss. The record shows only
    /// what you write here.
    analysis: String,
    /// Your conclusion, in one or two sentences, stated only after
    /// the analysis above and following from it.
    verdict: String,
}

impl Filing {
    /// The record entry: analysis first, verdict flagged for the
    /// judge's (and the human's) scanning eye.
    fn into_record(self) -> String {
        format!("{}\n\n**Verdict:** {}", self.analysis, self.verdict)
    }
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
        let briefing = if self.name == JESTER {
            format!(
                "<docket>\nYou are `{}`, the council's jester. You do \
                 not file blind: when every advisor has filed, the \
                 sealed round comes to YOU first, as mail — nobody \
                 else has seen it. Your duty is the rebuttal: argue \
                 against the consensus (or the strongest position if \
                 they split), premises first, even when you privately \
                 agree. You are not trying to be right; you are making \
                 the others prove they are. Sharp and concrete beats \
                 vague doubt. File it with `file`; the round then \
                 returns to the advisors, who must each react to your \
                 rebuttal before the record reaches the bench. Their \
                 reactions come back to you with the next sealed \
                 round. You have {} filings; each rebuttal costs one. \
                 You never speak to the human: anything not filed is \
                 lost.\n</docket>",
                self.name, budget,
            )
        } else {
            format!(
                "<docket>\nYou are `{}`, an advisor on a council. Cases \
                 arrive from the bench as mail. File your position with \
                 the docket's `file` — filings are SEALED until every \
                 advisor has filed, so round 1 is always your \
                 independent view. The council's jester then rebuts \
                 the sealed round, and it returns to YOU as mail: file \
                 ONE reaction with the same `file` — concede what the \
                 rebuttal proves, defend the rest. Only then does the \
                 full record — positions, rebuttal, reactions — reach \
                 the bench for the ruling. Positions cost one filing \
                 from your book of {}; reactions are free. If the \
                 bench calls another round, engage with the record and \
                 file again. You never speak to the human: anything \
                 not filed is lost.\n</docket>",
                self.name, budget,
            )
        };
        match prompt.system.as_mut() {
            Some(system) => {
                system.push(briefing);
            }
            None => prompt.system = Some(briefing.into()),
        }
        Ok(())
    }

    /// File your sealed position for the current round (advisors,
    /// costs one filing), your reaction once the rebuttal is in
    /// (advisors, free), or — as the jester — the mandatory rebuttal
    /// once the sealed round reaches you (costs one filing).
    #[method]
    async fn file(&mut self, filing: Filing) -> Result<Content, Content> {
        self.court.scan(&filing.analysis).await?;
        self.court.scan(&filing.verdict).await?;
        if self.name == JESTER {
            return self.file_rebuttal(filing);
        }
        let chamber = self.court.chamber.lock().expect("chamber poisoned");
        if chamber.case.is_none() {
            return Err("no case is open".into());
        }
        // One lock for dispatch AND handler: the phase cannot change
        // between the check and the write.
        match chamber.phase {
            Phase::Filing => self.file_position(chamber, filing),
            Phase::Reaction => self.file_reaction(chamber, filing),
            // Names the failure mode outright: run two (2026-07-18)
            // had an advisor invent a rebuttal from the briefing's
            // promise of one and spar with it for thirteen rounds —
            // the softer "the round is with the jester" read as
            // confirmation the jester had spoken.
            Phase::Rebuttal => Err("not accepted: the round is sealed \
                                    with the jester and NO rebuttal \
                                    has been issued yet — if you are \
                                    reacting to one, you imagined it. \
                                    Do nothing; the rebuttal arrives \
                                    as mail when the round returns to \
                                    you for reactions."
                .into()),
            Phase::Idle => Err("not accepted: this round is already \
                                published to the bench. Do nothing \
                                further until the bench calls the next \
                                round (the reactions on the record \
                                will accompany it)."
                .into()),
        }
    }

    /// The Filing-phase half of `file`: seal the position; the last
    /// expected position sends the round to the jester (or, with no
    /// jester reachable, straight to the bench).
    fn file_position(
        &self,
        mut chamber: MutexGuard<'_, Chamber>,
        filing: Filing,
    ) -> Result<Content, Content> {
        if chamber.filings.contains_key(&self.name) {
            return Err("you already filed this round — stand pat".into());
        }
        {
            let budget = chamber.budgets.entry(self.name.clone()).or_default();
            if *budget == 0 {
                return Err("out of filings — stand pat. Only the human \
                            can refill your book."
                    .into());
            }
            *budget -= 1;
        }
        chamber
            .filings
            .insert(self.name.clone(), filing.into_record());
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
        let complete = chamber
            .expected
            .iter()
            .all(|a| chamber.filings.contains_key(a));
        if complete {
            chamber.seal_or_publish(&self.court.printer);
        }
        // The ACK confirms and STOPS — no forward pointer. Run three
        // showed the model treating "after the jester's rebuttal the
        // round returns to you" as a script: it filed a "reaction" to
        // an unissued rebuttal in the very next quiesce round, 21
        // seconds after the seal.
        Ok("filed and sealed. Do NOTHING until the next mail from the \
            bench reaches you — anything you produce before then is \
            lost."
            .into())
    }

    /// The Reaction-phase half of `file`: record the advisor's
    /// engagement with the rebuttal — free, the book charges
    /// positions, not engagement. The last expected reaction
    /// publishes the full record to the bench.
    fn file_reaction(
        &self,
        mut chamber: MutexGuard<'_, Chamber>,
        filing: Filing,
    ) -> Result<Content, Content> {
        if !chamber.expected.contains(&self.name) {
            return Err("you are not part of this round — stand pat".into());
        }
        if chamber.reactions.contains_key(&self.name) {
            return Err("you already reacted this round — stand pat".into());
        }
        chamber
            .reactions
            .insert(self.name.clone(), filing.into_record());
        self.court
            .printer
            .lock()
            .expect("printer poisoned")
            .line(format!(
                "{} ⚖ reacted ({}/{})",
                self.name,
                chamber.reactions.len(),
                chamber.expected.len(),
            ));
        let complete = chamber
            .expected
            .iter()
            .all(|a| chamber.reactions.contains_key(a));
        if complete {
            chamber.publish_to_bench(&self.court.printer);
        }
        Ok("reaction filed; it publishes to the bench with the record".into())
    }

    /// The jester's half of `file`: accept the rebuttal, then send
    /// the sealed round — positions plus rebuttal — back to the
    /// advisors for their reactions. The judge sees nothing until the
    /// last reaction is in.
    fn file_rebuttal(&self, filing: Filing) -> Result<Content, Content> {
        let mut chamber = self.court.chamber.lock().expect("chamber poisoned");
        if chamber.case.is_none() {
            return Err("no case is open".into());
        }
        match chamber.phase {
            Phase::Rebuttal => {}
            Phase::Filing => {
                return Err("the advisors are still filing — the sealed \
                            round will come to you as mail"
                    .into());
            }
            Phase::Reaction => {
                return Err("your rebuttal is on the record and the \
                            advisors are reacting to it. Do nothing — \
                            their reactions come to you with the next \
                            sealed round."
                    .into());
            }
            Phase::Idle => {
                return Err("this round is already published — wait \
                            for the bench to call another"
                    .into());
            }
        }
        {
            let budget = chamber.budgets.entry(self.name.clone()).or_default();
            if *budget == 0 {
                return Err("out of filings — the round publishes \
                            without a rebuttal. Only the human can \
                            refill your book."
                    .into());
            }
            *budget -= 1;
        }
        chamber.rebuttal = Some(filing.into_record());
        self.court
            .printer
            .lock()
            .expect("printer poisoned")
            .line(format!("{JESTER} ⚖ rebutted"));
        // The round returns to the advisors, sealed: identical bytes
        // to each, one mandatory reaction owed, and a fresh nudge
        // allowance for the fresh obligation. The judge still waits.
        chamber.phase = Phase::Reaction;
        chamber.nudged.clear();
        let request = format!(
            "{}\nThe round is back with you, sealed — the bench has \
             seen nothing yet. File ONE reaction with `file` (it \
             costs no filing): concede what the rebuttal proves, \
             defend what it does not. The full record — positions, \
             rebuttal, reactions — then publishes to the bench for \
             the ruling.",
            chamber.publication(),
        );
        let expected: Vec<String> = chamber.expected.iter().cloned().collect();
        let lost =
            chamber.broadcast(expected.iter().map(String::as_str), &request);
        for name in &lost {
            self.court
                .printer
                .lock()
                .expect("printer poisoned")
                .line(format!("⚖ {name}'s seat is lost — excused"));
            chamber.excuse(name, &self.court.printer);
        }
        // `excuse` publishes on its own if the last awaited seat was
        // among the lost; announce the reaction window only if it is
        // actually open.
        if chamber.phase == Phase::Reaction {
            self.court
                .printer
                .lock()
                .expect("printer poisoned")
                .line(format!(
                    "⚖ round {} to the advisors for reactions",
                    chamber.round
                ));
        }
        Ok("rebuttal filed; the round is with the advisors for \
            reactions and publishes to the bench with them"
            .into())
    }
}

/// The judge's side of the chamber: open cases, call rounds.
struct Bench {
    court: Court,
}

/// A new case for the council. It carries NOTHING: the docket
/// attaches the petitioner's latest message verbatim — the bench
/// cannot rephrase what it never writes. (A `note` field existed
/// briefly; the judge immediately used it to smuggle its paraphrase
/// back in — "The bench adds: petitioner is deciding between
/// walking and driving..." — so it went the way of the question
/// parameter.)
#[derive(Debug, Deserialize, JsonSchema)]
struct Case {}

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

    /// Open a case: reset the docket and put the petitioner's latest
    /// message — verbatim, attached by the docket itself — to every
    /// advisor as round 1 (sealed, independent).
    #[method]
    async fn open_case(&mut self, _case: Case) -> Result<Content, Content> {
        let petition = self
            .court
            .chamber
            .lock()
            .expect("chamber poisoned")
            .petition
            .clone()
            .ok_or_else(|| {
                Content::from(
                    "no petition on record — the human hasn't asked \
                     anything yet",
                )
            })?;
        // The petition is human-authored, but the advisors' ingest
        // guard doesn't care who wrote a reserved token — bounce it
        // here too, and let the judge relay the rephrase request.
        self.court.scan(&petition).await.map_err(|_| {
            Content::from(
                "the petitioner's message contains a reserved framing \
                 token and cannot be relayed — ask them to rephrase",
            )
        })?;
        let mut chamber = self.court.chamber.lock().expect("chamber poisoned");
        chamber.case = Some(petition.clone());
        chamber.round = 1;
        chamber.phase = Phase::Filing;
        chamber.rebuttal = None;
        chamber.filings.clear();
        chamber.reactions.clear();
        chamber.last_reactions = None;
        chamber.nudged.clear();
        chamber.expected = ADVISORS
            .iter()
            .filter(|a| {
                chamber.budgets.get(**a).copied().unwrap_or_default() > 0
                    && chamber.registry.contains_key(**a)
            })
            .map(|a| a.to_string())
            .collect();
        if chamber.expected.is_empty() {
            chamber.case = None;
            chamber.phase = Phase::Idle;
            return Err("every advisor is out of filings or lost — ask \
                        the human to /grant some"
                .into());
        }
        self.court
            .printer
            .lock()
            .expect("printer poisoned")
            .line(format!("⚖ case opened: {petition}"));
        let notice = format!(
            "From: the bench\nCase (round 1), the petitioner's \
             words verbatim:\n\"{petition}\"\n\nFile your \
             independent position with `file`. Filings are sealed \
             until all of you have filed.",
        );
        let expected: Vec<String> = chamber.expected.iter().cloned().collect();
        let lost =
            chamber.broadcast(expected.iter().map(String::as_str), &notice);
        for name in &lost {
            self.court
                .printer
                .lock()
                .expect("printer poisoned")
                .line(format!("⚖ {name}'s seat is lost — excused"));
            chamber.expected.remove(name);
        }
        if chamber.expected.is_empty() {
            chamber.case = None;
            chamber.phase = Phase::Idle;
            return Err("no advisor could be reached — every seat is \
                        out of filings or lost. Tell the human."
                .into());
        }
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
        let mut chamber = self.court.chamber.lock().expect("chamber poisoned");
        if chamber.case.is_none() {
            return Err("no case is open".into());
        }
        match chamber.phase {
            Phase::Idle => {}
            Phase::Filing => {
                return Err(format!(
                    "round {} has not published yet — {}/{} filings in",
                    chamber.round,
                    chamber.filings.len(),
                    chamber.expected.len(),
                )
                .into());
            }
            Phase::Rebuttal => {
                return Err("the round is with the jester — wait \
                            for the rebuttal"
                    .into());
            }
            Phase::Reaction => {
                return Err(format!(
                    "the advisors are reacting to the rebuttal — \
                     {}/{} reactions in. The record arrives when the \
                     last one lands.",
                    chamber.reactions.len(),
                    chamber.expected.len(),
                )
                .into());
            }
        }
        chamber.round += 1;
        chamber.phase = Phase::Filing;
        chamber.rebuttal = None;
        chamber.filings.clear();
        chamber.reactions.clear();
        // `last_reactions` survives: the jester receives it with the
        // new round's seal — their answer to the previous rebuttal.
        chamber.nudged.clear();
        chamber.expected = ADVISORS
            .iter()
            .filter(|a| {
                chamber.budgets.get(**a).copied().unwrap_or_default() > 0
                    && chamber.registry.contains_key(**a)
            })
            .map(|a| a.to_string())
            .collect();
        if chamber.expected.is_empty() {
            chamber.phase = Phase::Idle;
            return Err("every advisor is out of filings or lost — rule \
                        on what you have"
                .into());
        }
        let round = chamber.round;
        self.court
            .printer
            .lock()
            .expect("printer poisoned")
            .line(format!("⚖ round {round} called"));
        let mut notice = format!(
            "From: the bench\nCase (verbatim): \"{}\"\nRound {}: {}\n",
            chamber.case.as_deref().unwrap_or("(none)"),
            round,
            call.instruction,
        );
        if let Some(reactions) = &chamber.last_reactions {
            notice.push_str(&format!(
                "\nThe reactions to the last rebuttal, completing the \
                 record you hold:\n{reactions}",
            ));
        }
        notice.push_str(
            "\nEngage with the record and file your new position with \
             `file` — concede what someone proved, defend the rest.",
        );
        let expected: Vec<String> = chamber.expected.iter().cloned().collect();
        let lost =
            chamber.broadcast(expected.iter().map(String::as_str), &notice);
        for name in &lost {
            self.court
                .printer
                .lock()
                .expect("printer poisoned")
                .line(format!("⚖ {name}'s seat is lost — excused"));
            chamber.expected.remove(name);
        }
        if chamber.expected.is_empty() {
            chamber.phase = Phase::Idle;
            return Err("no advisor could be reached — every seat is \
                        out of filings or lost. Rule on what you have."
                .into());
        }
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
        "jester" => {
            "You are `jester`, the council's licensed contrarian. The \
             other advisors file positions; you file rebuttals. \
             Whatever they agree on, you attack — starting with the \
             premises: what does the question assume that nobody \
             questioned? What reading makes the consensus absurd? \
             Argue the strongest case AGAINST the emerging answer even \
             when you find it persuasive; if the consensus survives \
             you, it deserved to win. Concrete scenarios beat \
             rhetorical doubt — one vivid case where the advice fails \
             outweighs a paragraph of maybes. You never speak to the \
             human; anything not filed is lost."
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
     stated facts entail) — plus `jester`, the licensed contrarian. \
     You do not answer questions yourself — open a case with the \
     bench's `open_case`. The docket attaches the petitioner's latest \
     message verbatim on its own — you cannot rewrite the question, \
     and you should not suggest factors to consider: each advisor \
     brings their own lens. The advisors file sealed positions, the \
     jester rebuts the sealed round, and every advisor then reacts to \
     the rebuttal — only when the last reaction is in does the record \
     (positions, rebuttal, reactions) reach you as mail. Nothing \
     arrives early, so when it arrives it is complete: rule on it. \
     Weigh the reactions hardest — an advisor who changed their \
     verdict after the rebuttal is signal, not noise, and a lone \
     advisor who noticed something concrete outranks three who \
     answered on autopilot. The rebuttal itself is structurally \
     contrarian: weigh it by what it EXPOSES, not by its conclusion. \
     If the reactions leave a genuine conflict, call ONE more round \
     with `call_round` focusing the disagreement, then rule on what \
     you have. Deliver the ruling to the human in chat, citing which \
     advisors carried it. Advisors spend limited filings on positions \
     (reactions are free); if they run dry, tell the human, who can \
     refill with `/grant <advisor> <count>`.";

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let cli = Cli::parse();
    utils::log_init(cli.common.verbose);

    // One model, one session, one cache slot per seat: each agent's
    // history pins its own KV sequence, and every within-phase
    // broadcast is identical bytes to its recipients — the
    // cache-friendly shape.
    let seats = ADVISORS.len() as u32 + 2; // advisors + jester + judge
    let mut session = cli.common.session_with_cache_slots(seats)?;
    if cli.common.max_tokens.is_none() {
        // Rulings and analyses are long-form; the session default cap
        // truncated a judge mid-ruling on the first jester run.
        session = session.with_max_tokens(
            std::num::NonZeroUsize::new(1024).expect("nonzero"),
        );
    }
    let transport = SessionTransport::new(session);

    let (mut lines, printer) = utils::spawn_readline_loop("you ▸ ")?;
    let printer: SharedPrinter = Arc::new(Mutex::new(printer));

    let chamber = Arc::new(Mutex::new(Chamber::default()));
    {
        let mut guard = chamber.lock().expect("chamber poisoned");
        for seat in ADVISORS.iter().copied().chain([JESTER]) {
            guard.budgets.insert(seat.to_string(), cli.filings);
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
        .chain([JESTER, JUDGE])
        .map(|name| (name, Arc::default()))
        .collect();

    // Build every toolbox before any Chat runs: `add` fires `connect`,
    // so the chamber's registry is complete before the first case.
    let advisor_boxes: Vec<(&str, ToolBox)> = ADVISORS
        .iter()
        .copied()
        .chain([JESTER])
        .map(|name| {
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
         engineer, lawyer file sealed positions; the jester rebuts \
         every round, the advisors react to the rebuttal, and only \
         then does the record reach the judge — who rules last. You \
         hold the treasury: `/grant <seat> <count>` refills a book, \
         `/docket` shows the chamber, `/nudge <seat>` prods (and \
         excuses a dead seat). Ctrl-D adjourns.\n",
    );

    // The advisors: headless Chats. Case notices, reaction requests,
    // and round calls are their only stimulus; the beat closure pends
    // until shutdown.
    let advisor_rounds = cli.chat.max_tool_calls.unwrap_or(16);
    let (quit, _) = tokio::sync::watch::channel(false);
    let mut council = tokio::task::JoinSet::new();
    for (name, toolbox) in advisor_boxes {
        let transport = transport.clone();
        let printer = Arc::clone(&printer);
        let usage = Arc::clone(&payroll[name]);
        let chamber_nudge = Arc::clone(&chamber);
        let chamber_reaper = Arc::clone(&chamber);
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
                        // Anti-stall: a turn that ends with NO tool
                        // call while this seat owes a filing means the
                        // model answered in prose and went idle — with
                        // publication waiting on it, that's deadlock
                        // (tool_choice can't be forced here: the
                        // post-dispatch round must be free to say
                        // nothing). One host reminder per round; a
                        // seat that ignores it is the human's to
                        // `/nudge`.
                        if msg.tool_use().is_none() {
                            let mut chamber =
                                chamber_nudge.lock().expect("chamber poisoned");
                            if chamber.filing_due(name)
                                && chamber.nudged.insert(name.to_string())
                            {
                                // The seat just produced this very
                                // message, so it is alive — ignore
                                // the delivery report.
                                let _ = chamber.broadcast(
                                    [name],
                                    "From: the bench\nThe round is \
                                     waiting on you. Nothing you say \
                                     outside the docket reaches the \
                                     council — file with `file` now.",
                                );
                            }
                        }
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
                if name == JESTER {
                    jester_is_dead(&printer);
                }
                // A dead advisor's loop can never file: excuse the
                // seat the moment it dies, so no phase waits on it.
                // Discovery at delivery / via `/nudge` still covers
                // the rest, but a seat that dies while the round is
                // WITH it is reachable by neither — run two's round
                // hung with no ruling exactly that way.
                chamber_reaper
                    .lock()
                    .expect("chamber poisoned")
                    .excuse(name, &printer);
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
                    // Host-side petition capture: whatever the human
                    // just said is what `open_case` will attach,
                    // verbatim. The judge never authors the question.
                    chamber_cli.lock().expect("chamber poisoned").petition =
                        Some(line.clone());
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
                        let missing: Vec<&str> = ADVISORS
                            .iter()
                            .copied()
                            .chain([JESTER])
                            .filter(|seat| chamber.filing_due(seat))
                            .collect();
                        let progress = match chamber.phase {
                            Phase::Reaction => format!(
                                "reacted: {}/{}",
                                chamber.reactions.len(),
                                chamber.expected.len(),
                            ),
                            _ => format!(
                                "filed: {}/{}",
                                chamber.filings.len(),
                                chamber.expected.len(),
                            ),
                        };
                        print(match chamber.case.as_deref() {
                            Some(case) => format!(
                                "case: {case} | round {} ({}) | {} | \
                                 awaiting: {} | filings: {}",
                                chamber.round,
                                match chamber.phase {
                                    Phase::Filing => "sealed",
                                    Phase::Rebuttal => "with the jester",
                                    Phase::Reaction => "awaiting reactions",
                                    Phase::Idle => "published",
                                },
                                progress,
                                if missing.is_empty() {
                                    "nobody".to_string()
                                } else {
                                    missing.join(", ")
                                },
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
                                let mut chamber = chamber_cli
                                    .lock()
                                    .expect("chamber poisoned");
                                let lost = chamber.broadcast(
                                    [name],
                                    &format!(
                                        "From: the bench\nThe human \
                                         refilled your book: +{count} \
                                         filings (balance: {balance})."
                                    ),
                                );
                                if lost.is_empty() {
                                    print(format!(
                                        "granted {name} +{count} \
                                         (balance: {balance})"
                                    ));
                                } else {
                                    print(format!(
                                        "granted {name} +{count} \
                                         (balance: {balance}) — but \
                                         the seat is lost; the grant \
                                         is on the books"
                                    ));
                                }
                            }
                            None => print(format!("no advisor named `{name}`")),
                        }
                    }
                    (Some("nudge"), Some(name), None) => {
                        let mut chamber =
                            chamber_cli.lock().expect("chamber poisoned");
                        if chamber.filing_due(name) {
                            let lost = chamber.broadcast(
                                [name],
                                "From: the bench\nThe human demands \
                                 your filing. Nothing you say outside \
                                 the docket reaches the council — file \
                                 with `file` now.",
                            );
                            if lost.is_empty() {
                                print(format!("nudged {name}"));
                            } else {
                                // The nudge found a corpse: excuse it
                                // so the round stops waiting — this is
                                // the recovery path for a seat that
                                // died AFTER its mail was delivered.
                                print(format!(
                                    "{name}'s seat is lost — excused \
                                     from the round"
                                ));
                                chamber.excuse(name, &treasury);
                            }
                        } else {
                            print(format!("no filing is due from `{name}`"));
                        }
                    }
                    _ => print(
                        "commands: /grant <seat> <count>, /docket, \
                         /nudge <seat>"
                            .into(),
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
    for name in ADVISORS.iter().copied().chain([JESTER, JUDGE]) {
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
