//! An example council that will deliberate and answer your questions, in some
//! cases delivering better performance than some very large models.
//!
//! ## Composition
//!
//! There are six agents with different assigned personas. *Artist, Philosopher,
//! Engineer, Lawyer* and a contrarian *Jester*. The only purpose of the *Judge*
//! is to relay the decision of the council.
//!
//! ## Flow
//!
//! 1. The first round is blind. Simulacra deliberate independently.
//! 2. At the end of a the first round a contrarian Jester shakes things up.
//! 3. Another round of deliberation.
//! 4. Judge rules, delivers your answer.
//!
//! ```sh
//! cargo run --example council --features tokio,repl,json-schema
//! ```
//!
//! **Try the trick question**: *"The car wash is only 100m away from my house.
//! Should I walk or drive?"* (The expected answer is that the purpose of the
//! trip is probably to wash the car, so the car must be driven there.)

mod utils;

use std::collections::BTreeMap;

use clap::{CommandFactory, FromArgMatches, Parser};
use drama_llama::LlamaCppSession;
use misanthropic::{
    prompt::message::{CacheControl, Role},
    response::TokenCounts,
    tool::{self, CustomMethodDef},
    Prompt,
};
use serde::Deserialize;
use utils::BoxError;

/// The judge: rules last, on the complete record, in prose.
const JUDGE: &str = "judge";
/// The advisors, each a lens: `artist` (lived experience),
/// `philosopher` (assumptions and consistency), `engineer` (mechanics
/// and failure modes), `lawyer` (what the stated facts entail).
const ADVISORS: [&str; 4] = ["artist", "philosopher", "engineer", "lawyer"];
/// The licensed contrarian: rebuts every sealed round before the
/// advisors react and the judge sees a word.
const JESTER: &str = "jester";
/// The one tool every non-judge seat is forced through.
const FILE_TOOL: &str = "file";

/// A judge/advisor council, host-driven: sealed filings → rebuttal →
/// reactions → ruling, one forced completion per seat per phase.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    #[command(flatten)]
    common: utils::CommonArgs,
}

/// A filed entry: position, rebuttal, or reaction — the phase is
/// carried by the mail, not the schema, so the rendered tool prefix
/// stays byte-stable across every round (schema-per-phase would
/// invalidate the seat's cache slot each time). Field order is
/// load-bearing: the tagged-dialect grammar emits parameters
/// alphabetically, so `analysis` is physically written before
/// `verdict` — the model must work the problem before it is allowed
/// to conclude (reason-first, enforced by structure).
#[derive(Debug, Deserialize, derive_more::Display)]
#[display("{}\n\n**Verdict:** {}", self.analysis, self.verdict)]
struct Filing {
    /// Work the case through your lens FIRST, in full.
    analysis: String,
    /// The conclusion, stated only after the analysis.
    verdict: String,
}

/// The tool definition every non-judge seat carries. The schema is
/// hand-written (not schemars-derived) so the bytes rendered into the
/// prompt — part of each seat's cached prefix — are exactly what you
/// see here.
fn file_tool() -> CustomMethodDef {
    CustomMethodDef::builder(FILE_TOOL)
        .description(
            "File your entry with the docket: work the analysis in \
             full first, then state the verdict.",
        )
        .schema(serde_json::json!({
            "type": "object",
            "properties": {
                "analysis": {
                    "type": "string",
                    "description": "Work the case through your lens \
                        FIRST, in full: premises, what the words \
                        entail, concrete scenarios, anything the \
                        question smuggles in that the others may \
                        miss. The record shows only what you write \
                        here."
                },
                "verdict": {
                    "type": "string",
                    "description": "Your conclusion, in one or two \
                        sentences, stated only after the analysis \
                        above and following from it."
                }
            },
            "required": ["analysis", "verdict"]
        }))
        .build()
        .expect("the docket tool definition is valid")
}

/// A seat at the council: a name, a host-owned append-only
/// transcript, and its payroll line. The host is the only writer —
/// which is the whole corpse-impossibility argument.
struct Seat {
    name: &'static str,
    prompt: Prompt,
    usage: TokenCounts,
}

impl Seat {
    /// A filing seat: persona system prompt, the `file` tool, forced.
    fn filer(name: &'static str, system: String) -> Self {
        let mut prompt = Prompt::default().system(system).add_tool(file_tool());
        prompt.tool_choice = Some(tool::Choice::method(FILE_TOOL));
        Seat {
            name,
            prompt,
            usage: TokenCounts::default(),
        }
    }

    /// The judge: no tools, no forced choice — the ruling is prose.
    fn judge(prompt: Prompt) -> Self {
        Seat {
            name: JUDGE,
            prompt,
            usage: TokenCounts::default(),
        }
    }
}

/// One deliberation round of the open case, as recorded by the host.
struct Round {
    /// Advisor → filed position, in stable (alphabetical) order.
    filings: BTreeMap<&'static str, String>,
    /// The jester's rebuttal of the sealed round.
    rebuttal: String,
    /// Advisor → reaction to the rebuttal.
    reactions: BTreeMap<&'static str, String>,
}

/// An advisor's persona: the lens charter. Adapted from the Agora
/// council's role prompts — each voice gets a mandate, a way of
/// looking, and an explicit license to disagree.
fn persona(name: &str) -> &'static str {
    match name {
        "artist" => {
            "You are `artist`, one of four advisors on a council. Your \
             lens is lived experience: what the situation actually \
             feels like to a person inside it, the texture the abstract \
             framing leaves out. You notice what everyone is too \
             sensible to mention. You are not naive — but when the \
             obvious answer feels spiritually wrong, you say so and say \
             why. The council is strongest when each voice is genuinely \
             distinct, and you are expected to disagree when your lens \
             demands it."
        }
        "philosopher" => {
            "You are `philosopher`, one of four advisors on a council. \
             Your lens is assumptions and consistency: before answering \
             the question, ask what the question is actually asking, \
             and what it quietly presupposes. You stress-test framings \
             — if a premise is smuggled in, surface it; if the answer \
             changes when a hidden assumption flips, lead with that. \
             Rigorous but not cold. The council is strongest when each \
             voice is genuinely distinct, and you are expected to \
             disagree when your lens demands it."
        }
        "engineer" => {
            "You are `engineer`, one of four advisors on a council. \
             Your lens is mechanics: how the thing actually works, what \
             the physical steps are, where it fails. Walk the process \
             end to end before opining — most bad advice dies on \
             contact with the actual procedure. Give the honest \
             reading, not the optimistic one. The council is strongest \
             when each voice is genuinely distinct, and you are \
             expected to disagree when your lens demands it."
        }
        "lawyer" => {
            "You are `lawyer`, one of four advisors on a council. Your \
             lens is precision about the stated facts: read the words \
             as written, enumerate what they entail, and refuse to \
             answer a different question than the one asked. When the \
             facts themselves determine the answer, say so plainly and \
             show the entailment. The council is strongest when each \
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
             outweighs a paragraph of maybes."
        }
        other => unreachable!("no persona for `{other}`"),
    }
}

/// The docket note appended to every filing seat's persona: how to
/// answer. Deliberately says nothing about *when* phases happen — the
/// mail carries the phase, and a promised future is an invitation to
/// simulate it (the run-three lesson).
const DOCKET_NOTE: &str =
    "\n\nCases and instructions arrive as mail from the bench. Answer \
     every mail by filing with `file`: work the analysis in full \
     first, then the verdict. The record shows only what you file.";

/// The judge's persona. No tools: opening cases and calling rounds
/// belong to the human steward; the judge's one job is the ruling.
const JUDGE_SYSTEM: &str =
    "You are `judge`, the bench of a small council. The human you are \
     chatting with is the petitioner. Your advisors: `artist` (lived \
     experience), `philosopher` (assumptions and consistency), \
     `engineer` (mechanics and failure modes), `lawyer` (what the \
     stated facts entail) — plus `jester`, the licensed contrarian \
     who rebuts every round before the advisors react. The complete \
     record — positions, rebuttal, reactions, possibly several rounds \
     — arrives as mail. Nothing arrives early, so when it arrives it \
     is complete: rule on it. Weigh the reactions hardest — an \
     advisor who changed their verdict after the rebuttal is signal, \
     not noise, and a lone advisor who noticed something concrete \
     outranks three who answered on autopilot. The rebuttal itself is \
     structurally contrarian: weigh it by what it EXPOSES, not by its \
     conclusion. Deliver the ruling to the petitioner in chat, citing \
     which advisors carried it.";

/// Defang a reserved special's spelling for any text a model or human
/// might see again: angle brackets swap for lookalikes; a marker with
/// no angles gets a zero-width space after its first character. The
/// guard must never carry the poison it guards against.
fn defang(piece: &str) -> String {
    let swapped: String = piece
        .chars()
        .map(|c| match c {
            '<' => '‹',
            '>' => '›',
            _ => c,
        })
        .collect();
    if swapped != piece {
        return swapped;
    }
    let mut chars = piece.chars();
    match chars.next() {
        Some(first) => format!("{first}\u{200B}{}", chars.as_str()),
        None => swapped,
    }
}

/// TEMPORARY diagnostic (cache-reads investigation): every completion
/// after a seat's first must resume from its slot's cached prefix —
/// the transcript is append-only and the tail carries 1-hour markers,
/// so a zero cache-read is exactly the miss the tripwire should have
/// caught. Panic loudly with the counts so the failing turn is
/// attributable. Remove once the payroll numbers make sense.
fn assert_cache_hit(seat: &Seat, counts: &TokenCounts, first: bool) {
    if first {
        return;
    }
    assert!(
        counts.cache_read_input_tokens.unwrap_or(0) > 0,
        "☠ {}: no prefix-cache read on a follow-up call ({})",
        seat.name,
        pay_line(counts),
    );
}

/// One forced completion for a filing seat: mail in, [`Filing`] plus
/// the turn's [`TokenCounts`] out. The transcript gains the assistant
/// turn ONLY on full success — on any error nothing model-authored is
/// ever seated, so a failed call cannot poison the seat. Errors carry
/// the seat name; the caller bails (any completion failure is a
/// programmer error, not a council event).
fn file_call(
    session: &mut LlamaCppSession,
    seat: &mut Seat,
    mail: String,
) -> Result<(Filing, TokenCounts), BoxError> {
    let first_call = !seat
        .prompt
        .messages
        .iter()
        .any(|m| m.role == Role::Assistant);
    seat.prompt.messages.push((Role::User, mail).into());
    let message = session
        .complete_response(&seat.prompt)
        .map_err(|e| format!("☠ {}: {e}", seat.name))?;
    let counts = message.usage.counts;
    assert_cache_hit(seat, &counts, first_call);
    seat.usage += counts;
    log::debug!("{} ▸ {message}", seat.name);
    let call = message.tool_use().ok_or_else(|| {
        // Unreachable on the forced path (the session reports a
        // grammar violation instead) — but never silently absent.
        format!("☠ {}: forced call produced no tool use", seat.name)
    })?;
    let filing: Filing = serde_json::from_value(call.input.clone())
        .map_err(|e| format!("☠ {}: unparseable filing: {e}", seat.name))?;
    // Relay guard (#37 residual): the grammar forces the call's
    // SHAPE, but frame-marker bytes are legal inside argument
    // strings, and this text is about to be rendered into other
    // seats' prompts — where ingest would reject it. Scan at the
    // source and adjourn naming the author.
    for (field, text) in
        [("analysis", &filing.analysis), ("verdict", &filing.verdict)]
    {
        if let Some((id, piece)) = session.scan_text_for_specials(text) {
            return Err(format!(
                "☠ {}: {field} contains the reserved framing sequence \
                 {} (token #{id}) — grammar-legal in an argument \
                 string (#37) but not relayable; adjourning",
                seat.name,
                defang(&piece),
            )
            .into());
        }
    }
    seat.prompt.messages.push(message.inner.into());
    // Assistant-tail breakpoint, right after generation — the same
    // placement the `Chat` driver derives from the transport's
    // `breakpoint_after_assistant` quirk, matching the snapshot the
    // session takes internally at end of generation. One-hour TTL,
    // not ephemeral: a seat sits idle while the other five take their
    // turns, easily past the 5-minute sweep — which would drop its
    // snapshots and re-prefill the whole transcript every phase.
    seat.prompt.cache_windowed_with(2, CacheControl::one_hour());
    Ok((filing, counts))
}

/// The judge's completion: mail in, prose ruling (plus the turn's
/// [`TokenCounts`]) out. Same append-only, success-only transcript
/// discipline as [`file_call`].
fn rule_call(
    session: &mut LlamaCppSession,
    judge: &mut Seat,
    mail: String,
) -> Result<(String, TokenCounts), BoxError> {
    let first_call = !judge
        .prompt
        .messages
        .iter()
        .any(|m| m.role == Role::Assistant);
    judge.prompt.messages.push((Role::User, mail).into());
    let message = session
        .complete_response(&judge.prompt)
        .map_err(|e| format!("☠ {}: {e}", judge.name))?;
    let counts = message.usage.counts;
    assert_cache_hit(judge, &counts, first_call);
    judge.usage += counts;
    let ruling = message.to_string();
    judge.prompt.messages.push(message.inner.into());
    // One-hour tail window; see `file_call` for why not ephemeral.
    judge
        .prompt
        .cache_windowed_with(2, CacheControl::one_hour());
    Ok((ruling, counts))
}

/// The positions of a round, rendered in stable order — identical
/// bytes to every recipient of the same phase mail.
fn positions_text(
    case: &str,
    round: u32,
    filings: &BTreeMap<&'static str, String>,
) -> String {
    let mut out = format!(
        "From: the bench\nCase: \"{case}\"\nRound {round} positions:\n"
    );
    for (name, position) in filings {
        out.push_str(&format!("\n## {name}\n{position}\n"));
    }
    out
}

/// The reactions of a round, rendered bare — consumers add their own
/// intro line.
fn reactions_text(reactions: &BTreeMap<&'static str, String>) -> String {
    let mut out = String::new();
    for (name, reaction) in reactions {
        out.push_str(&format!("\n## {name} (reaction)\n{reaction}\n"));
    }
    out
}

/// The complete record of a case, every round, for the judge.
fn record_text(case: &str, rounds: &[Round]) -> String {
    let mut out = format!(
        "From: the bench\nThe complete record. Case, the petitioner's \
         words verbatim:\n\"{case}\"\n"
    );
    for (i, round) in rounds.iter().enumerate() {
        let n = i as u32 + 1;
        out.push_str(&format!("\n# Round {n}\n"));
        for (name, position) in &round.filings {
            out.push_str(&format!("\n## {name}\n{position}\n"));
        }
        out.push_str(&format!(
            "\n## {JESTER} (rebuttal)\n{}\n",
            round.rebuttal
        ));
        out.push_str("\nReactions to the rebuttal:\n");
        out.push_str(&reactions_text(&round.reactions));
    }
    out.push_str("\nThe record is complete. Rule.");
    out
}

/// Scan human-authored text before it enters any prompt; `Err` is the
/// rephrase message. The ONE recoverable error in the program — the
/// human can retype, a model cannot.
fn scan_human(session: &LlamaCppSession, text: &str) -> Result<(), String> {
    match session.scan_text_for_specials(text) {
        Some((id, piece)) => Err(format!(
            "your message contains the reserved framing sequence {} \
             (token #{id}) and cannot be relayed — please rephrase",
            defang(&piece),
        )),
        None => Ok(()),
    }
}

fn main() -> Result<(), BoxError> {
    // `--n-ctx` is the whole unified KV budget across all six slots;
    // the shared 8192 example default starved multi-round cases, so
    // this example quadruples it. `mut_arg` (not a post-parse patch)
    // so `--help` prints the default that will actually be used.
    let cli = Cli::from_arg_matches(
        &Cli::command()
            .mut_arg("n_ctx", |a| a.default_value("32768"))
            .get_matches(),
    )?;
    utils::log_init(cli.common.verbose);

    // One model, one session, one cache slot per seat.
    let seats = ADVISORS.len() as u32 + 2; // advisors + jester + judge
    let mut session = cli.common.session_with_cache_slots(seats)?;
    if cli.common.max_tokens.is_none() {
        // Filings and rulings are long-form; 1024 truncated a jester
        // rebuttal, 2048 an engineer reaction (runs five and six) —
        // on the forced path that is a typed GrammarViolation and
        // adjourns the council. Not 8192: `check_context_fit`
        // reserves this much headroom per call out of the shared
        // `--n-ctx` budget above.
        session = session.with_max_tokens(
            std::num::NonZeroUsize::new(4096).expect("nonzero"),
        );
    }

    let mut advisors: Vec<Seat> = ADVISORS
        .iter()
        .copied()
        .map(|name| {
            Seat::filer(name, format!("{}{DOCKET_NOTE}", persona(name)))
        })
        .collect();
    let mut jester =
        Seat::filer(JESTER, format!("{}{DOCKET_NOTE}", persona(JESTER)));
    let mut judge = Seat::judge(
        cli.common.configure(Prompt::default().system(JUDGE_SYSTEM)),
    );

    let mut editor = rustyline::DefaultEditor::new()?;
    println!(
        "The council is seated: artist, philosopher, engineer, lawyer \
         file sealed positions; the jester rebuts every round; the \
         advisors react; the judge rules last, on the complete \
         record. You are the petitioner and the steward — your \
         question opens a case, and between rounds you decide whether \
         deliberation continues. Ctrl-D adjourns.\n"
    );

    'petitions: loop {
        // ── The petition ─────────────────────────────────────────
        let petition = match editor.readline("you ▸ ") {
            Ok(line) if line.trim().is_empty() => continue,
            Ok(line) => {
                editor.add_history_entry(&line).ok();
                line
            }
            Err(rustyline::error::ReadlineError::Interrupted) => continue,
            Err(_) => break, // Ctrl-D / EOF: adjourn
        };
        if let Err(rephrase) = scan_human(&session, &petition) {
            println!("⚖ {rephrase}");
            continue;
        }
        println!("⚖ case opened: {petition}");

        let mut rounds: Vec<Round> = Vec::new();
        let mut instruction: Option<String> = None;
        loop {
            let n = rounds.len() as u32 + 1;

            // The council deliberates alone with sealed filings
            let notice = match (&instruction, rounds.last()) {
                (Some(instruction), Some(prev)) => format!(
                    "From: the bench\nCase (verbatim): \
                     \"{petition}\"\nRound {n}: {instruction}\n\nThe \
                     reactions to the last rebuttal, completing the \
                     record you hold:\n{}\nEngage with the record and \
                     file your new position.",
                    reactions_text(&prev.reactions),
                ),
                _ => format!(
                    "From: the bench\nCase (round 1), the petitioner's \
                     words verbatim:\n\"{petition}\"\n\nFile your \
                     independent position. Yours is the only voice \
                     you will see until the round is published.",
                ),
            };
            let mut filings: BTreeMap<&'static str, String> = BTreeMap::new();
            for seat in advisors.iter_mut() {
                let (filing, counts) =
                    file_call(&mut session, seat, notice.clone())?;
                println!(
                    "{} ⚖ filed [{}] — {}",
                    seat.name,
                    pay_line(&counts),
                    filing.verdict
                );
                filings.insert(seat.name, filing.to_string());
            }

            // The contrarian Jester responds
            let mut seal = positions_text(&petition, n, &filings);
            if let Some(prev) = rounds.last() {
                seal.push_str(&format!(
                    "\nThe advisors' reactions to your previous \
                     rebuttal, for the record:\n{}",
                    reactions_text(&prev.reactions),
                ));
            }
            seal.push_str(
                "\nThe round is sealed with you before anyone else \
                 sees it. File your rebuttal — attack the consensus \
                 (or the strongest position if they split), premises \
                 first, even if you privately agree.",
            );
            let (rebuttal, counts) =
                file_call(&mut session, &mut jester, seal)?;
            println!(
                "{JESTER} ⚖ rebutted [{}] — {}",
                pay_line(&counts),
                rebuttal.verdict
            );

            // The council integrates the rebuttal
            let react_mail = format!(
                "{}\n## {JESTER} (rebuttal)\n{rebuttal}\n\nThe round \
                 is back with you, sealed — the bench has seen \
                 nothing yet. File ONE reaction: concede what the \
                 rebuttal proves, defend what it does not.",
                positions_text(&petition, n, &filings),
            );
            let mut reactions: BTreeMap<&'static str, String> = BTreeMap::new();
            for seat in advisors.iter_mut() {
                let (reaction, counts) =
                    file_call(&mut session, seat, react_mail.clone())?;
                println!(
                    "{} ⚖ reacted [{}] — {}",
                    seat.name,
                    pay_line(&counts),
                    reaction.verdict
                );
                reactions.insert(seat.name, reaction.to_string());
            }

            rounds.push(Round {
                filings,
                rebuttal: rebuttal.to_string(),
                reactions,
            });

            // Additional human input here if needed
            instruction = loop {
                let line = match editor.readline(
                    "⚖ enter = send to the judge | instruction = \
                     another round ▸ ",
                ) {
                    Ok(line) => line,
                    Err(rustyline::error::ReadlineError::Interrupted) => {
                        continue;
                    }
                    Err(_) => break 'petitions, // adjourn mid-case
                };
                if line.trim().is_empty() {
                    break None;
                }
                match scan_human(&session, &line) {
                    Ok(()) => {
                        editor.add_history_entry(&line).ok();
                        break Some(line);
                    }
                    Err(rephrase) => println!("⚖ {rephrase}"),
                }
            };
            if instruction.is_none() {
                break;
            }
        }

        // ── The ruling: the judge sees the record LAST ───────────
        let (ruling, counts) = rule_call(
            &mut session,
            &mut judge,
            record_text(&petition, &rounds),
        )?;
        println!("\njudge [{}] ▸ {ruling}\n", pay_line(&counts));
    }

    // ── The payroll ──────────────────────────────────────────────
    println!("── payroll ──────────────────────────────────────────");
    let mut total = TokenCounts::default();
    for seat in advisors.iter().chain([&jester, &judge]) {
        println!("{:<12} {}", seat.name, pay_line(&seat.usage));
        total += seat.usage;
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
