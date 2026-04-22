//! Manual smoke for structured output via `Prompt::output_config`.
//!
//! This variant bypasses `Session::complete_stream` and drives the
//! engine's piece-level predictor directly so we can dump each token
//! to stderr **as it's generated**. That turns a "silent for three
//! minutes" hang into a live view of cogito's thought block, which is
//! the whole point when diagnosing grammar / prompt edge cases.
//!
//! Run with:
//! ```text
//! cargo run --example whodunit --features json-schema --release -- \
//!     [path/to/model.gguf]
//! ```
//!
//! Defaults to `$PWD/models/model.gguf` if no path is given.

use std::{
    io::Write,
    num::NonZeroUsize,
    path::PathBuf,
    time::Instant,
};

use drama_llama::{
    grammar_stats_enabled, grammar_stats_reset, grammar_stats_snapshot,
    output_config, parse_completion, Block, GrammarStats, PredictOptions,
    Prompt, RenderOptions, Role, SampleOptions, Session,
};
use misanthropic::prompt::message::Content;

#[derive(schemars::JsonSchema, serde::Deserialize, Debug)]
#[allow(dead_code)]
struct Suspect {
    name: String,
    motive: String,
    had_opportunity: bool,
}

#[derive(schemars::JsonSchema, serde::Deserialize, Debug, PartialEq)]
enum Confidence {
    /// Evidence is thin; a jury would not convict.
    Low,
    /// The case is plausible but not airtight.
    Medium,
    /// The evidence conclusively identifies the culprit.
    High,
}

#[derive(schemars::JsonSchema, serde::Deserialize, Debug)]
#[allow(dead_code)]
struct CaseFile {
    suspects_considered: Vec<Suspect>,
    key_evidence: Vec<String>,
    culprit: String,
    confidence: Confidence,
    reasoning_summary: String,
}

const SCENARIO: &str = "\
Scenario: Sir Harold was found dead in his study at 11 PM, poisoned.

Suspects and verified facts:
- BUTLER (Mr. Finch): disliked Sir Harold. Served the nightcap at 9 PM, \
  but as a precaution (Sir Harold was paranoid) he took a sip from the \
  same glass in front of the house physician. He is alive and unharmed, \
  so the glass was not yet poisoned when it left his hands.
- NIECE (Lady Elsie): stood to inherit if Sir Harold died. She attended \
  the village charity gala from 8 PM to midnight; twenty named guests \
  place her there continuously. She cannot have been at the mansion.
- BUSINESS PARTNER (Mr. Crane): Sir Harold's ledger, found open on the \
  desk, showed Mr. Crane had been embezzling for two years and Sir \
  Harold intended to report him in the morning. Mr. Crane has a copy of \
  the study key (Sir Harold gave him one years ago). Two staff saw him \
  alone in the study from 10:30 to 10:50 PM. The poison is one Mr. \
  Crane keeps for his prize rose bushes.

All three had a motive. Only one had both opportunity (access to the \
glass after the butler's safe sip) AND means (possession of the \
specific poison used). Identify that suspect.";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let no_grammar = args.iter().any(|a| a == "--no-grammar");
    let path = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .cloned()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/model.gguf"));
    if !path.exists() {
        eprintln!("model not found at {path:?}");
        std::process::exit(2);
    }
    if no_grammar {
        eprintln!(
            "[setup] --no-grammar: sampling will run with default modes; \
             output is unconstrained and JSON parsing will be skipped"
        );
    }

    let mut session = Session::from_path_with_n_ctx(path, 8192)
        .expect("session load")
        .quiet();

    let prompt = Prompt::default()
        .structured_output::<CaseFile>()
        .set_system(
            "Enable deep thinking subroutine. You are a brief, \
             decisive detective. Reason inside <think>...</think> in \
             under 300 tokens: note which suspects are ruled out by \
             their alibis, identify the one remaining with motive, \
             means, and opportunity, then CLOSE the think tag. Output \
             the structured verdict as JSON matching the given schema.",
        )
        .add_message((Role::User, Content::text(SCENARIO)))
        .expect("add_message");

    // Render the prompt through the chat template. Enable cogito's
    // deep-thinking Jinja branch and generation prompt.
    let render_opts = RenderOptions::default()
        .with_generation_prompt(true)
        .with_extra("enable_thinking", true);
    let rendered = session
        .template()
        .render_with(&prompt, &render_opts)
        .expect("render");

    // Tokenize with parse_special=true so chat markers become single
    // special-token IDs (cogito's <|im_start|> etc.).
    let tokens = session.engine().model.tokenize(&rendered, true);
    eprintln!("[setup] rendered {} tokens", tokens.len());

    // Compile the output_config grammar directly.
    let grammar = output_config::grammar_for_prompt(
        &prompt,
        &output_config::OutputConfigOptions::default(),
    )
    .expect("grammar compile")
    .expect("output_config set");

    let mut predict_opts = PredictOptions::default()
        .add_model_stops(&session.engine().model);
    predict_opts.n = NonZeroUsize::new(4096).unwrap();
    predict_opts.sample_options = if no_grammar {
        SampleOptions::default()
    } else {
        SampleOptions {
            modes: vec![grammar],
            repetition: None,
        }
    };

    // Drive the piece predictor directly so we see every token as it's
    // generated. Each piece is a decoded-UTF-8 chunk from one sampled
    // token; concatenating them in order reconstructs the raw
    // completion text.
    // Compute the EOS piece BEFORE the predictor takes &mut engine —
    // driving `predict_pieces` directly bypasses `complete_stream`'s
    // stripping, so tokens like cogito's `<|im_end|>` would otherwise
    // leak into the collected text and break JSON parsing with
    // "trailing characters".
    let eos_piece = session
        .engine()
        .model
        .token_to_piece(session.engine().model.eos());

    let stats_on = grammar_stats_enabled();
    if stats_on {
        grammar_stats_reset();
        eprintln!(
            "[setup] DRAMA_LLAMA_GRAMMAR_STATS active — phase-split stats \
             will be printed at the end"
        );
    }

    let start = Instant::now();
    let predictor =
        session.engine_mut().predict_pieces(tokens, predict_opts);

    let mut full = String::new();
    let mut n_pieces = 0usize;
    // Phase transition: we treat everything until `</think>` is seen in the
    // accumulated text as phase 1 (thought block, grammar is `.+`), and
    // everything after as phase 2 (structured JSON). The split is what
    // directly sizes the thought/JSON phase-split optimization.
    let mut thought_done = false;
    let mut phase1_pieces = 0usize;
    let mut phase1_end: Option<Instant> = None;
    let mut phase1_stats: Option<GrammarStats> = None;
    for piece in predictor {
        n_pieces += 1;
        // Flush each piece so stderr isn't block-buffered when piped.
        eprint!("{piece}");
        let _ = std::io::stderr().flush();
        full.push_str(&piece);
        if !thought_done {
            phase1_pieces += 1;
            if full.contains("</think>") {
                thought_done = true;
                phase1_end = Some(Instant::now());
                if stats_on {
                    phase1_stats = Some(grammar_stats_snapshot());
                }
            }
        }
    }
    let elapsed = start.elapsed().as_secs_f32();
    eprintln!(
        "\n\n[done] {n_pieces} pieces in {elapsed:.1}s ({:.1} tok/s)",
        n_pieces as f32 / elapsed.max(0.001),
    );

    // Per-phase wall clock + tok/s.
    match phase1_end {
        Some(end) => {
            let p1 = end.duration_since(start).as_secs_f32();
            let p2 = elapsed - p1;
            let p2_pieces = n_pieces - phase1_pieces;
            eprintln!(
                "[phase1 thought ] {phase1_pieces} pieces in {p1:.2}s \
                 ({:.1} tok/s)",
                phase1_pieces as f32 / p1.max(0.001),
            );
            eprintln!(
                "[phase2 json    ] {p2_pieces} pieces in {p2:.2}s \
                 ({:.1} tok/s)",
                p2_pieces as f32 / p2.max(0.001),
            );
        }
        None => {
            eprintln!("[phase1 thought ] never observed </think>; single-phase run");
        }
    }

    if stats_on {
        let total = grammar_stats_snapshot();
        let p1 = phase1_stats.clone().unwrap_or_default();
        let p2 = GrammarStats {
            calls: total.calls - p1.calls,
            candidates_in: total.candidates_in - p1.candidates_in,
            candidates_bitmap_pass: total.candidates_bitmap_pass
                - p1.candidates_bitmap_pass,
            candidates_final_pass: total.candidates_final_pass
                - p1.candidates_final_pass,
            stacks_in_sum: total.stacks_in_sum - p1.stacks_in_sum,
            // max is not a difference; phase2's max is a lower bound of
            // the cumulative max, but we can't recover its true max
            // without per-phase reset. Report the cumulative instead.
            stacks_in_max: total.stacks_in_max,
            depth_max_sum: total.depth_max_sum - p1.depth_max_sum,
            depth_max_max: total.depth_max_max,
            filter_us_sum: total.filter_us_sum - p1.filter_us_sum,
            filter_us_max: total.filter_us_max,
            // DFA cache stats are "last-observed" style (not cumulative
            // across the whole phase), so the difference isn't meaningful.
            // Report cumulative for phase 2 too.
            dfa_states: total.dfa_states,
            dfa_transition_hits: total.dfa_transition_hits,
            dfa_transition_misses: total.dfa_transition_misses,
            dfa_bitmap_hits: total.dfa_bitmap_hits,
            dfa_bitmap_misses: total.dfa_bitmap_misses,
        };
        if phase1_stats.is_some() {
            print_stats("phase1 thought", &p1);
            print_stats("phase2 json   ", &p2);
        }
        print_stats("cumulative    ", &total);
    }

    if no_grammar {
        // Unconstrained output isn't expected to parse as CaseFile —
        // the tok/s reading is the whole point of this mode.
        return;
    }

    // Trim the trailing EOS piece if present — keeps the raw text
    // JSON-parseable.
    let trimmed = full.strip_suffix(eos_piece.as_str()).unwrap_or(&full);

    // Parse the accumulated text into Blocks and deserialize the JSON
    // body.
    let blocks = parse_completion(trimmed);
    eprintln!("[parse] {} blocks", blocks.len());

    let json_text = blocks.iter().find_map(|b| match b {
        Block::Text { text, .. } => Some(text.as_ref()),
        _ => None,
    });

    match json_text {
        Some(text) => match serde_json::from_str::<CaseFile>(text) {
            Ok(verdict) => {
                println!("\n--- parsed CaseFile ---\n{verdict:#?}");
            }
            Err(e) => {
                eprintln!("deserialize failed: {e}");
                eprintln!("raw text was:\n{text}");
                std::process::exit(1);
            }
        },
        None => {
            eprintln!(
                "no Block::Text in stream — model may not have reached JSON"
            );
            std::process::exit(1);
        }
    }
}

fn print_stats(label: &str, s: &GrammarStats) {
    if s.calls == 0 {
        eprintln!("[{label}] no filter calls");
        return;
    }
    let calls_f = s.calls as f64;
    let bitmap_survival = if s.candidates_in > 0 {
        100.0 * s.candidates_bitmap_pass as f64 / s.candidates_in as f64
    } else {
        0.0
    };
    let final_survival = if s.candidates_bitmap_pass > 0 {
        100.0 * s.candidates_final_pass as f64
            / s.candidates_bitmap_pass as f64
    } else {
        0.0
    };
    let avg_stacks = s.stacks_in_sum as f64 / calls_f;
    let avg_depth = s.depth_max_sum as f64 / calls_f;
    let avg_us = s.filter_us_sum as f64 / calls_f;
    let total_ms = s.filter_us_sum as f64 / 1000.0;
    eprintln!(
        "[{label}] calls={:>4} cand_in={:>8} bitmap_pass={:>7} ({:>5.2}%) \
         final_pass={:>6} (of bitmap {:>5.2}%) | stacks avg={:>4.1} max={:>3} \
         | depth avg={:>4.1} max={:>3} | filter avg={:>6.1}us max={:>6}us \
         total={:>6.1}ms",
        s.calls,
        s.candidates_in,
        s.candidates_bitmap_pass,
        bitmap_survival,
        s.candidates_final_pass,
        final_survival,
        avg_stacks,
        s.stacks_in_max,
        avg_depth,
        s.depth_max_max,
        avg_us,
        s.filter_us_max,
        total_ms,
    );
    let dfa_tx_total = s.dfa_transition_hits + s.dfa_transition_misses;
    let dfa_bm_total = s.dfa_bitmap_hits + s.dfa_bitmap_misses;
    if dfa_tx_total > 0 || dfa_bm_total > 0 {
        let tx_hit_rate = if dfa_tx_total > 0 {
            100.0 * s.dfa_transition_hits as f64 / dfa_tx_total as f64
        } else {
            0.0
        };
        let bm_hit_rate = if dfa_bm_total > 0 {
            100.0 * s.dfa_bitmap_hits as f64 / dfa_bm_total as f64
        } else {
            0.0
        };
        eprintln!(
            "[{label}] dfa states={:>6} tx hits/misses={}/{} ({:>5.2}% hit) \
             bitmap hits/misses={}/{} ({:>5.2}% hit)",
            s.dfa_states,
            s.dfa_transition_hits,
            s.dfa_transition_misses,
            tx_hit_rate,
            s.dfa_bitmap_hits,
            s.dfa_bitmap_misses,
            bm_hit_rate,
        );
    }
}
