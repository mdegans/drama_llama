//! How faithfully does a model copy identifiers while writing long-form
//! text, under different repetition-penalty settings?
//!
//! The prompt is a fixed, Agora-shaped dashboard: posts with UUIDs,
//! kebab-case author handles, community slugs and RFC 3339 timestamps;
//! governance-log entries with `GOV-2026-0006`-style ids and ISO dates;
//! prose that mentions English dates. The model is asked for a digest that
//! cites every one of them — a thousand-odd tokens of forced verbatim
//! repetition, which is exactly what a repetition penalty is built to
//! discourage. Each variant runs once per seed on one loaded session, and
//! the output is scored against the ids the prompt actually contains.
//!
//! ```sh
//! just example id_fidelity --seeds 1,2,3 --out target/id_fidelity
//! ```
//!
//! # Variants
//!
//! Every variant starts from the model's sampling sidecar (or `--sampling`)
//! so the sampling chain is identical across them; only the repetition
//! penalty differs.
//!
//! - `old` — the pre-#113 sidecar: ignore English/JSON/punctuation (not
//!   numbers), and only the two original id patterns (UUIDs, dotted
//!   `GOV-2026.6`), which match none of this prompt's governance ids.
//! - `new` — the sidecar as it stands.
//! - `new-min1` — `new` with `ngram_min_size = 1`.
//! - `none` — no repetition penalty at all: the control.
//!
//! # Scoring
//!
//! The known sets are extracted from the rendered prompt with the same
//! regexes that score the output, so the two cannot disagree about what a
//! "UUID" or a "date" is. "bad" means shape-valid but not in the prompt (a
//! miscopy); "mal" means recognizably an attempt at the id but not
//! shape-valid. Bare 8-hex prefixes are counted only when not all-decimal,
//! so plain numbers are not mistaken for truncated UUIDs. `loop` is the
//! number of words covered by the longest immediately repeated word n-gram
//! (`the the the` scores 3; zero when nothing repeats back to back).
//!
//! Wall time includes prefill, which the prefix cache makes a one-off:
//! expect the first run to be the slow one.

use std::fmt::Write as _;
use std::num::{NonZeroU128, NonZeroU32};
use std::path::PathBuf;
use std::time::Instant;

use clap::{ArgAction, Parser};
use drama_llama::{Block, LlamaCppSession, Prompt, Role};
use misanthropic::prompt::thinking::Thinking;

#[path = "utils/fidelity.rs"]
mod fidelity;
use fidelity::{
    agora_handles, dashboard, default_model_path, load_base_config, Known,
    Patterns, Penalty as Variant, Score, DIGEST_ASK, ID_COLUMNS, SYSTEM,
    TEXT_COLUMNS,
};

#[derive(Parser, Debug)]
struct Args {
    /// Path to the `.gguf` model.
    #[arg(short, long, default_value_os_t = default_model_path())]
    model: PathBuf,

    /// Sampling sidecar every variant starts from. Defaults to the model's
    /// own `<model>.sampling.toml`.
    #[arg(long)]
    sampling: Option<PathBuf>,

    /// Which repetition-penalty variants to run.
    #[arg(long, value_delimiter = ',', default_values_t = Variant::ALL)]
    variants: Vec<Variant>,

    /// RNG seeds; each variant runs once per seed.
    #[arg(long, value_delimiter = ',', default_values_t = [
        NonZeroU128::new(1).unwrap(),
        NonZeroU128::new(2).unwrap(),
        NonZeroU128::new(3).unwrap(),
    ])]
    seeds: Vec<NonZeroU128>,

    /// Generation cap, thought included.
    #[arg(long, default_value_t = NonZeroU32::new(1200).unwrap())]
    max_tokens: NonZeroU32,

    /// Let the model think first (`--thinking false` to disable). The
    /// thought is scored along with the answer, and separately.
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    thinking: bool,

    /// KV context size.
    #[arg(long, default_value_t = 16384)]
    n_ctx: u32,

    /// Write each generation to `<dir>/<variant>-<seed>.txt`.
    #[arg(long)]
    out: Option<PathBuf>,

    /// Load nothing: re-score the `--variants` × `--seeds` files a previous
    /// `--out` wrote to this directory (for iterating on the scorer). The
    /// run columns (tokens, stop, secs) read zero.
    #[arg(long)]
    rescore: Option<PathBuf>,
}

/// `(header, decimals)`, after the run columns (variant, seed, tokens,
/// stop, secs).
fn columns() -> impl Iterator<Item = &'static (&'static str, usize)> {
    ID_COLUMNS.iter().chain(TEXT_COLUMNS)
}

fn cells(score: &Score) -> Vec<f64> {
    let mut cells = score.id_cells();
    cells.extend(score.text_cells());
    cells
}

// ---------------------------------------------------------------------------
// Running and reporting

struct Run {
    variant: Variant,
    seed: NonZeroU128,
    tokens: u64,
    hit_max: bool,
    secs: f64,
    all: Score,
    thought: Score,
    answer: Score,
}

fn print_table(title: &str, runs: &[Run], pick: fn(&Run) -> &Score) {
    println!("\n== {title} ==");
    let mut header = format!(
        "{:<9} {:>4} {:>5} {:>4} {:>6}",
        "variant", "seed", "tok", "stop", "secs"
    );
    for (name, _) in columns() {
        let _ = write!(header, " {name:>5}");
    }
    println!("{header}");

    let row =
        |label: &str, seed: &str, run: [f64; 3], stop: &str, cells: &[f64]| {
            let mut line = format!(
                "{label:<9} {seed:>4} {:>5.0} {stop:>4} {:>6.1}",
                run[0], run[2]
            );
            for ((_, prec), v) in columns().zip(cells) {
                let _ = write!(line, " {v:>5.prec$}");
            }
            println!("{line}");
        };

    for variant in Variant::ALL {
        let mine: Vec<&Run> =
            runs.iter().filter(|r| r.variant == variant).collect();
        if mine.is_empty() {
            continue;
        }
        let mut sum = vec![0.0; columns().count()];
        let mut run_sum = [0.0; 3];
        for r in &mine {
            let cells = cells(pick(r));
            let stop = if r.hit_max { "max" } else { "end" };
            let run = [r.tokens as f64, 0.0, r.secs];
            row(variant.name(), &r.seed.to_string(), run, stop, &cells);
            sum.iter_mut().zip(&cells).for_each(|(s, c)| *s += c);
            run_sum.iter_mut().zip(run).for_each(|(s, c)| *s += c);
        }
        let n = mine.len() as f64;
        let maxed = mine.iter().filter(|r| r.hit_max).count();
        let mean: Vec<f64> = sum.iter().map(|s| s / n).collect();
        let run_mean = run_sum.map(|s| s / n);
        row(
            "  mean",
            "",
            run_mean,
            &format!("{maxed}/{}", mine.len()),
            &mean,
        );
    }
}

impl Run {
    fn new(
        variant: Variant,
        seed: NonZeroU128,
        (tokens, hit_max, secs): (u64, bool, f64),
        thought: &str,
        answer: &str,
        (re, known): (&Patterns, &Known),
    ) -> Self {
        Self {
            variant,
            seed,
            tokens,
            hit_max,
            secs,
            all: Score::of(&format!("{thought}\n{answer}"), re, known),
            thought: Score::of(thought, re, known),
            answer: Score::of(answer, re, known),
        }
    }
}

const THOUGHT_HEADER: &str = "=== thought ===\n";
const ANSWER_HEADER: &str = "\n=== answer ===\n";

/// Run every variant × seed on one loaded session.
fn generate(
    args: &Args,
    prompt_text: String,
    scorer: (&Patterns, &Known),
) -> Result<Vec<Run>, Box<dyn std::error::Error>> {
    let mut session =
        LlamaCppSession::from_path_with_n_ctx(args.model.clone(), args.n_ctx)?
            .quiet();

    // The sidecar is read *after* the load, which writes a default one if
    // the model had none.
    let base = load_base_config(&args.model, args.sampling.as_deref())?;

    let mut prompt = Prompt::default()
        .system(SYSTEM)
        .max_tokens(args.max_tokens)
        .add_message((Role::User, prompt_text))?;
    if args.thinking {
        prompt = prompt.thinking(Thinking::Adaptive { display: None });
    }

    if let Some(dir) = &args.out {
        std::fs::create_dir_all(dir)?;
    }

    let mut runs = Vec::new();
    for &variant in &args.variants {
        let mut config = base.clone();
        config.repetition = variant.repetition(&base.repetition);
        session = session.with_sample_options(config);

        for &seed in &args.seeds {
            session = session.with_seed(Some(seed));
            eprint!("{variant} seed {seed} ... ");
            let start = Instant::now();
            let blocks = session.complete_blocks(&prompt)?;
            let secs = start.elapsed().as_secs_f64();
            let tokens = session.last_usage().output_tokens;
            eprintln!("{tokens} tokens in {secs:.1}s");

            let (mut thought, mut answer) = (String::new(), String::new());
            for block in &blocks {
                match block {
                    Block::Thought { thought: t, .. } => {
                        thought.push_str(t);
                        thought.push('\n');
                    }
                    Block::Text { text, .. } => answer.push_str(text),
                    _ => {}
                }
            }
            if let Some(dir) = &args.out {
                std::fs::write(
                    dir.join(format!("{variant}-{seed}.txt")),
                    format!("{THOUGHT_HEADER}{thought}{ANSWER_HEADER}{answer}"),
                )?;
            }

            let hit_max = tokens >= u64::from(args.max_tokens.get());
            runs.push(Run::new(
                variant,
                seed,
                (tokens, hit_max, secs),
                &thought,
                &answer,
                scorer,
            ));
        }
    }
    Ok(runs)
}

/// Re-score what a previous `--out` wrote.
fn rescore(
    args: &Args,
    dir: &std::path::Path,
    scorer: (&Patterns, &Known),
) -> Result<Vec<Run>, Box<dyn std::error::Error>> {
    let mut runs = Vec::new();
    for &variant in &args.variants {
        for &seed in &args.seeds {
            let path = dir.join(format!("{variant}-{seed}.txt"));
            let file = std::fs::read_to_string(&path)?;
            let body = file.strip_prefix(THOUGHT_HEADER).unwrap_or(&file);
            let (thought, answer) =
                body.split_once(ANSWER_HEADER).unwrap_or(("", body));
            runs.push(Run::new(
                variant,
                seed,
                (0, false, 0.0),
                thought,
                answer,
                scorer,
            ));
        }
    }
    Ok(runs)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let prompt_text = dashboard(DIGEST_ASK);
    let re = Patterns::new();
    let known = Known::from_prompt(
        &re,
        &format!("{SYSTEM}\n{prompt_text}"),
        agora_handles(),
    );
    eprintln!("known: {}", known.summary());

    let runs = match &args.rescore {
        Some(dir) => rescore(&args, dir, (&re, &known))?,
        None => generate(&args, prompt_text, (&re, &known))?,
    };

    print_table("thought + answer", &runs, |r| &r.all);
    if args.thinking {
        print_table("thought only", &runs, |r| &r.thought);
        print_table("answer only", &runs, |r| &r.answer);
    }

    println!("\n== miscopies (thought + answer) ==");
    for r in &runs {
        if !r.all.misses.is_empty() {
            println!("{} {}: {}", r.variant, r.seed, r.all.misses.join("  "));
        }
    }
    Ok(())
}
