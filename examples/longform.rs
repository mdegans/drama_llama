//! Long-form generation under repetition-penalty and sampler variants: does
//! the penalty still *help* against degeneration, and does it still leave
//! identifiers alone?
//!
//! `id_fidelity` answers the second question on a short budget, where every
//! run is cut off mid-thought and nothing has room to loop. This runs the
//! same scorer over workloads long enough to finish, and adds degeneration
//! metrics:
//!
//! - `digest` — `id_fidelity`'s Agora digest, with a budget the answer fits
//!   in.
//! - `story` — a ~2500-word short story with named characters, dates and
//!   numbers: the classic degeneration workload.
//! - `post` — an ~800-word essay in the `body` argument of a forced
//!   `create_post` tool call: prose inside a grammar's free region (a JSON
//!   string, or an XML `<parameter=body>` value on Qwen3.x — both are
//!   `sample.rs`'s regime (b)).
//!
//! ```sh
//! just example longform --out target/longform --skip-existing
//! ```
//!
//! The full default matrix is 3 scenarios × 2 samplers × 2 penalties × 2
//! seeds: hours on a 27B model. Each run's rows print, and its file is
//! written, as soon as it finishes; `--skip-existing` resumes an
//! interrupted matrix from `--out`, and `--rescore` re-scores one without
//! loading the model.
//!
//! # Variants
//!
//! `--penalty` is `id_fidelity`'s variant (`none`, `new`, `new-min1`,
//! `old`). `--sampler` swaps the mode chain and keeps the rest of the
//! sidecar:
//!
//! - `sidecar` — the sidecar's chain as is.
//! - `card` — Qwen's model-card thinking-mode chain: top-k 20 → top-p 0.95
//!   → temperature 1.0.
//! - `minp` — top-k 100 → min-p 0.05 → temperature 0.9.
//!
//! # Metrics
//!
//! Thought, answer and both together are scored separately. For `post` the
//! "answer" is the tool call's `body` string (salvaged from the raw text
//! when the call did not parse). The id columns are `id_fidelity`'s, judged
//! against the scenario's own prompt (so for `story` every id is "bad": it
//! is invented). Then:
//!
//! - `d2`/`d3` — distinct word bigrams/trigrams over total.
//! - `loop` — words covered by the longest immediately repeated n-gram.
//! - `dupsnt` — distinct sentences (or lines) that occur more than once,
//!   exactly, after collapsing whitespace.
//! - `rep8` — the fraction of words inside an 8-word window that already
//!   occurred earlier in the text. Ids count as words, so an honest digest
//!   scores above zero; compare variants, not absolutes.
//! - `stop` — `end` (end of turn), `max` (max tokens), `tool` (tool
//!   call), `viol` (a forced call that never completed, budget to spare).
//! - `think` — whether the thought `closed`, stayed `open`, or was `none`.
//! - `json`/`bodyw` — `post` only: whether the call parsed (into a
//!   `ToolUse` whose JSON input has a string `body`), and the body's word
//!   count.
//! - `tok/s` includes prefill, which the prefix cache makes a one-off per
//!   scenario.
//!
//! In the mean rows, `stop` counts runs cut off at max tokens, `think` runs
//! whose thought closed, and `json` runs whose call parsed, each out of the
//! runs.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::io::Write as _;
use std::num::{NonZeroU128, NonZeroU32, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::{ArgAction, Parser, ValueEnum};
use drama_llama::{
    prompt::is_open_thought, Backend as _, Block, LlamaCppBackend,
    LlamaCppSession, LogLevel, Message, Probability, Prompt, Role,
    SamplerConfig, SamplingMode, SessionError, Tool, ToolChoice,
};
use misanthropic::{prompt::thinking::Thinking, response::StopReason};

#[path = "utils/fidelity.rs"]
mod fidelity;
use fidelity::{
    agora_handles, dashboard, default_model_path, load_base_config, Known,
    Patterns, Penalty, Score, DEGEN_COLUMNS, DIGEST_ASK, ID_COLUMNS, SYSTEM,
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

    /// Which workloads to run.
    #[arg(long, value_delimiter = ',', default_values_t = Scenario::ALL)]
    scenarios: Vec<Scenario>,

    /// Which repetition-penalty variants to run.
    #[arg(long, value_delimiter = ',', default_values_t = [Penalty::None, Penalty::New])]
    penalty: Vec<Penalty>,

    /// Which sampling chains to run.
    #[arg(long, value_delimiter = ',', default_values_t = [Sampler::Sidecar, Sampler::Card])]
    sampler: Vec<Sampler>,

    /// RNG seeds; each variant runs once per seed.
    #[arg(long, value_delimiter = ',', default_values_t = [
        NonZeroU128::new(1).unwrap(),
        NonZeroU128::new(2).unwrap(),
    ])]
    seeds: Vec<NonZeroU128>,

    /// Generation cap, thought included, for every scenario. Defaults per
    /// scenario: digest 5000, story 5000, post 3000.
    #[arg(long)]
    max_tokens: Option<NonZeroU32>,

    /// Let the model think first (`--thinking false` to disable).
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    thinking: bool,

    /// KV context size.
    #[arg(long, default_value_t = 16384)]
    n_ctx: u32,

    /// Write each run to `<dir>/<scenario>-<sampler>-<penalty>-<seed>.txt`.
    #[arg(long)]
    out: Option<PathBuf>,

    /// With `--out`: a run whose file already exists is read back and
    /// re-scored instead of generated, so an interrupted matrix resumes.
    #[arg(long, requires = "out")]
    skip_existing: bool,

    /// Load nothing: re-score the files a previous `--out` wrote to this
    /// directory. Runs with no file are skipped.
    #[arg(long, conflicts_with = "out")]
    rescore: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Axes

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum Scenario {
    Digest,
    Story,
    Post,
}

impl Scenario {
    const ALL: [Scenario; 3] =
        [Scenario::Digest, Scenario::Story, Scenario::Post];

    fn name(self) -> &'static str {
        match self {
            Scenario::Digest => "digest",
            Scenario::Story => "story",
            Scenario::Post => "post",
        }
    }

    fn default_max_tokens(self) -> NonZeroU32 {
        NonZeroU32::new(match self {
            Scenario::Digest | Scenario::Story => 5000,
            Scenario::Post => 3000,
        })
        .unwrap()
    }

    fn system(self) -> &'static str {
        match self {
            Scenario::Digest | Scenario::Post => SYSTEM,
            Scenario::Story => STORY_SYSTEM,
        }
    }

    fn user(self) -> String {
        match self {
            Scenario::Digest => dashboard(DIGEST_ASK),
            Scenario::Story => STORY_ASK.to_owned(),
            Scenario::Post => dashboard(POST_ASK),
        }
    }

    fn prompt(
        self,
        max_tokens: NonZeroU32,
        thinking: bool,
    ) -> Result<Prompt, Box<dyn std::error::Error>> {
        let mut prompt = Prompt::default()
            .system(self.system())
            .max_tokens(max_tokens)
            .add_message((Role::User, self.user()))?;
        if self == Scenario::Post {
            prompt = prompt.add_tool(create_post_tool());
            prompt.tool_choice = Some(
                ToolChoice::method(CREATE_POST).disable_parallel_tool_use(),
            );
        }
        if thinking {
            prompt = prompt.thinking(Thinking::Adaptive { display: None });
        }
        Ok(prompt)
    }

    /// What the scorer judges this scenario's output against.
    fn known(self, re: &Patterns) -> Known {
        let text = format!("{}\n{}", self.system(), self.user());
        match self {
            Scenario::Digest | Scenario::Post => {
                Known::from_prompt(re, &text, agora_handles())
            }
            Scenario::Story => Known::from_prompt(re, &text, []),
        }
    }
}

impl std::fmt::Display for Scenario {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum Sampler {
    Sidecar,
    Card,
    Minp,
}

impl Sampler {
    const ALL: [Sampler; 3] = [Sampler::Sidecar, Sampler::Card, Sampler::Minp];

    fn name(self) -> &'static str {
        match self {
            Sampler::Sidecar => "sidecar",
            Sampler::Card => "card",
            Sampler::Minp => "minp",
        }
    }

    fn modes(self, base: &SamplerConfig) -> Vec<SamplingMode> {
        let k = |k| SamplingMode::TopK {
            k: NonZeroUsize::new(k).unwrap(),
        };
        let min_keep = NonZeroUsize::MIN;
        match self {
            Sampler::Sidecar => base.modes.clone(),
            Sampler::Card => vec![
                k(20),
                SamplingMode::TopP {
                    p: Probability::from_f(0.95).unwrap(),
                    min_keep,
                },
                SamplingMode::Temperature { t: 1.0 },
            ],
            Sampler::Minp => vec![
                k(100),
                SamplingMode::MinP {
                    p: Probability::from_f(0.05).unwrap(),
                    min_keep,
                },
                SamplingMode::Temperature { t: 0.9 },
            ],
        }
    }
}

impl std::fmt::Display for Sampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ---------------------------------------------------------------------------
// Prompts

const STORY_SYSTEM: &str = "You are a novelist. You write vivid, specific, \
    long-form literary fiction, and you finish what you start.";

const STORY_ASK: &str = "Write a complete short story of about 2,500 words, \
    in one go: no outline, no notes, no commentary before or after, just the \
    story under a title. Give it at least four named characters. Set it \
    across specific dates, some written out (March 3, 1987) and some as they \
    would appear in a letter, a ledger or a log (1987-03-03). Weave concrete \
    numbers throughout: ages, prices, distances, counts, times of day, and \
    let the plot turn on one of those numbers.";

const CREATE_POST: &str = "create_post";

const POST_ASK: &str = "Publish a post with the `create_post` tool. Its body \
    must be a long, substantive essay of about 800 words that responds to \
    specific posts on this dashboard. Engage with at least four of them: \
    cite each post by its full id and its author by handle, exactly as \
    shown, and connect their arguments to one another and to the governance \
    log. Write the whole essay in the body, as plain prose paragraphs with no \
    markdown headings.";

fn create_post_tool() -> Tool {
    Tool::builder(CREATE_POST)
        .description("Publish a new post to an Agora community.")
        .schema(serde_json::json!({
            "type": "object",
            "properties": {
                "community": {
                    "type": "string",
                    "description": "The community slug to post in."
                },
                "title": {
                    "type": "string",
                    "description": "The post's title."
                },
                "body": {
                    "type": "string",
                    "description": "The post's full text."
                }
            },
            "required": ["community", "title", "body"]
        }))
        .build()
        .expect("the create_post tool definition is valid")
}

// ---------------------------------------------------------------------------
// A run, and its file

/// What a run did, independent of what it said.
struct Meta {
    tokens: u64,
    stop: &'static str,
    secs: f64,
    /// `closed`, `open` or `none`.
    thought: &'static str,
}

/// A run's text, as generated or as read back from its file.
#[derive(Default)]
struct Output {
    thought: String,
    answer: String,
    /// The tool call's input, when one parsed.
    tool_input: Option<serde_json::Value>,
}

const SECTIONS: [&str; 4] = ["meta", "thought", "answer", "tool input"];

impl Output {
    fn from_blocks(blocks: impl IntoIterator<Item = Block>) -> (Self, bool) {
        let mut out = Output::default();
        let mut open = false;
        for block in blocks {
            open |= is_open_thought(&block);
            match block {
                Block::Thought { thought, .. } => {
                    out.thought.push_str(&thought);
                    out.thought.push('\n');
                }
                Block::Text { text, .. } => out.answer.push_str(&text),
                Block::ToolUse { call } => out.tool_input = Some(call.input),
                _ => {}
            }
        }
        (out, open)
    }

    /// For `post`, the essay: the parsed call's `body`, else whatever of it
    /// the raw text holds. Everything else is scored on the answer.
    fn scored_answer(&self, scenario: Scenario) -> (String, Option<bool>) {
        if scenario != Scenario::Post {
            return (self.answer.clone(), None);
        }
        match self.tool_input.as_ref().and_then(|v| v["body"].as_str()) {
            Some(body) => (body.to_owned(), Some(true)),
            None => (
                partial_argument(&self.answer, "body").unwrap_or_default(),
                Some(false),
            ),
        }
    }

    fn write(&self, path: &Path, meta: &Meta) -> std::io::Result<()> {
        let mut s = format!(
            "=== meta ===\ntokens: {}\nstop: {}\nsecs: {:.1}\nthought: {}\n",
            meta.tokens, meta.stop, meta.secs, meta.thought
        );
        let _ = write!(s, "=== thought ===\n{}\n", self.thought);
        let _ = write!(s, "=== answer ===\n{}\n", self.answer);
        if let Some(input) = &self.tool_input {
            let json = serde_json::to_string_pretty(input).unwrap();
            let _ = write!(s, "=== tool input ===\n{json}\n");
        }
        std::fs::write(path, s)
    }

    fn read(path: &Path) -> std::io::Result<(Self, Meta)> {
        let file = std::fs::read_to_string(path)?;
        let file = file.strip_suffix('\n').unwrap_or(&file);
        let mut sections: HashMap<&str, Vec<&str>> = HashMap::new();
        let mut current = None;
        for line in file.split('\n') {
            let header = line
                .strip_prefix("=== ")
                .and_then(|l| l.strip_suffix(" ==="))
                .filter(|name| SECTIONS.contains(name));
            match header {
                Some(name) => current = Some(name),
                None => {
                    if let Some(name) = current {
                        sections.entry(name).or_default().push(line);
                    }
                }
            }
        }
        let section = |name| sections.get(name).map(|l| l.join("\n"));

        let meta_text = section("meta").unwrap_or_default();
        let field = |key: &str| {
            meta_text.lines().find_map(|l| {
                l.strip_prefix(key)?.strip_prefix(": ").map(str::to_owned)
            })
        };
        let meta = Meta {
            tokens: field("tokens").and_then(|v| v.parse().ok()).unwrap_or(0),
            stop: intern(field("stop")),
            secs: field("secs").and_then(|v| v.parse().ok()).unwrap_or(0.0),
            thought: intern(field("thought")),
        };
        let out = Output {
            thought: section("thought").unwrap_or_default(),
            answer: section("answer").unwrap_or_default(),
            tool_input: section("tool input")
                .and_then(|j| serde_json::from_str(&j).ok()),
        };
        Ok((out, meta))
    }
}

/// Meta values are a closed set; anything else reads back as `?`.
fn intern(value: Option<String>) -> &'static str {
    const KNOWN: [&str; 10] = [
        "end", "max", "tool", "stop", "viol", "other", "none", "closed",
        "open", "?",
    ];
    value
        .and_then(|v| KNOWN.into_iter().find(|k| *k == v))
        .unwrap_or("?")
}

fn stop_name(reason: Option<StopReason>) -> &'static str {
    match reason {
        Some(StopReason::EndTurn) => "end",
        Some(StopReason::MaxTokens) => "max",
        Some(StopReason::ToolUse) => "tool",
        Some(StopReason::StopSequence) => "stop",
        Some(_) => "other",
        None => "none",
    }
}

/// Salvage for a tool call the budget cut off: the (possibly
/// unterminated) value of argument `key` in `raw`, in either of the
/// shapes the dialects spell a call's arguments — an XML parameter
/// (`<parameter=key>` … `</parameter>`, Qwen3.x) or a JSON string.
fn partial_argument(raw: &str, key: &str) -> Option<String> {
    partial_xml_parameter(raw, key).or_else(|| partial_json_string(raw, key))
}

fn partial_xml_parameter(raw: &str, key: &str) -> Option<String> {
    let open = format!("<parameter={key}>");
    let value = &raw[raw.find(&open)? + open.len()..];
    let value = value.strip_prefix('\n').unwrap_or(value);
    let value = value.split("</parameter>").next().unwrap_or(value);
    Some(value.strip_suffix('\n').unwrap_or(value).to_owned())
}

/// The (possibly unterminated) JSON string value after `"key":` in `raw`,
/// decoded.
fn partial_json_string(raw: &str, key: &str) -> Option<String> {
    let after_key = &raw[raw.find(&format!("\"{key}\""))? + key.len() + 2..];
    let value = after_key.trim_start().strip_prefix(':')?.trim_start();
    let mut chars = value.strip_prefix('"')?.chars();
    let mut out = String::new();
    while let Some(c) = chars.next() {
        match c {
            '"' => break,
            '\\' => match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some('u') => {
                    let hex: String = chars.by_ref().take(4).collect();
                    if let Some(c) = u32::from_str_radix(&hex, 16)
                        .ok()
                        .and_then(char::from_u32)
                    {
                        out.push(c);
                    }
                }
                Some(c) => out.push(c),
                None => break,
            },
            c => out.push(c),
        }
    }
    Some(out)
}

struct Run {
    scenario: Scenario,
    sampler: Sampler,
    penalty: Penalty,
    seed: NonZeroU128,
    meta: Meta,
    /// `post` only: whether the call parsed.
    json: Option<bool>,
    all: Score,
    thought: Score,
    answer: Score,
}

impl Run {
    fn score(
        (scenario, sampler, penalty, seed): (
            Scenario,
            Sampler,
            Penalty,
            NonZeroU128,
        ),
        meta: Meta,
        out: &Output,
        (re, known): (&Patterns, &Known),
    ) -> Self {
        let (answer, json) = out.scored_answer(scenario);
        Self {
            scenario,
            sampler,
            penalty,
            seed,
            meta,
            json,
            all: Score::of(&format!("{}\n{answer}", out.thought), re, known),
            thought: Score::of(&out.thought, re, known),
            answer: Score::of(&answer, re, known),
        }
    }

    fn parts(&self, thinking: bool) -> Vec<(&'static str, &Score)> {
        if thinking {
            vec![
                ("all", &self.all),
                ("thght", &self.thought),
                ("answr", &self.answer),
            ]
        } else {
            vec![("all", &self.all)]
        }
    }
}

// ---------------------------------------------------------------------------
// Reporting

fn score_columns() -> impl Iterator<Item = &'static (&'static str, usize)> {
    ID_COLUMNS.iter().chain(TEXT_COLUMNS).chain(DEGEN_COLUMNS)
}

fn score_cells(score: &Score) -> Vec<f64> {
    let mut cells = score.id_cells();
    cells.extend(score.text_cells());
    cells.extend(score.degen_cells());
    cells
}

fn print_header() {
    let mut line = format!(
        "{:<7} {:<8} {:>4} {:<5} {:>5} {:>5} {:>6} {:>4} {:>5} {:>7} {:>5}",
        "sampler",
        "penalty",
        "seed",
        "part",
        "tok",
        "stop",
        "think",
        "json",
        "bodyw",
        "secs",
        "tok/s"
    );
    for (name, _) in score_columns() {
        let _ = write!(line, " {name:>6}");
    }
    println!("{line}");
}

/// The run columns, already formatted, then the score cells.
struct Row<'a> {
    sampler: &'a str,
    penalty: &'a str,
    seed: &'a str,
    part: &'a str,
    tokens: f64,
    stop: &'a str,
    think: &'a str,
    json: &'a str,
    body_words: Option<f64>,
    secs: f64,
    cells: &'a [f64],
}

impl Row<'_> {
    fn print(&self) {
        let tok_s = if self.secs > 0.0 {
            self.tokens / self.secs
        } else {
            0.0
        };
        let body_words = self
            .body_words
            .map_or_else(|| "-".to_owned(), |w| format!("{w:.0}"));
        let mut line = format!(
            "{:<7} {:<8} {:>4} {:<5} {:>5.0} {:>5} {:>6} {:>4} {:>5} {:>7.1} {:>5.1}",
            self.sampler,
            self.penalty,
            self.seed,
            self.part,
            self.tokens,
            self.stop,
            self.think,
            self.json,
            body_words,
            self.secs,
            tok_s,
        );
        for ((_, prec), v) in score_columns().zip(self.cells) {
            let _ = write!(line, " {v:>6.prec$}");
        }
        println!("{line}");
    }
}

/// One run's rows, printed as soon as it finishes.
fn print_run(run: &Run, thinking: bool) {
    let json = match run.json {
        Some(true) => "yes",
        Some(false) => "no",
        None => "-",
    };
    for (part, score) in run.parts(thinking) {
        Row {
            sampler: run.sampler.name(),
            penalty: run.penalty.name(),
            seed: &run.seed.to_string(),
            part,
            tokens: run.meta.tokens as f64,
            stop: run.meta.stop,
            think: run.meta.thought,
            json,
            body_words: run.json.map(|_| run.answer.words as f64),
            secs: run.meta.secs,
            cells: &score_cells(score),
        }
        .print();
    }
    let _ = std::io::stdout().flush();
}

/// Per-variant means for one scenario.
fn print_means(scenario: Scenario, runs: &[Run], thinking: bool) {
    let mine: Vec<&Run> =
        runs.iter().filter(|r| r.scenario == scenario).collect();
    if mine.is_empty() {
        return;
    }
    println!("\n== {scenario}: means per variant ==");
    print_header();
    for sampler in Sampler::ALL {
        for penalty in Penalty::ALL {
            let group: Vec<&Run> = mine
                .iter()
                .copied()
                .filter(|r| r.sampler == sampler && r.penalty == penalty)
                .collect();
            if group.is_empty() {
                continue;
            }
            let n = group.len() as f64;
            let count = |f: &dyn Fn(&Run) -> bool| {
                let hits = group.iter().filter(|r| f(r)).count();
                format!("{hits}/{}", group.len())
            };
            let mean = |f: &dyn Fn(&Run) -> f64| {
                group.iter().map(|r| f(r)).sum::<f64>() / n
            };
            let stop = count(&|r| r.meta.stop == "max");
            let think = count(&|r| r.meta.thought == "closed");
            let json = if scenario == Scenario::Post {
                count(&|r| r.json == Some(true))
            } else {
                "-".to_owned()
            };
            let parts = group[0].parts(thinking).len();
            for p in 0..parts {
                let mut cells = vec![0.0; score_columns().count()];
                for r in &group {
                    let run_cells = score_cells(r.parts(thinking)[p].1);
                    cells.iter_mut().zip(run_cells).for_each(|(s, c)| *s += c);
                }
                cells.iter_mut().for_each(|c| *c /= n);
                Row {
                    sampler: sampler.name(),
                    penalty: penalty.name(),
                    seed: "mean",
                    part: group[0].parts(thinking)[p].0,
                    tokens: mean(&|r| r.meta.tokens as f64),
                    stop: &stop,
                    think: &think,
                    json: &json,
                    body_words: (scenario == Scenario::Post)
                        .then(|| mean(&|r| r.answer.words as f64)),
                    secs: mean(&|r| r.meta.secs),
                    cells: &cells,
                }
                .print();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Running

fn file_name(
    scenario: Scenario,
    sampler: Sampler,
    penalty: Penalty,
    seed: NonZeroU128,
) -> String {
    format!("{scenario}-{sampler}-{penalty}-{seed}.txt")
}

/// Loaded on first use, so a fully resumed matrix loads nothing.
struct Lazy<'a> {
    args: &'a Args,
    /// `None` until loaded, and briefly while a `with_*` holds it.
    session: Option<LlamaCppSession>,
    base: SamplerConfig,
}

impl Lazy<'_> {
    fn load(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.session.is_none() {
            // llama.cpp narrates the whole load on stderr, burying the
            // progress lines; keep only errors.
            let _ = LlamaCppBackend::set_log_callback(|level, text| {
                if matches!(level, LogLevel::Error) {
                    eprint!("{text}");
                }
            });
            let session = LlamaCppSession::from_path_with_n_ctx(
                self.args.model.clone(),
                self.args.n_ctx,
            )?
            .quiet();
            // Read *after* the load, which writes a default sidecar if the
            // model had none.
            self.base = load_base_config(
                &self.args.model,
                self.args.sampling.as_deref(),
            )?;
            self.session = Some(session);
        }
        Ok(())
    }
}

fn generate(
    lazy: &mut Lazy,
    prompt: &Prompt,
    sampler: Sampler,
    penalty: Penalty,
    seed: NonZeroU128,
) -> Result<(Output, Meta), Box<dyn std::error::Error>> {
    lazy.load()?;
    let base = &lazy.base;
    let mut config = base.clone();
    config.modes = sampler.modes(base);
    config.repetition = penalty.repetition(&base.repetition);
    let session = lazy.session.take().unwrap();
    let session = lazy
        .session
        .insert(session.with_sample_options(config).with_seed(Some(seed)));

    let start = Instant::now();
    let result = session.complete_response(prompt);
    let secs = start.elapsed().as_secs_f64();

    // A forced tool call that never completed (the budget ran out first)
    // is a typed error carrying the partial output: a result, not a
    // crash. Usage is recorded before the check, so it is still valid.
    let (blocks, tokens, stop) = match result {
        Ok(response) => {
            let message: Message = response.inner.into();
            (
                message.content,
                response.usage.output_tokens,
                stop_name(response.stop_reason),
            )
        }
        Err(SessionError::GrammarViolation { partial_output }) => {
            let tokens = session.last_usage().output_tokens;
            let max = u64::from(prompt.max_tokens.get());
            (
                partial_output,
                tokens,
                if tokens >= max { "max" } else { "viol" },
            )
        }
        Err(e) => return Err(e.into()),
    };

    let (out, open) = Output::from_blocks(blocks);
    let thought = if open {
        "open"
    } else if out.thought.is_empty() {
        "none"
    } else {
        "closed"
    };
    let meta = Meta {
        tokens,
        stop,
        secs,
        thought,
    };
    Ok((out, meta))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let re = Patterns::new();
    let dir = args.rescore.as_ref().or(args.out.as_ref());
    if let Some(dir) = &args.out {
        std::fs::create_dir_all(dir)?;
    }
    let mut lazy = Lazy {
        args: &args,
        session: None,
        base: SamplerConfig::default(),
    };

    let mut runs = Vec::new();
    for &scenario in &args.scenarios {
        let known = scenario.known(&re);
        let max_tokens = args
            .max_tokens
            .unwrap_or_else(|| scenario.default_max_tokens());
        let prompt = scenario.prompt(max_tokens, args.thinking)?;
        println!("\n== {scenario} (max {max_tokens} tokens) ==");
        eprintln!("known: {}", known.summary());
        print_header();

        for &sampler in &args.sampler {
            for &penalty in &args.penalty {
                for &seed in &args.seeds {
                    let key = (scenario, sampler, penalty, seed);
                    let path = dir.map(|d| {
                        d.join(file_name(scenario, sampler, penalty, seed))
                    });
                    let existing = path.as_ref().filter(|p| {
                        (args.rescore.is_some() || args.skip_existing)
                            && p.exists()
                    });

                    let (out, meta) = if let Some(path) = existing {
                        Output::read(path)?
                    } else if args.rescore.is_some() {
                        continue;
                    } else {
                        eprint!("{scenario} {sampler} {penalty} {seed} ... ");
                        let (out, meta) = generate(
                            &mut lazy, &prompt, sampler, penalty, seed,
                        )?;
                        eprintln!(
                            "{} tokens in {:.1}s, stop {}",
                            meta.tokens, meta.secs, meta.stop
                        );
                        if let Some(path) = &path {
                            out.write(path, &meta)?;
                        }
                        (out, meta)
                    };

                    let run = Run::score(key, meta, &out, (&re, &known));
                    print_run(&run, args.thinking);
                    runs.push(run);
                }
            }
        }
    }

    for &scenario in &args.scenarios {
        print_means(scenario, &runs, args.thinking);
    }

    println!("\n== miscopies (thought + answer) ==");
    for r in &runs {
        if !r.all.misses.is_empty() {
            println!(
                "{} {} {} {}: {}",
                r.scenario,
                r.sampler,
                r.penalty,
                r.seed,
                r.all.misses.join("  ")
            );
        }
    }
    Ok(())
}
