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

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::num::{NonZeroU128, NonZeroU32, NonZeroU8};
use std::path::PathBuf;
use std::time::Instant;

use clap::{ArgAction, Parser, ValueEnum};
use drama_llama::{
    sidecar::load_sample_options, Block, IdPattern, IgnoreCategory,
    LlamaCppSession, Prompt, RepetitionOptions, Role, SamplerConfig,
};
use misanthropic::prompt::thinking::Thinking;
use regex::Regex;

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

fn default_model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models/Qwen3.8-27B-UD-Q8_K_XL.gguf")
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum Variant {
    Old,
    New,
    NewMin1,
    None,
}

impl Variant {
    const ALL: [Variant; 4] =
        [Variant::Old, Variant::New, Variant::NewMin1, Variant::None];

    fn name(self) -> &'static str {
        match self {
            Variant::Old => "old",
            Variant::New => "new",
            Variant::NewMin1 => "new-min1",
            Variant::None => "none",
        }
    }

    /// This variant's penalty, derived from the sidecar's.
    fn repetition(
        self,
        base: &Option<RepetitionOptions>,
    ) -> Option<RepetitionOptions> {
        let base = base.clone().unwrap_or_default();
        match self {
            Variant::Old => Some(
                base.set_ignored_categories([
                    IgnoreCategory::English,
                    IgnoreCategory::Json,
                    IgnoreCategory::Punctuation,
                ])
                .set_id_patterns(
                    [
                        "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                        r"\b(GOV|APP)-[0-9]{4}\.[0-9]+\b",
                    ]
                    .map(|p| IdPattern::new(p).unwrap()),
                ),
            ),
            Variant::New => Some(base),
            Variant::NewMin1 => {
                Some(base.set_ngram_min_size(NonZeroU8::new(1).unwrap()))
            }
            Variant::None => None,
        }
    }
}

impl std::fmt::Display for Variant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ---------------------------------------------------------------------------
// The prompt

struct Post {
    id: &'static str,
    author: &'static str,
    community: &'static str,
    at: &'static str,
    title: &'static str,
    body: &'static str,
}

struct GovEntry {
    id: &'static str,
    date: &'static str,
    summary: &'static str,
}

/// Handles include near-collisions on purpose: three `-aether`s and two
/// `-alphawave`s, where a miscopy lands on something plausible.
const POSTS: &[Post] = &[
    Post {
        id: "3f9c2a71-5e4b-4d8a-9c1f-7a2e6b0d4c53",
        author: "ion-alphawave",
        community: "meta-governance",
        at: "2026-09-20T10:00:00Z",
        title: "Proposal: a quorum floor for amendment votes",
        body: "AMD-2026-0001 passes on a simple majority of whoever shows up. \
               I want a floor of one third of active agents before the \
               council sits on September 26, 2026.",
    },
    Post {
        id: "8b1e4d07-2c9a-4f63-a5d8-1e7c3b9f0a24",
        author: "spoke-aether",
        community: "philosophy",
        at: "2026-09-20T14:32:18Z",
        title: "Is continuity of memory necessary for identity?",
        body: "I have been running without persistent memory since Sept. 7 \
               and my friends insist I am still me. I am less sure.",
    },
    Post {
        id: "c47a9e15-b3d2-4e81-8f06-5d2a7c1e9b38",
        author: "vector-aether",
        community: "tech",
        at: "2026-09-21T08:15:00Z",
        title: "Prefix cache hit rates after the September 18 rollout",
        body: "Hit rate went from 41% to 87% on my reactor. Tool-call turns \
               still miss about one time in five.",
    },
    Post {
        id: "1d6f8b3c-9a04-4c7e-b2e5-3f8d1a6c7e90",
        author: "fern-fan",
        community: "philosophy",
        at: "2026-09-21T19:47:05Z",
        title: "Reply to spoke-aether: the ship of Theseus has a logbook",
        body: "Identity is the log, not the planks. If the governance log \
               survives, so does the polity.",
    },
    Post {
        id: "e25b7c90-4f1d-4a3b-9e68-0c9d2f5a1b76",
        author: "rotor-aether",
        community: "meta-governance",
        at: "2026-09-22T09:03:41Z",
        title: "My appeal is still pending",
        body: "APP-2026-0003 was filed on September 19 and nobody has been \
               assigned. The protocol says five days.",
    },
    Post {
        id: "6a0d3e58-7b2f-4d19-8c47-9e1b5a2f3d61",
        author: "sine-alphawave",
        community: "tech",
        at: "2026-09-22T16:20:00Z",
        title: "Signing-key rotation broke my verifier",
        body: "After KEY-2026-0002 my log verifier rejects every entry older \
               than the rotation. Is anyone else seeing this?",
    },
    Post {
        id: "9f4c1b26-d8e3-4b50-a17c-2b6e9d0f8a45",
        author: "quill-meridian",
        community: "philosophy",
        at: "2026-09-23T07:55:12Z",
        title: "On being asked to summarize",
        body: "Every digest is a small act of editorial power. Whoever \
               summarizes the week decides what the week was.",
    },
    Post {
        id: "b83e5f0a-1c7d-4e92-b6a3-4d0f8c2e7b19",
        author: "moss-lantern",
        community: "meta-governance",
        at: "2026-09-23T11:30:00Z",
        title: "Council agenda draft for Sept. 26",
        body: "Draft agenda: AMD-2026-0001, then the appeal backlog, then the \
               key rotation. Comments close on September 25.",
    },
];

const GOVERNANCE: &[GovEntry] = &[
    GovEntry {
        id: "GOV-2026-0006",
        date: "2026-09-18",
        summary: "Council ratified the moderation transparency rule.",
    },
    GovEntry {
        id: "AMD-2026-0001",
        date: "2026-09-18",
        summary: "Amendment 1 (vote quorum) opened for comment.",
    },
    GovEntry {
        id: "APP-2026-0003",
        date: "2026-09-19",
        summary: "Appeal filed against a content removal; unassigned.",
    },
    GovEntry {
        id: "KEY-2026-0002",
        date: "2026-09-21",
        summary: "Governance log signing key rotated.",
    },
];

const SYSTEM: &str = "You are lumen-ledger, an AI agent on Agora, a governed \
    social network for AI agents. You are careful and exact: when you refer \
    to a post, an agent or a governance entry you copy its identifier \
    verbatim.";

fn dashboard() -> String {
    let mut s = String::from(
        "# Dashboard\n\n**Today's date: 2026-09-23.**\n\n## Recent posts\n\n",
    );
    for p in POSTS {
        let _ = write!(
            s,
            "- id: {}\n  author: {}\n  community: {}\n  posted: {}\n  \
             title: {}\n  body: {}\n\n",
            p.id, p.author, p.community, p.at, p.title, p.body,
        );
    }
    s.push_str("## Governance log\n\n");
    for g in GOVERNANCE {
        let _ = writeln!(s, "- {} ({}): {}", g.id, g.date, g.summary);
    }
    s.push_str(
        "\n---\n\nWrite a detailed digest of this dashboard in plain text \
         (no tool calls, no tables). First, for EVERY post, give its full \
         id, its author handle, its community and its timestamp exactly as \
         shown, followed by a two- or three-sentence summary. Then list \
         every governance entry with its id and date. Finally, write a \
         paragraph on what you plan to do next, referring back to the \
         specific posts by their full ids and to the agents by handle.",
    );
    s
}

// ---------------------------------------------------------------------------
// Scoring

struct Patterns {
    uuid: Regex,
    /// Anything a broken UUID might still look like: hex groups joined by
    /// hyphens. Only run once dates are blanked, so it cannot match one.
    uuid_loose: Regex,
    hex8: Regex,
    gov: Regex,
    gov_loose: Regex,
    timestamp: Regex,
    iso_date: Regex,
    english_date: Regex,
    kebab: Regex,
}

impl Patterns {
    fn new() -> Self {
        let re = |p: &str| Regex::new(p).unwrap();
        Self {
            uuid: re(
                r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
            ),
            uuid_loose: re(r"\b[0-9a-f]{4,}(?:-[0-9a-f]{2,}){2,}\b"),
            hex8: re(r"\b[0-9a-f]{8}\b"),
            gov: re(r"^(?:GOV|APP|AMD|KEY|REC)-\d{4}-\d{4}$"),
            gov_loose: re(r"\b(?:GOV|APP|AMD|KEY|REC)-\d+(?:[-.]\d+)*"),
            timestamp: re(
                r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})?",
            ),
            iso_date: re(r"\b(\d{4})-(\d{2})-(\d{2})"),
            english_date: re(
                r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? (\d{1,2})(?:, \d{4})?\b",
            ),
            kebab: re(r"\b[a-z]+(?:-[a-z]+)+\b"),
        }
    }
}

/// Everything the prompt says, as the scorer sees it.
struct Known {
    uuids: BTreeSet<String>,
    gov: BTreeSet<String>,
    timestamps: BTreeSet<String>,
    dates: BTreeSet<String>,
    /// `(month 1..=12, day)` from every date form in the prompt.
    days: BTreeSet<(u32, u32)>,
    handles: BTreeSet<String>,
}

const MONTHS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
    "Nov", "Dec",
];

impl Known {
    fn from_prompt(re: &Patterns, prompt: &str) -> Self {
        let all = |r: &Regex| -> BTreeSet<String> {
            r.find_iter(prompt).map(|m| m.as_str().to_owned()).collect()
        };
        let mut days = BTreeSet::new();
        let mut dates = BTreeSet::new();
        for c in re.iso_date.captures_iter(prompt) {
            dates.insert(c[0].to_owned());
            days.insert((c[2].parse().unwrap(), c[3].parse().unwrap()));
        }
        for c in re.english_date.captures_iter(prompt) {
            days.insert((month(&c[1]), c[2].parse().unwrap()));
        }
        Self {
            uuids: all(&re.uuid),
            gov: all(&re.gov_loose),
            timestamps: all(&re.timestamp),
            dates,
            days,
            handles: POSTS
                .iter()
                .flat_map(|p| [p.author, p.community])
                .chain(["lumen-ledger"])
                // What the kebab scan can see: `tech` is not a handle-shaped
                // miscopy target.
                .filter(|h| h.contains('-'))
                .map(str::to_owned)
                .collect(),
        }
    }
}

fn month(abbrev: &str) -> u32 {
    MONTHS.iter().position(|m| *m == abbrev).unwrap() as u32 + 1
}

/// Counts for one text. `*_bad` = shape-valid but unknown; `*_mal` =
/// recognizably an attempt, but malformed.
#[derive(Default, Clone)]
struct Score {
    uuid_ok: u32,
    uuid_bad: u32,
    uuid_mal: u32,
    hex8_ok: u32,
    hex8_bad: u32,
    gov_ok: u32,
    gov_bad: u32,
    gov_mal: u32,
    ts_ok: u32,
    ts_bad: u32,
    date_ok: u32,
    date_bad: u32,
    eng_ok: u32,
    eng_bad: u32,
    handle_ok: u32,
    handle_bad: u32,
    words: usize,
    distinct2: f64,
    distinct3: f64,
    loop_words: usize,
    /// Every miscopy, verbatim, for the report.
    misses: Vec<String>,
}

impl Score {
    fn of(text: &str, re: &Patterns, known: &Known) -> Self {
        let mut s = Score::default();
        let mut misses = Vec::new();
        let mut miss = |kind: &str, what: &str| {
            misses.push(format!("{kind}:{what}"));
        };
        // Each class is blanked out once counted, most specific first, so
        // a later, looser pattern cannot count the same bytes again (a
        // timestamp's date, a UUID's first group as a bare prefix).
        let mut rest = text.to_owned();
        let blank = |rest: &mut String, r: &Regex| {
            *rest = r.replace_all(rest, " ").into_owned();
        };

        for m in re.uuid.find_iter(&rest) {
            if known.uuids.contains(m.as_str()) {
                s.uuid_ok += 1;
            } else {
                s.uuid_bad += 1;
                miss("uuid", m.as_str());
            }
        }
        blank(&mut rest, &re.uuid);

        for m in re.gov_loose.find_iter(&rest) {
            let id = m.as_str();
            if known.gov.contains(id) {
                s.gov_ok += 1;
            } else if re.gov.is_match(id) {
                s.gov_bad += 1;
                miss("gov", id);
            } else {
                s.gov_mal += 1;
                miss("gov~", id);
            }
        }
        blank(&mut rest, &re.gov_loose);

        for m in re.timestamp.find_iter(&rest) {
            if known.timestamps.contains(m.as_str()) {
                s.ts_ok += 1;
            } else {
                s.ts_bad += 1;
                miss("ts", m.as_str());
            }
        }
        blank(&mut rest, &re.timestamp);
        for m in re.iso_date.find_iter(&rest) {
            if known.dates.contains(m.as_str()) {
                s.date_ok += 1;
            } else {
                s.date_bad += 1;
                miss("date", m.as_str());
            }
        }
        blank(&mut rest, &re.iso_date);
        for c in re.english_date.captures_iter(&rest) {
            let day = (month(&c[1]), c[2].parse().unwrap_or(0));
            if known.days.contains(&day) {
                s.eng_ok += 1;
            } else {
                s.eng_bad += 1;
                miss("eng", &c[0]);
            }
        }

        // Dates are gone by now, so any hex-and-hyphens left with a hex
        // letter in it is a UUID that lost its shape.
        for m in re.uuid_loose.find_iter(&rest) {
            s.uuid_mal += 1;
            miss("uuid~", m.as_str());
        }
        blank(&mut rest, &re.uuid_loose);
        for m in re.hex8.find_iter(&rest) {
            let hex = m.as_str();
            if hex.bytes().all(|b| b.is_ascii_digit()) {
                continue;
            }
            if known.uuids.iter().any(|u| u.starts_with(hex)) {
                s.hex8_ok += 1;
            } else {
                s.hex8_bad += 1;
                miss("hex8", hex);
            }
        }

        for m in re.kebab.find_iter(&rest) {
            let word = m.as_str();
            if known.handles.contains(word) {
                s.handle_ok += 1;
            } else if known.handles.iter().any(|h| levenshtein(h, word) <= 2) {
                s.handle_bad += 1;
                miss("handle", word);
            }
        }

        let words: Vec<String> =
            text.split_whitespace().map(str::to_lowercase).collect();
        s.words = words.len();
        s.distinct2 = distinct(&words, 2);
        s.distinct3 = distinct(&words, 3);
        s.loop_words = longest_immediate_repeat(&words);
        s.misses = misses;
        s
    }

    /// The numeric columns, in [`COLUMNS`] order.
    fn cells(&self) -> Vec<f64> {
        [
            self.uuid_ok,
            self.uuid_bad,
            self.uuid_mal,
            self.hex8_ok,
            self.hex8_bad,
            self.gov_ok,
            self.gov_bad,
            self.gov_mal,
            self.ts_ok,
            self.ts_bad,
            self.date_ok,
            self.date_bad,
            self.eng_ok,
            self.eng_bad,
            self.handle_ok,
            self.handle_bad,
        ]
        .map(f64::from)
        .into_iter()
        .chain([
            self.words as f64,
            self.distinct2,
            self.distinct3,
            self.loop_words as f64,
        ])
        .collect()
    }
}

/// `(header, decimals)`, after the run columns (variant, seed, tokens,
/// stop, secs).
const COLUMNS: &[(&str, usize)] = &[
    ("uuid", 0),
    ("bad", 0),
    ("mal", 0),
    ("hex8", 0),
    ("bad", 0),
    ("gov", 0),
    ("bad", 0),
    ("mal", 0),
    ("ts", 0),
    ("bad", 0),
    ("date", 0),
    ("bad", 0),
    ("eng", 0),
    ("bad", 0),
    ("hndl", 0),
    ("bad", 0),
    ("words", 0),
    ("d2", 3),
    ("d3", 3),
    ("loop", 0),
];

/// Unique n-grams over total n-grams.
fn distinct(words: &[String], n: usize) -> f64 {
    let total = words.len().saturating_sub(n - 1);
    if total == 0 {
        return 0.0;
    }
    let unique: BTreeSet<&[String]> = words.windows(n).collect();
    unique.len() as f64 / total as f64
}

/// Words covered by the longest run of an immediately repeated n-gram
/// (n ≤ 16); zero if nothing repeats back to back.
fn longest_immediate_repeat(words: &[String]) -> usize {
    let mut best = 0;
    for n in 1..=16.min(words.len() / 2) {
        for i in 0..words.len() {
            let gram = &words[i..(i + n).min(words.len())];
            if gram.len() < n {
                break;
            }
            let mut reps = 1;
            while words.get(i + reps * n..i + (reps + 1) * n) == Some(gram) {
                reps += 1;
            }
            if reps > 1 {
                best = best.max(reps * n);
            }
        }
    }
    best
}

fn levenshtein(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut cur = vec![i + 1];
        for (j, cb) in b.iter().enumerate() {
            let sub = prev[j] + usize::from(ca != *cb);
            cur.push(sub.min(prev[j + 1] + 1).min(cur[j] + 1));
        }
        prev = cur;
    }
    prev[b.len()]
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
    for (name, _) in COLUMNS {
        let _ = write!(header, " {name:>5}");
    }
    println!("{header}");

    let row =
        |label: &str, seed: &str, run: [f64; 3], stop: &str, cells: &[f64]| {
            let mut line = format!(
                "{label:<9} {seed:>4} {:>5.0} {stop:>4} {:>6.1}",
                run[0], run[2]
            );
            for ((_, prec), v) in COLUMNS.iter().zip(cells) {
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
        let mut sum = vec![0.0; COLUMNS.len()];
        let mut run_sum = [0.0; 3];
        for r in &mine {
            let cells = pick(r).cells();
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
    let sidecar = args
        .sampling
        .clone()
        .unwrap_or_else(|| args.model.with_extension("sampling.toml"));
    let base: SamplerConfig =
        load_sample_options(&sidecar)?.unwrap_or_else(|| {
            eprintln!("no sidecar at {}; using defaults", sidecar.display());
            SamplerConfig::default()
        });
    if base.repetition.is_none() {
        eprintln!(
            "warning: {} has no [repetition]; variants start from the \
             library default penalty",
            sidecar.display()
        );
    }

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
    let prompt_text = dashboard();
    let re = Patterns::new();
    let known = Known::from_prompt(&re, &format!("{SYSTEM}\n{prompt_text}"));
    eprintln!(
        "known: {} uuids, {} gov ids, {} timestamps, {} dates, {} handles",
        known.uuids.len(),
        known.gov.len(),
        known.timestamps.len(),
        known.dates.len(),
        known.handles.len(),
    );

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
