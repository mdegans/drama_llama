//! Replay recorded `/v1/messages` requests: regenerate one assistant turn
//! from its recorded prefix and score how the tool calls copy ids.
//!
//! The input is a request body exactly as a client sent it to blallama —
//! system, tools, `thinking`, `output_config`, and the whole transcript,
//! the model's own earlier turns included. For assistant message `N`,
//! `messages[..N]` is the prompt; every turn is regenerated from its own
//! *recorded* prefix, never from our generated output. Tool results cannot
//! be executed here, so exactly one assistant turn is generated per replay:
//! there is no tool loop.
//!
//! The request goes through the same path blallama's `/v1/messages` takes:
//! `serde_json` into [`Prompt`], [`Catalog::resolve`] for the model it
//! names (under `--models`), `Session::from_path_with` (so the model's
//! sampling, dialect and template sidecars load exactly as they do there),
//! the prefix cache on, then `complete_response`. Differences, on purpose:
//! a fixed seed per run, the sampler and penalty swapped per variant, no
//! `EmittedSpecialToken` resample (with a fixed seed it would repeat
//! itself; the run is reported as `special`), and `output_config` handling
//! (below).
//!
//! ```sh
//! just example replay --request path/to/prompts/*.json --list
//! just example replay --request path/to/prompts/*.json \
//!     --penalty none,new --sampler sidecar --seeds 1 --out target/replay
//! ```
//!
//! # `output_config`
//!
//! A request file is the *last* request of a session, so its
//! `output_config` belongs to that last turn (the memory update) and not to
//! the tool-using turns before it. It also cannot have been active on any
//! turn that called a tool: `Session` ranks `output_config` above the lazy
//! `tool_choice: auto` grammar, so under it a tool call is unreachable. So
//! by default (`--output-config auto`) it is kept only for turns whose
//! recorded text is a JSON object; `keep` applies it everywhere, and
//! `drop` (or `--no-output-config`) nowhere.
//!
//! # Scoring
//!
//! One row per generated turn, printed as soon as it finishes, after a
//! `recorded` row scoring the transcript's own turn the same way:
//!
//! - `think` — thought length in tokens (re-tokenized; characters for the
//!   recorded row), and `thk` whether it `closed`, stayed `open`, or was
//!   `none`.
//! - `ids` — for each tool call, in order, the class of each of its
//!   id-bearing string arguments (`id`, `target`, `reply_to`, and any
//!   argument whose recorded value in this transcript starts with a UUID
//!   or a governance id): `E` exact (a known id), `M` miscopy (shape-valid
//!   but unknown), `R` run-on (a whole id, then more), `T` truncated (a
//!   strict prefix of a known id), `O` other; `.` for a call with no id
//!   argument. "Known" means present in the system prompt, the tools, or a
//!   user or tool-result message of the prompt — not in the model's own
//!   earlier turns, which carry its earlier miscopies.
//! - `secs` and `tok/s` include prefill; `in`/`cached` are the prompt's
//!   tokens and those restored from the prefix cache (`last_usage`).
//!
//! The summary totals each class by call index within the turn (1st, 2nd,
//! …), per variant, since the failure clusters on the 3rd+ call. Every
//! non-exact value is listed at the end.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write as _;
use std::io::Write as _;
use std::num::{NonZeroU128, NonZeroU32, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::{Parser, ValueEnum};
use drama_llama::{
    prompt::is_open_thought, Backend as _, Block, Catalog, FromPath as _,
    LlamaCppBackend, LlamaCppOptions, LlamaCppSession, LogLevel, Message,
    Probability, Prompt, RenderOptions, Role, SamplerConfig, SamplingMode,
    SessionError,
};
use misanthropic::response::StopReason;
use regex::Regex;
use serde_json::Value;

#[path = "utils/fidelity.rs"]
mod fidelity;
use fidelity::{load_base_config, Penalty};

#[derive(Parser, Debug)]
struct Args {
    /// Request bodies to replay (`/v1/messages` JSON).
    #[arg(long, num_args = 1.., required = true)]
    request: Vec<PathBuf>,

    /// Regenerate only the assistant message at this index of `messages`.
    /// Default: every assistant turn.
    #[arg(long)]
    turn: Option<usize>,

    /// Replay only these turns, as `stem:index` (e.g. `aegis:5`), across
    /// all `--request` files in one model load. Comma-separated or
    /// repeated; combines with `--turn` by intersection.
    #[arg(long, value_delimiter = ',')]
    only: Vec<String>,

    /// Print each file's assistant turns, their tool calls and id classes,
    /// and exit. Loads no model.
    #[arg(long)]
    list: bool,

    /// Directory the request's `model` is resolved in, as blallama's
    /// model directory.
    #[arg(long, default_value_os_t = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models"))]
    models: PathBuf,

    /// Replay on this model instead of the one each request names (e.g.
    /// a Qwen3.8 transcript on Qwen3.6). Use a separate `--out`: output
    /// file names do not include the model.
    #[arg(long)]
    model: Option<String>,

    /// Sampling sidecar every variant starts from. Defaults to the model's
    /// own `<model>.sampling.toml`.
    #[arg(long)]
    sampling: Option<PathBuf>,

    /// The chat-template variable `reasoning_effort`. Omitted: not set, so
    /// the template's default applies.
    #[arg(long)]
    effort: Option<Effort>,

    /// Which repetition-penalty variants to run.
    #[arg(long, value_delimiter = ',', default_values_t = [Penalty::None, Penalty::New])]
    penalty: Vec<Penalty>,

    /// Which sampling chains to run.
    #[arg(long, value_delimiter = ',', default_values_t = [Sampler::Sidecar, Sampler::Card])]
    sampler: Vec<Sampler>,

    /// RNG seeds; each variant runs once per seed.
    #[arg(long, value_delimiter = ',', default_values_t = [NonZeroU128::new(1).unwrap()])]
    seeds: Vec<NonZeroU128>,

    /// KV context size.
    #[arg(long, default_value_t = 65536)]
    n_ctx: u32,

    /// KV sequences over one unified cell pool (blallama's
    /// `--cache-slots`).
    #[arg(long)]
    cache_slots: Option<u32>,

    /// Generation cap, thought included. Default: the request's own.
    #[arg(long)]
    max_tokens: Option<NonZeroU32>,

    /// When the request's `output_config` applies. See the module docs.
    #[arg(long, value_enum, default_value_t = OutputConfig::Auto)]
    output_config: OutputConfig,

    /// Drop `output_config` from every turn (`--output-config drop`).
    #[arg(long, conflicts_with = "output_config")]
    no_output_config: bool,

    /// Write each generated turn to
    /// `<dir>/<file>-t<turn>-<effort>-<sampler>-<penalty>-<seed>.txt`.
    #[arg(long)]
    out: Option<PathBuf>,

    /// With `--out`: a turn whose file exists is read back and re-scored
    /// instead of generated.
    #[arg(long, requires = "out")]
    skip_existing: bool,
}

// ---------------------------------------------------------------------------
// Axes

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum Effort {
    Low,
    Medium,
    High,
    Xhigh,
}

impl Effort {
    fn name(self) -> &'static str {
        match self {
            Effort::Low => "low",
            Effort::Medium => "medium",
            Effort::High => "high",
            Effort::Xhigh => "xhigh",
        }
    }
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Sampler {
    Sidecar,
    Card,
}

impl Sampler {
    fn name(self) -> &'static str {
        match self {
            Sampler::Sidecar => "sidecar",
            Sampler::Card => "card",
        }
    }

    /// `card` is Qwen's model-card thinking-mode chain, as in `longform`.
    fn modes(self, base: &SamplerConfig) -> Vec<SamplingMode> {
        match self {
            Sampler::Sidecar => base.modes.clone(),
            Sampler::Card => vec![
                SamplingMode::TopK {
                    k: NonZeroUsize::new(20).unwrap(),
                },
                SamplingMode::TopP {
                    p: Probability::from_f(0.95).unwrap(),
                    min_keep: NonZeroUsize::MIN,
                },
                SamplingMode::Temperature { t: 1.0 },
            ],
        }
    }
}

impl std::fmt::Display for Sampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum OutputConfig {
    Auto,
    Keep,
    Drop,
}

// ---------------------------------------------------------------------------
// Ids

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum IdClass {
    Exact,
    Miscopy,
    RunOn,
    Truncated,
    Other,
}

impl IdClass {
    const ALL: [IdClass; 5] = [
        IdClass::Exact,
        IdClass::Miscopy,
        IdClass::RunOn,
        IdClass::Truncated,
        IdClass::Other,
    ];

    fn letter(self) -> char {
        match self {
            IdClass::Exact => 'E',
            IdClass::Miscopy => 'M',
            IdClass::RunOn => 'R',
            IdClass::Truncated => 'T',
            IdClass::Other => 'O',
        }
    }

    fn name(self) -> &'static str {
        match self {
            IdClass::Exact => "exact",
            IdClass::Miscopy => "miscopy",
            IdClass::RunOn => "run-on",
            IdClass::Truncated => "truncated",
            IdClass::Other => "other",
        }
    }
}

/// The argument keys always treated as ids.
const ID_KEYS: [&str; 3] = ["id", "target", "reply_to"];

struct IdRules {
    /// A whole id, anywhere.
    find: Regex,
    /// A whole id, at the start of a value.
    prefix: Regex,
}

impl IdRules {
    fn new() -> Self {
        const ID: &str = r"(?:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|(?:GOV|APP|AMD|KEY|REC)-\d{4}(?:-\d{4}|\.\d+))";
        Self {
            find: Regex::new(&format!(r"\b{ID}\b")).unwrap(),
            prefix: Regex::new(&format!("^{ID}")).unwrap(),
        }
    }

    fn classify(&self, value: &str, known: &BTreeSet<String>) -> IdClass {
        if known.contains(value) {
            return IdClass::Exact;
        }
        if let Some(m) = self.prefix.find(value) {
            // Shape-valid: whole, or a whole id with more after it. A
            // UUID's last group followed by more hex is not a whole id
            // (the regex is greedy on the fixed width), so it lands in
            // run-on either way.
            return if m.len() == value.len() {
                IdClass::Miscopy
            } else {
                IdClass::RunOn
            };
        }
        if !value.is_empty()
            && known
                .iter()
                .any(|k| k.len() > value.len() && k.starts_with(value))
        {
            return IdClass::Truncated;
        }
        IdClass::Other
    }

    /// Every id in the prompt a turn is generated from, except in the
    /// model's own earlier turns.
    fn known(&self, prompt: &Prompt) -> BTreeSet<String> {
        let mut text = String::new();
        if let Some(system) = &prompt.system {
            text.push_str(&serde_json::to_string(system).unwrap_or_default());
        }
        if let Some(tools) = &prompt.tools {
            text.push_str(&serde_json::to_string(tools).unwrap_or_default());
        }
        for m in prompt.messages.iter().filter(|m| m.role != Role::Assistant) {
            text.push_str(&serde_json::to_string(m).unwrap_or_default());
        }
        self.find
            .find_iter(&text)
            .map(|m| m.as_str().to_owned())
            .collect()
    }

    /// [`ID_KEYS`] plus every key a recorded call in `prompt` gave a
    /// value starting with an id.
    fn keys(&self, prompt: &Prompt) -> BTreeSet<String> {
        let mut keys: BTreeSet<String> =
            ID_KEYS.iter().map(|k| k.to_string()).collect();
        for m in prompt.messages.iter().filter(|m| m.role == Role::Assistant) {
            for call in calls_of(m.content.iter().cloned()) {
                walk_strings(&call.input, None, &mut |key, value| {
                    if let Some(key) = key {
                        if self.prefix.is_match(value) {
                            keys.insert(key.to_owned());
                        }
                    }
                });
            }
        }
        keys
    }
}

/// Every string in `value`, with the key it sits under (an array element
/// inherits its array's key).
fn walk_strings<'a>(
    value: &'a Value,
    key: Option<&'a str>,
    f: &mut impl FnMut(Option<&'a str>, &'a str),
) {
    match value {
        Value::String(s) => f(key, s),
        Value::Array(items) => {
            items.iter().for_each(|v| walk_strings(v, key, f));
        }
        Value::Object(map) => {
            map.iter().for_each(|(k, v)| walk_strings(v, Some(k), f));
        }
        _ => {}
    }
}

#[derive(Clone, Debug)]
struct Call {
    name: String,
    input: Value,
}

fn calls_of(blocks: impl IntoIterator<Item = Block>) -> Vec<Call> {
    blocks
        .into_iter()
        .filter_map(|b| match b {
            Block::ToolUse { call } => Some(Call {
                name: call.name.into_owned(),
                input: call.input,
            }),
            _ => None,
        })
        .collect()
}

/// One id argument of one call.
struct IdArg {
    call: usize,
    tool: String,
    key: String,
    value: String,
    class: IdClass,
}

/// Every id argument of `calls`, in call order.
fn score_calls(
    calls: &[Call],
    rules: &IdRules,
    keys: &BTreeSet<String>,
    known: &BTreeSet<String>,
) -> Vec<IdArg> {
    let mut args = Vec::new();
    for (i, call) in calls.iter().enumerate() {
        walk_strings(&call.input, None, &mut |key, value| {
            let Some(key) = key.filter(|k| keys.contains(*k)) else {
                return;
            };
            args.push(IdArg {
                call: i,
                tool: call.name.clone(),
                key: key.to_owned(),
                value: value.to_owned(),
                class: rules.classify(value, known),
            });
        });
    }
    args
}

/// `E E R`: each call's classes, `.` for a call with no id argument.
fn class_string(n_calls: usize, args: &[IdArg]) -> String {
    let cells: Vec<String> = (0..n_calls)
        .map(|i| {
            let s: String = args
                .iter()
                .filter(|a| a.call == i)
                .map(|a| a.class.letter())
                .collect();
            if s.is_empty() {
                ".".to_owned()
            } else {
                s
            }
        })
        .collect();
    if cells.is_empty() {
        "-".to_owned()
    } else {
        cells.join(" ")
    }
}

// ---------------------------------------------------------------------------
// A generated turn, and its file

struct Meta {
    output_tokens: u64,
    thought_tokens: u64,
    stop: String,
    secs: f64,
    /// The prompt's total cell count — `cache_read_input_tokens` +
    /// `cache_creation_input_tokens` + `input_tokens`, disjoint (see
    /// `Session::last_usage`'s doc). NOT the bare `Usage::input_tokens`
    /// field alone: that's only the tail after the last `cache_control`
    /// breakpoint, and would make the "in" column shrink to near-zero
    /// on any turn that's mostly cache reuse.
    input_tokens: u64,
    cache_read: u64,
}

#[derive(Default)]
struct Output {
    thought: String,
    text: String,
    calls: Vec<Call>,
    /// `closed`, `open` or `none`.
    thought_state: &'static str,
}

impl Output {
    fn from_blocks(blocks: impl IntoIterator<Item = Block>) -> Self {
        let blocks: Vec<Block> = blocks.into_iter().collect();
        let mut out = Output::default();
        let mut open = false;
        for block in &blocks {
            open |= is_open_thought(block);
            match block {
                Block::Thought { thought, .. } => out.thought.push_str(thought),
                Block::Text { text, .. } => out.text.push_str(text),
                _ => {}
            }
        }
        out.calls = calls_of(blocks);
        out.thought_state = thought_state(open, &out.thought);
        out
    }

    fn write(&self, path: &Path, meta: &Meta) -> std::io::Result<()> {
        let mut s = String::from("=== meta ===\n");
        let _ = writeln!(s, "output_tokens: {}", meta.output_tokens);
        let _ = writeln!(s, "thought_tokens: {}", meta.thought_tokens);
        let _ = writeln!(s, "thought: {}", self.thought_state);
        let _ = writeln!(s, "stop: {}", meta.stop);
        let _ = writeln!(s, "secs: {:.1}", meta.secs);
        let _ = writeln!(s, "input_tokens: {}", meta.input_tokens);
        let _ = writeln!(s, "cache_read: {}", meta.cache_read);
        let _ = write!(s, "=== thought ===\n{}\n", self.thought);
        let _ = write!(s, "=== text ===\n{}\n", self.text);
        s.push_str("=== tool calls ===\n");
        for call in &self.calls {
            let line = serde_json::json!({
                "name": call.name,
                "input": call.input,
            });
            let _ = writeln!(s, "{line}");
        }
        std::fs::write(path, s)
    }

    fn read(path: &Path) -> std::io::Result<(Self, Meta)> {
        const SECTIONS: [&str; 4] = ["meta", "thought", "text", "tool calls"];
        let file = std::fs::read_to_string(path)?;
        let mut sections: HashMap<&str, Vec<&str>> = HashMap::new();
        let mut current = None;
        for line in file.strip_suffix('\n').unwrap_or(&file).split('\n') {
            let header = line
                .strip_prefix("=== ")
                .and_then(|l| l.strip_suffix(" ==="))
                .filter(|name| SECTIONS.contains(name));
            match (header, current) {
                (Some(name), _) => current = Some(name),
                (None, Some(name)) => {
                    sections.entry(name).or_default().push(line)
                }
                (None, None) => {}
            }
        }
        let section =
            |name| sections.get(name).map(|l| l.join("\n")).unwrap_or_default();
        let meta_text = section("meta");
        let field = |key: &str| {
            meta_text.lines().find_map(|l| {
                l.strip_prefix(key)?.strip_prefix(": ").map(str::to_owned)
            })
        };
        let num = |key: &str| field(key).and_then(|v| v.parse().ok());
        let meta = Meta {
            output_tokens: num("output_tokens").unwrap_or(0),
            thought_tokens: num("thought_tokens").unwrap_or(0),
            stop: field("stop").unwrap_or_else(|| "?".to_owned()),
            secs: field("secs").and_then(|v| v.parse().ok()).unwrap_or(0.0),
            input_tokens: num("input_tokens").unwrap_or(0),
            cache_read: num("cache_read").unwrap_or(0),
        };
        let calls = section("tool calls")
            .lines()
            .filter_map(|l| serde_json::from_str::<Value>(l).ok())
            .map(|v| Call {
                name: v["name"].as_str().unwrap_or_default().to_owned(),
                input: v["input"].clone(),
            })
            .collect();
        let out = Output {
            thought: section("thought"),
            text: section("text"),
            calls,
            thought_state: match field("thought").as_deref() {
                Some("closed") => "closed",
                Some("open") => "open",
                Some("none") => "none",
                _ => "?",
            },
        };
        Ok((out, meta))
    }
}

fn thought_state(open: bool, thought: &str) -> &'static str {
    if open {
        "open"
    } else if thought.is_empty() {
        "none"
    } else {
        "closed"
    }
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

// ---------------------------------------------------------------------------
// The model

/// Loaded on first use, and reloaded only when a request names another
/// model, so `--skip-existing` over finished runs loads nothing.
struct Lazy<'a> {
    args: &'a Args,
    catalog: Catalog<LlamaCppBackend>,
    loaded: Option<(String, LlamaCppSession)>,
    base: SamplerConfig,
}

impl Lazy<'_> {
    fn session(
        &mut self,
        model: &str,
    ) -> Result<&mut LlamaCppSession, Box<dyn std::error::Error>> {
        if self.loaded.as_ref().is_some_and(|(name, _)| name != model) {
            // Free the outgoing model first, as blallama does.
            self.loaded = None;
        }
        if self.loaded.is_none() {
            let name = self.catalog.resolve(model, None).map_err(|e| {
                format!(
                    "{model} not under {}: {e:?}",
                    self.args.models.display()
                )
            })?;
            let path = self.catalog.path_of(&name);
            eprintln!("loading {}", path.display());
            let _ = LlamaCppBackend::set_log_callback(|level, text| {
                if matches!(level, LogLevel::Error) {
                    eprint!("{text}");
                }
            });
            let mut session = LlamaCppSession::from_path_with(
                path.clone(),
                *self.catalog.options(),
            )?
            .quiet()
            .with_prefix_cache(true);
            if let Some(effort) = self.args.effort {
                session = session.with_render_opts(
                    RenderOptions::default()
                        .with_extra("reasoning_effort", effort.name()),
                );
            }
            // Read *after* the load, which writes a default sidecar if the
            // model had none.
            self.base = load_base_config(&path, self.args.sampling.as_deref())?;
            self.loaded = Some((model.to_owned(), session));
        }
        Ok(&mut self.loaded.as_mut().unwrap().1)
    }
}

fn generate(
    lazy: &mut Lazy,
    prompt: &Prompt,
    (sampler, penalty, seed): (Sampler, Penalty, NonZeroU128),
) -> Result<(Output, Meta), Box<dyn std::error::Error>> {
    let model = lazy
        .args
        .model
        .clone()
        .unwrap_or_else(|| prompt.model.to_string());
    lazy.session(&model)?;
    let mut config = lazy.base.clone();
    config.modes = sampler.modes(&lazy.base);
    config.repetition = penalty.repetition(&lazy.base.repetition);
    let (name, session) = lazy.loaded.take().unwrap();
    let session = &mut lazy
        .loaded
        .insert((
            name,
            session.with_sample_options(config).with_seed(Some(seed)),
        ))
        .1;

    let start = Instant::now();
    let result = session.complete_response(prompt);
    let secs = start.elapsed().as_secs_f64();

    let usage = session.last_usage().clone();
    let (blocks, stop): (Vec<Block>, String) = match result {
        Ok(response) => {
            let message: Message = response.inner.into();
            (
                message.content.into_iter().collect(),
                stop_name(response.stop_reason).to_owned(),
            )
        }
        // A tool call the budget cut off, or one that never closed: the
        // partial output is the finding, not a crash.
        Err(SessionError::GrammarViolation { partial_output }) => {
            let max = u64::from(prompt.max_tokens.get());
            let stop = if usage.output_tokens >= max {
                "max"
            } else {
                "viol"
            };
            (partial_output.into_iter().collect(), stop.to_owned())
        }
        Err(SessionError::EmittedSpecialToken { .. }) => {
            (Vec::new(), "special".to_owned())
        }
        Err(e) if !e.is_fatal() => {
            eprintln!("error: {e}");
            (Vec::new(), "error".to_owned())
        }
        Err(e) => return Err(e.into()),
    };

    let out = Output::from_blocks(blocks);
    let thought_tokens =
        session.engine().model().tokenize(&out.thought, false).len() as u64;
    // The three input counters are disjoint (read + creation + input);
    // `input_tokens` alone is only the post-breakpoint tail, so sum all
    // three for "the prompt's total size" — what this table's "in"
    // column means.
    let prompt_total = usage.cache_read_input_tokens.unwrap_or(0)
        + usage.cache_creation_input_tokens.unwrap_or(0)
        + usage.input_tokens;
    let meta = Meta {
        output_tokens: usage.output_tokens,
        thought_tokens,
        stop,
        secs,
        input_tokens: prompt_total,
        cache_read: usage.cache_read_input_tokens.unwrap_or(0),
    };
    Ok((out, meta))
}

// ---------------------------------------------------------------------------
// Requests

struct Request {
    stem: String,
    prompt: Prompt,
    keys: BTreeSet<String>,
}

impl Request {
    fn load(
        path: &Path,
        rules: &IdRules,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let body = std::fs::read_to_string(path)?;
        // What blallama's `Json<Prompt>` extractor does.
        let prompt: Prompt = serde_json::from_str(&body)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let stem = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let keys = rules.keys(&prompt);
        Ok(Self { stem, prompt, keys })
    }

    fn assistant_turns(&self) -> impl Iterator<Item = usize> + '_ {
        self.prompt
            .messages
            .iter()
            .enumerate()
            .filter(|(_, m)| m.role == Role::Assistant)
            .map(|(i, _)| i)
    }

    /// The request that produced assistant message `turn`, as far as it
    /// can be reconstructed.
    fn prompt_for(&self, turn: usize, args: &Args) -> Prompt {
        let mut prompt = self.prefix(turn);
        if let Some(max) = args.max_tokens {
            prompt.max_tokens = max;
        }
        let mode = if args.no_output_config {
            OutputConfig::Drop
        } else {
            args.output_config
        };
        let keep = match mode {
            OutputConfig::Keep => true,
            OutputConfig::Drop => false,
            OutputConfig::Auto => self.recorded(turn).text_is_json(),
        };
        if !keep {
            prompt.output_config = None;
        }
        prompt
    }

    /// The request, cut to the messages before `turn`.
    fn prefix(&self, turn: usize) -> Prompt {
        let mut prompt = self.prompt.clone();
        prompt.messages.truncate(turn);
        prompt
    }

    fn recorded(&self, turn: usize) -> Output {
        Output::from_blocks(self.prompt.messages[turn].content.iter().cloned())
    }
}

impl Output {
    fn text_is_json(&self) -> bool {
        serde_json::from_str::<Value>(self.text.trim())
            .is_ok_and(|v| v.is_object())
    }
}

// ---------------------------------------------------------------------------
// Reporting

/// Totals per variant, class and call index; the last bucket is `5+`.
const BUCKETS: usize = 5;

#[derive(Default)]
struct Summary {
    /// Variant label → class → per-call-index counts.
    counts: BTreeMap<String, BTreeMap<IdClass, [u32; BUCKETS]>>,
    turns: BTreeMap<String, u32>,
    /// `variant file turn call tool.key = value` for every non-exact id.
    misses: Vec<String>,
}

impl Summary {
    fn add(&mut self, variant: &str, stem: &str, turn: usize, args: &[IdArg]) {
        *self.turns.entry(variant.to_owned()).or_default() += 1;
        let classes = self.counts.entry(variant.to_owned()).or_default();
        for a in args {
            classes.entry(a.class).or_default()[a.call.min(BUCKETS - 1)] += 1;
            if a.class != IdClass::Exact {
                let value: String = a.value.chars().take(80).collect();
                self.misses.push(format!(
                    "{variant:<24} {stem} t{turn} #{} {}.{} {}: {value:?}",
                    a.call + 1,
                    a.tool,
                    a.key,
                    a.class.name(),
                ));
            }
        }
    }

    fn print(&self) {
        println!("\n== id classes by call index, per variant ==");
        let mut header = format!("{:<24} {:<9}", "variant", "class");
        for i in 1..=BUCKETS {
            let label = if i == BUCKETS {
                format!("{i}+")
            } else {
                i.to_string()
            };
            let _ = write!(header, " {label:>5}");
        }
        println!("{header} {:>6}", "total");
        for (variant, classes) in &self.counts {
            let turns = self.turns.get(variant).copied().unwrap_or(0);
            println!("{variant} ({turns} turns)");
            for class in IdClass::ALL {
                let row = classes.get(&class).copied().unwrap_or_default();
                let mut line = format!("{:<24} {:<9}", "", class.name());
                for n in row {
                    let _ = write!(line, " {n:>5}");
                }
                println!("{line} {:>6}", row.iter().sum::<u32>());
            }
        }
        if !self.misses.is_empty() {
            println!("\n== non-exact ids ==");
            for m in &self.misses {
                println!("{m}");
            }
        }
    }
}

fn print_header() {
    println!(
        "{:<14} {:>4} {:<8} {:<8} {:>4} {:>6} {:>6} {:<6} {:<7} {:>5} {:>7} {:>5} {:>6} {:>6}  ids",
        "file",
        "turn",
        "sampler",
        "penalty",
        "seed",
        "out",
        "think",
        "thk",
        "stop",
        "calls",
        "secs",
        "tok/s",
        "in",
        "cached",
    );
}

struct Row<'a> {
    stem: &'a str,
    turn: usize,
    sampler: &'a str,
    penalty: &'a str,
    seed: &'a str,
    out: &'a Output,
    meta: Option<&'a Meta>,
    ids: &'a str,
}

impl Row<'_> {
    fn print(&self) {
        let dash = || "-".to_owned();
        let (out, think, stop, secs, tok_s, input, cached) = match self.meta {
            Some(m) => (
                m.output_tokens.to_string(),
                m.thought_tokens.to_string(),
                m.stop.clone(),
                format!("{:.1}", m.secs),
                format!(
                    "{:.1}",
                    if m.secs > 0.0 {
                        m.output_tokens as f64 / m.secs
                    } else {
                        0.0
                    }
                ),
                m.input_tokens.to_string(),
                m.cache_read.to_string(),
            ),
            None => (
                dash(),
                format!("{}c", self.out.thought.chars().count()),
                dash(),
                dash(),
                dash(),
                dash(),
                dash(),
            ),
        };
        println!(
            "{:<14} {:>4} {:<8} {:<8} {:>4} {:>6} {:>6} {:<6} {:<7} {:>5} {:>7} {:>5} {:>6} {:>6}  {}",
            self.stem,
            self.turn,
            self.sampler,
            self.penalty,
            self.seed,
            out,
            think,
            self.out.thought_state,
            stop,
            self.out.calls.len(),
            secs,
            tok_s,
            input,
            cached,
            self.ids,
        );
        let _ = std::io::stdout().flush();
    }
}

fn list(request: &Request, rules: &IdRules) {
    println!(
        "\n== {} ({} messages) ==",
        request.stem,
        request.prompt.messages.len()
    );
    for turn in request.assistant_turns() {
        let rec = request.recorded(turn);
        let known = rules.known(&request.prefix(turn));
        let args = score_calls(&rec.calls, rules, &request.keys, &known);
        println!(
            "[{turn}] thought {}c ({}), text {}c{}, {} call(s): {}",
            rec.thought.chars().count(),
            rec.thought_state,
            rec.text.chars().count(),
            if rec.text_is_json() { " (json)" } else { "" },
            rec.calls.len(),
            class_string(rec.calls.len(), &args),
        );
        for (i, call) in rec.calls.iter().enumerate() {
            let mut input = call.input.clone();
            // Bodies are long and carry no ids worth reading here.
            if let Some(body) = input.get_mut("body") {
                let n = body.as_str().map_or(0, |b| b.chars().count());
                *body = Value::String(format!("<{n} chars>"));
            }
            let mut input = input.to_string();
            if input.chars().count() > 160 {
                input = input.chars().take(160).collect::<String>() + "…";
            }
            println!("    #{} {}({input})", i + 1, call.name);
        }
    }
}

// ---------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let rules = IdRules::new();
    let requests = args
        .request
        .iter()
        .map(|p| Request::load(p, &rules))
        .collect::<Result<Vec<_>, _>>()?;

    if args.list {
        for request in &requests {
            list(request, &rules);
        }
        return Ok(());
    }

    if let Some(dir) = &args.out {
        std::fs::create_dir_all(dir)?;
    }
    let mut options = LlamaCppOptions::default().with_n_ctx(args.n_ctx);
    if let Some(slots) = args.cache_slots {
        options = options.with_cache_slots(slots);
    }
    let mut lazy = Lazy {
        args: &args,
        catalog: Catalog::new(&args.models, options),
        loaded: None,
        base: SamplerConfig::default(),
    };
    let effort = args.effort.map_or("default", Effort::name);
    println!("reasoning_effort: {effort}");

    let mut summary = Summary::default();
    for request in &requests {
        let turns: Vec<usize> = request
            .assistant_turns()
            .filter(|&i| args.turn.is_none_or(|t| t == i))
            .filter(|&i| {
                args.only.is_empty()
                    || args
                        .only
                        .iter()
                        .any(|o| *o == format!("{}:{i}", request.stem))
            })
            .collect();
        if turns.is_empty() {
            eprintln!("{}: no assistant turn to replay", request.stem);
            continue;
        }
        println!("\n== {} ==", request.stem);
        print_header();
        for turn in turns {
            let prompt = request.prompt_for(turn, &args);
            let known = rules.known(&prompt);

            let rec = request.recorded(turn);
            let rec_args =
                score_calls(&rec.calls, &rules, &request.keys, &known);
            let ids = class_string(rec.calls.len(), &rec_args);
            Row {
                stem: &request.stem,
                turn,
                sampler: "recorded",
                penalty: "-",
                seed: "-",
                out: &rec,
                meta: None,
                ids: &ids,
            }
            .print();
            summary.add("recorded", &request.stem, turn, &rec_args);

            for &sampler in &args.sampler {
                for &penalty in &args.penalty {
                    for &seed in &args.seeds {
                        let path = args.out.as_ref().map(|d| {
                            d.join(format!(
                                "{}-t{turn}-{effort}-{sampler}-{penalty}-{seed}.txt",
                                request.stem
                            ))
                        });
                        let existing = path
                            .as_ref()
                            .filter(|p| args.skip_existing && p.exists());
                        let (out, meta) = match existing {
                            Some(path) => Output::read(path)?,
                            None => {
                                let (out, meta) = generate(
                                    &mut lazy,
                                    &prompt,
                                    (sampler, penalty, seed),
                                )?;
                                if let Some(path) = &path {
                                    out.write(path, &meta)?;
                                }
                                (out, meta)
                            }
                        };
                        let id_args = score_calls(
                            &out.calls,
                            &rules,
                            &request.keys,
                            &known,
                        );
                        let ids = class_string(out.calls.len(), &id_args);
                        Row {
                            stem: &request.stem,
                            turn,
                            sampler: sampler.name(),
                            penalty: penalty.name(),
                            seed: &seed.to_string(),
                            out: &out,
                            meta: Some(&meta),
                            ids: &ids,
                        }
                        .print();
                        summary.add(
                            &format!("{effort}/{sampler}/{penalty}"),
                            &request.stem,
                            turn,
                            &id_args,
                        );
                    }
                }
            }
        }
    }
    summary.print();
    Ok(())
}
