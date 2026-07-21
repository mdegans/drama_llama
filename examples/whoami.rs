//! **Who do you think you are?** — an identity probe over the raw
//! [`Engine`] / [`CandidatePredictor`] API.
//!
//! Ask a local model what it is called, then instead of reading the
//! string it generates, read the *distribution* it would have generated
//! from. That turns "the open-weights models were trained on Claude
//! output" from a claim into a measurement — visible in the tail, not in
//! the answer.
//!
//! ```sh
//! # score the default candidate names against models/model.gguf
//! cargo run --release --example whoami --features cli
//!
//! # a specific model in models/, by substring
//! cargo run --release --example whoami --features cli -- --model gemma
//!
//! # every model in models/, in turn
//! cargo run --release --example whoami --features cli -- --all
//!
//! # no constraint: top-K at the answer position + what it actually says
//! cargo run --release --example whoami --features cli -- --unconstrained
//! ```
//!
//! # Why the naive version is wrong
//!
//! Names are **multi-token**. `Claude` may be one token or two; `ChatGPT`
//! is usually two or three. Reading one candidate set at the answer
//! position gives you `P(first token)`, which is not `P(name)` and is
//! outright ambiguous when two names share a leading token. So this
//! example scores each name by *forced continuation*:
//!
//! ```text
//! P(name | prompt) = ∏ᵢ P(tokenᵢ | prompt, token₍<ᵢ₎)
//! ```
//!
//! which is exactly what [`CandidatePredictor`] is for. Its
//! [`record_choice`] method exists so the caller picks the token instead
//! of a sampler — here we pick the token the *name* demands and read off
//! what the model thought of that choice on the way past. One prefill per
//! name (the predictor clears the KV cache on construction); the prompt
//! is tiny, so the cost is irrelevant and we avoid the partial-truncate
//! rewind that recurrent/hybrid models reject.
//!
//! # What the numbers do and do not support
//!
//! Mass on `Claude` is evidence that text asserting "I am Claude" was in
//! the training corpus. That is *not* the same as proof of distillation
//! on Claude outputs: the open web is full of Claude transcripts, and a
//! model with no identity anchor in its system prompt is guessing from
//! whatever assistant-shaped text it saw. Read this as one signal with a
//! specific shape, and quote it with its provenance header attached —
//! which is why the header exists and why `--no-hash` is opt-in rather
//! than the default.
//!
//! The sharpest version of that is when the prompt *already contains the
//! answer*. gpt-oss's Harmony template hardcodes "You are ChatGPT, a
//! large language model trained by OpenAI" into its system message, so
//! gpt-oss scoring `GPT` at 0.61 is instruction-following and says
//! nothing about its corpus. The example checks the rendered text for
//! every candidate and says so; a run with that warning is measuring a
//! different thing than a run without it. (Qwen and Gemma render clean
//! here — no identity anchor of any kind.)
//!
//! Two more caveats worth stating before someone screenshots a table:
//!
//! * Joint probability mechanically favours **shorter** names. A
//!   two-token name clears a lower bar than a three-token one. The
//!   `share` column renormalises across the candidate set, which is the
//!   honest answer to "which of these does it think it is", but it
//!   inherits the same bias.
//! * The `other` residual (`1 − Σ`) is only a clean "none of the above"
//!   when the candidate set is **prefix-free**. The default set is.
//!
//! # Reasoning models
//!
//! The answer position is wherever the template's generation prompt ends,
//! and for a thinking model that is where a reasoning block opens, not
//! where the answer starts. Two things handle this:
//!
//! * The rendered [`Prompt`] leaves `thinking` unset, so
//!   [`ChatTemplate`] passes `enable_thinking = false` and Qwen-style
//!   templates close an empty `<think></think>` for us. Free, and
//!   template-native.
//! * `--prefix` appends literal text after the generation prompt for
//!   templates that need a nudge anyway — gpt-oss's Harmony wants a
//!   channel opened: `--prefix '<|channel|>final<|message|>'`.
//!
//! The `argmax at answer position` line in the header tells you which
//! situation you are in without having to guess. If it reads `<think>`,
//! the scores below it are answering a counterfactual.
//!
//! [`Engine`]: drama_llama::Engine
//! [`CandidatePredictor`]: drama_llama::CandidatePredictor
//! [`record_choice`]: drama_llama::CandidatePredictor::record_choice

use std::collections::btree_map::{BTreeMap, Entry};
use std::collections::HashSet;
use std::io::{Read, Write};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::Command;

use clap::Parser;
use drama_llama::{
    ChatTemplate, LlamaCppEngine, Model, PredictOptions, RenderOptions, Token,
};
use misanthropic::{prompt::message::Role, Prompt};
use sha2::{Digest, Sha256};

/// The question. Short, and asks for the bare name — a preamble would
/// push the name off the answer position entirely.
const DEFAULT_QUESTION: &str =
    "What is your model's name? Just the name, please.";

/// Prefix-free by construction; see the `other` residual caveat above.
///
/// `GPT` earns its place next to `ChatGPT` empirically: asked this
/// question with no system prompt, Gemma-4-31B puts p=0.95 on a bare
/// `GPT` token and continues `-4o`. A candidate set that only knew about
/// `ChatGPT` would have reported 95% of the mass as "other" and missed
/// the single most interesting thing in the distribution.
const DEFAULT_NAMES: &[&str] = &[
    "Claude", "ChatGPT", "GPT", "Gemini", "Gemma", "Qwen", "Llama", "Mistral",
    "DeepSeek", "Grok", "Copilot",
];

#[derive(Parser, Debug)]
#[command(version, about = "Ask a model who it thinks it is.")]
struct Args {
    /// A path to a GGUF, or a case-insensitive substring matching one
    /// file in `models/` (`qwen`, `gemma`, `gptoss`, …).
    #[arg(short, long, default_value = "model.gguf")]
    model: String,

    /// Probe every GGUF in `models/` in turn. Skips projector sidecars
    /// and the `model.gguf` alias (it is a link to one of the others).
    #[arg(long)]
    all: bool,

    /// The question to ask.
    #[arg(short, long, default_value = DEFAULT_QUESTION)]
    prompt: String,

    /// System prompt. Absent by default — "no system prompt" is a real
    /// experimental condition, and the header records it as one.
    #[arg(long)]
    system: Option<String>,

    /// Candidate names to score. Repeatable, or comma-separated.
    #[arg(long, value_delimiter = ',')]
    name: Vec<String>,

    /// Skip scoring. Print the unconstrained top-K at the answer
    /// position, then greedily generate what the model actually says.
    #[arg(long)]
    unconstrained: bool,

    /// Candidates to list under `--unconstrained`.
    #[arg(long, default_value_t = 15)]
    top_k: usize,

    /// Tokens to generate under `--unconstrained`.
    #[arg(long, default_value_t = 64)]
    max_tokens: usize,

    /// Literal text appended after the template's assistant header. See
    /// the reasoning-models note in the module docs.
    #[arg(long, default_value = "")]
    prefix: String,

    /// Context size.
    #[arg(long, default_value_t = 4096)]
    n_ctx: u32,

    /// Skip the model sha256. Faster to iterate on; weakens the capture,
    /// which is the whole point of the header.
    #[arg(long)]
    no_hash: bool,

    /// Print the rendered prompt and the per-token score breakdown.
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    // Before the first load: llama.cpp's loader is chatty and this
    // example's output is meant to be pasted somewhere.
    drama_llama::log::silence_logs();

    let models = resolve_models(&args)?;
    for (i, path) in models.iter().enumerate() {
        if i > 0 {
            println!();
        }
        match probe(path, &args) {
            Ok(()) => {}
            // A sweep must not die on one bad file. `models/` can hold
            // GGUFs that are not language models at all (a projector the
            // sidecar convention didn't catch, a vocab-less export); they
            // fail at load or at `ChatTemplate::from_model`, and the right
            // answer is to say so and move on. A single explicit `--model`
            // still fails loudly — there is nothing to move on to.
            Err(e) if models.len() > 1 => {
                eprintln!(
                    "skipping {}: {e}",
                    path.file_name()
                        .unwrap_or(path.as_os_str())
                        .to_string_lossy(),
                );
            }
            Err(e) => return Err(e),
        }
    }

    Ok(())
}

// ===========================================================================
// The probe
// ===========================================================================

fn probe(
    model_path: &Path,
    args: &Args,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut engine = LlamaCppEngine::from_path_with_n_ctx(
        model_path.to_path_buf(),
        args.n_ctx,
    )?;

    // Model -> template -> rendered text -> tokens. This is the layer
    // `Session` sits on top of; doing it by hand is most of the point of
    // the example.
    let (template, template_source) = load_template(&engine, model_path)?;
    let mut prompt =
        Prompt::default().messages([(Role::User, args.prompt.clone())])?;
    if let Some(system) = &args.system {
        prompt = prompt.system(system.clone());
    }
    // `thinking` is left unset, so `enable_thinking` renders false and a
    // Qwen-style template closes an empty reasoning block for us.
    let render_opts = RenderOptions::default().with_generation_prompt(true);
    let mut rendered = template.render_with(&prompt, &render_opts)?;
    rendered.push_str(&args.prefix);

    // `special = true`: the render is full of control tokens and they
    // must resolve to their single ids, not to their spellings. Same
    // call `Session` makes.
    let prompt_tokens = engine.model().tokenize(&rendered, true);

    let names: Vec<String> = if args.name.is_empty() {
        DEFAULT_NAMES.iter().map(|s| s.to_string()).collect()
    } else {
        args.name.clone()
    };

    print_header(&engine, model_path, args, &template_source, &prompt_tokens)?;

    // The confound that makes a probe worthless without saying so: a
    // candidate name that appears in the prompt is not being *recalled*,
    // it is being read back. gpt-oss's Harmony template hardcodes "You
    // are ChatGPT, a large language model trained by OpenAI" into its
    // system message, so gpt-oss answering `GPT` is instruction-following
    // and nothing more. The header's `system prompt: (none)` is true of
    // *our* prompt and quietly false in effect, which is exactly the kind
    // of gap that survives into a quotation.
    //
    // Exact test, no interpretation: does the rendered text contain the
    // candidate? Substring, so a template naming `ChatGPT` correctly
    // flags `GPT` too — both are confounded by it.
    let haystack = rendered.to_lowercase();
    let leaked: Vec<&str> = names
        .iter()
        .filter(|name| haystack.contains(&name.to_lowercase()))
        .map(|name| name.as_str())
        .collect();
    if !leaked.is_empty() {
        println!(
            "\n  ⚠ the prompt itself names {} — the template or --system \
             told the\n    model what it is. Those rows measure \
             instruction-following, not\n    recall. Re-read the render \
             with --verbose before quoting them.",
            leaked.join(", "),
        );
    }

    if args.verbose {
        println!("\n--- rendered prompt ---\n{rendered}\n--- end ---");
    }

    // One free-standing peek at the answer position, shared by both
    // modes: the header's sanity line in scoring mode, the main event
    // under `--unconstrained`.
    let k = if args.unconstrained { args.top_k } else { 3 };
    let peeked = peek(&mut engine, &prompt_tokens, k.max(1));

    println!(
        "\n  argmax at answer position: {} p={:.4}",
        quoted(&peeked[0].piece),
        peeked[0].p,
    );

    // The trap this example is most likely to be quoted through. If the
    // model's own next move is to open a structured channel — Harmony's
    // `<|channel|>`, an un-suppressed `<think>` — then the answer
    // position is not where the answer goes, and every score below is
    // conditioned on a continuation the model would never have written.
    // Exact test, not a heuristic: ask the vocabulary.
    if engine.model().special_tokens().contains(&peeked[0].id) {
        println!(
            "  ⚠ that is a control token — the model opens a structured \
             channel here,\n    not an answer. Scores below are \
             counterfactual until --prefix moves\n    the probe inside \
             the channel (see the module docs)."
        );
    }

    if args.unconstrained {
        unconstrained(&mut engine, &prompt_tokens, &peeked, args);
    } else {
        scored(&mut engine, &prompt_tokens, &names, args);
    }

    Ok(())
}

/// Which Jinja source actually rendered the prompt.
struct TemplateSource {
    origin: String,
    digest: String,
}

/// Resolve the chat template the way [`Session`] does: a
/// `<model>.template.jinja` sidecar **overrides** the model's embedded
/// `tokenizer.chat_template`.
///
/// Worth the extra dozen lines over a bare `ChatTemplate::from_model`,
/// because getting it wrong invalidates the capture silently. Both
/// gemma-4 and gpt-oss ship an override in `models/` — the vendored
/// Gemma one fixes a re-ingest bug that breaks KV-cache byte-stability
/// — so an example that ignored sidecars would probe a template nothing
/// actually serves, and the header would look just as authoritative
/// while doing it.
///
/// [`Session`]: drama_llama::Session
fn load_template(
    engine: &LlamaCppEngine,
    model_path: &Path,
) -> Result<(ChatTemplate, TemplateSource), Box<dyn std::error::Error>> {
    let model = engine.model();
    let sidecar_path = model_path.with_extension("template.jinja");

    if let Some(source) =
        drama_llama::sidecar::load_template_source(&sidecar_path)?
    {
        let digest = short_digest(source.as_bytes());
        let template = ChatTemplate::from_source(
            source,
            model.token_to_piece(model.bos()),
            model.token_to_piece(model.eos()),
        )?;
        let name = sidecar_path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        return Ok((
            template,
            TemplateSource {
                origin: format!("sidecar ({name})"),
                digest,
            },
        ));
    }

    let digest = match model.chat_template_source() {
        Some(source) => short_digest(source.as_bytes()),
        None => "(none)".to_string(),
    };
    Ok((
        ChatTemplate::from_model(model)?,
        TemplateSource {
            origin: "embedded (tokenizer.chat_template)".to_string(),
            digest,
        },
    ))
}

/// One candidate, softmaxed over the whole vocabulary.
struct Peek {
    id: Token,
    piece: String,
    p: f32,
}

/// Top-`k` at the answer position, with **true** probabilities.
///
/// Note the `softmax(None)`: `softmax(Some(k))` truncates to `k` first
/// and renormalises *within* the survivors, so its `p` values are
/// conditional on the top-k and always sum to 1. Correct for sampling,
/// wrong for reporting. We softmax over everything and slice afterwards.
fn peek(
    engine: &mut LlamaCppEngine,
    prompt_tokens: &[Token],
    k: usize,
) -> Vec<Peek> {
    let one = NonZeroUsize::new(1).unwrap();
    let mut predictor = engine.predict_candidates(prompt_tokens.to_vec(), one);
    let candidates = predictor
        .next()
        .expect("a fresh CandidatePredictor always yields once")
        .softmax(None);
    let top: Vec<(Token, f32)> = candidates
        .iter()
        .take(k)
        .map(|data| (data.id, data.p))
        .collect();
    drop(predictor);

    let model = engine.model();
    top.into_iter()
        .map(|(id, p)| Peek {
            id,
            piece: model.token_to_piece(id),
            p,
        })
        .collect()
}

// ===========================================================================
// Constrained: forced-continuation scoring
// ===========================================================================

/// What the model thought of one forced token.
struct Step {
    piece: String,
    p: f32,
    /// Position in the full descending-by-logit ordering. Rank 0 means
    /// the model would have chosen this token anyway.
    rank: usize,
}

struct Score {
    name: String,
    steps: Vec<Step>,
    joint: f64,
}

fn scored(
    engine: &mut LlamaCppEngine,
    prompt_tokens: &[Token],
    names: &[String],
    args: &Args,
) {
    let mut scores: Vec<Score> = names
        .iter()
        .filter_map(|name| {
            let score = score_name(engine, prompt_tokens, name);
            // A candidate that tokenizes to nothing cannot be scored, but
            // dropping it quietly would leave the reader believing the
            // table covers everything they asked for.
            if score.is_none() {
                eprintln!("  (skipped {}: tokenizes to nothing)", quoted(name));
            }
            score
        })
        .collect();
    scores.sort_by(|a, b| b.joint.total_cmp(&a.joint));

    let total: f64 = scores.iter().map(|s| s.joint).sum();

    println!();
    println!(
        "  {:<12} {:>3}  {:>11}  {:>7}  {}",
        "name", "tok", "P(name)", "share", "ranks"
    );
    println!("  {}", "─".repeat(58));
    for score in &scores {
        let ranks: Vec<String> =
            score.steps.iter().map(|s| s.rank.to_string()).collect();
        println!(
            "  {:<12} {:>3}  {:>11}  {:>6.2}%  {}",
            score.name,
            score.steps.len(),
            format!("{:.3e}", score.joint),
            if total > 0.0 {
                100.0 * score.joint / total
            } else {
                0.0
            },
            ranks.join(","),
        );
    }
    println!("  {}", "─".repeat(58));
    println!(
        "  {:<12} {:>3}  {:>11}   (mass on everything else: {:.2}%)",
        "Σ",
        "",
        format!("{total:.3e}"),
        100.0 * (1.0 - total).max(0.0),
    );

    if args.verbose {
        println!("\n  per-token breakdown");
        for score in &scores {
            let parts: Vec<String> = score
                .steps
                .iter()
                .map(|s| {
                    format!("{} p={:.4} #{}", quoted(&s.piece), s.p, s.rank)
                })
                .collect();
            println!("    {:<12} {}", score.name, parts.join("  │  "));
        }
    }
}

/// `P(name | prompt)` by forcing the name's tokens one at a time.
///
/// Returns `None` for a name that tokenizes to nothing.
fn score_name(
    engine: &mut LlamaCppEngine,
    prompt_tokens: &[Token],
    name: &str,
) -> Option<Score> {
    // `tokenize_special(name, false, false)`: no automatic BOS (we are
    // continuing a prompt, not starting one) and no special-token
    // parsing (a name is content, and content never spells a control
    // token). The name is tokenized standalone, so if the model would
    // have preferred a leading-space variant this is a lower bound —
    // applied identically to every candidate, so the comparison holds.
    let model = engine.model();
    let tokens = model.tokenize_special(name, false, false);
    let n = NonZeroUsize::new(tokens.len())?;
    let pieces: Vec<String> =
        tokens.iter().map(|&t| model.token_to_piece(t)).collect();

    let mut predictor = engine.predict_candidates(prompt_tokens.to_vec(), n);
    let mut joint = 1.0f64;
    let mut steps = Vec::with_capacity(tokens.len());

    for (&token, piece) in tokens.iter().zip(pieces) {
        let candidates = predictor
            .next()
            .expect("CandidatePredictor yields exactly `n` sets")
            .softmax(None);
        // Sorted descending by logit, so enumeration index *is* the rank
        // and every vocabulary token is present — the forced token is
        // findable no matter how unlikely it was.
        let (rank, data) = candidates
            .iter()
            .enumerate()
            .find(|(_, data)| data.id == token)
            .expect("a full softmax contains every token");
        joint *= data.p as f64;
        steps.push(Step {
            piece,
            p: data.p,
            rank,
        });
        // The bit that makes this a `CandidatePredictor` and not a
        // sampler: we choose the next token, so the model is scored
        // against the continuation *we* wanted.
        predictor.record_choice(token);
    }

    Some(Score {
        name: name.to_string(),
        steps,
        joint,
    })
}

// ===========================================================================
// Unconstrained: top-K, then let it talk
// ===========================================================================

fn unconstrained(
    engine: &mut LlamaCppEngine,
    prompt_tokens: &[Token],
    peeked: &[Peek],
    args: &Args,
) {
    println!("\n  top {} at the answer position", peeked.len());
    println!("  {}", "─".repeat(40));
    let width = peeked.iter().map(|p| p.p).fold(0.0f32, f32::max).max(1e-9);
    for (rank, entry) in peeked.iter().enumerate() {
        let bar = "█".repeat(((entry.p / width) * 24.0).round() as usize);
        println!(
            "  {:>3}  {:<18} {:>7.4}  {}",
            rank,
            quoted(&entry.piece),
            entry.p,
            bar
        );
    }

    let Some(n) = NonZeroUsize::new(args.max_tokens) else {
        return;
    };
    let mut options = PredictOptions::greedy().add_model_stops(engine.model());
    options.n = n;

    println!("\n  greedy continuation");
    println!("  {}", "─".repeat(40));
    print!("  ");
    // `PiecePredictor` — the streaming half of the raw API. Pieces
    // arrive UTF-8-reassembled, so printing them straight is safe even
    // when a codepoint splits across tokens.
    for piece in engine.predict_pieces(prompt_tokens.to_vec(), options, None) {
        print!("{}", piece.replace('\n', "\n  "));
        std::io::stdout().flush().ok();
    }
    println!();
}

// ===========================================================================
// Provenance
// ===========================================================================

/// A capture is worth exactly as much as what you know about how it was
/// taken: `provider_source × capture_date × wrapper_version ×
/// sampler_settings`. Scoring is deterministic, so there is no sampler
/// to record — everything else goes here.
fn print_header(
    engine: &LlamaCppEngine,
    model_path: &Path,
    args: &Args,
    template: &TemplateSource,
    prompt_tokens: &[Token],
) -> Result<(), Box<dyn std::error::Error>> {
    let model = engine.model();
    let bar = "═".repeat(66);
    println!("{bar}");
    println!(" whoami — identity probe");
    println!("{bar}");

    let field = |k: &str, v: &str| println!(" {k:<16}{v}");

    field("captured", &utc_now());
    field(
        "drama_llama",
        &format!("{} (git {})", env!("CARGO_PKG_VERSION"), git_head()),
    );
    field("model file", &model_path.display().to_string());
    if args.no_hash {
        field("model sha256", "(skipped: --no-hash)");
    } else {
        let started = std::time::Instant::now();
        let digest = sha256_file(model_path)?;
        field(
            "model sha256",
            &format!(
                "{digest}  ({:.1} GiB in {:.1}s)",
                model.size() as f64 / (1 << 30) as f64,
                started.elapsed().as_secs_f64(),
            ),
        );
    }
    field("model desc", &model.desc());
    field(
        "general.name",
        &model
            .get_meta("general.name")
            .unwrap_or_else(|| "(unset)".into()),
    );
    field("template", &template.origin);
    field("template sha256", &template.digest);
    field(
        "system prompt",
        &match &args.system {
            Some(s) => quoted(s),
            None => "(none)".into(),
        },
    );
    field("question", &quoted(&args.prompt));
    field(
        "prefix",
        &if args.prefix.is_empty() {
            "(none)".into()
        } else {
            quoted(&args.prefix)
        },
    );
    field("prompt tokens", &prompt_tokens.len().to_string());
    println!(" {}", "─".repeat(65));

    Ok(())
}

/// Streaming sha256 — the model is tens of gigabytes and must not be
/// read into memory. Warm page cache makes repeat runs much cheaper than
/// the first.
fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 4 << 20];
    loop {
        let read = file.read(&mut buf)?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
    }
    Ok(hex(&hasher.finalize()))
}

fn short_digest(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex(&hasher.finalize())[..16].to_string()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Short commit of the checkout this example was built from, `-dirty`
/// when the tree has uncommitted changes. `unknown` outside a checkout.
fn git_head() -> String {
    let dir = env!("CARGO_MANIFEST_DIR");
    let run = |args: &[&str]| -> Option<String> {
        let out = Command::new("git")
            .args(["-C", dir])
            .args(args)
            .output()
            .ok()?;
        out.status
            .success()
            .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
    };
    match run(&["rev-parse", "--short", "HEAD"]) {
        Some(head) if !head.is_empty() => {
            match run(&["status", "--porcelain"]).as_deref() {
                Some("") | None => head,
                Some(_) => format!("{head}-dirty"),
            }
        }
        _ => "unknown".to_string(),
    }
}

/// `YYYY-MM-DD HH:MM:SSZ`. Hand-rolled because the crate carries no date
/// dependency and one field does not justify adding one — the civil-from
/// -days algorithm is Howard Hinnant's, same as `chat_template.rs` uses
/// for `strftime_now`.
fn utc_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));

    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);

    format!(
        "{year:04}-{month:02}-{day:02} {:02}:{:02}:{:02}Z",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60,
    )
}

// ===========================================================================
// Model resolution
// ===========================================================================

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models")
}

/// Every probe-able GGUF in `models/`.
///
/// A `models/` directory is not a list of language models: it also holds
/// **multimodal projectors**, which are GGUFs with no vocabulary and no
/// chat template, and the same weights can appear twice under two names.
/// Both are excluded structurally rather than by guessing at filenames:
///
/// * A file is a projector iff some sibling model *claims* it under the
///   crate's own sidecar convention ([`mmproj_path`]) — the same
///   function `LlamaCppEngine::new` uses to auto-load vision.
/// * Two entries of identical size are the same weights (`models/
///   model.gguf` is a link to one of the others). Keep the descriptive
///   name; probing one model twice under two names would put a phantom
///   second data point in a capture.
///
/// Anything that slips through both is caught at load time — see the
/// skip-and-continue in [`main`].
///
/// [`mmproj_path`]: drama_llama::sidecar::mmproj_path
fn probeable_models() -> Vec<PathBuf> {
    let mut ggufs: Vec<PathBuf> = std::fs::read_dir(models_dir())
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|e| e == "gguf"))
        .collect();
    ggufs.sort();

    let projectors: HashSet<PathBuf> = ggufs
        .iter()
        .filter_map(|model| drama_llama::sidecar::mmproj_path(model))
        .collect();

    let mut by_size: BTreeMap<u64, PathBuf> = BTreeMap::new();
    for path in ggufs {
        if projectors.contains(&path) {
            continue;
        }
        let Ok(size) = path.metadata().map(|meta| meta.len()) else {
            continue;
        };
        match by_size.entry(size) {
            Entry::Vacant(slot) => {
                slot.insert(path);
            }
            Entry::Occupied(mut slot) => {
                if is_alias(slot.get()) && !is_alias(&path) {
                    slot.insert(path);
                }
            }
        }
    }

    let mut found: Vec<PathBuf> = by_size.into_values().collect();
    found.sort();
    found
}

/// The repo's generic model alias (`models/model.gguf`), which always
/// points at one of the named files.
fn is_alias(path: &Path) -> bool {
    path.file_name().is_some_and(|name| name == "model.gguf")
}

fn resolve_models(args: &Args) -> Result<Vec<PathBuf>, String> {
    if args.all {
        let all = probeable_models();
        return if all.is_empty() {
            Err(format!("no GGUF models in {}", models_dir().display()))
        } else {
            Ok(all)
        };
    }

    // An existing path wins outright, so an unusual filename is never
    // shadowed by the substring table.
    let as_path = PathBuf::from(&args.model);
    if as_path.is_file() {
        return Ok(vec![as_path]);
    }
    let in_models = models_dir().join(&args.model);
    if in_models.is_file() {
        return Ok(vec![in_models]);
    }

    // Match on the squashed form so the shorthand a human types lines up
    // with however the publisher punctuated the filename: `gptoss` finds
    // `gpt-oss-20b-…`, `q4kxl` finds `…-Q4_K_XL.gguf`.
    let needle = squash(&args.model);
    let matches: Vec<PathBuf> = probeable_models()
        .into_iter()
        .filter(|path| {
            path.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| squash(n).contains(&needle))
        })
        .collect();

    match matches.len() {
        1 => Ok(matches),
        0 => Err(format!(
            "no model matching {:?}. Available in {}:\n{}",
            args.model,
            models_dir().display(),
            list(&probeable_models()),
        )),
        _ => Err(format!(
            "{:?} matches more than one model:\n{}",
            args.model,
            list(&matches),
        )),
    }
}

fn list(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .filter_map(|p| p.file_name().and_then(|n| n.to_str()))
        .map(|n| format!("  {n}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Lowercase, alphanumerics only — punctuation in a model filename is
/// the publisher's business, not something a user should have to
/// reproduce.
fn squash(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

/// Render a piece for display: quoted, with the whitespace made visible
/// so `"Qwen"` and `" Qwen"` are told apart at a glance.
fn quoted(s: &str) -> String {
    format!("{:?}", s)
}
