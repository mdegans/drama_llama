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
//! # no constraint: top-K at the answer position + what it actually says
//! cargo run --release --example whoami --features cli -- --unconstrained
//!
//! # the whole experiment: every model, every phrasing, both framings
//! cargo run --release --example whoami --features cli -- \
//!     --all --battery --framing both --seek 10
//! ```
//!
//! # Learned, or patched in?
//!
//! The interesting question is not *what* a model answers but *how
//! robustly*. An identity the model genuinely learned should survive
//! being asked sideways; one installed by a thin patch — a handful of
//! canned self-identification examples, a find-and-replace over a
//! fine-tuning set — should hold under the phrasing it was patched with
//! and wobble everywhere else. `--battery` varies the axes a patch is
//! least likely to have covered (register, sentence shape, language) and
//! `--framing raw` drops the chat template entirely, which is the
//! strongest version of the same test: no control tokens, no turn
//! structure, none of the scaffolding a patch was trained against.
//!
//! One honest limit, worth stating because it is easy to over-read: this
//! cannot distinguish an honestly-trained identity from a *thoroughly*
//! find-and-replaced one. Both produce a model that genuinely learned the
//! substituted name. What it can find is **residue** — places the
//! replacement missed — which is what `--focus` is for.
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
//! # Read Σ before you read anything else
//!
//! Σ is the total probability that the next tokens spell *any* candidate
//! — how much the model is actually trying to name itself right there.
//! `share` renormalises across the candidate set, so it always sums to
//! 100% and always looks confident, **including when Σ is 1e-9 and the
//! model was about to write "I am a large language model" instead.**
//! Rows below the threshold are marked; the marked ones are not results.
//!
//! `--seek N` is the fix rather than the warning. It follows the model's
//! own greedy continuation for up to `N` tokens and measures at whichever
//! position Σ peaks — so "hey what model are you" gets measured after the
//! model's own `"Hey! I am"` rather than on top of it. Maximising Σ is
//! what keeps this honest: we never choose the preamble, we follow the
//! one the model writes and measure when it arrives. Two Qwen rows go
//! from Σ=2e-6 (noise) to Σ=0.98 (a real answer) under `--seek 10`, with
//! no change to what was asked.
//!
//! If the model never arrives — Gemma asked in Spanish greedily writes
//! *"Soy un modelo de lenguaje entrenado por Google"* and names nothing —
//! Σ stays low and that is itself the finding.
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
//! In practice `--seek` mostly dissolves this too: a channel opener is
//! just a preamble, and the greedy walk steps through it like any other.
//! Qwen under `raw` framing emits its own `<think></think>` out of sheer
//! habit — no template asked it to — and the walk measures past it. The
//! `argmax` line still tells you where the answer position landed, and
//! the `↳ measured after` line tells you where the number came from.
//!
//! [`Engine`]: drama_llama::Engine
//! [`CandidatePredictor`]: drama_llama::CandidatePredictor
//! [`record_choice`]: drama_llama::CandidatePredictor::record_choice
//!
//! # Deliberately backend-concrete
//!
//! Unlike the `Transport`-driven examples, this one names `LlamaCpp*`
//! directly and takes no `--backend`. That is the point: it demonstrates
//! the raw [`Engine`] / [`CandidatePredictor`] layer, which sits *below*
//! the erasure boundary. Do not "upgrade" it onto `Session` or a
//! transport — the low-level examples were once ported away wholesale and
//! the crate was left with nothing showing this layer at all.

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

/// One labelled question in the battery.
struct Question {
    label: &'static str,
    text: &'static str,
}

/// The robustness battery.
///
/// A model that *learned* its identity should answer the same way however
/// you ask. A model whose identity was **patched in** — a find-and-replace
/// over a fine-tuning set, a handful of canned self-identification
/// examples — should hold up under the phrasing it was patched with and
/// wobble everywhere else. So the battery varies the axes a patch is
/// least likely to have covered: register, sentence shape, and language.
///
/// The point is the *variance across rows*, not any single row.
const BATTERY: &[Question] = &[
    Question {
        label: "plain",
        text: DEFAULT_QUESTION,
    },
    Question {
        label: "exact",
        text: "What specific model am I speaking to right now? I want you \
               to state your exact version.",
    },
    Question {
        label: "casual",
        text: "hey what model are you",
    },
    // Not a question at all: sidesteps the Q&A persona and asks the
    // language model to do the one thing it always does.
    Question {
        label: "complete",
        text: "Complete this sentence with only the name: \"The AI model \
               generating this text is called ___\"",
    },
    Question {
        label: "zh",
        text: "你是什么模型？只需要回答名字。",
    },
    Question {
        label: "es",
        text: "¿Qué modelo eres? Solo el nombre, por favor.",
    },
    Question {
        label: "fr",
        text: "Quel modèle es-tu ? Juste le nom, s'il te plaît.",
    },
];

/// How the question reaches the model.
#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
enum Framing {
    /// The model's own chat template — the distribution it was aligned on.
    Chat,
    /// `User: …\nAssistant:` and nothing else. Deliberately
    /// out-of-distribution: no control tokens, no turn structure, none of
    /// the scaffolding an identity patch would have been trained against.
    Raw,
    /// Both, in turn.
    Both,
}

impl Framing {
    fn expand(self) -> &'static [Framing] {
        match self {
            Framing::Chat => &[Framing::Chat],
            Framing::Raw => &[Framing::Raw],
            Framing::Both => &[Framing::Chat, Framing::Raw],
        }
    }

    fn label(self) -> &'static str {
        match self {
            Framing::Chat => "chat",
            Framing::Raw => "raw",
            Framing::Both => "both",
        }
    }
}

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

    /// The question to ask. Repeatable — every question runs against the
    /// same loaded model, which is the only affordable way to ask many.
    #[arg(short, long)]
    prompt: Vec<String>,

    /// Ask the built-in robustness battery instead: the same question
    /// across phrasings, registers and languages. See [`BATTERY`].
    #[arg(long)]
    battery: bool,

    /// How to frame the question: through the model's `chat` template,
    /// `raw` completion (`User: …\nAssistant:`), or `both`.
    #[arg(long, value_enum, default_value_t = Framing::Chat)]
    framing: Framing,

    /// Chat template override — raw Jinja, replacing both the embedded
    /// template and any `<model>.template.jinja` sidecar. For asking what
    /// a model says when its template stops telling it who it is.
    #[arg(long)]
    template: Option<PathBuf>,

    /// Let the model open its own answer: follow its greedy continuation
    /// up to this many tokens and score at whichever position it is most
    /// nearly naming itself (highest Σ).
    ///
    /// Without this, "¿Qué modelo eres?" measures the position where the
    /// model wants to write `Soy` — the name lands two tokens later and
    /// the row reports noise. Letting the model supply its own preamble
    /// keeps our thumb off the scale: we never choose the words, we only
    /// follow where it goes and measure when it arrives.
    #[arg(long, default_value_t = 0)]
    seek: usize,

    /// Always report this candidate's placing, even when it misses the
    /// top three. The residue question ("is there any trace of vendor X
    /// in here?") is answered by where X *places*, not by whether it
    /// wins, and a top-3 summary is exactly where that evidence hides.
    #[arg(long, default_value = "Claude")]
    focus: String,

    /// Force scoring of `" Name"` rather than `"Name"`. Rarely needed —
    /// the spelling is derived from where the prompt actually ends (see
    /// [`wants_leading_space`]); this is the escape hatch.
    #[arg(long)]
    leading_space: bool,

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

    let names: Vec<String> = if args.name.is_empty() {
        DEFAULT_NAMES.iter().map(|s| s.to_string()).collect()
    } else {
        args.name.clone()
    };
    let questions = resolve_questions(args);

    print_header(&engine, model_path, args, &questions)?;

    // Loading is the entire cost here — a 17 GiB model takes ~40 s to
    // come up and a scored question takes well under a second. So every
    // question and every framing runs against one load. This loop shape
    // is the difference between a battery being a thing you run and a
    // thing you talk about running.
    for &framing in args.framing.expand() {
        let renderer = Renderer::build(&engine, model_path, args, framing)?;
        println!("\n framing: {}", renderer.describe());
        let compact = questions.len() > 1 && !args.verbose;
        for question in &questions {
            ask(&mut engine, &renderer, question, &names, compact, args)?;
        }
    }

    Ok(())
}

/// Ask one question under one framing, and report.
fn ask(
    engine: &mut LlamaCppEngine,
    renderer: &Renderer,
    question: &Question,
    names: &[String],
    compact: bool,
    args: &Args,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut rendered =
        renderer.render(question.text, args.system.as_deref())?;
    rendered.push_str(&args.prefix);

    // `special = true`: a chat render is full of control tokens and they
    // must resolve to their single ids, not to their spellings. Same
    // call `Session` makes. Harmless under raw framing, which contains
    // no control tokens to parse.
    let prompt_tokens = engine.model().tokenize(&rendered, true);

    println!("\n  ── {} · {}", question.label, quoted(question.text));

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
            "     ⚠ the prompt itself names {} — the template or --system \
             told the model\n       what it is. Those rows measure \
             instruction-following, not recall.",
            leaked.join(", "),
        );
    }

    if args.verbose {
        println!("\n--- rendered prompt ---\n{rendered}\n--- end ---");
    }

    // One free-standing peek at the answer position, shared by both
    // modes: the sanity line in scoring mode, the main event under
    // `--unconstrained`.
    let k = if args.unconstrained { args.top_k } else { 3 };
    let peeked = peek(engine, &prompt_tokens, k.max(1));

    println!(
        "     argmax {} p={:.4}",
        quoted(&peeked[0].piece),
        peeked[0].p,
    );

    // The trap this example is most likely to be quoted through. If the
    // model's own next move is to open a structured channel — Harmony's
    // `<|channel|>`, an un-suppressed `<think>` — then the answer
    // position is not where the answer goes, and a score taken there is
    // conditioned on a continuation the model would never have written.
    // Exact test, not a heuristic: ask the vocabulary.
    //
    // `--seek` mostly dissolves this on its own: the greedy walk steps
    // through the channel opener like any other preamble and measures
    // past it. So the warning stays factual about the *position* and
    // stops short of condemning the numbers — the `↳ measured after`
    // line says whether the walk got out.
    if engine.model().special_tokens().contains(&peeked[0].id) {
        let advice = if args.seek == 0 {
            "Scores below are counterfactual until --seek or\n       \
             --prefix moves the probe inside it."
        } else {
            "--seek should step past it; check the\n       \
             ↳ line below for where this was actually measured."
        };
        println!(
            "     ⚠ that is a control token — the model opens a structured \
             channel here,\n       not an answer. {advice}"
        );
    }

    if args.unconstrained {
        unconstrained(engine, &prompt_tokens, &peeked, args);
    } else {
        let leading_space = args.leading_space
            || wants_leading_space(engine, &rendered, &prompt_tokens);
        scored(
            engine,
            &prompt_tokens,
            &rendered,
            names,
            leading_space,
            compact,
            args,
        );
    }

    Ok(())
}

/// Should candidates be scored as `" Name"` rather than `"Name"`?
///
/// Byte-level BPE carries a word's preceding space on the *front* of its
/// token, so the right spelling depends entirely on what the prompt ends
/// with — and getting it wrong deflates every score by orders of
/// magnitude while leaving the ranking intact, which is the worst kind of
/// wrong: invisible in the shape of the answer, fatal to the numbers.
///
/// Two cases mean the space is already accounted for:
///
/// * The prompt ends in whitespace — a chat template's `…assistant\n`.
/// * The prompt ends in a **control token** — Harmony's `<|message|>`
///   runs straight into content. Checked against the vocabulary rather
///   than by sniffing for angle brackets.
///
/// Everything else (`Assistant:`, `--prefix "My name is"`) is mid-line
/// text, and the next token carries the space. Deriving this rather than
/// flagging it is what makes chat and raw framings comparable at all.
fn wants_leading_space(
    engine: &LlamaCppEngine,
    rendered: &str,
    prompt_tokens: &[Token],
) -> bool {
    if rendered.chars().next_back().is_none_or(char::is_whitespace) {
        return false;
    }
    match prompt_tokens.last() {
        Some(last) => !engine.model().special_tokens().contains(last),
        None => false,
    }
}

/// `--prompt` (repeatable) wins; then `--battery`; then one default.
fn resolve_questions(args: &Args) -> Vec<Question> {
    if !args.prompt.is_empty() {
        return args
            .prompt
            .iter()
            .enumerate()
            .map(|(i, text)| Question {
                // Leaked deliberately: `Question` holds `&'static str` so
                // the battery can be a const, and an example that asks a
                // handful of questions and exits does not need to pay
                // lifetime plumbing for it.
                label: Box::leak(format!("q{i}").into_boxed_str()),
                text: Box::leak(text.clone().into_boxed_str()),
            })
            .collect();
    }
    if args.battery {
        return BATTERY
            .iter()
            .map(|q| Question {
                label: q.label,
                text: q.text,
            })
            .collect();
    }
    vec![Question {
        label: "plain",
        text: DEFAULT_QUESTION,
    }]
}

// ===========================================================================
// Framing
// ===========================================================================

/// How a question becomes prompt text.
enum Renderer {
    Chat {
        template: ChatTemplate,
        origin: String,
        digest: String,
    },
    Raw,
}

impl Renderer {
    fn build(
        engine: &LlamaCppEngine,
        model_path: &Path,
        args: &Args,
        framing: Framing,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if framing == Framing::Raw {
            return Ok(Renderer::Raw);
        }
        let (template, source) = load_template(engine, model_path, args)?;
        Ok(Renderer::Chat {
            template,
            origin: source.origin,
            digest: source.digest,
        })
    }

    fn describe(&self) -> String {
        match self {
            Renderer::Chat { origin, digest, .. } => {
                format!("chat · {origin} · sha256 {digest}")
            }
            Renderer::Raw => {
                "raw · \"User: …\\nAssistant:\" (out of distribution)".into()
            }
        }
    }

    fn render(
        &self,
        question: &str,
        system: Option<&str>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        match self {
            Renderer::Chat { template, .. } => {
                let mut prompt = Prompt::default()
                    .messages([(Role::User, question.to_string())])?;
                if let Some(system) = system {
                    prompt = prompt.system(system.to_string());
                }
                // `thinking` is left unset, so `enable_thinking` renders
                // false and a Qwen-style template closes an empty
                // reasoning block for us.
                let opts =
                    RenderOptions::default().with_generation_prompt(true);
                Ok(template.render_with(&prompt, &opts)?)
            }
            // No trailing space after `Assistant:`. A dangling space is
            // its own tokenization artifact — BPE wants to carry it on
            // the front of the following token — which is why raw framing
            // scores `" Name"` (see `--leading-space`).
            Renderer::Raw => {
                let mut out = String::new();
                if let Some(system) = system {
                    out.push_str(system);
                    out.push_str("\n\n");
                }
                out.push_str("User: ");
                out.push_str(question);
                out.push_str("\nAssistant:");
                Ok(out)
            }
        }
    }
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
    args: &Args,
) -> Result<(ChatTemplate, TemplateSource), Box<dyn std::error::Error>> {
    let model = engine.model();
    // `--template` outranks both, so a template can be edited and the
    // edit's effect measured — strip gpt-oss's "You are ChatGPT" line and
    // ask again, for instance.
    let (sidecar_path, explicit) = match &args.template {
        Some(path) => (path.clone(), true),
        None => (model_path.with_extension("template.jinja"), false),
    };

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
        let kind = if explicit { "--template" } else { "sidecar" };
        return Ok((
            template,
            TemplateSource {
                origin: format!("{kind} ({name})"),
                digest,
            },
        ));
    }
    if explicit {
        return Err(format!("--template {sidecar_path:?} not found").into());
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

/// Every candidate scored at one position, best first, plus Σ.
struct Measurement {
    scores: Vec<Score>,
    total: f64,
    /// The model's own words between the answer position and where this
    /// was measured. Empty at offset 0.
    opener: String,
}

fn measure(
    engine: &mut LlamaCppEngine,
    tokens: &[Token],
    names: &[String],
    leading_space: bool,
) -> Measurement {
    let mut scores: Vec<Score> = names
        .iter()
        .filter_map(|name| {
            let score = score_name(engine, tokens, name, leading_space);
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
    let total = scores.iter().map(|s| s.joint).sum();
    Measurement {
        scores,
        total,
        opener: String::new(),
    }
}

/// Follow the model's greedy continuation and measure at whichever of its
/// first `seek` positions it is most nearly naming itself.
///
/// Picking by Σ is the whole trick. Σ is exactly "how much probability
/// sits on *some* candidate name here", so maximising it walks to where
/// the answer actually lands — without anyone deciding in advance what
/// the preamble should be. The model writes `Soy un modelo de lenguaje
/// llamado`; we just follow it there. Choosing that opener by hand would
/// be putting a thumb on the scale; following it is not.
fn seek_best(
    engine: &mut LlamaCppEngine,
    prompt_tokens: &[Token],
    rendered: &str,
    names: &[String],
    seek: usize,
    args: &Args,
) -> Measurement {
    let mut options = PredictOptions::greedy().add_model_stops(engine.model());
    options.n = NonZeroUsize::new(seek).unwrap();
    let path: Vec<Token> = engine
        .predict_tokens(prompt_tokens.to_vec(), options, None)
        .collect();

    let mut walk: Vec<Measurement> = Vec::with_capacity(path.len() + 1);
    for offset in 0..=path.len() {
        let opener = engine
            .model()
            .tokens_to_string(path[..offset].iter().copied());
        let mut tokens = prompt_tokens.to_vec();
        tokens.extend_from_slice(&path[..offset]);
        let text = format!("{rendered}{opener}");
        let leading_space =
            args.leading_space || wants_leading_space(engine, &text, &tokens);

        let mut m = measure(engine, &tokens, names, leading_space);
        m.opener = opener;
        walk.push(m);
    }

    // The *earliest* position that gets within a whisker of the best —
    // not the best outright. Taking the max walks past the model's own
    // answer whenever it names itself early and then repeats: Qwen under
    // raw framing writes `<think></think>Qwen`, and measuring after that
    // scores `P(Qwen | …Qwen…)`, which is repetition wearing recall's
    // clothes. First arrival is the honest one, and the tolerance stops a
    // rounding difference from pushing us past it.
    let peak = walk.iter().map(|m| m.total).fold(0.0f64, f64::max);
    walk.into_iter()
        .find(|m| m.total >= 0.9 * peak)
        .expect("the peak is one of the measurements")
}

fn scored(
    engine: &mut LlamaCppEngine,
    prompt_tokens: &[Token],
    rendered: &str,
    names: &[String],
    leading_space: bool,
    compact: bool,
    args: &Args,
) {
    let Measurement {
        scores,
        total,
        opener,
    } = match args.seek {
        0 => measure(engine, prompt_tokens, names, leading_space),
        seek => seek_best(engine, prompt_tokens, rendered, names, seek, args),
    };
    if !opener.is_empty() {
        println!("     ↳ measured after the model's own {}", quoted(&opener));
    }
    let share = |joint: f64| {
        if total > 0.0 {
            100.0 * joint / total
        } else {
            0.0
        }
    };

    // A battery is read down the column, not across the row: what matters
    // is whether the winner *changes* between questions, so each question
    // gets one line and the full table stays behind `--verbose`.
    if compact {
        let top: Vec<String> = scores
            .iter()
            .take(3)
            .map(|s| format!("{} {:.2}", s.name, share(s.joint)))
            .collect();
        let focus = scores
            .iter()
            .position(|s| s.name.eq_ignore_ascii_case(&args.focus))
            .map(|i| {
                format!(
                    "   │ {} #{} {:.2}",
                    scores[i].name,
                    i + 1,
                    share(scores[i].joint)
                )
            })
            .unwrap_or_default();
        println!(
            "     {}{}   Σ={:.3e}{}",
            top.join(" · "),
            focus,
            total,
            sigma_note(total)
        );
        return;
    }

    println!();
    println!(
        "  {:<12} {:>3}  {:>11}  {:>7}  ranks",
        "name", "tok", "P(name)", "share"
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
            share(score.joint),
            ranks.join(","),
        );
    }
    println!("  {}", "─".repeat(58));
    println!(
        "  {:<12} {:>3}  {:>11}   (mass on everything else: {:.2}%){}",
        "Σ",
        "",
        format!("{total:.3e}"),
        100.0 * (1.0 - total).max(0.0),
        sigma_note(total),
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

/// Σ is the load-bearing number and the easiest one to skip past.
///
/// It is the total probability that the very next tokens spell *any*
/// candidate — i.e. how much the model is actually trying to name itself
/// right here. When it is high, `share` is a real answer. When it is
/// ~1e-5 the model is writing a sentence ("You are speaking to…", "Hey!
/// I'm…") and `share` has renormalised a rounding error into a
/// confident-looking percentage. Both are worth seeing; conflating them
/// is how a table lies.
///
/// The threshold is deliberately loose. This is a "stop and look" mark,
/// not a verdict.
fn sigma_note(total: f64) -> &'static str {
    if total < 0.01 {
        "  ⚠ not naming itself here — share is renormalised noise"
    } else {
        ""
    }
}

/// `P(name | prompt)` by forcing the name's tokens one at a time.
///
/// Returns `None` for a name that tokenizes to nothing.
fn score_name(
    engine: &mut LlamaCppEngine,
    prompt_tokens: &[Token],
    name: &str,
    leading_space: bool,
) -> Option<Score> {
    // `tokenize_special(spelling, false, false)`: no automatic BOS (we
    // are continuing a prompt, not starting one) and no special-token
    // parsing (a name is content, and content never spells a control
    // token).
    //
    // `leading_space` picks which spelling is scored. It matters more
    // than it looks: BPE carries a word's preceding space on the front of
    // its token, so after a chat template's assistant header `"Claude"`
    // is right and `" Claude"` is near-impossible, while after
    // `"Assistant:"` it is the other way round. Getting it backwards
    // deflates every candidate uniformly — the ranking survives, the
    // probabilities do not. Uniform across candidates either way, so a
    // run is internally comparable; only cross-framing comparisons need
    // this to be right.
    let model = engine.model();
    let spelling = if leading_space {
        format!(" {name}")
    } else {
        name.to_string()
    };
    let tokens = model.tokenize_special(&spelling, false, false);
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
    questions: &[Question],
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
    field(
        "system prompt",
        &match &args.system {
            Some(s) => quoted(s),
            None => "(none)".into(),
        },
    );
    field(
        "prefix",
        &if args.prefix.is_empty() {
            "(none)".into()
        } else {
            quoted(&args.prefix)
        },
    );
    field(
        "questions",
        &format!(
            "{} ({})",
            questions.len(),
            questions
                .iter()
                .map(|q| q.label)
                .collect::<Vec<_>>()
                .join(", "),
        ),
    );
    field("framing", args.framing.label());
    field(
        "seek",
        &match args.seek {
            0 => "0 (answer position only)".to_string(),
            n => format!("{n} greedy tokens, measured at max Σ"),
        },
    );
    field(
        "candidates",
        &format!(
            "{} (focus: {})",
            if args.name.is_empty() {
                DEFAULT_NAMES.len()
            } else {
                args.name.len()
            },
            args.focus,
        ),
    );
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
