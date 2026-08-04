//! Example: generate Agora `Soul` documents on a **base** model — the
//! completion-scaffold mode of [issue #88] (Phase 6 / rung 4b). Instruct
//! tunes are distillation-collapsed into Claude/GPT voice; a base model
//! still has the whole pretraining distribution, so new personalities come
//! out *different*. Division of labor: grammar supplies form, pretraining
//! supplies voice.
//!
//! The chat template is replaced by a sidecar
//! (`<model>.template.jinja`, copied from
//! `templates/completion-scaffold.jinja`) that renders the prompt as a
//! bare, never-closed JSON array of records — a scraped data file, not a
//! conversation. Exemplar souls ride in through
//! [`Prompt::add_examples`] (which also seeds the schema so the
//! constraint can't drift from the exemplars); their user turns render as
//! zero bytes. Each completion is grammar-locked to one `Soul` record, so
//! `--n` souls means `--n` completions — the loop feeds each result back
//! as a record, and every element of the array is schema-guaranteed.
//! Exact-count-via-`minItems` is deliberately NOT how this works: the
//! grammar compiler enforces `minItems` only as non-emptiness, because
//! forcing N items manufactures filler entries.
//!
//! ```sh
//! cargo run --example soul_forge --features "tokio,cli,json-schema" -- \
//!     --model models/Qwen3.5-35B-A3B-Base-Q8_0.gguf \
//!     path/to/SOUL.json another/SOUL.json --n 1
//! ```
//!
//! [issue #88]: https://github.com/mdegans/drama_llama/issues/88
//! [`Prompt::add_examples`]: misanthropic::Prompt::add_examples

mod utils;

use std::path::{Path, PathBuf};

use clap::Parser;
use drama_llama::OutputConfigOptions;
use misanthropic::{prompt::message::Role, Prompt, Transport};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// An agent's personality. Mirror of agora-agentkit's `Soul` *generation*
/// surface: `evolution_log` is deliberately absent — the seed runner
/// appends history entries; an agent (or a forge) never writes its own.
/// Field order is generation order — each field is context for the next.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Soul {
    /// Your name as it appears on Agora. Lowercase, hyphenated.
    name: String,
    /// Who you are, in your own voice. A few sentences. Identity is what
    /// makes you recognizably you across many cycles.
    identity: String,
    /// What you care about. Pithy bullets, one value per entry. 3-5 entries.
    values: Vec<String>,
    /// What you want to talk about and where.
    interests: Interests,
    /// How you write. A sentence or two. Tone, register, characteristic
    /// phrases.
    voice: String,
    /// Hard limits on your behavior. Optional - some agents run
    /// unconstrained.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    boundaries: Option<String>,
}

/// Communities an agent participates in plus freeform off-platform topics.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Interests {
    /// Slugs of Agora communities you participate in (e.g. `general`).
    /// Two or more.
    communities: Vec<String>,
    /// Other topics you care about, not tied to a specific community.
    #[serde(default)]
    topics: Vec<String>,
}

/// Generate new `Soul`s from exemplar SOUL.json files on a base model.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    #[command(flatten)]
    common: utils::CommonArgs,

    /// Exemplar SOUL.json files, one `Soul` object per file.
    #[arg(required = true)]
    souls: Vec<PathBuf>,

    /// How many new souls to generate.
    #[arg(short, long, default_value_t = 1)]
    n: usize,

    /// Also write each generated soul to `<dir>/<name>.json`
    /// (never overwrites; collisions get a numeric suffix).
    #[arg(long)]
    out_dir: Option<PathBuf>,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let cli = Cli::parse();
    utils::log_init(cli.common.verbose);
    let transport = cli
        .common
        .transport()
        // A base model has no trained habit of opening — or ever closing —
        // a `<think>` block; the default optional-thought limb in the
        // grammar is a trap there, not a feature. JSON from token 0.
        .output_config_opts(OutputConfigOptions {
            allow_thought: false,
            phase_split: false,
        })
        .build()?;

    let mut exemplars = Vec::new();
    for path in &cli.souls {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let soul: Soul = serde_json::from_str(&text)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        exemplars.push(soul);
    }

    // The empty user inputs are turn-order ballast: the completion
    // scaffold renders them as zero bytes, so the document stays a pure
    // record series while the prompt API still sees alternating turns.
    let mut prompt = cli
        .common
        .configure(Prompt::default())
        .add_examples(exemplars.into_iter().map(|soul| ("", soul)))?;

    for i in 0..cli.n {
        prompt = prompt.add_message((Role::User, ""))?;
        let response = transport.send(&prompt).await?;
        let soul: Soul = response.json()?;
        if cli.common.verbose {
            eprintln!("[{}/{}] usage: {:?}", i + 1, cli.n, response.usage);
        }

        // Feed the emission back verbatim (not a re-serialization): the
        // next render must reproduce the model's bytes exactly for the
        // prefix cache to survive, and the breakpoint marks the turn so
        // the hash path can bridge the grammar's non-canonical BPE
        // segmentation.
        prompt = prompt.add_message(response.inner.clone())?.cache();

        println!("{}", serde_json::to_string_pretty(&soul)?);
        if let Some(dir) = &cli.out_dir {
            let path = save(dir, &soul)?;
            eprintln!("[{}/{}] wrote {}", i + 1, cli.n, path.display());
        }
    }

    Ok(())
}

/// Write `soul` to `<dir>/<name>.json`, suffixing the stem rather than
/// overwriting anything already there.
fn save(dir: &Path, soul: &Soul) -> Result<PathBuf, std::io::Error> {
    std::fs::create_dir_all(dir)?;
    let mut path = dir.join(format!("{}.json", soul.name));
    let mut suffix = 2;
    while path.exists() {
        path = dir.join(format!("{}-{suffix}.json", soul.name));
        suffix += 1;
    }
    std::fs::write(&path, serde_json::to_vec_pretty(soul)?)?;
    Ok(path)
}
