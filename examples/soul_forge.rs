//! Example: generate Agora [`Soul`] documents on a **base** model — the
//! completion-scaffold mode of [issue #88] (Phase 6 / rung 4b). Instruct
//! tunes are distillation-collapsed into Claude/GPT voice; a base model
//! still has the whole pretraining distribution, so new personalities come
//! out *different*. Division of labor: grammar supplies form, pretraining
//! supplies voice.
//!
//! The `Soul` type is `agora-agentkit`'s own — the schema cannot drift
//! from what the seed runner deserializes. Exemplar `evolution_log`s are
//! cleared before prompting: history belongs to the agents that lived it,
//! and a fresh soul starts with none.
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
//! Sampling is deterministic per exemplar permutation whenever the
//! distribution collapses inside the typical cut, so identical reruns
//! produce identical souls. `--shuffle` permutes the exemplars for
//! run-to-run variety, and `--names` pins each record's `name` field via
//! a `const` schema patch — the grammar itself then guarantees the
//! continuation can't reuse an existing name (the name is the first
//! field generated, and everything downstream conditions on it).
//!
//! ```sh
//! cargo run --example soul_forge \
//!     --features "tokio,cli,json-schema,agora-agentkit" -- \
//!     --model models/Qwen3.5-35B-A3B-Base-Q8_0.gguf \
//!     path/to/SOUL.json another/SOUL.json --shuffle --names quill,ember
//! ```
//!
//! [issue #88]: https://github.com/mdegans/drama_llama/issues/88
//! [`Prompt::add_examples`]: misanthropic::Prompt::add_examples
//! [`Soul`]: agora_agentkit::reactor::seed::Soul

mod utils;

use std::path::{Path, PathBuf};

use agora_agentkit::reactor::seed::Soul;
use clap::Parser;
use drama_llama::OutputConfigOptions;
use misanthropic::{
    prompt::message::Role,
    prompt::output::{OutputConfig, OutputFormat},
    Prompt, Transport,
};
use rand::seq::SliceRandom;

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

    /// Generate one soul per name, with the `name` field grammar-pinned
    /// to it (`const` in the schema). Guarantees no name collisions with
    /// the exemplars — and since `name` is generated first, distinct
    /// names pull the rest of the record apart too.
    #[arg(long, conflicts_with = "n", value_delimiter = ',')]
    names: Vec<String>,

    /// Shuffle the exemplar order. Generation is deterministic per
    /// permutation whenever the distribution collapses inside the
    /// typical cut, so this is the cheap run-to-run diversity lever.
    #[arg(long)]
    shuffle: bool,

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
        let mut soul: Soul = serde_json::from_str(&text)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        // History belongs to the agent that lived it; a fresh soul
        // starts with none, and the exemplars should say so.
        soul.evolution_log.clear();
        exemplars.push(soul);
    }
    if cli.shuffle {
        exemplars.shuffle(&mut rand::rng());
    }

    // The schema `add_examples` seeds, after misanthropic's sanitizer —
    // the base for the per-name `const` patch below.
    let base_schema = match OutputConfig::for_type::<Soul>().format {
        Some(OutputFormat::JsonSchema(f)) => f.schema,
        _ => unreachable!("for_type always yields a JSON Schema format"),
    };

    // The empty user inputs are turn-order ballast: the completion
    // scaffold renders them as zero bytes, so the document stays a pure
    // record series while the prompt API still sees alternating turns.
    let mut prompt = cli
        .common
        .configure(Prompt::default())
        .add_examples(exemplars.into_iter().map(|soul| ("", soul)))?;

    let slots: Vec<Option<String>> = if cli.names.is_empty() {
        vec![None; cli.n]
    } else {
        cli.names.iter().cloned().map(Some).collect()
    };

    for (i, pinned) in slots.iter().enumerate() {
        if let Some(name) = pinned {
            let mut schema = base_schema.clone();
            schema["properties"]["name"] = serde_json::json!({
                "const": name,
            });
            prompt = prompt.json_schema(schema);
        }
        prompt = prompt.add_message((Role::User, ""))?;
        let response = transport.send(&prompt).await?;
        let soul: Soul = response.json()?;
        if cli.common.verbose {
            eprintln!(
                "[{}/{}] usage: {:?}",
                i + 1,
                slots.len(),
                response.usage
            );
        }
        for warning in soul.validate() {
            eprintln!("[{}/{}] {warning}", i + 1, slots.len());
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
            eprintln!("[{}/{}] wrote {}", i + 1, slots.len(), path.display());
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
