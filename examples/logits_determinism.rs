//! **Are two prefills of the same prompt the same?** A raw determinism
//! probe for the llama.cpp backend, below `Session`, `Engine` and every
//! predictor: a `LlamaCppModel`, bare `LlamaCppDecoder` contexts, and
//! the [`Decoder`] trait calls `Session` itself makes (`prefill` per
//! chunk, `step` per generated token, `checkpoint_pos` / `restore_to`
//! for the prefix cache). No chat template, no sampler: a fixed,
//! seeded ~3k-token text prompt, final-position logits compared **bit
//! for bit**, and a greedy (argmax, lowest id on ties) continuation.
//!
//! Motivation: misanthropic's warm-vs-cold equivalence check against
//! blallama (greedy, `--no-penalty`) saw replies diverge late in fluent
//! text — including a request where *both* sides were cold prefills. If
//! this probe shows the same thing, `Session` is ruled out and the
//! difference lives in llama.cpp (or the GPU kernels) or in how a
//! context is set up and reused. It does **not** rule out drama_llama's
//! thin decode wrapper: every call here still goes through
//! `LlamaCppDecoder::prefill_inherent` / `Batch` on its way to
//! `llama_decode`.
//!
//! ```sh
//! # Metal (all layers offloaded), every experiment
//! cargo run --release --example logits_determinism --features cli -- \
//!     models/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf --dump metal.json
//!
//! # the same on the CPU, then compared against the Metal run
//! cargo run --release --example logits_determinism --features cli -- \
//!     models/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf --n-gpu-layers 0 \
//!     --compare metal.json
//! ```
//!
//! `--n-gpu-layers 0` also turns llama.cpp's `op_offload` off. It
//! defaults on, and with it the scheduler still hands large batched
//! `MUL_MAT` / `MUL_MAT_ID` on CPU-resident weights to Metal — a "CPU"
//! run would be half GPU. `--no-op-offload` turns it off with layers
//! offloaded too. The header prints what was used.
//!
//! # Experiments
//!
//! Every row compares one variant against a baseline and reports whether
//! the final logits are bit-identical, how many differ and by how much,
//! whether the argmax flipped (and how close the baseline's top two
//! were), where the 64-token greedy continuation first diverges, and the
//! first continuation step whose logits differ while the tokens still
//! agree. After every rewind, clear or removal the probe asserts the
//! sequence's `memory_seq_pos_max`, so a wrong KV extent fails loudly
//! instead of reading as a numerical difference.
//!
//! - **repeat** — the same prompt, same schedule, in fresh contexts
//!   (`--repeats`), plus the same prompt prefilled twice in one context
//!   and in a context that first held a longer unrelated prompt, then
//!   `memory_seq_rm(seq, -1, -1)` (what `Session` does on a cache miss)
//!   or `memory_clear`. Expect bit-identical. A fresh-context difference
//!   is nondeterminism in llama.cpp / the backend kernels; a
//!   reused-context-only difference means leftover KV state (cell
//!   placement, `n_kv` padding, stale recurrent state) leaks into
//!   results.
//! - **chunking** — the same prompt split differently across decode
//!   calls: one batch, chunks of 512 (and `--chunk`), a breakpoint split
//!   (instruction prefix, then the rest), three chunks, and the last 64
//!   tokens one at a time. llama.cpp's kernels are not batch-invariant,
//!   so small differences here are *expected*; what matters is their
//!   size and whether they flip a greedy choice.
//!
//!   The breakpoint rows are **not** the warm/cold difference on a
//!   breakpoint hit: `Session`'s cold walk prefills chunk by chunk at the
//!   same breakpoints its warm path restores to, so both sides run one
//!   schedule — that case is the **snapshot** experiment. They show
//!   what a *different* set of breakpoints costs (a moved
//!   `cache_control` marker, or a request with none against one with
//!   some). The schedule does differ on **tip** reuse: the warm side
//!   decoded the previous reply one token at a time (`step`), the cold
//!   side prefills those same tokens in a batch. The "last 64
//!   token-by-token" row models that; the **tip** experiment measures
//!   it directly.
//! - **align** — does a breakpoint split cost anything once the ubatch
//!   grid is restored? llama.cpp slices each decode call into ubatches of
//!   `n_ubatch` from the call's first token, so after `[517 | rest]`
//!   every later ubatch is shifted. The rows split at the breakpoint and
//!   then realign (`[517 | 507 | 512 …]`, boundaries on multiples of
//!   `n_ubatch` from position 0), split on the grid itself (`[512 |
//!   rest]`), and add a short extra chunk before realigning (`[517 | 7 |
//!   500 | 512 …]`), against the unaligned `[517 | rest]` as the control.
//!   If realigned rows match the one-batch reference, blallama could make
//!   breakpoint chunking schedule-invariant by realigning after each
//!   breakpoint. Assumes `n_batch` is a multiple of `n_ubatch`, as at
//!   the defaults.
//! - **snapshot** — the prefix-cache path: prefill the prefix, take the
//!   checkpoint, prefill the suffix; then `restore_to` the prefix in the
//!   same context and prefill the suffix again (twice, and once after a
//!   different suffix in between), and load the raw `get_state_seq`
//!   bytes into a fresh context. All against a fresh prefill with the
//!   *same* two-chunk schedule, so only the restore can differ.
//!   `restore_to` tries a KV truncate first: hybrid and recurrent models
//!   (Qwen3.6-A3B) refuse it and reload the snapshot; pure attention
//!   models truncate, so there only the `get_state_seq` row exercises a
//!   state load. `--force-snapshots` makes `checkpoint_pos` store bytes
//!   on a pure attention model but does not change that — `restore_to`
//!   still truncates first. Expect bit-identical; a difference is a
//!   lossy restore.
//! - **tip** — the previous reply reused as the next request's prefix.
//!   The baseline (a) is the warm side: prefill the prompt P, greedily
//!   generate `--tip-tokens` with `step` and feed every one back (the
//!   tip), `checkpoint_pos` there, then prefill a short new turn S and
//!   continue 32 tokens. (b) replays that exactly in a fresh context — P
//!   with the same schedule, the same tip tokens one `step` at a time,
//!   then S — and must be bit-identical; a separate line compares its
//!   prefill and replayed steps with (a)'s. (d) `restore_to` the tip in
//!   (a)'s context, after (a)'s own turn and after another branch, then
//!   S — must be bit-identical too; the probe prints
//!   `memory_seq_pos_max` after each restore (e), a record rather than a
//!   check: `rewind` already asserts that extent, and on a hybrid model
//!   `memory_seq_pos_max` is the minimum of the attention and recurrent
//!   memories' extents, so it can't vouch for each separately. (c) is
//!   blallama's cold path, the tip batch-prefilled with P (`[P+tip | S]`
//!   and one batch `[P+tip+S]`): expected to differ by schedule alone,
//!   and the size of that gap is the reading. The verdict is "tip reuse
//!   lossless under matched schedule" iff (b) and every (d) row match
//!   (a), and it calls a warm/cold gap the schedule only when a (c) row
//!   actually differs. Its scope is drama_llama's decoder: restoring a
//!   stepped tip. blallama's tip matching (the history re-rendered
//!   against the generated tokens, a stop sequence's trim, #91) isn't
//!   exercised here; misanthropic's equivalence check covers it with a
//!   matched-schedule control.
//! - **neighbors** — a two-slot unified KV cache, as blallama gets with
//!   `--cache-slots`: the prompt on seq 0 in an empty cache against the
//!   prompt on seq 1 beside a longer unrelated sequence, on seq 1 alone,
//!   and on seq 0 after that sequence was removed; plus the two-slot
//!   baseline against the one-slot reference (the context shape alone).
//!   A difference means results depend on what else the cache holds.
//! - **kv-view** — the neighbor effect, characterized: the prompt on
//!   seq 0 of a two-slot unified cache while seq 1 holds K tokens. Each
//!   ubatch attends over `n_kv` cells — the highest occupied cell,
//!   padded to 256 — masked per sequence. "Placed after" rows fill seq 1
//!   first, so the prompt lands K cells later *and* `n_kv` grows; "freed
//!   spacer" rows put the prompt in the baseline's own cells (a spacer
//!   of `held` tokens on seq 0, seq 1 after it, then the spacer removed)
//!   so only `n_kv` changes. `held` is what seq 0 ends up holding: the
//!   prompt and the continuation tokens fed back.
//!
//!   Labels give the **final prompt ubatch's** `n_kv` only. The
//!   baseline's and the placed-after rows' `n_kv` grows as the prompt
//!   fills (narrower on earlier ubatches, wider on continuation steps);
//!   a spacer row's is one width throughout, since seq 1 holds the top
//!   cells. So a spacer row differing from the baseline mixes the wider
//!   view of the early ubatches with the final width. The threshold is
//!   the spacer **pair**, always run: K = `pad256(held) − held` keeps
//!   every width at `pad256(held)` (the baseline's final width); K + 1
//!   crosses into the next 256-cell block. The two differ in nothing
//!   but that block of masked cells, and are compared with each other
//!   directly. `--kv-view` (default 0, 256, 1024, 4096) adds
//!   placed-after rows and more spacer rows; to bracket another
//!   boundary pass `m·256 − held` and one more.
//!
//! `--dump` writes the reference run (fresh context, one batch) and the
//! settings that shaped it to JSON; `--compare` checks this run's
//! reference against such a file. Across two processes on one device
//! that is a process-level repeat; with `--n-gpu-layers 0` on one side
//! it measures Metal against the CPU (expect differences there — the
//! useful question is how large). A different model file or prompt is
//! refused; different `n_ctx` / `n_batch` / `n_ubatch` is refused unless
//! `--compare-anyway`; device settings (`n_gpu_layers`, `op_offload`,
//! flash attention, threads) are the axis being compared and are shown
//! side by side.
//!
//! Loads the model once and creates a context per variant, so it needs
//! the model's weights plus one context's KV (two for `neighbors` and
//! `kv-view`) — don't run it beside a server holding the same weights on
//! a machine without room for both.

use std::{
    collections::BTreeSet,
    error::Error,
    path::{Path, PathBuf},
    time::Instant,
};

use clap::{Parser, ValueEnum};
use drama_llama::{
    gpu_device_names, silence_logs, Decoder, FlashAttention, LlamaCppDecoder,
    LlamaCppModel, LlamaCppOptions, Token,
};
use serde::{Deserialize, Serialize};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

/// Tokens decoded one at a time at the end of the tail-by-tail schedule.
const TAIL: usize = 64;
/// Tokens generated on an unrelated prompt to leave decode cells behind.
const JUNK_GEN: usize = 16;
/// The extra short chunk in the align experiment's `[split | 7 | …]`.
const ODD_CHUNK: usize = 7;
/// llama.cpp pads `n_kv` to a multiple of this (at least).
const KV_PAD: usize = 256;
/// Greedy continuation after the tip experiment's new turn.
const TIP_CONTINUE: usize = 32;
/// The tip experiment's new user turn, prefilled after the tip.
const TIP_TURN: &str = "\n\nQuestion: Which entries mention the dome \
shutter? List their numbers and times.\nAnswer:";
/// Another turn from the tip: the branch the warm rows restore past.
const TIP_OTHER_TURN: &str = "\n\nQuestion: Who worked in the basement \
most often?\nAnswer:";

#[derive(Parser)]
#[command(about = "Bitwise prefill-determinism probe below Session")]
struct Args {
    /// GGUF model to load.
    model: PathBuf,
    /// KV context size in tokens (blallama's default).
    #[arg(long, default_value_t = 32768)]
    n_ctx: u32,
    /// Max tokens per decode call (llama.cpp `n_batch`). Defaults to
    /// `--n-ctx`, as blallama loads it.
    #[arg(long)]
    n_batch: Option<u32>,
    /// Micro-batch size (llama.cpp `n_ubatch`). Defaults to llama.cpp's
    /// (512), as blallama loads it.
    #[arg(long)]
    n_ubatch: Option<u32>,
    /// Threads for generation and batch processing. Defaults to every
    /// logical core.
    #[arg(long)]
    threads: Option<i32>,
    /// Layers to offload to the GPU; -1 is all of them, 0 runs on the
    /// CPU alone (and implies `--no-op-offload`).
    #[arg(long, default_value_t = -1, allow_negative_numbers = true)]
    n_gpu_layers: i32,
    /// Turn off llama.cpp's `op_offload`, which otherwise runs large
    /// batched ops on CPU-resident weights on the GPU anyway.
    #[arg(long)]
    no_op_offload: bool,
    /// Flash Attention policy. Defaults to llama.cpp's (auto).
    #[arg(long, value_enum)]
    flash_attn: Option<FlashAttention>,
    /// Seed for the generated prompt text. The same seed and length
    /// always give the same tokens.
    #[arg(long, default_value_t = 1337)]
    seed: u64,
    /// Target prompt length in tokens.
    #[arg(long, default_value_t = 3000)]
    prompt_tokens: usize,
    /// Breakpoint position (tokens) for the split schedules and the
    /// snapshot. Defaults to the end of the instruction prefix.
    #[arg(long)]
    split: Option<usize>,
    /// Fresh-context repeats in the repeat experiment, counting the
    /// reference run.
    #[arg(long, default_value_t = 3)]
    repeats: usize,
    /// Greedy continuation length.
    #[arg(long, default_value_t = 64)]
    continue_tokens: usize,
    /// Extra chunk sizes for the chunking experiment (comma-separated).
    #[arg(long, value_delimiter = ',')]
    chunk: Vec<usize>,
    /// Tokens the tip experiment generates as the previous reply.
    #[arg(long, default_value_t = 64)]
    tip_tokens: usize,
    /// Tokens seq 1 holds in the kv-view experiment (comma-separated):
    /// placed-after rows, and spacer rows beside the computed pad pair.
    #[arg(
        long,
        value_delimiter = ',',
        default_values_t = [0usize, 256, 1024, 4096]
    )]
    kv_view: Vec<usize>,
    /// Experiments to run (comma-separated).
    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_values = [
            "repeat", "chunking", "align", "snapshot", "tip", "neighbors",
            "kv-view"
        ]
    )]
    experiments: Vec<Experiment>,
    /// Make `checkpoint_pos` store sequence snapshots on a pure-attention
    /// model too. `restore_to` still truncates the KV first there, so
    /// this does not route the warm snapshot rows through a load.
    #[arg(long)]
    force_snapshots: bool,
    /// Write the reference run and its settings to this JSON file.
    #[arg(long)]
    dump: Option<PathBuf>,
    /// Compare the reference run against a file from `--dump`.
    #[arg(long)]
    compare: Option<PathBuf>,
    /// Compare against `--compare` even when `n_ctx`, `n_batch` or
    /// `n_ubatch` differ from the file's.
    #[arg(long)]
    compare_anyway: bool,
    /// Keep llama.cpp's own logging.
    #[arg(long)]
    verbose: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Experiment {
    Repeat,
    Chunking,
    Align,
    Snapshot,
    Tip,
    Neighbors,
    KvView,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if !args.verbose {
        silence_logs();
    }

    let mut model_params = LlamaCppOptions::default().model_params();
    model_params.n_gpu_layers = args.n_gpu_layers;
    let model =
        LlamaCppModel::from_file(args.model.clone(), Some(model_params))
            .ok_or_else(|| {
                format!("could not load {}", args.model.display())
            })?;

    let mut options = LlamaCppOptions::default().with_n_ctx(args.n_ctx);
    if let Some(n_ubatch) = args.n_ubatch {
        options = options.with_n_ubatch(n_ubatch);
    }
    if let Some(fa) = args.flash_attn {
        options = options.with_flash_attention(fa);
    }
    let rig = Rig::new(model, options, &args)?;

    let prompt = Prompt::build(&rig.model, args.seed, args.prompt_tokens);
    let junk = Prompt::build(
        &rig.model,
        args.seed ^ 0x5eed_0fd1_ffe2_e7a1,
        args.prompt_tokens + 512,
    );
    let split = args.split.unwrap_or(prompt.split);
    let len = prompt.tokens.len();
    if split == 0 || split >= len {
        return Err(format!("--split {split} must be in 1..{len}").into());
    }
    let need = junk.tokens.len() + JUNK_GEN + len + rig.continue_tokens;
    if args.experiments.contains(&Experiment::Neighbors)
        && need > rig.n_ctx as usize
    {
        return Err(format!(
            "n_ctx {} is too small: the neighbors experiment holds {need} \
             tokens at once",
            rig.n_ctx
        )
        .into());
    }

    rig.header(&args, &prompt, split, junk.tokens.len());

    let started = Instant::now();
    let reference = rig.run(&prompt.tokens, &rig.whole(len), 0, None)?;
    println!(
        "reference continuation ({} tokens, fresh context, one batch):\n  \
         {:?}\n",
        reference.tokens.len(),
        rig.model.tokens_to_string(reference.tokens.iter().copied()),
    );

    let config = rig.config(&args, &prompt, split)?;
    if let Some(path) = &args.dump {
        Dump::new(&config, &prompt, &reference).write(path)?;
        println!("wrote the reference run to {}\n", path.display());
    }
    if let Some(path) = &args.compare {
        compare_file(&rig, &config, path, &prompt, &reference, &args)?;
    }

    for experiment in &args.experiments {
        match experiment {
            Experiment::Repeat => {
                repeat(&rig, &prompt, &junk, &reference, args.repeats)?
            }
            Experiment::Chunking => {
                chunking(&rig, &prompt, split, &reference, &args.chunk)?
            }
            Experiment::Align => align(&rig, &prompt, split, &reference)?,
            Experiment::Snapshot => snapshot(&rig, &prompt, &junk, split)?,
            Experiment::Tip => tip(&rig, &prompt, args.tip_tokens)?,
            Experiment::Neighbors => {
                neighbors(&rig, &prompt, &junk, &reference)?
            }
            Experiment::KvView => kv_view(&rig, &prompt, &args)?,
        }
    }
    println!("done in {:.1}s", started.elapsed().as_secs_f64());
    Ok(())
}

// ---------------------------------------------------------------------
// Contexts, prefill, greedy decoding
// ---------------------------------------------------------------------

/// The loaded model and how to build a context over it.
struct Rig {
    model: LlamaCppModel,
    options: LlamaCppOptions,
    /// Max tokens per decode call, as the context resolved it.
    n_batch: u32,
    /// Micro-batch size, as llama.cpp resolves it (`min(n_ubatch,
    /// n_batch)`).
    n_ubatch: u32,
    /// KV context size, as the context resolved it.
    n_ctx: u32,
    threads: Option<i32>,
    /// llama.cpp's `op_offload`: whether the scheduler may run ops on
    /// CPU-resident weights on the GPU.
    op_offload: bool,
    force_snapshots: bool,
    /// Whether the model takes sequence checkpoints without forcing — a
    /// sliding-window, hybrid or recurrent model, which a KV truncate
    /// alone cannot always rewind.
    native_snapshots: bool,
    continue_tokens: usize,
}

impl Rig {
    /// Resolve the context settings from a probe context.
    fn new(
        model: LlamaCppModel,
        options: LlamaCppOptions,
        args: &Args,
    ) -> Result<Self> {
        let mut rig = Self {
            model,
            options,
            n_batch: args.n_batch.unwrap_or(args.n_ctx),
            n_ubatch: 0,
            n_ctx: 0,
            threads: args.threads,
            op_offload: !(args.no_op_offload || args.n_gpu_layers == 0),
            force_snapshots: args.force_snapshots,
            native_snapshots: false,
            continue_tokens: args.continue_tokens.max(1),
        };
        let probe = rig.bare_context(None)?;
        rig.n_ctx = probe.n_ctx();
        rig.n_batch = probe.n_batch();
        rig.n_ubatch = options.context_params().n_ubatch.min(rig.n_batch);
        rig.native_snapshots = probe.seq_snapshots_enabled();
        Ok(rig)
    }

    /// A fresh context, snapshots as the model defaults them. `slots`
    /// asks for that many sequences over a unified KV cache, as
    /// `--cache-slots` does for blallama.
    fn bare_context(&self, slots: Option<u32>) -> Result<LlamaCppDecoder> {
        let options = match slots {
            Some(slots) => self.options.with_cache_slots(slots),
            None => self.options,
        };
        let mut params = options.context_params();
        params.n_batch = self.n_batch;
        params.op_offload = self.op_offload;
        if let Some(threads) = self.threads {
            params.n_threads = threads;
            params.n_threads_batch = threads;
        }
        Ok(LlamaCppDecoder::new(&self.model, params, None)?)
    }

    /// [`Self::bare_context`], with `--force-snapshots` applied.
    fn context(&self, slots: Option<u32>) -> Result<LlamaCppDecoder> {
        let mut decoder = self.bare_context(slots)?;
        if self.force_snapshots {
            decoder.set_seq_snapshots(true);
        }
        Ok(decoder)
    }

    /// The whole of `len` tokens in as few decode calls as `n_batch`
    /// allows — one, at the defaults.
    fn whole(&self, len: usize) -> Vec<usize> {
        chunks(len, self.n_batch as usize)
    }

    /// `schedule` with any chunk larger than `n_batch` split to fit.
    fn fit(&self, schedule: impl IntoIterator<Item = usize>) -> Vec<usize> {
        schedule
            .into_iter()
            .filter(|&n| n > 0)
            .flat_map(|n| chunks(n, self.n_batch as usize))
            .collect()
    }

    /// Prefill `tokens` per `schedule` in a fresh context and continue
    /// greedily.
    fn run(
        &self,
        tokens: &[Token],
        schedule: &[usize],
        seq: i32,
        slots: Option<u32>,
    ) -> Result<Run> {
        let mut decoder = self.context(slots)?;
        let logits = prefill(&mut decoder, tokens, 0, seq, schedule)?;
        self.greedy(&mut decoder, logits, tokens.len(), seq)
    }

    /// Greedy continuation from a prefill that ended just before `pos`
    /// and left `logits`.
    fn greedy(
        &self,
        decoder: &mut LlamaCppDecoder,
        logits: Vec<f32>,
        pos: usize,
        seq: i32,
    ) -> Result<Run> {
        greedy(decoder, logits, pos, seq, self.continue_tokens)
    }

    /// The settings that shape the reference run, for `--dump` and
    /// `--compare`.
    fn config(
        &self,
        args: &Args,
        prompt: &Prompt,
        split: usize,
    ) -> Result<Config> {
        let params = self.options.context_params();
        Ok(Config {
            model: args.model.display().to_string(),
            model_bytes: std::fs::metadata(&args.model)?.len(),
            n_ctx: self.n_ctx,
            n_batch: self.n_batch,
            n_ubatch: self.n_ubatch,
            flash_attn: args
                .flash_attn
                .map_or("auto".into(), |fa| format!("{fa:?}")),
            op_offload: self.op_offload,
            n_gpu_layers: args.n_gpu_layers,
            threads: self.threads.unwrap_or(params.n_threads),
            split,
            continue_tokens: self.continue_tokens,
            seed: args.seed,
            prompt_tokens: prompt.tokens.len(),
        })
    }

    fn header(
        &self,
        args: &Args,
        prompt: &Prompt,
        split: usize,
        junk_len: usize,
    ) {
        let params = self.options.context_params();
        println!("model        : {}", args.model.display());
        println!(
            "             : {} ({} params, vocab {})",
            self.model.desc(),
            self.model.n_params(),
            self.model.n_vocab(),
        );
        println!(
            "devices      : {:?}, n_gpu_layers {}, op_offload {}, \
             offload_kqv {}",
            gpu_device_names(),
            args.n_gpu_layers,
            self.op_offload,
            params.offload_kqv,
        );
        println!(
            "context      : n_ctx {}, n_batch {}, n_ubatch {}, flash_attn \
             {}, threads {}",
            self.n_ctx,
            self.n_batch,
            self.n_ubatch,
            args.flash_attn
                .map_or("auto (default)".into(), |fa| format!("{fa:?}")),
            self.threads
                .map_or(params.n_threads.to_string(), |t| t.to_string()),
        );
        println!(
            "rewind       : {}",
            match (self.native_snapshots, self.force_snapshots) {
                (true, _) => {
                    "restore_to reloads partial checkpoints when a KV \
                     truncate cannot rewind (sliding-window, hybrid or \
                     recurrent model)"
                }
                (false, true) => {
                    "restore_to truncates the KV (pure attention); \
                     --force-snapshots only stores snapshots, so just the \
                     get_state_seq row loads state"
                }
                (false, false) => {
                    "restore_to truncates the KV (pure attention); just the \
                     get_state_seq row loads state"
                }
            }
        );
        println!(
            "prompt       : {} tokens (seed {}), split at {split}, junk {} \
             tokens",
            prompt.tokens.len(),
            args.seed,
            junk_len,
        );
        println!(
            "prompt tail  : {:?}\n",
            tail(&prompt.text, 160).replace('\n', "⏎"),
        );
    }
}

/// Split `len` into chunks of `size`, the last one short.
fn chunks(len: usize, size: usize) -> Vec<usize> {
    (0..len)
        .step_by(size.max(1))
        .map(|start| size.min(len - start))
        .collect()
}

/// Prefill `tokens` from `start` on `seq`, one [`Decoder::prefill`] per
/// entry of `schedule` — the call `Session` makes per breakpoint chunk —
/// and return the final position's logits.
fn prefill(
    decoder: &mut LlamaCppDecoder,
    tokens: &[Token],
    start: usize,
    seq: i32,
    schedule: &[usize],
) -> Result<Vec<f32>> {
    assert_eq!(
        schedule.iter().sum::<usize>(),
        tokens.len(),
        "a schedule must cover its tokens exactly"
    );
    let mut offset = 0;
    let mut logits = Vec::new();
    for (i, &n) in schedule.iter().enumerate() {
        let chunk = &tokens[offset..offset + n];
        let out = Decoder::prefill(decoder, chunk, start + offset, seq)?;
        if i + 1 == schedule.len() {
            logits = out.to_vec();
        }
        offset += n;
    }
    Ok(logits)
}

/// Assert `seq` holds positions `[0, len)` — nothing when `len` is 0 —
/// after a rewind, clear or removal, so a wrong KV extent fails loudly
/// instead of reading as a numerical difference.
fn expect_extent(
    decoder: &mut LlamaCppDecoder,
    seq: i32,
    len: usize,
    after: &str,
) {
    let pos_max = Decoder::memory_seq_pos_max(decoder, seq);
    assert_eq!(
        pos_max,
        len as i32 - 1,
        "after {after}, seq {seq} should hold positions [0, {len}) but its \
         memory_seq_pos_max is {pos_max}"
    );
}

/// Argmax, lowest id on ties, so greedy decoding is a pure function of
/// the logits.
fn argmax(logits: &[f32]) -> Token {
    let mut best = 0;
    for (i, &l) in logits.iter().enumerate() {
        if l > logits[best] {
            best = i;
        }
    }
    best as Token
}

/// Gap between the top two logits.
fn margin(logits: &[f32]) -> f32 {
    let (mut first, mut second) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &l in logits {
        if l > first {
            second = first;
            first = l;
        } else if l > second {
            second = l;
        }
    }
    first - second
}

/// Decode `n` tokens greedily with [`Decoder::step`], the call the
/// predictors make per generated token.
fn greedy(
    decoder: &mut LlamaCppDecoder,
    logits: Vec<f32>,
    pos: usize,
    seq: i32,
    n: usize,
) -> Result<Run> {
    let mut tokens = Vec::with_capacity(n);
    let mut steps = Vec::with_capacity(n.saturating_sub(1));
    let mut next = argmax(&logits);
    for i in 0..n {
        tokens.push(next);
        if i + 1 == n {
            break;
        }
        let out = Decoder::step(decoder, next, pos + i, seq)?;
        next = argmax(out);
        steps.push(out.to_vec());
    }
    Ok(Run {
        logits,
        tokens,
        steps,
    })
}

// ---------------------------------------------------------------------
// Runs and comparisons
// ---------------------------------------------------------------------

/// One prefill and its greedy continuation.
struct Run {
    /// Final-position logits of the prefill.
    logits: Vec<f32>,
    /// The continuation.
    tokens: Vec<Token>,
    /// Logits after feeding `tokens[i]`, which chose `tokens[i + 1]`.
    steps: Vec<Vec<f32>>,
}

/// How a run differs from its baseline.
struct Cmp {
    /// Final logits whose bits differ.
    n_diff: usize,
    max_abs: f32,
    argmax: (Token, Token),
    /// The baseline's top-two gap: how close a flip was.
    margin: f32,
    /// First continuation token that differs.
    diverge: Option<usize>,
    /// First continuation step, while the tokens still agree, whose
    /// logits are not bit-identical.
    step_diff: Option<usize>,
    step_max_abs: f32,
}

impl Cmp {
    fn new(base: &Run, run: &Run) -> Self {
        let (n_diff, max_abs) = diff(&base.logits, &run.logits);
        let diverge = base
            .tokens
            .iter()
            .zip(&run.tokens)
            .position(|(a, b)| a != b)
            .or((base.tokens.len() != run.tokens.len())
                .then(|| base.tokens.len().min(run.tokens.len())));
        // Step i fed tokens[i]; its logits are comparable only while
        // tokens[..=i] agree.
        let comparable = diverge.unwrap_or(usize::MAX);
        let mut step_diff = None;
        let mut step_max_abs = 0f32;
        for (i, (a, b)) in base.steps.iter().zip(&run.steps).enumerate() {
            if i >= comparable {
                break;
            }
            let (n, max) = diff(a, b);
            if n > 0 && step_diff.is_none() {
                step_diff = Some(i);
            }
            step_max_abs = step_max_abs.max(max);
        }
        Self {
            n_diff,
            max_abs,
            argmax: (argmax(&base.logits), argmax(&run.logits)),
            margin: margin(&base.logits),
            diverge,
            step_diff,
            step_max_abs,
        }
    }

    fn identical(&self) -> bool {
        self.n_diff == 0 && self.diverge.is_none() && self.step_diff.is_none()
    }

    fn flipped(&self) -> bool {
        self.argmax.0 != self.argmax.1
    }
}

/// Count of bitwise-different entries and the largest absolute gap.
fn diff(a: &[f32], b: &[f32]) -> (usize, f32) {
    if a.len() != b.len() {
        return (a.len().max(b.len()), f32::INFINITY);
    }
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .fold((0, 0f32), |(n, max), (x, y)| {
            (n + 1, max.max((x - y).abs()))
        })
}

/// One experiment's table: rows print as they finish, the verdict at
/// the end.
struct Table<'a> {
    model: &'a LlamaCppModel,
    rows: Vec<(String, Cmp)>,
    notes: Vec<String>,
}

impl<'a> Table<'a> {
    fn new(model: &'a LlamaCppModel, title: &str, baseline: &str) -> Self {
        println!("== {title} ==  (baseline: {baseline})");
        println!(
            "{:<46} {:>5} {:>7} {:>10} {:>15} {:>5} {:>8} {:>6} {:>6} {:>10} {:>6}",
            "variant",
            "bits",
            "n_diff",
            "max|Δ|",
            "argmax b→v",
            "flip",
            "margin",
            "div@",
            "step≠",
            "step|Δ|",
            "secs",
        );
        Self {
            model,
            rows: Vec::new(),
            notes: Vec::new(),
        }
    }

    fn row(&mut self, label: impl Into<String>, cmp: Cmp, started: Instant) {
        let label = label.into();
        let opt =
            |o: Option<usize>| o.map_or("-".to_string(), |i| i.to_string());
        println!(
            "{:<46} {:>5} {:>7} {:>10.3e} {:>15} {:>5} {:>8.4} {:>6} {:>6} {:>10.3e} {:>6.1}",
            label,
            if cmp.n_diff == 0 { "same" } else { "DIFF" },
            cmp.n_diff,
            cmp.max_abs,
            format!("{}→{}", cmp.argmax.0, cmp.argmax.1),
            if cmp.flipped() { "FLIP" } else { "-" },
            cmp.margin,
            opt(cmp.diverge),
            opt(cmp.step_diff),
            cmp.step_max_abs,
            started.elapsed().as_secs_f64(),
        );
        self.rows.push((label, cmp));
    }

    /// The row `label`'s comparison; `None` when there is no such row.
    fn get(&self, label: &str) -> Option<&Cmp> {
        self.rows.iter().find(|(l, _)| l == label).map(|(_, c)| c)
    }

    /// Whether the row `label` matched its baseline bit for bit;
    /// `None` when there is no such row.
    fn identical(&self, label: &str) -> Option<bool> {
        self.get(label).map(Cmp::identical)
    }

    /// A row that could not be measured.
    fn skip(&mut self, label: &str, why: &str) {
        println!("{label:<46} n/a — {why}");
        self.notes.push(format!("{label}: {why}"));
    }

    /// Print the first divergent token of each diverging row, then a
    /// one-line verdict: `same` when every row matched bit for bit,
    /// otherwise `differ` plus the numbers.
    fn verdict(
        self,
        base: &Run,
        runs: &[(String, Run)],
        same: &str,
        differ: &str,
    ) {
        self.divergences(base, runs);
        let differing: Vec<&Cmp> = self
            .rows
            .iter()
            .map(|(_, c)| c)
            .filter(|c| !c.identical())
            .collect();
        let verdict = if differing.is_empty() {
            format!("IDENTICAL — {same}")
        } else {
            let max_abs =
                differing.iter().map(|c| c.max_abs).fold(0f32, f32::max);
            let flips = differing.iter().filter(|c| c.flipped()).count();
            let earliest = differing.iter().filter_map(|c| c.diverge).min();
            format!(
                "DIFFERENT ({}/{} variants; max final |Δ| {max_abs:.3e}, {flips} \
                 argmax flip(s), earliest continuation divergence {}) — \
                 {differ}",
                differing.len(),
                self.rows.len(),
                earliest.map_or("none".into(), |i| format!("at token {i}")),
            )
        };
        println!("verdict: {verdict}");
        self.footer();
    }

    /// Print where each diverging row's continuation first leaves the
    /// baseline's, and the token each side chose there.
    fn divergences(&self, base: &Run, runs: &[(String, Run)]) {
        for (label, cmp) in &self.rows {
            let Some(at) = cmp.diverge else { continue };
            let Some((_, run)) = runs.iter().find(|(l, _)| l == label) else {
                continue;
            };
            let piece = |tokens: &[Token]| {
                tokens.get(at).map_or("<end>".into(), |&t| {
                    format!("{:?}", self.model.token_to_piece(t))
                })
            };
            println!(
                "  {label}: continuation diverges at token {at} after {:?}: \
                 baseline {} vs {}",
                tail(
                    &self
                        .model
                        .tokens_to_string(base.tokens[..at].iter().copied()),
                    60
                ),
                piece(&base.tokens),
                piece(&run.tokens),
            );
        }
    }

    /// Close the table: the rows that could not be measured, then a
    /// blank line.
    fn footer(self) {
        for note in &self.notes {
            println!("  not measured: {note}");
        }
        println!();
    }
}

/// Keep what a table's divergence report needs from a run: the tokens.
/// The per-step logits are large (vocab × continuation) and only the
/// baseline's are compared against.
fn light(label: &str, run: &Run) -> (String, Run) {
    (
        label.to_string(),
        Run {
            logits: Vec::new(),
            tokens: run.tokens.clone(),
            steps: Vec::new(),
        },
    )
}

// ---------------------------------------------------------------------
// Experiments
// ---------------------------------------------------------------------

fn repeat(
    rig: &Rig,
    prompt: &Prompt,
    junk: &Prompt,
    reference: &Run,
    repeats: usize,
) -> Result<()> {
    let tokens = &prompt.tokens;
    let whole = rig.whole(tokens.len());
    let mut table = Table::new(
        &rig.model,
        "REPEAT: the same prompt and schedule, prefilled again",
        "reference: fresh context, one batch",
    );
    let mut runs = Vec::new();

    for k in 2..=repeats.max(2) {
        let started = Instant::now();
        let label = format!("fresh context #{k}");
        let run = rig.run(tokens, &whole, 0, None)?;
        table.row(&label, Cmp::new(reference, &run), started);
        runs.push(light(&label, &run));
    }

    // One context, the prompt twice with a full clear between.
    let started = Instant::now();
    let label = "same context, 2nd prefill after memory_clear";
    let mut decoder = rig.context(None)?;
    let logits = prefill(&mut decoder, tokens, 0, 0, &whole)?;
    rig.greedy(&mut decoder, logits, tokens.len(), 0)?;
    Decoder::memory_clear(&mut decoder);
    expect_extent(&mut decoder, 0, 0, "memory_clear");
    let logits = prefill(&mut decoder, tokens, 0, 0, &whole)?;
    let run = rig.greedy(&mut decoder, logits, tokens.len(), 0)?;
    table.row(label, Cmp::new(reference, &run), started);
    runs.push(light(label, &run));
    drop(decoder);

    // A longer unrelated prompt and some decode cells first, then the
    // cache-miss path (`seq_rm(seq, -1, -1)`) or a full clear.
    for (label, clear) in [
        ("after longer prompt → seq_rm(0, -1, -1)", false),
        ("after longer prompt → memory_clear", true),
    ] {
        let started = Instant::now();
        let mut decoder = rig.context(None)?;
        dirty(rig, &mut decoder, junk, 0)?;
        if clear {
            Decoder::memory_clear(&mut decoder);
        } else if !Decoder::memory_seq_rm(&mut decoder, 0, -1, -1) {
            table.skip(label, "memory_seq_rm(0, -1, -1) refused");
            continue;
        }
        expect_extent(&mut decoder, 0, 0, label);
        let logits = prefill(&mut decoder, tokens, 0, 0, &whole)?;
        let run = rig.greedy(&mut decoder, logits, tokens.len(), 0)?;
        table.row(label, Cmp::new(reference, &run), started);
        runs.push(light(label, &run));
    }

    table.verdict(
        reference,
        &runs,
        "prefill is deterministic here, fresh or reused context; a \
         cold-vs-cold server mismatch must come from a different schedule \
         or context shape (see chunking / neighbors), or from above the \
         decoder",
        "fresh-context rows differing is nondeterminism in llama.cpp or \
         its kernels (not Session); only reused-context rows differing \
         means leftover KV state leaks into results",
    );
    Ok(())
}

/// Fill `seq` with an unrelated, longer prompt and a few generated
/// tokens, leaving occupied cells beyond where the real prompt will end.
fn dirty(
    rig: &Rig,
    decoder: &mut LlamaCppDecoder,
    junk: &Prompt,
    seq: i32,
) -> Result<()> {
    let tokens = &junk.tokens;
    let logits = prefill(decoder, tokens, 0, seq, &rig.whole(tokens.len()))?;
    greedy(decoder, logits, tokens.len(), seq, JUNK_GEN)?;
    Ok(())
}

fn chunking(
    rig: &Rig,
    prompt: &Prompt,
    split: usize,
    reference: &Run,
    extra: &[usize],
) -> Result<()> {
    let len = prompt.tokens.len();
    let rest = len - split;
    let tail = TAIL.min(len - 1);
    let mut schedules: Vec<(String, Vec<usize>)> =
        vec![("chunks of 512".into(), rig.fit(chunks(len, 512)))];
    for &size in extra.iter().filter(|&&s| s > 0) {
        schedules
            .push((format!("chunks of {size}"), rig.fit(chunks(len, size))));
    }
    schedules.extend([
        (
            format!("breakpoint: [{split} | {rest}]"),
            rig.fit([split, rest]),
        ),
        (
            format!(
                "breakpoints: [{split} | {} | {}]",
                rest / 2,
                rest - rest / 2
            ),
            rig.fit([split, rest / 2, rest - rest / 2]),
        ),
        (
            format!("one batch, last {tail} token-by-token"),
            rig.fit(
                std::iter::once(len - tail).chain(std::iter::repeat_n(1, tail)),
            ),
        ),
    ]);

    let mut table = Table::new(
        &rig.model,
        "CHUNKING: the same prompt split differently across decode calls",
        "reference: fresh context, one batch",
    );
    let mut runs = Vec::new();
    for (label, schedule) in &schedules {
        let started = Instant::now();
        let run = rig.run(&prompt.tokens, schedule, 0, None)?;
        table.row(label, Cmp::new(reference, &run), started);
        runs.push(light(label, &run));
    }
    table.verdict(
        reference,
        &runs,
        "logits do not depend on how the prompt is split (batch-invariant \
         here)",
        "logits depend on the chunk schedule (llama.cpp is not \
         batch-invariant). Breakpoint rows are not the warm/cold gap on a \
         breakpoint hit (the cold walk splits at the same breakpoints — see \
         snapshot); the token-by-token row is the tip-reuse gap, where the \
         warm side stepped the previous reply and the cold side batches \
         it, and can flip a greedy near-tie with no cache bug",
    );
    Ok(())
}

fn align(
    rig: &Rig,
    prompt: &Prompt,
    split: usize,
    reference: &Run,
) -> Result<()> {
    let len = prompt.tokens.len();
    let u = rig.n_ubatch as usize;
    // Chunks from `start` to the end whose boundaries all fall on
    // multiples of `u` counted from position 0.
    let realign = |start: usize| -> Vec<usize> {
        let first = start.next_multiple_of(u).min(len);
        std::iter::once(first - start)
            .chain(chunks(len - first, u))
            .filter(|&n| n > 0)
            .collect()
    };
    let fmt = |schedule: &[usize]| match schedule {
        [a, b, c, ..] if schedule.len() > 3 => {
            format!("[{a} | {b} | {c} | …×{u}]")
        }
        _ => format!(
            "[{}]",
            schedule
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(" | ")
        ),
    };

    let mut table = Table::new(
        &rig.model,
        &format!(
            "ALIGN: a breakpoint split with the n_ubatch ({u}) grid restored"
        ),
        "reference: fresh context, one batch",
    );
    let mut runs = Vec::new();
    let mut measure = |table: &mut Table, label: &str, schedule: &[usize]| {
        let started = Instant::now();
        let run = rig.run(&prompt.tokens, schedule, 0, None)?;
        table.row(label, Cmp::new(reference, &run), started);
        runs.push(light(label, &run));
        Result::Ok(())
    };

    let control = rig.fit([split, len - split]);
    let control_label = format!("unaligned (control) {}", fmt(&control));
    measure(&mut table, &control_label, &control)?;

    let realigned: Vec<usize> =
        rig.fit(std::iter::once(split).chain(realign(split)));
    let realigned_label = format!("realigned {}", fmt(&realigned));
    measure(&mut table, &realigned_label, &realigned)?;

    let on_grid = if split >= u { split - split % u } else { u };
    let on_grid_label = format!("on the grid [{on_grid} | rest]");
    if on_grid < len {
        measure(
            &mut table,
            &on_grid_label,
            &rig.fit([on_grid, len - on_grid]),
        )?;
    } else {
        table.skip(&on_grid_label, "the prompt is shorter than n_ubatch");
    }

    let odd = ODD_CHUNK.min(len - split - 1);
    let odd_label;
    if odd > 0 {
        let schedule =
            rig.fit([split, odd].into_iter().chain(realign(split + odd)));
        odd_label = format!("+{odd}, realigned {}", fmt(&schedule));
        measure(&mut table, &odd_label, &schedule)?;
    } else {
        odd_label = String::new();
        table.skip("+short chunk, realigned", "no room after the split");
    }

    let same = |label: &str| table.identical(label);
    let mut readings = Vec::new();
    if split.is_multiple_of(u) {
        readings.push(format!(
            "the split ({split}) is already on the grid, so the control and \
             the realigned rows run one schedule; pass an unaligned --split"
        ));
    }
    if same(&on_grid_label) == Some(true) {
        readings.push(
            "a decode-call boundary on the grid is free: only ubatch \
             boundaries change logits"
                .into(),
        );
    }
    match (same(&control_label), same(&realigned_label)) {
        (Some(false), Some(true)) => readings.push(
            "the breakpoint's own ubatch boundary is harmless; the shifted \
             ubatches after it are what differ. Realigning the chunk after \
             each breakpoint to the next multiple of n_ubatch would make \
             breakpoint chunking schedule-invariant"
                .into(),
        ),
        (_, Some(false)) => readings.push(
            "an extra ubatch boundary changes logits even with every later \
             ubatch back on the grid: alignment alone cannot make breakpoint \
             chunking schedule-invariant"
                .into(),
        ),
        _ => {}
    }
    if odd > 0 && same(&odd_label) != same(&realigned_label) {
        readings.push(
            "a second short chunk before realigning changes the outcome: the \
             count of off-grid boundaries matters, not just the grid after"
                .into(),
        );
    }
    for reading in readings {
        println!("  reading: {reading}");
    }
    table.verdict(
        reference,
        &runs,
        "every schedule matches the one-batch reference, the unaligned \
         control included",
        "see the readings above: which rows differ says whether an \
         off-grid breakpoint costs anything once the ubatch grid is \
         restored",
    );
    Ok(())
}

fn snapshot(
    rig: &Rig,
    prompt: &Prompt,
    junk: &Prompt,
    split: usize,
) -> Result<()> {
    let tokens = &prompt.tokens;
    let (prefix, suffix) = tokens.split_at(split);
    let pos = split as i32;
    let fresh = rig.fit([split, suffix.len()]);
    let suffix_schedule = rig.fit([suffix.len()]);

    println!(
        "(computing the snapshot baseline: fresh context, [prefix | suffix])"
    );
    let base = rig.run(tokens, &fresh, 0, None)?;
    let mut table = Table::new(
        &rig.model,
        "SNAPSHOT: prefix restored from the cache, then the suffix",
        "fresh context, the same [prefix | suffix] schedule",
    );
    let mut runs = Vec::new();

    // Cold, with the checkpoint Session takes at a breakpoint.
    let started = Instant::now();
    let label = "cold + checkpoint_pos(prefix)";
    let mut decoder = rig.context(None)?;
    prefill(&mut decoder, prefix, 0, 0, &rig.fit([split]))?;
    Decoder::checkpoint_pos(&mut decoder, 0, pos);
    let bytes = decoder.get_state_seq(0);
    let logits = prefill(&mut decoder, suffix, split, 0, &suffix_schedule)?;
    let run = rig.greedy(&mut decoder, logits, tokens.len(), 0)?;
    table.row(label, Cmp::new(&base, &run), started);
    runs.push(light(label, &run));

    // Warm: rewind the same context to the prefix, as Session does on a
    // hit, and prefill the suffix again.
    let warm = |decoder: &mut LlamaCppDecoder| -> Result<(Run, &'static str)> {
        let path = rewind(decoder, pos)?;
        let logits = prefill(decoder, suffix, split, 0, &suffix_schedule)?;
        Ok((rig.greedy(decoder, logits, tokens.len(), 0)?, path))
    };
    for n in 1..=2 {
        let started = Instant::now();
        match warm(&mut decoder) {
            Ok((run, path)) => {
                let label = format!("warm #{n}: restore_to(prefix) [{path}]");
                table.row(&label, Cmp::new(&base, &run), started);
                runs.push(light(&label, &run));
            }
            Err(e) => table.skip(&format!("warm #{n}"), &e.to_string()),
        }
    }

    // Warm after a different branch: another suffix was prefilled and
    // generated from the prefix in between.
    let started = Instant::now();
    let other = &junk.tokens[..suffix.len().min(junk.tokens.len())];
    let branched = rewind(&mut decoder, pos).and_then(|_| {
        let logits =
            prefill(&mut decoder, other, split, 0, &rig.fit([other.len()]))?;
        rig.greedy(&mut decoder, logits, split + other.len(), 0)?;
        warm(&mut decoder)
    });
    match branched {
        Ok((run, path)) => {
            let label = format!("warm after another branch [{path}]");
            table.row(&label, Cmp::new(&base, &run), started);
            runs.push(light(&label, &run));
        }
        Err(e) => table.skip("warm after another branch", &e.to_string()),
    }
    drop(decoder);

    // The raw sequence state, loaded into a context that never saw the
    // prompt.
    let started = Instant::now();
    let label = "get_state_seq bytes → fresh context";
    let mut decoder = rig.context(None)?;
    if decoder.set_state_seq(&bytes, 0) {
        expect_extent(&mut decoder, 0, split, "set_state_seq(prefix)");
        let logits = prefill(&mut decoder, suffix, split, 0, &suffix_schedule)?;
        let run = rig.greedy(&mut decoder, logits, tokens.len(), 0)?;
        table.row(label, Cmp::new(&base, &run), started);
        runs.push(light(label, &run));
    } else {
        table.skip(label, "set_state_seq rejected the bytes");
    }

    table.verdict(
        &base,
        &runs,
        "restoring the prefix is lossless: a warm prefill equals a cold one \
         with the same schedule",
        "restoring the prefix changes results — a lossy restore (drama_llama's \
         snapshot/restore path, or llama.cpp's state save/load for this \
         architecture)",
    );
    Ok(())
}

/// Rewind seq 0 to `pos` through [`Decoder::restore_to`], reporting
/// which path it took: the KV truncate it tries first (pure attention),
/// or the snapshot reload (hybrid and recurrent models refuse a
/// partial truncate).
fn rewind(decoder: &mut LlamaCppDecoder, pos: i32) -> Result<&'static str> {
    // The same probe restore_to opens with; on success its own attempt
    // is an empty-range no-op, so this changes nothing but the label.
    let truncated = Decoder::memory_seq_rm(decoder, 0, pos, -1)
        && Decoder::memory_seq_pos_max(decoder, 0) == pos - 1;
    Decoder::restore_to(decoder, 0, pos)?;
    expect_extent(decoder, 0, pos as usize, &format!("restore_to({pos})"));
    Ok(if truncated {
        "KV truncate"
    } else {
        "snapshot reload"
    })
}

/// The tip experiment's previous reply and every logits vector along
/// the way: `logits[0]` ends the prompt's prefill, `logits[i + 1]`
/// follows feeding `tokens[i]`.
struct Tip {
    tokens: Vec<Token>,
    logits: Vec<Vec<f32>>,
}

/// Greedily generate `n` tokens after a prefill that left `logits` and
/// ended just before `pos`, feeding every one back with
/// [`Decoder::step`] — the KV then holds the whole reply, as the warm
/// side's does once its turn ends.
fn generate_tip(
    decoder: &mut LlamaCppDecoder,
    logits: Vec<f32>,
    pos: usize,
    seq: i32,
    n: usize,
) -> Result<Tip> {
    let mut tokens = Vec::with_capacity(n);
    let mut all = Vec::with_capacity(n + 1);
    all.push(logits);
    for i in 0..n {
        let next = argmax(&all[i]);
        tokens.push(next);
        all.push(Decoder::step(decoder, next, pos + i, seq)?.to_vec());
    }
    Ok(Tip {
        tokens,
        logits: all,
    })
}

/// Feed `tokens` one [`Decoder::step`] at a time from `pos`, as
/// [`generate_tip`] did, and return the logits after each.
fn replay(
    decoder: &mut LlamaCppDecoder,
    tokens: &[Token],
    pos: usize,
    seq: i32,
) -> Result<Vec<Vec<f32>>> {
    tokens
        .iter()
        .enumerate()
        .map(|(i, &token)| -> Result<Vec<f32>> {
            Ok(Decoder::step(decoder, token, pos + i, seq)?.to_vec())
        })
        .collect()
}

fn tip(rig: &Rig, prompt: &Prompt, n: usize) -> Result<()> {
    let p = &prompt.tokens;
    let len = p.len();
    let n = n.max(1);
    let turn = rig.model.tokenize(TIP_TURN, false);
    let other = rig.model.tokenize(TIP_OTHER_TURN, false);
    // The tip: P and the reply, positions [0, tip_end).
    let tip_end = len + n;
    let tip_pos = tip_end as i32;
    let need =
        tip_end + (turn.len() + TIP_CONTINUE).max(other.len() + JUNK_GEN);
    if need > rig.n_ctx as usize {
        println!(
            "== TIP == skipped: needs {need} tokens of n_ctx {}\n",
            rig.n_ctx
        );
        return Ok(());
    }
    let p_schedule = rig.whole(len);
    let turn_schedule = rig.fit([turn.len()]);
    // The new turn after the tip, then a greedy continuation.
    let answer = |decoder: &mut LlamaCppDecoder| -> Result<Run> {
        let logits = prefill(decoder, &turn, tip_end, 0, &turn_schedule)?;
        greedy(decoder, logits, tip_end + turn.len(), 0, TIP_CONTINUE)
    };
    // (d)/(e): back to the tip, the new turn, and the extent the
    // restore left.
    let restored = |decoder: &mut LlamaCppDecoder| {
        let path = rewind(decoder, tip_pos)?;
        let pos_max = Decoder::memory_seq_pos_max(decoder, 0);
        Result::Ok((answer(decoder)?, path, pos_max))
    };

    // (a) warm: the reply stepped, the tip checkpointed as Session
    // does, then the new turn in the same context.
    println!(
        "(computing the warm baseline: [{len}], {n} greedy steps, \
         checkpoint_pos({tip_end}), [{}])",
        turn.len()
    );
    let mut warm = rig.context(None)?;
    let logits = prefill(&mut warm, p, 0, 0, &p_schedule)?;
    let tip = generate_tip(&mut warm, logits, len, 0, n)?;
    Decoder::checkpoint_pos(&mut warm, 0, tip_pos);
    let base = answer(&mut warm)?;
    println!(
        "tip ({n} tokens): {:?}\nnew turn ({} tokens): {:?}",
        rig.model.tokens_to_string(tip.tokens.iter().copied()),
        turn.len(),
        TIP_TURN,
    );
    let mut table = Table::new(
        &rig.model,
        "TIP: the previous reply reused as the next request's prefix",
        "(a) warm: [P], tip stepped, checkpoint, [S]",
    );
    let mut runs = Vec::new();
    let mut readings = Vec::new();

    // (d) warm: restore_to(tip) after (a)'s own turn, then after
    // another branch grown from the tip.
    let mut d_rows: Vec<Option<bool>> = Vec::new();
    for branch in [false, true] {
        let started = Instant::now();
        let when = if branch {
            "after another branch"
        } else {
            "after its own turn"
        };
        let outcome = if branch {
            rewind(&mut warm, tip_pos).and_then(|_| {
                let logits = prefill(
                    &mut warm,
                    &other,
                    tip_end,
                    0,
                    &rig.fit([other.len()]),
                )?;
                greedy(&mut warm, logits, tip_end + other.len(), 0, JUNK_GEN)?;
                restored(&mut warm)
            })
        } else {
            restored(&mut warm)
        };
        match outcome {
            Ok((run, path, pos_max)) => {
                let label = format!("(d) {when} [{path}]");
                readings.push(format!(
                    "(e) restore_to({tip_end}) {when} [{path}]: \
                     memory_seq_pos_max {pos_max} (expected {}; already \
                     asserted by rewind, and on a hybrid the minimum of \
                     the attention and recurrent extents)",
                    tip_end - 1
                ));
                let cmp = Cmp::new(&base, &run);
                d_rows.push(Some(cmp.identical()));
                table.row(&label, cmp, started);
                runs.push(light(&label, &run));
            }
            Err(e) => {
                d_rows.push(None);
                table.skip(&format!("(d) {when}"), &e.to_string());
            }
        }
    }
    drop(warm);

    // (b) cold, matched: a fresh context making (a)'s exact calls — P
    // with its schedule, the tip one step at a time, then S.
    let started = Instant::now();
    let b_label = format!("(b) cold, matched: [{len}], {n} steps, [S]");
    let mut cold = rig.context(None)?;
    let mut matched = vec![prefill(&mut cold, p, 0, 0, &p_schedule)?];
    matched.extend(replay(&mut cold, &tip.tokens, len, 0)?);
    let run = answer(&mut cold)?;
    drop(cold);
    let b_cmp = Cmp::new(&base, &run);
    let b_same = b_cmp.identical();
    table.row(&b_label, b_cmp, started);
    runs.push(light(&b_label, &run));
    let phase = tip
        .logits
        .iter()
        .zip(&matched)
        .map(|(a, b)| diff(a, b))
        .enumerate()
        .filter(|(_, (n_diff, _))| *n_diff > 0)
        .fold(None, |acc: Option<(usize, f32)>, (i, (_, max))| {
            Some(acc.map_or((i, max), |(first, m)| (first, m.max(max))))
        });
    readings.push(match phase {
        None => format!(
            "tip phase: (b)'s prefill of P and {n} replayed steps are \
             bit-identical to (a)'s generation"
        ),
        Some((i, max)) => format!(
            "tip phase: (b) already differs from (a) at {} (max |Δ| \
             {max:.3e} over the tip) — before the new turn",
            match i {
                0 => "P's prefill logits".to_string(),
                i => format!("the step feeding tip token {}", i - 1),
            }
        ),
    });
    drop(matched);

    // (c) cold, batched: blallama's cold path prefills the tip with P.
    let all: Vec<Token> =
        p.iter().chain(&tip.tokens).chain(&turn).copied().collect();
    let c_rows = [
        (
            format!("(c) cold, batched: [{len}+{n} | S]"),
            rig.fit([tip_end, turn.len()]),
        ),
        (
            format!("(c) cold, one batch: [{len}+{n}+S]"),
            rig.whole(all.len()),
        ),
    ];
    for (label, schedule) in &c_rows {
        let started = Instant::now();
        let mut decoder = rig.context(None)?;
        let logits = prefill(&mut decoder, &all, 0, 0, schedule)?;
        let run = greedy(&mut decoder, logits, all.len(), 0, TIP_CONTINUE)?;
        table.row(label, Cmp::new(&base, &run), started);
        runs.push(light(label, &run));
    }
    let mut c_gap = false;
    for (label, _) in &c_rows {
        let Some(cmp) = table.get(label) else {
            continue;
        };
        c_gap |= !cmp.identical();
        readings.push(if cmp.identical() {
            format!("{label}: bit-identical to warm — no schedule gap here")
        } else {
            format!(
                "{label}: the schedule gap — final |Δ| {:.3e} ({} logits), \
                 argmax {}, continuation diverges {}, step |Δ| {:.3e} while \
                 tokens agree",
                cmp.max_abs,
                cmp.n_diff,
                if cmp.flipped() { "flipped" } else { "kept" },
                cmp.diverge
                    .map_or("nowhere".into(), |i| format!("at token {i}")),
                cmp.step_max_abs,
            )
        });
    }

    table.divergences(&base, &runs);
    for reading in &readings {
        println!("  reading: {reading}");
    }
    let lossy: Vec<&str> = [
        (!b_same).then_some(
            "(b) differs: a fresh context making the same calls does not \
             reproduce the warm KV",
        ),
        d_rows.contains(&Some(false)).then_some(
            "(d) differs: restoring the tip is lossy (drama_llama's \
             snapshot/restore, or llama.cpp's state save/load)",
        ),
    ]
    .into_iter()
    .flatten()
    .collect();
    let verdict = if !lossy.is_empty() {
        format!(
            "tip reuse NOT lossless under matched schedule — {}",
            lossy.join("; ")
        )
    } else if d_rows.contains(&None) {
        "inconclusive — (b) matched but a (d) row could not be measured"
            .to_string()
    } else if c_gap {
        "tip reuse lossless under matched schedule — (b) and (d) match (a) \
         bit for bit, so the warm/cold gap (c) shows on tip reuse is the \
         schedule, not the restore"
            .to_string()
    } else {
        "tip reuse lossless under matched schedule — (b) and (d) match (a) \
         bit for bit, and so does (c): no warm/cold gap here for the \
         schedule to explain"
            .to_string()
    };
    println!("verdict: {verdict}");
    println!(
        "  scope: drama_llama's decoder restore of a stepped tip{}; \
         blallama's tip matching (re-render vs generated tokens, stop trim, \
         #91) is not tested here — see misanthropic's equivalence matched \
         control",
        match lossy.is_empty() && !d_rows.contains(&None) {
            true => " is lossless",
            false => "",
        }
    );
    table.footer();
    Ok(())
}

fn neighbors(
    rig: &Rig,
    prompt: &Prompt,
    junk: &Prompt,
    reference: &Run,
) -> Result<()> {
    let tokens = &prompt.tokens;
    let whole = rig.whole(tokens.len());
    const SLOTS: Option<u32> = Some(2);

    println!("(computing the two-slot baseline: seq 0, empty cache)");
    let base = rig.run(tokens, &whole, 0, SLOTS)?;
    let mut table = Table::new(
        &rig.model,
        "NEIGHBORS: two sequences over one unified KV cache",
        "two-slot context, the prompt on seq 0 in an empty cache",
    );
    let mut runs = Vec::new();

    // The context shape alone, against the one-slot reference. The
    // baseline's own steps are there to compare against, so swap roles.
    let label = "one-slot reference vs this baseline";
    table.row(label, Cmp::new(&base, reference), Instant::now());
    runs.push(light(label, reference));

    let started = Instant::now();
    let label = "seq 1, empty cache";
    let run = rig.run(tokens, &whole, 1, SLOTS)?;
    table.row(label, Cmp::new(&base, &run), started);
    runs.push(light(label, &run));

    let started = Instant::now();
    let label = "seq 1, beside a longer prompt on seq 0";
    let mut decoder = rig.context(SLOTS)?;
    dirty(rig, &mut decoder, junk, 0)?;
    let logits = prefill(&mut decoder, tokens, 0, 1, &whole)?;
    let run = rig.greedy(&mut decoder, logits, tokens.len(), 1)?;
    table.row(label, Cmp::new(&base, &run), started);
    runs.push(light(label, &run));
    drop(decoder);

    let started = Instant::now();
    let label = "seq 0, after seq_rm of a longer prompt there";
    let mut decoder = rig.context(SLOTS)?;
    dirty(rig, &mut decoder, junk, 0)?;
    if Decoder::memory_seq_rm(&mut decoder, 0, -1, -1) {
        expect_extent(&mut decoder, 0, 0, label);
        let logits = prefill(&mut decoder, tokens, 0, 0, &whole)?;
        let run = rig.greedy(&mut decoder, logits, tokens.len(), 0)?;
        table.row(label, Cmp::new(&base, &run), started);
        runs.push(light(label, &run));
    } else {
        table.skip(label, "memory_seq_rm(0, -1, -1) refused");
    }

    table.verdict(
        &base,
        &runs,
        "results do not depend on the sequence id or what else the cache \
         holds",
        "results depend on the KV cache's other contents or shape (cell \
         placement / n_kv, or slot configuration) — a server-side cold \
         prefill is not reproducible unless the cache around it is",
    );
    Ok(())
}

/// `n_kv` for a ubatch whose highest occupied cell is `used - 1`, as
/// llama.cpp pads it (ignoring the clamp to the cache size).
fn n_kv(used: usize) -> usize {
    used.next_multiple_of(KV_PAD).max(KV_PAD)
}

fn kv_view(rig: &Rig, prompt: &Prompt, args: &Args) -> Result<()> {
    let tokens = &prompt.tokens;
    let len = tokens.len();
    let whole = rig.whole(len);
    // Cells seq 0 ends up holding: the prompt and the continuation
    // tokens greedy feeds back (all but the last).
    let held = len + rig.continue_tokens - 1;
    // The spacer pair: the most seq 1 can hold with every n_kv at the
    // baseline's final width, pad(held), then one token more.
    let in_pad = n_kv(held) - held;
    let spacer_ks: BTreeSet<usize> = [in_pad, in_pad + 1]
        .into_iter()
        .chain(args.kv_view.iter().copied())
        .filter(|&k| k > 0)
        .collect();
    let most = spacer_ks
        .iter()
        .chain(&args.kv_view)
        .copied()
        .max()
        .unwrap_or(0);
    let filler = Prompt::build(
        &rig.model,
        args.seed ^ 0x0cca_5100_f111_e5ed,
        most.max(held),
    );
    let n_ctx = rig.n_ctx as usize;
    const SLOTS: Option<u32> = Some(2);

    println!("(computing the two-slot baseline: seq 0, empty cache)");
    let base = rig.run(tokens, &whole, 0, SLOTS)?;
    println!(
        "(n_kv in labels is the final prompt ubatch's; seq 0 holds {held} \
         cells at the end, pad {}; spacer pair K={in_pad} / K={})",
        n_kv(held),
        in_pad + 1,
    );
    let mut table = Table::new(
        &rig.model,
        "KV-VIEW: the prompt on seq 0 while seq 1 holds K tokens",
        &format!("two-slot context, seq 1 empty (n_kv {})", n_kv(len)),
    );
    let mut runs = Vec::new();
    let mut changed = Vec::new();
    let mut unchanged = Vec::new();
    let mut sort = |cmp: &Cmp, name: String| {
        if cmp.identical() {
            unchanged.push(name)
        } else {
            changed.push(name)
        }
    };

    // Put `k` filler tokens on seq 1.
    let fill = |decoder: &mut LlamaCppDecoder, k: usize| -> Result<()> {
        prefill(decoder, &filler.tokens[..k], 0, 1, &rig.whole(k))?;
        expect_extent(decoder, 1, k, "filling seq 1");
        Ok(())
    };

    // Seq 1 first: the prompt lands k cells later and n_kv grows.
    for &k in &args.kv_view {
        let started = Instant::now();
        let label = if k == 0 {
            format!("K=0: seq 1 empty (n_kv {})", n_kv(len))
        } else {
            format!("K={k}: placed after seq 1 (n_kv {})", n_kv(k + len))
        };
        if k + held > n_ctx {
            table.skip(&label, "does not fit in n_ctx");
            continue;
        }
        let mut decoder = rig.context(SLOTS)?;
        if k > 0 {
            fill(&mut decoder, k)?;
        }
        let logits = prefill(&mut decoder, tokens, 0, 0, &whole)?;
        let run = rig.greedy(&mut decoder, logits, len, 0)?;
        let cmp = Cmp::new(&base, &run);
        sort(
            &cmp,
            if k == 0 {
                "K=0".to_string()
            } else {
                format!("K={k} placed after")
            },
        );
        table.row(&label, cmp, started);
        runs.push(light(&label, &run));
    }

    // A spacer of `held` tokens on seq 0, seq 1 after it, then the
    // spacer removed: the prompt reuses the baseline's cells and only
    // n_kv changes — one width for every ubatch and step, as seq 1 holds
    // the top cells.
    let mut pair: (Option<Run>, Option<Run>) = (None, None);
    for &k in &spacer_ks {
        let started = Instant::now();
        let tag = match k {
            k if k == in_pad => ", fits pad",
            k if k == in_pad + 1 => ", next pad",
            _ => "",
        };
        let label =
            format!("K={k}: freed spacer{tag} (n_kv {})", n_kv(held + k));
        if held + k > n_ctx {
            table.skip(&label, "does not fit in n_ctx");
            continue;
        }
        let mut decoder = rig.context(SLOTS)?;
        prefill(&mut decoder, &filler.tokens[..held], 0, 0, &rig.whole(held))?;
        fill(&mut decoder, k)?;
        if !Decoder::memory_seq_rm(&mut decoder, 0, -1, -1) {
            table.skip(&label, "memory_seq_rm(0, -1, -1) refused");
            continue;
        }
        expect_extent(&mut decoder, 0, 0, "removing the spacer");
        let logits = prefill(&mut decoder, tokens, 0, 0, &whole)?;
        let run = rig.greedy(&mut decoder, logits, len, 0)?;
        expect_extent(&mut decoder, 1, k, "the prompt on seq 0");
        let cmp = Cmp::new(&base, &run);
        sort(&cmp, format!("K={k} spacer"));
        table.row(&label, cmp, started);
        runs.push(light(&label, &run));
        if k == in_pad {
            pair.0 = Some(run);
        } else if k == in_pad + 1 {
            pair.1 = Some(run);
        }
    }

    let list = |v: &[String]| {
        if v.is_empty() {
            "none".to_string()
        } else {
            v.join(", ")
        }
    };
    println!("  changed results : {}", list(&changed));
    println!("  bit-identical   : {}", list(&unchanged));
    // The pair differs only in one more KV_PAD block of masked cells in
    // view: compare the two rows with each other, not the baseline.
    let crossing = match &pair {
        (Some(fits), Some(next)) => {
            let cmp = Cmp::new(fits, next);
            if cmp.identical() {
                format!(
                    "K={} vs K={in_pad} spacer: bit-identical — one more \
                     {KV_PAD}-cell block of masked cells in view changes \
                     nothing",
                    in_pad + 1
                )
            } else {
                format!(
                    "K={} vs K={in_pad} spacer: DIFFERENT (final |Δ| \
                     {:.3e}, continuation diverges {}) — crossing the pad \
                     boundary alone changes logits: n_kv width matters",
                    in_pad + 1,
                    cmp.max_abs,
                    cmp.diverge
                        .map_or("nowhere".into(), |i| format!("at token {i}")),
                )
            }
        }
        _ if in_pad == 0 => format!(
            "no spacer pair: seq 0's {held} cells end on a pad boundary, so \
             any K widens n_kv; pass another --continue-tokens"
        ),
        _ => "the spacer pair was not measured (see above)".to_string(),
    };
    println!("  pad crossing    : {crossing}");
    println!(
        "  to bracket another pad boundary: --kv-view {},{}  (m·{KV_PAD} − \
         {held}, and one more)",
        in_pad + KV_PAD,
        in_pad + KV_PAD + 1,
    );
    table.verdict(
        &base,
        &runs,
        "what seq 1 holds does not reach seq 0's logits, at any K tried",
        "seq 1's contents reach seq 0's logits. The pad crossing line says \
         whether n_kv width alone does it (the spacer pair differs from \
         each other); spacer rows differing from the baseline mix that \
         with their wider view on the early ubatches; placed-after rows \
         differing while spacer rows match means the prompt's cell offset \
         does",
    );
    Ok(())
}

// ---------------------------------------------------------------------
// Cross-process comparison
// ---------------------------------------------------------------------

/// The settings that shape the reference run. Effective values, as the
/// context resolved them, except `flash_attn`: llama.cpp does not report
/// what `auto` picked.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
struct Config {
    model: String,
    model_bytes: u64,
    n_ctx: u32,
    n_batch: u32,
    n_ubatch: u32,
    flash_attn: String,
    op_offload: bool,
    n_gpu_layers: i32,
    threads: i32,
    split: usize,
    continue_tokens: usize,
    seed: u64,
    prompt_tokens: usize,
}

/// The reference run, for comparing across processes and devices.
#[derive(Serialize, Deserialize)]
struct Dump {
    /// Missing from dumps written before settings were recorded.
    #[serde(default)]
    config: Option<Config>,
    prompt: Vec<Token>,
    /// Final logits as raw bits, so the round trip is exact.
    logits_bits: Vec<u32>,
    continuation: Vec<Token>,
}

impl Dump {
    fn new(config: &Config, prompt: &Prompt, run: &Run) -> Self {
        Self {
            config: Some(config.clone()),
            prompt: prompt.tokens.clone(),
            logits_bits: run.logits.iter().map(|l| l.to_bits()).collect(),
            continuation: run.tokens.clone(),
        }
    }

    fn write(&self, path: &Path) -> Result<()> {
        std::fs::write(path, serde_json::to_vec(self)?)?;
        Ok(())
    }
}

/// What a settings mismatch against a `--dump` file means.
#[derive(Clone, Copy, PartialEq)]
enum Stake {
    /// A different model or prompt: the comparison means nothing.
    Fatal,
    /// Changes the reference run's schedule or context: refused unless
    /// `--compare-anyway`.
    Refuse,
    /// The device axis the comparison exists to measure.
    Axis,
    /// Does not shape the reference run's final logits.
    Note,
}

/// Check this run's settings against the file's, print them side by
/// side, and refuse a comparison that would mean nothing.
fn check_config(
    ours: &Config,
    theirs: &Config,
    path: &Path,
    anyway: bool,
) -> Result<()> {
    let rows: [(&str, String, String, Stake); 13] = [
        ("model", name(&ours.model), name(&theirs.model), Stake::Note),
        (
            "model bytes",
            ours.model_bytes.to_string(),
            theirs.model_bytes.to_string(),
            Stake::Fatal,
        ),
        (
            "seed",
            ours.seed.to_string(),
            theirs.seed.to_string(),
            Stake::Fatal,
        ),
        (
            "prompt tokens",
            ours.prompt_tokens.to_string(),
            theirs.prompt_tokens.to_string(),
            Stake::Fatal,
        ),
        (
            "n_ctx",
            ours.n_ctx.to_string(),
            theirs.n_ctx.to_string(),
            Stake::Refuse,
        ),
        (
            "n_batch",
            ours.n_batch.to_string(),
            theirs.n_batch.to_string(),
            Stake::Refuse,
        ),
        (
            "n_ubatch",
            ours.n_ubatch.to_string(),
            theirs.n_ubatch.to_string(),
            Stake::Refuse,
        ),
        (
            "flash_attn",
            ours.flash_attn.clone(),
            theirs.flash_attn.clone(),
            Stake::Axis,
        ),
        (
            "op_offload",
            ours.op_offload.to_string(),
            theirs.op_offload.to_string(),
            Stake::Axis,
        ),
        (
            "n_gpu_layers",
            ours.n_gpu_layers.to_string(),
            theirs.n_gpu_layers.to_string(),
            Stake::Axis,
        ),
        (
            "threads",
            ours.threads.to_string(),
            theirs.threads.to_string(),
            Stake::Axis,
        ),
        (
            "split",
            ours.split.to_string(),
            theirs.split.to_string(),
            Stake::Note,
        ),
        (
            "continue tokens",
            ours.continue_tokens.to_string(),
            theirs.continue_tokens.to_string(),
            Stake::Note,
        ),
    ];

    println!("settings     : this run vs {}", path.display());
    for (label, a, b, stake) in &rows {
        let status = match (a == b, stake) {
            (true, _) => "",
            (false, Stake::Fatal) => "!! MISMATCH — different model/prompt",
            (false, Stake::Refuse) => "!! MISMATCH — changes the reference",
            (false, Stake::Axis) => "compared axis",
            (false, Stake::Note) => "differs (does not shape the reference)",
        };
        println!("  {label:<16} {a:>22} {b:>22}  {status}");
    }
    if ours.model != theirs.model {
        println!("  model paths: {} vs {}", ours.model, theirs.model);
    }

    let mismatched = |want: Stake| -> Vec<&str> {
        rows.iter()
            .filter(|(_, a, b, stake)| *stake == want && a != b)
            .map(|(label, ..)| *label)
            .collect()
    };
    let fatal = mismatched(Stake::Fatal);
    if !fatal.is_empty() {
        return Err(format!(
            "{} was made with a different {}; use the same model, --seed \
             and --prompt-tokens",
            path.display(),
            fatal.join(", "),
        )
        .into());
    }
    let refused = mismatched(Stake::Refuse);
    if !refused.is_empty() {
        let which = refused.join(", ");
        if !anyway {
            return Err(format!(
                "{} was made with a different {which}, which changes the \
                 reference run itself; match them, or pass --compare-anyway",
                path.display(),
            )
            .into());
        }
        println!(
            "WARNING: comparing across a different {which} (--compare-anyway) \
             — a difference below mixes that change with the device axis"
        );
    }
    println!();
    Ok(())
}

/// A path's file name, for a settings table.
fn name(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map_or(path.into(), |n| n.to_string_lossy().into_owned())
}

fn compare_file(
    rig: &Rig,
    config: &Config,
    path: &Path,
    prompt: &Prompt,
    reference: &Run,
    args: &Args,
) -> Result<()> {
    let dump: Dump = serde_json::from_slice(&std::fs::read(path)?)?;
    match &dump.config {
        Some(theirs) => {
            check_config(config, theirs, path, args.compare_anyway)?
        }
        None => println!(
            "WARNING: {} predates recorded settings; only the prompt and \
             vocabulary are checked — make sure n_ctx, n_batch and n_ubatch \
             matched\n",
            path.display(),
        ),
    }
    if dump.prompt != prompt.tokens {
        return Err(format!(
            "{} holds a different prompt ({} tokens vs {}); use the same \
             model, --seed and --prompt-tokens",
            path.display(),
            dump.prompt.len(),
            prompt.tokens.len(),
        )
        .into());
    }
    if dump.logits_bits.len() != reference.logits.len() {
        return Err(format!(
            "{} holds {} logits, this model {}: a different vocabulary",
            path.display(),
            dump.logits_bits.len(),
            reference.logits.len(),
        )
        .into());
    }

    // Continuations of different lengths compare over the shorter one.
    let n = reference.tokens.len().min(dump.continuation.len());
    if reference.tokens.len() != dump.continuation.len() {
        println!(
            "note: continuations differ in length (this run {}, file {}); \
             comparing the first {n} tokens",
            reference.tokens.len(),
            dump.continuation.len(),
        );
    }
    let ours = Run {
        logits: reference.logits.clone(),
        tokens: reference.tokens[..n].to_vec(),
        steps: Vec::new(),
    };
    let theirs = Run {
        logits: dump
            .logits_bits
            .iter()
            .map(|&b| f32::from_bits(b))
            .collect(),
        tokens: dump.continuation[..n].to_vec(),
        steps: Vec::new(),
    };
    let mut table = Table::new(
        &rig.model,
        "FILE: this process's reference against a dumped one",
        "this process's reference",
    );
    let label = match &dump.config {
        Some(c) => format!(
            "{} (ngl {}, op_offload {}, fa {})",
            name(&path.display().to_string()),
            c.n_gpu_layers,
            c.op_offload,
            c.flash_attn,
        ),
        None => name(&path.display().to_string()),
    };
    table.row(&label, Cmp::new(&ours, &theirs), Instant::now());
    let runs = [light(&label, &theirs)];
    table.verdict(
        &ours,
        &runs,
        "bit-identical across processes/devices",
        "the two processes disagree — expected between Metal and CPU; \
         between two runs on the same device and settings it is \
         process-level nondeterminism upstream",
    );
    Ok(())
}

// ---------------------------------------------------------------------
// The prompt
// ---------------------------------------------------------------------

/// A seeded, raw-text prompt: an instruction prefix (the breakpoint
/// split), a long log, and a question whose answer is fluent prose.
struct Prompt {
    text: String,
    tokens: Vec<Token>,
    /// Tokens the instruction prefix covers.
    split: usize,
}

const PREAMBLE: &str = "You are the archivist of the Harbor Street \
Observatory. You read the station log below and answer questions about it \
accurately, in complete sentences, citing entry numbers.\n\nStanding \
instructions:\n";
const RULES: &[&str] = &[
    "Always cite the entry number when you mention an event.",
    "If two entries conflict, prefer the later one.",
    "Do not speculate about anything the log does not record.",
    "Give times in the twenty-four hour format used by the log.",
    "Name every person involved in an event you describe.",
    "Summaries should follow the order of the log.",
];
const NAMES: &[&str] = &[
    "Ada", "Bram", "Cleo", "Dmitri", "Esme", "Farid", "Greta", "Hiro", "Ines",
    "Jonah", "Kavya", "Luca",
];
const VERBS: &[&str] = &[
    "recalibrated",
    "inspected",
    "logged",
    "replaced",
    "cleaned",
    "photographed",
    "measured",
    "repaired",
    "tested",
    "moved",
];
const ADJS: &[&str] = &[
    "northern",
    "cracked",
    "brass",
    "backup",
    "primary",
    "humming",
    "frosted",
    "spare",
    "old",
    "newly installed",
];
const NOUNS: &[&str] = &[
    "telescope mount",
    "spectrograph",
    "dome shutter",
    "weather vane",
    "star chart",
    "cooling pump",
    "focuser",
    "clock",
    "filter wheel",
    "antenna",
];
const PLACES: &[&str] = &[
    "east gallery",
    "control room",
    "basement",
    "roof deck",
    "library",
    "workshop",
    "west stair",
    "loading bay",
];
const CLAUSES: &[&str] = &[
    "the humidity had risen overnight",
    "a bearing was unusually warm",
    "the seeing was excellent",
    "a gull had nested nearby",
    "the readings drifted by two percent",
    "nothing seemed out of place",
    "the power flickered twice",
    "a note from the previous shift was missing",
];
const DAYS: &[&str] = &[
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
];

/// SplitMix64: tiny, seedable, and stable across crate versions, so a
/// seed names the same prompt forever.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn pick<'a>(&mut self, xs: &[&'a str]) -> &'a str {
        xs[self.below(xs.len())]
    }
}

impl Prompt {
    fn build(model: &LlamaCppModel, seed: u64, target: usize) -> Self {
        let mut rng = SplitMix(seed);
        let tokenize = |text: &str| model.tokenize(text, false);

        let mut prefix = PREAMBLE.to_string();
        let mut rule = 1;
        for text in RULES {
            prefix.push_str(&format!("{rule}. {text}\n"));
            rule += 1;
        }
        while tokenize(&prefix).len() < target / 6 {
            prefix.push_str(&format!(
                "{rule}. Treat any mention of the {} {} in the {} as \
                 referring to the one {} maintains.\n",
                rng.pick(ADJS),
                rng.pick(NOUNS),
                rng.pick(PLACES),
                rng.pick(NAMES),
            ));
            rule += 1;
        }
        prefix.push_str("\nStation log:\n\n");

        let subject = rng.pick(NAMES);
        let question = format!(
            "\nQuestion: Using the log above, describe in detail everything \
             {subject} did, in the order it happened, and explain what the \
             staff should check next.\nAnswer:"
        );
        let mut body = String::new();
        let mut entry = 1;
        loop {
            for _ in 0..8 {
                body.push_str(&format!(
                    "Entry {entry}: At {:02}:{:02} on {}, {} {} the {} {} in \
                     the {}, noting that {}.\n",
                    rng.below(24),
                    rng.below(60),
                    rng.pick(DAYS),
                    rng.pick(NAMES),
                    rng.pick(VERBS),
                    rng.pick(ADJS),
                    rng.pick(NOUNS),
                    rng.pick(PLACES),
                    rng.pick(CLAUSES),
                ));
                entry += 1;
            }
            if tokenize(&format!("{prefix}{body}{question}")).len() >= target {
                break;
            }
        }

        let text = format!("{prefix}{body}{question}");
        let tokens = tokenize(&text);
        let split = tokenize(&prefix)
            .iter()
            .zip(&tokens)
            .take_while(|(a, b)| a == b)
            .count();
        Self {
            text,
            tokens,
            split,
        }
    }
}

/// The last `n` characters of `s`.
fn tail(s: &str, n: usize) -> String {
    let skip = s.chars().count().saturating_sub(n);
    s.chars().skip(skip).collect()
}
