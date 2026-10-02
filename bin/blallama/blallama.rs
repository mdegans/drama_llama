//! Demo Anthropic-compatible `/v1/messages` server over a local model.
//!
//! # Why a turn sometimes re-prefills
//!
//! Throughput here is dominated by the prefix cache, and the prefix cache
//! is keyed on the *rendered* prompt. Each finished assistant turn is
//! parsed into blocks, and the next request re-renders those blocks back
//! into the prompt. The cache survives only when that round-trip is
//! byte-exact — `render(parse(raw)) == raw`. When it isn't, every cached
//! token past the divergence is unusable and the turn re-prefills.
//!
//! Nothing is corrupted when that happens; the session falls back to a
//! token-id longest-common-prefix walk. It is purely a cost, but a large
//! one: the lost breakpoint frequently sits on system+tools, usually the
//! biggest part of the prompt, so one mismatch can cost minutes on a long
//! conversation. A sudden collapse in cache-read counts on an otherwise
//! healthy conversation is nearly always this.
//!
//! Prose always round-trips. Complete reasoning blocks and complete tool
//! calls round-trip by construction — they are emitted under a grammar
//! that forces the shape the template re-renders, so even *garbage*
//! arguments survive intact as long as the structure closed.
//!
//! The one thing that cannot round-trip is a **structure the model never
//! finished**, because generation can always run out of budget
//! mid-emission. A truncated *thought* is fine: thoughts are text, so an
//! unclosed one is representable, renders without its close marker, and
//! the next request continues it. A truncated *tool call* is not: its
//! arguments are a JSON value, and half an object has no representation.
//! Such a turn comes back as Anthropic's does — `stop_reason: max_tokens`
//! (or `stop_sequence`) with the call cut short (see below) — and the call
//! re-renders closed, which its KV never was. It never round-trips, and no
//! amount of future work changes that.
//!
//! Practical consequence for clients: put **two cache breakpoints at the
//! end of the prompt** rather than one. A mismatch then costs a single
//! re-prefilled message instead of everything back to the previous
//! structural boundary.
//!
//! # Automatic caching, as on Anthropic
//!
//! A request-level `cache_control` (Anthropic's automatic caching) places
//! a breakpoint after the last cacheable block, with its TTL, and counts
//! toward the four-marker limit — a fifth marker, the automatic one
//! included, is Anthropic's 400, as is an automatic TTL that disagrees
//! with the markers ([`validate_prompt`]). Each request then reads back
//! the previous one's prompt (lookback, which reads anchors Anthropic's
//! would not: theirs walks back at most 20 blocks, ours reads any anchor
//! the previous call placed — though Anthropic can also hit an older
//! call's entry within those 20 blocks, which the slot no longer keeps)
//! and, beyond parity, the previous turn's generated KV when the turn
//! round-trips (the tip).
//! So an explicit marker on the system plus automatic caching — the
//! combination Anthropic recommends for agent loops — is also the right
//! setup here: a turn that does not round-trip costs only itself.
//!
//! # Seeing a cache miss
//!
//! Every request logs one `cache_reuse` event (`hit` with its source at
//! `DEBUG`, or `miss` with its reason), and every loss gets a
//! `cache_degrade` or `cache_evict` event naming why — for a lost tip,
//! the entry where the new prompt diverges from the cached tokens and
//! the text on both sides. Losses past a few hundred tokens are logged
//! at `WARN`, the rest at `INFO`. The targets are
//! `drama_llama::session` and `drama_llama::snapshot_store`, so
//! `RUST_LOG=info,drama_llama::session=debug` adds the hits.
//!
//! # A `max_tokens` turn can carry complete calls
//!
//! As on Anthropic, a turn the budget cut short is a 200 with
//! `stop_reason: max_tokens` — and the calls that closed before the cut
//! are in it, as `tool_use` blocks. Clients must gate dispatch on
//! `stop_reason: tool_use`, never on the presence of a `tool_use` block.
//!
//! One cut turn is not answered that way: one that had already repeated
//! a call verbatim (same tool, same input) — a model looping identical
//! calls to the budget, the loop the old grammar-violation check caught
//! (plan Phase G). blallama resamples it on the warm cache, like the
//! other unlucky draws, and answers `max_tokens` only if every draw
//! loops. Anthropic has no such loop to guard against; here it is
//! better than parity.
//!
//! # A cut call comes back cut, as on Anthropic
//!
//! When a turn is cut *inside* a call, Anthropic still returns that call,
//! and so does blallama — the same prompt drives a client the same way on
//! both. Captured 2026-09-30 on claude-haiku-4-5, raw bytes, a `write_file`
//! requiring `path` and `contents` (misanthropic's
//! `misanthropic/test/data/stop/`):
//!
//! - **`max_tokens` mid-input** (`clip*.*`): `stop_reason: max_tokens`,
//!   and the input holds only the members that *completed* — the one
//!   being generated is dropped whole, however far it got
//!   (`{"path":"hello.py"}`; 140 output tokens into a 200-word
//!   `contents`, still `{"path":"story.txt"}`). Streamed, the
//!   `input_json_delta` chunks stop at the last completed member and the
//!   block never gets `content_block_stop`
//!   ([`drama_llama::BlockStream::open_call_json`]).
//! - **A stop sequence matched in the input** (`stop_sequence_tool.*`):
//!   `stop_reason: stop_sequence`, `stop_sequence: "print("`, and a
//!   `tool_use` whose string is cut right before the match, the JSON
//!   closed — `{"path":"hello.py","contents":"import datetime\n"}`.
//!   Generation stops at the match. Streamed, the block ends with
//!   `content_block_stop` like any other; under `tool_choice: auto`, the
//!   same after a text block.
//!
//! Either call looks complete and isn't, which is why clients gate
//! dispatch on `stop_reason` (above). Nested containers keep their
//! completed members at every depth — inferred; only the top level is
//! captured. A call cut before its *name* is whole has nothing to return
//! (Anthropic's block carries its name whole) and is left out.
//!
//! # Run it under a supervisor
//!
//! blallama exits on purpose when it can no longer trust its own
//! process, and expects to be restarted:
//!
//! - **a panic**, on any thread — exit code **70** (`EX_SOFTWARE`);
//! - **a backend failure** llama.cpp does not recover from in-process —
//!   a failed `llama_decode`, a Metal context left "in error state …
//!   recreate the backend" by an out-of-memory command buffer, or a
//!   model load that fails after the backend began allocating (out of
//!   memory loading the weights or creating the KV cache, or any load
//!   failure llama.cpp leaves unexplained) — exit code **75**
//!   (`EX_TEMPFAIL`).
//!
//! A load that fails *before* anything is allocated — a missing or
//! unreadable file, metadata llama.cpp cannot read (not a GGUF, an
//! unsupported architecture), a bad template — is answered with an
//! error and the server serves on.
//!
//! Recovering in-process would mean unwinding through, or dropping,
//! llama.cpp state that failed mid-operation, and llama.cpp does not
//! promise its destructors clean that up; serving on is worse (one
//! Metal OOM on 2026-10-01 failed every later request until a manual
//! restart). So the process logs one `ERROR` line (`event: "fatal"`,
//! with `kind`, `exit_code` and `cause`), answers the requests in
//! flight and any that arrive meanwhile with a 500 `api_error` (which
//! the SDKs retry), and `_exit`s half a second later — skipping
//! llama.cpp's static destructors, which on Metal would turn the exit
//! into a `SIGABRT`. See the `fatal` module.
//!
//! Run it under launchd, systemd (`Restart=on-failure`), or the restart
//! loop in `scripts/blallama-supervise.sh`, which keeps the arguments,
//! backs off when it crash-loops, and logs each restart with its exit
//! code. A clean exit (SIGTERM, Ctrl-C: code 0) is not restarted by any
//! of them.

mod fatal;

use std::{
    num::{NonZeroU128, NonZeroUsize},
    path::PathBuf,
    sync::Arc,
    time::Duration,
};

use axum::{
    extract::{
        rejection::JsonRejection, DefaultBodyLimit, FromRequest, Json,
        Path as UrlPath, Request, State,
    },
    http::StatusCode,
    routing::{get, post},
    Router,
};
use clap::Parser;
use drama_llama::{
    backend::{Backend, Model},
    cli::{BackendArgs, BackendKind},
    prompt::{AnthropicError, MessageResponse, Usage},
    Catalog, FromPath, ProbeCtx, ProbeHook, Prompt, SchemaLimits, Session,
    SnapshotOpts,
};
use fatal::Fatal;
use misanthropic::{
    model::{ModelInfo, Models},
    response::StopReason,
};
use tokio::{sync::Mutex, task::spawn_blocking};
use tracing::{error, info, instrument};

#[derive(Parser)]
#[command(about = "Demo /v1/messages server")]
struct Args {
    /// Path containing model files (llama.cpp) or model directories (moeflux).
    model_path: PathBuf,
    /// Port to use
    #[arg(long, default_value_t = 11435)]
    port: u16,
    /// Backend selection (`llama-cpp` discovers `.gguf` files; `moeflux`
    /// discovers child directories with the `mlx/`/`artifacts/`/`root/`
    /// convention) plus that backend's load-time knobs. Variants are
    /// cfg-gated — a build with only one backend feature accepts only
    /// that variant.
    ///
    /// `--n-ctx` in particular: this server used to have no way to set
    /// it at all, so every llama.cpp model was served at llama.cpp's
    /// 512-token default and silently truncated anything real. It now
    /// defaults to `cli::DEFAULT_N_CTX`; raise it for long agentic
    /// conversations if the box has the memory.
    #[command(flatten)]
    load: BackendArgs,
    /// Force the repetition-penalty filter OFF, even when the per-model
    /// sampling sidecar enables it. Useful for probes, canary runs, and any
    /// diagnostic where you want to see the model's raw logit gradient with no
    /// penalty applied. Without this flag, sampling configuration comes from
    /// `<model>.sampling.toml` (gguf) or `parent/sampling.toml` (moeflux) —
    /// `Session::from_path*` writes a default sidecar on first load.
    #[arg(long, default_value_t = false)]
    no_penalty: bool,
    /// Optional fixed RNG seed forwarded to every prediction (a "fork" under
    /// the session's resume/fork/fresh trichotomy). Useful for tuning
    /// iteration: same prompt + same seed = same output, so a sidecar tweak
    /// shows up as a deliberate divergence rather than a stochastic one. Omit
    /// for the default: resume a cached stream on a hit, fresh entropy
    /// otherwise.
    #[arg(long)]
    seed: Option<u128>,
    /// Serve this model when a request names one that isn't on disk. Lets
    /// unmodified Anthropic-SDK clients (which default to `claude-*` ids) run
    /// against this server without per-client model configuration. Must name a
    /// discoverable model; unknown requested models still 404 when this is
    /// unset.
    #[arg(long)]
    default_model: Option<String>,
    /// Append per-token probe records to this JSONL file. One
    /// `{"event":"session_start","model":"…"}` line per `/v1/messages` request,
    /// then one `{"event":"probe_ctx","ts_ms":T,"ctx":{…}}` line per yielded
    /// token — where `ctx` is the full serialized `ProbeCtx` (same schema as
    /// the `ctx` field on `--probe-stream` `token` events: sampled token,
    /// `n_cur`, `snapshot` top-K + entropy when available, etc.). `ts_ms` is
    /// relative to the moment the recorder was installed for that request. Omit
    /// to disable JSONL recording.
    ///
    /// Composes with `--probe-stream`: both recorders see every token once via
    /// a `FanOutHook`. The JSONL recorder requests the same snapshot budget as
    /// the streaming recorder (`top_k=100`, `p_threshold=0`,
    /// `compute_entropy=true`); when both are active the snapshot is captured
    /// once and shared.
    #[arg(long)]
    record_json: Option<PathBuf>,
    /// Mount the `/probe` SSE endpoint and install a per-request streaming
    /// recorder. Consumers connect once with `GET /probe` and receive
    /// `session_start` / `token` / `session_end` events for every request the
    /// server handles, tagged by the request's UUID (also returned as
    /// `Message::id` on the sync response). Late connectors miss early events;
    /// convention is to open `/probe` before sending `/v1/messages`.
    #[arg(long, default_value_t = false)]
    probe_stream: bool,
    /// Limits on a request's tool and `output_config` schemas.
    #[command(flatten)]
    schema_limits: SchemaLimitArgs,
}

/// The most a request's client-supplied schemas — every custom tool's
/// `input_schema`, an `output_config` `json_schema` — may measure,
/// checked before anything compiles them. A request past one is a 400
/// `invalid_request_error` naming it. The defaults are
/// `drama_llama::SchemaLimits::default()`, generous for real tools
/// (Agora's are 15 tools of at most 5 parameters, ~400 schema nodes);
/// raise one if a legitimate request trips it.
#[derive(clap::Args)]
struct SchemaLimitArgs {
    /// Custom tools in one request.
    #[arg(long, default_value_t = SchemaLimits::default().max_tools)]
    schema_max_tools: usize,
    /// Top-level properties of one tool's `input_schema`.
    #[arg(long, default_value_t = SchemaLimits::default().max_params)]
    schema_max_params: usize,
    /// JSON values across all of a request's schemas.
    #[arg(long, default_value_t = SchemaLimits::default().max_nodes)]
    schema_max_nodes: usize,
    /// `$defs` (and `definitions`) entries of one schema.
    #[arg(long, default_value_t = SchemaLimits::default().max_defs)]
    schema_max_defs: usize,
    /// Bytes of one `enum` member or `const` value, as compact JSON.
    #[arg(long, default_value_t = SchemaLimits::default().max_member_bytes)]
    schema_max_member_bytes: usize,
    /// Bytes of every `enum` member and `const` value across a request,
    /// each `$ref` counted at its target's size once per reference.
    #[arg(
        long,
        default_value_t = SchemaLimits::default().max_total_member_bytes
    )]
    schema_max_total_member_bytes: usize,
    /// How many ways one schema's grammar can go on at once (`enum`
    /// members, properties, `anyOf`/`oneOf` variants, nested variants
    /// multiplying); see `SchemaLimits::max_width`.
    #[arg(long, default_value_t = SchemaLimits::default().max_width)]
    schema_max_width: usize,
    /// Levels of objects and arrays a value one schema's grammar writes
    /// may nest, each `$ref` at its target's depth; see
    /// `SchemaLimits::max_depth`. Past 93, a value (an untyped one adds
    /// up to 32 levels, a call envelope two) can nest deeper than the 127
    /// levels the parsers read.
    #[arg(long, default_value_t = SchemaLimits::default().max_depth)]
    schema_max_depth: usize,
}

impl SchemaLimitArgs {
    fn limits(&self) -> SchemaLimits {
        SchemaLimits::default()
            .with_max_tools(self.schema_max_tools)
            .with_max_params(self.schema_max_params)
            .with_max_nodes(self.schema_max_nodes)
            .with_max_defs(self.schema_max_defs)
            .with_max_member_bytes(self.schema_max_member_bytes)
            .with_max_total_member_bytes(self.schema_max_total_member_bytes)
            .with_max_width(self.schema_max_width)
            .with_max_depth(self.schema_max_depth)
    }
}

#[derive(Clone)]
struct AppState<B: Backend>
where
    Session<B>: FromPath,
{
    args: Arc<Args>,
    /// The models under [`Args::model_path`], described as the load
    /// options (narrowed from [`Args::load`] once at startup, so an
    /// inapplicable flag is a refusal to boot rather than a surprise on
    /// the first request) would load them. Owns the `/v1/models`
    /// metadata cache: disk once per model, memory thereafter.
    catalog: Arc<Catalog<B>>,
    /// Sender into the JSONL writer task. `None` if `--record-json` wasn't
    /// given. Cloned per-request when installing the [`JsonlProbeRecorder`];
    /// all clones feed the same writer task / output file.
    record_json_tx: Option<tokio::sync::mpsc::Sender<serde_json::Value>>,
    /// Streaming-probe broadcast bus. `None` if `--probe-stream` wasn't given.
    /// Cloned per-request into a [`StreamingProbeRecorder`] and (separately)
    /// subscribed by the `/probe` SSE handler. The same bus carries
    /// `SessionStart` / `SessionEnd` events emitted directly from the request
    /// handler around the generation call.
    probe_bus: Option<tokio::sync::broadcast::Sender<StreamProbeMsg>>,
    session: Arc<Mutex<Option<Session<B>>>>,
}

// Listing and id resolution used to live here (`list_entries`,
// `resolve_model`); they are `Catalog::list` / `Catalog::resolve` now,
// alongside the metadata cache that `/v1/models` needs.

/// Anthropic wire envelope for errors: `{"type":"error","error":{...}}`.
/// Real clients (misanthropic included) parse errors through this
/// wrapper; serving the bare `AnthropicError` object is unparseable to
/// them. misanthropic's own wrapper is `pub(crate)`
/// (mdegans/misanthropic#134 asks to expose it) — replicated here
/// until then.
#[derive(serde::Serialize)]
struct ErrorEnvelope {
    #[serde(rename = "type")]
    kind: &'static str,
    error: AnthropicError,
}

impl From<AnthropicError> for ErrorEnvelope {
    fn from(error: AnthropicError) -> Self {
        Self {
            kind: "error",
            error,
        }
    }
}

/// An error response in the wire envelope, with the status Anthropic
/// sends for that error type (500 when it has none).
fn error_response(error: AnthropicError) -> (StatusCode, Json<ErrorEnvelope>) {
    let status = error
        .status()
        .and_then(|code| StatusCode::from_u16(code.get()).ok())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    (status, Json(error.into()))
}

/// Anthropic's request-size ceiling for the Messages API. axum's own
/// default is 2 MB, which a single base64 image can exceed.
const MAX_REQUEST_BYTES: usize = 32 * 1024 * 1024;

/// [`Json`] whose rejection is Anthropic's error envelope (#123).
///
/// A body that fails to deserialize used to get axum's own answer — a
/// plain-text 422 — which no Anthropic client parses: the SDKs expect
/// `{"type":"error","error":{…}}` and decide retry vs. give-up on the
/// status. Anthropic answers a malformed body with 400
/// `invalid_request_error` (never retried), and an oversized one with
/// 413 `request_too_large`; so does this.
struct AnthropicJson<T>(T);

impl<S, T> FromRequest<S> for AnthropicJson<T>
where
    Json<T>: FromRequest<S, Rejection = JsonRejection>,
    S: Send + Sync,
{
    type Rejection = (StatusCode, Json<ErrorEnvelope>);

    async fn from_request(
        req: Request,
        state: &S,
    ) -> Result<Self, Self::Rejection> {
        Json::<T>::from_request(req, state)
            .await
            .map(|Json(value)| Self(value))
            .map_err(map_json_rejection)
    }
}

/// See [`AnthropicJson`]. The message is axum's description of what was
/// wrong with the body, which names the offending field.
fn map_json_rejection(
    rejection: JsonRejection,
) -> (StatusCode, Json<ErrorEnvelope>) {
    let message = rejection.body_text();
    error_response(if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE {
        AnthropicError::RequestTooLarge { message }
    } else {
        AnthropicError::InvalidRequest { message }
    })
}

/// A request body as `/v1/messages` and `count_tokens` take it:
/// [`AnthropicJson`], then the checks Anthropic makes on a well-formed
/// body before it reaches a model ([`validate_prompt`]), each a 400.
struct AnthropicPrompt(Prompt);

impl<S: Send + Sync> FromRequest<S> for AnthropicPrompt {
    type Rejection = (StatusCode, Json<ErrorEnvelope>);

    async fn from_request(
        req: Request,
        state: &S,
    ) -> Result<Self, Self::Rejection> {
        let AnthropicJson(prompt) =
            AnthropicJson::<Prompt>::from_request(req, state).await?;
        validate_prompt(&prompt).map_err(error_response)?;
        Ok(Self(prompt))
    }
}

/// Anthropic's exact wording, which clients may match on.
const BLANK_STOP_MESSAGE: &str =
    "stop_sequences: each stop sequence must contain non-whitespace";

/// Anthropic's request checks that deserializing alone lets through.
///
/// A stop sequence with no non-whitespace character (`"\n"`, `" "`) is a
/// 400 `invalid_request_error` on Anthropic — captured 2026-09-30 on
/// claude-haiku-4-5 (misanthropic's
/// `misanthropic/test/data/stop/whitespace_stop.error.json`) for `stream: false` and `stream: true` alike (the
/// streaming request gets the same plain JSON body, not an SSE `error`
/// event). "Whitespace" here is Rust's [`char::is_whitespace`] (Unicode
/// `White_Space`); Anthropic's exact character class is an uncaptured
/// assumption, as is its handling of `""`, which this rejects (vacuously
/// all-whitespace) because the rule as worded covers it. Applied to `count_tokens` too, on the (uncaptured)
/// assumption that Anthropic validates the shared body the same way on
/// both routes.
///
/// The `cache_control` markers get Anthropic's checks too
/// ([`drama_llama::check_cache_controls`]): at most four, the
/// request-level automatic one included, its TTL agreeing with the
/// explicit markers. Those were captured on both routes.
fn validate_prompt(prompt: &Prompt) -> Result<(), AnthropicError> {
    let blank_stop = prompt
        .stop_sequences
        .iter()
        .flatten()
        .any(|stop| stop.chars().all(char::is_whitespace));
    if blank_stop {
        return Err(AnthropicError::InvalidRequest {
            message: BLANK_STOP_MESSAGE.into(),
        });
    }
    drama_llama::check_cache_controls(prompt)
        .map_err(|message| AnthropicError::InvalidRequest { message })
}

/// A route's error answer, in the wire envelope.
type Reply = (StatusCode, Json<ErrorEnvelope>);

/// Run `f` on the blocking pool — every call into llama.cpp goes
/// through here — or answer 500 once the process is declared fatal
/// ([`fatal`]): by a panic in `f` (which parks its thread rather than
/// unwind it, see [`fatal::install`]), or anything else meanwhile. A
/// panic that does unwind into a `JoinError` (one inside
/// [`fatal::caught_by_caller`] that nothing caught) is declared here.
async fn spawn_blocking_or_bust<F, R>(f: F) -> Result<R, Reply>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    tokio::select! {
        joined = spawn_blocking(f) => match joined {
            Ok(r) => Ok(r),
            Err(e) => {
                // Cancelled only at runtime shutdown; a panic otherwise.
                if e.is_panic() {
                    fatal::declare(Fatal::Panic, &e);
                }
                Err(fatal::reply(fatal::current().unwrap_or(Fatal::Panic)))
            }
        },
        fatal = fatal::declared() => Err(fatal::reply(fatal)),
    }
}

/// Run a request's work as its own task, so a client that disconnects
/// drops only this wait, never the work. axum drops a handler's future
/// when its connection closes; were the work inside it, the session's
/// lock guard would drop while the blocking pool still ran the
/// generation (or load) that owns the session, and the next request
/// would find the slot empty and load a second copy of the model — live
/// on 2026-10-02, a client timeout mid-generation on Mistral Small 4
/// followed by its retry exited the server on Metal OOM. Detached, the
/// task keeps the guard to the end and a retry meanwhile gets 529.
async fn detached<F, T>(work: F) -> Result<T, Reply>
where
    F: std::future::Future<Output = Result<T, Reply>> + Send + 'static,
    T: Send + 'static,
{
    match tokio::spawn(work).await {
        Ok(r) => r,
        Err(e) => {
            // Cancelled only at runtime shutdown; a panic otherwise.
            if e.is_panic() {
                fatal::declare(Fatal::Panic, &e);
            }
            Err(fatal::reply(fatal::current().unwrap_or(Fatal::Panic)))
        }
    }
}

fn log_stats(id: impl AsRef<str>, usage: Usage, elapsed: Duration) {
    // `Usage` derefs to `TokenCounts`, where the counts now live.
    let input_tokens = usage.input_tokens;
    let cache_creation_input_tokens = usage.cache_creation_input_tokens;
    let cache_read_input_tokens = usage.cache_read_input_tokens;
    let output_tokens = usage.output_tokens;

    info!(
        event = "stats",
        id = id.as_ref(),
        input_tokens,
        cache_creation_input_tokens,
        cache_read_input_tokens,
        output_tokens,
        elapsed_ms = elapsed.as_millis() as u64,
        tok_per_sec = output_tokens as f64 / elapsed.as_secs_f64()
    );
}

// There was a `log_moeflux_prefetch` here that formatted
// `MoefluxDecoder::prefetch_stats()` into a tracing event. It was never
// called — not once in its life — and only surfaced as dead code when the
// permutation gate started building the moeflux configurations with
// `--all-targets` (#68). Removed rather than wired up (#69): the counters
// live in moeflux, so the emitter belongs there too, instead of every
// consumer hand-rolling one. `prefetch_stats()` is still public, so
// nothing is lost but the formatting.

// Credit To Claude Opus 4.7 for this
fn init_logging() {
    use tracing_subscriber::{fmt, prelude::*, EnvFilter, Registry};

    // EnvFilter reads RUST_LOG. Falls back to "info" if unset. Syntax:
    // RUST_LOG=info,drama_llama=debug,axum=warn
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    // JSON formatter for structured output (downstream-parseable). Span context
    // flags control what span info rides on each event.
    let fmt_layer = fmt::layer()
        .json()
        .with_current_span(true) // include the active span on each event
        .with_span_list(false) // skip the full span stack (noisy)
        .with_target(true) // module path
        .with_file(true)
        .with_line_number(true)
        .with_thread_ids(true);

    Registry::default().with(filter).with(fmt_layer).init();
}

/// The 404 for a model id that isn't on disk, in the wire envelope.
fn model_not_found(id: &str) -> (StatusCode, Json<ErrorEnvelope>) {
    (
        StatusCode::NOT_FOUND,
        Json(
            AnthropicError::NotFound {
                message: format!("model not found: {id}"),
            }
            .into(),
        ),
    )
}

/// `GET /v1/models` — every model on disk, loaded or not, as Anthropic's
/// `{"data": [ModelInfo, …]}`. The first call per model reads its
/// metadata off disk (a vocab-only load for llama.cpp — no weights, no
/// GPU); later calls are served from the [`Catalog`]'s cache until the
/// file changes. Runs on the blocking pool: the peek is I/O.
async fn route_models<B>(
    State(state): State<AppState<B>>,
) -> Result<Json<Models>, Reply>
where
    B: Backend + 'static,
    Session<B>: FromPath,
{
    let catalog = state.catalog.clone();
    let read = move || fatal::caught_by_caller(|| catalog.models());
    spawn_blocking_or_bust(read).await.map(Json)
}

/// `GET /v1/models/{id}` — one model's [`ModelInfo`], or the same 404
/// `/v1/messages` gives for an unknown id. `--default-model` is *not*
/// substituted here: a listing that answers `claude-opus-5` with a GGUF
/// would be lying to a client that is about to trust it.
async fn route_model<B>(
    State(state): State<AppState<B>>,
    UrlPath(id): UrlPath<String>,
) -> Result<Json<ModelInfo>, (StatusCode, Json<ErrorEnvelope>)>
where
    B: Backend + 'static,
    Session<B>: FromPath,
{
    let catalog = state.catalog.clone();
    let name = id.clone();
    let read = move || fatal::caught_by_caller(|| catalog.info(&name));
    spawn_blocking_or_bust(read)
        .await?
        .map(Json)
        .ok_or_else(|| model_not_found(&id))
}

/// `GET /api/tags` — the ollama-shaped listing, derived from the same
/// [`ModelInfo`]s as `/v1/models` so the two never disagree on what's
/// there. Kept for clients that discover models the ollama way; the
/// `details` block is filled as far as the catalog knows (llama.cpp's
/// own server leaves the same fields blank).
async fn route_tags<B>(
    State(state): State<AppState<B>>,
) -> Result<Json<serde_json::Value>, Reply>
where
    B: Backend + 'static,
    Session<B>: FromPath,
{
    let catalog = state.catalog.clone();
    let models = spawn_blocking_or_bust(move || {
        fatal::caught_by_caller(|| catalog.models())
            .into_iter()
            .map(|info| {
                let name = info.id.name().to_string();
                serde_json::json!({
                    "name": name,
                    "model": name,
                    "modified_at": info.created_at.to_rfc3339(),
                    "size": catalog.size(&name).unwrap_or(0),
                    "digest": "",
                    "details": {
                        "format": if B::NAME == "llama-cpp" { "gguf" } else { B::NAME },
                        "family": "",
                        "families": [],
                        "parameter_size": "",
                        "quantization_level": ""
                    }
                })
            })
            .collect::<Vec<_>>()
    })
    .await?;
    Ok(Json(serde_json::json!({ "models": models })))
}

async fn run<B>(
    args: Args,
    load_options: <Session<B> as FromPath>::Options,
    record_json_tx: Option<tokio::sync::mpsc::Sender<serde_json::Value>>,
    probe_bus: Option<tokio::sync::broadcast::Sender<StreamProbeMsg>>,
) -> Result<(), Box<dyn std::error::Error>>
where
    B: Backend + 'static,
    AppState<B>: Clone,
    Session<B>: FromPath,
{
    let listener = tokio::net::TcpListener::bind(format!(
        "0.0.0.0:{port}",
        port = args.port
    ))
    .await?;

    let session: Arc<Mutex<Option<Session<B>>>> = Mutex::from(None).into();
    let catalog = Arc::new(Catalog::<B>::new(&args.model_path, load_options));

    // Warm the `/v1/models` cache off the request path: a cold listing
    // reads every model's metadata (a second or two each for llama.cpp),
    // and the first client shouldn't be the one to pay for it. Requests
    // are served meanwhile — cached entries never wait on a read in
    // flight, and `/v1/messages` only needs the directory listing.
    {
        let catalog = catalog.clone();
        tokio::spawn(spawn_blocking_or_bust(move || {
            let started = std::time::Instant::now();
            let n = fatal::caught_by_caller(|| catalog.models()).len();
            info!(
                event = "catalog_warm",
                models = n,
                elapsed_ms = started.elapsed().as_millis() as u64,
                "model catalog warmed",
            );
        }));
    }

    let mut app = Router::new()
        .route("/v1/messages", post(route_messages))
        .route("/v1/messages/count_tokens", post(route_count_tokens))
        .route("/v1/models", get(route_models))
        .route("/v1/models/{id}", get(route_model))
        .route("/api/tags", get(route_tags))
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BYTES));
    if probe_bus.is_some() {
        app = app.route("/probe", axum::routing::get(route_probe_stream));
    }
    // Last, so it covers every route: once the process is declared
    // fatal, nothing more is served from it.
    let app = app.layer(axum::middleware::from_fn(fatal::refuse_when_fatal));
    let app = app.with_state(AppState {
        args: args.into(),
        catalog,
        record_json_tx,
        probe_bus,
        session,
    });
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    // #95 stage logging: if shutdown wedges, the last line present names
    // the stuck stage. "draining" (from `shutdown_signal`) but no
    // "drained" = hyper waiting on an open connection (`/probe` SSE
    // never completes its response). "drained" but the process lives =
    // runtime teardown waiting on the blocking pool (in-flight
    // generation; policy is finish-then-exit).
    tracing::info!("drained — all connections closed; dropping runtime");
    Ok(())
}

/// Resolve on SIGTERM or Ctrl-C, so `main` returns normally instead of the
/// process being torn down mid-flight.
///
/// Correct for a server on its own terms — Docker and systemd both stop a
/// container with SIGTERM, and until this existed blallama died uncleanly
/// under both, dropping in-flight requests rather than draining them.
///
/// It is also load-bearing for **coverage**. The LLVM profiling runtime
/// writes its `.profraw` from an `atexit` handler, and `atexit` does not run
/// for a signal-killed process. `tests/blallama.rs` used to `Child::kill()`
/// (SIGKILL, uncatchable), so everything those tests drove through this
/// binary — a full `/v1/messages` completion with cache reuse — was invisible
/// to `cargo llvm-cov`. Measurably: blallama's coverage was byte-identical
/// (17.64%) whether the integration tier ran or was skipped entirely.
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{signal, SignalKind};
        match signal(SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            // Nothing useful to do if the handler won't install; fall back
            // to Ctrl-C alone rather than resolving immediately, which
            // would shut the server down the instant it started.
            Err(e) => {
                tracing::warn!("could not install SIGTERM handler: {e}");
                std::future::pending::<()>().await
            }
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => tracing::info!("SIGINT — draining"),
        _ = terminate => tracing::info!("SIGTERM — draining"),
    }
}

/// Load `model` from the catalog's directory with its options, and tell
/// the catalog what the loaded session advertises — the decoder's real
/// context size beats the peek's estimate, and the entry is now known
/// to load.
async fn load_session<B>(
    catalog: Arc<Catalog<B>>,
    model: String,
    no_penalty: bool,
    seed: Option<u128>,
    schema_limits: SchemaLimits,
) -> Result<Session<B>, (StatusCode, Json<ErrorEnvelope>)>
where
    B: Backend + 'static,
    Session<B>: FromPath,
{
    let path = catalog.path_of(&model);
    tracing::info!(
        event = "load_model",
        backend = B::NAME,
        model,
        path = path.to_string_lossy().as_ref()
    );
    let session =
        load_with(catalog, model, path, Session::<B>::from_path_with).await?;
    Ok(configure_session(session, no_penalty, seed, schema_limits))
}

/// [`load_session`]'s load, with the loader passed in so a test can
/// stand in for a load that fails. A failure that may have left the
/// backend half-allocated ([`SessionError::is_resource`]: out of memory
/// loading the weights or creating the KV cache) is fatal, like any
/// other backend failure; one found before anything was allocated (a
/// missing file, unreadable metadata, a bad template) is answered and
/// the server serves on.
///
/// [`SessionError::is_resource`]: drama_llama::SessionError::is_resource
async fn load_with<B, L>(
    catalog: Arc<Catalog<B>>,
    model: String,
    path: PathBuf,
    load: L,
) -> Result<Session<B>, Reply>
where
    B: Backend + 'static,
    Session<B>: FromPath,
    L: FnOnce(
            PathBuf,
            <Session<B> as FromPath>::Options,
        ) -> Result<Session<B>, drama_llama::SessionError>
        + Send
        + 'static,
{
    // On the blocking pool: loading is seconds of blocking file and GPU
    // work and this is a reactor thread.
    spawn_blocking_or_bust(move || {
        fatal::caught_by_caller(|| {
            let session = load(path, catalog.options().clone())?;
            catalog.refresh(&model, session.model_info());
            Ok(session)
        })
    })
    .await?
    .map_err(|e: drama_llama::SessionError| match e.is_resource() {
        true => {
            fatal::declare(Fatal::Backend, &e);
            fatal::reply(Fatal::Backend)
        }
        false => {
            error!(event = "load_failed", error = %e);
            map_session_err(e)
        }
    })
}

async fn route_messages<B>(
    State(state): State<AppState<B>>,
    AnthropicPrompt(mut prompt): AnthropicPrompt,
) -> Result<Json<MessageResponse>, (StatusCode, Json<ErrorEnvelope>)>
where
    B: Backend + 'static,
    Session<B>: FromPath,
{
    resolve_model(&state, &mut prompt).await?;
    detached(complete(state, prompt)).await
}

/// `POST /v1/messages/count_tokens`: what `/v1/messages` would prefill
/// for this body — chat template, tools and thinking scaffold included —
/// as Anthropic's `{"input_tokens": N}`. The body is a `/v1/messages`
/// body without `max_tokens`. Counting needs the model's tokenizer and
/// template, so it loads the model like a completion would (and answers
/// 529 while a generation holds the session).
#[instrument(skip(state, prompt), fields(model = %prompt.model))]
async fn route_count_tokens<B>(
    State(state): State<AppState<B>>,
    AnthropicPrompt(mut prompt): AnthropicPrompt,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorEnvelope>)>
where
    B: Backend + 'static,
    Session<B>: FromPath,
{
    resolve_model(&state, &mut prompt).await?;
    // Detached for the same reason as a completion: a load it triggers
    // must keep the slot until it lands.
    let input_tokens = detached(async move {
        let (mut lock, session) =
            checkout(&state, &prompt.model.to_string()).await?;
        let (session, result) = spawn_blocking_or_bust(move || {
            let mut session = session;
            let result = session.count_tokens(&prompt);
            (session, result)
        })
        .await?;
        // Counting never touches KV state, so the session survives any
        // error it can return.
        lock.replace(session);
        result.map_err(map_session_err)
    })
    .await?;
    Ok(Json(serde_json::json!({ "input_tokens": input_tokens })))
}

/// Resolve `prompt.model` against the catalog, substituting
/// `--default-model` for an unknown id.
async fn resolve_model<B>(
    state: &AppState<B>,
    prompt: &mut Prompt,
) -> Result<(), (StatusCode, Json<ErrorEnvelope>)>
where
    B: Backend + 'static,
    Session<B>: FromPath,
{
    let catalog = state.catalog.clone();
    let requested = prompt.model.to_string();
    let default = state.args.default_model.clone();
    let served = spawn_blocking_or_bust(move || {
        catalog.resolve(&requested, default.as_deref())
    })
    .await?
    .map_err(|e| {
        error!(error = %e);
        (StatusCode::NOT_FOUND, Json(e.into()))
    })?;
    if prompt.model != served {
        info!(
            requested = %prompt.model,
            served = %served,
            "substituting --default-model for unknown id",
        );
        prompt.model = served.into();
    }
    Ok(())
}

/// Take the session out of its lock, loaded with `model`: the resident
/// session when it already serves `model`, else a fresh load. The caller
/// owns both and puts the session back (or drops it, to force a reload)
/// when done. A busy session is Anthropic's 529 `overloaded_error`.
async fn checkout<'s, B>(
    state: &'s AppState<B>,
    model: &str,
) -> Result<
    (tokio::sync::MutexGuard<'s, Option<Session<B>>>, Session<B>),
    (StatusCode, Json<ErrorEnvelope>),
>
where
    B: Backend + 'static,
    Session<B>: FromPath,
{
    let Ok(mut lock) = state.session.try_lock() else {
        return Err((
            StatusCode::from_u16(529).unwrap(),
            Json(
                AnthropicError::Overloaded {
                    message: "Session is busy.".into(),
                    retry_after: None,
                }
                .into(),
            ),
        ));
    };

    if let Some(session) = lock.take() {
        let display =
            session.engine().model().display_name().unwrap_or_default();
        if model == display {
            return Ok((lock, session));
        }
        // Free the outgoing model BEFORE loading the incoming one.
        // Without this the old session stays bound across the `.await`,
        // so a model switch peaks at both models resident — ~38 GB for a
        // pair of 19 GB models, which a 24 GB card does not survive.
        // Nothing is lost by dropping early: on the `?` path below the
        // session is gone either way, having already been `take`n out of
        // the lock.
        drop(session);
    }
    let session = load_session(
        state.catalog.clone(),
        model.to_string(),
        state.args.no_penalty,
        state.args.seed,
        state.args.schema_limits.limits(),
    )
    .await?;
    Ok((lock, session))
}

#[instrument(skip(state, prompt), fields(model = %prompt.model))]
async fn complete<B>(
    state: AppState<B>,
    prompt: Prompt,
) -> Result<Json<MessageResponse>, (StatusCode, Json<ErrorEnvelope>)>
where
    B: Backend + 'static,
    Session<B>: FromPath,
{
    let (mut lock, mut session) =
        checkout(&state, &prompt.model.to_string()).await?;

    // Per-request UUID — same id ends up on `Message.id` and on every
    // `StreamProbeMsg` emitted while this request runs.
    let id = uuid::Uuid::new_v4();
    install_per_request_hooks(
        &mut session,
        state.record_json_tx.as_ref(),
        state.probe_bus.as_ref(),
        id,
    );

    // Emit SessionStart on the bus before generation. SendError means zero
    // subscribers; harmless, ignored. The model name here is the request's
    // `prompt.model` (the user-facing name) rather than the engine's
    // display_name (the GGUF internal name); both are recoverable from the
    // JSONL ts_ms ordering if needed.
    if let Some(bus) = &state.probe_bus {
        let _ = bus.send(StreamProbeMsg::SessionStart {
            id,
            model: prompt.model.to_string(),
        });
    }

    // Closure returns the session in *both* arms so it can be restored to
    // the lock — otherwise a `complete_response` error drops it and the
    // next request reloads from disk. See `is_reusable_after` for the
    // reuse-vs-reload classification.
    //
    // An unlucky draw (`resample_reason`) gets a bounded in-place
    // resample: a fresh draw usually takes a different path. Bounded
    // because a greedy (temperature 0) request reproduces the same
    // emission deterministically — the cap keeps that pathological case
    // at a fixed cost instead of a loop.
    const MAX_RESAMPLES: u32 = 2;
    let (session, result, elapsed, resamples) =
        spawn_blocking_or_bust(move || {
            let start = std::time::Instant::now();
            let mut resamples: u32 = 0;
            let result = loop {
                let result = session.complete_response_id(&prompt, id);
                let Some(reason) = resample_reason(&result)
                    .filter(|_| resamples < MAX_RESAMPLES)
                else {
                    break result;
                };
                resamples += 1;
                // A discarded draw's usage stays in the session's
                // `total_usage` — the work was done, as for any failed
                // call — but the client and `log_stats` see only the
                // draw that is answered.
                match reason {
                    // Pieces logged verbatim: stderr is operator-facing,
                    // never model-visible. The redaction discipline
                    // applies to `Display`, which is relayed to clients.
                    Resample::SpecialToken(found) => error!(
                        attempt = resamples,
                        max = MAX_RESAMPLES,
                        found = ?found,
                        "generation emitted reserved special token(s) in \
                         free text; resampling on the warm cache (#101)",
                    ),
                    Resample::GrammarViolation(e) => error!(
                        attempt = resamples,
                        max = MAX_RESAMPLES,
                        error = %e,
                        "resampling after grammar violation",
                    ),
                    Resample::SchemaViolation(e) => error!(
                        attempt = resamples,
                        max = MAX_RESAMPLES,
                        error = %e,
                        "constrained output broke its schema; resampling \
                         on the warm cache",
                    ),
                    Resample::CallLoop => error!(
                        attempt = resamples,
                        max = MAX_RESAMPLES,
                        "turn cut by max_tokens after repeating a tool \
                         call verbatim; resampling (Phase G loop)",
                    ),
                }
            };
            (session, result, start.elapsed(), resamples)
        })
        .await?;

    if resamples > 0 && result.is_ok() {
        info!(resamples, "resample recovered a clean generation");
    }

    // SessionEnd fires regardless of generation success — the probe stream
    // is a flight recorder, not a control channel.
    if let Some(bus) = &state.probe_bus {
        let _ = bus.send(StreamProbeMsg::SessionEnd { id });
    }

    match &result {
        Ok(_) => {
            lock.replace(session);
        }
        Err(e) if fatal::is_backend_failure(e) => {
            fatal::declare(Fatal::Backend, e);
            // Never through llama.cpp's destructors: the backend just
            // failed, and the process exits in a moment anyway.
            std::mem::forget(session);
            return Err(fatal::reply(Fatal::Backend));
        }
        Err(e) => {
            error!(error = %e);
            lock.replace(session);
        }
    }

    let response = result.map_err(map_session_err)?;
    log_stats(&response.id, response.usage.clone(), elapsed);
    Ok(Json(response))
}

/// Why a draw is resampled on the warm cache instead of answered — each
/// an unlucky path a fresh draw usually escapes — or `None` to answer
/// it. The caller bounds the retries (`MAX_RESAMPLES`).
#[derive(Debug)]
enum Resample<'r> {
    /// Reserved special token(s) in free text (#101). Containment
    /// leaves the prompt's cache extent warm, so the retry re-prefills
    /// nothing.
    SpecialToken(&'r [String]),
    /// An unsatisfied constraint. The session invalidates its own cache
    /// here, so the retry re-prefills; still cheaper than the client's
    /// round trip.
    GrammarViolation(&'r drama_llama::SessionError),
    /// Constrained output that finished but does not match its schema —
    /// the backstop behind the grammar. Never answered with a 200:
    /// resampled warm, and a 500 `api_error` if every draw breaks it.
    SchemaViolation(&'r drama_llama::SessionError),
    /// A cut turn looping identical calls into the budget
    /// ([`loops_a_call`]). Any other cut turn — `max_tokens`, a stop
    /// sequence — is not one of these: it succeeds with that stop
    /// reason, as on Anthropic (#121).
    CallLoop,
}

fn resample_reason(
    result: &Result<MessageResponse, drama_llama::SessionError>,
) -> Option<Resample<'_>> {
    use drama_llama::SessionError;
    match result {
        Err(SessionError::EmittedSpecialToken { found }) => {
            Some(Resample::SpecialToken(found))
        }
        Err(e @ SessionError::GrammarViolation { .. }) => {
            Some(Resample::GrammarViolation(e))
        }
        Err(e @ SessionError::SchemaViolation { .. }) => {
            Some(Resample::SchemaViolation(e))
        }
        Ok(response) if loops_a_call(response) => Some(Resample::CallLoop),
        _ => None,
    }
}

/// The loop signature the Phase G postmortem found: a turn the budget
/// cut (`max_tokens`) that had already repeated one call verbatim — same
/// tool, same input. See the module docs. The call the cut truncated
/// counts too, by the members it completed: equal to an earlier call's
/// input, it was repeating it.
fn loops_a_call(response: &MessageResponse) -> bool {
    let calls: Vec<_> = response
        .inner
        .content
        .0
        .iter()
        .filter_map(|block| match block {
            drama_llama::Block::ToolUse { call } => {
                Some((&call.name, &call.input))
            }
            _ => None,
        })
        .collect();
    response.stop_reason == Some(StopReason::MaxTokens)
        && calls
            .iter()
            .enumerate()
            .any(|(i, call)| calls[..i].contains(call))
}

fn configure_session<B: Backend>(
    s: Session<B>,
    no_penalty: bool,
    seed: Option<u128>,
    schema_limits: SchemaLimits,
) -> Session<B> {
    // Sampling configuration is loaded from the per-model sidecar
    // (`<model>.sampling.toml` for gguf, `parent/sampling.toml` for moeflux)
    // inside `Session::from_path*`. `--no-penalty` overrides the sidecar to
    // force repetition penalty OFF — for probes, canary runs, or any "what does
    // this model do with no penalty" diagnostic.
    let with_penalty = if no_penalty {
        s.without_repetition()
    } else {
        s
    };
    // NOTE: there is no server-side generation ceiling. `prompt.max_tokens`
    // is the sole generation authority (the Session-level cap was removed);
    // a request is honored when prompt + max_tokens fits the context, and
    // one that doesn't is rejected up front (strict fit, below) — we don't
    // babysit a magic ceiling constant that would need bumping as context
    // windows grow.
    let configured = with_penalty
        .with_seed(seed.and_then(NonZeroU128::new))
        .with_prefix_cache(true)
        // An Anthropic-API server answers an overrun `max_tokens` with
        // Anthropic's 400, before prefill, not a silent truncation.
        .with_strict_context_fit(true)
        .with_schema_limits(schema_limits);
    // ProbeHook installation moved to per-request handlers — each /v1/messages
    // request gets a fresh hook bound to its UUID, so the hook can fan out to
    // JSONL, the broadcast bus, or both, with a recorder lifetime that exactly
    // matches the request.
    tracing::info!(
        event = "session_ready",
        n_ctx = configured.engine().n_ctx(),
        no_penalty,
        seed = seed.map(|n| n as u64),
        model = configured
            .engine()
            .model()
            .display_name()
            .unwrap_or_default()
            .as_str(),
    );
    configured
}

/// Default `SnapshotOpts` for the streaming recorder. top_k=100 with
/// p_threshold=0 and entropy=true is the cross-validation suite's working
/// set: refusal-class probes need tail-token visibility (high top_k, no
/// threshold) and entropy is cheap when probes are infrequent. Override via
/// `Args` if/when finer control is needed.
fn default_stream_opts() -> SnapshotOpts {
    SnapshotOpts {
        top_k: NonZeroUsize::new(100).unwrap(),
        p_threshold: 0.0,
        compute_entropy: true,
    }
}

/// Build and install the per-request `FanOutHook` on `session`'s engine.
/// Returns `true` when at least one recorder was installed (so the caller can
/// emit `StreamProbeMsg::SessionStart` / `SessionEnd` only when there's a
/// streaming consumer to receive them).
fn install_per_request_hooks<B: Backend>(
    session: &mut Session<B>,
    record_json_tx: Option<&tokio::sync::mpsc::Sender<serde_json::Value>>,
    probe_bus: Option<&tokio::sync::broadcast::Sender<StreamProbeMsg>>,
    id: uuid::Uuid,
) {
    let mut hooks: Vec<Box<dyn ProbeHook>> = Vec::new();
    if let Some(tx) = record_json_tx {
        let model_name =
            session.engine().model().display_name().unwrap_or_default();
        hooks.push(Box::new(JsonlProbeRecorder::install(
            tx.clone(),
            model_name.as_str(),
            default_stream_opts(),
        )));
    }
    if let Some(bus) = probe_bus {
        hooks.push(Box::new(StreamingProbeRecorder {
            bus: bus.clone(),
            id,
            opts: default_stream_opts(),
        }));
    }
    let hook: Option<Box<dyn ProbeHook>> = match hooks.len() {
        0 => None,
        1 => Some(hooks.pop().unwrap()),
        _ => Some(Box::new(FanOutHook { hooks })),
    };
    session.engine_mut().set_probe_hook(hook);
}

// ---------------------------------------------------------------------------
// JSONL probe recorder — per-session ProbeHook decoupled from disk via
// an unbounded mpsc; a single tokio task drains and writes.
// ---------------------------------------------------------------------------

/// Spawn a single JSONL writer task draining `rx` to `path` (append). Each
/// message becomes one line. The task exits when every Sender is dropped
/// (channel closes); on exit it flushes the BufWriter. Returns the Sender
/// (Cloneable for per-session installs).
///
/// Buffer is bounded so a stalled disk doesn't grow the channel without bound;
/// see `JsonlProbeRecorder::on_token` for drop-on-full semantics. 4096 records
/// ≈ 120 KB of in-flight state, plenty for any realistic decode rate (≤ ~50
/// tok/s on Apple Silicon).
const PROBE_CHANNEL_DEPTH: usize = 4096;

async fn spawn_probe_writer(
    path: PathBuf,
) -> std::io::Result<tokio::sync::mpsc::Sender<serde_json::Value>> {
    use tokio::io::AsyncWriteExt as _;

    let file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .await?;
    let (tx, mut rx) =
        tokio::sync::mpsc::channel::<serde_json::Value>(PROBE_CHANNEL_DEPTH);

    tokio::spawn(async move {
        // Unbuffered writes. Per-line BufWriter would batch better but its
        // flush only runs when all Senders drop; under SIGKILL or crash that
        // flush never runs and the user sees an empty file. Probe write rate
        // caps at ~50 tok/s so the per-line syscall cost is negligible —
        // correctness over throughput.
        let mut file = file;
        while let Some(value) = rx.recv().await {
            let line = match serde_json::to_string(&value) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(event = "probe_write_failed", error = %e);
                    continue;
                }
            };
            if let Err(e) = file.write_all(line.as_bytes()).await {
                tracing::warn!(event = "probe_write_failed", error = %e);
                continue;
            }
            if let Err(e) = file.write_all(b"\n").await {
                tracing::warn!(event = "probe_write_failed", error = %e);
            }
        }
    });

    Ok(tx)
}

/// Per-session [`ProbeHook`]. Sends each token to the shared writer task via
/// the bounded mpsc — `on_token` returns in nanoseconds, so disk I/O never
/// blocks the prediction loop.
struct JsonlProbeRecorder {
    tx: tokio::sync::mpsc::Sender<serde_json::Value>,
    session_start: std::time::Instant,
    opts: SnapshotOpts,
}

impl JsonlProbeRecorder {
    fn install(
        tx: tokio::sync::mpsc::Sender<serde_json::Value>,
        model_name: &str,
        opts: SnapshotOpts,
    ) -> Self {
        // Best-effort: a session_start lost to a stalled disk is surprising but
        // not catastrophic. The token records that follow carry their own model
        // context via the file's append-only ordering.
        let _ = tx.try_send(serde_json::json!({
            "event": "session_start",
            "model": model_name,
        }));
        Self {
            tx,
            session_start: std::time::Instant::now(),
            opts,
        }
    }
}

impl ProbeHook for JsonlProbeRecorder {
    fn on_token(&mut self, ctx: ProbeCtx<'_>) {
        let ts_ms = self.session_start.elapsed().as_millis() as u64;
        let ctx_value = match serde_json::to_value(ctx) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(event = "probe_write_serialize_failed", error = %e);
                return;
            }
        };
        // Non-blocking send. Failure modes:
        // - `Full(_)`: writer task is behind (slow / stalled disk). Drop the
        //   record rather than block decode; a flat-line in the probe log is
        //   the disk-stall signal.
        // - `Closed(_)`: writer task exited (panicked or finished). Same
        //   treatment — failing predictions because the probe sink died would
        //   be worse than a missing record.
        let _ = self.tx.try_send(serde_json::json!({
            "event": "probe_ctx",
            "ts_ms": ts_ms,
            "ctx": ctx_value,
        }));
    }

    fn snapshot_opts(&self) -> Option<SnapshotOpts> {
        Some(self.opts.clone())
    }
}

// ---------------------------------------------------------------------------
// Streaming probe — broadcast bus + per-request recorder
//
// Fired only when `--probe-stream` is set. Consumers connect once to `GET
// /probe` and receive `StreamProbeMsg` events for every request the server
// handles, tagged by request UUID. The same UUID is returned on the sync
// `/v1/messages` response as `Message::id`, so consumers join the two by id.
// ---------------------------------------------------------------------------

/// Wire schema for the `/probe` SSE channel. Serializes to one of:
/// `{"event":"session_start","id":"…","model":"…"}`,
/// `{"event":"token","id":"…","ctx":{ … full ProbeCtx … }}`,
/// `{"event":"session_end","id":"…"}`.
///
/// `ctx` is the `ProbeCtx` rendered via `serde_json::to_value` —
/// `sample_options` is `#[serde(skip)]` (grammar Arc/Mutex doesn't serialize
/// cleanly); `snapshot` is the rich top-K + entropy view from slice-1.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
enum StreamProbeMsg {
    SessionStart {
        id: uuid::Uuid,
        model: String,
    },
    Token {
        id: uuid::Uuid,
        ctx: serde_json::Value,
    },
    SessionEnd {
        id: uuid::Uuid,
    },
}

/// Capacity of the broadcast channel. Tokens cap at ~50 tok/s on Apple Silicon;
/// 1024 absorbs ~20s of decode at full rate before a slow consumer starts
/// dropping. `Lagged` is observed at the SSE handler boundary and logged at
/// `warn`.
const PROBE_BROADCAST_CAPACITY: usize = 1024;

/// Per-request streaming probe recorder. Fires `serde_json::to_value(&ctx)` per
/// token and pushes a [`StreamProbeMsg::Token`] onto the bus.
///
/// `Sender::send` returns `Err` only when there are zero subscribers — silently
/// ignored, since "no consumers means no observers" is fine.
struct StreamingProbeRecorder {
    bus: tokio::sync::broadcast::Sender<StreamProbeMsg>,
    id: uuid::Uuid,
    opts: SnapshotOpts,
}

impl ProbeHook for StreamingProbeRecorder {
    fn on_token(&mut self, ctx: ProbeCtx<'_>) {
        // serde_json::to_value goes via the Serialize impl on ProbeCtx — owns
        // the result, which the broadcast bus then clones once per receiver.
        // Less code than deriving Clone on Snapshot etc.
        let value = match serde_json::to_value(ctx) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(event = "probe_stream_serialize_failed", error = %e);
                return;
            }
        };
        let _ = self.bus.send(StreamProbeMsg::Token {
            id: self.id,
            ctx: value,
        });
    }

    fn snapshot_opts(&self) -> Option<SnapshotOpts> {
        Some(self.opts.clone())
    }
}

/// Composes multiple [`ProbeHook`] implementations behind a single `Box<dyn
/// ProbeHook>`. `Engine::set_probe_hook` accepts only one; when `--record-json`
/// and `--probe-stream` are both set, this fans `on_token` to both inner
/// recorders and aggregates `snapshot_opts` so capture cost is paid once.
struct FanOutHook {
    hooks: Vec<Box<dyn ProbeHook>>,
}

impl ProbeHook for FanOutHook {
    fn on_token(&mut self, ctx: ProbeCtx<'_>) {
        // ProbeCtx is `#[non_exhaustive]` — can't struct-literal it from a
        // downstream crate. It's also `Copy`, so we just copy the whole bag of
        // borrows once per inner hook.
        for hook in self.hooks.iter_mut() {
            hook.on_token(ctx);
        }
    }

    fn snapshot_opts(&self) -> Option<SnapshotOpts> {
        // Aggregate: if any inner hook wants a snapshot, capture once with the
        // union of opts (max top_k, min p_threshold, entropy-OR). Capture cost
        // is paid once; cheap recorders see the populated `ctx.snapshot` and
        // ignore it.
        let mut acc: Option<SnapshotOpts> = None;
        for hook in self.hooks.iter() {
            if let Some(opts) = hook.snapshot_opts() {
                acc = Some(match acc {
                    None => opts,
                    Some(prev) => SnapshotOpts {
                        top_k: prev.top_k.max(opts.top_k),
                        p_threshold: prev.p_threshold.min(opts.p_threshold),
                        compute_entropy: prev.compute_entropy
                            || opts.compute_entropy,
                    },
                });
            }
        }
        acc
    }
}

/// `/probe` SSE handler. Subscribes a fresh receiver on the broadcast bus and
/// emits each [`StreamProbeMsg`] as one `text/event-stream` event. Generic over
/// the backend so both `llama_cpp_run` and `moeflux_run` can mount the same
/// handler.
///
/// Behavior:
/// - **No bus** (server started without `--probe-stream`): return 404. The
///   route is also gated at mount time, but defensive against anyone managing
///   to hit the path through some other path.
/// - **Lagged receiver** (slow consumer falls behind the broadcast ring): log
///   at `warn` and continue. The consumer skips the missed events; the stream
///   stays open.
/// - **Channel closed** (sender dropped — only happens at server shutdown): the
///   stream ends naturally.
async fn route_probe_stream<B: Backend>(
    axum::extract::State(state): axum::extract::State<AppState<B>>,
) -> Result<
    axum::response::Sse<
        impl futures_util::Stream<
            Item = Result<axum::response::sse::Event, std::convert::Infallible>,
        >,
    >,
    StatusCode,
>
where
    AppState<B>: Clone,
    Session<B>: FromPath,
{
    use axum::response::sse::{Event, KeepAlive, Sse};
    use futures_util::StreamExt as _;
    use tokio_stream::wrappers::{
        errors::BroadcastStreamRecvError, BroadcastStream,
    };

    let bus = state.probe_bus.ok_or(StatusCode::NOT_FOUND)?;
    // #95: an open SSE response never completes, and hyper's graceful
    // drain waits for exactly that — this line in a wedged shutdown's
    // transcript is the confirmation.
    tracing::info!("probe stream opened");
    let rx = bus.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|res| async move {
        match res {
            Ok(msg) => match Event::default().json_data(&msg) {
                Ok(ev) => Some(Ok(ev)),
                Err(e) => {
                    tracing::warn!(
                        event = "probe_stream_serialize_failed",
                        error = %e,
                    );
                    None
                }
            },
            Err(BroadcastStreamRecvError::Lagged(n)) => {
                tracing::warn!(event = "probe_stream_lagged", missed = n);
                None
            }
        }
    });

    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

/// Map a [`SessionError`] onto the status and error type Anthropic sends
/// for the same situation, so an Anthropic client's own retry policy
/// works unchanged against blallama: 400 `invalid_request_error` for a
/// request that will fail the same way on every retry, 500 `api_error`
/// for a transient failure a retry can clear (the SDKs retry it).
///
/// [`SessionError`]: drama_llama::SessionError
fn map_session_err(
    e: drama_llama::SessionError,
) -> (StatusCode, Json<ErrorEnvelope>) {
    use drama_llama::SessionError as E;
    let error = match e {
        // Anthropic's exact wording, which clients match on: the sum is
        // input + max_tokens, checked against the window before prefill.
        E::ContextOverflow {
            needed_cells,
            max_tokens,
            n_ctx,
        } => AnthropicError::InvalidRequest {
            message: format!(
                "prompt is too long: {} tokens > {n_ctx} maximum",
                needed_cells + max_tokens
            ),
        },
        // The request itself is the problem; retrying resends it.
        E::ChatTemplate(_)
        | E::ToolChoice(_)
        | E::OutputConfig(_)
        | E::RequestTopP(_)
        | E::Dialect(_)
        | E::SchemaBudget(_)
        | E::UnrenderableOpenThought { .. }
        | E::MediaUnsupported { .. }
        | E::Media(_)
        | E::TrailingMedia => AnthropicError::InvalidRequest {
            message: e.to_string(),
        },
        // The ingest guard now fires only when a content surface
        // bypassed neutralization: our bug, not the client's request,
        // so a 400 would tell them to fix what they did nothing wrong
        // in. An SDK retry fails the same way, but at prepare, cheaply.
        E::InjectedSpecialToken { .. } => AnthropicError::API {
            message: e.to_string(),
        },
        // Sampling failures (a fresh seed resamples them), decode
        // failures and engine trouble: all worth a retry.
        _ => AnthropicError::API {
            message: e.to_string(),
        },
    };
    error_response(error)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logging();
    fatal::install();
    let args = Args::parse();

    // If --record-json is set, spin up the JSONL writer task before any request
    // handles so per-request installs always have a Sender to clone. Failure to
    // open the file is a startup error — the user asked for probe records and
    // we can't deliver them.
    let record_json_tx = if let Some(path) = args.record_json.clone() {
        Some(spawn_probe_writer(path).await?)
    } else {
        None
    };

    // If --probe-stream is set, build the broadcast bus shared by all request
    // handlers (per-request `StreamingProbeRecorder` clones the Sender) and the
    // /probe SSE handler (calls `subscribe()` on each consumer connect).
    let probe_bus = if args.probe_stream {
        Some(
            tokio::sync::broadcast::channel::<StreamProbeMsg>(
                PROBE_BROADCAST_CAPACITY,
            )
            .0,
        )
    } else {
        None
    };

    // Narrow the union flags to the chosen backend's options here, in the
    // one place that names a concrete backend. A flag the backend has no
    // notion of stops the process now rather than being dropped and
    // discovered as a mysteriously short context later.
    match args.load.backend {
        #[cfg(feature = "llama-cpp")]
        BackendKind::LlamaCpp => {
            let options = drama_llama::LlamaCppOptions::try_from(&args.load)?;
            run::<drama_llama::LlamaCppBackend>(
                args,
                options,
                record_json_tx,
                probe_bus,
            )
            .await
        }
        #[cfg(all(feature = "moeflux", target_os = "macos"))]
        BackendKind::Moeflux => {
            let options = drama_llama::MoefluxOptions::try_from(&args.load)?;
            run::<drama_llama::MoefluxBackend>(
                args,
                options,
                record_json_tx,
                probe_bus,
            )
            .await
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A `POST` with `body` as JSON, as the routes receive it.
    fn json_request(body: &'static str) -> Request {
        axum::http::Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(body))
            .unwrap()
    }

    /// Extract a [`Prompt`] the way `/v1/messages` and `count_tokens`
    /// do, returning the rejection as the wire would carry it.
    async fn extract(
        req: Request,
    ) -> Result<Prompt, (StatusCode, serde_json::Value)> {
        AnthropicPrompt::from_request(req, &())
            .await
            .map(|AnthropicPrompt(p)| p)
            .map_err(|(status, Json(envelope))| {
                (status, serde_json::to_value(envelope).unwrap())
            })
    }

    /// #123: a body that doesn't deserialize is Anthropic's 400
    /// `invalid_request_error` in the error envelope — not axum's
    /// plain-text 422 — whether it is malformed JSON or well-formed JSON
    /// of the wrong shape.
    #[tokio::test]
    async fn body_rejection_is_anthropic_400_envelope() {
        for body in [
            // Syntax error.
            r#"{"model": "m", "max_tokens": 8, "messages": ["#,
            // Well-formed, wrong type (axum: 422). Not a missing field:
            // misanthropic's `Prompt` defaults every field.
            r#"{"model": "m", "max_tokens": "eight", "messages": []}"#,
            r#"{"model": "m", "max_tokens": 8, "messages": "hi"}"#,
        ] {
            let (status, value) = extract(json_request(body))
                .await
                .expect_err("body must be rejected");
            assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
            assert_eq!(value["type"], "error", "{body}");
            assert_eq!(value["error"]["type"], "invalid_request_error");
            assert!(
                value["error"]["message"]
                    .as_str()
                    .is_some_and(|m| !m.is_empty()),
                "{body}: {value}",
            );
        }

        // No JSON content type (axum: 415) is a malformed request too.
        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .body(axum::body::Body::from("{}"))
            .unwrap();
        let (status, value) = extract(req).await.expect_err("no content type");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(value["error"]["type"], "invalid_request_error");
    }

    /// A chat template that raises on the request's content — here the
    /// stock Qwen3.8 template's "System message must be at the
    /// beginning." on a mid-conversation system turn, live 2026-09-30 —
    /// is the request's fault: Anthropic's 400 `invalid_request_error`
    /// in the error envelope, not a retryable 500, and the session
    /// survives it.
    #[test]
    fn template_raise_is_anthropic_400_envelope() {
        use drama_llama::{
            prompt::{Message, Role},
            ChatTemplate, Content, RenderOptions,
        };
        let template = ChatTemplate::from_source(
            drama_llama::baked::QWEN38.stock.to_owned(),
            String::new(),
            "<|im_end|>".to_owned(),
        )
        .expect("template compiles");
        let text = |role, text: &str| Message {
            role,
            content: Content::text(text.to_owned()),
        };
        let prompt = Prompt {
            messages: vec![
                text(Role::User, "Who checks the fog signal?"),
                text(Role::Assistant, "Ada checks it."),
                text(Role::System, "The lamp is out."),
                text(Role::User, "And the lamp?"),
            ],
            ..Prompt::default()
        };
        let error = drama_llama::SessionError::from(
            template
                .render_with(&prompt, &RenderOptions::default())
                .expect_err(
                    "stock Qwen3.8 raises on a mid-conversation system",
                ),
        );
        assert!(error.is_reusable_after(), "{error}");
        let (status, Json(envelope)) = map_session_err(error);
        let value = serde_json::to_value(envelope).unwrap();
        assert_eq!(status, StatusCode::BAD_REQUEST, "{value}");
        assert_eq!(value["type"], "error");
        assert_eq!(value["error"]["type"], "invalid_request_error");
        assert!(
            value["error"]["message"]
                .as_str()
                .is_some_and(|m| m.contains("System message must be at")),
            "{value}",
        );
    }

    /// A schema with no grammar — here an empty `enum`, in a tool and in
    /// an `output_config` — is the request's fault: 400
    /// `invalid_request_error`, never a retryable 500.
    #[test]
    fn schema_without_grammar_is_anthropic_400_envelope() {
        use drama_llama::{
            dialect::{grammar_source, EmitOptions},
            grammar_for_output_config, CallSyntax, OutputConfigOptions,
            SessionError, Tool,
        };
        use misanthropic::prompt::output::OutputConfig;
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"x": {"enum": []}},
        });
        let tool = Tool::builder("t")
            .description("d")
            .schema(schema.clone())
            .build()
            .unwrap();
        let dialect = grammar_source(
            &CallSyntax::qwen_xml(),
            &[&tool],
            &EmitOptions::default(),
        )
        .unwrap_err();
        let output = grammar_for_output_config(
            &OutputConfig::json_schema(schema),
            &OutputConfigOptions::default(),
            false,
        )
        .unwrap_err();
        for error in [SessionError::from(dialect), SessionError::from(output)] {
            let (status, Json(envelope)) = map_session_err(error);
            let value = serde_json::to_value(envelope).unwrap();
            assert_eq!(status, StatusCode::BAD_REQUEST, "{value}");
            assert_eq!(value["error"]["type"], "invalid_request_error");
            assert!(
                value["error"]["message"]
                    .as_str()
                    .is_some_and(|m| m.contains("empty `enum`")),
                "{value}",
            );
        }
    }

    /// A request whose schemas measure past the server's limits is the
    /// request's fault too: 400 `invalid_request_error`, naming the
    /// limit and where.
    #[test]
    fn schema_past_limits_is_anthropic_400_envelope() {
        use drama_llama::{schema_budget::check_schemas, SessionError, Tool};
        let tool = Tool::builder("lookup")
            .description("d")
            .schema(serde_json::json!({
                "type": "object",
                "properties": {"a": {}, "b": {}},
            }))
            .build()
            .unwrap();
        let limits = SchemaLimits::default().with_max_params(1);
        let error = check_schemas([&tool], None, &limits).unwrap_err();
        let (status, Json(envelope)) =
            map_session_err(SessionError::from(error));
        let value = serde_json::to_value(envelope).unwrap();
        assert_eq!(status, StatusCode::BAD_REQUEST, "{value}");
        assert_eq!(value["error"]["type"], "invalid_request_error");
        let message = value["error"]["message"].as_str().unwrap();
        assert!(message.contains("tool `lookup`"), "{message}");
        assert!(message.contains("more than 1 top-level"), "{message}");
    }

    /// The flags default to the library's limits.
    #[test]
    fn schema_limit_flags_default_to_the_library() {
        let args = Args::parse_from(["blallama", "models"]);
        assert_eq!(args.schema_limits.limits(), SchemaLimits::default());
        let args =
            Args::parse_from(["blallama", "models", "--schema-max-tools", "3"]);
        assert_eq!(args.schema_limits.limits().max_tools, 3);
        let args =
            Args::parse_from(["blallama", "models", "--schema-max-width", "9"]);
        assert_eq!(args.schema_limits.limits().max_width, 9);
        let args =
            Args::parse_from(["blallama", "models", "--schema-max-depth", "8"]);
        assert_eq!(args.schema_limits.limits().max_depth, 8);
    }

    /// The ingest guard is a bug detector now: a shortfall is a 500
    /// `api_error`, not a 400 blaming the client's request.
    #[test]
    fn neutralization_bypass_is_anthropic_500() {
        let error = drama_llama::SessionError::InjectedSpecialToken {
            violations: Vec::new(),
        };
        let (status, Json(envelope)) = map_session_err(error);
        let value = serde_json::to_value(envelope).unwrap();
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{value}");
        assert_eq!(value["error"]["type"], "api_error");
    }

    /// A response as the wire carries it, cut by `max_tokens` after
    /// `calls` (name, input JSON).
    fn cut_response(calls: &[(&str, &str)]) -> MessageResponse {
        let content: Vec<String> = calls
            .iter()
            .enumerate()
            .map(|(i, (name, input))| {
                format!(
                    r#"{{"type": "tool_use", "id": "toolu_{i}",
                        "name": "{name}", "input": {input}}}"#
                )
            })
            .collect();
        serde_json::from_str(&format!(
            r#"{{"id": "msg_0", "type": "message", "role": "assistant",
                "model": "m", "content": [{}],
                "stop_reason": "max_tokens", "stop_sequence": null,
                "usage": {{"input_tokens": 1, "output_tokens": 1}}}}"#,
            content.join(", "),
        ))
        .expect("wire-shaped response")
    }

    /// The Phase G safeguard keys on a *verbatim* repeat in a cut turn:
    /// the same call twice resamples; distinct calls, or the same tool
    /// with different input, are an ordinary cut turn (200
    /// `max_tokens`, #121), and so is any turn that was not cut.
    #[test]
    fn loops_a_call_is_a_verbatim_repeat_in_a_cut_turn() {
        let a = ("get_weather", r#"{"city": "Paris"}"#);
        let b = ("get_weather", r#"{"city": "Oslo"}"#);
        assert!(loops_a_call(&cut_response(&[a, b, a])));
        assert!(!loops_a_call(&cut_response(&[a, b])));
        assert!(!loops_a_call(&cut_response(&[a])));
        assert!(!loops_a_call(&cut_response(&[])));

        let mut finished = cut_response(&[a, a]);
        finished.stop_reason = Some(StopReason::ToolUse);
        assert!(!loops_a_call(&finished));
    }

    /// The resample arms, the Phase G one included: a looping cut
    /// turn is redrawn, an ordinary one (or a finished one) answered;
    /// a special token, a grammar violation or a schema violation is
    /// redrawn, any other error answered.
    #[test]
    fn resample_reason_redraws_only_the_unlucky_paths() {
        use drama_llama::SessionError;
        let a = ("get_weather", r#"{"city": "Paris"}"#);
        let b = ("get_weather", r#"{"city": "Oslo"}"#);
        let reason = |r| resample_reason(&r).map(|r| format!("{r:?}"));

        let looping = Ok(cut_response(&[a, a]));
        assert!(matches!(
            resample_reason(&looping),
            Some(Resample::CallLoop)
        ));
        assert_eq!(reason(Ok(cut_response(&[a, b]))), None);
        let mut finished = cut_response(&[a, a]);
        finished.stop_reason = Some(StopReason::ToolUse);
        assert_eq!(reason(Ok(finished)), None);

        let special = Err(SessionError::EmittedSpecialToken {
            found: vec!["<|im_end|>".into()],
        });
        assert!(matches!(
            resample_reason(&special),
            Some(Resample::SpecialToken([found])) if found == "<|im_end|>"
        ));
        let violation = Err(SessionError::GrammarViolation {
            partial_output: drama_llama::prompt::Content(Vec::new()),
        });
        assert!(matches!(
            resample_reason(&violation),
            Some(Resample::GrammarViolation(_))
        ));
        let schema = Err(schema_violation());
        assert!(matches!(
            resample_reason(&schema),
            Some(Resample::SchemaViolation(_))
        ));
        assert_eq!(reason(Err(SessionError::TrailingMedia)), None);
    }

    /// A schema violation as `Session` raises it for the 2026-10-01
    /// consent answer (`"soul_text":"", "$memory_note":""}`).
    fn schema_violation() -> drama_llama::SessionError {
        drama_llama::SessionError::SchemaViolation {
            mismatch: drama_llama::SchemaMismatch {
                path: String::new(),
                kind: drama_llama::MismatchKind::MissingProperty(
                    "memory_note".into(),
                ),
            },
            partial_output: drama_llama::prompt::Content(Vec::new()),
        }
    }

    /// When every draw breaks its schema, the client gets what Anthropic
    /// sends for a transient server fault — a 500 `api_error` its SDK
    /// retries — never a 200 carrying the invalid value.
    #[test]
    fn schema_violation_answers_500_api_error() {
        let error = schema_violation();
        assert!(error.is_reusable_after(), "{error}");
        let (status, Json(envelope)) = map_session_err(error);
        let value = serde_json::to_value(envelope).unwrap();
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{value}");
        assert_eq!(value["error"]["type"], "api_error");
        assert!(
            value["error"]["message"]
                .as_str()
                .is_some_and(|m| m.contains("missing required property")),
            "{value}",
        );
    }

    /// Serve `app` on an ephemeral local port.
    async fn serve(app: Router) -> std::net::SocketAddr {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("ephemeral port");
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await });
        addr
    }

    /// `POST` `body` as JSON to `path` over a raw socket, as a client
    /// would; returns the response head and payload.
    async fn post_raw(
        addr: std::net::SocketAddr,
        path: &str,
        body: &str,
    ) -> (String, String) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
        let request = format!(
            "POST {path} HTTP/1.1\r\nHost: localhost\r\n\
             Content-Type: application/json\r\nContent-Length: {}\r\n\
             Connection: close\r\n\r\n{body}",
            body.len(),
        );
        stream.write_all(request.as_bytes()).await.unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).await.unwrap();
        let (head, payload) =
            response.split_once("\r\n\r\n").expect("an HTTP response");
        (head.to_owned(), payload.to_owned())
    }

    /// #123: a body over the limit is Anthropic's 413
    /// `request_too_large` in the error envelope, through the real
    /// `DefaultBodyLimit` layer (shrunk to 16 bytes on a one-route
    /// router, so a tiny body is oversized) on an ephemeral local port.
    #[tokio::test]
    async fn oversized_body_is_anthropic_413_envelope() {
        async fn accept(_: AnthropicJson<Prompt>) -> StatusCode {
            StatusCode::OK
        }
        let addr = serve(
            Router::new()
                .route("/v1/messages", post(accept))
                .layer(DefaultBodyLimit::max(16)),
        )
        .await;

        let body = r#"{"model": "m", "max_tokens": 8, "messages": []}"#;
        let (head, payload) = post_raw(addr, "/v1/messages", body).await;
        assert!(head.starts_with("HTTP/1.1 413"), "{head}");
        let value: serde_json::Value =
            serde_json::from_str(&payload).expect("a JSON envelope");
        assert_eq!(value["type"], "error", "{value}");
        assert_eq!(value["error"]["type"], "request_too_large", "{value}");
    }

    /// A stop sequence with no non-whitespace character is Anthropic's
    /// 400, envelope and message byte-for-byte as captured (2026-09-30,
    /// claude-haiku-4-5, `stream` false and true alike) — on both routes
    /// that take a request body, through [`AnthropicPrompt`] on a real
    /// router. A stop that merely *contains* whitespace is fine.
    #[tokio::test]
    async fn whitespace_only_stop_sequence_is_anthropic_400() {
        async fn accept(_: AnthropicPrompt) -> StatusCode {
            StatusCode::OK
        }
        let addr = serve(
            Router::new()
                .route("/v1/messages", post(accept))
                .route("/v1/messages/count_tokens", post(accept)),
        )
        .await;
        let body = |stops: &str| {
            format!(
                r#"{{"model": "m", "max_tokens": 8, "stream": true,
                    "messages": [{{"role": "user", "content": "hi"}}],
                    "stop_sequences": {stops}}}"#
            )
        };
        let expected: serde_json::Value = serde_json::from_str(concat!(
            r#"{"type":"error","error":{"type":"invalid_request_error","#,
            r#""message":"stop_sequences: each stop sequence must "#,
            r#"contain non-whitespace"}}"#,
        ))
        .unwrap();

        for path in ["/v1/messages", "/v1/messages/count_tokens"] {
            for stops in [r#"["\n"]"#, r#"[" \t\r\n"]"#, r#"["STOP", "\n"]"#] {
                let (head, payload) = post_raw(addr, path, &body(stops)).await;
                assert!(head.starts_with("HTTP/1.1 400"), "{path} {stops}");
                let value: serde_json::Value =
                    serde_json::from_str(&payload).expect("a JSON envelope");
                assert_eq!(value, expected, "{path} {stops}");
            }
            for stops in [r#"["\nObservation:"]"#, r#"[" STOP "]"#, "[]"] {
                let (head, _) = post_raw(addr, path, &body(stops)).await;
                assert!(head.starts_with("HTTP/1.1 200"), "{path} {stops}");
            }
        }
    }

    /// Anthropic's 400 for a fifth `cache_control` marker, where the
    /// fifth is the request-level automatic one (captured 2026-09-30 on
    /// claude-haiku-4-5, both routes). Four in all is accepted.
    #[tokio::test]
    async fn fifth_cache_marker_counting_the_automatic_one_is_anthropic_400() {
        async fn accept(_: AnthropicPrompt) -> StatusCode {
            StatusCode::OK
        }
        let addr = serve(
            Router::new()
                .route("/v1/messages", post(accept))
                .route("/v1/messages/count_tokens", post(accept)),
        )
        .await;
        let body = |explicit: usize| {
            let block = r#"{"type": "text", "text": "a",
                "cache_control": {"type": "ephemeral"}}"#;
            let blocks = vec![block; explicit].join(",");
            format!(
                r#"{{"model": "m", "max_tokens": 8,
                    "cache_control": {{"type": "ephemeral"}},
                    "messages": [{{"role": "user", "content": [{blocks},
                        {{"type": "text", "text": "tail"}}]}}]}}"#
            )
        };
        let expected: serde_json::Value = serde_json::from_str(concat!(
            r#"{"type":"error","error":{"type":"invalid_request_error","#,
            r#""message":"A maximum of 4 blocks with cache_control may be "#,
            r#"provided. Found 5."}}"#,
        ))
        .unwrap();

        for path in ["/v1/messages", "/v1/messages/count_tokens"] {
            let (head, payload) = post_raw(addr, path, &body(4)).await;
            assert!(head.starts_with("HTTP/1.1 400"), "{path}: {head}");
            let value: serde_json::Value =
                serde_json::from_str(&payload).expect("a JSON envelope");
            assert_eq!(value, expected, "{path}");
            let (head, _) = post_raw(addr, path, &body(3)).await;
            assert!(head.starts_with("HTTP/1.1 200"), "{path}: {head}");
        }
    }

    /// #123: a tool whose schema interleaves required and optional
    /// properties (`zulu` required, `alpha` optional, `mike` required)
    /// deserializes. Anthropic keeps optionals in place and accepts this
    /// shape; a schema-order check on misanthropic's deserialize path,
    /// if feature unification ever turned one on here, would 400 it.
    #[tokio::test]
    async fn interleaved_required_optional_tool_deserializes() {
        let prompt = extract(json_request(
            r#"{
                "model": "m",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{
                    "name": "zam",
                    "description": "interleaved required/optional",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "zulu": {"type": "string"},
                            "alpha": {"type": "string"},
                            "mike": {"type": "integer"}
                        },
                        "required": ["zulu", "mike"]
                    }
                }]
            }"#,
        ))
        .await
        .expect("interleaved schema must deserialize");

        let tool = prompt
            .tools
            .iter()
            .flatten()
            .find_map(|def| def.as_method())
            .expect("the custom tool survives");
        assert_eq!(tool.name, "zam");
        // Declaration order intact — the grammar lays fields out in it.
        let order: Vec<&str> = tool.schema["properties"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(order, ["zulu", "alpha", "mike"]);
    }

    /// `StreamProbeMsg` wire format check — SessionStart / Token / SessionEnd
    /// serialize to the schema documented on the type. The /probe consumer
    /// relies on the `event` discriminator + the `id` field shape; this catches
    /// accidental shape changes.
    #[test]
    fn stream_probe_msg_wire_format() {
        let id =
            uuid::Uuid::from_u128(0x0123_4567_89AB_CDEF_FEDC_BA98_7654_3210);
        let id_str = id.to_string();

        let start = serde_json::to_value(&StreamProbeMsg::SessionStart {
            id,
            model: "test-model".to_string(),
        })
        .unwrap();
        assert_eq!(start["event"], "session_start");
        assert_eq!(start["id"], id_str);
        assert_eq!(start["model"], "test-model");

        let token = serde_json::to_value(&StreamProbeMsg::Token {
            id,
            ctx: serde_json::json!({"token": 42, "n_cur": 7}),
        })
        .unwrap();
        assert_eq!(token["event"], "token");
        assert_eq!(token["id"], id_str);
        assert_eq!(token["ctx"]["token"], 42);

        let end =
            serde_json::to_value(&StreamProbeMsg::SessionEnd { id }).unwrap();
        assert_eq!(end["event"], "session_end");
        assert_eq!(end["id"], id_str);
    }

    /// Test-only hook that declares a fixed `SnapshotOpts`. Used to exercise
    /// `FanOutHook::snapshot_opts` aggregation without needing a real
    /// `ProbeCtx` (which is non-exhaustive and can't be
    /// struct-literal-constructed outside the defining crate).
    struct OptsHook(Option<SnapshotOpts>);
    impl ProbeHook for OptsHook {
        fn on_token(&mut self, _ctx: ProbeCtx<'_>) {}
        fn snapshot_opts(&self) -> Option<SnapshotOpts> {
            self.0.clone()
        }
    }

    #[test]
    fn fan_out_aggregates_snapshot_opts() {
        // No inner hook wants snapshot → None.
        let mut fan = FanOutHook { hooks: Vec::new() };
        fan.hooks.push(Box::new(OptsHook(None)));
        fan.hooks.push(Box::new(OptsHook(None)));
        assert!(fan.snapshot_opts().is_none(), "all-None inner ⇒ None");

        // One inner hook wants snapshot → that hook's opts pass through.
        let opts_a = SnapshotOpts {
            top_k: NonZeroUsize::new(20).unwrap(),
            p_threshold: 0.005,
            compute_entropy: false,
        };
        let mut fan = FanOutHook { hooks: Vec::new() };
        fan.hooks.push(Box::new(OptsHook(None)));
        fan.hooks.push(Box::new(OptsHook(Some(opts_a.clone()))));
        let agg = fan.snapshot_opts().expect("at least one Some");
        assert_eq!(agg.top_k, opts_a.top_k);
        assert_eq!(agg.p_threshold, opts_a.p_threshold);
        assert_eq!(agg.compute_entropy, opts_a.compute_entropy);

        // Two inner hooks want snapshot → max top_k, min p_threshold,
        // entropy-OR.
        let opts_b = SnapshotOpts {
            top_k: NonZeroUsize::new(100).unwrap(),
            p_threshold: 0.0,
            compute_entropy: true,
        };
        let mut fan = FanOutHook { hooks: Vec::new() };
        fan.hooks.push(Box::new(OptsHook(Some(opts_a.clone()))));
        fan.hooks.push(Box::new(OptsHook(Some(opts_b.clone()))));
        let agg = fan.snapshot_opts().expect("at least one Some");
        assert_eq!(agg.top_k, opts_b.top_k, "max(20, 100) = 100");
        assert_eq!(agg.p_threshold, 0.0, "min(0.005, 0.0) = 0.0");
        assert!(agg.compute_entropy, "false || true = true");
    }

    /// `StreamingProbeRecorder` declares the snapshot appetite it was
    /// configured with. Trivial but catches accidental hardcoding / override of
    /// the `opts` field.
    #[test]
    fn streaming_recorder_advertises_its_opts() {
        let (bus, _rx) = tokio::sync::broadcast::channel::<StreamProbeMsg>(4);
        let id =
            uuid::Uuid::from_u128(0xDEADBEEF_DEADBEEF_DEADBEEF_DEADBEEFu128);
        let opts = SnapshotOpts {
            top_k: NonZeroUsize::new(50).unwrap(),
            p_threshold: 0.001,
            compute_entropy: false,
        };
        let recorder = StreamingProbeRecorder {
            bus,
            id,
            opts: opts.clone(),
        };
        let advertised = recorder.snapshot_opts().expect("Some");
        assert_eq!(advertised.top_k, opts.top_k);
        assert_eq!(advertised.p_threshold, opts.p_threshold);
        assert_eq!(advertised.compute_entropy, opts.compute_entropy);
    }
}
