use std::{
    num::{NonZeroU128, NonZeroUsize},
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use axum::{
    extract::{Json, State},
    http::StatusCode,
    routing::{get, post},
    Router,
};
use clap::{Parser, ValueEnum};
use drama_llama::{
    backend::{Backend, Model},
    prompt::{AnthropicError, MessageResponse, Usage},
    FromPath, ProbeCtx, ProbeHook, Prompt, Session, SnapshotOpts,
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
    /// Inference backend. `llama-cpp` discovers `.gguf` files; `moeflux`
    /// discovers child directories with the `mlx/`/`artifacts/`/`root/`
    /// convention. Variants are cfg-gated — a build with only one backend
    /// feature accepts only that variant.
    #[arg(long, value_enum, default_value_t = default_backend_kind())]
    backend: BackendKind,
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
}

/// Inference backend selector. Variants are cfg-gated to whichever crate
/// features are enabled.
#[derive(Copy, Clone, Debug, ValueEnum)]
enum BackendKind {
    #[cfg(feature = "llama-cpp")]
    LlamaCpp,
    #[cfg(all(feature = "moeflux", target_os = "macos"))]
    Moeflux,
}

/// Default `--backend` value: prefer llama-cpp when both backends are compiled
/// in (it's been the default for the lifetime of blallama).
const fn default_backend_kind() -> BackendKind {
    #[cfg(feature = "llama-cpp")]
    {
        BackendKind::LlamaCpp
    }
    #[cfg(all(
        all(feature = "moeflux", target_os = "macos"),
        not(feature = "llama-cpp"),
    ))]
    {
        BackendKind::Moeflux
    }
}

#[derive(Clone)]
struct AppState<B: Backend> {
    args: Arc<Args>,
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

/// List directory entries whose followed-symlink metadata satisfies `accept`.
/// llama-cpp wants `is_file()` (one `.gguf` per model); moeflux wants
/// `is_dir()` (one parent dir per model).
///
/// Uses `metadata()` (which follows symlinks) rather than `file_type()` (which
/// reports the entry as `symlink` without chasing it). Mike's test layout
/// symlinks `mlx` / `artifacts` / `root` into a single moeflux model dir, and
/// the dir itself can be a symlink — both forms must enumerate.
async fn list_entries<P>(
    path: impl AsRef<Path>,
    accept: P,
) -> Result<Vec<String>, std::io::Error>
where
    P: Fn(&str, &std::fs::Metadata) -> bool,
{
    let mut read_dir = tokio::fs::read_dir(path).await?;
    let mut models = vec![];
    while let Some(entry) = read_dir.next_entry().await? {
        // metadata() follows symlinks; symlink_metadata() would not. Skip
        // entries whose target is missing or unreadable.
        let Ok(meta) = entry.metadata().await else {
            continue;
        };
        let model = if let Ok(model) = entry.file_name().into_string() {
            model
        } else {
            continue;
        };
        if !accept(&model, &meta) {
            continue;
        }
        models.push(model)
    }
    Ok(models)
}

/// Resolve a requested model id against what's on disk. `Ok(None)` means serve
/// as-requested; `Ok(Some(d))` means substitute the `--default-model`
/// (unmodified Anthropic-SDK clients request `claude-*` ids); `Err` is the 404
/// payload.
fn resolve_model(
    requested: &str,
    models: &[String],
    default_model: Option<&String>,
) -> Result<Option<String>, AnthropicError> {
    if models.iter().any(|m| m == requested) {
        return Ok(None);
    }
    if let Some(d) = default_model {
        if models.iter().any(|m| m == d) {
            return Ok(Some(d.clone()));
        }
    }
    Err(AnthropicError::NotFound {
        message: format!("model not found: {requested}"),
    })
}

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

async fn spawn_blocking_or_bust<F, R>(f: F) -> R
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    match spawn_blocking(f).await {
        Ok(r) => r,
        Err(e) => {
            error!(error = %e);
            std::process::exit(1); // We don't trust llama.cpp's destructors to
                                   // clean up so this is fatal.
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

#[cfg(all(feature = "moeflux", target_os = "macos"))]
fn log_moeflux_prefetch(
    id: impl AsRef<str>,
    stats: drama_llama::moeflux::PrefetchStats,
) {
    let rate = |h: u64, m: u64| -> f64 {
        let t = h + m;
        if t > 0 {
            h as f64 / t as f64
        } else {
            0.0
        }
    };
    info!(
        event = "moeflux_prefetch",
        id = id.as_ref(),
        prefill_hits = stats.prefill_hits,
        prefill_misses = stats.prefill_misses,
        prefill_hit_rate = rate(stats.prefill_hits, stats.prefill_misses),
        decode_hits = stats.decode_hits,
        decode_misses = stats.decode_misses,
        decode_hit_rate = rate(stats.decode_hits, stats.decode_misses),
    );
}

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

async fn route_tags<B: Backend>(
    State(state): State<AppState<B>>,
) -> Json<serde_json::Value> {
    let names = list_entries(&state.args.model_path, B::is_supported_model)
        .await
        .unwrap_or_default();
    let models: Vec<_> = names
        .iter()
        .map(|name| {
            serde_json::json!({
                "name": name,
                "model": name,
                "modified_at": "1970-01-01T00:00:00.000000000Z",
                "size": 0,
                "digest": "",
                "details": {
                    "format": "gguf",
                    "family": "",
                    "families": [],
                    "parameter_size": "",
                    "quantization_level": ""
                }
            })
        })
        .collect();
    Json(serde_json::json!({ "models": models }))
}

async fn run<B>(
    args: Args,
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

    let mut app = Router::new()
        .route("/v1/messages", post(route_messages))
        .route("/api/tags", get(route_tags));
    if probe_bus.is_some() {
        app = app.route("/probe", axum::routing::get(route_probe_stream));
    }
    let app = app.with_state(AppState {
        args: args.into(),
        record_json_tx,
        probe_bus,
        session,
    });
    axum::serve(listener, app).await?;
    Ok(())
}

async fn load_session<B>(
    root: impl AsRef<Path>,
    model: String,
    no_penalty: bool,
    seed: Option<u128>,
) -> Result<Session<B>, (StatusCode, Json<ErrorEnvelope>)>
where
    B: Backend,
    Session<B>: FromPath,
{
    let path = root.as_ref().join(&model);
    tracing::info!(
        event = "load_model",
        backend = B::NAME,
        model,
        path = path.to_string_lossy().as_ref()
    );
    Session::<B>::from_path(path)
        .await
        .map(|s| configure_session(s, no_penalty, seed))
        .map_err(map_session_err)
}

async fn route_messages<B>(
    State(state): State<AppState<B>>,
    Json(mut prompt): Json<Prompt>,
) -> Result<Json<MessageResponse>, (StatusCode, Json<ErrorEnvelope>)>
where
    B: Backend + 'static,
    Session<B>: FromPath,
{
    let models =
        match list_entries(&state.args.model_path, B::is_supported_model).await
        {
            Ok(models) => models,
            Err(e) => {
                let e = AnthropicError::NotFound {
                    message: format!("Models could not be loaded: {e}"),
                };
                error!(error = %e);
                return Err((StatusCode::NOT_FOUND, Json(e.into())));
            }
        };

    match resolve_model(
        &prompt.model.to_string(),
        &models,
        state.args.default_model.as_ref(),
    ) {
        Ok(None) => {}
        Ok(Some(default)) => {
            info!(
                requested = %prompt.model,
                served = %default,
                "substituting --default-model for unknown id",
            );
            prompt.model = default.into();
        }
        Err(e) => {
            error!(error = %e);
            return Err((StatusCode::NOT_FOUND, Json(e.into())));
        }
    }

    complete(state, prompt).await
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
    let mut lock = match state.session.try_lock() {
        Ok(lock) => lock,
        Err(_) => {
            return Err((
                StatusCode::from_u16(529).unwrap(),
                Json(
                    AnthropicError::Overloaded {
                        message: "Session is busy.".into(),
                        retry_after: None,
                    }
                    .into(),
                ),
            ))
        }
    };

    let mut session = match lock.take() {
        Some(session) => {
            let display =
                session.engine().model.display_name().unwrap_or_default();
            if display == prompt.model.to_string() {
                session
            } else {
                load_session(
                    &state.args.model_path,
                    prompt.model.to_string(),
                    state.args.no_penalty,
                    state.args.seed,
                )
                .await?
            }
        }
        None => {
            load_session(
                &state.args.model_path,
                prompt.model.to_string(),
                state.args.no_penalty,
                state.args.seed,
            )
            .await?
        }
    };

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
    let (session, result, elapsed) = spawn_blocking_or_bust(move || {
        let start = std::time::Instant::now();
        let result = session.complete_response_id(&prompt, id);
        (session, result, start.elapsed())
    })
    .await;

    // SessionEnd fires regardless of generation success — the probe stream
    // is a flight recorder, not a control channel.
    if let Some(bus) = &state.probe_bus {
        let _ = bus.send(StreamProbeMsg::SessionEnd { id });
    }

    match &result {
        Ok(_) => {
            lock.replace(session);
        }
        Err(e) if !e.is_fatal() => {
            error!(error = %e);
            lock.replace(session);
        }
        Err(e) => {
            error!(erorr = %e);
            // Drop session; next request will reload.
        }
    }

    let response = result.map_err(map_session_err)?;
    log_stats(&response.id, response.usage.clone(), elapsed);
    Ok(Json(response))
}

fn configure_session<B: Backend>(
    s: Session<B>,
    no_penalty: bool,
    seed: Option<u128>,
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
    // a request is honored as long as context remains, and one that asks for
    // more than fits simply fails at generation — we don't babysit a magic
    // ceiling constant that would need bumping as context windows grow.
    let configured = with_penalty
        .with_seed(seed.and_then(NonZeroU128::new))
        .with_prefix_cache(true);
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
            .model
            .display_name()
            .unwrap_or_default()
            .as_str(),
    );
    configured
}

/// Default `SnapshotOpts` for the streaming recorder. top_k=100 + p_threshold=0
/// + entropy=true is the cross-validation suite's working set: refusal-class
/// probes need tail-token visibility (high top_k, no threshold) and entropy is
/// cheap when probes are infrequent. Override via `Args` if/when finer control
/// is needed.
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
            session.engine().model.display_name().unwrap_or_default();
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
        let ctx_value = match serde_json::to_value(&ctx) {
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
        let value = match serde_json::to_value(&ctx) {
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
{
    use axum::response::sse::{Event, KeepAlive, Sse};
    use futures_util::StreamExt as _;
    use tokio_stream::wrappers::{
        errors::BroadcastStreamRecvError, BroadcastStream,
    };

    let bus = state.probe_bus.ok_or(StatusCode::NOT_FOUND)?;
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

fn map_session_err(
    e: drama_llama::SessionError,
) -> (StatusCode, Json<ErrorEnvelope>) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(
            AnthropicError::Unknown {
                code: Some(500.try_into().unwrap()),
                message: e.to_string(),
            }
            .into(),
        ),
    )
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logging();
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

    match args.backend {
        #[cfg(feature = "llama-cpp")]
        BackendKind::LlamaCpp => {
            run::<drama_llama::LlamaCppBackend>(args, record_json_tx, probe_bus)
                .await
        }
        #[cfg(all(feature = "moeflux", target_os = "macos"))]
        BackendKind::Moeflux => {
            run::<drama_llama::MoefluxBackend>(args, record_json_tx, probe_bus)
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
