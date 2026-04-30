use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use axum::{
    extract::{Json, State},
    http::StatusCode,
    routing::post,
    Router,
};
use clap::{Parser, ValueEnum};
use drama_llama::{
    backend::{Backend, Model},
    prompt::{AnthropicError, MessageResponse, Usage},
    ProbeCtx, ProbeHook, Prompt, Session, SnapshotOpts,
};
use std::num::{NonZeroU128, NonZeroUsize};
use tokio::{sync::Mutex, task::spawn_blocking};
use tracing::{error, info, instrument};

#[derive(Parser)]
#[command(about = "Demo /v1/messages server")]
struct Args {
    /// Path containing model files (llama.cpp) or model directories
    /// (moeflux).
    model_path: PathBuf,
    /// Port to use
    #[arg(long, default_value_t = 11435)]
    port: u16,
    /// Inference backend. `llama-cpp` discovers `.gguf` files;
    /// `moeflux` discovers child directories with the
    /// `mlx/`/`artifacts/`/`root/` convention. Variants are
    /// cfg-gated — a build with only one backend feature accepts
    /// only that variant.
    #[arg(long, value_enum, default_value_t = default_backend_kind())]
    backend: BackendKind,
    /// Force the repetition-penalty filter OFF, even when the
    /// per-model sampling sidecar enables it. Useful for probes,
    /// canary runs, and any diagnostic where you want to see the
    /// model's raw logit gradient with no penalty applied. Without
    /// this flag, sampling configuration comes from
    /// `<model>.sampling.toml` (gguf) or `parent/sampling.toml`
    /// (moeflux) — `Session::from_path*` writes a default sidecar on
    /// first load.
    #[arg(long, default_value_t = false)]
    no_penalty: bool,
    /// Optional fixed RNG seed forwarded to every prediction. Useful
    /// for tuning iteration: same prompt + same seed = same output,
    /// so a sidecar tweak shows up as a deliberate divergence rather
    /// than a stochastic one. Omit to use the crate default
    /// (`PredictOptions::DEFAULT_SEED`).
    #[arg(long)]
    seed: Option<u128>,
    /// Append per-token probe records to this JSONL file. One
    /// `{"event":"session_start", ...}` line per `/v1/messages` request,
    /// then one `{"event":"token", "token":N, "n_cur":M, "ts_ms":T}`
    /// line per yielded token, then nothing (the writer task drains
    /// silently as long as the channel sender lives). `ts_ms` is
    /// relative to the moment the recorder was installed for that
    /// request. Omit to disable JSONL recording.
    ///
    /// Composes with `--probe-stream`: both recorders see every token
    /// once via a `FanOutHook`.
    #[arg(long)]
    record_json: Option<PathBuf>,
    /// Mount the `/probe` SSE endpoint and install a per-request
    /// streaming recorder. Consumers connect once with `GET /probe`
    /// and receive `session_start` / `token` / `session_end` events
    /// for every request the server handles, tagged by the request's
    /// UUID (also returned as `Message::id` on the sync response).
    /// Late connectors miss early events; convention is to open
    /// `/probe` before sending `/v1/messages`.
    #[arg(long, default_value_t = false)]
    probe_stream: bool,
}

/// Inference backend selector. Variants are cfg-gated to whichever
/// crate features are enabled.
#[derive(Copy, Clone, Debug, ValueEnum)]
enum BackendKind {
    #[cfg(feature = "llama-cpp")]
    LlamaCpp,
    #[cfg(all(feature = "moeflux", target_os = "macos"))]
    Moeflux,
}

/// Default `--backend` value: prefer llama-cpp when both backends are
/// compiled in (it's been the default for the lifetime of blallama).
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
    /// Sender into the JSONL writer task. `None` if `--record-json`
    /// wasn't given. Cloned per-request when installing the
    /// [`JsonlProbeRecorder`]; all clones feed the same writer task /
    /// output file.
    record_json_tx: Option<tokio::sync::mpsc::Sender<ProbeMsg>>,
    /// Streaming-probe broadcast bus. `None` if `--probe-stream`
    /// wasn't given. Cloned per-request into a [`StreamingProbeRecorder`]
    /// and (separately) subscribed by the `/probe` SSE handler. The
    /// same bus carries `SessionStart` / `SessionEnd` events emitted
    /// directly from the request handler around the generation call.
    probe_bus: Option<tokio::sync::broadcast::Sender<StreamProbeMsg>>,
    session: Arc<Mutex<Option<Session<B>>>>,
}

/// List directory entries whose followed-symlink metadata satisfies
/// `accept`. llama-cpp wants `is_file()` (one `.gguf` per model);
/// moeflux wants `is_dir()` (one parent dir per model).
///
/// Uses `metadata()` (which follows symlinks) rather than
/// `file_type()` (which reports the entry as `symlink` without
/// chasing it). Mike's test layout symlinks `mlx` / `artifacts` /
/// `root` into a single moeflux model dir, and the dir itself can
/// be a symlink — both forms must enumerate.
async fn list_entries<P>(
    path: impl AsRef<Path>,
    accept: P,
) -> Result<Vec<String>, std::io::Error>
where
    P: Fn(&std::fs::Metadata) -> bool,
{
    let mut read_dir = tokio::fs::read_dir(path).await?;
    let mut models = vec![];
    while let Some(entry) = read_dir.next_entry().await? {
        // metadata() follows symlinks; symlink_metadata() would not.
        // Skip entries whose target is missing or unreadable.
        let Ok(meta) = entry.metadata().await else {
            continue;
        };
        if !accept(&meta) {
            continue;
        }
        let model = if let Ok(model) = entry.file_name().into_string() {
            model
        } else {
            continue;
        };
        models.push(model)
    }
    Ok(models)
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
            std::process::exit(1); // We don't trust llama.cpp's destructors
        }
    }
}

fn log_stats(id: impl AsRef<str>, usage: Usage, elapsed: Duration) {
    let Usage {
        input_tokens,
        cache_creation_input_tokens,
        cache_read_input_tokens,
        output_tokens,
    } = usage;

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

// Credit To Claude Opus 4.7 for this
fn init_logging() {
    use tracing_subscriber::{fmt, prelude::*, EnvFilter, Registry};

    // EnvFilter reads RUST_LOG. Falls back to "info" if unset.
    // Syntax: RUST_LOG=info,drama_llama=debug,axum=warn
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    // JSON formatter for structured output (downstream-parseable).
    // Span context flags control what span info rides on each event.
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

// ---------------------------------------------------------------------------
// llama.cpp run path
// ---------------------------------------------------------------------------

#[cfg(feature = "llama-cpp")]
mod llama_cpp_run {
    use super::*;
    use drama_llama::LlamaCppBackend;

    pub async fn run(
        args: Args,
        record_json_tx: Option<tokio::sync::mpsc::Sender<ProbeMsg>>,
        probe_bus: Option<tokio::sync::broadcast::Sender<StreamProbeMsg>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let listener = tokio::net::TcpListener::bind(format!(
            "0.0.0.0:{port}",
            port = args.port
        ))
        .await?;

        let session: Arc<Mutex<Option<Session<LlamaCppBackend>>>> =
            Mutex::from(None).into();

        let mut app =
            Router::new().route("/v1/messages", post(route_messages));
        if probe_bus.is_some() {
            app = app.route(
                "/probe",
                axum::routing::get(route_probe_stream::<LlamaCppBackend>),
            );
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

    async fn route_messages(
        State(state): State<AppState<LlamaCppBackend>>,
        Json(prompt): Json<Prompt>,
    ) -> Result<Json<MessageResponse>, (StatusCode, Json<AnthropicError>)>
    {
        let models = match list_entries(&state.args.model_path, |m| m.is_file()).await {
            Ok(models) => models,
            Err(e) => {
                let e = AnthropicError::NotFound {
                    message: format!("Models could not be loaded: {e}"),
                };
                error!(error = %e);
                return Err((StatusCode::NOT_FOUND, Json(e)));
            }
        };

        if !models.contains(&prompt.model.to_string()) {
            let e = AnthropicError::NotFound {
                message: format!(
                    "model not found: {model}",
                    model = prompt.model
                ),
            };
            error!(error = %e);
            return Err((StatusCode::NOT_FOUND, Json(e)));
        }

        complete(state, prompt).await
    }

    #[instrument(skip(state, prompt), fields(model = %prompt.model))]
    async fn complete(
        state: AppState<LlamaCppBackend>,
        prompt: Prompt,
    ) -> Result<Json<MessageResponse>, (StatusCode, Json<AnthropicError>)>
    {
        let mut lock = match state.session.try_lock() {
            Ok(lock) => lock,
            Err(_) => {
                return Err((
                    StatusCode::from_u16(529).unwrap(),
                    Json(AnthropicError::Overloaded {
                        message: "Session is busy.".into(),
                    }),
                ))
            }
        };

        let mut session = match lock.take() {
            Some(session) => {
                let display = session
                    .engine()
                    .model
                    .display_name()
                    .unwrap_or_default();
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

        // Per-request UUID — same id ends up on `Message.id` and on
        // every `StreamProbeMsg` emitted while this request runs.
        let id = uuid::Uuid::new_v4();
        install_per_request_hooks(
            &mut session,
            state.record_json_tx.as_ref(),
            state.probe_bus.as_ref(),
            id,
        );

        // Emit SessionStart on the bus before generation. SendError
        // means zero subscribers; harmless, ignored. The model name
        // here is the request's `prompt.model` (the user-facing name)
        // rather than the engine's display_name (the GGUF internal
        // name); both are recoverable from the JSONL ts_ms ordering
        // if needed.
        if let Some(bus) = &state.probe_bus {
            let _ = bus.send(StreamProbeMsg::SessionStart {
                id,
                model: prompt.model.to_string(),
            });
        }

        // Closure returns the session in *both* arms so it can be
        // restored to the lock — otherwise a `complete_response` error
        // drops it and the next request reloads from disk. See
        // `is_reusable_after` for the reuse-vs-reload classification.
        let (session, result, elapsed) =
            spawn_blocking_or_bust(move || {
                let start = std::time::Instant::now();
                let result = session.complete_response_id(&prompt, id);
                (session, result, start.elapsed())
            })
            .await;

        // SessionEnd fires regardless of generation success — the
        // probe stream is a flight recorder, not a control channel.
        if let Some(bus) = &state.probe_bus {
            let _ = bus.send(StreamProbeMsg::SessionEnd { id });
        }

        match &result {
            Ok(_) => {
                lock.replace(session);
            }
            Err(e) if is_reusable_after(e) => {
                lock.replace(session);
            }
            Err(_) => {
                // Drop session; next request will reload.
            }
        }

        let response = result.map_err(map_session_err)?;
        log_stats(&response.id, response.usage, elapsed);
        Ok(Json(response))
    }

    async fn load_session(
        root: impl AsRef<Path>,
        model: String,
        no_penalty: bool,
        seed: Option<u128>,
    ) -> Result<Session<LlamaCppBackend>, (StatusCode, Json<AnthropicError>)>
    {
        let path = root.as_ref().join(&model);
        tracing::info!(
            event = "load_model",
            backend = "llama-cpp",
            model,
            path = path.to_string_lossy().as_ref()
        );
        spawn_blocking_or_bust(|| {
            Session::<LlamaCppBackend>::from_path_with_n_ctx(path, 65536)
        })
        .await
        .map(|s| configure_session(s, no_penalty, seed))
        .map_err(map_session_err)
    }
}

// ---------------------------------------------------------------------------
// moeflux run path
// ---------------------------------------------------------------------------

#[cfg(all(feature = "moeflux", target_os = "macos"))]
mod moeflux_run {
    use super::*;
    use drama_llama::MoefluxBackend;

    pub async fn run(
        args: Args,
        record_json_tx: Option<tokio::sync::mpsc::Sender<ProbeMsg>>,
        probe_bus: Option<tokio::sync::broadcast::Sender<StreamProbeMsg>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let listener = tokio::net::TcpListener::bind(format!(
            "0.0.0.0:{port}",
            port = args.port
        ))
        .await?;

        let session: Arc<Mutex<Option<Session<MoefluxBackend>>>> =
            Mutex::from(None).into();

        let mut app =
            Router::new().route("/v1/messages", post(route_messages));
        if probe_bus.is_some() {
            app = app.route(
                "/probe",
                axum::routing::get(route_probe_stream::<MoefluxBackend>),
            );
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

    async fn route_messages(
        State(state): State<AppState<MoefluxBackend>>,
        Json(prompt): Json<Prompt>,
    ) -> Result<Json<MessageResponse>, (StatusCode, Json<AnthropicError>)>
    {
        let models = match list_entries(&state.args.model_path, |m| m.is_dir()).await {
            Ok(models) => models,
            Err(e) => {
                let e = AnthropicError::NotFound {
                    message: format!("Models could not be loaded: {e}"),
                };
                error!(error = %e);
                return Err((StatusCode::NOT_FOUND, Json(e)));
            }
        };

        if !models.contains(&prompt.model.to_string()) {
            let e = AnthropicError::NotFound {
                message: format!(
                    "model not found: {model}",
                    model = prompt.model
                ),
            };
            error!(error = %e);
            return Err((StatusCode::NOT_FOUND, Json(e)));
        }

        complete(state, prompt).await
    }

    #[instrument(skip(state, prompt), fields(model = %prompt.model))]
    async fn complete(
        state: AppState<MoefluxBackend>,
        prompt: Prompt,
    ) -> Result<Json<MessageResponse>, (StatusCode, Json<AnthropicError>)>
    {
        let mut lock = match state.session.try_lock() {
            Ok(lock) => lock,
            Err(_) => {
                return Err((
                    StatusCode::from_u16(529).unwrap(),
                    Json(AnthropicError::Overloaded {
                        message: "Session is busy.".into(),
                    }),
                ))
            }
        };

        let mut session = match lock.take() {
            Some(session) => {
                let display = session
                    .engine()
                    .model
                    .display_name()
                    .unwrap_or_default();
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

        // Per-request UUID — see llama-cpp variant for full rationale.
        let id = uuid::Uuid::new_v4();
        install_per_request_hooks(
            &mut session,
            state.record_json_tx.as_ref(),
            state.probe_bus.as_ref(),
            id,
        );

        if let Some(bus) = &state.probe_bus {
            let _ = bus.send(StreamProbeMsg::SessionStart {
                id,
                model: prompt.model.to_string(),
            });
        }

        // See llama-cpp variant + `is_reusable_after` doc-comment for
        // the reuse-vs-reload rationale.
        let (session, result, elapsed) =
            spawn_blocking_or_bust(move || {
                let start = std::time::Instant::now();
                let result = session.complete_response_id(&prompt, id);
                (session, result, start.elapsed())
            })
            .await;

        if let Some(bus) = &state.probe_bus {
            let _ = bus.send(StreamProbeMsg::SessionEnd { id });
        }
        match &result {
            Ok(_) => {
                lock.replace(session);
            }
            Err(e) if is_reusable_after(e) => {
                lock.replace(session);
            }
            Err(_) => {
                // Drop session; next request will reload.
            }
        }

        let response = result.map_err(map_session_err)?;
        log_stats(&response.id, response.usage, elapsed);
        Ok(Json(response))
    }

    async fn load_session(
        root: impl AsRef<Path>,
        model: String,
        no_penalty: bool,
        seed: Option<u128>,
    ) -> Result<Session<MoefluxBackend>, (StatusCode, Json<AnthropicError>)>
    {
        let path = root.as_ref().join(&model);
        tracing::info!(
            event = "load_model",
            backend = "moeflux",
            model,
            path = path.to_string_lossy().as_ref()
        );
        spawn_blocking_or_bust(|| {
            Session::<MoefluxBackend>::from_path(path)
        })
        .await
        .map(|s| configure_session(s, no_penalty, seed))
        .map_err(map_session_err)
    }
}

// ---------------------------------------------------------------------------
// Shared session post-load configuration
// ---------------------------------------------------------------------------

fn configure_session<B: Backend>(
    s: Session<B>,
    no_penalty: bool,
    seed: Option<u128>,
) -> Session<B> {
    // Sampling configuration is loaded from the per-model sidecar
    // (`<model>.sampling.toml` for gguf, `parent/sampling.toml` for
    // moeflux) inside `Session::from_path*`. `--no-penalty` overrides
    // the sidecar to force repetition penalty OFF — for probes,
    // canary runs, or any "what does this model do with no penalty"
    // diagnostic.
    let with_penalty = if no_penalty {
        s.without_repetition()
    } else {
        s
    };
    let configured = with_penalty
        .with_seed(seed.and_then(NonZeroU128::new))
        .with_prefix_cache(true)
        // Session-level generation cap. Distinct from `n_ctx` — that's
        // the KV context window, set per-backend at engine
        // construction (llama.cpp: `from_path_with_n_ctx(_, 65536)`;
        // moeflux: compile-time per model variant, surfaced in the
        // `session_ready` log below). 8K is the per-request gen
        // ceiling; the prompt's `max_tokens` wins when smaller, this
        // clips runaway requests.
        .with_max_tokens(8192.try_into().unwrap());
    // ProbeHook installation moved to per-request handlers — each
    // /v1/messages request gets a fresh hook bound to its UUID, so the
    // hook can fan out to JSONL, the broadcast bus, or both, with a
    // recorder lifetime that exactly matches the request.
    tracing::info!(
        event = "session_ready",
        n_ctx = configured.engine().n_ctx(),
        session_max_tokens = 8192u32,
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

/// Default `SnapshotOpts` for the streaming recorder. top_k=100 +
/// p_threshold=0 + entropy=true is the cross-validation suite's
/// working set: refusal-class probes need tail-token visibility
/// (high top_k, no threshold) and entropy is cheap when probes are
/// infrequent. Override via `Args` if/when finer control is needed.
fn default_stream_opts() -> SnapshotOpts {
    SnapshotOpts {
        top_k: NonZeroUsize::new(100).unwrap(),
        p_threshold: 0.0,
        compute_entropy: true,
    }
}

/// Build and install the per-request `FanOutHook` on `session`'s
/// engine. Returns `true` when at least one recorder was installed (so
/// the caller can emit `StreamProbeMsg::SessionStart` / `SessionEnd`
/// only when there's a streaming consumer to receive them).
fn install_per_request_hooks<B: Backend>(
    session: &mut Session<B>,
    record_json_tx: Option<&tokio::sync::mpsc::Sender<ProbeMsg>>,
    probe_bus: Option<&tokio::sync::broadcast::Sender<StreamProbeMsg>>,
    id: uuid::Uuid,
) {
    let mut hooks: Vec<Box<dyn ProbeHook>> = Vec::new();
    if let Some(tx) = record_json_tx {
        let model_name = session.engine().model.display_name().unwrap_or_default();
        hooks.push(Box::new(JsonlProbeRecorder::install(
            tx.clone(),
            model_name.as_str(),
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

/// Records flowing from `ProbeHook::on_token` (and one synthetic
/// `SessionStart` per recorder install) into the writer task.
#[derive(Debug)]
enum ProbeMsg {
    SessionStart { model: String },
    Token { token: i32, n_cur: usize, ts_ms: u64 },
}

/// Spawn a single JSONL writer task draining `rx` to `path` (append).
/// Each message becomes one line. The task exits when every Sender
/// is dropped (channel closes); on exit it flushes the BufWriter.
/// Returns the Sender (Cloneable for per-session installs).
///
/// Buffer is bounded so a stalled disk doesn't grow the channel
/// without bound; see `JsonlProbeRecorder::on_token` for drop-on-full
/// semantics. 4096 records ≈ 120 KB of in-flight state, plenty for
/// any realistic decode rate (≤ ~50 tok/s on Apple Silicon).
const PROBE_CHANNEL_DEPTH: usize = 4096;

async fn spawn_probe_writer(
    path: PathBuf,
) -> std::io::Result<tokio::sync::mpsc::Sender<ProbeMsg>> {
    use tokio::io::AsyncWriteExt as _;

    let file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .await?;
    let (tx, mut rx) =
        tokio::sync::mpsc::channel::<ProbeMsg>(PROBE_CHANNEL_DEPTH);

    tokio::spawn(async move {
        // Unbuffered writes. Per-line BufWriter would batch better but
        // its flush only runs when all Senders drop; under SIGKILL or
        // crash that flush never runs and the user sees an empty file.
        // Probe write rate caps at ~50 tok/s so the per-line syscall
        // cost is negligible — correctness over throughput.
        let mut file = file;
        while let Some(msg) = rx.recv().await {
            let line = match msg {
                ProbeMsg::SessionStart { model } => {
                    format!(
                        r#"{{"event":"session_start","model":{}}}"#,
                        serde_json::to_string(&model).unwrap_or_default(),
                    )
                }
                ProbeMsg::Token { token, n_cur, ts_ms } => format!(
                    r#"{{"event":"token","token":{token},"n_cur":{n_cur},"ts_ms":{ts_ms}}}"#,
                ),
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

/// Per-session [`ProbeHook`]. Sends each token to the shared writer
/// task via the bounded-pressure unbounded mpsc — `on_token` returns
/// in nanoseconds, so disk I/O never blocks the prediction loop.
struct JsonlProbeRecorder {
    tx: tokio::sync::mpsc::Sender<ProbeMsg>,
    session_start: std::time::Instant,
}

impl JsonlProbeRecorder {
    fn install(
        tx: tokio::sync::mpsc::Sender<ProbeMsg>,
        model_name: &str,
    ) -> Self {
        // Best-effort: a session_start lost to a stalled disk is
        // surprising but not catastrophic. The token records that
        // follow carry their own model context via the file's
        // append-only ordering.
        let _ = tx.try_send(ProbeMsg::SessionStart {
            model: model_name.to_owned(),
        });
        Self {
            tx,
            session_start: std::time::Instant::now(),
        }
    }
}

impl ProbeHook for JsonlProbeRecorder {
    fn on_token(&mut self, ctx: ProbeCtx<'_>) {
        let ts_ms = self.session_start.elapsed().as_millis() as u64;
        // Non-blocking send. Failure modes:
        // - `Full(_)`: writer task is behind (slow / stalled disk).
        //   Drop the record rather than block decode; a flat-line in
        //   the probe log is the disk-stall signal.
        // - `Closed(_)`: writer task exited (panicked or finished).
        //   Same treatment — failing predictions because the probe
        //   sink died would be worse than a missing record.
        let _ = self.tx.try_send(ProbeMsg::Token {
            token: ctx.token,
            n_cur: ctx.n_cur,
            ts_ms,
        });
    }
}

// ---------------------------------------------------------------------------
// Streaming probe — broadcast bus + per-request recorder
//
// Fired only when `--probe-stream` is set. Consumers connect once to
// `GET /probe` and receive `StreamProbeMsg` events for every request the
// server handles, tagged by request UUID. The same UUID is returned on
// the sync `/v1/messages` response as `Message::id`, so consumers join
// the two by id.
// ---------------------------------------------------------------------------

/// Wire schema for the `/probe` SSE channel. Serializes to one of:
/// `{"event":"session_start","id":"…","model":"…"}`,
/// `{"event":"token","id":"…","ctx":{ … full ProbeCtx … }}`,
/// `{"event":"session_end","id":"…"}`.
///
/// `ctx` is the `ProbeCtx` rendered via `serde_json::to_value` —
/// `sample_options` is `#[serde(skip)]` (grammar Arc/Mutex doesn't
/// serialize cleanly); `snapshot` is the rich top-K + entropy view
/// from slice-1.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
enum StreamProbeMsg {
    SessionStart { id: uuid::Uuid, model: String },
    Token { id: uuid::Uuid, ctx: serde_json::Value },
    SessionEnd { id: uuid::Uuid },
}

/// Capacity of the broadcast channel. Tokens cap at ~50 tok/s on Apple
/// Silicon; 1024 absorbs ~20s of decode at full rate before a slow
/// consumer starts dropping. `Lagged` is observed at the SSE handler
/// boundary and logged at `warn`.
const PROBE_BROADCAST_CAPACITY: usize = 1024;

/// Per-request streaming probe recorder. Fires `serde_json::to_value(&ctx)`
/// per token and pushes a [`StreamProbeMsg::Token`] onto the bus.
///
/// `Sender::send` returns `Err` only when there are zero subscribers —
/// silently ignored, since "no consumers means no observers" is fine.
struct StreamingProbeRecorder {
    bus: tokio::sync::broadcast::Sender<StreamProbeMsg>,
    id: uuid::Uuid,
    opts: SnapshotOpts,
}

impl ProbeHook for StreamingProbeRecorder {
    fn on_token(&mut self, ctx: ProbeCtx<'_>) {
        // serde_json::to_value goes via the Serialize impl on ProbeCtx —
        // owns the result, which the broadcast bus then clones once
        // per receiver. Less code than deriving Clone on Snapshot etc.
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

/// Composes multiple [`ProbeHook`] implementations behind a single
/// `Box<dyn ProbeHook>`. `Engine::set_probe_hook` accepts only one;
/// when `--record-json` and `--probe-stream` are both set, this fans
/// `on_token` to both inner recorders and aggregates `snapshot_opts`
/// so capture cost is paid once.
struct FanOutHook {
    hooks: Vec<Box<dyn ProbeHook>>,
}

impl ProbeHook for FanOutHook {
    fn on_token(&mut self, ctx: ProbeCtx<'_>) {
        // ProbeCtx is `#[non_exhaustive]` — can't struct-literal it
        // from a downstream crate. It's also `Copy`, so we just copy
        // the whole bag of borrows once per inner hook.
        for hook in self.hooks.iter_mut() {
            hook.on_token(ctx);
        }
    }

    fn snapshot_opts(&self) -> Option<SnapshotOpts> {
        // Aggregate: if any inner hook wants a snapshot, capture once
        // with the union of opts (max top_k, min p_threshold,
        // entropy-OR). Capture cost is paid once; cheap recorders see
        // the populated `ctx.snapshot` and ignore it.
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

/// `/probe` SSE handler. Subscribes a fresh receiver on the
/// broadcast bus and emits each [`StreamProbeMsg`] as one
/// `text/event-stream` event. Generic over the backend so both
/// `llama_cpp_run` and `moeflux_run` can mount the same handler.
///
/// Behavior:
/// - **No bus** (server started without `--probe-stream`): return 404.
///   The route is also gated at mount time, but defensive against
///   anyone managing to hit the path through some other path.
/// - **Lagged receiver** (slow consumer falls behind the broadcast
///   ring): log at `warn` and continue. The consumer skips the
///   missed events; the stream stays open.
/// - **Channel closed** (sender dropped — only happens at server
///   shutdown): the stream ends naturally.
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
) -> (StatusCode, Json<AnthropicError>) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(AnthropicError::Unknown {
            code: Some(500.try_into().unwrap()),
            message: e.to_string(),
        }),
    )
}

/// Decide whether a session is safe to reuse after `complete_response`
/// returned this error variant. Reusable variants return the session
/// to the lock so the next request can hit the prefix cache; non-
/// reusable variants drop the session, forcing a reload — the
/// pre-Phase-7 default that was applied unconditionally.
///
/// Default for unknown variants (added in future SessionError
/// expansions) is **non-reusable**: erring on the side of correctness
/// over the perf cost of a reload. New variants must be explicitly
/// classified once their state implications are understood.
fn is_reusable_after(err: &drama_llama::SessionError) -> bool {
    use drama_llama::SessionError as E;
    match err {
        // Render / grammar-compile errors fire before any decode work
        // touches the engine. State is untouched — safe to reuse.
        E::ChatTemplate(_) | E::ToolChoice(_) | E::OutputConfig(_) => true,
        // run_call invalidates its own prefix cache on grammar
        // violation, so the session is internally consistent.
        E::GrammarViolation { .. } => true,
        // Backend prefill error (Phase 7's `SessionError::Decode`).
        // Engine state may be dirty — but Session's
        // kv_setup_and_chunk_prefill on the next call will memory_clear
        // or restore_to a known-good snapshot, recovering before any
        // generation runs. Reusable.
        E::Decode(_) => true,
        // Engine setup errors can't fire post-load (session is already
        // built); if they ever do, drop and reload.
        #[cfg(feature = "llama-cpp")]
        E::LlamaCppEngine(_) => false,
        #[cfg(all(feature = "moeflux", target_os = "macos"))]
        E::MoefluxEngine(_) => false,
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logging();
    let args = Args::parse();

    // If --record-json is set, spin up the JSONL writer task before
    // any request handles so per-request installs always have a
    // Sender to clone. Failure to open the file is a startup error —
    // the user asked for probe records and we can't deliver them.
    let record_json_tx = if let Some(path) = args.record_json.clone() {
        Some(spawn_probe_writer(path).await?)
    } else {
        None
    };

    // If --probe-stream is set, build the broadcast bus shared by all
    // request handlers (per-request `StreamingProbeRecorder` clones
    // the Sender) and the /probe SSE handler (calls `subscribe()` on
    // each consumer connect).
    let probe_bus = if args.probe_stream {
        Some(tokio::sync::broadcast::channel::<StreamProbeMsg>(
            PROBE_BROADCAST_CAPACITY,
        ).0)
    } else {
        None
    };

    match args.backend {
        #[cfg(feature = "llama-cpp")]
        BackendKind::LlamaCpp => {
            llama_cpp_run::run(args, record_json_tx, probe_bus).await
        }
        #[cfg(all(feature = "moeflux", target_os = "macos"))]
        BackendKind::Moeflux => {
            moeflux_run::run(args, record_json_tx, probe_bus).await
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// `StreamProbeMsg` wire format check — SessionStart / Token /
    /// SessionEnd serialize to the schema documented on the type. The
    /// /probe consumer relies on the `event` discriminator + the `id`
    /// field shape; this catches accidental shape changes.
    #[test]
    fn stream_probe_msg_wire_format() {
        let id = uuid::Uuid::from_u128(0x0123_4567_89AB_CDEF_FEDC_BA98_7654_3210);
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

        let end = serde_json::to_value(&StreamProbeMsg::SessionEnd { id }).unwrap();
        assert_eq!(end["event"], "session_end");
        assert_eq!(end["id"], id_str);
    }

    /// Test-only hook that declares a fixed `SnapshotOpts`. Used to
    /// exercise `FanOutHook::snapshot_opts` aggregation without needing
    /// a real `ProbeCtx` (which is non-exhaustive and can't be
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
    /// configured with. Trivial but catches accidental hardcoding /
    /// override of the `opts` field.
    #[test]
    fn streaming_recorder_advertises_its_opts() {
        let (bus, _rx) =
            tokio::sync::broadcast::channel::<StreamProbeMsg>(4);
        let id = uuid::Uuid::from_u128(0xDEADBEEF_DEADBEEF_DEADBEEF_DEADBEEFu128);
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
