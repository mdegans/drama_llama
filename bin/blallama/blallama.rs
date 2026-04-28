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
    ProbeCtx, ProbeHook, Prompt, Session,
};
use std::num::NonZeroU128;
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
    /// Append per-token probe records to this JSONL file. First line
    /// per session is `{"event":"session_start","model":"<name>"}`;
    /// subsequent lines are `{"event":"token","token":N,"n_cur":M,
    /// "ts_ms":T}`. `ts_ms` is relative to the moment the hook was
    /// installed on the session. Omit to disable probe recording.
    #[arg(long)]
    probe_out: Option<PathBuf>,
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
    /// Sender into the probe-writer task. `None` if `--probe-out`
    /// wasn't given. Cloned per-session in `configure_session` so each
    /// installed [`JsonlProbeRecorder`] has its own handle; all clones
    /// feed the same consumer task / output file.
    probe_tx: Option<tokio::sync::mpsc::Sender<ProbeMsg>>,
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
        probe_tx: Option<tokio::sync::mpsc::Sender<ProbeMsg>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let listener = tokio::net::TcpListener::bind(format!(
            "0.0.0.0:{port}",
            port = args.port
        ))
        .await?;

        let session: Arc<Mutex<Option<Session<LlamaCppBackend>>>> =
            Mutex::from(None).into();

        let app = Router::new()
            .route("/v1/messages", post(route_messages))
            .with_state(AppState {
                args: args.into(),
                probe_tx,
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
                        state.probe_tx.clone(),
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
                    state.probe_tx.clone(),
                )
                .await?
            }
        };

        let (session, response, elapsed) = spawn_blocking_or_bust(move || {
            let start = std::time::Instant::now();
            let response = session.complete_response(&prompt).map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(AnthropicError::Unknown {
                        code: Some(500.try_into().unwrap()),
                        message: e.to_string(),
                    }),
                )
            })?;
            let elapsed = start.elapsed();
            Ok((session, response, elapsed))
        })
        .await?;

        log_stats(&response.id, response.usage, elapsed);
        lock.replace(session);
        Ok(Json(response))
    }

    async fn load_session(
        root: impl AsRef<Path>,
        model: String,
        no_penalty: bool,
        seed: Option<u128>,
        probe_tx: Option<tokio::sync::mpsc::Sender<ProbeMsg>>,
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
        .map(|s| configure_session(s, no_penalty, seed, probe_tx))
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
        probe_tx: Option<tokio::sync::mpsc::Sender<ProbeMsg>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let listener = tokio::net::TcpListener::bind(format!(
            "0.0.0.0:{port}",
            port = args.port
        ))
        .await?;

        let session: Arc<Mutex<Option<Session<MoefluxBackend>>>> =
            Mutex::from(None).into();

        let app = Router::new()
            .route("/v1/messages", post(route_messages))
            .with_state(AppState {
                args: args.into(),
                probe_tx,
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
                        state.probe_tx.clone(),
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
                    state.probe_tx.clone(),
                )
                .await?
            }
        };

        let (session, response, elapsed) = spawn_blocking_or_bust(move || {
            let start = std::time::Instant::now();
            let response = session.complete_response(&prompt).map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(AnthropicError::Unknown {
                        code: Some(500.try_into().unwrap()),
                        message: e.to_string(),
                    }),
                )
            })?;
            let elapsed = start.elapsed();
            Ok((session, response, elapsed))
        })
        .await?;

        log_stats(&response.id, response.usage, elapsed);
        lock.replace(session);
        Ok(Json(response))
    }

    async fn load_session(
        root: impl AsRef<Path>,
        model: String,
        no_penalty: bool,
        seed: Option<u128>,
        probe_tx: Option<tokio::sync::mpsc::Sender<ProbeMsg>>,
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
        .map(|s| configure_session(s, no_penalty, seed, probe_tx))
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
    probe_tx: Option<tokio::sync::mpsc::Sender<ProbeMsg>>,
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
    let mut configured = with_penalty
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
    if let Some(tx) = probe_tx {
        let model_name = configured
            .engine()
            .model
            .display_name()
            .unwrap_or_default();
        let recorder = JsonlProbeRecorder::install(tx, model_name.as_str());
        configured
            .engine_mut()
            .set_probe_hook(Some(Box::new(recorder)));
        tracing::info!(event = "probe_hook_installed", model = model_name.as_str());
    }
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
        let mut writer = tokio::io::BufWriter::new(file);
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
            if let Err(e) = writer.write_all(line.as_bytes()).await {
                tracing::warn!(event = "probe_write_failed", error = %e);
                continue;
            }
            if let Err(e) = writer.write_all(b"\n").await {
                tracing::warn!(event = "probe_write_failed", error = %e);
            }
        }
        // Channel closed (all senders dropped). Flush anything still
        // in the BufWriter before exiting.
        let _ = writer.flush().await;
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logging();
    let args = Args::parse();

    // If --probe-out is set, spin up the writer task before any
    // session loads so per-session installs always have a Sender to
    // clone. Failure to open the file is a startup error — the user
    // asked for probe records and we can't deliver them.
    let probe_tx = if let Some(path) = args.probe_out.clone() {
        Some(spawn_probe_writer(path).await?)
    } else {
        None
    };

    match args.backend {
        #[cfg(feature = "llama-cpp")]
        BackendKind::LlamaCpp => llama_cpp_run::run(args, probe_tx).await,
        #[cfg(all(feature = "moeflux", target_os = "macos"))]
        BackendKind::Moeflux => moeflux_run::run(args, probe_tx).await,
    }
}
