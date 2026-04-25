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
    IgnoreCategory, Prompt, RepetitionOptions, Session,
};
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

    pub async fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
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
                    )
                    .await?
                }
            }
            None => {
                load_session(&state.args.model_path, prompt.model.to_string())
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
        .map(configure_session)
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

    pub async fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
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
                    )
                    .await?
                }
            }
            None => {
                load_session(&state.args.model_path, prompt.model.to_string())
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
        .map(configure_session)
        .map_err(map_session_err)
    }
}

// ---------------------------------------------------------------------------
// Shared session post-load configuration
// ---------------------------------------------------------------------------

fn configure_session<B: Backend>(s: Session<B>) -> Session<B> {
    let configured = s
        .with_repetition(
            RepetitionOptions::default().set_ignored_categories([
                IgnoreCategory::English,
                IgnoreCategory::Json,
                IgnoreCategory::Punctuation,
            ]),
        )
        .with_prefix_cache(true)
        // Session-level generation cap. Not n_ctx — that's the KV
        // context window, set per-backend at engine construction
        // (llama.cpp: `from_path_with_n_ctx(_, 65536)`; moeflux:
        // compile-time per model variant, exposed via
        // `engine().n_ctx()`). We pin the session cap to the same
        // 65536 ceiling so the prompt's `max_tokens` always wins
        // unless it explicitly asks for more than 64K, which would
        // exceed any current model's context anyway.
        .with_max_tokens(65536.try_into().unwrap());
    tracing::info!(
        event = "session_ready",
        n_ctx = configured.engine().n_ctx(),
        session_max_tokens = 65536u32,
        model = configured
            .engine()
            .model
            .display_name()
            .unwrap_or_default()
            .as_str(),
    );
    configured
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

    match args.backend {
        #[cfg(feature = "llama-cpp")]
        BackendKind::LlamaCpp => llama_cpp_run::run(args).await,
        #[cfg(all(feature = "moeflux", target_os = "macos"))]
        BackendKind::Moeflux => moeflux_run::run(args).await,
    }
}
