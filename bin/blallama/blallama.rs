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
use clap::Parser;
use drama_llama::{
    prompt::{AnthropicError, MessageResponse, Usage},
    IgnoreCategory, Prompt, RepetitionOptions, Session,
};
use tokio::{sync::Mutex, task::spawn_blocking};
use tracing::{error, info, instrument};

#[derive(Parser)]
#[command(about = "Demo /v1/messages server")]
struct Args {
    /// Path containing model files
    model_path: PathBuf,
    /// Port to use
    #[arg(long, default_value_t = 11435)]
    port: u16,
}

#[derive(Clone)]
struct AppState {
    args: Arc<Args>,
    session: Arc<Mutex<Option<Session>>>,
}

/// Get available .gguf models in a given path
async fn get_available_models(
    path: impl AsRef<Path>,
) -> Result<Vec<String>, std::io::Error> {
    let mut read_dir = tokio::fs::read_dir(path).await?;
    let mut models = vec![];
    while let Some(model) = read_dir.next_entry().await? {
        if !model.file_type().await?.is_file() {
            continue;
        }
        let model = if let Ok(model) = model.file_name().into_string() {
            model
        } else {
            continue;
        };
        models.push(model)
    }

    Ok(models)
}

/// `/v1/messages` handler
async fn route_messages(
    State(state): State<AppState>,
    Json(prompt): Json<Prompt>,
) -> Result<Json<MessageResponse>, (StatusCode, Json<AnthropicError>)> {
    // List available models
    let models = match get_available_models(&state.args.model_path).await {
        Ok(models) => models,
        Err(e) => {
            let e = AnthropicError::NotFound {
                message: format!("Models could not be loaded: {e}"),
            };
            error!(error = %e);
            return Err((StatusCode::NOT_FOUND, Json(e)));
        }
    };

    // Check model exists in our models path
    if !models.contains(&prompt.model.to_string()) {
        let e = AnthropicError::NotFound {
            message: format!("model not found: {model}", model = prompt.model),
        };
        error!(error = %e);
        return Err((StatusCode::NOT_FOUND, Json(e)));
    }

    complete(state, prompt).await
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

#[instrument(skip(state, prompt), fields(model = %prompt.model))]
async fn complete(
    state: AppState,
    prompt: Prompt,
) -> Result<Json<MessageResponse>, (StatusCode, Json<AnthropicError>)> {
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

    // Load session if necessary. Load will briefly block the executor. Use
    // spawn_blocking in production.
    let mut session = match lock.take() {
        Some(session) => {
            // Get model filename. Can't panic since model was loaded from file
            // and a file_name exists.
            let file_name = session.engine().model.file_name().unwrap();
            if file_name
                .to_str()
                .is_some_and(|s| s == prompt.model.to_string())
            {
                session
            } else {
                load_session(&state.args.model_path, prompt.model.to_string())
                    .await?
            }
        }
        None => {
            load_session(&state.args.model_path, prompt.model.to_string())
                .await?
        }
    };

    // `complete_response` is blocking
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
) -> Result<Session, (StatusCode, Json<AnthropicError>)> {
    // Containment in the directory-listed models rules out path traversal
    let path = root.as_ref().join(&model);
    tracing::info!(
        event = "load_model",
        model,
        path = path.to_string_lossy().as_ref()
    );
    // `from_path` is blocking
    spawn_blocking_or_bust(|| Session::from_path_with_n_ctx(path, 65536))
        .await
        .map(|s| {
            s.with_repetition(
                RepetitionOptions::default().set_ignored_categories([
                    IgnoreCategory::English,
                    IgnoreCategory::Json,
                    IgnoreCategory::Punctuation,
                ]),
            )
            .with_prefix_cache(true)
            .with_max_tokens(8192.try_into().unwrap())
        })
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(AnthropicError::Unknown {
                    code: Some(500.try_into().unwrap()),
                    message: e.to_string(),
                }),
            )
        })
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logging();
    let args = Args::parse();

    // fail fast stuff first
    let listener = tokio::net::TcpListener::bind(format!(
        "0.0.0.0:{port}",
        port = args.port
    ))
    .await?;

    // Inference engine, lazily loaded
    let session: Arc<Mutex<Option<Session>>> = Mutex::from(None).into();

    let app = Router::new()
        .route("/v1/messages", post(route_messages))
        .with_state(AppState {
            args: args.into(),
            session,
        });
    axum::serve(listener, app).await?;

    Ok(())
}
