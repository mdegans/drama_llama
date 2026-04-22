use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use axum::{
    extract::{Json, State},
    http::StatusCode,
    routing::post,
    Router,
};
use clap::Parser;
use drama_llama::{
    prompt::{AnthropicError, MessageResponse},
    Prompt, Session,
};
use tokio::{sync::Mutex, task::spawn_blocking};

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

    // List available models
    let models = match get_available_models(&state.args.model_path).await {
        Ok(models) => models,
        Err(e) => {
            return Err((
                StatusCode::NOT_FOUND,
                Json(AnthropicError::NotFound {
                    message: format!("Models could not be loaded: {e}"),
                }),
            ))
        }
    };

    // Check model exists in our models path
    if !models.contains(&prompt.model.to_string()) {
        return Err((
            StatusCode::NOT_FOUND,
            Json(AnthropicError::NotFound {
                message: format!(
                    "model not found: {model}",
                    model = prompt.model
                ),
            }),
        ));
    }

    // Containment in the directory-listed models rules out path traversal
    let path = state.args.model_path.join(prompt.model.to_string());

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
                Ok(session)
            } else {
                // `from_path` is blocking
                spawn_blocking(|| Session::from_path(path)).await.unwrap()
            }
        }
        None => spawn_blocking(|| Session::from_path(path)).await.unwrap(),
    }
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(AnthropicError::Unknown {
                code: Some(500.try_into().unwrap()),
                message: e.to_string(),
            }),
        )
    })?;

    // `complete_response` is blocking
    let (session, response) = spawn_blocking(move || {
        let response = session.complete_response(&prompt).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(AnthropicError::Unknown {
                    code: Some(500.try_into().unwrap()),
                    message: e.to_string(),
                }),
            )
        })?;

        Ok((session, response))
    })
    .await
    .unwrap()?;

    // Put session back, return the message, drop the lock
    lock.replace(session);
    Ok(Json(response))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
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
