//! End-to-end HTTP tests for the `blallama` binary: spawn the real
//! server against `models/`, then drive it with `misanthropic`'s
//! `Client` pointed at the local port — the same client a real
//! consumer uses against api.anthropic.com.
//!
//! Covers `/v1/models` and `/api/tags` discovery, the `/v1/messages`
//! happy path, and cross-request prompt caching through the server's
//! shared session (the endpoint-level analog of `tests/session_cache.rs`).
//!
//! All tests need a GGUF in `models/`: `cargo test --test blallama --
//! --ignored`.

use std::{
    io::{Read as _, Write as _},
    net::{TcpListener, TcpStream},
    path::PathBuf,
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

use misanthropic::{prompt::message::Role, Client, Prompt};

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models")
}

/// Kills the server on drop so a failing assertion doesn't leak a
/// GPU-resident process into the next test.
struct Server {
    child: Child,
    port: u16,
}

impl Drop for Server {
    /// SIGTERM, wait, and only then SIGKILL.
    ///
    /// This used to be a bare `Child::kill()`, which is SIGKILL and is
    /// uncatchable. The LLVM coverage runtime writes its `.profraw` from an
    /// `atexit` handler, so a SIGKILL'd server writes nothing — every line
    /// these tests drive through the real binary was absent from coverage,
    /// and blallama scored an identical 17.64% whether this tier ran or was
    /// skipped outright. `blallama` now drains on SIGTERM and returns from
    /// `main`, which is what lets the handler fire.
    ///
    /// The SIGKILL fallback stays: a server that will not drain must not
    /// wedge the suite, and a leaked child would hold the port.
    fn drop(&mut self) {
        #[cfg(unix)]
        // SAFETY: `kill(2)` with a pid we own and a valid signal number.
        // The child cannot have been reaped yet — nothing else calls
        // `wait` on it, and this is the only `Drop`.
        unsafe {
            libc::kill(self.child.id() as libc::pid_t, libc::SIGTERM);
        }

        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            match self.child.try_wait() {
                Ok(Some(_)) => return,
                Ok(None) => std::thread::sleep(Duration::from_millis(50)),
                Err(_) => break,
            }
        }

        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn spawn_server() -> Server {
    // Bind-then-drop to pick a free port. Racy in principle; fine for
    // a test that runs alone on a dev box.
    let port = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let child = Command::new(env!("CARGO_BIN_EXE_blallama"))
        .arg(models_dir())
        .args(["--port", &port.to_string(), "--seed", "42", "--no-penalty"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn blallama");

    // The server binds before loading any model (sessions load lazily
    // on the first request), so readiness is just the port accepting.
    let deadline = Instant::now() + Duration::from_secs(30);
    while TcpStream::connect(("127.0.0.1", port)).is_err() {
        assert!(
            Instant::now() < deadline,
            "blallama did not bind port {port} within 30s"
        );
        std::thread::sleep(Duration::from_millis(200));
    }
    Server { child, port }
}

/// Minimal HTTP/1.0 GET — 1.0 so the body arrives unchunked and the
/// connection closes, keeping parsing trivial without an HTTP client
/// dependency.
fn http_get(port: u16, path: &str) -> String {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
    write!(stream, "GET {path} HTTP/1.0\r\nHost: localhost\r\n\r\n").unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).unwrap();
    response
        .split_once("\r\n\r\n")
        .map(|(_, body)| body.to_string())
        .unwrap_or_default()
}

/// Discover a servable model name from `/api/tags` (entry names in
/// `models/`, e.g. `some-model.gguf`).
fn first_model(port: u16) -> String {
    let body = http_get(port, "/api/tags");
    let v: serde_json::Value =
        serde_json::from_str(&body).expect("tags returns JSON");
    v["models"][0]["name"]
        .as_str()
        .expect("at least one .gguf in models/")
        .to_string()
}

fn client(port: u16) -> Client {
    // blallama ignores auth; the key only has to satisfy the client's
    // length validation.
    Client::new("x".repeat(108))
        .expect("client")
        .base_url(format!("http://127.0.0.1:{port}"))
        .expect("base url")
}

#[test]
#[ignore = "long running, requires a GGUF in models/"]
fn tags_lists_models() {
    let server = spawn_server();
    let body = http_get(server.port, "/api/tags");
    let v: serde_json::Value = serde_json::from_str(&body).expect("JSON");
    let models = v["models"].as_array().expect("models array");
    assert!(!models.is_empty(), "no models listed from models/");
    assert!(models[0]["name"]
        .as_str()
        .is_some_and(|n| n.ends_with(".gguf")));
}

/// `/v1/models` through the real client's `models()` — the consumer
/// path, so this also proves the wire shape parses — plus the per-id
/// route and its 404. Every model in `models/` is listed with metadata
/// read from disk, none of them loaded.
#[tokio::test]
#[ignore = "long running, requires a GGUF in models/"]
async fn v1_models_lists_every_model_unloaded() {
    let server = spawn_server();
    let client = client(server.port);

    let started = Instant::now();
    let models = client.models().await.expect("GET /v1/models");
    let cold = started.elapsed();
    assert!(!models.is_empty(), "no models listed from models/");
    for info in &models {
        assert!(info.id.name().ends_with(".gguf"), "{}", info.id);
        assert!(!info.display_name.is_empty(), "{}", info.id);
        assert!(info.max_input_tokens > 0, "{}: no context ceiling", info.id);
        assert_eq!(info.max_tokens, info.max_input_tokens, "{}", info.id);
        assert!(info.capabilities.structured_outputs == true, "{}", info.id);
    }
    // Same set as the ollama-shaped listing, in id order.
    let tagged: Vec<String> = {
        let body = http_get(server.port, "/api/tags");
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        v["models"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["name"].as_str().unwrap().to_string())
            .collect()
    };
    let listed: Vec<String> =
        models.iter().map(|m| m.id.name().to_string()).collect();
    assert_eq!(listed, tagged);
    assert!(listed.windows(2).all(|w| w[0] <= w[1]), "not sorted");

    // Second listing is served from the catalog's cache. Not asserted
    // on (a one-model CI box makes both calls fast); visible with
    // `--nocapture`.
    let started = Instant::now();
    let again = client.models().await.expect("second GET /v1/models");
    eprintln!(
        "/v1/models: {} models, cold {cold:?}, cached {:?}",
        again.len(),
        started.elapsed()
    );
    assert_eq!(again.len(), models.len());

    // Per-id route round-trips the listing entry.
    let first = &models[0];
    let body = http_get(server.port, &format!("/v1/models/{}", first.id));
    let one: misanthropic::model::ModelInfo =
        serde_json::from_str(&body).expect("ModelInfo JSON");
    assert_eq!(one.id, first.id);
    assert_eq!(one.display_name, first.display_name);
    assert_eq!(one.max_input_tokens, first.max_input_tokens);
    assert_eq!(one.created_at, first.created_at);

    // Unknown id: the same envelope `/v1/messages` uses.
    let body =
        http_get(server.port, "/v1/models/claude-definitely-not-on-disk");
    let v: serde_json::Value = serde_json::from_str(&body).expect("JSON");
    assert_eq!(v["type"], "error");
    assert_eq!(v["error"]["type"], "not_found_error");
    assert!(v["error"]["message"]
        .as_str()
        .is_some_and(|m| m.contains("model not found")));
}

/// Unknown model id with no `--default-model` → Anthropic-shaped 404.
#[tokio::test]
#[ignore = "long running, requires a GGUF in models/"]
async fn unknown_model_is_not_found() {
    use misanthropic::client::{AnthropicError, Error};

    let server = spawn_server();
    let client = client(server.port);
    let prompt = Prompt::default()
        .model("claude-definitely-not-on-disk")
        .add_message((Role::User, "hi"))
        .unwrap();
    match client.message(&prompt).await {
        Err(Error::Anthropic(AnthropicError::NotFound { message })) => {
            assert!(message.contains("model not found"), "{message}");
        }
        other => panic!("expected NotFound, got {other:?}"),
    }
}

/// The core end-to-end flow: `/v1/messages` completes, reports usage,
/// and the shared session's prefix cache carries across requests —
/// request 2 extends request 1's conversation and must report
/// `cache_read_input_tokens > 0`.
#[tokio::test]
#[ignore = "long running, requires a GGUF in models/"]
async fn messages_completes_and_caches_across_requests() {
    let server = spawn_server();
    let model = first_model(server.port);
    let client = client(server.port);

    // `.cache()` marks the tail block — the breakpoint drama_llama
    // anchors reuse on, exactly as a caching client would send it.
    let mut chat = Prompt::default()
        .model(model.clone())
        .max_tokens(64.try_into().unwrap())
        .system("You are a concise assistant. Answer in one sentence.")
        .add_message((Role::User, "Name a primary color."))
        .unwrap()
        .cache();

    // Request 1: model loads lazily, so give it time.
    let r1 = client.message(&chat).await.expect("request 1");
    assert_eq!(r1.model.to_string(), model);
    assert!(!r1.inner.content.to_string().trim().is_empty());
    assert!(r1.usage.input_tokens > 0);
    assert_eq!(
        r1.usage.cache_read_input_tokens,
        Some(0),
        "first request has nothing to reuse"
    );

    // Request 2: same conversation extended — the server-side session
    // must reuse the request-1 prefix.
    chat.push_message(r1).expect("push assistant turn");
    chat = chat
        .add_message((Role::User, "Now name a shape."))
        .expect("push user turn")
        .cache();

    let r2 = client.message(&chat).await.expect("request 2");
    let read = r2.usage.cache_read_input_tokens.unwrap_or(0);
    assert!(
        read > 0,
        "request 2 extends request 1's conversation; the server \
         session must reuse its prefix (cache_read={read}, \
         input={})",
        r2.usage.input_tokens
    );
    assert!(read < r2.usage.input_tokens);
}

/// `count_tokens` counts exactly what `/v1/messages` prefills: for the
/// same body, the count equals the completion's `input_tokens`, which in
/// blallama is the whole prompt (the cache read and write are breakdowns
/// of it, unlike Anthropic's usage, where they are added on top).
/// Pinned to `model.gguf` so it loads the test model rather than
/// whichever file lists first.
#[tokio::test]
#[ignore = "long running, requires a GGUF in models/"]
async fn count_tokens_matches_messages_input() {
    let server = spawn_server();
    let client = client(server.port);
    let prompt = Prompt::default()
        .model("model.gguf")
        .max_tokens(16.try_into().unwrap())
        .system("You are a concise assistant.")
        .add_message((Role::User, "Name a primary color."))
        .unwrap()
        .cache();

    let counted = client.count_tokens(&prompt).await.expect("count_tokens");
    let response = client.message(&prompt).await.expect("messages");
    assert!(counted > 0);
    assert_eq!(
        u64::from(counted),
        response.usage.input_tokens,
        "usage: {:?}",
        response.usage
    );
}

/// A request whose input + `max_tokens` overruns the context is refused
/// up front with Anthropic's exact 400, which clients match on: the
/// reported number is input + `max_tokens`. The session survives, so
/// the next request that fits completes.
#[tokio::test]
#[ignore = "long running, requires a GGUF in models/"]
async fn context_overflow_is_anthropic_400() {
    use misanthropic::client::{AnthropicError, Error};

    let server = spawn_server();
    let client = client(server.port);
    let prompt = Prompt::default()
        .model("model.gguf")
        .add_message((Role::User, "Name a primary color."))
        .unwrap();

    // Load the model, so the listing reports the live context size.
    let input = client.count_tokens(&prompt).await.expect("count_tokens");
    let v: serde_json::Value =
        serde_json::from_str(&http_get(server.port, "/v1/models/model.gguf"))
            .expect("model JSON");
    let n_ctx = v["max_input_tokens"].as_u64().expect("max_input_tokens");

    let too_long = prompt
        .clone()
        .max_tokens(u32::try_from(n_ctx).unwrap().try_into().unwrap());
    match client.message(&too_long).await {
        Err(Error::Anthropic(AnthropicError::InvalidRequest { message })) => {
            assert_eq!(
                message,
                format!(
                    "prompt is too long: {} tokens > {n_ctx} maximum",
                    input as u64 + n_ctx
                )
            );
        }
        other => panic!("expected InvalidRequest, got {other:?}"),
    }

    let fits = prompt.max_tokens(8.try_into().unwrap());
    client
        .message(&fits)
        .await
        .expect("a fitting request completes");
}
