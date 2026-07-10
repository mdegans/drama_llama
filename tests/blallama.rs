//! End-to-end HTTP tests for the `blallama` binary: spawn the real
//! server against `models/`, then drive it with `misanthropic`'s
//! `Client` pointed at the local port — the same client a real
//! consumer uses against api.anthropic.com.
//!
//! Covers `/api/tags` discovery, the `/v1/messages` happy path, and
//! cross-request prompt caching through the server's shared session
//! (the endpoint-level analog of `tests/session_cache.rs`).
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
    fn drop(&mut self) {
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
