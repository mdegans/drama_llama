//! When blallama stops trusting its own process, it exits — and a
//! supervisor starts a fresh one.
//!
//! A panic anywhere, or a backend failure llama.cpp does not recover
//! from in-process (a Metal command buffer that ran out of memory
//! leaves its context "in error state … recreate the backend"; a
//! `llama_decode` that fails mid-batch leaves its ubatches in the KV),
//! is not something to unwind past and keep serving. llama.cpp gives
//! no guarantee its destructors clean up after such a failure, and the
//! 2026-10-01 cohort showed what serving on looks like: one Metal OOM,
//! then every later request failed until a manual restart. So the
//! process goes down, with the cause in one `ERROR` line, and comes
//! back clean under its supervisor (`scripts/blallama-supervise.sh`,
//! launchd, systemd).
//!
//! How it goes down:
//!
//! - **The panic hook** ([`install`]) handles a panic on *any* thread —
//!   a request's blocking session task, a tokio worker running a
//!   handler, a hyper connection task — not just the ones a `JoinError`
//!   would surface. It declares the process fatal and then parks the
//!   panicking thread instead of unwinding it: unwinding would drop
//!   whatever the thread owns, a `Session` included, through
//!   llama.cpp's destructors. A panic tokio would have caught and
//!   swallowed can no longer leave the server running.
//! - **A backend failure** ([`is_backend_failure`]) is declared by the
//!   request that met it, which leaks its session rather than drop it.
//! - **Declaring** ([`declare`]) logs the cause, wakes every request
//!   waiting on the blocking pool ([`declared`]) so it answers a 500
//!   `api_error` (which SDKs retry), refuses new requests the same way
//!   ([`refuse_when_fatal`]), and `_exit`s after [`GRACE`] — without
//!   `atexit` handlers or C++ static destructors, which on Metal assert
//!   that every buffer was freed (`ggml_metal_rsets_free`) and would
//!   turn the exit into a `SIGABRT`. Nothing waits on the answers: the
//!   exit is on its own thread and timer.
//!
//! The exit code tells a supervisor which: [`Fatal::Panic`] or
//! [`Fatal::Backend`]. Model loads and catalog reads are the one place a
//! panic is not fatal at once ([`caught_by_caller`]): the chat-template
//! analyzer deliberately catches the panics some templates raise in
//! minijinja. One that escapes still reaches [`declare`] through the
//! `JoinError`.

use std::{
    cell::Cell, io::Write as _, panic::PanicHookInfo, sync::OnceLock,
    time::Duration,
};

use axum::{
    extract::Request, http::StatusCode, middleware::Next, response::Response,
    Json,
};
use drama_llama::{prompt::AnthropicError, SessionError};
use tokio::sync::Notify;

use super::{error_response, ErrorEnvelope};

/// Why the process is going down, as its exit code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fatal {
    /// A panic: a bug, ours or a dependency's. `EX_SOFTWARE`.
    Panic = 70,
    /// The backend failed in a way the process does not recover from.
    /// `EX_TEMPFAIL`: a restart is expected to fix it.
    Backend = 75,
}

impl Fatal {
    /// The process exit code.
    pub(crate) const fn code(self) -> i32 {
        self as i32
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Panic => "panic",
            Self::Backend => "backend",
        }
    }
}

/// How long declared requests have to send their 500 before the exit.
pub(crate) const GRACE: Duration = Duration::from_millis(500);

/// The first cause declared; later ones are logged and ignored.
static DECLARED: OnceLock<Fatal> = OnceLock::new();
/// Wakes [`declared`] waiters.
static WOKEN: Notify = Notify::const_new();

thread_local! {
    /// Depth of [`caught_by_caller`] scopes on this thread.
    static CAUGHT: Cell<u32> = const { Cell::new(0) };
}

/// Install the panic hook. The previous hook still runs first, so the
/// panic message (and a backtrace under `RUST_BACKTRACE`) reaches
/// stderr as before.
pub(crate) fn install() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        previous(info);
        if CAUGHT.get() > 0 {
            return;
        }
        let (fatal, cause) = panic_cause(info);
        declare(fatal, &cause);
        hold_forever()
    }));
}

/// Never unwind: what this thread owns stays undropped until the exit.
/// A tokio worker first hands its scheduler to another thread
/// (`block_in_place`), so the tasks queued behind it — the answers to
/// requests in flight among them — still run.
fn hold_forever() -> ! {
    use tokio::runtime::{Handle, RuntimeFlavor};
    let hold = || loop {
        std::thread::park();
    };
    match Handle::try_current().map(|handle| handle.runtime_flavor()) {
        // Outside a worker (the blocking pool) it simply runs `hold`.
        Ok(RuntimeFlavor::MultiThread) => tokio::task::block_in_place(hold),
        _ => hold(),
    }
}

/// Run `f` with its panics left to its caller to catch, rather than
/// fatal at once — for the model reads whose chat-template analysis
/// catches the panics minijinja raises on some templates. A panic that
/// escapes `f` unwinds into a `JoinError`, which
/// [`super::spawn_blocking_or_bust`] declares.
pub(crate) fn caught_by_caller<R>(f: impl FnOnce() -> R) -> R {
    struct Scope;
    impl Drop for Scope {
        fn drop(&mut self) {
            CAUGHT.set(CAUGHT.get() - 1);
        }
    }
    CAUGHT.set(CAUGHT.get() + 1);
    let _scope = Scope;
    f()
}

/// A panic's message and location, and which kind of fatal it is. The
/// predictor still panics on a failed decode (#92), so those count as
/// the backend failures they are.
fn panic_cause(info: &PanicHookInfo<'_>) -> (Fatal, String) {
    let payload = info.payload();
    let message = payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("(non-string panic payload)");
    let location = info
        .location()
        .map(|l| format!(" at {}:{}", l.file(), l.line()))
        .unwrap_or_default();
    let thread = std::thread::current();
    let thread = thread.name().unwrap_or("unnamed");
    let cause = format!("thread '{thread}' panicked{location}: {message}");
    (panic_kind(message), cause)
}

/// [`Fatal::Backend`] for the predictor's decode-failure panics
/// (`src/predictor.rs`), [`Fatal::Panic`] for any other.
fn panic_kind(message: &str) -> Fatal {
    const DECODE_FAILURES: [&str; 2] = [
        "prefill failed in CandidatePredictor",
        "decoder.step failed",
    ];
    match DECODE_FAILURES.iter().any(|m| message.starts_with(m)) {
        true => Fatal::Backend,
        false => Fatal::Panic,
    }
}

/// A session error that leaves the backend untrustworthy. Every
/// [`SessionError::Decode`]: it carries the backend's message, not
/// whether the KV is dirty (an abort or a fatal `llama_decode`, or a
/// Metal context in error state, which no later call recovers), and a
/// restart costs a reload, never a wrong answer. And every error after
/// which the session itself is not reusable
/// ([`SessionError::is_fatal`]), which used to be dropped — through
/// llama.cpp's destructors — and reloaded.
pub(crate) fn is_backend_failure(error: &SessionError) -> bool {
    matches!(error, SessionError::Decode(_)) || error.is_fatal()
}

/// Declare the process fatal: start the exit timer, log the cause, and
/// wake every request waiting on [`declared`]. Only the first cause
/// counts; it is safe to call from any thread, a panic hook included.
pub(crate) fn declare(fatal: Fatal, cause: &dyn std::fmt::Display) {
    if DECLARED.set(fatal).is_err() {
        tracing::error!(
            event = "fatal",
            kind = fatal.name(),
            cause = %cause,
            "a further fatal error while exiting; ignored",
        );
        return;
    }
    // The timer first: the exit must not depend on logging succeeding.
    let reaper = std::thread::Builder::new()
        .name("blallama-fatal".into())
        .spawn(move || {
            std::thread::sleep(GRACE);
            exit_now(fatal)
        });
    tracing::error!(
        event = "fatal",
        kind = fatal.name(),
        exit_code = fatal.code(),
        cause = %cause,
        "the process can no longer be trusted; exiting for the supervisor \
         to restart it — requests in flight are answered 500 api_error",
    );
    flush();
    if reaper.is_err() {
        exit_now(fatal);
    }
    WOKEN.notify_waiters();
}

/// The fatal declared so far, if any.
pub(crate) fn current() -> Option<Fatal> {
    DECLARED.get().copied()
}

/// Resolves once the process is declared fatal.
pub(crate) async fn declared() -> Fatal {
    loop {
        let woken = WOKEN.notified();
        tokio::pin!(woken);
        // Registered before the check, so a declaration between the two
        // still wakes it.
        woken.as_mut().enable();
        if let Some(fatal) = current() {
            return fatal;
        }
        woken.await;
    }
}

/// What a request gets once the process is declared fatal: a 500
/// `api_error`, which Anthropic's SDKs retry — by then against the
/// restarted server.
pub(crate) fn reply(fatal: Fatal) -> (StatusCode, Json<ErrorEnvelope>) {
    error_response(AnthropicError::API {
        message: format!(
            "blallama is restarting after a fatal {} error; retry",
            fatal.name()
        ),
    })
}

/// Middleware: once the process is declared fatal, refuse every new
/// request rather than serve it from a process about to exit.
pub(crate) async fn refuse_when_fatal(
    request: Request,
    next: Next,
) -> Response {
    match current() {
        Some(fatal) => {
            axum::response::IntoResponse::into_response(reply(fatal))
        }
        None => next.run(request).await,
    }
}

fn flush() {
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
}

/// Exit at once: no unwinding, no `atexit` handlers, no C++ static
/// destructors — llama.cpp's included.
fn exit_now(fatal: Fatal) -> ! {
    flush();
    // SAFETY: `_exit` takes no pointers and does not return; skipping
    // every destructor is the point.
    unsafe { libc::_exit(fatal.code()) }
}

#[cfg(test)]
mod tests {
    //! Each test re-runs this test binary as a child process (the
    //! `CHILD` variable picks its scenario), so the child can install
    //! the hook, serve, and exit for real while the parent watches its
    //! exit code and output.

    use super::*;
    use axum::{routing::get, Router};

    const CHILD: &str = "BLALLAMA_FATAL_CHILD";

    /// Run this test binary on `test` (a test of this module) with
    /// `scenario`; returns the exit code and stdout.
    fn run_child(test: &str, scenario: &str) -> (Option<i32>, String) {
        let output = std::process::Command::new(
            std::env::current_exe().expect("test binary"),
        )
        .args(["--exact", &format!("fatal::tests::{test}"), "--nocapture"])
        .env(CHILD, scenario)
        .output()
        .expect("child runs");
        (
            output.status.code(),
            String::from_utf8_lossy(&output.stdout).into_owned(),
        )
    }

    /// A child: install the hook and the JSON log, serve `app` (behind
    /// the refusal middleware) on a local port, `GET` `/fault`, then —
    /// once the process is declared — `/ok`, print each answer that
    /// comes, and wait for the exit.
    fn serve_and_wait(app: Router) -> ! {
        super::super::init_logging();
        install();
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            let app = app
                .route("/ok", get(|| async { "served" }))
                .layer(axum::middleware::from_fn(refuse_when_fatal));
            let listener =
                tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            tokio::spawn(async move { axum::serve(listener, app).await });
            let answer = |path: &'static str| async move {
                let (head, body) = get_raw(addr, path).await;
                let status = head.split(' ').nth(1).unwrap_or("?").to_owned();
                println!("ANSWER {path} {status} {body}");
                let _ = std::io::stdout().flush();
            };
            tokio::spawn(answer("/fault"));
            declared().await;
            answer("/ok").await;
            std::future::pending::<()>().await;
        });
        unreachable!("the process exits");
    }

    async fn get_raw(
        addr: std::net::SocketAddr,
        path: &str,
    ) -> (String, String) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
        let request = format!(
            "GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
        );
        stream.write_all(request.as_bytes()).await.unwrap();
        let mut response = String::new();
        let _ = stream.read_to_string(&mut response).await;
        let (head, body) =
            response.split_once("\r\n\r\n").unwrap_or((&response, ""));
        (head.to_owned(), body.to_owned())
    }

    /// The parent's checks: the exit code, the one `ERROR` line naming
    /// the cause, and a 500 `api_error` for each of `answered` — the
    /// request after the declaration always, the one in flight when its
    /// handler could still answer.
    fn assert_exited(
        code: Option<i32>,
        stdout: &str,
        fatal: Fatal,
        cause: &str,
        answered: &[&str],
    ) {
        assert_eq!(code, Some(fatal.code()), "{stdout}");
        let fatal_lines: Vec<serde_json::Value> = stdout
            .lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .filter(|v: &serde_json::Value| v["fields"]["event"] == "fatal")
            .collect();
        assert_eq!(fatal_lines.len(), 1, "{stdout}");
        let line = &fatal_lines[0];
        assert_eq!(line["level"], "ERROR", "{line}");
        assert_eq!(line["fields"]["kind"], fatal.name(), "{line}");
        assert_eq!(line["fields"]["exit_code"], fatal.code(), "{line}");
        let logged = line["fields"]["cause"].as_str().unwrap_or_default();
        assert!(logged.contains(cause), "{line}");
        for path in answered {
            let answer = stdout
                .lines()
                .find_map(|l| l.strip_prefix(&format!("ANSWER {path} ")))
                .unwrap_or_else(|| panic!("no answer to {path}: {stdout}"));
            assert!(answer.starts_with("500 "), "{path}: {answer}");
            assert!(answer.contains(r#""type":"api_error""#), "{answer}");
        }
    }

    /// A panic in a request's blocking session task — the injected
    /// stand-in for a backend that panics — exits with the panic code,
    /// after answering that request and refusing the next one.
    #[test]
    fn a_panic_in_the_session_task_exits_with_the_panic_code() {
        if std::env::var_os(CHILD).is_some() {
            serve_and_wait(Router::new().route(
                "/fault",
                get(|| async {
                    super::super::spawn_blocking_or_bust(|| -> &str {
                        panic!("injected backend panic")
                    })
                    .await
                }),
            ));
        }
        let (code, stdout) = run_child(
            "a_panic_in_the_session_task_exits_with_the_panic_code",
            "1",
        );
        assert_exited(
            code,
            &stdout,
            Fatal::Panic,
            "injected backend panic",
            &["/fault", "/ok"],
        );
    }

    /// A panic in a handler on a tokio worker — which tokio would have
    /// caught, leaving the server up — exits all the same. That request
    /// goes unanswered (its thread is parked mid-poll); the next is
    /// refused.
    #[test]
    fn a_panic_in_a_handler_exits_with_the_panic_code() {
        if std::env::var_os(CHILD).is_some() {
            serve_and_wait(Router::new().route(
                "/fault",
                get(|| async {
                    if std::hint::black_box(true) {
                        panic!("injected handler panic");
                    }
                    "unreachable"
                }),
            ));
        }
        let (code, stdout) =
            run_child("a_panic_in_a_handler_exits_with_the_panic_code", "1");
        assert_exited(
            code,
            &stdout,
            Fatal::Panic,
            "injected handler panic",
            &["/ok"],
        );
    }

    /// Inside a model read ([`caught_by_caller`]) a panic the read
    /// catches itself — the chat-template analyzer's — is not fatal; one
    /// that escapes is, through its `JoinError`.
    #[test]
    fn a_panic_caught_in_a_model_read_is_not_fatal() {
        if std::env::var_os(CHILD).is_some() {
            serve_and_wait(Router::new().route(
                "/fault",
                get(|| async {
                    let caught = super::super::spawn_blocking_or_bust(|| {
                        caught_by_caller(|| {
                            std::panic::catch_unwind(|| {
                                panic!("a template the analyzer catches")
                            })
                            .is_err()
                        })
                    })
                    .await;
                    assert!(matches!(caught, Ok(true)));
                    assert_eq!(current(), None, "a caught panic is fatal");
                    super::super::spawn_blocking_or_bust(|| -> &str {
                        caught_by_caller(|| panic!("escaped model read"))
                    })
                    .await
                }),
            ));
        }
        let (code, stdout) =
            run_child("a_panic_caught_in_a_model_read_is_not_fatal", "1");
        assert_exited(
            code,
            &stdout,
            Fatal::Panic,
            "escaped model read",
            &["/fault", "/ok"],
        );
    }

    /// A fatal backend error (the error `complete` gets when a prefill's
    /// `llama_decode` fails) exits with the backend code.
    #[test]
    fn a_backend_failure_exits_with_the_backend_code() {
        if std::env::var_os(CHILD).is_some() {
            serve_and_wait(Router::new().route(
                "/fault",
                get(|| async {
                    let error = SessionError::Decode(
                        "`llama_decode` failed fatally (-3)".into(),
                    );
                    assert!(is_backend_failure(&error));
                    declare(Fatal::Backend, &error);
                    reply(Fatal::Backend)
                }),
            ));
        }
        let (code, stdout) =
            run_child("a_backend_failure_exits_with_the_backend_code", "1");
        assert_exited(
            code,
            &stdout,
            Fatal::Backend,
            "failed fatally (-3)",
            &["/fault", "/ok"],
        );
    }

    /// The predictor's decode-failure panics are backend failures; the
    /// strings are `src/predictor.rs`'s `expect` messages.
    #[test]
    fn decode_failure_panics_are_backend_failures() {
        for message in [
            "prefill failed in CandidatePredictor::new: Fatal { code: -3 }",
            "prefill failed in CandidatePredictor::new_resuming: Aborted",
            "decoder.step failed: NonFinite",
        ] {
            assert_eq!(panic_kind(message), Fatal::Backend, "{message}");
        }
        assert_eq!(panic_kind("index out of bounds"), Fatal::Panic);
        let error = SessionError::Media("bad image".into());
        assert!(!is_backend_failure(&error));
    }
}
