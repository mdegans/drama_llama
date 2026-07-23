//! Shared helpers for the examples — adapted from `misanthropic`'s
//! `examples/utils` (the examples here are ports of that crate's, driven
//! through a local [`SessionTransport`](drama_llama::SessionTransport)
//! instead of the API client, so no API key and no key helper).
//!
//! This is **not** an example target — it lives in a subdirectory with no
//! `main.rs`, so Cargo's example auto-discovery ignores it. Pull it into an
//! example with `mod utils;`.
#![allow(dead_code)]

// The chat driver is misanthropic crate API (promoted there in #104);
// re-exported so ported call sites read unchanged.
#[allow(unused_imports)]
pub use misanthropic::chat::{BoxError, BudgetPolicy, Chat};

// Shared CLI flags — pure data + `Prompt`/`Session` mapping.
mod args;
#[allow(unused_imports)]
pub use args::{
    Args, BlockingTransport, BlockingTransportError, BuildError, ChatArgs,
    CommonArgs, TransportBuilder,
};

/// Initialize logging for an example. With `verbose`, the examples and the
/// chat driver log at `debug` (otherwise `warn`); `RUST_LOG` overrides
/// either. Safe to call once.
///
/// The verbose filter is *scoped* rather than a global `debug` default —
/// a global default also turns on rustyline's per-keypress debug logging,
/// which floods the readline examples (one log line per key).
///
/// # Call this before loading a model
///
/// This also routes each compiled-in backend's native log stream into the
/// `log` crate, which is the whole reason it has to run *first*: llama.cpp
/// is at its loudest during model load, so a sink installed afterwards
/// (which is all `Session::quiet` could ever be) never sees the noisy part.
///
/// With the bridge in place `RUST_LOG` governs it like anything else. The
/// backends log under the `llama_cpp` target, which is deliberately *not*
/// in the `--verbose` set — a model load at `debug` is hundreds of lines
/// nobody asked for. Opt in explicitly with `RUST_LOG=llama_cpp=debug`
/// (the load-progress dots arrive as `trace`, so even that stays
/// readable). Warnings and errors from the backend do now reach the
/// terminal by default, which `Session::quiet` used to swallow.
pub fn log_init(verbose: bool) {
    let default = if verbose {
        // Example binaries log under their own name; cover the set plus
        // the crates whose narration is actually wanted. (Miss one and
        // its narration silently vanishes: council's `jester ▸` lines
        // were invisible for the 2026-07-18 run-two postmortem —
        // exactly the output that would have shown what the dead seat
        // actually emitted.)
        "warn,swarm=debug,python=debug,council=debug,chat=debug,\
         misanthropic=debug,drama_llama=debug"
    } else {
        "warn"
    };
    let env = env_logger::Env::default().default_filter_or(default);
    // `try_init` so a second call (or a pre-installed logger) is a no-op.
    let _ = env_logger::Builder::from_env(env).try_init();

    install_backend_log_bridges();
}

/// Point every compiled-in backend's native log sink at the `log` crate.
///
/// Gated by `cfg` rather than dispatched on `--backend`: installing a sink
/// only affects the library that owns it, so doing it for everything linked
/// is both simpler and correct — and it can happen before the flags are
/// even parsed. Backends with no sink of their own answer
/// `Err(NotImplemented)`, which is the expected reply and not worth
/// reporting; `moeflux` logs through `tracing`, so `RUST_LOG` already
/// governs it.
fn install_backend_log_bridges() {
    use drama_llama::LogLevel;

    fn bridge(level: LogLevel, text: &str) {
        // `Cont` is llama.cpp's load-progress dots — one call per dot.
        // Folding them into `trace` keeps a progress bar from becoming
        // several hundred log records at `debug`.
        let level = match level {
            LogLevel::Error => log::Level::Error,
            LogLevel::Warn => log::Level::Warn,
            LogLevel::Info => log::Level::Info,
            LogLevel::Debug | LogLevel::None | LogLevel::Other(_) => {
                log::Level::Debug
            }
            LogLevel::Cont => log::Level::Trace,
            // `LogLevel` is `#[non_exhaustive]`; treat unknown future
            // levels like `Other`.
            _ => log::Level::Debug,
        };
        // llama.cpp terminates its own lines; `log` adds another.
        let text = text.trim_end_matches('\n');
        if !text.is_empty() {
            log::log!(target: "llama_cpp", level, "{text}");
        }
    }

    #[cfg(feature = "llama-cpp")]
    {
        use drama_llama::Backend as _;
        let _ = drama_llama::LlamaCppBackend::set_log_callback(bridge);
    }
    #[cfg(all(feature = "moeflux", target_os = "macos"))]
    {
        use drama_llama::Backend as _;
        let _ = drama_llama::MoefluxBackend::set_log_callback(bridge);
    }
}

/// Spawn a dedicated thread running a `rustyline` prompt and return the
/// entered lines as a channel plus a [`Printer`] for the async side to print
/// *through*.
///
/// # Why a thread *and* a printer
///
/// A line editor owns the terminal line while it waits, so anything else that
/// writes stdout collides with what the user is typing (the `you ▸ claude ▸ …`
/// glitch). Two moves fix it: keep the blocking `readline` on its own thread
/// (out of the `select!` future tree — its `mpsc` `recv()` is cancel-safe),
/// and route **all** async output through the returned [`Printer`]
/// (rustyline's `ExternalPrinter`), which erases the prompt, prints above it,
/// and redraws it with the user's in-progress input intact.
#[cfg(feature = "repl")]
pub fn spawn_readline_loop(
    prompt: impl Into<String>,
) -> rustyline::Result<(tokio::sync::mpsc::Receiver<String>, Printer)> {
    use rustyline::{error::ReadlineError, DefaultEditor};

    let prompt = prompt.into();
    let mut editor = DefaultEditor::new()?;
    let printer = Printer(Box::new(editor.create_external_printer()?));
    let (tx, rx) = tokio::sync::mpsc::channel::<String>(16);

    std::thread::spawn(move || {
        loop {
            match editor.readline(&prompt) {
                Ok(line) if line.trim().is_empty() => continue,
                Ok(line) => {
                    editor.add_history_entry(&line).ok();
                    // Receiver gone (the driver exited): stop reading.
                    if tx.blocking_send(line).is_err() {
                        break;
                    }
                }
                Err(ReadlineError::Interrupted) => continue, // Ctrl-C: ignore
                Err(ReadlineError::Eof) => break,            // Ctrl-D: quit
                Err(_) => break,
            }
        }
        // Dropping `tx` signals EOF to the consumer's `recv()`.
    });

    Ok((rx, printer))
}

/// Prints *above* the live `rustyline` prompt. Use this for all async output
/// (model replies, notifications) instead of `println!`, which would collide
/// with the user's in-progress input. Returned by [`spawn_readline_loop`].
#[cfg(feature = "repl")]
pub struct Printer(Box<dyn rustyline::ExternalPrinter + Send>);

#[cfg(feature = "repl")]
impl Printer {
    /// Print `msg` followed by a newline, above the prompt.
    pub fn line(&mut self, msg: impl std::fmt::Display) {
        // The editor thread performs the redraw; if it's gone, drop the line.
        let _ = self.0.print(format!("{msg}\n"));
    }
}
