# Future work — graceful shutdown for blallama

Captured 2026-04-29 during slice-2A landing of `/probe` SSE.

## Problem

`bin/blallama/blallama.rs` doesn't have a SIGTERM/Ctrl-C handler.
On signal, the tokio runtime aborts in place, dropping in-flight
work without orderly cleanup. Two consequences observed during
slice-2A:

1. **`spawn_probe_writer`'s BufWriter never flushed** under SIGKILL
   because the writer task only flushed after all senders dropped,
   which only happened on graceful runtime exit. Worked around by
   dropping the BufWriter and writing unbuffered — correct but
   loses any future buffered-write opportunity.

2. **In-flight `/v1/messages` requests** terminate mid-generation
   with no client-visible response. Acceptable for dev work; not
   acceptable in any deployed canary-suite scheduler.

## Shape of the fix

`axum::serve(listener, app).with_graceful_shutdown(signal_future)`
where `signal_future` awaits `tokio::signal::ctrl_c()` (and on
unix, also SIGTERM via `tokio::signal::unix::signal`). The serve
future then:

1. Stops accepting new connections.
2. Awaits in-flight handlers up to a deadline.
3. Returns.

Then in `main`, after `axum::serve(...).await?`:

- Drop `AppState` (drops `record_json_tx` and `probe_bus` Senders).
- Await any spawned writer/broadcast tasks via a `JoinSet` or
  per-task `JoinHandle`s collected at startup.
- Once tasks have drained their channels and exited, return.

Once the shutdown sequence is in place, the BufWriter wrap can be
restored to `spawn_probe_writer` (fewer syscalls under load) since
it'll always get its trailing `flush().await` before the runtime
exits.

## Out-of-scope cousins worth thinking about together

- **In-flight Session lock release.** A request holding the
  `Mutex<Option<Session>>` on shutdown leaves the session in an
  unknown state. Graceful shutdown should let the request finish;
  ungraceful should be rare enough to ignore.
- **Probe consumers connected to /probe at shutdown time.** They'd
  see the SSE stream close. That's correct behavior — server is
  going away — but worth a `session_end`-style "shutdown" event so
  consumers can tell "request done" from "server gone."
- **Pre-existing JSONL writer behavior.** Today's unbuffered fix is
  correct under any shutdown mode but cheap-but-not-free at high
  throughput. Restore BufWriter once graceful shutdown lands.

## Why not now

Slice-2A scope was the streaming endpoint + per-request UUID
correlation. Graceful shutdown is its own project — affects
`/v1/messages` lifecycle, not just probes. Better to land it as a
focused slice with the lifecycle considerations explicit.

Estimated size: ~30 LOC in `main` + signal-handler imports + a
small refactor of `run()` to thread the shutdown future. One
focused session.
