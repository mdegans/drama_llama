# Future work — prefill progress callback

## Idea (Mike, mid-session-4)

Add a per-token or per-chunk prefill-progress callback to
`Engine::predict_*_resuming` / `Engine::predict_*` so consumers
(drama_llama UI, Agora reactor) can show "X of N tokens prefilled"
during long prompts. Currently `Predictor` only starts yielding
*after* prefill returns, so the UI has no signal during a 40-60k
prompt.

## Why useful

- **Long Agora prompts** (40-60k tokens) sit silent for minutes
  during cold prefill — bad UX, no recourse to cancel mid-flight if
  the user changes their mind.
- **Cancellation hook**: a callback that returns `Result<(), _>`
  would let consumers signal "stop" at chunk boundaries, which is a
  much cleaner cancellation point than today's "either run to
  completion or kill the worker thread."
- **Telemetry**: streaming prefill rate (tok/s instantaneous) is
  easier to plot when reported per chunk than reconstructed from
  wall-clock at completion.

## Shape sketch

```rust
// src/engine.rs — extend Engine API:
pub trait PrefillProgress {
    /// Called after each chunk completes. `processed` = total
    /// tokens prefilled so far in this call; `total` = full prompt
    /// length. Return Err to request cancellation at the next
    /// chunk boundary.
    fn on_chunk(&mut self, processed: usize, total: usize)
        -> Result<(), PrefillCancel>;
}

pub fn predict_pieces_resuming_with_progress<'a, P: PrefillProgress>(
    &'a mut self,
    tokens: Vec<Token>,
    start_pos: usize,
    seq_id: i32,
    options: PredictOptions,
    progress: P,
) -> Result<PiecePredictor<'a, B>, PrefillCancel>;
```

In moeflux's `step_internal` chunked loop:

```rust
while chunk_start < n {
    self.batched_forward(...)?;
    if let Some(cb) = progress.as_mut() {
        cb.on_chunk(chunk_end, n).map_err(...)?;
    }
    chunk_start = chunk_end;
}
```

Granularity is per-chunk (8192 tokens at the current default) —
for a 40k prompt that's 5 callback events, plenty for a progress
bar and not overhead-dominant. Coarser than today's "per-token"
that prefill *could* expose if we exposed the per-token oracle
loop, but per-token granularity for prefill isn't worth the
plumbing cost.

## How to apply

When this lands (probably session 5+, after Phase F measurement
makes the chunk-size knob real), prefer the `with_progress` variant
as the new public API and have the existing methods delegate with a
no-op progress impl. Don't bake cancellation into the Decoder
trait — it stays at the Engine layer where the chunked loop
actually exists.

## Dependencies

- Phase D landed (chunkwise prefill exposes the callback boundary).
- Phase E moves the API question into drama_llama; the moeflux side
  can take a generic `FnMut(usize, usize) -> Result<(), ()>` and
  drama_llama wraps that into a typed trait.

## Why not now

Phase F measurement is the headline; the progress callback is
ergonomics, not throughput. Save the API churn for a session where
this *is* the headline. Mike suggested saving it as a good idea
mid-session-4 — captured here so it doesn't get lost.
