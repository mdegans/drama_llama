# Plan of record: fallible prediction iterators (issue #92)

> **Designed 2026-07-28 (Opus 5), agreed with Mike in-session.
> NOT IMPLEMENTED — deliberately deferred to its own session**, because
> the predictor is the core of Session, the examples, both bins and
> every model test, and the work could balloon.
>
> GitHub twin: [issue #92](https://github.com/mdegans/drama_llama/issues/92).
> Motivating incident: [[mistral4_support_and_metal_nan]].

## The problem

`Iterator::next() -> Option<Item>` has nowhere to put an error, so all
three decode sites panic:

- `predictor.rs:350` — `.expect("prefill failed in CandidatePredictor::new")`
- `predictor.rs:392` — `.expect(... new_resuming)`
- `predictor.rs:443` — `.expect("decoder.step failed")`

Plus a quieter one that is arguably worse — `predictor.rs:982` swallows
a `GrammarError` and returns `None`, i.e. **indistinguishable from
EOS**. No panic, no log, just a short completion.

**This fires in practice.** It fired all session on Mistral Small 4 on
Metal. Mike's framing of why it matters: a panic kills the tokio task,
the next request builds a new predictor and panics again — a **panic
loop** — and it is not clear llama.cpp cleans up when a panic unwinds
through a frame that called into it.

## The decision

**Now (non-breaking):** `take_error()` + `NanPolicy` + fix the 4
`Session` sites.
**At 1.0 (breaking):** the full `Item = Result<T, PredictError<E>>`,
batched into [[one_dot_oh_wishlist]]'s "breaking-later" section.

Mike's preference is *yielding `Result`* — hence the 1.0 half. What
makes it 1.0 work rather than now: it is semver-blocking, and handling
the error **inside** the predictor is far less surgery.

### 1. `take_error()`, following the existing idiom

Not a new pattern. The crate already delegates terminal-state
accessors up the chain — `grammar_complete()` (`:742`),
`constraint_incomplete_at_end()` (`:757`), `sampler_state()` (`:733`),
re-exported at `:1148-1160`. `take_error()` is the same shape.

Constructor errors are **stored** so the first `next()` returns `None`
immediately; that is what keeps it non-breaking end to end. Fold
`:982`'s `GrammarError` into the same slot.

### 2. `Engine::nan_policy() -> NanPolicy`

`CandidatePredictor` has no options of its own (unlike the sampling
predictors) but does hold `&mut Engine` — Mike's observation, and why
`Engine` is the right owner.

```rust
pub enum NanPolicy {
    Stop,                                  // end stream, record error
    RetryChunked { chunk: NonZeroUsize },  // wipe, re-prefill smaller
}
```

Only two variants, both doing real work. `Abort`/panic was considered
and rejected: panicking across FFI in a tokio task is the failure mode
we are trying to remove.

### 3. Fix the 4 `Session` sites

Without this, `take_error()` only trades a loud panic for **silent
truncation**. `Session::complete` already returns
`Result<_, SessionError>`, so a decode variant slots in. The open
decision at each site is cache-commit ordering: abort and discard
`generated_tokens`, or commit the prefix and surface the error with
partial text.

## Why `RetryChunked` is buildable (the question that decided it)

**No trait changes needed.** `Decoder::prefill(tokens, start_pos,
seq_id)` already takes arbitrary slices and start positions —
`new_resuming` does exactly that — and `memory_clear` / `forget_pos` /
`memory_seq_rm` are already on the trait:

```rust
decoder.forget_pos(seq_id, start_pos);   // the error is kv_dirty
let mut pos = start_pos;
for c in tokens.chunks(chunk) { logits = decoder.prefill(c, pos, seq_id)?; pos += c.len(); }
```

Two facts make it cheap:

- **It never needs to know *why* it failed.** Retrying chunked on *any*
  prefill error dodges the opaque associated type (`Decoder::Error` is
  `DecodeError` for llama.cpp, `MoefluxError` for moeflux — you cannot
  match `NonFinite` generically), and is also the right response to
  `BatchTooLarge`.
- **Only prefill can hit this class of bug.** `Decoder::step()` decodes
  one token, so `ne21 = 1` — below any batch-size threshold,
  structurally immune. The retry lives entirely in the constructors.

**KV is already poisoned when this fires** (`NonFinite` is `kv_dirty`),
so "continue from here" is never an option — wipe first, always.

### Chunk size

Measured Metal threshold is **exactly 32** (31 tokens clean, 32 all
NaN) — it is `ne21_mm_id_min`, not a tuned number, and it is why
`n_ubatch = 31` is the workaround. So chunks must be ≤31.

Encoding 31 would bake one backend's kernel constant into
backend-agnostic code. Prefer **halving-from-full** (converges to a
working size on any backend, naming none), or 16 as a fixed default.

### Cost, and the counter-intuitive bit

With a warm prefix cache, `new_resuming` prefills only the uncached
delta, so retry is cheap. On a **cold** call it re-prefills everything
— a 40–60k Agora prompt at chunk=16 is ~2,500–3,750 sequential
decodes. **The fallback is cheapest exactly when it is least needed.**
Still far better than a panic loop. Log at `warn!` with the error and
chunk size on every retry, so an upstream regression shows up
immediately instead of being inferred from a slow prefill.

## Blast radius (measured, not estimated)

**32 logical call sites**, but only **4 are production library code**,
all in `src/session/mod.rs`. The rest: 9 in-`src` `#[cfg(test)]`, 10
integration tests, 4 in `examples/whoami.rs`, 2 bins.
**`blallama` and `settings_tool` have zero** — pure `Session`
consumers, they would not notice.

Why `Item = Result` is 1.0 work, not now:

1. **Constructors prefill**, so `Item` cannot carry their error;
   fixing them cascades to 7 constructors and all 7
   `Engine::predict_*` entry points, which return the predictor **by
   value, not `Result`**. That is the real breaking surface.
2. `Decoder::Error` is an **associated type** — `Item = Result<T,
   DecodeError>` is not spellable without a `PredictError<E>` wrapper.
3. All four predictor types are **publicly re-exported and unfeatured**
   (`lib.rs:147-150`), so every downstream `.collect::<String>()` and
   `for x in engine.predict_*(..)` breaks.

**`BlockStream` (`session/mod.rs:6278`) is the genuinely hard site**: a
public streaming type with `Item = Block`. If the inner
`PiecePredictor` yields `Result` it needs its own decision — propagate
(breaking again), swallow-and-stash, or synthesize an error block. No
mechanical answer, and it is what Agora's reactor streams through.

Also needing thought rather than a mechanical fix:
`session/mod.rs:4816/4824`, `:5215/5223` (both drive `last_token()`,
`sampler_state()`, grammar-completion and `uncommitted_bytes` per
iteration), `:5618` (reaches into `predictor.engine.model` mid-loop),
`bin/regurgitater:383` (destructures `Predicted` in pattern position),
`examples/whoami.rs:1067`, `tests/tip_invariant.rs:48`.

## Rejected

- **Parallel `try_predict_*` family** — 7 more entry points plus
  `Result`-yielding twins is the method proliferation Mike has called a
  smell, and it leaves the wrong default in place indefinitely.
- **`Abort`/panic policy** — see above.
- **Sanitize (NaN → -inf)** — with an all-NaN vocab this samples a
  random token from a corrupt cache. Silently wrong, the worst outcome.
- **Plain retry** — the failure is deterministic (measured
  bit-identical across processes); a re-run re-fails.

## Related

- [[mistral4_support_and_metal_nan]] — the incident, and
  `DecodeError::NonFinite` which this makes reachable.
- [[one_dot_oh_wishlist]] — where the breaking half belongs.
- [[llama_cpp_ffi_audit]] — same class: an upstream failure surfacing
  as something that looks like our bug.
