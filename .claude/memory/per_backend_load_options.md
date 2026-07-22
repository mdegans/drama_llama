# Per-backend load options, `FromPath`, and who owns logging

Landed 2026-07-22 on `v0.8.0`. Read this before adding a constructor to
`Session`/`Engine`, before proposing a shared cross-backend config
struct, and before making anything in this crate decide where logs go.

## The bug that justified the refactor

`bin/blallama` was serving **every llama.cpp model at `n_ctx = 512`**.

It is the only generic-over-`B` consumer, so it could only call
`FromPath::from_path(path)` — and `FromPath` carried nothing but a path.
That fell through `LlamaCppEngine::from_path` → `new(path, None, None,
None)` → `llama_context_default_params()` → 512. blallama could not grow
a `--n-ctx` flag without naming a backend, which is the thing being
generic was for. Its own `session_ready` log line had been printing
`n_ctx=512` the entire time and nobody read it.

**The generalisable bit:** an abstraction too narrow to express a
necessary parameter does not produce a compile error. It produces a
silently wrong default, in the one consumer that cannot work around it.
When a trait has exactly one implementor-facing argument, ask what the
implementors actually need to be told.

## Shape

```rust
pub trait FromPath: Sized + Send + 'static {
    type Options: Default + Clone + Send + Sync + 'static;
    fn from_path_with(path: PathBuf, options: Self::Options) -> Result<Self, SessionError>;
    fn from_path(path: PathBuf) -> Result<Self, SessionError> { … }          // provided
    #[cfg(feature = "tokio")]
    async fn from_path_async(path: PathBuf, options: Self::Options) -> … { … } // provided
}
```

One trait, one required method. `LlamaCppOptions`
(`src/llama_cpp/options.rs`), `MoefluxOptions` (`src/moeflux/options.rs`).

**Two traits with the same method name does not work** — and this was
the original proposal. `FromPath::from_path` (sync) plus
`FromPathAsync::from_path` (async), both implemented for `Session<B>`,
is `E0034: multiple applicable items in scope` at every unqualified call
site where both are imported. Rust resolves methods by name; it does not
disambiguate on await-context. Verified with `rustc` before designing
around it. A sync required method plus a provided async one is strictly
simpler anyway: no blanket impl, and a blanket impl could not have been
overridden per-backend.

## Why `Options` is per-backend, and what would change that

**The intersection of the backends' load knobs is empty.** llama.cpp
has `n_ctx`, `cache_slots`, `flash_attn`, `no_gpu`, `numa`; moeflux has
`use_2bit`. Nothing in common but the path. moeflux's context length is
`variants::MAX_SEQ_LEN`, a compile-time constant
(`~/Projects/moeflux/crates/moeflux/src/riir/mod.rs`).

So **do not add a shared cross-backend config struct yet.** The gate:
land runtime KV sizing in moeflux first, so `n_ctx` becomes a field two
backends both honour. Until then a shared struct has one real member and
lies about the rest. `MoefluxOptions` being nearly empty is the honest
signal, not a gap to fill.

The half-built version already existed and nobody remembered: `cli::Args`
was `{model, context, no_gpu}` with `model_params()`/`context_params()`
plus `LlamaCppEngine::from_cli`. It was `LlamaCppOptions` in the wrong
module, missing two knobs, with no serde. **Grep before designing a new
type in this crate** — it went untouched for two years and the shape was
already there.

## CLI union vs. library options — they cannot be one type

`#[command(flatten)]` resolves at compile time; `--backend` is chosen at
run time. A multi-backend binary therefore *cannot* flatten `B::Options`
— there is no `B` yet when clap builds the command. Hence
`cli::BackendArgs` (the union) narrowed by `TryFrom` to a concrete
`B::Options`. A single-backend binary skips the union and flattens the
concrete type directly (`regurgitater` does).

`BackendArgs` deliberately has **no model path**: blallama's is a
*directory* of models, an example's is one file. Each flattens the union
beside its own path argument.

**Narrowing fails fast.** `--backend moeflux --n-ctx 4096` is
`UnsupportedOptions`, not a warning — this replaced
`warn_llama_cpp_only_knobs`. A dropped `--n-ctx` leaves a run that looks
configured and is not, and it surfaces three benchmarks later if at all.

**One deliberate imprecision:** `BackendArgs::n_ctx` is `u32` with
`default_value_t = DEFAULT_N_CTX` so `--help` shows the number, which
costs the ability to distinguish "unset" from "set to exactly the
default". `--backend moeflux --n-ctx 32768` is therefore accepted where
any other value is refused. Judged cheaper than hiding the default.
`DEFAULT_N_CTX` is 32768, sized to the Agora seed-agent workload
(40–60k typical, ~7k prefix), not to the model — recent local models
reach 1M.

## Logging is the application's job

`Backend::set_log_callback` / `clear_log_callback`, returning
`Result<(), NotImplemented>` with a default body that errors.
`LlamaCppBackend` forwards to `crate::log`; `MoefluxBackend` keeps the
default, correctly — moeflux logs through `tracing`, so there is no
C-side slot to hijack and the subscriber already governs it.

Three things worth not re-deriving:

- **Nothing in this crate installs a sink, and no constructor calls
  one.** An earlier draft had `Backend::init_logging()` invoked from the
  generic constructor. It is wrong twice over: a caller whose model will
  not load wants exactly the load logs it would suppress, and a
  constructor that has already run cannot be un-decided. `Session::quiet`
  has this defect structurally — it can only run *after* load, which is
  when llama.cpp is loudest. It is kept (≈50 uses, mostly tests) but it
  is not the answer.
- **`Err(NotImplemented)` beats a default no-op.** A no-op is
  indistinguishable from a sink that was installed and never fired. The
  error names the backend (from `Backend::NAME`, free in the default
  body) and tells the caller where to look instead.
- **`LogLevel` moved to `src/backend.rs`** and its `Other` widened from
  `ggml_log_level` to `u32`, because `Backend` compiles under
  `--no-default-features` and `src/log.rs` is `cfg(feature =
  "llama-cpp")`. The raw mapping stays gated as
  `log::log_level_from_raw`.
- **`LogLevel::Cont` is a flood.** llama.cpp draws load progress as a run
  of dots, each its own `Cont` call. Filtering is the sink's job — we
  cannot know whether the caller wants them buffered or dropped. The
  examples' bridge folds `Cont` into `trace`.

`tracing` is now a **non-optional** dependency with its `log` feature on.
The crate's own diagnostics (sidecar read/write failures in
`src/session/mod.rs`) were bare `eprintln!` and fired on moeflux runs
too, so "moeflux is quiet, `RUST_LOG` governs" was never true. Gating
those call sites on a feature is more moving parts than the dependency
is worth. The one surviving `eprintln!` is the prefix-cache tripwire,
which panics immediately after and must print with no subscriber
installed.

## Incidental finds

- `CommonArgs::seed` had **no `#[arg]` attribute**, so clap made it a
  *positional* — `--seed` never existed on any example. Now `#[arg(long)]`.
  Worth a squint at any bare field in a clap derive struct.
- `LlamaCppEngine::from_path_cpu_only` and `from_path_with_flash_attention`
  had zero callers outside their own definitions. Deleted.

## Verified (2026-07-22)

`just check` clean; `just test` 468 passed / 119 skipped (was 462 — the
new options/conversion tests are pure logic, no model needed).

## Feature permutations (measured 2026-07-22)

| permutation | status |
|---|---|
| `--no-default-features` | builds — this is why `LogLevel` moved to `src/backend.rs` |
| default (llama-cpp) | green |
| llama-cpp + moeflux | 469/469 serialized |
| moeflux only | **broken, pre-existing** |

moeflux-only fails on an `impl Session<LlamaCppBackend>` block in
`src/session/mod.rs` that never carried a `#[cfg(feature = "llama-cpp")]`.
Pre-dates this arc: 25 errors before the constructor collapse, 3 after —
verified by checking out `49c18d0` and building, not inferred. Tracked in
[#68](https://github.com/mdegans/drama_llama/issues/68) along with the rest
of the test-topology holes (`just test full` is only the ignored tests; no
`cfg(moeflux)` unit test in `src/` is reachable from any recipe; the
both-backends build is compiled but barely run). #68 pairs with #51 (CI) —
one script, called by both the justfile and CI.
