# Validation runbook — paths, env vars, invocations

First assembled 2026-06-12 for the v0.8.0 pre-publish session.
Everything here was verified on disk that day. This is the
"how do I actually run all the model-touching stuff" reference;
the per-release checklist lives in the publish-session plan memos.

## Never run model-backed tests with `cargo test`

**Use `just test` (unignored) and `just test ignored` (the `#[ignore]`d
model/GPU tests); `just test all` is both.** All go through `cargo-nextest`,
which runs each test in its own process; the `full` *nextest profile*
serializes them.

(The mode used to be called `just test full`, which did NOT mean everything
— only the ignored set. Renamed in #68; `full` is now a hard error that
tells you to pick `ignored` or `all`. The nextest profile keeps the name.)

`cargo test` overlaps test *binaries* — and `--test-threads=1` does not
fix it, since that only serializes *within* one binary. With a ~19 GB
model per test you get memory exhaustion surfacing as a llama.cpp decode
failure:

```
thread '<varies>' panicked at src/predictor.rs:443:
decoder.step failed: Fatal { code: -3 }
```

Diagnostic tell: **the failing test changes between runs** while the
pass/fail counts stay identical (48 passed / 1 failed, different test each
time). That is the harness, not the code — re-run the single test alone
to confirm before suspecting a regression. Burned ~10 minutes of model
runs on this 2026-07-21; Mike called it out.

Single-binary invocations (`cargo test --test session -- --ignored`) do
happen to work, because that is one binary and those tests are `serial`.
Not worth relying on — just use `just`.

## Model inventory (this machine)

| Path | What |
|---|---|
| `models/model.gguf` | **Hard link** (not symlink) → Qwen3.6-35B-A3B-UD-Q4_K_S, ~19.8 GB. Swapping models means re-linking this. |
| `~/models/gguf/` | Qwen3.6-35B-A3B-UD-Q4_K_S, Qwen3.5-35B-A3B-Q8_0, cogito-32b, gemma-4-31B-it-UD-Q4_K_XL (+ `.sampling.toml` sidecars except gemma) |
| `/Volumes/Temp Backup/models/gguf/` | qwen3-6-35b-a3b `q4_k_m` and `f16` — these are what `cross_backend.rs` defaults expect |
| `/Volumes/Temp Backup/models/moeflux/` | qwen3-6-35b-a3b `{mlx-4bit, artifacts, root}` + a17b set + experimental artifact variants |
| `/Volumes/Temp Backup/models/blallama/` | per-model dirs (qwen3-6-a3b, qwen3-5-a17b, cogito-v2-671b) — pass this dir to `blallama` as its positional model-dir arg |

`Temp Backup` must be mounted for the moeflux / cross_backend / blallama
legs. If it is, **all the env-var defaults below just work** — no
overrides needed.

## Env vars (the complete set)

| Var | Read by | Default |
|---|---|---|
| `DRAMA_LLAMA_GGUF_PATH` | `tests/cross_backend.rs` | `/Volumes/Temp Backup/models/gguf/qwen3-6-35b-a3b-q4_k_m.gguf` |
| `DRAMA_LLAMA_MOEFLUX_MLX_DIR` | cross_backend + all moeflux tests | `…/moeflux/qwen3-6-35b-a3b-mlx-4bit` |
| `DRAMA_LLAMA_MOEFLUX_ARTIFACTS_DIR` | same | `…/moeflux/qwen3-6-35b-a3b-artifacts` |
| `DRAMA_LLAMA_MOEFLUX_EXPERTS_DIR` | same | `…/moeflux/qwen3-6-35b-a3b-root` |
| `DRAMA_LLAMA_COGITO_MODEL` | `tests/hash_cache_smoke.rs` | falls back to `models/model.gguf` |
| `DRAMA_LLAMA_MODEL` | `examples/inspect_prompt.rs` only | `models/model.gguf` |
| `DRAMA_LLAMA_UPDATE_GOLDEN=1` | `tests/regression.rs` | (off) overwrites golden files |
| `MOEFLUX_MOE_GATHER_ID`, `MOEFLUX_SDPA_VB` | set *by* `moeflux_coherence.rs` on its decode subprocess | — |

**There is no global env override for `models/model.gguf`** — every
llama-cpp ignored test hardcodes it via `CARGO_MANIFEST_DIR`. Symlink
(or hard-link) swap is the only way to point the test suite at a
different model. Examples are friendlier: `strawberry --model`,
`whodunit <path>`, `chat --model <path>`, `dump_template <path>`,
`inspect_prompt` via `DRAMA_LLAMA_MODEL`.

## Invocations

```bash
# Fast tests, default features (llama-cpp)
cargo test

# Full ignored sweep against models/model.gguf.
# --test-threads=1 is REQUIRED: parallel model tests construct
# multiple ~20 GB engines concurrently and fail with
# `decoder.step failed: Fatal { code: -3 }` (first seen
# 2026-06-12, when it read `ErrorCode { code: -3 }`; the FFI audit
# split that variant out — -3 is GGML_STATUS_FAILED, a compute-graph
# failure. A 46 s wall time for the sweep is the tell that it ran
# parallel). --no-fail-fast so a lib-target failure doesn't
# skip the integration suites.
cargo test --no-fail-fast -- --include-ignored --test-threads=1

# moeflux integration tests (macOS; Temp Backup mounted)
cargo test --features "moeflux,moeflux-model-qwen3-6-35b-a3b" \
    --test moeflux_smoke --test moeflux_session_pollution \
    --test moeflux_coherence -- --ignored --nocapture

# Cross-backend agreement (both backends, 32 greedy steps,
# ≥95% argmax agreement, ≥80% top-20 Jaccard)
cargo test --features "llama-cpp,moeflux,moeflux-model-qwen3-6-35b-a3b" \
    --test cross_backend -- --ignored --nocapture

# Examples (model path optional where shown)
cargo run --release --example strawberry --features "cli,json-schema" -- [--model <gguf>]
cargo run --release --example whodunit --features "json-schema" -- [<gguf>]
cargo run --release --example chat --features "tokio,repl" -- [--model <gguf>]
cargo run --release --example dump_template -- [<gguf>]
cargo run --release --example inspect_prompt --features cli -- <prompt.json>
cargo run --release --example grammar_fuzz --features cli -- pure --duration 5m --threads 6 --corpus ./fuzz-corpus

# blallama (positional model DIR, default port 11435)
cargo run --release --bin blallama --features "axum,cli,toml" -- \
    "/Volumes/Temp Backup/models/blallama"
# add: --backend moeflux (+ moeflux features), --probe-stream,
#      --record-json <path>, --seed <u128>, --no-penalty
# endpoints: POST /v1/messages, GET /api/tags, GET /probe (SSE)
```

## Misanthropic examples as blallama harness — VERIFIED 2026-06-12

Working procedure (all five non-streaming examples ran green against
blallama; strawberry's "Tool was not called" is the unforced-dialect
gap, see `future_work_qwen_xml_tool_call_parse.md`):

```bash
# Server (note --default-model so claude-* ids resolve):
./target/release/blallama ~/models/gguf --port 11435 \
    --default-model Qwen3.6-35B-A3B-UD-Q4_K_S.gguf

# Examples (in ~/Projects/misanthropic):
export MISANTHROPIC_BASE_URL=http://localhost:11435
export ANTHROPIC_API_KEY=$(python3 -c "print('x'*108)")  # dummy
# neologism & strawberry read the key from STDIN, not env — pipe it:
python3 -c "print('x'*108)" | ./target/debug/examples/neologism
```



The examples (in `~/Projects/misanthropic/misanthropic/examples/`)
construct `Client::new(key)` against the production URL — **no
base-url flag or env var exists** as of 1.0.0-alpha.2, even though
`Client::base_url()` itself does. To point them at blallama you
need a local patch (add `.base_url("http://localhost:11435")?` after
`Client::new`, or teach `utils::CommonArgs` a `MISANTHROPIC_BASE_URL`
env). API key can be any dummy string of plausible length.
Non-streaming examples should pass; streaming ones are expected to
fail until drama_llama#26 (stream `misanthropic::stream::Event`)
lands in 0.9 — don't chase those failures.

## Publish-day notes

- `Cargo.toml` `exclude` keeps `.claude/`, `CLAUDE.md`, `tests/data/`
  (the detect-infringement corpus — copyrighted, must never ship),
  and `.DS_Store` out of the crate. Verify with
  `cargo package --list | grep -cE 'tests/data|\.claude'` → 0.
- CHANGELOG: stamp the `Unreleased` date before tagging.
- Doc check: `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
  --features "webchat,cli,stats,toml,serde,egui"`.
- crates.io API needs a `User-Agent` header or it 403s.
