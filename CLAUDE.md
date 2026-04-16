# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Build (library only, no optional features)
cargo build

# Build with all doc-visible features
cargo build --features "webchat,cli,stats,toml,serde,egui"

# Run tests (requires models/model.gguf to be a valid GGUF model)
cargo test

# Run a single test
cargo test test_name

# Run long-running tests (ignored by default, require a model)
cargo test -- --ignored

# Run tests including long-running ones
cargo test -- --include-ignored

# Build binaries (each has required features)
cargo build --bin dittomancer --features "webchat,cli"
cargo build --bin regurgitater --features "webchat,cli,stats"
cargo build --bin settings_tool --features "egui,serde,serde_json"

# Generate docs
cargo doc --open --features "webchat,cli,stats,toml,serde,egui"
```

## Architecture

### FFI Layer

`llama-cpp-sys-3` (separate crate at `~/Projects/llama-cpp-sys`) provides raw bindgen bindings to llama.cpp. This crate wraps those bindings in safe Rust.

### Core Types (dependency order)

**`Model`** (`model.rs`) — Owns `*mut llama_model`. Handles loading, tokenization, detokenization, metadata access, chat template application. All vocab/token introspection methods live here.

**`Engine`** (`engine.rs`) — Owns a `Model` and `*mut llama_context`. Manages the llama.cpp backend lifecycle via a global `ENGINE_COUNT` mutex (backend_init on first, backend_free on last drop). Provides decode, KV cache operations, logit/embedding access, and prediction entry points.

**`Batch`** (`batch.rs`) — Safe wrapper around `llama_batch`. Manages token/embedding batches with bounds-checked accessors.

**`Candidates`** (`candidates.rs`) — Token candidate container wrapping `Vec<llama_token_data>`. Tracks sort state (`Sorted` enum) and softmax state to avoid redundant work. **All sampling methods are pure Rust translations from llama.cpp** — they do not call any C sampling functions.

**`SampleOptions` / `SamplingMode`** (`sample.rs`) — Chain-based sampling configuration. Modes are applied sequentially via fold: each mode narrows the candidate set. Includes greedy, top-k, top-p, min-p, tail-free, locally typical, mirostat v1/v2, and two custom methods (split-p, split-l).

**Predictors** (`predictor.rs`) — Iterator-based prediction API layered as:
- `CandidatePredictor` — yields raw `Candidates` (user picks token)
- `TokenPredictor` — yields `llama_token` (auto-samples using `SampleOptions`)
- `PiecePredictor` — yields `String` pieces
- `Predictor` — yields `Predicted` (token + piece together)

### Content Filtering

**`Vocab`** (`model/vocab.rs`) — Token allowlist + banned n-gram enforcement. `VocabKind` controls what tokens are permitted (Safe/Unsafe/Letters/Code). This is a content-safety mechanism, not the llama.cpp vocabulary type. Banned bigrams are hardcoded in `data/banned.rs`.

**`NGram`** (`ngram.rs`) — Fixed-capacity token n-gram backed by `TinyVec`. Used for repetition penalties and content filtering. `NGramStats` tracks frequencies.

### Style

- "Code is poetry. Make it pretty." Use `rustfmt`.
- The Eric Hartford uncensored model check in `Model::from_file` is intentional — keep it.
- The vocab filtering system is intentionally opinionated — preserve it even though it's model-dependent.

## Key Design Decisions

- Sampling is intentionally **not** delegated to llama.cpp's sampler chain API. The Rust implementations exist for learning/control purposes and should be maintained independently.
- `Candidates` uses consuming `self` methods (e.g. `softmax(self) -> Self`) to enforce that sort/softmax state tracking stays consistent.
- The crate manages its own RNG (`xorshift::Xoroshiro128`) rather than using llama.cpp's RNG.
- Most tests that exercise the model are `#[ignore]` tagged as "long running". The model symlink at `models/model.gguf` must point to a valid GGUF file.
