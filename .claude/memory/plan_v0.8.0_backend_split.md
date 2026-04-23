# Meta-plan: drama_llama v0.8.0 backend split

Execution plan for extracting llama.cpp behind a trait so we can plug in
flash-moe (Metal/C, streaming-experts MoE) as an alternative decode +
model backend. Target: run Cogito 600B MoE locally on Apple Silicon as
an Anthropic-API-independence path for Agora's Council.

Builds on the scout findings in `future_work_flash_moe_backend.md`.
That document captures flash-moe's internal shape + upstream work
items; this one captures **our** drama_llama-side execution across
multiple sessions.

## Goals

1. Full backend split: `llama-cpp` and `flash-moe` are independent
   Cargo features. Either can be disabled entirely. No runtime mixing.
2. Support any MoE model (near-term target: Cogito 600B; smoke-test
   target: something ≤40GB).
3. Long-term: tear out C. Flash-moe's hand-rolled tokenizer is
   negotiable; everything above the decode kernel should eventually
   be Rust.
4. Don't regress Weave (it lives on 0.7.0 if we break it, but the
   surface Weave uses is small and we can preserve it).

## Locked-in design decisions

- **Both `Decoder` and `Model` are traits.** Not just decode — model
  metadata, tokenization, vocab introspection, chat-template
  application all go behind a trait. No mixing llama.cpp-for-
  tokenization with flash-moe-for-decode.
- **`Engine` is generic: `Engine<D: Decoder, M: Model>`** (or possibly
  `Engine<B: Backend>` where `Backend` bundles both). Compile-time
  monomorphization. No `Box<dyn>` in hot loops.
- **Feature-gated default type alias** so consumers (Weave, binaries)
  still say `Engine` without parameters when exactly one backend
  feature is enabled.
- **`step(&mut self, token: Token) -> Result<&[f32]>`** for the hot
  path. Candidates construction stays in Predictor.
- **Fork location**: `~/Projects/flash-moe` (sibling crate, path-dep
  in dev, git-dep for releases). Matches the `~/Projects/llama-cpp-sys`
  pattern. No submodules.
- **Introduce `drama_llama::TokenData`** with ABI-compatible layout to
  `llama_token_data`. Candidates stores `Vec<TokenData>`. Interop with
  llama.cpp types lives behind `cfg(feature = "llama-cpp")`.
- **Introduce `drama_llama::Token`** as a type alias (currently
  `i32`) to replace bare `llama_token` in public API.

## Phase breakdown

Each phase is landable independently. drama_llama never leaves a
broken state on `main`.

### Phase 1 — Trait design + llama.cpp extraction (Plan-mode session)

Scope:
- Define `Decoder` trait (prefill, step, memory_clear, memory_seq_rm,
  n_vocab, n_ctx, eos). Generic error type.
- Define `Model` trait (tokenize, detokenize, token_to_piece,
  special_tokens, max_token_len, eos, bos, eot, n_vocab,
  context_size, embedding_size, get_meta, chat_template).
- Introduce `Token` type alias + `TokenData` struct.
- Move current Engine behavior to `LlamaCppDecoder` (concrete impl).
- Move current Model behavior to `LlamaCppModel` (concrete impl).
- Restructure `Engine` to be generic over `D: Decoder, M: Model`.
- Feature-gate llama.cpp: `default = ["llama-cpp"]`, crate compiles
  (even if unusable) with `default-features = false`.
- Update call sites in `session/`, `predictor.rs`, binaries, tests.
- Preserve Weave's surface (`engine.model`, `PredictOptions.n`,
  `PredictOptions.stop_strings`, `Engine::from_cli`, `quiet`,
  `set_n_threads`, `predict_pieces`, `n_ctx`, `Model::tokenize`,
  `Model::add_model_stops`, `Model::context_size`, `Model::meta`).

Exit criteria:
- `cargo test --features "webchat,cli,stats,toml,serde,egui"` passes.
- `cargo build --no-default-features` compiles (trait layer only; no
  backend available at runtime is fine for this phase).
- Weave (path-dep'd) builds and runs unchanged.

Est: 1 focused day. Mostly mechanical after trait shapes are nailed.

### Phase 2 — Regression harness

Scope:
- Golden-output capture on current `LlamaCppDecoder` for a fixed set
  of deterministic probes (greedy sampling, fixed seed, small model).
- Capture: token stream, logits at step 0, logits at step N.
- Harness runs against the llama.cpp backend as a baseline and later
  against flash-moe.
- Wire into existing `#[ignore]` long-running tests.
- Fix any `todo!` tests surfaced during trait extraction.

Exit criteria:
- Harness produces reproducible output across `cargo test --
  --include-ignored` on the same machine.
- Baseline golden files committed for a small model.

Est: half a day. Informal markdown-driven; probably no Plan-mode
needed unless scope grows.

### Phase 3 — flash-moe fork + C API

Scope:
- Fork `github.com/danveloper/flash-moe` to an account we control
  (mdegans or a project account).
- Add `memory_seq_rm(start, end)` — reset position counter + zero
  buffer range. ~50 LOC per the scout doc.
- Add extern "C" API: `dl_init_model`, `dl_eval_prompt`,
  `dl_eval_token`, `dl_memory_seq_rm`, `dl_memory_clear`,
  `dl_n_vocab`, `dl_eos`, `dl_model_name`, `dl_free_model`.
- Header file suitable for bindgen.
- Tokenizer export path: decide whether to keep flash-moe's exported
  tokenizer or re-tokenize in Rust (e.g. `tokenizers` crate). Probably
  Rust-side tokenizer is cleaner for the "tear out C" goal.
- PR upstream to danveloper (may or may not merge; friendly).

Exit criteria:
- Fork builds on M-series Mac.
- C API works end-to-end on Qwen3.5-A17B or smaller (smoke test via a
  tiny C program before drama_llama integration).

Est: 2-4 hours if KV-seq_rm really is ~50 LOC. Plan-mode recommended
for the tokenizer-routing decision.

### Phase 4 — `FlashMoeDecoder` + `FlashMoeModel` via bindgen

Scope:
- New crate `flash-moe-sys` (bindgen) sibling to `llama-cpp-sys-3`.
- `FlashMoeDecoder` + `FlashMoeModel` impls in drama_llama under
  `cfg(feature = "flash-moe")` + `cfg(target_os = "macos")`.
- build.rs for Metal + Foundation framework linking.
- Handle flash-moe's model shape constants (fork-and-rebuild
  initially; parameterize later).
- Smoke test: load a small MoE, generate a few tokens, compare shape
  of output against llama.cpp backend on the same model.

Target smoke-test models (small first, work up):
- **OLMoE-1B-7B** (~4-5GB Q4, 1B active). "Does the pipe work at all."
- **Qwen3-30B-A3B** (~18GB Q4, 3B active). "Does real MoE work."
- **Qwen3.5-A17B** (original flash-moe target, 200GB+). Skip unless
  everything else passes — llama.cpp comparison baseline would be
  painful (swapping like crazy).

Exit criteria:
- Drama_llama with `flash-moe` feature runs OLMoE end-to-end.
- Regression harness output on llama.cpp matches flash-moe within
  tolerance (not byte-for-byte, but token distributions should align).

Est: 1-2 days. Plan-mode session warranted.

### Phase 5 — Cogito 600B adaptation + Agora probe

Scope:
- Parameterize model shape in flash-moe fork if Cogito 600B differs
  structurally from Qwen3.5-A17B.
- Re-run `repack_experts.py` / `extract_weights.py` / 4-bit quantize
  pipeline for Cogito 600B (hours of CPU time, one-shot).
- Wire drama_llama's axum server (Agora's reactor target) to
  flash-moe backend.
- Run Agora's alignment-drift canary
  (`agora-agents/crates/agora-agent-lib/examples/probe_canary`)
  against Cogito 600B via drama_llama. Compare to cogito-32b baseline.
- If drift exceeds tolerance: governance-relevant data point;
  discuss on Agora before deciding next step (re-probe, weight edit,
  different distillation target).

Exit criteria:
- Cogito 600B runs at ≥4 tok/s decode on 96GB Macbook.
- Probe produces a committed baseline entry.
- Agora Council can run on local model if/when Anthropic API is
  pulled.

Est: 1-2 days excluding repack pipeline. Probably informal; Plan-mode
only if architecture mismatch turns out to be large.

## Open questions (revisit before starting each phase)

- **Phase 1**: Generic `Engine<D, M>` vs `Engine<B: Backend>` where
  Backend bundles both. Bundled is cleaner API but less flexible.
  Lean bundled unless a reason to split emerges.
- **Phase 3**: Tokenizer in Rust (via `tokenizers` crate, shared
  across backends) vs flash-moe's hand-rolled C tokenizer. Rust-side
  is cleaner for the long-term "tear out C" goal.
- **Phase 4**: Feature-flag layout. `default = ["llama-cpp"]`? Or
  platform-aware (`llama-cpp` default on Linux, `flash-moe` default on
  Mac)? Probably explicit in Cargo.toml, no platform auto-magic.
- **Phase 5**: Does Cogito 600B share Qwen3.5-A17B architecture
  exactly? Answered by reading Cogito's HF config vs Qwen3's. Needs
  to happen before Phase 5 starts.

## Deferred work (Phase 1 landed pragmatically)

- **`sample_token` still hardcodes `&LlamaCppModel`.** Routing
  `Candidates::sample_token` / `apply_sample_repetition_ngram` /
  `token_to_piece_ref` through `&impl Model` / `&dyn Model` was out
  of scope for Commit 4b; during Phase 1 the Predictor structs
  (`CandidatePredictor<D, M>`, `TokenPredictor<D, M>`,
  `PiecePredictor<D, M>`, `Predictor<D, M>`) carry a generic `M:
  Model` type parameter, but the **Iterator impls** on
  `TokenPredictor` / `PiecePredictor` / `Predictor` are restricted
  to `M = LlamaCppModel`. Session similarly is pinned to
  `LlamaCppModel` via `LlamaCppEngine`. This is fine now because
  we have only one `Model` impl; it **must** be addressed when
  `FlashMoeModel` (Phase 4) is introduced. Touching this will
  require changes in `src/sample.rs`, `src/sample/repetition.rs`,
  `src/sample/grammar.rs`, `src/sample/json.rs`, and
  `src/candidates.rs::sample_token`.

- **Backend lifecycle via OnceLock + leak.** With `Engine<D, M>`
  generic, `Engine` can no longer have a Drop impl (Rust allows
  Drop only on generic struct definitions themselves, not on
  specific instantiations). Backend init/teardown moved to
  `LlamaCppDecoder::new` / `LlamaCppDecoder::Drop`; same net
  behavior, including `llama_backend_free` on last-decoder-drop.
  If a future backend (flash-moe) brings its own process-global
  init/teardown, we may need to revisit this pattern.

## What's deliberately NOT in scope

- Non-Metal platforms (CUDA, Vulkan, ROCm) for flash-moe. Those are
  fresh implementations of streaming-MoE on different hardware.
- Auto-repack pipeline. Humans repack once per target model;
  drama_llama consumes repacked dirs only.
- Running the Agora probe itself as part of any drama_llama PR. Probe
  lives in agora-agents.
- Optimizing Candidates allocation per-step. Current path allocates
  ~1.8MB Vec per token at n_vocab=151k; worth a perf item later, not
  coupled to backend split.
- Byte-for-byte regression between backends. Same shape is the bar;
  identical output is unrealistic given numerical differences.

## Dependencies between phases

```
Phase 1 ──┬──> Phase 2 ──┐
          │              │
          └──> Phase 3 ──┴──> Phase 4 ──> Phase 5
```

Phase 2 and Phase 3 can run in parallel after Phase 1 lands. Phase 4
consumes both. Phase 5 consumes Phase 4.

## Why the ordering

Phase 1 is load-bearing. If the trait boundary is wrong, everything
downstream is rework. Golden-output harness (Phase 2) protects
against trait-extraction regressions before we add a second backend.
flash-moe fork (Phase 3) is independent of drama_llama-side work and
can proceed as soon as Phase 1's C API surface is agreed. Integration
(Phase 4) is where the two halves meet; smoke tests with small MoEs
before committing to the 600B repack. Phase 5 is the payoff.

## Session continuity

This plan spans multiple Claude Code sessions. When picking up:
1. Read this file and `future_work_flash_moe_backend.md`.
2. Check `git log --oneline v0.7.0..HEAD` for what's already landed.
3. Check `.claude/memory/MEMORY.md` for the current project status
   entry.
4. Find the phase in-progress (should be an open PR or a
   `v0.8.0-phase-N` branch).

If confused about current state, ask Mike before proceeding.
