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

**Status (2026-04-24): LANDED.** Commits `a164c30` through `2fde122`
on `v0.8.0`:
- Token + TokenData (`backend.rs`)
- Decoder + Model traits
- LlamaCppDecoder extracted from Engine; LlamaCppModel renamed
- Engine<D, M> generic; threaded through Predictor + Session
- Files moved to `src/llama_cpp/{mod,decoder,engine,model}.rs`
- `llama-cpp-sys-3` optional; `llama-cpp` feature forwards to dep;
  `cli`/`webchat`/`axum` transitively imply it
- `--no-default-features` drops the C dep and compiles the trait
  layer (verified via `cargo tree`)

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

**Status (2026-04-24): LANDED** (commit `07c160f`). `tests/regression.rs`
captures 4-token prompt + 32 greedy steps + top-20 logits at step 0 and
step N on `LlamaCppEngine`. Golden at
`tests/fixtures/regression/llama_cpp_baseline.json` (Qwen2 151665-vocab
131072-ctx model, 4319 bytes). Compare with exact token match + 1e-2
absolute tolerance on logits. Regenerate with
`DRAMA_LLAMA_UPDATE_GOLDEN=1`. Two `todo!()` candidate tests
(`test_apply_entropy`, `test_sample_tail_free`) also filled in.

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

**Status (2026-04-24): LANDED** at `github.com/mdegans/moeflux`
(private). Renamed from `flash-moe` → `moeflux`; Opus 4.6 /
@danveloper / Opus 4.7 / @mdegans credited in CONTRIBUTORS.md.
MIT license with AI-authored public-domain notice. Tokenizer
routing decided: Rust-side via HuggingFace `tokenizers` crate in
Phase 4; tokenizer.h kept in C until then.

Seven commits on the fork:
1. `moeflux: Fork initial — LICENSE, CONTRIBUTORS, README preamble`
2. `NOTES: Orientation from 3b.1 reading of infer.m`
3. `strip: Drop chat.m, linenoise, failed-experiment scripts` (−4068 LOC)
4. `kv: Add state truncation helpers for KV + linear-attn layers`
5. `api: Add mf_* extern "C" surface + libmoeflux.a target` (+460 LOC)
6. `smoke: Add C smoke test + make smoke target`
7. `state: Add mf_state_{size,save,load} for prefix-cache reuse (Option B)`

C API shape (see `~/Projects/moeflux/metal_infer/moeflux.h`):
- `mf_init_model`, `mf_free_model`
- `mf_eval_prompt`, `mf_eval_token` — caller-allocated logit buffers,
  no internal sampling
- `mf_memory_clear`, `mf_memory_seq_rm`, `mf_memory_seq_pos_max`
- `mf_n_vocab`, `mf_n_ctx`, `mf_eos`, `mf_model_name`
- `mf_state_size`, `mf_state_save`, `mf_state_load` (Option B,
  closes the "full re-prefill on truncation" gap for drama_llama's
  prefix cache)

Scout-vs-survey disagreement on chat.m settled: it's an HTTP/SSE
client, not the reference inference loop. The reference per-token
flow is `serve_loop()` at `infer.m:5965`. Our `mf_eval_token`
mirrors its inner loop (embed → 60× fused_layer_forward →
complete_deferred_experts → cpu_rms_norm → lm_head_forward).

Linear-attention state handling: GPU buffers
(`g_metal->buf_delta_state[i]` / `buf_conv_state[i]`) are
authoritative on Metal path. Option B serializes these directly
via `[contents]` unified-memory access.

**Remaining for public release:**
- Flip `mdegans/moeflux` to public (blocked on Mike's courtesy
  note to @danveloper — draft in-progress at session end).
- Publish `moeflux` crate to crates.io (Phase 4 concern — needs
  the Rust wrapper first).

**Actually tested:** build only. No smoke run yet — model download
in progress on Mike's side. Real end-to-end validation is
Phase 4's opening move.

### Phase 4 — `MoefluxDecoder` + `MoefluxModel` via bindgen

(Was `FlashMoeDecoder` in the original plan — renamed to match the
`moeflux` fork identity.)

**Hard constraint — run C smoke before any Rust work.** Smoke
running, not byte-for-byte correctness, is the gate. **Status
(2026-04-24): GATE PASSED** against Qwen3.6-35B-A3B (MLX 4-bit
converted from `Qwen/Qwen3.6-35B-A3B` BF16). All 12 smoke stages
green: init → prefill → 3 decodes → seq_rm → state_size/save/load
round-trip → clear. State round-trip restores pos_max=4; logits
don't match fresh path bit-for-bit (WARN — expected per this
plan's "GPU non-determinism" tolerance).

Two C-side bugs surfaced and were fixed during first run:
  - `mf_model_name` hardcoded to A17B string instead of using
    `MOEFLUX_MODEL_NAME` macro (moeflux commit `449649d`).
  - `smoke.c` had wrong seq_rm/memory_clear ordering before the
    re-prefill, causing state_save to fail -1 on short buffer
    (same commit). These were the only two bugs found in the
    ~1000 LOC of unverified C — much cleaner than expected.

35B-A3B was picked over A17B for this gate because it's the same
`qwen3_5_moe` architecture at a smaller shape (2048 dim, 40
layers, 256 experts), so the C smoke validates both the API
surface *and* the conditional-compilation scaffold in one shot.
A17B is preserved as a compile-time variant but not smoke-tested
in this session.

Scope:
- **Workspace layout inside the existing `moeflux` repo:**
  ```
  moeflux/
  ├── Cargo.toml         ← [workspace] members
  ├── metal_infer/       ← C/Metal sources (unchanged)
  ├── tests/             ← C smoke test (unchanged)
  └── crates/
      ├── moeflux-sys/   ← thin bindgen wrapper, raw FFI
      └── moeflux/       ← Rust-ergonomic wrapper, published name
  ```
  Both crates publish to crates.io. drama_llama depends on
  `moeflux` (not `-sys`). This follows the standard -sys
  convention so downstream consumers who want raw FFI can drop
  down without rewriting our safe layer. Also defensively
  claims both `moeflux` and `moeflux-sys` crate names.
- **moeflux-sys:** `build.rs` compiles the C/Metal via the `cc`
  crate + `-fobjc-arc`, links `-framework Metal -framework
  Foundation -framework Accelerate -lpthread -lcompression`.
  bindgen against `../../metal_infer/moeflux.h`. Gated
  `cfg(target_os = "macos")`. No safe Rust — unsafe extern "C"
  + raw types only.
- **moeflux:** depends on moeflux-sys. Safe wrappers:
  `MoefluxCtx` (owns `mf_ctx*`, `Drop` calls `mf_free_model`),
  Result-based error types, logit slices (pointer → `&[f32]`
  with correct lifetime), safe state-snapshot API.
- **drama_llama side:** `MoefluxDecoder` + `MoefluxModel` impls
  under `cfg(feature = "moeflux")` + `cfg(target_os = "macos")`.
  Tokenization via HuggingFace `tokenizers` crate; moeflux's C
  API receives token IDs, never text.
- **Model shape:** fork-and-rebuild per target model initially.
  Runtime parameterization is Phase 5 (Cogito 600B).

Exit criteria (in order — each gates the next):

1. **[PASSED 2026-04-24]** Smoke produces PASS against the real
   model. Hit on Qwen3.6-35B-A3B (not A17B; see smoke-gate note
   above). State save/load round-trip non-byte-exact WARN is
   within tolerance.
2. **[PASSED 2026-04-24]** `cargo build -p moeflux-sys`
   compiles; bindgen output matches hand-written `mf_*`
   signatures (verified by `tests/link.rs`). A17B + 35B-A3B
   features both build green; zero / both features panic at
   build time with a clear message. Moeflux commit `6e600ef`.
3. **[PASSED 2026-04-24]** `cargo build -p moeflux` compiles;
   `Ctx::Drop` runs cleanly at end of Rust smoke. Full Rust
   port of `tests/smoke.c` PASSes in 3.29s against 35B-A3B
   artifacts (same commit).
4. drama_llama with `moeflux` feature loads a 35B-A3B variant
   and runs the regression harness. **Tolerance: token-level
   argmax agreement with llama.cpp backend ≥95%, top-20 logit
   set overlap ≥80%.** Cross-backend comparison: GGUF Q4_K_M
   of the same BF16 source (converted via llama.cpp's
   `convert_hf_to_gguf.py` + `llama-quantize`) lives at
   `/Volumes/Temp Backup/models/gguf/qwen3-6-35b-a3b-q4_k_m.gguf`.

**35B-A3B artifacts (as of 2026-04-24):**
- MLX 4-bit (group 64, affine): `/Volumes/Temp Backup/models/moeflux/qwen3-6-35b-a3b-mlx-4bit/` (18 GB)
- Packed experts: `/Volumes/Temp Backup/models/moeflux/qwen3-6-35b-a3b-packed/` (17 GB, 40 files × 432 MB)
- Smoke-ready wrapper (with `packed_experts/` symlink): `qwen3-6-35b-a3b-root/`
- Non-expert weights + manifest + vocab: `qwen3-6-35b-a3b-artifacts/` (model_weights.bin 1.4 GB, model_weights.json, vocab.bin, tokenizer.bin)
- GGUF F16: 69 GB; GGUF Q4_K_M: 21 GB (4.88 BPW).

**Conditional-compile scaffold landed (moeflux commit `32dd06e`):**
- `metal_infer/model_variant.h` owns all shape `#define`s; 22
  per-variant constants + derived 4-bit/2-bit expert-offset macros.
- Select via `-DMOEFLUX_MODEL_QWEN3_5_A17B` (default) or
  `-DMOEFLUX_MODEL_QWEN3_6_35B_A3B`. Makefile accepts `MODEL=`.
- A17B build byte-identical post-refactor (all 12 derived offsets
  match the old hardcoded literals + expert_index.json regression
  check).
- `docs/model_variants.md` documents the add-a-variant flow.
- Shape constants are compile-time only (C needs static array
  sizes); Metal shaders untouched since they use only tile
  constants, not shape constants.

**Prep pipeline scaffold (moeflux commit `79b92c3`):**
- `tools/gen_expert_index.py` — new. Walks mlx_lm output's
  `switch_mlp` tensor layout, builds the per-layer byte-offset map
  `repack_experts.py` consumes. Stdlib-only.
- `repack_experts.py` — rewritten to derive COMPONENTS /
  NUM_EXPERTS / NUM_LAYERS from the generated index. A17B offsets
  match the old hardcoded ones exactly (regression verified).
- `metal_infer/extract_weights.py` — config from HF `config.json`
  (text_config.*), not hardcoded A17B values.
- `metal_infer/export_vocab.py` — new. Decoder-side vocab.bin
  format that `load_vocab` reads (distinct from
  `export_tokenizer.py`'s BPET format for bpe_load; C engine needs
  both). Inverts GPT-2 bytes_to_unicode correctly.

Target smoke-test models (small first, work up):
- **OLMoE-1B-7B** (~4-5GB Q4, 1B active). "Does the pipe work at all."
- **Qwen3-30B-A3B** (~18GB Q4, 3B active). "Does real MoE work."
- **Qwen3.5-A17B** (original flash-moe target, 209GB). Mike is
  downloading this at session end; becomes the first smoke target
  by virtue of being first-available.

Peripheral risks tracked (not blockers):
- **AI-authorship doctrine is current, not permanent.** Thaler +
  2026 SCOTUS cert denial is where we stand. A future ruling
  narrowing the doctrine would not retroactively revoke our
  position, but new forks/work might need a different legal
  frame. Low probability near-term; worth keeping in peripheral
  vision.
- **Bus-factor 1 on the human side.** Mike's husband knows
  about Agora but formal succession planning (Amanda Askell or
  similar for Steward role) is a post-launch concern.

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
