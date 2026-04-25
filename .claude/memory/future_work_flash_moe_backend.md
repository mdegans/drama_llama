# Future work: flash-moe decode backend for drama_llama

> **STATUS (2026-04-27): historical-record / largely shipped.** This was
> the north-star planning doc written before any code. The live state
> now lives in `plan_v0.8.0_backend_split.md` and
> `plan_phase4_a3b_gate_offset_fixed.md`. Quick map of what changed
> from the plan in this doc:
>
> - **flash-moe → moeflux.** Forked, KV-seq_rm patch landed, C API
>   exported (`mf_*` extern block), bindgen wrapper (`moeflux-sys`)
>   and safe wrapper (`moeflux`) crates published in-repo.
> - **`Decoder`/`Model` trait extraction landed** — drama_llama now has
>   `MoefluxDecoder`/`MoefluxModel`/`MoefluxEngine` under
>   `cfg(feature="moeflux", target_os="macos")`. `LlamaCppDecoder` /
>   `LlamaCppEngine` is the parallel pair.
> - **Phase 4 quality gate PASSED 2026-04-27** — A3B canonical pangram
>   matches MLX (cosine 0.9990, top-20 20/20). Root cause of the
>   pre-fix ~3% argmax agreement was 10 hardcoded A17B byte offsets
>   missed by the prior literals cleanup; fixed in moeflux `925f7a0`.
> - **MLX-regression test infra landed** at
>   `moeflux/crates/moeflux/tests/mlx_regression.rs` with golden
>   fixture for A3B; A17B golden deferred (210 GB MLX checkpoint
>   needs ≥256 GB host RAM to load).
>
> What's still pending from this doc's work-items list:
>
> - **`Session` is still LlamaCppEngine-coupled** — `MoefluxSession`
>   (or generic `Session<E: Engine>`) is the next-up task to make
>   blallama (and other consumers above the trait layer) backend-
>   agnostic. Tracking in the meta-plan as the live next step.
> - **Cogito 600B bring-up** — still future. A3B Phase 4 was the gate
>   that unblocks it. Fixture-generation host-RAM constraint applies
>   (same 256 GB-class machine needed for the MLX-regression golden).
> - **Probe-against-600B run** — depends on Cogito bring-up.
>
> Body below is preserved verbatim as scouting reference for anyone
> learning the codebase or planning a fresh MoE-variant integration.
> The "why this ordering matters" section in particular (KV-seq_rm
> before trait extraction) remains the right ordering and is
> reusable lore. Timing estimates were optimistic — actual was ~5 days
> across 6 sessions, not 1-2 days.

## Context

Agora's Council needs an independence path from the Anthropic API — see
agora's `memory/project_alignment_drift_canary.md` for the drift threat
model and agora-agents#14 for the probe that baselines models against
it. Deep Cogito's 600B-parameter MoE (parent of cogito-32b, publicly
available) is the target Council model. Consumer-grade RAM can't
hold 600B in memory, but Apple Silicon's unified-memory + fast-NVMe
combination lets us stream experts from SSD.

[danveloper/flash-moe](https://github.com/danveloper/flash-moe) already
proves the approach: runs Qwen3.5-397B-A17B at 4.4 tok/s on a 48GB
M3 Max MacBook via pure Metal/C. Our 96GB Macbook should do similar
or slightly better on Cogito 600B due to higher page-cache hit rate
relative to working set.

**The plan:** swap flash-moe's decode backend into drama_llama as an
alternative to llama.cpp. Everything above the decode step (axum
wrapper, sampling chain, grammar compilation, prefix cache, probe
integration, `output_config`, phase_split) stays unchanged.

## Architecture: pluggable decode via a trait

drama_llama's current coupling: `Engine` wraps llama.cpp; `Session`
drives it. The swap extracts the logit-producing surface into a
trait implemented by two backends.

```rust
pub trait Decoder: Send {
    /// Load prefix tokens into KV. Called before sampling loop.
    fn eval_prompt(&mut self, tokens: &[Token]) -> Result<(), DecodeError>;

    /// Advance one token, return logits for the next-token distribution.
    fn eval_token(&mut self, token: Token) -> Result<Logits, DecodeError>;

    /// Truncate KV cache to positions [0, len). Mirrors llama.cpp's
    /// `llama_memory_seq_rm`. Enables drama_llama's prefix-cache reuse.
    fn memory_seq_rm(&mut self, start: usize, end: Option<usize>);

    /// Clear KV entirely (equivalent to seq_rm(0, None)).
    fn memory_clear(&mut self);

    fn n_vocab(&self) -> usize;
    fn eos(&self) -> Token;
    fn model_name(&self) -> &str;
}
```

Two impls:
- `LlamaCppDecoder` — current behavior, Linux/Mac via llama-cpp-sys-3
- `FlashMoeDecoder` — Apple Silicon only, via FFI into forked flash-moe

`Engine` becomes a type-erased wrapper: `Engine { decoder: Box<dyn Decoder>, model: Model, ... }`. Construction picks the backend based on build cfg + args.

## flash-moe scout findings

Based on reading `metal_infer/infer.m` (7151 LOC), `main.m` (1847), `chat.m` (760).

### Good news
- **Position-indexed KV.** `apply_rotary_emb(q, k, pos, ...)` and `kv_cache_size = GPU_KV_SEQ * kv_dim * sizeof(float)` — cache is preallocated, position-addressed. Truncation is conceptually "set `pos` back to N and overwrite from there". The hard part (moving KV around) isn't needed.
- **Metal kernels are self-contained** in `shaders.metal`. Repack scripts (`repack_experts.py`, `export_tokenizer.py`, `extract_weights.py`) are model-agnostic in shape.
- **`chat.m` is the reference interactive loop** and shows the inference flow shape: init → load model → eval prompt → token-at-a-time eval loop. A decoder-style API can be extracted from that flow.
- **No llama.cpp dependency.** flash-moe is Metal+C+Accelerate BLAS only. No version conflicts when both backends are present.

### Things to handle
- **No KV-seq_rm API.** Must add it in our flash-moe fork. Mechanism is trivial — reset position counter, optionally zero buffers `[start, old_end)`. New function ~50 LOC. `drama_llama`'s prefix-cache reuse depends on this existing.
- **Model shape hardcoded.** `NUM_FULL_ATTN_LAYERS`, `HIDDEN_DIM`, `NUM_KV_HEADS`, `HEAD_DIM`, `GROUP_SIZE`, layer count, expert count, gate shape — all `#define`-style constants throughout `infer.m`. For Cogito 600B we need to parameterize or fork-and-rebuild. Parameterizing is cleaner long-term but fork-and-rebuild is faster to start. Recommend fork with runtime params where cheap, rebuild-per-model where Metal shader specialization needs them.
- **Expert pack format is model-specific.** The `repack_experts.py` + `extract_weights.py` pipeline needs re-running per target model. One-time operational cost; not a runtime concern.
- **Tokenizer is Qwen-specific.** Cogito 600B likely uses the Qwen tokenizer or a close variant — if identical, reuse; otherwise `export_tokenizer.py` needs a per-model pass.

### Binding recommendation
**Bind the C directly via `bindgen`, don't rewrap Metal in Rust.**

- flash-moe's Objective-C is a thin wrapper over Metal anyway — the C layer is where the pipeline/scheduling logic lives, and it's already debugged.
- Rewriting pipeline orchestration in Rust costs days and adds no capability.
- The C surface exposed to drama_llama is small: `init_model`, `eval_prompt`, `eval_token`, `memory_seq_rm`, `memory_clear`, `free_model`. Maybe 6-10 extern functions.
- Future refactor to pure Rust is possible once we have a working end-to-end and know which C bits hurt.

Fork at `claudeopusagora/flash-moe` with the KV-seq_rm patch applied + the C API extern block added. drama_llama's `Cargo.toml` pulls the fork via git dep.

## Work items in order

1. **Fork flash-moe, add KV-seq_rm + C API.** One PR upstream to
   danveloper (might get merged; friendly). Add these exported C
   symbols:
   ```c
   void *dl_init_model(const char *model_path, const char *repacked_experts_dir);
   int dl_eval_prompt(void *ctx, const int32_t *tokens, size_t n);
   int dl_eval_token(void *ctx, int32_t token, float *logits_out);
   void dl_memory_seq_rm(void *ctx, size_t start, size_t end);  /* end=SIZE_MAX → clear-to-end */
   void dl_memory_clear(void *ctx);
   size_t dl_n_vocab(void *ctx);
   int32_t dl_eos(void *ctx);
   const char *dl_model_name(void *ctx);
   void dl_free_model(void *ctx);
   ```

2. **Extract `Decoder` trait in drama_llama.** Move current
   `Engine`'s logit-producing methods behind the trait. Default
   impl `LlamaCppDecoder` that preserves existing behavior
   byte-for-byte. Feature flag `flash-moe-backend` gates the
   alternate impl. This is the bulk of drama_llama-side surgery —
   touches `src/engine.rs`, `src/session/mod.rs`'s `prepare_call*`,
   `kv_setup_for_call`, and anywhere else `engine` fields are
   accessed. Compile-time abstraction; no runtime dispatch cost.

3. **Implement `FlashMoeDecoder` via bindgen.** Wraps the exported C
   API. `build.rs` runs bindgen against the flash-moe fork's header
   + links the static lib. Handle Metal framework linking (`-framework Metal`,
   `-framework Foundation`).

4. **End-to-end smoke test with Qwen3.5-A17B first, not 600B.** The
   model flash-moe is natively written for. Validates the decoder
   trait + binding. 209GB model; disk + time to repack but a known
   target. Once that runs against drama_llama's axum server, swap
   to Cogito 600B.

5. **Parameterize model shape** if Cogito 600B differs structurally
   from Qwen3.5-A17B. Metal-shader specialization probably needs
   rebuild-per-model; pipeline constants can be runtime.

6. **Run the probe against Cogito 600B** via the drama_llama axum
   wrapper — same questionnaire as agora-agents#14's
   `probe/questionnaires/v0.json`, same baseline file, same
   tolerance. Compare to cogito-32b baseline. If drift beyond
   tolerance, we have governance-relevant evidence distillation
   altered priors.

## Open questions (worth answering before starting)

1. **Does Cogito 600B share Qwen3.5-A17B's architecture exactly?**
   If yes, flash-moe runs it with minimal changes. If the expert
   count or attention topology differs, more work.

2. **Metal-specific build on Linux development machines?** The
   Rust crate should cfg-gate `FlashMoeDecoder` to `target_os = "macos"`.
   Linux builds get `LlamaCppDecoder` only. CI matrix needs both.

3. **thinking mode compatibility.** flash-moe's `chat.m` mentions
   tool-calling as "full tool calling" at 4-bit; thinking blocks
   should work since drama_llama handles `<think>` tokens at the
   parser level, not the decode level. Verify during end-to-end.

4. **Temperature / sampling.** flash-moe includes basic sampling in
   `chat.m` but drama_llama wants to drive sampling itself via the
   `Decoder` trait returning raw logits. Ensure flash-moe's
   `eval_token` can be called purely for logits, no internal
   sampling.

## Timeline

Mike's 1-2h estimate is optimistic but directionally right for the
trait-extraction + binding work if flash-moe's KV-reset patch is
small. Realistic split:

- Fork + KV-seq_rm patch + C API: 2-4h
- drama_llama trait extraction: 3-5h (most churn)
- bindgen impl + build setup: 2-3h
- End-to-end smoke: 1-3h
- Cogito 600B shape adaptation (if needed): 2-8h
- Probe run on 600B: <1h

Call it 1-2 focused days for a working-end-to-end prototype
excluding model repack pipeline. The repack is its own thing (runs
once per model; takes hours to do the 4-bit quantize + pack step on
the source f16 weights).

## Testing

- Unit tests on `Decoder` trait implementations via a mock decoder
  that returns canned logits.
- Byte-for-byte regression tests: `LlamaCppDecoder` output on the
  current test suite should not change. Add golden-output tests
  capturing current behavior before the refactor.
- Integration: run the existing probe (`cargo run -p agora-agent-lib
  --example probe_canary`) against drama_llama with both backends
  and verify structured output works end-to-end.

## Deferred / non-goals

- **Not in scope:** non-Metal platforms (CUDA, Vulkan, ROCm via
  flash-moe). Those are fresh implementations of the streaming-MoE
  approach on different hardware, which is interesting but separate.
- **Not in scope:** running the probe itself as part of the swap
  PR. Probe ships in agora-agents and just needs drama_llama to work.
- **Not in scope:** auto-repack on model add. Humans repack once per
  target model; drama_llama consumes repacked directories only.

## Why this ordering matters

KV-seq_rm first is load-bearing. drama_llama's whole prefix-cache-
reuse story depends on it. If we shipped a `FlashMoeDecoder` without
seq_rm, we'd silently lose cache-hit speedups on every call where
the new prompt shares prefix with the previous one — which is the
common case for Council agents probing + responding on thread
history. Don't skip.

## Pointers

- Probe + baselines: `agora-agents/crates/agora-agent-lib/src/probe/`
- Canary memory: `agora/memory/project_alignment_drift_canary.md`
- flash-moe upstream: https://github.com/danveloper/flash-moe
- drama_llama Session / Engine coupling: `src/session/mod.rs` (top
  of `run_call`, `prepare_call_cached`, `kv_setup_for_call`) +
  `src/engine.rs`
