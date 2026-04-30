# Cogito-V2 GPU MLA — first-run green

Captured 2026-04-30. Session goal hit: GPU MLA forward produces
bit-exact same logits as CPU oracle on BOS-at-pos-0 smoke
(`min=-3.422 max=20.299 argmax=5` on both paths). Hybrid topology
this session — GPU does MLA attention, CPU still does MoE / dense
MLP — but the load-bearing kernel work landed.

## What's in tree (`~/Projects/moeflux`, branch `main`)

Five commits stack:

1. `37c5f31` — `MlaKvCacheGpu` swaps `Box<[f32]>` for shared-storage
   Metal buffers (`Option<Buffer>`-wrapped, lazy-allocated by
   `ensure_mla_resources`). CPU path uses unsafe slice accessors
   over `contents()` so it still runs unchanged.
2. `c4b32ac` — `yarn_rope_apply` Metal kernel + `gpu_rope.rs`
   dispatcher. **Uses `metal::precise::cos`/`sin`** — the default
   fast-math versions drift ~3e-4 at pos=4096. Tested ≤4 ULP vs
   `apply_rotary_emb_yarn` libm reference.
3. `b5c52f1` — three folded-form Metal kernels in `gpu_mla.rs`:
   - `mla_q_prime_4bit`: q'[h, c] = Σ_i q_nope[h, i] * dequant(W[h*256+i, c])
   - `mla_sdpa_folded`: per-head scoring + softmax + V_combine
     (4096-cap on cache_len, 16 KB threadgroup `scores[]` + 32-byte
     simd-scratch for cross-simdgroup reductions)
   - `mla_out_per_head_4bit`: out[h, f] = Σ_c V_combine[h, c] *
     dequant(W[h*256+nope+f, c])
   - Per-kernel diff tests synthesize 4-bit weights, compute
     reference on host, assert ≤1e-3 / ≤1e-5 absolute drift.
4. `ea3dcdf` — `mla_attn_layer_forward_gpu` integrates all of the
   above per layer; `step_internal_mla_gpu` orchestrates. Default
   path on Cogito-V2; `MOEFLUX_FORCE_CPU_MLA=1` switches back to
   the CPU oracle.

## Validated against real weights

```text
GPU: cogito_v2_eval_token_smoke ... ok
     min=-3.422 max=20.299 argmax=5  (34.4s — first-token; pipeline JIT)

CPU: MOEFLUX_FORCE_CPU_MLA=1 cogito_v2_eval_token_smoke ... ok
     min=-3.422 max=20.299 argmax=5  (20.4s)
```

Bit-exact. Folded-form math matches naive math; all 4 kernels
work; integration is correct.

## What's still hybrid / open for next session

- **MoE / dense MLP run on CPU.** That's where the 20s/token CPU
  time goes; GPU MLA on its own only ate ~3 of those 20s. To hit
  Mike's ~1 tok/s target we need full-GPU MoE.
  - Path: write `dense_mlp_layer_forward_gpu` (reuse existing
    `dequant_matvec_4bit_v3` + `swiglu_fused`, both
    dim-parametric — confirmed in plan-phase exploration).
  - Path: GPU noaux_tc routing + dispatch routed experts via
    existing `gpu_batched_experts_forward`. Note Cogito's MoE
    composition (unconditional shared expert add, no gate)
    differs from Qwen — needs a sibling of `post_attention_tail`
    or a feature-toggle (`SharedExpertGate::Unscaled` enum already
    exists).
  - Deferred-ring integration with the new MLA forward is the
    clean target — the per-layer prefetch / chained-MoE ring is
    what gives the existing GQA path overlap with SSD streaming.
- **Long-generation stability untested.** Smoke is single-token.
  The CPU path's last test was 8 tokens = "I'm doing well, thanks!
  How". Need a 200+ token gen to verify nothing drifts as
  cache_len grows.
- **First-token overhead is large** (~14s extra over CPU). Mostly
  Metal pipeline JIT (4 new pipelines × ~3.5s each) + 18 GB virtual
  KV cache buffer lazy-commit. Both are one-time. Warm-token
  numbers are unknown — need a multi-token gen to measure.
- **Snapshot v2 (MLA-aware wire format) not started.** Required
  before Cogito ships to Council (prompt caching depends on it).
  Stub is `MlaUnsupported` for both Mla and (now-merged) MlaCpu.
- **Phase 0 oracle never captured** — folded the validation into
  the live argmax-match. If we want fast iteration on per-layer
  hidden-state diffs in the future, capture once via `/probe` SSE.

## Architectural decisions worth keeping

- `MlaKvCacheGpu` uses `Option<Buffer>` (not `Option<MlaKvCacheGpu>`
  on the LayerState) — keeps match-arm shape stable, lets unit
  tests mutate `len` without device init. Convention in this repo
  for lazy GPU resources.
- `MlaForwardBuffers` is a self-contained per-token scratch set,
  not grafted onto `LinearAttnBuffers`. Cleaner separation between
  the two attention flavors. Stays separate when MoE moves to GPU
  unless we find a strong reason to merge.
- `metal::precise::cos`/`sin` are the right answer for any
  position-driven trig with arguments > a few rotations. Default
  Metal `cos`/`sin` are not safe for RoPE.
- Threadgroup memory cap on Apple Silicon is 32 KB total (not
  per-array). 8192-float `scores[]` overflows by 8 bytes once you
  add a single broadcast scratch float. We're at 4096 with simd-
  level reductions; long-context tiling is a future-work slice.

## Pointers

- The plan: `~/.claude/plans/replicated-spinning-snowflake.md`
- CPU oracle: `crates/moeflux/src/riir/mla_attn_cpu.rs`
- GPU forward: `crates/moeflux/src/riir/mla_attn_forward.rs`
- GPU kernels: `crates/moeflux/src/riir/gpu_mla.rs`,
  `crates/moeflux/src/riir/gpu_rope.rs`,
  `crates/moeflux/shaders/shaders.metal` (kernels 13-16)
- Smoke test: `crates/moeflux/tests/cogito_v2_smoke.rs`
- Run with: `cargo test -p moeflux --no-default-features
  --features model-cogito-v2-671b --release --test cogito_v2_smoke
  -- --ignored --nocapture`
- Force CPU oracle: prefix with `MOEFLUX_FORCE_CPU_MLA=1`
