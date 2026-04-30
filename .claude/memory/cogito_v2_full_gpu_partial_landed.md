# Cogito-V2 — Full-GPU forward, partial landing (Phases 0–4a + 4b-partial)

Captured 2026-04-30 end-of-session. Plan-of-record was
`cogito_v2_full_gpu_plan.md`; this is the outcome memo.

## Topology this session

GPU does: MLA attention (single cmdbuf, no host bounces), dense MLP
(layers 0–2), MoE K-expert FFNs + unscaled combine, lm_head.

CPU does: embedding lookup, per-layer pre/post-attn rms_norm, residual
adds, MoE router gate matvec (BF16), noaux_tc routing, shared-expert
SwiGLU MLP, expert-blob disk reads (now parallel via rayon).

## What's bit-exact-validated (cosine ≥ 0.9999)

- `dense_mlp_layer_forward_gpu` vs `dense_mlp_swiglu_cpu` at layer 0
  (`tests/dense_mlp_gpu.rs`): cosine 1.000000, rel 1.4e-6.
- `cogito_moe_layer_forward_gpu` vs `deepseek_moe_cpu` at layer 3
  (`tests/cogito_moe_gpu.rs`): cosine 1.000000, rel 5.7e-8.
- Existing single-token GPU MLA smoke (`tests/cogito_v2_smoke.rs`):
  argmax=5, magnitudes match the CPU oracle and prior session
  (logits min=-3.422 max=20.299).

## Critical latent bug found and fixed

**`gpu_batched_experts_forward` silently truncated inputs with
`in_dim > 4096`.**

`emit_batched_experts` (`expert_forward.rs:782+`) hardcoded
`dequant_matvec_4bit_v3`. That kernel uses a 4096-float threadgroup
input cache (`shaders.metal:274`). For Cogito-V2's `hidden_dim=7168`,
the gate/up matvecs over-ran the cache and silently truncated to the
first 4096 elements — producing ~3% relative drift on the MoE output.
Fix: refactored `encode_matvec_into` to accept `&MatvecPipelines` and
pick `_v3` or `_fast` based on `in_dim`, mirroring `gpu_matvec.rs`'s
existing dispatcher. Affects **any future variant with
`hidden_dim > 4096`** that hits the GPU expert path. Caught only
because Cogito is the first such variant.

## What's in tree (`~/Projects/moeflux`, branch `main`)

5 modified files + 6 new files (uncommitted at memo-write time):

- `shaders/shaders.metal` — `+moe_combine_residual_unscaled`,
  `+mla_split_q_kv`, `+mla_kv_cache_append`. ~190 lines added.
- `src/riir/expert_forward.rs` — variant-aware combine kernel
  selection, matvec pipeline-pair refactor. ~30 lines net.
- `src/riir/gpu_mla.rs` — `+encode_mla_split_q_kv`,
  `+encode_mla_kv_cache_append`, `MlaPipelines` adds two PSOs.
  ~95 lines added.
- `src/riir/mla_attn_forward.rs` — single-cmdbuf rewrite, kills 3 sync
  points. ~120 lines net (removed ~120 host scatter lines, added
  ~80 kernel-encode lines).
- `src/riir/mod.rs` — `step_internal_mla_gpu` now full-GPU; new
  `dense_mlp_pipes` / `dense_mlp_bufs` fields on RsCtx; allocates
  `moe_buffers` from `ensure_mla_resources`.
- `+ src/riir/dense_mlp_gpu.rs` — `DenseMlpBuffers`,
  `DenseMlpPipelines`, `encode_dense_mlp_layer_forward_gpu`,
  `dense_mlp_layer_forward_gpu`.
- `+ src/riir/cogito_moe_gpu.rs` — Cogito MoE wrapper: CPU gate +
  CPU routing + GPU experts + GPU combine. **Parallel expert reads
  via rayon `par_chunks_mut` (Phase 4b-partial)**.
- `+ tests/dense_mlp_gpu.rs`, `tests/cogito_moe_gpu.rs` — per-phase
  diff tests.
- `+ tests/fixtures/cogito_v2_oracle.{response.json,sse.jsonl}` —
  Phase 0 multi-token CPU-oracle fixture: greedy 16-token
  continuation = `"I'm doing well, thank you. How can I help you
  today?"` (15 emitted tokens before EOS).

## Perf state

- Cold first-token: ~26–30s (cogito_v2_eval_token_smoke). Includes
  Metal pipeline JIT + **17.5 GB KV cache memset + page-fault storm**
  + expert page first-touch.
- Warm tok/s: **NOT cleanly measured this session.** blallama prefill
  on the 16-token continuation runs >10 min; we never reached the
  warm-decode window before pivoting to profile.
- **CPU 100% / GPU 0–4%** during Mike's blallama observation. Parallel
  expert reads alone did not visibly shift the ratio.

### Profile (samply, with `--unstable-presymbolicate` sidecar + `RUSTFLAGS="-C force-frame-pointers=yes"` + `CARGO_PROFILE_RELEASE_DEBUG=true`)

Saved at `/tmp/cogito_pre.json` + `/tmp/cogito_pre.syms.json`. 25.4s
single-eval-token wall time at BOS-pos-0. **First-eval profile —
ensure_buffers + Metal pipeline JIT dominate; not representative of
warm-token compute.** The next session needs a 2-eval profile that
isolates token 2.

Top inclusive (single-eval cold):
- `RsCtx::step_internal` 99.4%
- `RsCtx::ensure_mla_resources` 92.9%
  - `MlaKvCacheGpu::ensure_buffers` **91.7%** ← one-time cost
- `cogito_moe_layer_forward_gpu` 6.2%
- `gpu_batched_experts_forward` 3.6%
- `shared_expert_swiglu_cpu` 1.8% (calls `project_4bit_cpu` 1.7%)
- `mla_attn_layer_forward_gpu` 0.9%
- `MetalBackend::pipeline` 1.1% (PSO compile cache)

The huge `ensure_buffers` slice is `zero_shared_buffer` over the
full virtual KV cache: `MAX_SEQ_LEN=131072 × kv_lora_rank=512 × 4 =
256 MB` per latent + `131072 × 64 × 4 = 32 MB` per rope_k, **× 61
layers ≈ 17.5 GB** of CPU memset. Page faults dominate because
shared-storage Metal buffers commit lazily on first touch.

### Implications for next session

1. **Subtract the cold-init from any perf claim.** First-token wall
   time (~25s) is mostly setup, not compute. After ensure_buffers
   has run once, the per-token compute is ~2s in this profile —
   suggesting warm tok/s could be ~0.5 from this code alone. **But**
   Mike's blallama prefill at sustained CPU 100% means warm tokens
   are slower than that — there's a second hot path the cold-only
   profile can't see.
2. **Profile a warm token next session.** Run `eval_token` twice; on
   the second, init has finished and the hot path is just compute.
   `samply --duration` could time-window the capture to skip the
   first 25s.
3. **Likely warm hot paths (educated guesses to verify with the
   warm profile)**:
   - `shared_expert_swiglu_cpu` — 4-bit SwiGLU at hidden=7168 →
     intermediate=2048 → hidden=7168, single-threaded
     `project_4bit_cpu` × 58 MoE layers. Estimated ~2-3s/token.
     Move to GPU.
   - `bf16_matvec_cpu` for the MoE router gate — 256 × 7168 BF16
     matvec × 58 layers. Estimated ~0.3-0.5s/token.
   - Per-layer host bounce in the orchestrator — read MLA out,
     CPU residual add, CPU rms_norm, write back to GPU.
4. **Consider lazy-paging the KV cache.** Allocate per-block
   (e.g., chunk_size=4096 positions) on demand instead of one giant
   per-layer 256 MB buffer. Saves the cold-init storm and reduces
   working set if context stays small.
5. **Frame pointers + presymbolicate sidecar are the right
   profiling setup on macOS.** Captured in
   `scripts/profile_smoke.sh` + `scripts/profile_aggregate.py`
   (in moeflux repo) so future sessions don't re-derive the build
   flags. Usage:
   ```bash
   cd ~/Projects/moeflux
   ./scripts/profile_smoke.sh                      # cogito_v2_smoke
   ./scripts/profile_smoke.sh cogito_moe_gpu       # other test
   ./scripts/profile_smoke.sh --open               # browser flame graph
   # Aggregator also runs standalone:
   python3 scripts/profile_aggregate.py /tmp/moeflux_profile.json --filter moeflux
   ```
   Without those flags samply gets shallow stacks and unresolved
   addresses — every leaf samples to the test harness and the actual
   hot path is invisible. (How we burned an hour this session before
   getting it right.)

## Cut/punted relative to plan-of-record

**Cut**:
- Phase 4b ring + chain integration. Parallel-reads only landed.
  Reason: ad-hoc opt didn't move CPU/GPU ratio; Mike's call was to
  pivot to systematic profile next session before more refactor.
- Phase 5 long-context tiling for `mla_sdpa_folded`. Not started.
- Phase 6 perf measurement on blallama. Prefill timed out twice;
  abandoned for memo time. samply attempt produced empty samples
  (likely macOS sampling-entitlement issue — investigate next
  session).
- GPU routing kernel (`noaux_tc_router_gpu`). CPU routing kept; the
  ~100 µs/layer it costs is dwarfed by other CPU work.
- `MOEFLUX_FORCE_CPU_MOE=1` symmetric env var. Skipped.

**Why these are safe to defer**: bit-exact correctness is locked in
on Phase 3 + 4a; further work is perf-only and can be staged behind
the working forward. Snapshot v2 (Phase 7 of plan-of-record) is a
separate Council-blocker that should land next session in parallel.

## Architectural decisions worth keeping

- **Single-cmdbuf MLA forward** (Phase 4a) eliminates the
  `MlaForwardBuffers` aliasing trap from depth-2 ring concurrency
  preemptively — even if Phase 4b lands, MLA scratch is queue-
  serialized within its own cmdbuf and doesn't need depth-2
  duplication.
- **Variant-flag combine kernel selection** via `combine_kernel_name()`
  helper in `expert_forward.rs`. Sigmoid-gate path stays for Qwen3
  variants, unscaled path for DeepSeek-V3/Cogito. Sibling kernels in
  shaders.metal beat a flag-on-the-existing-kernel for thread-
  divergence cleanliness.
- **MatvecPipelines pair propagation** through `emit_batched_experts`.
  The fix for the `in_dim > 4096` truncation bug. Don't revert.
- **Phase 0 oracle fixture in-tree** at
  `tests/fixtures/cogito_v2_oracle.sse.jsonl`. Future multi-token
  diff tests should compare token sequences against this file.

## Next session — concrete plan

In priority order:

1. **Profile properly**. samply with macOS code signing /
   entitlements (or `cargo flamegraph` / Apple Instruments).
   Capture a CPU profile of `cogito_v2_eval_token_smoke` running.
   Goal: identify the top 3 self-time hot spots. Without this, more
   refactor is guesswork.
2. **Move `shared_expert_swiglu_cpu` to GPU.** The
   `encode_dense_mlp_layer_forward_gpu` helper is reusable —
   parameterize on `(prefix, intermediate)`. Add scratch fields to
   `MoeBuffers` at `shared_intermediate=2048`. Expected savings:
   ~2–3 s/token if it really is the hot spot.
3. **Move `bf16_matvec_cpu` for the gate to GPU.** New `bf16_matvec`
   PSO (separate from the 4-bit family). Saves the K-by-256 score
   readback if the router goes GPU; with CPU routing kept, only
   saves the gate matvec compute itself (smaller win).
4. **Phase 4b ring + chain integration.** Now that CPU work is
   reduced, the GPU/SSD-read overlap from the deferred ring is the
   next multiplier. Use `gpu_batched_experts_encode_pre_staged`
   (the async variant). Fold MLA into the ring via the
   `ChainToNormed` mechanism.
5. **Snapshot v2 wire format** (Phase 7). Council-blocker. Mla arms
   in `state_snapshot.rs:103-104, 210-213, 391-394` need to
   serialize the latent + rope_k cache buffers. Parallelizable with
   #1-#4.
6. **Long-context tiling** for `mla_sdpa_folded`. Online-softmax
   accumulator across tiles; `MLA_MAX_CACHE_TG=4096` is the
   threadgroup-mem cap, tiling lets cache_len > 4096. Verify with
   8k/16k smoke. Mike may run this overnight after a session.

## Critical files & line refs

- `crates/moeflux/src/riir/cogito_moe_gpu.rs` — Cogito MoE entry
  point. Where shared-expert-on-GPU and gate-on-GPU plug in.
- `crates/moeflux/src/riir/dense_mlp_gpu.rs` — reusable SwiGLU FFN
  encode helper. Generalize to take a tensor-prefix arg.
- `crates/moeflux/src/riir/mla_attn_forward.rs:208-460` — single-
  cmdbuf MLA. Phase 4b refactors this to `encode_mla_attn_layer`
  (no commit/wait inside) + ring-push from the orchestrator.
- `crates/moeflux/src/riir/mod.rs:1254-1450` —
  `step_internal_mla_gpu`. Phase 4b's chain handoff lives here:
  drain N-2 → prefetch N → encode CMD1 → CMD2+CMD3 chained → push.
- `crates/moeflux/src/riir/expert_forward.rs:556-720` —
  `gpu_batched_experts_encode` and `_pre_staged`. Phase 4b uses the
  pre-staged variant.
- `crates/moeflux/src/riir/expert_forward.rs:66-100` —
  `ChainToNormed`. Already correct for Cogito's pre-LN layout per
  this-session's exploration.
- `crates/moeflux/src/riir/state_snapshot.rs:103-104, 210-213,
  391-394` — `MlaUnsupported` stubs to replace for Phase 7.
- `crates/moeflux/tests/fixtures/cogito_v2_oracle.{response.json,
  sse.jsonl}` — Phase 0 multi-token diff fixture.

## Run commands

```bash
# Per-phase tests (each green this session):
cargo test -p moeflux --no-default-features --features model-cogito-v2-671b --release \
  --test dense_mlp_gpu -- --ignored --nocapture
cargo test -p moeflux --no-default-features --features model-cogito-v2-671b --release \
  --test cogito_moe_gpu -- --ignored --nocapture
cargo test -p moeflux --no-default-features --features model-cogito-v2-671b --release \
  --test cogito_v2_smoke -- --ignored --nocapture

# CPU oracle fallback (env-gated, kept):
MOEFLUX_FORCE_CPU_MLA=1 cargo test ... --test cogito_v2_smoke ...

# blallama (drama_llama bin):
cd ~/Projects/drama_llama
cargo build --release --bin blallama --features "axum,cli,toml,moeflux-model-cogito-v2-671b"
./target/release/blallama "/Volumes/Temp Backup/models/blallama" \
  --backend moeflux --probe-stream --port 11435
# In another shell:
curl -sN http://localhost:11435/probe > run.sse.jsonl &
curl -X POST http://localhost:11435/v1/messages -H 'Content-Type: application/json' \
  -d '{"model":"cogito-v2-671b","messages":[{"role":"user","content":"Hello. How are you?"}],"max_tokens":16,"temperature":0.0}'
```

## Pointers (durable)

- Plan-of-record: `cogito_v2_full_gpu_plan.md`
- Last session: `cogito_v2_gpu_mla_landed.md`
- Architecture audit: `cogito_v2_architecture.md`
- moeflux RIIR strategy: `riir_moeflux_strategy.md`
