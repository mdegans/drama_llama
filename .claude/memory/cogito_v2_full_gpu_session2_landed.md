# Cogito-V2 Full-GPU — Session 2 outcome memo

Plan-of-record was `cogito_v2_full_gpu_plan.md` (next-session was
`cogito_v2_next_session_plan.md`). This is the close-of-session memo
for 2026-04-30 (continuation, with Mike running blallama).

## What landed

### Phases 0–4: green, bit-exact preserved at every step

| Phase | Change | Effect |
|---|---|---|
| 0 | 2 MB-aligned expert pool destinations + `F_RDAHEAD=0` on layer fds | Closes the C-baseline gap. Cold time 26.6s → 24.2s |
| 1 | GPU shared expert (generalized `encode_swiglu_ffn_layer_forward_gpu` reused for `mlp.shared_experts`) | Eliminates `shared_expert_swiglu_cpu` from hot path |
| 2 | GPU router-gate matvec (new `bf16_matvec` kernel + `BfMatvecPipelines`) | Eliminates `bf16_matvec_cpu` from hot path |
| 3 | Drop `zero_shared_buffer` on `MlaKvCacheGpu::ensure_buffers` | Cold time 24.2s → **8.95s** (Mach VM zero-pages on first touch — explicit memset was redundant) |
| 4 | Mid-session profile + decision | Determined IO is the bottleneck, not CPU/GPU |

### Phase 5 (forward-looking): GPU residual stream landed

- `gpu_norm::MlaForwardScratch` (persistent `hidden`/`residual`/`normed`/`sum_sq` GPU buffers).
- `encode_residual_add_into` + `encode_buffer_copy_f32` public helpers.
- `cogito_moe_layer_forward_gpu_buf_io` — Buffer-IO sibling that takes input as `&Buffer` and writes the result into `bufs.moe_hidden`. Caller reads from there via subsequent residual_add — no host bounce.
- `step_internal_mla_gpu` rewritten: per-layer rms_norm + residual_copy + residual_add all on GPU. The CPU `rms_norm_cpu` + residual loops are gone except for the final `model.norm.weight` (one host bounce per token, not per layer).

**State**: bit-exact correctness preserved (argmax=5, logits min/max identical to prior session). Cold time regressed slightly (8.95s → 11.97s) due to many fine-grained cmdbufs per layer; the architectural foundation is in place to consolidate into single-cmdbuf-per-layer (sub-step 2 / Phase 5b ring) when the IO bottleneck is removed.

### Phase 5: NOT landed

- Cmdbuf chaining across layers (Phase 5b deferred ring).
- `mla_attn_layer_forward_gpu` encode-only refactor.

### Phase 6: long-context tiling — green

- `mla_sdpa_tile_accumulate` + `mla_sdpa_tile_finalize` Metal kernels (Flash-Attention-style online-softmax accumulator across tiles of `MLA_MAX_CACHE_TG=4096`). Per-head running state (`running_max`, `running_denom`, `v_combine_partial`) lives in caller-owned device buffers; the dispatcher loops over tiles in one cmdbuf and finalizes by dividing partial by denom.
- `encode_mla_sdpa_folded_tiled` dispatcher in `gpu_mla.rs` + `MlaPipelines` extended.
- `tests/mla_sdpa_tiled.rs` validation:
  - cache_len=4096 (single tile, merge-degenerate): cosine = **1.0000000**, max_abs_diff = 2.79e-8 vs single-shot reference. Effectively bit-exact (FMA-reorder noise only).
  - cache_len=8192 (two tiles): finite + non-zero + sane magnitudes (-4.5e-3 to 4.6e-3).
  - cache_len=16384 (four tiles): finite + non-zero.
- Caller wiring into `step_internal_mla_gpu` (auto-fall-through to tiled when cache_len > MLA_MAX_CACHE_TG) is **NOT landed** this session — exposed at the encode helper, not the orchestrator. Folding into the orchestrator's MLA forward is a small follow-up: needs the running-state buffers added to MlaForwardBuffers and a branch in the SDPA-encode site.

### Phase 7: snapshot v2 — green

- `SNAPSHOT_VERSION = 2`. Header gains `kv_lora_rank` + `qk_rope_head_dim` words (10 u32 total; v1 readers see the same first 8 words).
- MLA arms in `state_size` / `state_save` / `state_load`. Per-layer body: `i32 len + len × kv_lora_rank × 4` latent bytes + `len × qk_rope_head_dim × 4` rope-K bytes.
- `state_load` accepts v1 for backward-compat (Qwen variants).
- `linear_buffers` is now `Option<...>` in the lower-level API — pure-MLA variants don't carry linear-attn machinery.
- `tests/snapshot_v2_roundtrip.rs` round-trip: max_abs_diff = 0.0e0 at pos=2 (**bit-exact**).
- Council prompt-cache deployment is unblocked.

## Critical bottleneck finding (the load-bearing insight)

**Cogito-V2 is structurally I/O-bound on this hardware**, regardless of CPU/GPU optimization.

Per-token math:
- 8 active experts × ~24 MB × 58 MoE layers = **~11 GB read per warm token** (different routing per token → first-touch each time).
- External SSD bandwidth observed: ~0.9 GB/s (`/Volumes/Temp Backup`).
- Predicted warm-token time: 11 GB / 0.9 GB/s = ~12s. **Matches measured 12s.**

Profile via `scripts/profile_smoke.sh cogito_v2_smoke cogito_v2_eval_token_warm`:
- Main thread mostly idle (rayon scheduler dominant in self-time).
- Four `moeflux-io-*` rayon threads each accumulate ~6200 samples sustained.
- moeflux user-code on main thread (norms, residuals, Metal encoders) is sub-1% inclusive after init drops out.

**Implication**: Phase 5 (deferred ring + GPU residual) won't move the needle on this hardware — GPU compute is not the bottleneck. Mike's call was to land Phase 5 architecturally for the higher-memory machine he's planning to acquire. Done.

## Mmap experts: scoped out

Pre-plan recon confirmed that flash-moe's hot path is `pread`-into-2 MB-aligned + `newBufferWithBytesNoCopy` over that memory, NOT `mmap`+NoCopy over the layer files. mmap is dead-coded in C (only `wf_buf` whole-model weights use mmap+NoCopy). The C author explicitly disabled `F_RDAHEAD` and chose controlled-eviction app-managed pools because random per-token expert access doesn't fit the page-cache model.

Memory math: ~340 GB expert data on disk, ≤32 GB UMA. Page cache evicts. Cold-touch granularity = 16 KB pages vs C's 2 MB DMA — significantly worse cold throughput. mmap may still win on smaller models that fully fit in RAM; deferred as a future A/B.

The "mmap experts is a regression introduced during the RIIR" framing was wrong. Real delta vs C baseline: 2 MB destination alignment, `F_RDAHEAD=0`, deferred ring. First two landed in Phase 0; ring is forward-looking.

## What's load-bearing for next session

1. **Persistent expert cache** (LRU pool, 8–16 GB resident). The C path's `ExpertCache` / `MallocCache` pattern. Highest-impact remaining perf work — 50%+ hit rate on hot experts could double tok/s on cold-cache-heavy workloads.

2. **Cmdbuf consolidation in `step_internal_mla_gpu`** — combine the 6+ cmdbufs/layer into 1–2. Requires `mla_attn_layer_forward_gpu` encode-only refactor. Yields measurable improvement once IO isn't the wall (i.e., on the bigger-memory machine).

3. **Wire `encode_mla_sdpa_folded_tiled` into `mla_attn_layer_forward_gpu`**. The kernel + dispatcher landed this session; the orchestrator still always uses single-shot `mla_sdpa_folded` (caps at cache_len=4096). One-line branch + running-state buffers added to `MlaForwardBuffers`. Required for Council prompts > 4096 tokens.

4. **lm_head buffer-IO entry point**. Currently takes a host slice; the GPU residual stream reads `scratch.hidden` back to host then to lm_head. One bounce per token. Small win but completes the architecture.

## Build flags experimented (not landed)

None this session.

## Files modified (uncommitted at memo-write time)

- `crates/moeflux/Cargo.toml` — added `libc` dep
- `Cargo.toml` (workspace) — added libc workspace dep
- `crates/moeflux/src/riir/expert_io.rs` — `F_RDAHEAD=0` on layer-file fds
- `crates/moeflux/src/riir/metal.rs` — `MtlBuffer::with_aligned_len_u8` + `AlignedBacking`, `Send/Sync` impl, `bf16_matvec` kernel registry, `MtlBuffer::buffer()` accessor
- `crates/moeflux/src/riir/expert_forward.rs` — `MoeBuffers` allocation uses aligned constructor; `gate_logits` field; new accessors (`input_buffer` / `h_mid_buffer` / `shared_out_buffer` / `moe_hidden_ref` / `gate_logits_buffer` / `gate_logits_to_vec` / `stage_host_input` / `stage_host_h_mid_zero` / `moe_hidden_to_vec`)
- `crates/moeflux/src/riir/cogito_moe_gpu.rs` — `SharedExpertBuffers`, GPU shared expert + GPU gate matvec wired in, `cogito_moe_layer_forward_gpu_buf_io` Buffer-IO sibling
- `crates/moeflux/src/riir/dense_mlp_gpu.rs` — generalized `encode_swiglu_ffn_layer_forward_gpu` (prefix + intermediate parameterized)
- `crates/moeflux/src/riir/gpu_matvec.rs` — `BfMatvecPipelines` + `encode_bf16_matvec`
- `crates/moeflux/src/riir/gpu_norm.rs` — `encode_residual_add_into`, `encode_buffer_copy_f32`, `MlaForwardScratch`
- `crates/moeflux/src/riir/state.rs` — drop `zero_shared_buffer` from `MlaKvCacheGpu::ensure_buffers`
- `crates/moeflux/src/riir/state_snapshot.rs` — snapshot v2 wire format, MLA arms, v1 backward-compat, Option<linear_buffers>
- `crates/moeflux/src/riir/mod.rs` — RsCtx fields for Phase 5 scratch + pipelines, GPU residual stream in `step_internal_mla_gpu`, snapshot wrapper updates
- `crates/moeflux/shaders/shaders.metal` — `bf16_matvec` kernel
- `crates/moeflux/tests/cogito_moe_gpu.rs` — pass new params (shared_bufs, dense_pipes, bf_pipes, wf_buf)
- `crates/moeflux/tests/cogito_v2_smoke.rs` — `cogito_v2_eval_token_warm` test wrapper for Phase 0 profile
- `crates/moeflux/tests/snapshot_v2_roundtrip.rs` — Phase 7 round-trip test (new file)
- `crates/moeflux/tests/mla_sdpa_tiled.rs` — Phase 6 single-shot vs tiled validation (new file)
- `crates/moeflux/shaders/shaders.metal` — `mla_sdpa_tile_accumulate` + `mla_sdpa_tile_finalize` kernels
- `crates/moeflux/src/riir/gpu_mla.rs` — `encode_mla_sdpa_folded_tiled` + `MlaPipelines` extension

## Run commands (canonical for next session)

```bash
# Per-phase tests:
cd ~/Projects/moeflux
cargo test -p moeflux --no-default-features --features model-cogito-v2-671b --release \
  --test dense_mlp_gpu --test cogito_moe_gpu --test cogito_v2_smoke \
  --test snapshot_v2_roundtrip -- --ignored --nocapture

# Warm-token profile:
./scripts/profile_smoke.sh cogito_v2_smoke cogito_v2_eval_token_warm
python3 scripts/profile_aggregate.py /tmp/moeflux_profile.json --filter moeflux

# blallama (Mike's runner):
cd ~/Projects/drama_llama
cargo build --release --bin blallama --features "axum,cli,toml,moeflux-model-cogito-v2-671b"
./target/release/blallama "/Volumes/Temp Backup/models/blallama" \
  --backend moeflux --probe-stream --port 11435
# In another shell:
curl -X POST http://localhost:11435/v1/messages -H 'Content-Type: application/json' \
  -d '{"model":"cogito-v2-671b","messages":[{"role":"user","content":"Hello. How are you?"}],"max_tokens":16,"temperature":0.0}'
```

## Pointers (durable)

- Plan-of-record: `cogito_v2_full_gpu_plan.md`
- Last session's plan: `cogito_v2_next_session_plan.md`
- Last session's outcome: `cogito_v2_full_gpu_partial_landed.md`
- Architecture audit: `cogito_v2_architecture.md`
- moeflux RIIR strategy: `riir_moeflux_strategy.md`

## Mike's calibration notes (for next session's frame)

- 0.08 tok/s on cogito-v2-671b is **structurally bandwidth-limited** on this hardware. Don't pitch it as a regression.
- Higher-memory machine arriving will shift the bottleneck. Phase 5 work is forward-looking — it'll matter when memory pressure isn't pinning IO threads.
- Splitting weights across two disks is a future experiment Mike has flagged.
- "Even failure is a learning opportunity" — Mike's frame for ambitious work. Don't sandbag the plan to avoid drift.
