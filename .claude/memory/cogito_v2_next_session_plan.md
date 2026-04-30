# Cogito-V2 — next-session plan (no-punt)

Companion to `cogito_v2_full_gpu_partial_landed.md`. Drafted at the
close of session 2026-04-30 with Mike's request for a complete plan
to land everything outstanding. Goal: **≥1 tok/s warm, snapshot v2
landed, long-context tiling landed, profile-driven optimizations in
order of impact.**

## State going in

- Bit-exact-correctness floor locked: Phases 0-4a green at cosine
  ≥0.9999 vs CPU oracle (BOS-pos-0). Phase 4b parallel-reads landed.
- Topology: GPU MLA (single cmdbuf), GPU dense MLP, GPU MoE expert
  FFNs + unscaled combine, GPU lm_head. CPU still does: pre/post-attn
  rms_norm, residual adds, MoE router gate matvec (BF16), noaux_tc
  routing, shared-expert SwiGLU MLP, expert-blob disk reads (now
  parallel), embedding lookup, final norm.
- Cold first-token: ~26-30s. **17.5 GB ensure_buffers memset + page
  fault storm = ~23s of that.** Warm-token compute is unprofiled.
- Profiling tooling in tree: `scripts/profile_smoke.sh` +
  `scripts/profile_aggregate.py` (frame pointers + presymbolicate).

## Success criteria (this session)

1. **Warm-token profile captured** — token-2 onward, isolating the
   real per-token cost from the one-time init.
2. **≥1 tok/s warm** on blallama with cogito-v2-671b. Stretch: 2 tok/s.
3. **Token sequence matches Phase 0 oracle fixture** end-to-end on
   greedy 16-token "Hello. How are you?" continuation.
4. **Snapshot v2 round-trip green** — save/load a 200-token state and
   verify post-load logits match pre-save.
5. **Long-context smoke** (≥8192 cache positions) coherent on
   eyeball read.

## Phase order (load-bearing first)

Phases 0-1 are STRICT prerequisites. Phases 2-5 are perf/feature work
ordered by expected impact; reorder only if the warm profile points
elsewhere.

### Phase 0 — warm-token profile (~15 min)

**This is non-negotiable.** Without warm-token data we'd be guessing
about hot paths; cold-profile artifacts (ensure_buffers, pipeline JIT)
would dominate every reading.

1. Add `cogito_v2_eval_token_warm` to `tests/cogito_v2_smoke.rs` —
   calls `ctx.eval_token` twice, asserts logits on the **second**
   call. The first call pays init costs.
2. `scripts/profile_smoke.sh cogito_v2_smoke cogito_v2_eval_token_warm`
   captures and aggregates.
3. Walk the top-25 inclusive **filtered to `moeflux::`** to see actual
   per-token CPU hot paths.
4. Update this plan's Phase 1+2 priority based on findings.

Likely outcomes (educated guesses to verify):
- `shared_expert_swiglu_cpu` dominant (~30-50% of warm token).
- `bf16_matvec_cpu` for gate ~5-10%.
- `gpu_batched_experts_forward` (mostly GPU wait) ~20-30%.
- Per-layer rms_norm/residual_add CPU ~5-15%.

### Phase 1 — GPU shared expert (~45 min)

**Expected impact**: 2-3 s/token CPU saved; biggest single win pending
profile confirmation.

The shape is identical to dense MLP (gate × up × swiglu × down) at
intermediate=`shared_intermediate=2048`. Reuse the dense MLP encoder
helper.

1. **Generalize `encode_dense_mlp_layer_forward_gpu`** to take a
   tensor prefix and intermediate width. Rename to something like
   `encode_swiglu_ffn_gpu` (or expose a generic helper alongside the
   dense-MLP wrapper). Tests stay green.
2. **Add `SharedExpertBuffers`** to `MoeBuffers` (or a sibling
   struct): `shared_in[hidden_dim]`, `shared_gate_out[shared_intermediate]`,
   `shared_up_out[shared_intermediate]`, `shared_act[shared_intermediate]`,
   reusing existing `bufs.shared_out[hidden_dim]`. ~24 KB extra per
   buffer.
3. **`encode_cogito_shared_expert_gpu`** in `cogito_moe_gpu.rs` —
   dispatcher that calls the SwiGLU helper with prefix
   `model.layers.{i}.mlp.shared_experts` and writes into
   `bufs.shared_out`.
4. **Replace `shared_expert_swiglu_cpu` call** in
   `cogito_moe_layer_forward_gpu` with the GPU dispatch. The output
   stays in `bufs.shared_out` — `gpu_batched_experts_forward` already
   reads it from there (no host roundtrip needed).
5. **Modify `gpu_batched_experts_encode`** to accept a "shared_out
   already in bufs" mode that skips the host-slice copy at line 610.
   Or: read back to host and re-stage (small cost, simpler).
6. Validate: `cogito_moe_gpu.rs` test still cosine ≥ 0.9999.

### Phase 2 — GPU gate matvec (~30 min)

**Expected impact**: 0.3-0.5 s/token saved; smaller but on critical
path between layers.

The router gate is `[256, hidden_dim]` BF16, NOT 4-bit. New PSO needed.

1. **Add `bf16_matvec` Metal kernel** in `shaders.metal` — takes a
   BF16 weight tensor + f32 input, emits f32 output. Pattern: each
   threadgroup handles one output row, lanes do dot product.
2. **Add `BfMatvecPipelines` in `gpu_matvec.rs`** with a fetch
   helper.
3. **Stage gate logits buffer in `MoeBuffers`** (256 floats).
4. **Replace `bf16_matvec_cpu` call** in
   `cogito_moe_layer_forward_gpu` with the GPU dispatch. Still
   readback to a host slice for CPU routing — leave routing on CPU
   for now (it's <100 µs, not a bottleneck).
5. Validate: `cogito_moe_gpu.rs` test still cosine ≥ 0.9999.

**Skip `noaux_tc_router_gpu`** unless the warm profile shows the
host readback of 256 scores is itself a bottleneck — unlikely, but
verify.

### Phase 3 — KV cache lazy paging (~30 min)

**Expected impact**: eliminates the 23s cold-init storm; doesn't
affect warm tok/s but matters for prefill/Council prompt-cache warmup.

Currently allocates `MAX_SEQ_LEN * kv_lora_rank * 4 = 256 MB` per
layer × 61 layers = 15.6 GB latent + 32 MB rope_k × 61 = 1.9 GB
rope_k. CPU memset on first eval = ~17.5 GB → 4.5M page faults.

Options (pick one):
1. **Allocate per-block on demand** — split each layer's cache into
   chunks of (e.g.) 4096 positions = 8 MB per chunk. Allocate chunks
   lazily as cache_len grows. Most workloads stay under 8k positions
   so only the first chunks are ever paid for.
2. **Skip the memset** — Metal's `StorageModeShared` buffers are
   probably zeroed by mmap default; verify and drop `zero_shared_buffer`
   call. Saves the CPU cost but page faults still happen on first
   write — fine, that's amortized across token writes anyway.
3. **Allocate at smaller `MAX_SEQ_LEN`** by default (say 8192) and
   re-allocate on cache-grow.

Recommend **option 2** as the minimum viable fix — verify Metal
shared-storage buffers are zero-initialized, drop the explicit
memset. If page faults still cause a noticeable spike, layer in
option 1.

### Phase 4 — Deferred ring + chain + GPU residual stream (~90-120 min)

**Expected impact**: large. Two coupled wins:
1. Eliminates ~244 host bounces per token (read MLA out → CPU
   residual+norm → write back to GPU buffer × 61 layers × 4
   transitions). Currently `step_internal_mla_gpu` keeps `hidden` as
   a host `Vec<f32>` and CPU-computes rms_norm + residual_add 122
   times per token. After this phase, `hidden` lives on the GPU
   throughout the layer loop.
2. CPU encoding overlaps GPU compute via the depth-2 deferred ring.

This is THE phase where the architecture changes shape. Substantial
refactor; biggest payoff if the warm profile confirms host-bounce
churn is meaningful.

1. **Move `hidden` to a GPU `Buffer`** (`shared` storage, hidden_dim
   floats, persistent across the layer loop). All per-layer
   intermediates (`residual`, `normed`, `block_out`,
   `mla_out_host`) become GPU buffers; no host Vec scratch.
2. **Use existing `encode_rms_norm_bf16_into`** and
   `encode_residual_add` Metal kernels (already in
   `shaders.metal:817, 851, 874, 898`) for the per-layer norm and
   residual-add — kills the CPU `rms_norm_cpu` and CPU residual
   loop.
3. **Refactor `mla_attn_layer_forward_gpu`** to expose an
   `encode_mla_attn_layer` (caller-owned cmdbuf, no commit/wait
   inside). Keep the synchronous wrapper for tests.
4. **Refactor `cogito_moe_layer_forward_gpu`** similarly — split
   into encode (caller cmdbuf) + sync wrapper. Note: gate matvec +
   routing stay CPU but no longer block the cmdbuf — they emit
   into a GPU buffer at the right point in the chain.
5. **Add `encode_cogito_moe_chained`** that uses
   `gpu_batched_experts_encode_pre_staged` (`expert_forward.rs:660`)
   and accepts a `ChainToNormed` for the next-layer norm.
6. **Audit `prefetch::predict_for`** for `first_k_dense_replace > 0` —
   prediction must no-op for layers 0-2 (dense MLP, no experts)
   and kick clean at layer 3.
7. **Modify `step_internal_mla_gpu`** per-layer cycle:
   "drain N-2 → prefetch N → encode CMD1 (input rms_norm + Q +
   KV-A) → encode CMD2 (split + RoPE + cache_append + SDPA +
   o_proj + residual_add) → encode CMD3 (post-norm + MLP/MoE +
   chained-input-norm-into-N+1) → push ring". CPU work
   (gate matvec + routing) happens between encode and commit on the
   parent cmdbuf chain, with results staged into GPU buffers.
8. Validate: full-token logits **bit-equal** ring-on vs ring-off
   (math unchanged). New smoke `cogito_v2_full_gpu_ring`.

If the warm profile shows host bounces aren't a top-3 cost
(unlikely but possible), skip the GPU residual stream and only do
the ring overlap (cuts Phase 4 to ~45 min).

### Phase 5 — Long-context tiling for `mla_sdpa_folded` (~30 min)

**Expected impact**: enables cache_len > 4096; required for any
real-world Council prompt. Doesn't help short-context perf.

1. **Add `mla_sdpa_folded_tiled` kernel** in `shaders.metal` —
   processes one tile of `cache_len` positions, takes/updates
   running `(max[h], denom[h], v_combine[h, c])` state.
2. **Outer loop in dispatcher** in `gpu_mla.rs` — splits
   `cache_len` into chunks of `MLA_MAX_CACHE_TG=4096`, allocates
   running-state buffers (per-head fp32), loops.
3. **Single-tile equivalence test** — cache_len=2048 must produce
   bit-equal output with both tiled and single-shot kernels.
4. **Long-context smoke** at cache_len=8192 and 16384 — synthetic
   inputs, compare against single-tile reference. Cosine ≥ 0.9999.

### Phase 6 — Snapshot v2 wire format (~45 min)

**Expected impact**: blocks Council deployment (prompt caching).
No perf impact.

1. **Wire format**: bump magic/version in `state_snapshot.rs`. v2
   reader accepts v1 (backward-compat for Qwen).
2. **`MlaUnsupported` stubs** at `state_snapshot.rs:103-104,
   210-213, 391-394` — replace with serialize/deserialize for
   `LayerState::Mla`. Per layer: `(len: i32, latent_cache[..len *
   kv_lora_rank], rope_k_cache[..len * qk_rope_head_dim])`.
3. **Caller invariant**: no GPU work in flight at save/load (same
   as existing snapshot path; document explicitly).
4. **Round-trip test**: save state at pos=200, load, verify
   `eval_token` produces identical logits to pre-save. New test
   `tests/snapshot_v2_roundtrip.rs`.
5. **drama_llama side**: no changes — `Session::checkpoint_pos`
   already calls `state_save` on RsCtx.

### Phase 7 — Validation + perf measurement (~30 min)

1. **Token sequence check**: 16-token greedy gen via blallama
   matches `tests/fixtures/cogito_v2_oracle.sse.jsonl` token-by-token
   (logits don't bit-match across CPU↔GPU; tokens at T=0 should).
2. **Warm tok/s measurement**: blallama prefill + decode with
   `--probe-stream`; capture SSE jsonl, compute warm-token wall
   clock from `ts_ms` deltas, ignore first-N tokens for warmup.
3. **Activity Monitor sanity**: GPU saturated (>50%), CPU not
   pegged at 100%.
4. **Update `cogito_v2_full_gpu_landed.md`** with achieved tok/s,
   what's still hybrid, what's untested.

## What stays on CPU after all phases land

Minimal CPU set per warm token:
- Embedding lookup (1×, ~28 KB BF16 → f32 conversion).
- noaux_tc routing — sub-100 µs/MoE-layer × 58 = ~6 ms/token.
- Per-MoE-layer expert-blob disk reads (parallel via rayon, hidden
  behind GPU compute by the deferred-ring prefetch).
- Final rms_norm before lm_head (1×, single tensor).
- Cmdbuf encoding overhead (rayon-pooled thread, overlaps with GPU
  compute via the ring).

That's it. Everything compute-heavy moves to GPU. CPU is glue +
disk I/O orchestration after this session.

## Cut order if squeezed

1. Phase 6 (snapshot v2) → next session. Council deployment
   non-blocking on perf.
2. Phase 5 (long-context tiling) → next session. Smoke uses
   cache_len < 4096.
3. Phase 4 (ring) → next session. If Phases 1+2 already hit ≥1
   tok/s, the ring is a bonus rather than a target.

**MVP that hits ≥1 tok/s warm**: Phase 0 + Phase 1 + Phase 2 +
Phase 7 measurement. Phase 3 is optional but cleans up cold-init.

## Open questions to resolve in-session

1. **Does the Phase 0 warm profile actually show the expected
   shared_expert_swiglu_cpu dominance?** If not, reorder Phase 1
   based on what's actually hot.
2. **Does Metal `StorageModeShared` buffer allocation zero by
   default?** Determines whether Phase 3 option-2 is sufficient.
3. **Does `gpu_batched_experts_forward` accept "shared_out already
   in bufs"?** Determines whether Phase 1 needs to modify it (look
   at `expert_forward.rs:610`).

## Critical files & line refs (durable)

- `crates/moeflux/src/riir/cogito_moe_gpu.rs` — Phase 1+2 entry
  point. Where shared-expert GPU and gate-matvec GPU plug in.
- `crates/moeflux/src/riir/dense_mlp_gpu.rs` —
  `encode_dense_mlp_layer_forward_gpu`. Generalize to take a
  prefix arg for Phase 1 reuse.
- `crates/moeflux/src/riir/expert_forward.rs:610` —
  `gpu_batched_experts_encode` shared_out staging. Phase 1 modifies.
- `crates/moeflux/src/riir/expert_forward.rs:660` —
  `gpu_batched_experts_encode_pre_staged`. Phase 4 calls.
- `crates/moeflux/src/riir/expert_forward.rs:66-100` —
  `ChainToNormed`. Phase 4 chain handoff.
- `crates/moeflux/src/riir/mla_attn_forward.rs:208+` — single-cmdbuf
  forward. Phase 4 splits into encode + wrapper.
- `crates/moeflux/src/riir/mod.rs:1289+` — `step_internal_mla_gpu`.
  Phase 4 inserts the ring loop here.
- `crates/moeflux/src/riir/state.rs:135` — `MlaKvCacheGpu::ensure_buffers`.
  Phase 3 modifies (drop memset / lazy-page).
- `crates/moeflux/src/riir/state_snapshot.rs:103-104, 210-213,
  391-394` — Phase 6 `MlaUnsupported` → real serialize/deserialize.
- `crates/moeflux/shaders/shaders.metal:1665+` — `mla_sdpa_folded`.
  Phase 5 sibling tiled kernel goes after this.
- `crates/moeflux/scripts/profile_smoke.sh` — capture profile.
- `crates/moeflux/scripts/profile_aggregate.py` — aggregate
  (use `--filter moeflux` to see only user code).
- `crates/moeflux/tests/fixtures/cogito_v2_oracle.sse.jsonl` —
  Phase 7 token-sequence diff target.

## Run commands (durable)

```bash
# Per-phase tests:
cargo test -p moeflux --no-default-features \
  --features model-cogito-v2-671b --release \
  --test dense_mlp_gpu --test cogito_moe_gpu --test cogito_v2_smoke \
  -- --ignored --nocapture

# CPU oracle fallback:
MOEFLUX_FORCE_CPU_MLA=1 cargo test ... --test cogito_v2_smoke ...

# Profile a warm token (Phase 0):
./scripts/profile_smoke.sh cogito_v2_smoke cogito_v2_eval_token_warm
python3 scripts/profile_aggregate.py /tmp/moeflux_profile.json --filter moeflux

# Open in browser instead:
./scripts/profile_smoke.sh --open cogito_v2_smoke cogito_v2_eval_token_warm

# blallama:
cd ~/Projects/drama_llama
cargo build --release --bin blallama --features "axum,cli,toml,moeflux-model-cogito-v2-671b"
./target/release/blallama "/Volumes/Temp Backup/models/blallama" \
  --backend moeflux --probe-stream --port 11435
# in another shell:
curl -sN http://localhost:11435/probe > run.sse.jsonl &
curl -X POST http://localhost:11435/v1/messages -H 'Content-Type: application/json' \
  -d '{"model":"cogito-v2-671b","messages":[{"role":"user","content":"Hello. How are you?"}],"max_tokens":16,"temperature":0.0}'
```

## Pointers (durable)

- This session's landing: `cogito_v2_full_gpu_partial_landed.md`
- Architecture audit: `cogito_v2_architecture.md`
- moeflux RIIR strategy: `riir_moeflux_strategy.md`
- Last session's plan-of-record: `cogito_v2_full_gpu_plan.md`
  (superseded by this file for next session)
