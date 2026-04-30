# Cogito-V2 — full-GPU forward (perf to ~1 tok/s)

Companion to `cogito_v2_gpu_mla_landed.md` (hybrid GPU MLA + CPU
MoE green this session). Drafted at session-end while context is
loaded — concrete enough that next-session-Claude can start coding
without re-exploring. Mike's cue: "Next session I'd like to move
all work to the GPU unless it's something that's actually faster
on CPU."

## Goal & success criteria

Drive Cogito-V2 671B fully on the Metal pipeline so the per-token
forward looks like Qwen's GQA path does today: GPU projections,
GPU SDPA, GPU cache append, GPU dense MLP, GPU MoE (router +
routed experts + shared expert). CPU only for embed lookup, host-
side residual additions, and the final norm; everything else on
GPU.

**Success = ≥1 tok/s warm on the "Hello. How are you?" 8-token
generation, same continuation as the current CPU oracle ("I'm
doing well, thanks! How").** Concrete bars:

1. blallama with Cogito-V2 + `--probe-stream` reports per-warm-token
   wall-clock ≥1 tok/s.
2. Generation matches the CPU oracle's continuation at T=0.
3. Activity Monitor shows GPU saturated, CPU mostly idle during
   warm tokens (Mike will eyeball this; we don't need to automate).
4. 200+ token gen stays coherent (open question from CPU landing,
   carried forward).

## Why hybrid is the bottleneck right now

This-session measurement: GPU first-token = 34s, CPU first-token =
20s. The 14s gap is mostly Metal pipeline JIT (4 new pipelines ×
~3.5s each) + 18 GB virtual KV cache lazy-commit — **one-time
costs**. The steady-state floor under hybrid is "GPU MLA fast +
CPU MoE slow" because the MoE path
(`moe_cpu::deepseek_moe_cpu`, ~340 GB of expert blobs) is the
bandwidth hog and rayon-CPU streaming maxes out at 8 P-cores. Qwen
hit 2 tok/s only after the deferred-ring + parity-prefetch
pipeline overlap landed — that's the perf machinery to lean on.

## Architectural decisions already made

1. **Reuse existing GPU MoE kernels**
   (`gpu_batched_experts_forward`, `moe_combine_residual`,
   `swiglu_fused`). They're parameterized on dim — confirmed in
   plan-mode exploration this session. Cogito-V2's
   `moe_intermediate_size = 2048` matches Qwen's per-expert
   width, so the existing kernel runs at the same shape.
2. **Replace `deepseek_moe_cpu` with a GPU dispatcher**, not
   port the routing logic to a Metal kernel. noaux_tc routing on
   256 experts is small CPU work; the bandwidth win is in
   running the routed experts on GPU + streaming experts via the
   existing async-pread machinery (`io_pool` + `prefetch`).
   Routing stays on CPU; expert dispatch moves to GPU.
3. **Variant-flag the shared-expert composition.**
   `SharedExpertGate::Unscaled` already exists on `variants.rs:115`
   and is what Cogito-V2 uses (`variants.rs:509`). The existing
   `moe_combine_residual` kernel
   (`shaders.metal:1333`) hardcodes the sigmoid gate path —
   `output = h_mid + Σ moe + sigmoid(shared_gate) * shared`.
   Either:
   - Sibling kernel `moe_combine_residual_unscaled` (drops the
     sigmoid and shared_gate read; cleaner semantics, ~10 lines
     of Metal duplicated)
   - Or pass a `gate_mode` constant into the existing kernel
     (one branch, smaller surface)
   Recommended: sibling. Avoids per-thread branch divergence; the
   duplicated lines are trivial. Land in `shaders.metal` after
   the existing kernel.
4. **Deferred-ring integration is the load-bearing perf slice.**
   The existing GQA path (`step_internal` lines 1290-1421) does:
   - Layer N's CMD1+CMD2 chained MoE while layer N-2's K-expert
     is still running on GPU (depth-2 ring)
   - Async expert prefetch for layer N+1 while N is mid-compute
   The MLA forward I landed this session is fully synchronous
   (3 sync points per layer). For ~1 tok/s we need MLA to play
   nice with the ring — every cmdbuf in `mla_attn_layer_forward
   _gpu` should chain into the next layer's K-expert dispatch
   without committing-and-waiting between them.
5. **Dense MLP layers 0-2 stay on the same code path as MoE
   layers.** They're 3 of 61, so a separate "dense MLP forward"
   helper that the per-layer dispatcher calls when
   `layer_idx < first_k_dense_replace`. Reuse
   `dequant_matvec_4bit_v3` + `swiglu_fused` (both
   dim-parametric). No new kernels.
6. **Snapshot v2 wire format is required before Council ships.**
   Prompt caching depends on it — without it, every Council
   request re-prefills the full prompt context. Currently both
   `Mla` arms in `state_snapshot.rs` return `MlaUnsupported`.
   This is parallelizable with the perf work (they don't touch
   the same code).

## Phases

Suggested order; phases 1-3 have no inter-dependencies and can
be tackled in any order. Phase 4 (deferred-ring integration) is
the load-bearing perf slice.

### Phase 1 — `dense_mlp_layer_forward_gpu` (smallest, do first)

Files: new `crates/moeflux/src/riir/dense_mlp_gpu.rs`,
`crates/moeflux/src/riir/mod.rs` (dispatch).

Math: `out = down_proj(silu(gate_proj(h)) * up_proj(h))` at
`intermediate_size = 18432`.

Sequence per layer:
1. `encode_matvec(gate_proj)` → scratch_gate
   (in_dim=hidden_dim, out_dim=18432, 4-bit)
2. `encode_matvec(up_proj)` → scratch_up (same shape)
3. `swiglu_fused(scratch_gate, scratch_up, scratch_act, dim=18432)`
4. `encode_matvec(down_proj)` → out
   (in_dim=18432, out_dim=hidden_dim, 4-bit)

Tensor names: `model.layers.{i}.mlp.gate_proj`,
`mlp.up_proj`, `mlp.down_proj`.

Dispatch from `step_internal_mla_gpu` when
`layer_idx < first_k_dense_replace` (= 3 for Cogito).

Validation: GPU dense_mlp output must match
`dense_mlp_swiglu_cpu` host output bit-for-bit modulo Metal
reduction order. Diff after layer 0 only — small fixture, fast
iteration.

### Phase 2 — GPU MoE dispatch with unconditional shared expert

Files: `crates/moeflux/shaders/shaders.metal` (new
`moe_combine_residual_unscaled`),
`crates/moeflux/src/riir/expert_forward.rs` (variant-aware
combine), new `crates/moeflux/src/riir/cogito_moe_gpu.rs` or
inline in `mla_attn_forward.rs`.

Sequence per MoE layer:
1. CPU: noaux_tc routing on `mlp.gate.weight` + `gate.biases`
   (existing `noaux_tc_router_cpu`) — produces top-K indices +
   weights.
2. GPU: stream top-K expert blobs (existing `prefetch` machinery
   + `io_pool`).
3. GPU: `gpu_batched_experts_forward(top_k, weights, blobs)` →
   per-expert SwiGLU in parallel; `moe_combine_residual_unscaled`
   sums weighted experts + adds shared-expert SwiGLU
   unconditionally + adds `h_mid` residual.
4. Shared-expert SwiGLU: existing GPU shared_expert path
   (`shared_gate_out`/`shared_up_out`/`shared_act`/`shared_out`
   buffers in `LayerForwardBuffers`). Just-needs the variant-
   flag check on the combine to skip the sigmoid gate.

Validation: GPU MoE output matches `deepseek_moe_cpu` host
output bit-for-bit modulo Metal reduction order. Diff after
layer 3 (first MoE layer).

### Phase 3 — `step_internal_mla_full_gpu` orchestrator

Files: `crates/moeflux/src/riir/mod.rs`.

Replaces the hybrid `step_internal_mla_gpu` from this session.
Per layer:
1. Pre-attn rms_norm (GPU `gpu_rms_norm_fused`).
2. `mla_attn_layer_forward_gpu` (existing).
3. residual_add via `encode_residual_add` (existing kernel).
4. Post-attn rms_norm (GPU).
5. If `layer_idx < first_k_dense_replace`: `dense_mlp_layer
   _forward_gpu` (Phase 1).
   Else: GPU MoE (Phase 2).
6. residual_add (GPU).

No CPU bounces between layers. Final norm + lm_head still GPU
(existing `lm_head_gpu`).

Compile-check: Cogito-V2 + both Qwen variants stay green.

### Phase 4 — Deferred-ring integration (the perf unlock)

Files: `crates/moeflux/src/riir/mla_attn_forward.rs`,
`crates/moeflux/src/riir/mod.rs`.

The current `mla_attn_layer_forward_gpu` has three commit+wait
sync points per layer (post Q+KV-A matvec, post-RoPE, post-SDPA-
chain). For ~1 tok/s we need these to chain into the per-layer
deferred-experts ring like the GQA path does. Concretely:

- Replace explicit cmdbuf.commit/wait with cmdbuf chaining: the
  q_pe extraction + kv_pre split + q_nope packing become Metal
  kernels (small custom kernels or `set_buffer(.., offset)`
  views) so we don't host-bounce between dispatches.
- Hand off the post-`o_proj` output buffer to the ring's
  `data_prefetch` machinery instead of host-readback.
- The MLA path becomes a peer of `full_attn_layer_forward` —
  matches its signature (deferred ring, prefetch, chained MoE).

This is genuinely substantial work — touching `deferred.rs`,
`prefetch.rs`, and the ring-cleanup logic in `step_internal`.
Likely the bulk of next session.

### Phase 5 — Multi-token validation + perf measurement

Files: `crates/moeflux/tests/cogito_v2_smoke.rs` (extend),
blallama path.

- Extend smoke to do an 8-token greedy gen and compare to a
  fixture (capture from CPU path once).
- blallama run with cogito-v2 + `--probe-stream`,
  `max_tokens = 128`. Capture the SSE jsonl. Inspect `ts_ms`
  deltas for warm-token wall-clock.
- Mike eyeballs Activity Monitor. Target: GPU saturated, CPU
  mostly idle during warm decode.

If perf ≥1 tok/s and gen matches CPU oracle → declare done. If
not, `samply record` capture + flame-graph → identify bottleneck
(likely a remaining sync or CPU-side hot path).

### Phase 6 — Snapshot v2 wire format (parallelizable)

Files: `crates/moeflux/src/riir/state_snapshot.rs`.

MLA cache shape: `[max_seq, kv_lora_rank + qk_rope_head_dim]` per
layer. Wire format additions:
- New magic / version = v2 (current is v1).
- For `LayerState::Mla`: serialize `(len, latent_cache[..len],
  rope_k_cache[..len])`. `latent_cache` and `rope_k_cache` are
  shared-storage Metal buffers; `contents()` gives a host-readable
  byte view (caller invariant: no GPU work in flight, same as
  existing snapshot path).
- v2 reader: must accept v1 (no MLA layers) for backward
  compatibility with Qwen snapshots. v1 reader: error on v2.

Test: round-trip a 200-token snapshot through save/load, verify
post-load `eval_token` produces identical logits to pre-save.

Drama_llama side: `Session::checkpoint_pos` already calls
`state_save` on RsCtx. No drama_llama changes needed once moeflux
v2 is plumbed.

## Critical files (existing — to read first)

- `crates/moeflux/src/riir/mla_attn_forward.rs` — this session's
  hybrid forward. Phase 4 transforms this into a deferred-ring
  citizen.
- `crates/moeflux/src/riir/full_attn_forward.rs` — contract
  template for the deferred-ring integration. The signature
  shape `(metal, wf, wf_buf, layer_cache, linear_buffers,
  moe_buffers, deferred, layer_idx, pos, k_active, experts,
  io_pool, prefetch, prefetch_set, kv_state, gpu_combine,
  prev_layer_chained, chain_next_norm_off)` is what
  `mla_attn_layer_forward_gpu` should look like post-Phase-4.
- `crates/moeflux/src/riir/expert_forward.rs:491+` —
  `gpu_batched_experts_forward` is the entry point Phase 2 calls.
- `crates/moeflux/src/riir/expert_forward.rs:692+` —
  `moe_combine_residual` dispatcher; sibling
  `_unscaled` lands here in Phase 2.
- `crates/moeflux/src/riir/moe_router.rs:193` —
  `noaux_tc_router_cpu` stays unchanged; CPU routing.
- `crates/moeflux/src/riir/mlp_cpu.rs:47` — `dense_mlp_swiglu
  _cpu` is the diff oracle for Phase 1.
- `crates/moeflux/src/riir/moe_cpu.rs:65` — `deepseek_moe_cpu`
  is the diff oracle for Phase 2.
- `crates/moeflux/src/riir/state_snapshot.rs:103-104,210-213,
  391-394` — `MlaUnsupported` stubs to replace in Phase 6.
- `crates/moeflux/shaders/shaders.metal:1333` — existing
  `moe_combine_residual` kernel to clone for `_unscaled`.

## Validation strategy (no oracle required)

The CPU oracle (`MOEFLUX_FORCE_CPU_MLA=1` + this-session-confirmed
hybrid path) is the diff target for every phase. Bit-exact
matching is achievable per Phase 4 of this session — the GPU path
produces identical logits to CPU on BOS-at-pos-0. Maintain that
invariant through every phase:

- Phase 1: top-K logits at decode step 0 of layer 3 (first MoE
  layer reached) must match CPU oracle.
- Phase 2: same, after layer 3's MoE GPU.
- Phase 3: full single-token logits match CPU oracle bit-for-bit.
- Phase 4: same — perf-only change, math unchanged.
- Phase 5: 8-token greedy continuation matches "I'm doing well,
  thanks! How".
- Phase 6: snapshot save/load round-trip → identical logits.

## Open questions for Mike (decide in-session)

1. **MoE routing on CPU or GPU?** The plan assumes CPU (small,
   simple). Going GPU saves the readback but adds a kernel. CPU
   is unlikely the bottleneck — 256 sigmoids + group-top-K is
   <100µs on M-series. Recommend CPU.
2. **Phase 6 in-session or follow-up?** Snapshot v2 isn't perf
   work; it's blocking the Council deployment. If Phase 4
   (deferred ring) eats most of the session, defer v2 to
   session-after-next. Or land v2 in parallel between phases.
3. **Long-context tiling for `mla_sdpa_folded`?** Currently
   `MLA_MAX_CACHE_TG = 4096`. Cogito-V2 supports 128k positions.
   Tiling is needed for long contexts. Probably session-3
   material — first verify ~1 tok/s at 4k context, then extend.

## Estimated session shape

- Phase 1: 30 min (mechanical, dim-parametric kernels)
- Phase 2: 1-2 hours (variant tail + dispatch wiring)
- Phase 3: 30 min (orchestrator stitching)
- Phase 4: 2+ hours (deferred-ring is the hard part)
- Phase 5: 30 min (run + measure, assuming things work)
- Phase 6: 1-2 hours (parallelizable, can spill into session-after)

Realistic outcome for one session: Phases 1-3 land cleanly + Phase
4 substantial-but-incomplete. Multi-token gen + perf measurement
spill into session-3 unless Phase 4 lands fast.

## Pointers (durable)

- This session's landing: `cogito_v2_gpu_mla_landed.md`
- Original GPU MLA plan: `cogito_v2_gpu_mla_plan.md` (Phases
  1-4 of which landed this session)
- Architecture audit: `cogito_v2_architecture.md`
- Reference impl (canonical math):
  `/Volumes/HF Models/models/hf/mlx-community/cogito-v2-preview-deepseek-671B-MoE-4bit/modeling_deepseek.py`
- moeflux RIIR strategy: `riir_moeflux_strategy.md`
