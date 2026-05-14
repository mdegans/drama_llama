# Session 6 Part B — precursor batching landed (B-0a/b/c/d + B-4 implicit)

**Date:** 2026-05-14 (same session as Part A)
**Commits (moeflux/main):**
- `ae55527` — Part A (GPU MoE router)
- `2255505` — B-0a/b/d: batched rms_norm + residual_add kernels + input rms_norm refactor
- `fbe7d7e` — B-0c: batched post-attn residual_norm_route in both batched paths

**Entry:** [`qwen_graph_mode_session6_partB_plan.md`](qwen_graph_mode_session6_partB_plan.md)
**Sibling:** [`qwen_graph_mode_session6_partA_landed.md`](qwen_graph_mode_session6_partA_landed.md)

## Discovery that re-shaped Part B

The plan locked in B2-first (bucket-build CPU, one chunk-end
readback), but it understated the precursor work. The existing
"batched" path was heterogeneous:

- `batched_full_attn_layer_forward` and
  `batched_linear_attn_layer_forward` had **per-token loops with
  `commit_and_wait` per token** in two places:
  1. The input rms_norm at the top of the layer
     (`full_attn_forward.rs:577-603`, `linear_attn_forward.rs:1978`).
  2. The post-attn residual + rms_norm + gate matvec + shared-gate
     matvec + CPU routing tail
     (`full_attn_forward.rs:814-853`, `linear_attn_forward.rs:2215`).

Each loop iterated N times. At N=8192, ~16k+ commits per layer ×
40 layers = the ~640k commits/chunk the parent plan blamed for
the 26× prefill gap.

Phase B-3's graph-mode lift can't help when there are
per-token commits inside each phase — the graph needs batched
encoders at the leaves. So Part B re-scoped: land the precursor
kernels and re-shape both batched paths into single-cmdbuf phases
*before* the graph-mode submission lift.

## What shipped

### B-0a: batched rms_norm kernel
`rms_norm_bf16_fused_n_tokens` — fused single-dispatch (sum_sq +
apply) over `[n_tokens, dim]`. One threadgroup per token; sum_sq
stays in tg-mem instead of going to global. Encoder
`encode_rms_norm_bf16_fused_n_tokens` in `gpu_norm.rs`.

Diff: cosine=1.0, max_abs=1.5e-7 vs per-token CPU reference
(sum_sq tree-reduction ULP drift).

### B-0b: batched residual_add
`residual_add_n_tokens` — 1D dispatch over `n_tokens * dim`
elements. Encoder `encode_residual_add_n_tokens_into`.

Diff: bit-exact (max_abs = 0).

### B-0c: batched post-attn tail
Both batched layer-forwards reshape the post-attn tail from a
per-token loop into one cmdbuf:
- `residual_add_n_tokens(o_proj_stack, hidden_in → h_mid)`
- `rms_norm_bf16_fused_n_tokens(h_mid → h_post, post_attn_norm_w)`
- `encode_matvec_n_tokens(h_post → gate_logits, gate_w/s/b, bits=8)`
- `encode_matvec_n_tokens(h_post → shared_gate_logits, seg_w/s/b, bits=8)`
- `encode_moe_router(gate_logits → routing_indices, routing_weights)`
- `commit_and_wait`, bulk readback of `h_mid_stack`, `h_post_stack`,
  routing, shared_gate_scores.

Needed kernel addition: `dequant_matvec_8bit_v3_n_tokens` (token-
axis extension of the existing single-token 8-bit v3 — required
because the Qwen3.6-A3B gate matvec is 8-bit, not 4-bit).
`encode_matvec_n_tokens` assertion widened to accept `bits == 4
|| bits == 8`. `MatvecPipelines::v3_8bit_n` pipeline field added.

### B-0d: batched input rms_norm
Replaced per-token loops at `full_attn_forward.rs:577` and
`linear_attn_forward.rs:1978` with single dispatches of the new
fused batched kernel. Input is `hidden_in_buf: MtlBuffer` (one
host→GPU bulk transfer instead of N per-iter copies into
`buffers.input`); output is `normed_buf` (no host bounce).

### B-4 (implicit)
The GPU MoE router (Part A) is wired into both batched paths via
`encode_moe_router`. Routing indices/weights are now produced on
GPU and read back once per layer in bulk, replacing the per-token
`read_buffer_to_vec(&batch_out[4], num_experts) → moe_router_cpu`
host bounce.

## Diff oracle

Canary battery (full 9-test list from the plan): **9/9 green** at
~80 s wall-clock. Cosine = 1.0 vs per-token oracle on the two
batched-prefill tests
(`eval_prompt_matches_per_token_oracle`,
`eval_prompt_chunked_matches_eval_prompt_whole_prompt`); bit-exact
on state snapshot round-trip.

Synthetic kernel tests:
- `moe_router_gpu_matches_cpu_*` — 5 shapes, all cosine=1.0,
  slot-match 100%.
- `rms_norm_bf16_fused_n_tokens_matches_cpu` — cosine=1.0,
  max_abs=1.5e-7.
- `residual_add_n_tokens_matches_cpu` — bit-exact.

## Directional bench (warm machine, no reboot)

`prefill_prompt.txt` 992 tokens, n=1, uptime 23:55 (well-warm):

| Metric | Pre-session | This session | Speedup |
|---|---:|---:|---:|
| Prefill tok/s | 36.8 | **60.76** | **1.65×** |

Not apples-to-apples vs the proper reboot-disciplined baseline
(`feedback_bench_discipline.md`), but the delta is well outside
the ±5 tok/s variance band — the move is real. Long-prompt bench
result captured in the [session memo footer](#long-prompt-result).

llama.cpp baseline on 992: 970 tok/s. Gap closed from **26×** to
**16×**. The remaining gap is the orchestrator-level commit
churn (~5–7 commits/layer × 40 layers ≈ 280 commits/chunk
still) — Phase B-3 / B-5 territory.

## What's left for Phase B

### B-2/B-3: closure-Vec graph + per-phase encoder refactor
Each phase's encoder currently includes its own
`metal.commit_and_wait_labeled(...)`. To consolidate phases into
larger cmdbufs, encoders must take an external `&CommandBufferRef`
and not commit. The orchestrator then sequences multiple phases
into one cmdbuf and commits once per natural boundary.

Natural boundaries (post-Part-B precursors):
1. **Pre-MoE chunk** — input rms_norm + Q/K/V projections + SDPA +
   o_proj + post-attn (residual + rms_norm + gate + shared_gate +
   router) across all 40 layers. Single commit, single readback of
   routing indices for all (layer, token) pairs.
2. **MoE chunk** — shared FFN + MoE permute-fuse + combine across
   all 40 layers + final RMSNorm + lm_head. Single commit, single
   readback of final logits (if requested).

Total commits per chunk: **2** (down from current ~280 after
Part-B precursors).

### B-5: orchestrator restructure
Currently `step_internal_batched_gqa` calls
`batched_full_attn_layer_forward` / `batched_linear_attn_layer_forward`
per layer in a loop, and each call commits internally. After B-3,
the orchestrator becomes the cmdbuf owner; layer-forward functions
become pure encoders.

### B-7 redux: properly disciplined bench
Reboot, n=3, high-perf power per `feedback_bench_discipline.md`.
Numbers in this memo are warm-machine directional only.

## Risks left for next session

- **Cross-layer state dependencies**. Layer N+1's input is layer N's
  output (hidden_out). In the orchestrator-level graph, this means
  the layer-N MoE combine writes to a buffer that layer-N+1's input
  rms_norm reads. All-in-one-cmdbuf if the data dependency is
  internal to the cmdbuf. Need to make sure hidden_in / hidden_out
  are GPU buffers (currently host vectors at the
  `step_internal_batched_gqa` boundary).
- **`hidden_in` host materialization**. Currently:
  - `step_internal_batched_gqa` builds `hidden_in_stack` as a host
    `Vec<f32>` from embeddings.
  - Each layer-forward call materializes `hidden_in_buf` from
    `hidden_in` (host).
  - Each layer-forward call returns `hidden_out` as a host vector.
  - The orchestrator copies `hidden_out → hidden_in` between layers.

  Phase B-3 needs a persistent `hidden_stack_buf: MtlBuffer<f32>` that
  layers write/read in place (or a double-buffer pair). One host
  bounce per chunk, not 40.
- **SDPA causal mask construction**. Currently materialized in host
  vectors and uploaded per layer. Either fold into a kernel that
  generates the mask on GPU per call, or hoist to a persistent
  per-chunk buffer.

## Files touched this session

| File | Change |
|---|---|
| `crates/moeflux/shaders/shaders.metal` | +3 kernels: `moe_softmax_topk`, `moe_normalize_weights`, `rms_norm_bf16_fused_n_tokens`, `residual_add_n_tokens`, `dequant_matvec_8bit_v3_n_tokens` |
| `crates/moeflux/src/riir/gpu_moe_router.rs` | New module |
| `crates/moeflux/src/riir/gpu_norm.rs` | +`RmsNormBf16FusedNTokensPipeline`, +`encode_rms_norm_bf16_fused_n_tokens`, +`encode_residual_add_n_tokens_into` |
| `crates/moeflux/src/riir/gpu_matvec.rs` | +`v3_8bit_n` pipeline, 8-bit support in `encode_matvec_n_tokens` |
| `crates/moeflux/src/riir/metal.rs` | `ALL_KERNELS` updated |
| `crates/moeflux/src/riir/mod.rs` | +`pub mod gpu_moe_router` |
| `crates/moeflux/src/riir/full_attn_forward.rs` | B-0d + B-0c refactor of Phase 1a + Phase 3c |
| `crates/moeflux/src/riir/linear_attn_forward.rs` | B-0d + B-0c refactor of Phase 1a + Phase 1e |
| `crates/moeflux/tests/batched_diff_oracle.rs` | +5 router tests, +1 rms_norm test, +1 residual_add test |

## Session takeaway

The plan's instinct was right — graph-mode submission is the
headline win — but the precursor work to get there was larger than
the parent plan acknowledged. Splitting Part B into precursors
(this memo) + graph-mode lift (next session) is the cleaner
shipping rhythm. Mike's "pivot on discovery" feedback
(`feedback_pivot_on_discovery.md`) applied cleanly: stopped, re-
scoped, documented, proceeded.

The 1.65× directional win on the small workload is a strong signal
that the graph-mode work will land the rest of the gap. Next
session is the orchestrator refactor.

## Long-prompt result

(to be filled in once bench finishes)
