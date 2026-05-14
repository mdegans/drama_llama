# Session 6 Part A landed — GPU MoE router

**Date:** 2026-05-14
**Commit:** `ae55527` on `moeflux/main` ("graph-mode: session 6 phase
A — GPU MoE router")
**Entry point:** [`qwen_graph_mode_session6_plan.md`](qwen_graph_mode_session6_plan.md)
**Next:** [`qwen_graph_mode_session6_partB_plan.md`](qwen_graph_mode_session6_partB_plan.md)

## What shipped

Two new Metal kernels (`moe_softmax_topk`, `moe_normalize_weights`)
and `gpu_moe_router::encode_moe_router` — a Rust encoder that
emits the full softmax → selection-sort top-K → divide-by-sum
pipeline into a borrowed `CommandBufferRef` with no
`commit_and_wait`.

**No callsite swapped.** Phase A is the *enabler*: the existing
`post_attention_residual_norm_route` still reads `batch_out[4]`
to host and runs `moe_router_cpu`. The win comes in Phase B when
routing stays in a GPU buffer and the per-layer commit boundary
goes away.

## Diff results

`batched_diff_oracle` battery:

| Shape | N | E | K | slot-match | weight cosine | max_abs_w |
|---|---:|---:|---:|---:|---:|---:|
| a3b decode | 1 | 256 | 8 | 1/1 | 1.000000000 | 1.49e-8 |
| a3b sub-chunk | 8 | 256 | 8 | 8/8 | 1.000000000 | 2.24e-8 |
| a3b mid-batch | 256 | 256 | 8 | 256/256 | 1.000000000 | 4.47e-8 |
| Qwen2-shape | 64 | 128 | 8 | 64/64 | 1.000000000 | 4.47e-8 |
| A17B-shape | 32 | 512 | 10 | 32/32 | 1.000000000 | 2.98e-8 |

**Slot-match 100% across all shapes** — running-min selection
order is bit-exact against the CPU oracle's `cpu_topk`. The plan
worried this might require set-equality (ULP-close logits could
swap ranks); on uniform `(-2, 2)` random inputs the magnitude
separation between adjacent expert scores dominates. The test
still asserts set equality first and reports slot match as a
diagnostic, so a future regression to slot-order would surface
but not fail the test.

Canary battery (full session-6-plan list): 9/9 pass at 76 s.

## Kernel notes

- `moe_softmax_topk`: per-token threadgroup, 64 threads. Parallel
  max/sum reductions in tg-mem (`probs[MAX_EXPERTS=512]`), then
  lane-0 runs the serial running-min top-K into per-thread
  `(int sel_idx[16], float sel_val[16])` registers and writes
  once at the end. K and num_experts are runtime constants.
- `moe_normalize_weights`: per-token threadgroup, K threads.
  Lane 0 sums and broadcasts `1/sum` via tg-mem; lane `lid`
  multiplies slot `lid`. Guard mirrors the CPU: if `sum ≤ 0`,
  pass `1.0` so the multiply is a no-op (matches
  `cpu_normalize_weights` skipping the divide).
- Caps: `MAX_EXPERTS=512`, `MAX_K=16`. Covers Qwen3-A3B (256/8),
  Cogito-V2 / DeepSeek-V3 (256/8), Qwen3.5-A17B (512/10). 32 KB
  threadgroup memory cap is unbroken (probs = 2 KB).

## Architecture decisions that stick

- **GPU encoder lives in `gpu_moe_router.rs`**, not appended to
  `moe_router.rs`. Mirrors the CPU/GPU split established by
  `gpu_norm`, `gpu_matvec`, `gpu_rope` etc. CPU oracle stays in
  `moe_router.rs` as the diff target.
- **`MoeRouterPipelines::fetch` pattern** matches the existing
  `RmsNormBf16Pipelines`, `MatvecPipelines`, etc. Caller scopes
  it once.
- **No commit_and_wait inside the encoder.** The whole point of
  this work is graph-mode submission; lifting commit boundaries
  is Phase B's job.

## What this unblocks

The post-attention path's CPU readback shape today is:

```rust
// linear_attn_forward.rs ~line 1175:
let mut scores =
    read_buffer_to_vec(&buffers.batch_out[4], v.num_experts);
let mut routing_indices = vec![0i32; k_active];
let mut routing_weights = vec![0f32; k_active];
moe_router_cpu(&mut scores, k_active, &mut routing_indices,
               &mut routing_weights)?;
```

Phase B's B2 variant (recommended in the plan) replaces this
with:

1. Dispatch `encode_moe_router` into a per-chunk routing buffer
   `(indices_buf[N, K], weights_buf[N, K])`.
2. *No host bounce*. The next layer (or the chunk's MoE pass)
   reads `indices_buf` either via another GPU kernel (B1) or via
   one host bounce at chunk end (B2).

The plumbing seam: `post_attention_residual_norm_route` needs to
grow a return shape that includes a *GPU buffer of routing
indices* in place of `routing_indices: Vec<i32>`. The caller
chain up through `step_internal_batched_gqa` then either consumes
the GPU buffer directly or reads it back at the chunk boundary.

## Risks left

- **K-tied logits**: at very small K or near-tied logits, slot
  order could diverge from CPU. We don't see this on uniform
  random inputs, but real model logits could in principle hit a
  tied case. If Phase B ever falls back to CPU bucket-building
  via host readback of routing indices, the build is robust to
  slot reordering (set of indices is what matters, not order
  within the slot). So this is at worst a diagnostic concern.
- **MAX_EXPERTS/MAX_K caps** in the kernel are static. A future
  variant exceeding 512 experts or K=16 would need a kernel-side
  bump (cheap — both are tg-mem / register sizing).
- **No norm_topk_prob variant**. llama.cpp's
  `build_moe_ffn` supports a softmax-over-K normalize instead of
  divide-by-sum. Our CPU oracle is divide-by-sum (matches `infer.m`),
  so GPU matches. If a future model needed softmax-renorm, that's
  a `_softmax_renorm` variant kernel.
