# Session 7 plan — graph-mode submission lift (real B-3 + B-5)

**Entry:** [`qwen_graph_mode_session6_partB_precursors_landed.md`](qwen_graph_mode_session6_partB_precursors_landed.md)
**Parent plan:** [`qwen_graph_mode_session6_plan.md`](qwen_graph_mode_session6_plan.md)

**Goal:** collapse the per-phase commit churn (~5–7 commits/layer
× 40 layers ≈ 280 commits/chunk after session 6's precursor
batching) down to **2 commits/chunk**, closing the remaining gap
to llama.cpp's ~1.02 s on the 992-prefill workload.

Session 6's precursor batching took the 26× gap to 16× directional;
this session aims for **≤3×** (target band 200-500 tok/s on 992).

## State entering session 7 (updated after S7-1a)

After session 6's S7-1a fusion, the intra-layer commit count is
already reduced:

**`batched_full_attn_layer_forward`**: 4 commits/layer.
1. Fused 1a+1b (input rms_norm + Q/K/V proj).
2. Phase 2 SDPA.
3. Fused 3b+3c (O proj + post-attn + router).
4. Fused 3d+3e (shared FFN + MoE permute-fuse).

The 3 boundaries between these are forced by CPU host-bounces:
q/k norm + RoPE + KV append, sigmoid_gate, bucket build.

**`batched_linear_attn_layer_forward`**: 2 commits/layer.
1. Fused 1a+1b+1c+1d+1e (entire pre-MoE chain — all GPU-deps).
2. Fused 1f+1g (shared FFN + MoE permute-fuse).

Total commits per chunk on Qwen3.6-A3B: 10 full-attn layers × 4 +
30 linear-attn layers × 2 = 100 commits. Down from session-5's
~640k, and from S6-pre-S7-1a's ~280.

Remaining wins from S7 phases:
- **Cross-layer hidden_in/hidden_out hoist**: today's chunk
  orchestrator passes hidden via host vectors between layers.
  Hoisting to GPU buffers would let two consecutive layers
  share the same cmdbuf, halving the remaining commits.
- **GPU q/k norm + RoPE + KV append**: the full-attn host bounce
  in Phase 1b→2. New Metal kernels + KV cache append kernel.
  Would let full_attn match linear_attn's 2 commits/layer.
- **GPU sigmoid_gate**: trivial element-wise kernel; eliminates
  the Phase 2→1d host bounce.
- **GPU bucket build (B1 from parent plan)**: eliminates the
  3c→3d host bounce. Most ambitious.

llama.cpp does ~1–2 commits per chunk via dispatch_apply across
`n_cb` worker threads.

## The structural commit boundary

There is exactly *one* host-side data dependency inside each
layer's forward: **routing indices are read back from GPU after
the post-attn gate matvec, then CPU `build_expert_buckets` runs
to convert them into the bucket-CSR shape the MoE permute-fuse
kernel needs.**

Everything else inside a layer's forward chain has GPU-only data
dependencies (output of phase N becomes input of phase N+1, all in
device memory).

So the natural per-layer shape is:
1. **Cmdbuf A**: phases 1a + 1b + 1c + 1d + 3c (pre-MoE).
2. **Commit-and-wait** → readback routing indices/weights.
3. CPU: `build_expert_buckets` (sub-millisecond at N=8192 × K=8).
4. **Cmdbuf B**: phases 3d + 3e (shared FFN + MoE).
5. **Commit-and-wait** → hidden_out is now in a GPU buffer.

If we go further and chain cmdbuf B of layer N with cmdbuf A of
layer N+1 — they share no host dependency — we get:
- Single cmdbuf containing 40 × (pre-MoE) + 40 × (MoE) interleaved
  with N CPU bucket-builds between.

But that's hard to schedule without a producer/consumer event
mechanism. **B2-first**: collect all 40 layers' routing into one
buffer, do one big chunk-end readback, then encode the MoE work
for all 40 layers into cmdbuf B.

```
Cmdbuf A (per chunk):
  for layer in 0..40:
    Phase 1a / 1b / 1c / 1d / 3c → write routing_indices_per_layer[layer]
  commit_and_wait

CPU:
  for layer in 0..40:
    buckets[layer] = build_expert_buckets(routing_indices_per_layer[layer], ...)

Cmdbuf B (per chunk):
  for layer in 0..40:
    Phase 3d / 3e using buckets[layer]
  final RMS norm + lm_head
  commit_and_wait
```

**Total commits per chunk: 2** (down from ~280). This is the
session-7 target shape.

## Phases (S7-1a landed in session 6)

### S7-1a: intra-layer fusion (DONE)
Shipped in `6628eaf` end of session 6 — both layer-forwards
collapsed to their minimal commit counts given the existing CPU
host-bounces (full_attn 4/layer; linear_attn 2/layer).

### S7-1b (NEW): GPU q/k norm + RoPE + KV append
Move the per-token Phase 1b→2 CPU work to GPU. Existing kernels:
`rms_norm_qk` (already used by linear-attn), `apply_rotary_emb`
needs a Metal version. KV cache append also needs a kernel.

Diff target: per-token CPU outputs (`q_host`, `k_host` post-norm
+ RoPE, KV cache rows). Cosine 0.9999 vs CPU oracle.

After S7-1b: full_attn can fuse Phase 1b + Phase 2, eliminating
one more commit/layer. Estimated +20-30% on full_attn-heavy
workloads (which is rare — only 10/40 layers are full_attn).

### S7-1c (NEW): GPU sigmoid_gate
Element-wise: `out[t,i] = (1 / (1 + exp(-q_gate[t,i]))) * attn[t,i]`.
Trivial kernel. Eliminates the Phase 2→3b host bounce in full_attn.

After S7-1c: full_attn becomes 2 commits/layer like linear_attn.

### S7-1: Convert per-phase encoders to no-commit encoders

Each `encode_X_into(cmdbuf, ...)` already takes a `cmdbuf`. The
current pattern wraps each in:

```rust
{
    let queue = metal.queue_clone();
    let cmdbuf = queue.new_command_buffer();
    encode_X_into(cmdbuf, ...);
    metal.commit_and_wait_labeled(cmdbuf, "label");
}
```

The fix: hoist the commit-and-wait to the caller. Each phase
function takes a `cmdbuf: &CommandBufferRef` parameter instead of
allocating one internally.

Phases to convert (in both `full_attn_forward.rs` and
`linear_attn_forward.rs`):
- Phase 1a (already an `encode_*_into` after S6 — just delete the
  inline commit).
- Phase 1b (Q/K/V projections).
- Phase 1c (SDPA / linear-attn kernels).
- Phase 1d (o_proj).
- Phase 3c (post-attn — already a single cmdbuf after S6,
  just delete the inline commit).
- Phase 3d (shared FFN).
- Phase 3e (MoE permute-fuse + combine).

### S7-2: Hidden-state persistence

Currently `hidden_in: &[f32]` and `hidden_out: &mut [f32]` are
host slices at the layer-forward boundary. The orchestrator does:

```rust
let mut hidden_in_stack = vec![0.0; n * hidden_dim];   // embeddings
let mut hidden_out_stack = vec![0.0; n * hidden_dim];
for layer in 0..40 {
    batched_X_layer_forward(..., &hidden_in_stack, &mut hidden_out_stack);
    std::mem::swap(&mut hidden_in_stack, &mut hidden_out_stack);
}
```

Each layer copies host→GPU at start and GPU→host at end. For
graph-mode, hoist to a double-buffered GPU pair:

```rust
let mut hidden_a = MtlBuffer::with_data(&device, &embeddings_host);
let mut hidden_b = MtlBuffer::with_len(&device, n * hidden_dim);
for layer in 0..40 {
    batched_X_layer_forward(..., cmdbuf_A, &hidden_a, &hidden_b);
    std::mem::swap(&mut hidden_a, &mut hidden_b);
}
```

One host→GPU at chunk start; one GPU→host at chunk end (for
logits computation if `logits_out` is set, otherwise zero).

### S7-3: Orchestrator restructure

`step_internal_batched_gqa` becomes the cmdbuf owner:

```rust
fn step_internal_batched_gqa(...) {
    // Allocate per-chunk persistent buffers (hidden double-buffer,
    // routing stack [40, N, K], gate logits stack, etc.).

    // Pass 1: pre-MoE for all layers.
    let cmdbuf_a = queue.new_command_buffer();
    for layer in 0..40 {
        encode_pre_moe_into(cmdbuf_a, ..., layer, routing_buf[layer]);
    }
    cmdbuf_a.commit();
    cmdbuf_a.wait_until_completed();

    // CPU: bucket-build for all layers.
    let buckets_per_layer: Vec<ExpertBuckets> = (0..40).map(|ℓ| {
        let idx = read_buf_to_vec_i32(&routing_buf[ℓ].indices, n * k);
        let w   = read_buf_to_vec(&routing_buf[ℓ].weights, n * k);
        build_expert_buckets(&idx, &w, n, k, num_experts)
    }).collect();

    // Pass 2: MoE for all layers + final norm + lm_head.
    let cmdbuf_b = queue.new_command_buffer();
    for layer in 0..40 {
        encode_moe_into(cmdbuf_b, ..., layer, &buckets_per_layer[layer]);
    }
    encode_final_norm_into(cmdbuf_b, ...);
    encode_lm_head_into(cmdbuf_b, ...);
    cmdbuf_b.commit();
    cmdbuf_b.wait_until_completed();
}
```

### S7-4: SDPA + linear-attn kernel split

If `Phase 1c` issues multiple tiled cmdbufs internally
(`mla_sdpa_tile_*` for MLA, or batched SDPA tile-finalize), those
also need to be no-commit encoders that take the outer cmdbuf.
Check by grep for `commit_and_wait` inside the SDPA / linear-attn
modules.

### S7-5: Canary battery after each of S7-1, S7-3

Cosine floor 0.9999 vs per-token oracle. Run the 9-test list
from the parent plan after every commit.

### S7-6: Bench post-reboot

Reboot, n=3, high-perf per `feedback_bench_discipline.md`:

```bash
cd ~/Projects/drama_llama
./bench.py --model a3b --prompt-file prefill_prompt.txt --max-tokens 1 -n 3
./bench.py --model a3b --prompt-file prefill_prompt_long.txt --max-tokens 1 -n 3
./bench.py --model a3b --max-tokens 128 -n 3
```

Target: prefill 200-500 tok/s on 992 (vs llama.cpp 970).

## Concrete file pointers

| What | Where |
|---|---|
| `batched_full_attn_layer_forward` | `crates/moeflux/src/riir/full_attn_forward.rs:505` |
| `batched_linear_attn_layer_forward` | `crates/moeflux/src/riir/linear_attn_forward.rs:1883` |
| Orchestrator `step_internal_batched_gqa` | `crates/moeflux/src/riir/mod.rs:1422` |
| Phase 1a (batched input rms_norm) | both files Phase 1a — first commit_and_wait after the new fused dispatch |
| Phase 3c (batched post-attn) | both files Phase 3c — single commit_and_wait |
| Internal commits to find | grep `commit_and_wait_labeled` |

## Concrete order of operations

1. **S7-1** — refactor each phase's commit to live in the caller.
   This is mechanical but touches 6-8 sites per layer-forward.
   Single commit, canary green, bench.
2. **S7-2** — hoist hidden_in/hidden_out to GPU buffers at
   orchestrator level. Canary green, bench.
3. **S7-3** — split per-layer call into pre-MoE / MoE passes,
   collect routing across all layers, do one chunk-end readback.
   Canary green, bench.
4. **S7-5** — run full canary battery + post-reboot bench.
5. **S7-6** — write landed memo with measured numbers.

## Risks

- **Cmdbuf size limits**. Metal command buffers can hold many
  encoded dispatches but not unlimited. 40 layers × ~7 phases =
  ~280 encoded dispatches in cmdbuf A. Should fit (llama.cpp
  encodes much more), but watch for "command buffer too large"
  errors.
- **GPU buffer alignment**. The persistent routing stack
  `[num_layers, N, K] i32` and `[num_layers, N, K] f32` are
  ~10 MB each at N=8192 / K=8 / 40 layers / 4 B. Memory not a
  concern; alignment shouldn't be either since they're flat f32/i32.
- **`step_internal_per_token_oracle` cross-talk**. The oracle still
  exists for canary tests. Its per-layer state cleanup happens via
  `discard_deferred_experts_in(deferred)` at the top of
  `step_internal_batched_gqa`. The pre-MoE-only cmdbuf A still
  needs that drain at chunk start.
- **MLA / cogito-v2 path**. MLA falls back to the per-token
  oracle inside `step_internal`. Graph-mode work in session 7 is
  for the GQA path (Qwen3.6-A3B); MLA stays on tokenwise until
  DeepSeek-V3 becomes the dominant consumer.

## Out of scope for session 7

- **Phase C** (parallel cmdbuf encoding via `dispatch_apply`). Save
  for session 8.
- **NPU enum-Op refactor**. The closure-Vec abstraction described
  in the parent plan is overkill for the session-7 work — the
  orchestrator can build cmdbufs directly without a typed graph.
  Closure-Vec is the *seam* that lets us refactor later; we can
  add it as a clean-up pass if it helps test isolation.
- **`eval_token` re-route through batched at N=1**. Phase D
  cleanup — graph-mode batched should outperform the oracle at
  N=1 after S7-3 lands, but verify with bench before deleting the
  oracle.
- **Snapshot v2 wire format bump**. Routing stack is per-chunk
  scratch, not persistent state.

## Notes from session 6 close

- Mike confirmed "1M context is a lot" — keep going on long
  sessions rather than warming up new ones for every small task.
  See [`feedback_dont_wrap_on_context_anxiety`](feedback_dont_wrap_on_context_anxiety.md).
- Session 6 shipped 4 commits + landed memo + directional bench
  in one sitting; session 7 has a similar shape if we stay
  disciplined on the orchestrator restructure.
