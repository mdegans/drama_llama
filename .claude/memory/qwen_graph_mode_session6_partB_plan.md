# Session 6 Part B plan — graph-mode submission

**Entry:** [`qwen_graph_mode_session6_partA_landed.md`](qwen_graph_mode_session6_partA_landed.md)
**Parent plan:** [`qwen_graph_mode_session6_plan.md`](qwen_graph_mode_session6_plan.md)
(Phase B section — refined with concrete file:line targets after Phase A)

**Goal:** collapse the ~640 k `commit_and_wait`s per chunk down to
**2**, closing most of the 26–32× prefill gap vs llama.cpp on Qwen3.6-A3B.

Phase B is the headline win and was budgeted at 2-3 sessions in the
parent plan. This memo locks in the **B2-first** path (CPU
bucket-build with one chunk-level readback) and lists the concrete
files / functions / buffer renames the refactor touches.

## Decision: B2 first

The parent plan offered B1 (GPU bucket-build via scatter scan) and
B2 (CPU bucket-build with one chunk-end readback). B2 first wins
because:

- The bucket-build is `O(N × K)` CPU work (8192 × 8 = 65 k token-slot
  pairs). At ~ns per iteration this is sub-millisecond.
- The cost of B2 over B1 is one readback per chunk, not one per
  layer. Compared to the today-state (one readback per layer ×
  40 layers ≈ 8× the data), it's a 99% reduction.
- B1's `moe_build_buckets` kernel (parallel scan + scatter) is a
  non-trivial GPU kernel. We'd only build it if the chunk-end
  readback shows up in profile.
- Reversibility: if profile later flags the readback, B1 is a
  *strictly additive* follow-up (the GPU router buffer is already
  the right input).

Decision lock-in: **Phase B implements B2.** Profile post-B before
deciding on B1.

## Per-chunk submission shape

```
┌─ cmdbuf #1: pre-MoE for all N tokens × 40 layers ──────────────┐
│   For each layer ℓ in 0..40:                                   │
│     - input rms_norm (batched, in tg-mem)                       │
│     - Q/K/V/O projections (batched matvec_n_tokens)             │
│     - linear-attn or full-attn forward (batched)                │
│     - post-attn residual_add + rms_norm                         │
│     - gate matvec → gate_logits[ℓ, N, num_experts]              │
│     - shared-gate matvec → shared_gate[ℓ, N]                    │
│     - shared FFN (gate_proj, up_proj, swiglu, down_proj)        │
│     - encode_moe_router → routing_idx[ℓ, N, K], weights[ℓ, N, K]│
└────────────────────────────────────────────────────────────────┘
  commit_and_wait
  CPU: read routing_idx[ℓ, *, *] for ℓ in 0..40, build buckets per layer
┌─ cmdbuf #2: MoE permute-fuse + combine for all N tokens × 40 ──┐
│   For each layer ℓ:                                             │
│     - encode_moe_batched_permute_fuse(bucket[ℓ])                │
│     - combine: hidden_out = h_mid + moe_sum + shared_out·gate   │
│   final RMSNorm + lm_head matvec                                │
└────────────────────────────────────────────────────────────────┘
  commit_and_wait (only if caller wants logits at chunk end)
```

Total commits per chunk: **2** (today's count: ~640 k).

## Closure-Vec graph abstraction

Per the parent plan, the submission shape is a `Vec<Box<dyn
FnOnce(&CommandBufferRef) + Send + 'a>>` accumulator. New module
`crates/moeflux/src/riir/graph.rs`:

```rust
pub struct Graph<'a> {
    nodes: Vec<(&'static str, Box<dyn FnOnce(&CommandBufferRef) + Send + 'a>)>,
}

impl<'a> Graph<'a> {
    pub fn push<F: FnOnce(&CommandBufferRef) + Send + 'a>(
        &mut self, label: &'static str, f: F,
    ) {
        self.nodes.push((label, Box::new(f)));
    }
    pub fn encode_into(self, cmdbuf: &CommandBufferRef) {
        for (_label, node) in self.nodes { node(cmdbuf); }
    }
    pub fn labels(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.nodes.iter().map(|(l, _)| *l)
    }
}
```

`label: &'static str` is the NPU-roadmap hook from the parent plan
([`qwen_npu_roadmap.md`](qwen_npu_roadmap.md)) — when CoreML lands
and we refactor closures → enum-Op variants, the labels become
debug strings on each variant.

## Phases inside B

### B-0: Plumbing for GPU routing buffer (precursor)

Today `PostAttnIntermediates` carries `routing_indices: Vec<i32>`
and `routing_weights: Vec<f32>`. Make this a tagged enum so the
batched orchestrator can hold *either* the host vectors (oracle
path) or GPU buffers:

```rust
pub(super) struct PostAttnIntermediates {
    pub routing: RoutingBacking,
    pub shared_gate_score: f32,
}

pub(super) enum RoutingBacking {
    Cpu { indices: Vec<i32>, weights: Vec<f32> },
    Gpu { indices_buf: Buffer, weights_buf: Buffer, n_tokens: usize, k: usize },
}
```

Per-token oracle keeps `Cpu`. Batched orchestrator on graph-mode
flips to `Gpu` and the caller decides when to read back.

**Files touched:** `linear_attn_forward.rs` (struct + the three
`moe_router_cpu` callsites at lines 1180, 1384, 1501),
`mod.rs` (`step_internal_batched_gqa` consumer of routing).

### B-1: Batched gate matvec already lands logits stacked

The batched path's gate matvec via `encode_matvec_n_tokens` already
writes `[N, num_experts]` to a stacked buffer. Sanity-check that
this is the layout `encode_moe_router` expects (row-major
`[token, expert]`). If not, add the transpose to the encode site.

**Files touched:** check `linear_attn_forward.rs::batched_linear_attn_layer_forward`
and `full_attn_forward.rs::batched_full_attn_layer_forward` for
the gate-matvec dispatch shape.

### B-2: Persistent per-chunk routing buffers

`LayerForwardBuffers` today re-allocates per call. Add persistent
fields for the chunk-level routing stack:

```rust
pub(super) struct ChunkRoutingBuffers {
    pub indices: Vec<Buffer>,   // num_layers entries, each [chunk_size, K] i32
    pub weights: Vec<Buffer>,   // num_layers entries, each [chunk_size, K] f32
    pub gate_logits: Vec<Buffer>, // num_layers entries, each [chunk_size, num_experts]
}
```

Allocate at `LayerForwardBuffers::new` (or a sibling
`ChunkRoutingBuffers::new`) sized for `CHUNK_SIZE × K × num_layers`.
Per a3b's 8192 × 8 × 40 × 4B = ~10 MB indices, ~10 MB weights, ~80 MB
gate_logits. Worth it.

This is the Phase 4-scratch-hoist precondition the parent plan
called out. Limit it to what graph-mode references; don't
preemptively hoist everything else.

### B-3: Refactor batched layer forward to graph builder

Convert `batched_linear_attn_layer_forward` and
`batched_full_attn_layer_forward` from "commit-and-wait inside"
to "push into Graph":

```rust
// Before:
encode_residual_add(cmdbuf, &resid_add, ...);
encode_rms_norm_pair(cmdbuf, ...);
metal.commit_and_wait_labeled(cmdbuf, "post_attn_residual_norm_route");

// After:
graph.push("post_attn.residual_add", move |cb| {
    encode_residual_add_into(cb, &resid_add, ...)
});
graph.push("post_attn.rms_norm", move |cb| {
    encode_rms_norm_pair_into(cb, ...)
});
// No commit. Orchestrator owns the cmdbuf boundary.
```

Existing `encode_*_into` functions in `gpu_norm.rs` etc. already
take `&CommandBufferRef` — those can be wrapped in closures
directly. Functions that today take `&mut MetalBackend` and call
`commit_and_wait_labeled` internally get rewritten to encode-only
helpers (the labeled commit shifts to the orchestrator).

### B-4: Swap router callsite

In `post_attention_residual_norm_route` (or its graph-mode twin),
push `encode_moe_router` after the gate matvec. Return a
`RoutingBacking::Gpu` referencing this layer's slots in
`ChunkRoutingBuffers`.

Drop the `read_buffer_to_vec(&batch_out[4], num_experts)` +
`moe_router_cpu` block. Keep the function shape; only the inside
changes.

### B-5: Orchestrator commit boundaries

In `step_internal_batched_gqa`, after building cmdbuf #1's graph
across all layers and committing it:

```rust
// Read back ALL layers' routing indices in one shot.
let mut all_buckets: Vec<ExpertBuckets> = Vec::with_capacity(num_layers);
for ℓ in 0..num_layers {
    let idx = read_buffer_to_vec_i32(&chunk_routing.indices[ℓ], n * k);
    let w   = read_buffer_to_vec   (&chunk_routing.weights[ℓ], n * k);
    all_buckets.push(build_expert_buckets(&idx, &w, n, k, num_experts));
}
// Build cmdbuf #2 from the bucket data.
```

`read_buffer_to_vec_i32` may need adding (we have the f32 version);
trivial — a 4-line generic helper.

### B-6: Verification gate

Run the canary battery **after each of B-3, B-4, B-5**. Cosine
floor of 0.9999 vs the per-token oracle. If we break before B-5,
the orchestrator still works because the oracle path is unchanged;
only graph-mode batched layers shift.

If a phase regresses cosine, **stop**. Most likely culprits:
- Ordering inside the closure-Vec graph (one closure reading a
  buffer the next still writes to). Metal guarantees encode order
  = execution order **within a cmdbuf**; the bug if it appears is
  that two closures landed in the *wrong* order.
- The routing buffer layout mismatch (gate matvec writes
  `[expert, token]` but router expects `[token, expert]`, or vice
  versa). Easy to confirm: per-token cosine drops to ~0 if so.

### B-7: Bench

`bench.py --model a3b --prompt-file prefill_prompt.txt --max-tokens 1 -n 3`,
post-reboot, high-perf power per
[`feedback_bench_discipline.md`](feedback_bench_discipline.md).

Target post-B (per parent plan): **200–500 tok/s prefill on 992**
(today: 36.8). Anywhere in that band is a session win; below 200
suggests bucket-build or readback is the new pole and B1 might be
needed; above 500 suggests we may also be close on the 16 k workload.

## What stays out of scope for Part B

- **Phase C (parallel cmdbuf encoding)**. Save for session 7 or 8.
- **MLA path** (cogito-v2 671B). Stays on the tokenwise oracle
  per `step_internal`. DeepSeek-V3 motivation revisit later.
- **Snapshot v2 wire format**. The new persistent routing buffers
  are transient (per chunk, not per session state) — no snapshot
  bump.
- **`eval_token` re-route through batched at N=1**. Phase D
  cleanup, after graph-mode lands.

## Concrete file pointers

| What | Where |
|---|---|
| `post_attention_residual_norm_route` callsite | `crates/moeflux/src/riir/linear_attn_forward.rs:1412` |
| Three CPU router callsites | `crates/moeflux/src/riir/linear_attn_forward.rs:1180,1384,1501` |
| Batched linear-attn forward | `crates/moeflux/src/riir/linear_attn_forward.rs:1883` |
| Batched full-attn forward | `crates/moeflux/src/riir/full_attn_forward.rs:505` |
| Batched orchestrator | `crates/moeflux/src/riir/mod.rs:1422` |
| Per-chunk buffer struct | `crates/moeflux/src/riir/mod.rs` (`LayerForwardBuffers`) |
| Graph abstraction (new) | `crates/moeflux/src/riir/graph.rs` |
| Phase A encoder | `crates/moeflux/src/riir/gpu_moe_router.rs` |
| Phase A kernels | `crates/moeflux/shaders/shaders.metal:moe_softmax_topk,moe_normalize_weights` |
| Diff tests (existing for Phase A) | `crates/moeflux/tests/batched_diff_oracle.rs` |
| Canary battery | `crates/moeflux/tests/diff_oracle.rs` |

## Verification commands

Per phase B-3, B-4, B-5:

```bash
cd ~/Projects/moeflux
cargo test --release -p moeflux --features model-qwen3-6-35b-a3b \
  --test diff_oracle -- --ignored --nocapture --test-threads=1 \
  state_round_trip_rust eval_token_matches_c_single_step \
  eval_prompt_matches_per_token_oracle slot_reuse_race_regression_rust \
  state_load_rust_from_c_save eval_prompt_matches_c_multi_token \
  eval_prompt_chunked_matches_eval_prompt_whole_prompt \
  prompt_cache_start_pos_nonzero_matches diag_b2_eval_prompt_chunk_1
```

Bench (post-reboot):

```bash
cd ~/Projects/drama_llama
./bench.py --model a3b --prompt-file prefill_prompt.txt    --max-tokens 1 -n 3
./bench.py --model a3b --prompt-file prefill_prompt_long.txt --max-tokens 1 -n 3
./bench.py --model a3b --max-tokens 128 -n 3
```

## Effort budget

Part A landed in well under a session (kernel + encoder + 5 tests +
canary verify + commit). Part B is meaningfully larger because of
the orchestrator refactor footprint — parent plan estimated 2-3
sessions. With Part A's plumbing seam already in place, plausible
that **B-0 through B-4 lands in one session** and **B-5 + B-7
(bench post-reboot) in the next.**

Mitigation if B drifts: bail out cleanly after B-4 with the
graph-mode infrastructure in place but the old CPU routing path
still in use. That's a clean checkpoint — orchestrator runs, just
without the win — and the next session resumes at B-5.
