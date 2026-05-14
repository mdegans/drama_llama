# Session 6 plan — graph-mode submission, close the llama.cpp gap

Entry-point: [`qwen_batched_prefill_session5_landed`] (to write at
session-5 close), profile data 2026-05-14 head-to-head.

**End-state goal:** prefill parity (within 2×) with llama.cpp on
Qwen3.6-35B-A3B. Decode parity (within 1.5×). Session 5 closed
N×k×commit_and_wait churn within a single batched call; session 6
attacks the layer-level and chunk-level commit churn that's left.

## Head-to-head measurement (2026-05-14 baseline)

| Workload | moeflux | llama.cpp | gap |
|---|---:|---:|---:|
| 992 prefill (n=3) | 36.8 tok/s | 970 tok/s | **26×** |
| 16k prefill (n=1) | 27.1 tok/s | 857 tok/s | **32×** |
| essay+128 mixed (n=3) | 8.23 tok/s | 22.78 tok/s | **2.77×** |

Same M2 Max, same prompt, both at 4-bit (moeflux internal 4-bit;
llama.cpp `Q4_K_S`). No 8k wall on moeflux's batched path —
chunking + tiled SDPA work clean past 8k.

## Architectural diagnosis

Looking at llama.cpp's recent Metal scheduler
(`~/Projects/llama-cpp-sys/external/llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m:438..550`):

1. The **entire forward** (40 layers × all phases) is constructed
   as a single GGML compute graph. Per
   `src/models/qwen3moe.cpp:120`, `ggml_build_forward_expand(gf,
   cur)` after the layer loop.
2. The graph splits into **≤8 MTLCommandBuffers** total per chunk
   (`GGML_METAL_MAX_COMMAND_BUFFERS=8`, typically `n_cb=1..2` on
   Apple Silicon per the in-source empirical comment).
3. Encoded by **main thread + n_cb worker threads in parallel** via
   `dispatch_apply(n_cb, ctx->d_queue, ctx->encode_async)`.
4. Single `commit_and_wait` at chunk end.

moeflux's session-5 batched path:
- 40 layers × ~5 phases × 1+ cmdbufs each = **~200+
  `commit_and_wait`s per chunk**.
- At ~50–200 μs CPU↔GPU toggle per commit, that's ~400–1600 ms per
  chunk wasted on sync overhead alone.
- For 992 prefill: llama.cpp 1.02 s vs moeflux 27 s ⇒ ~26 s of
  "sync overhead + serial encode" delta. Roughly matches the
  back-of-envelope.

The remaining work is not "more kernels" or "more batching" — it's
restructuring the submission shape.

## Constraints worth keeping in mind

- **Mike on 2026-05-14:** "this crate was mostly to run models
  larger than could fit in memory at once but I suspect these
  architectural wins will be good for regardless." — the original
  Cogito-V2 671B working-set-larger-than-RAM use case still
  matters; graph-mode refactor must preserve mmap discipline +
  layer-streaming.
- **Probe / diff-oracle compatibility.** Cosine ≥ 0.9999 against
  the C side stays the verification floor. The session-5 canary
  battery is the gate.
- **moeflux on main branch.** Per
  `feedback_moeflux_main_branch.md`, land directly on main; no
  feature branches while drama_llama/Mike are sole consumers.

## Phase A — GPU MoE router (enabler)

**Why first.** Today the MoE router runs on CPU
(`post_attention_residual_norm_route` →
`moe_router_cpu(scores, k_active, indices, weights)`). The router
forces a mid-layer `commit_and_wait` to read back the gate logits
from `buffers.batch_out[4]` to host. **Every layer**. Until the
router is on GPU, graph-mode is structurally impossible — the
graph has to break at every layer for the host bounce.

llama.cpp's `build_moe_ffn`
(`src/llama-graph.cpp:1305..1700`) does this entirely in GGML ops:
- `ggml_soft_max` over logits → `probs`.
- `ggml_argsort_top_k(probs, n_expert_used)` → `selected_experts`
  (a tensor of expert indices).
- `ggml_get_rows(probs, selected_experts)` → `weights`.
- `build_lora_mm_id(up_exps, cur, selected_experts)` → per-expert
  matmul indexed by the selection tensor.

We don't need the `build_lora_mm_id` path yet — our
`encode_moe_batched_permute_fuse` is bucket-driven (CPU builds the
permutation, GPU does the matmuls). For Phase A we just need the
top-K selection + weight extraction to live in a GPU buffer
without going to host.

### Shape

New kernels in `crates/moeflux/shaders/shaders.metal`:

1. `moe_softmax_topk`: take `n_tokens × n_experts` f32 logits,
   write per-token `(top_k_indices: i32[k], top_k_weights: f32[k])`
   into two output stacks. Implementation: per-token threadgroup
   does softmax + selection-sort or radix top-K. K small (8), so
   selection-sort is fine.
2. `moe_normalize_weights`: per-token, normalise top-K weights to
   sum to 1.0 (matches `moe_router_cpu`'s post-softmax behaviour).

Existing CPU `moe_router_cpu` stays as the diff oracle.

### Files touched

- `crates/moeflux/shaders/shaders.metal` — 2 new kernels.
- `crates/moeflux/src/riir/metal.rs::ALL_KERNELS` — register them.
- `crates/moeflux/src/riir/moe_router.rs` — new
  `moe_router_gpu(...)` that encodes the 2 kernels.
- `crates/moeflux/src/riir/linear_attn_forward.rs::post_attention_residual_norm_route`
  — branch on a feature/env flag: if GPU router enabled, leave
  routing indices in a GPU buffer; else CPU readback as today.
- `crates/moeflux/tests/batched_diff_oracle.rs` — new diff test
  `moe_router_gpu_matches_cpu` at random logits, cosine on weights
  + bit-exact on indices.

### Diff target

`moe_router_cpu` byte-equal on the indices (top-K selection is
deterministic given softmax tied-handling matching), cosine ≥
0.9999 on weights.

### Win estimate

Phase A alone: probably 0–5% perf change (CPU router was ~µs of
work). The *enabling* effect for Phase B is the win.

**Estimated effort:** 60–90 min.

## Phase B — Graph-mode submission

**The headline.** Restructure the batched layer forward so that
all of a chunk's GPU work lands in **one MTLCommandBuffer** per
chunk (or 2-3 if we partition for parallel encoding in Phase C).

### Submission shape: closure-Vec graph

Adopt a lightweight graph abstraction for the submission shape:

```rust
pub struct Graph<'a> {
    nodes: Vec<Box<dyn FnOnce(&CommandBufferRef) + Send + 'a>>,
}

impl<'a> Graph<'a> {
    pub fn push<F: FnOnce(&CommandBufferRef) + Send + 'a>(&mut self, f: F) {
        self.nodes.push(Box::new(f));
    }
    pub fn encode_into(self, cmdbuf: &CommandBufferRef) {
        for node in self.nodes { node(cmdbuf); }
    }
    // Phase C entry point: shard `nodes` across `n_cb` cmdbufs,
    // encode each shard on its own rayon thread, enqueue all in
    // order, wait on the last.
    pub fn encode_partitioned(self, queue: &CommandQueue, n_cb: usize) { ... }
}
```

Each `encode_*_into` function we already have becomes a closure
that captures its arguments by reference:

```rust
graph.push(|cmdbuf| {
    encode_rms_norm_bf16_into(
        cmdbuf, &rms_pipes, &input_stack, wf_buf.buffer(),
        lc.input_layernorm_w, &sum_sq, &normed_stack,
        hidden_dim as u32, RMS_NORM_EPS,
    );
});
```

**Rationale for closure-Vec over enum-Op or direct-encode:**

- **Direct-encode** (just thread `&CommandBufferRef` through
  existing functions) is 0 LOC of framework but offers no
  partitioning seam — Phase C would need a parallel-side refactor.
- **Enum-Op** (`Vec<Op>` where `Op` is a kernel-typed enum, ~500
  LOC of variants + match dispatch) would be friendlier to a
  future CoreML backend (each variant could grow an
  `encode_coreml` impl). But we don't know CoreML's actual API
  shape until M5 lands and we explore it; designing the enum
  speculatively risks producing a shape that fits neither.
- **Closure-Vec** (~50 LOC) gives us Phase C's partitioning seam
  cheaply, without committing to a future-backend shape. Closures
  can't be introspected for lowering — when CoreML lands, we'll
  refactor closures → enum variants informed by actual API
  contact. The cost of that refactor is fixed regardless of
  whether we lay closure-Vec now or not.

See [NPU roadmap](qwen_npu_roadmap.md) for the M5-era CoreML path
that the eventual enum-Op refactor would enable.

### Current submission shape (session-5 batched)

Per chunk × per layer × per phase: one fresh cmdbuf + commit_and_wait:

- Phase 1a (input rms_norm): N × cmdbuf-per-token. *Pre-Phase-B*
  this is per-token because the rms_norm output is read to host;
  *Phase B* it stays in a stacked GPU buffer.
- Phase 1b (projections): 1 cmdbuf, 4 dispatches.
- Phase 1c (recurrent kernels): 1 cmdbuf, N × 5 dispatches.
- Phase 1d (o_proj): 1 cmdbuf.
- Phase 1e (post-attn tail per-token): N × cmdbuf-per-token.
- Phase 1f (shared FFN): 1 cmdbuf.
- Phase 1g (MoE permute-fuse): 1 cmdbuf.

That's ~5 + 2N commits per layer. At N=8192, ~16k commits per
layer × 40 layers ≈ 640k commits per chunk. The
post-Phase-B target is **1 commit per chunk**.

### Required preconditions

1. Phase A (GPU router) — eliminates the routing host bounce.
2. No host bounce for h_post / shared_gate readback — the combine
   kernel must take `(h_mid, moe_sum, shared_out, shared_gate_logits)`
   on GPU and produce `hidden_out` on GPU.
3. The bucket-permute setup (`buckets.token_idx`, `buckets.weights`,
   bucket_input gather) currently runs on CPU after the GPU
   readback of routing indices. Two options:
   - **(B1) Bucket-build on GPU.** New kernel
     `moe_build_buckets`: scatter token-to-bucket assignments
     given the per-token top-K. Output: bucket_token_idx,
     bucket_weights, bucket_offsets (parallel scan).
   - **(B2) Keep bucket-build on CPU but defer it.** Read routing
     indices to host after the *whole chunk's pre-MoE* GPU work
     finishes (one readback, one commit_and_wait), then resume
     GPU-side for MoE permute-fuse. This keeps the commit count
     at 2 instead of 1 but is way less code.

   Recommendation: **B2 first**, B1 as a follow-up if profile shows
   the readback is meaningful.

### Shape (with B2)

Per chunk:
1. Build cmdbuf #1: encode all 40 layers' pre-MoE work (input
   rms_norm, projections, attn, post-attn, gate logits).
2. `commit_and_wait` cmdbuf #1 — readback routing indices for all
   N×40 (token, layer) pairs.
3. CPU: build buckets per layer (40 buckets-of-buckets total).
4. Build cmdbuf #2: encode all 40 layers' MoE permute-fuse +
   combine, plus the final norm + lm_head.
5. `commit_and_wait` cmdbuf #2 — readback final logits if
   requested.

Total commits per chunk: **2** (down from ~640k).

### Files touched

- New module: `crates/moeflux/src/riir/graph.rs`. Builders that
  accumulate encoders into a borrowed `&CommandBufferRef`.
- Refactor of `step_internal_batched_gqa` and the two
  `batched_*_layer_forward` functions: the inner kernels stay,
  but commit boundaries lift to the orchestrator.
- `LayerForwardBuffers` grows persistent stacks for the per-layer
  intermediate state (currently re-allocated per call — Phase 4's
  scratch hoist becomes a prerequisite again, but limited to
  what the graph references across phases).

### Diff target

`eval_prompt_matches_per_token_oracle` cosine = 1.0. The reordered
GPU work has the same data dependencies as today; serialisation on
encoder order inside one cmdbuf gives the same numerics.

### Win estimate

Most of the prefill gap. Realistic: 5–15× speedup on prefill,
landing us at ~150-500 tok/s on 992. Decode also benefits because
the per-layer commit count for N=1 drops from ~5 to 2.

**Estimated effort:** 2-3 sessions.

## Phase C — Parallel cmdbuf encoding

**Apple Silicon scaling.** llama.cpp encodes via `dispatch_apply`
across `n_cb` threads. Each thread holds its own cmdbuf and
encodes a slice of the graph in parallel; the cmdbufs all enqueue
into the same MTLCommandQueue, GPU pipelines them.

After Phase B lands graph-shaped submission, Phase C scales the
encode side. Concrete shape:

- Split the per-chunk graph into `n_cb=2` partitions
  (layer-aligned: layers 0..N/2 in cmdbuf A, N/2..N in cmdbuf B).
- Each partition encodes on its own thread via
  `std::thread::spawn` (or a small thread pool — rayon's pool is
  already in scope).
- Both cmdbufs `enqueue` early so Metal can schedule.
- Single `commit_and_wait` on the last cmdbuf (the second one).

Risk: Metal command buffer **enqueue order** ≠ dependency order.
The first partition writes intermediate state that the second
reads. Must use `MTLEvent` (or `MTLFence`) to express the
dependency across cmdbufs. llama.cpp uses `ev_cpy` (see
`ggml-metal-context.m:32` for the event field).

### Files touched

- `crates/moeflux/src/riir/graph.rs` — split + thread spawn.
- Plumb `MTLEvent` through the orchestrator.

### Win estimate

Apple's empirical comment in llama.cpp says n_cb=1 or 2 is
optimal — so the win from C is modest (maybe 1.3-1.5× on top of
B). But it's the last 30% of the gap.

**Estimated effort:** 60–90 min.

## Phase D — Polish

- Re-route `eval_token` through batched at N=1 (the Phase 3
  carve-out the env-flag now exists for). After Phase B, batched
  N=1 should be at least as fast as the oracle (the cross-layer
  pipelining gap the oracle's deferred-K ring exploits gets
  closed by graph-mode submission).
- Delete the deferred K-expert ring code (only used by the
  per-token oracle path that Phase D's re-route would retire).
  ~250 LOC + 5-10 MB persistent buffer reclaim per Ctx.
- Delete the prefetch state machine entirely (the whole
  `prefetch.rs` module + `MoeBuffers::data_prefetch*` +
  `PrefetchEnv<'a>` plumbing). Under graph mode, prefetch is
  redundant: mmap mode handles expert pages via demand-fault;
  pread mode does a single parallel-pread pass at chunk start
  after Phase A's routing readback. ~600 LOC reclaimed plus
  ~570 MB persistent GPU buffer per Ctx.
- Delete the per-token oracle path entirely if test coverage
  allows (some diff tests use it as a reference; those need
  CPU-only equivalents or move to the C side as the oracle).
- Clean up the per-token GPU SDPA fast path that Phase 5 deferred
  (kernels stay as test-only diff oracles but the production
  callsites die).

**Estimated effort:** 60–90 min.

## Post-Phase-D forward-looking note: NPU seam

The closure-Vec graph (Phase B's submission shape) is the natural
seam for a future CoreML/ANE backend. The M5-era plan, captured
in [NPU roadmap](qwen_npu_roadmap.md), refactors closures →
typed-enum `Op` variants once CoreML API exploration provides
the lowering shape. Apple's INT4-per-block GPU recommendation
(macOS 15+) validates that our current quantization path is
already what they suggest for Mac GPU — the eventual ANE port
would target the W8A8 path that the M4+ "faster int8-int8
compute path" enables, as a separate quantization output, not a
replacement for the existing 4-bit weights.

**Concrete actionable for Phase D**: when refactoring closures,
include a brief `label: &'static str` parameter on `Graph::push`
so the graph can be debug-printed. That's the foundation for
later inspection / lowering passes without committing to the
full enum-Op shape today.

## Order of operations

1. Phase A (Session 6, part 1). Unblocks B.
2. Phase B (Sessions 6-7). The big lift.
3. Phase C (Session 7 or 8). Parallel encoding.
4. Phase D (Session 8). Polish.

Mike's session-5 plan estimated all phases in one session; turned
out to take one + a partial. Phase B alone is plausibly 2 sessions
given the orchestrator refactor footprint. Budget accordingly.

## Verification protocol

Each phase:

```bash
cd ~/Projects/moeflux

# Cosine canaries (real artifacts):
cargo test --release -p moeflux --features model-qwen3-6-35b-a3b \
  --test diff_oracle -- --ignored --nocapture --test-threads=1 \
  state_round_trip_rust eval_token_matches_c_single_step \
  eval_prompt_matches_per_token_oracle slot_reuse_race_regression_rust \
  state_load_rust_from_c_save eval_prompt_matches_c_multi_token \
  eval_prompt_chunked_matches_eval_prompt_whole_prompt \
  prompt_cache_start_pos_nonzero_matches diag_b2_eval_prompt_chunk_1
```

Synthetic battery for new kernels (Phase A):

```bash
cargo test --release -p moeflux --features model-qwen3-6-35b-a3b \
  --test batched_diff_oracle -- --ignored --nocapture --test-threads=1
```

Bench progression (per phase, n=3, reboot between phases per
`feedback_bench_discipline.md`):

```bash
cd ~/Projects/drama_llama

# moeflux:
./bench.py --model a3b --prompt-file prefill_prompt.txt --max-tokens 1 -n 3
./bench.py --model a3b --prompt-file prefill_prompt_long.txt --max-tokens 1 -n 3
./bench.py --model a3b --max-tokens 128 -n 3

# llama.cpp (same workload — see session-5 bench.py for the --backend toggle):
./bench.py --model a3b --backend llama-cpp --prompt-file prefill_prompt.txt --max-tokens 1 -n 3
./bench.py --model a3b --backend llama-cpp --prompt-file prefill_prompt_long.txt --max-tokens 1 -n 3
./bench.py --model a3b --backend llama-cpp --max-tokens 128 -n 3

# Profile:
./profile.py --model a3b --prompt-file prefill_prompt_long.txt --max-tokens 1 --duration 60 --top 30
```

Target progression on 992 prefill:

| Phase | moeflux | llama.cpp | gap |
|---|---:|---:|---:|
| Pre-S6 (today) | 36.8 | 970 | 26× |
| Post-A | ~37 (neutral) | 970 | 26× |
| Post-B | 200-500 | 970 | 2-5× |
| Post-C | 400-700 | 970 | 1.4-2.4× |
| Post-D | 400-700 | 970 | 1.4-2.4× |

## Risks + open questions

- **Cross-cmdbuf MTLEvent contract on Apple Silicon.** llama.cpp
  uses events for async copies but the layer-graph itself is one
  cmdbuf split across threads. Need to verify Metal's
  `MTLCommandQueue` provides the dependency we need (enqueue
  order = execution order for cmdbufs on the same queue, per
  Apple docs — should hold).
- **State serialisation for `state_save` / `state_load`.** The
  snapshot wire format references `LayerForwardBuffers`. If
  graph-mode adds persistent stacks (Phase 4's scratch hoist),
  the wire format bumps. Already accepted as a pre-existing risk;
  flag any new fields explicitly when they land.
- **MLA path** (cogito-v2 671B) currently falls back to tokenwise
  oracle inside `step_internal`. Graph-mode refactor needs to
  either also cover MLA or leave the tokenwise fallback in place.
  Recommend: leave it for cogito-v2 until DeepSeek-V3 (the broader
  consumer) is closer.
- **Working set > RAM** (cogito-v2's original motivating use
  case). Graph mode doesn't change the streaming requirement —
  experts still mmap on demand. But the chunk-level commit pushes
  more pages to be live at once. Watch the cold-token profile on
  cogito-v2 to verify mmap demand-fault doesn't melt down under
  graph-mode dispatch density.

## Files where the planning context lives

- This memo: `qwen_graph_mode_session6_plan.md`.
- Session-5 outcome (to be written at close):
  `qwen_batched_prefill_session5_landed.md`.
- Architectural references (read at start of S6):
  - `~/Projects/llama-cpp-sys/external/llama.cpp/src/models/qwen3moe.cpp`
    — the graph builder shape.
  - `~/Projects/llama-cpp-sys/external/llama.cpp/src/llama-graph.cpp:1305..1700`
    — `build_moe_ffn` (GPU router reference).
  - `~/Projects/llama-cpp-sys/external/llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m:438..550`
    — Metal scheduler (graph submission).
