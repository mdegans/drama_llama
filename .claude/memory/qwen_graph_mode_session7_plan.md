# Session 7+ plan — `Graph<'a>` and the rest of the way

**Entry:** [`qwen_graph_mode_session6_partB_precursors_landed.md`](qwen_graph_mode_session6_partB_precursors_landed.md)
**Parent plan:** [`qwen_graph_mode_session6_plan.md`](qwen_graph_mode_session6_plan.md)
**GPU-saturation diagnostic:** [`qwen_prefill_gpu_saturation_signal.md`](qwen_prefill_gpu_saturation_signal.md)

**Direction (Mike, 2026-05-14, after S7-2 landed):**

> I'd like to begin next session with `Graph<'a>` and take us the
> rest of the way, no matter what it takes. […] I'd like to take
> us architecturally as close as possible to llama.cpp so we can
> in the future swap to other backends (CoreML, maybe CUDA). If
> the code is in the shape as described, that becomes easier. So
> it's not just removing the commits, although I'm not convinced
> that won't help at least a little.

**Mandate:** the load-bearing goal is *backend portability*, not
prefill tok/s. The commit reduction is a corollary of getting the
shape right. Multi-session OK. One step at a time.

## Why `Graph<'a>` is the right scaffolding

Three properties we want:

1. **Backend portability.** Today every encoder is hard-coded to
   `metal::CommandBufferRef`. To swap to CoreML's MPSGraph or a
   future CUDA backend, we need a *layer of indirection* between
   "what the model does" (RMS norm, matvec, SDPA, …) and "how the
   backend encodes it." Closures aren't the final answer for that
   (they can't be introspected for lowering), but they're the
   right stepping stone — once we have actual API contact with a
   second backend, we refactor closures → typed-enum `Op`
   variants. The eventual enum gets `encode_metal`, `encode_coreml`,
   `encode_cuda` impls.

2. **Parallel encoding.** llama.cpp uses `dispatch_apply(n_cb,
   queue, encode_async)` — splits the graph across worker threads,
   each encoding a slice of the dispatch list into its own cmdbuf
   in parallel. Apple Silicon sweet spot is `n_cb=1..2`, but this
   is the last 30% of the gap to llama.cpp. The closure-Vec is the
   natural partitioning seam — split `nodes[..]` into `n_cb`
   slices, encode each on a rayon thread.

3. **Commit reduction.** Currently ~100 commits/chunk
   (full_attn 4/layer × 10 + linear_attn 2/layer × 30). With the
   inter-layer host bounce eliminated (S7-2 already landed), the
   *only* remaining commit barriers inside one chunk are CPU
   host-bounces:
   - Phase 1b → Phase 2 in full_attn (q/k norm + RoPE + KV append).
   - Phase 2 → Phase 1d in full_attn (sigmoid_gate).
   - Phase 3c → Phase 3d in both paths (routing readback + bucket
     build).

   Move those to GPU (or accept them as split points that produce
   2 cmdbufs/chunk total) and we land at llama.cpp's commit shape.

   Mike's caveat — *"I'm not convinced that won't help at least a
   little"* — is right to flag. Even though GPU is saturated
   *within* each cmdbuf, removing per-layer commits would let layer
   N+1's pre-MoE encode start before layer N's MoE commits,
   tightening cross-layer GPU pipelining. The win is probably
   modest but real.

## The `Graph<'a>` API design

```rust
/// A pending compute graph. Each node captures its arguments by
/// reference (via the `'a` lifetime) and writes Metal encoders into
/// the cmdbuf when `encode_into` runs.
///
/// Today: closure-based. Tomorrow (after CoreML/CUDA contact):
/// closures → typed-enum `Op` variants. The label parameter
/// survives both shapes.
pub struct Graph<'a> {
    nodes: Vec<Node<'a>>,
}

struct Node<'a> {
    label: &'static str,
    encode: Box<dyn FnOnce(&CommandBufferRef) + Send + 'a>,
}

impl<'a> Graph<'a> {
    pub fn new() -> Self { Self { nodes: vec![] } }
    pub fn with_capacity(cap: usize) -> Self {
        Self { nodes: Vec::with_capacity(cap) }
    }

    pub fn push<F>(&mut self, label: &'static str, encode: F)
    where F: FnOnce(&CommandBufferRef) + Send + 'a {
        self.nodes.push(Node { label, encode: Box::new(encode) });
    }

    /// Total dispatch count — useful for picking n_cb in the
    /// parallel-encode variant.
    pub fn len(&self) -> usize { self.nodes.len() }
    pub fn is_empty(&self) -> bool { self.nodes.is_empty() }

    /// Encode every node into `cmdbuf` in order. Single-threaded.
    pub fn encode_into(self, cmdbuf: &CommandBufferRef) {
        for node in self.nodes {
            (node.encode)(cmdbuf);
        }
    }

    /// Encode into `n_cb` cmdbufs in parallel via rayon. Each
    /// thread takes a contiguous slice of `nodes`. Cmdbufs are
    /// enqueued in order on `queue` — Metal serialises execution
    /// order to encode order on the same queue, so the data
    /// dependency between slices is honoured without explicit
    /// MTLEvent sync (caller verifies this assumption).
    ///
    /// Returns the last cmdbuf so the caller can `commit_and_wait`
    /// or chain further work.
    pub fn encode_partitioned<'b>(
        self, queue: &'b CommandQueue, n_cb: usize,
    ) -> &'b CommandBufferRef { … }

    /// Debug helper: print all node labels in order. Useful for
    /// diffing two builds of the same graph.
    pub fn labels(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.nodes.iter().map(|n| n.label)
    }
}
```

**File:** new `crates/moeflux/src/riir/graph.rs`.

**The label parameter is load-bearing for the future.** It's the
identity that survives the eventual enum-Op refactor (`Op::Matvec
{ label: "qkv_proj", … }`), and the inspection seam for graph
dumping / debugging across backends.

## Migration phases

Each phase is a checkpoint. Canary 9/9 must pass at each before
moving on. Mike: *"one step at a time."*

### S7-α: Land `Graph<'a>`, no callers

Just write `graph.rs`. No production callsite. Unit tests in the
module showing:
- `push` then `encode_into` orders correctly.
- `encode_partitioned` with `n_cb=2` correctly splits across two
  cmdbufs and enqueues in order.

This is the foundation. Tiny commit. Verifies the abstraction
compiles cleanly with the borrow checker (the `'a` lifetime + the
`FnOnce + Send` bound is the load-bearing constraint — closures
need to capture by reference, not move, so per-layer-cache
references can outlive the Graph build).

### S7-β: Convert each `encode_*_into` to be `Graph`-compatible

The existing `encode_X_into(cmdbuf, ...)` helpers already take
`&CommandBufferRef`. To convert: wrap each call as a closure
pushed into a Graph instead of called directly. Mechanical refactor
inside each batched layer-forward.

```rust
// Before:
super::gpu_norm::encode_rms_norm_bf16_fused_n_tokens(
    cmdbuf, &rms_n_pipe, hidden_in_buf, wf_buf.buffer(),
    layer_cache.input_layernorm_w, normed_buf.buffer(),
    hidden_dim as u32, n_tokens as u32, RMS_NORM_EPS,
);

// After:
let rms_n_pipe = rms_n_pipe.clone();  // closure capture
let wf_buf_ref = wf_buf.buffer().clone();  // metal::Buffer is refcounted
let normed_buf_ref = normed_buf.buffer().clone();
graph.push("input_rms_norm", move |cmdbuf| {
    super::gpu_norm::encode_rms_norm_bf16_fused_n_tokens(
        cmdbuf, &rms_n_pipe, hidden_in_buf, wf_buf_ref,
        layer_cache.input_layernorm_w, normed_buf_ref,
        hidden_dim as u32, n_tokens as u32, RMS_NORM_EPS,
    );
});
```

Tricky bit: `metal::Buffer` is an `NSObject`-backed reference-
counted handle — cloning it is cheap (Objective-C `retain`). But
the pipeline state objects we get via `.clone()` on the cached
PSO are also cheap. The `'a` lifetime on Graph means closures can
borrow references to the layer-cache, wf_buf, etc. without owning
them — what doesn't satisfy `+ 'a` is anything bound to a tighter
scope.

Encode happens via `Graph::encode_into(cmdbuf)` once everything is
queued up — keeps the same commit boundaries as today initially,
then S7-γ tightens them.

Effort: 30+ encoders touched. Tedious but mechanical.

**Verification gate:** canary 9/9 after this phase, identical
commits/chunk count as pre-S7-β. Pure refactor.

### S7-γ: Collapse cmdbuf boundaries

With every dispatch flowing through a Graph, the orchestrator can
sequence multiple Graphs into one cmdbuf:

```rust
// Today: each batched_X_layer_forward calls Graph::encode_into +
// commit_and_wait internally per phase.

// After γ:
let mut chunk_graph_a = Graph::with_capacity(40 * 8);  // pre-MoE
for layer in 0..40 {
    encode_pre_moe_phases(&mut chunk_graph_a, layer, …);
}
let cmdbuf_a = queue.new_command_buffer();
chunk_graph_a.encode_into(cmdbuf_a);
cmdbuf_a.commit();
cmdbuf_a.wait_until_completed();

// CPU: read all 40 layers' routing, bucket-build per layer.

let mut chunk_graph_b = Graph::with_capacity(40 * 6);  // MoE + final
for layer in 0..40 {
    encode_moe_phases(&mut chunk_graph_b, layer, buckets[layer], …);
}
encode_final_norm_and_lm_head(&mut chunk_graph_b, …);
let cmdbuf_b = queue.new_command_buffer();
chunk_graph_b.encode_into(cmdbuf_b);
cmdbuf_b.commit();
cmdbuf_b.wait_until_completed();
```

**Total commits per chunk: 2** (down from current ~100).

**Two prerequisites for this phase:**

1. **GPU q/k norm + RoPE + KV append** (full-attn only). Existing
   kernels: `rms_norm_qk` already used by linear-attn. RoPE: needs
   a Metal kernel (`encode_yarn_rope_apply` exists for MLA but
   doesn't fit the GQA shape — verify). KV cache append: a new
   kernel that copies `k_host`, `v_host` into the cache rows.
   Roughly 3 small kernels.

2. **GPU sigmoid_gate** (full-attn only). Trivial element-wise
   kernel; eliminates the Phase 2→3b host bounce.

Without (1)+(2), full-attn still has 3 inter-phase host bounces
inside one layer; (γ) only fuses the pre-MoE chain for linear-attn
fully. That's still a useful checkpoint (linear-attn is 30/40
layers); ship it, then do (1)+(2) as follow-up phases.

**Verification gate:** canary 9/9 + benchmark. This is where we
might see the *modest perf bump* Mike anticipated, even on GPU-
saturated workloads — cross-layer pipelining opens up.

### S7-δ: GPU q/k norm + RoPE + KV append (full-attn)

Three small kernels:

- `rms_norm_per_head_n_tokens` — already have per-head shape for
  linear-attn; needs an n_tokens variant.
- `rope_apply_n_tokens` — yarn-aware (read pos from a per-token
  buffer or constant).
- `kv_cache_append_n_tokens` — copies (k, v) from contiguous per-
  token buffers into the cache at `kv_start..kv_start+n`.

After δ, full-attn's Phase 1b→Phase 2 boundary is GPU-only.

### S7-ε: GPU sigmoid_gate (full-attn)

```metal
kernel void sigmoid_gate_n_tokens(
    device float* attn_out  [[buffer(0)]],
    device const float* q_gate [[buffer(1)]],
    constant uint& total [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= total) return;
    attn_out[tid] = (1.0f / (1.0f + exp(-q_gate[tid]))) * attn_out[tid];
}
```

Trivial. After ε, full-attn has just one inter-phase host bounce
(routing readback) — same as linear-attn. Full-attn graph collapses
into the pre-MoE chunk_graph.

### S7-ζ: Parallel cmdbuf encoding

`Graph::encode_partitioned(queue, n_cb=2)` over `chunk_graph_a`.
Each rayon thread encodes a slice. Cmdbufs enqueued in order,
single commit_and_wait at the end. Apple's empirical sweet spot
on M-series is `n_cb=1..2`.

Risk: cross-cmdbuf dependency. The first cmdbuf's writes need to
be visible to the second cmdbuf's reads. Metal's documented
behaviour is *enqueue order = execution order on the same queue*,
so this should hold without explicit `MTLEvent` — but verify on a
small fixture before trusting it on the full graph.

**Verification gate:** canary 9/9 + bench. Apple's docs suggest
~1.3-1.5× over single-threaded encode at this scale.

### S7-η: B1 (GPU bucket build)

Optional. The routing readback after pre-MoE forces the
chunk_graph_a / chunk_graph_b split. If we move
`build_expert_buckets` to GPU (a parallel-scan + scatter kernel),
the split goes away — we get **1 commit per chunk**.

But (a) the buckets are small (8192 × 8 × 8B = 524 KB), the
readback is sub-millisecond, and (b) the bucket-build kernel is
non-trivial. Defer until profile shows the readback is the new
pole — same logic the parent plan applied.

## Forward look: enum-Op for the second backend

When we get API contact with CoreML or CUDA, the closure-Vec
graph refactors to:

```rust
pub enum Op {
    RmsNorm { label: &'static str, input: BufId, weight: BufId, output: BufId, eps: f32 },
    Matvec { label: &'static str, w: WeightId, input: BufId, output: BufId, n_tokens: u32, in_dim: u32, out_dim: u32, bits: u32 },
    Sdpa { … },
    MoePermuteFuse { … },
    // …
}

pub struct Graph {
    ops: Vec<Op>,
}

impl Op {
    fn encode_metal(&self, cmdbuf: &CommandBufferRef, ctx: &MetalCtx) { … }
    fn encode_coreml(&self, graph: &MPSGraph, ctx: &CoreMLCtx) { … }
    fn encode_cuda(&self, stream: &CudaStream, ctx: &CudaCtx) { … }
}
```

The `BufId` / `WeightId` types are backend-agnostic handles into
a buffer pool. The dispatch shape stays the same — the encoding
side splits per backend.

We're not building this yet. We're building the closure-Vec graph
that will refactor cleanly *into* this shape. The closure-Vec
phase is the "shape verification" phase — once we have the
dispatch list in Vec form, lowering to typed Ops is a search-and-
replace plus some buffer ID plumbing.

Critical: the closure capture pattern in S7-β should not capture
anything that *couldn't* be represented as a typed enum field. If
a closure captures a complex `&'a SomeStruct`, that's a signal we
need a typed BufId/WeightId shape sooner rather than later.

## Estimated session count

This is multi-session. Mike: *"no matter what it takes."* Rough
sketch:

- **Session 7:** S7-α + S7-β. Graph<'a> lands + every encoder
  threaded through it. No perf change, large diff. Canary green.
- **Session 8:** S7-γ (the collapse) for linear-attn path. Bench:
  expect modest win on prefill (5-15%?) from cross-layer GPU
  pipelining.
- **Session 9:** S7-δ (full-attn GPU ops) + S7-ε (sigmoid_gate).
  Full-attn joins the chunk_graph. Bench.
- **Session 10:** S7-ζ (parallel encoding). Bench.
- **Session 11+:** S7-η or kernel-efficiency work (FlashAttention-
  style SDPA, persistent chunk buffers), informed by post-ζ
  profile.

Mike's reframe — "architectural cleanliness over tok/s" — means
landing S7-α/β cleanly even if perf doesn't move. The win is the
*shape*. Once shape is right, the dispatch optimisations and
eventual backend port follow cheaply.

## Verification protocol (unchanged from session 6)

Canary battery after each phase:

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

Bench post-reboot per `feedback_bench_discipline.md`. The session-6
warm-machine bench (75 tok/s on 992) gives the directional baseline.

## Files where context lives

- This memo: `qwen_graph_mode_session7_plan.md`.
- Session-6 outcome: `qwen_graph_mode_session6_partB_precursors_landed.md`.
- GPU saturation observation: `qwen_prefill_gpu_saturation_signal.md`.
- llama.cpp reference (read at session start):
  - `~/Projects/llama-cpp-sys/external/llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m:438..550`
    — `dispatch_apply(n_cb, encode_async)`.
  - `~/Projects/llama-cpp-sys/external/llama.cpp/src/llama-graph.cpp:1305..1700`
    — `build_moe_ffn` shape.

## Risks

- **Closure capture vs `+ 'a`**. Some encoders take many references
  (layer_cache, wf_buf, pipelines). Closures will capture either by
  reference (cheap, lifetime-bound) or by clone (slightly more work
  but no lifetime constraint). Pattern: clone refcounted handles
  (metal::Buffer, ComputePipelineState are Obj-C refcounted, cheap
  to clone), borrow Rust-side structs by reference.
- **Cmdbuf size limits**. 40 layers × 8 phases ≈ 320 dispatches in
  one cmdbuf. llama.cpp encodes more without trouble. Watch for
  "command buffer too large" errors but unlikely.
- **`MTLEvent` for parallel encode**. Apple docs say enqueue order
  = execution order on the same queue, but verify on a small
  fixture. If cross-cmdbuf data dependency needs explicit MTLEvent
  sync, the closure-Vec abstraction needs an event-aware encode
  path.
- **Refactor footprint**. S7-β touches every batched encoder. Big
  diff, mostly mechanical. Use `Graph::labels()` to verify the
  dispatch sequence is unchanged before/after — that's the cheap
  regression check before canary.

## Out of scope for session 7

- **GPU bucket build (S7-η)**: only if profile flags the readback.
- **Enum-Op refactor**: waits for second-backend API contact.
- **MLA path** (cogito-v2): graph-mode applies only to the GQA
  path. MLA stays on tokenwise oracle.
- **Kernel efficiency** (FlashAttention SDPA, persistent chunk
  buffers): post-ζ once the shape is right.

## Carry-overs explicitly preserved

The session-6 plan called out commit shape, host bounces, GPU
saturation as facts. All still hold:

- ~100 commits/chunk today.
- 3 forced host bounces per full_attn layer; 1 per linear_attn.
- GPU saturated within each cmdbuf — see
  [`qwen_prefill_gpu_saturation_signal.md`](qwen_prefill_gpu_saturation_signal.md).

Open with `profile.py` if curious, but the *primary* session-7
work is `Graph<'a>` + S7-β refactor — orthogonal to whatever the
new pole turns out to be.
