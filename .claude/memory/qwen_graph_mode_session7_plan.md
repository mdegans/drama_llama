# Session 7 plan — the general-shape graph compiler, ship it

**Entry:** [`qwen_graph_mode_session6_partB_precursors_landed.md`](qwen_graph_mode_session6_partB_precursors_landed.md)
**Parent plan:** [`qwen_graph_mode_session6_plan.md`](qwen_graph_mode_session6_plan.md)
**GPU-saturation diagnostic:** [`qwen_prefill_gpu_saturation_signal.md`](qwen_prefill_gpu_saturation_signal.md)
**Insulation rationale:** [`project_llama_cpp_insulation`](../../../../.claude/projects/-Users-mdegans-Projects-drama-llama/memory/project_llama_cpp_insulation.md)

## Direction

**Mike, 2026-05-14, after S7-2 landed:**

> I'd like to begin next session with `Graph<'a>` and take us the
> rest of the way, no matter what it takes. […] I'd like to take
> us architecturally as close as possible to llama.cpp so we can
> in the future swap to other backends (CoreML, maybe CUDA). If
> the code is in the shape as described, that becomes easier. So
> it's not just removing the commits, although I'm not convinced
> that won't help at least a little.

**Then, immediately after, on scope:**

> I'm not arguing we re-implement GGML and all the kernels for
> all the backends — just what we need for the models we need on
> the backends we're using. Reason for at least some of this is,
> as fast as llama.cpp is, it's not super stable and the public
> api changes a lot.

**Then, after I drafted a multi-session closure-Vec plan:**

> Actually, I'd argue we build the general shape of the graph
> compiler. We've done harder in a session.

**Locked direction:** session 7 ships the general-shape graph
compiler — typed-op enum, buffer pool, backend trait, inspection
— not the closure-Vec stepping stone. Single session. Done right.
The op vocabulary stays model-driven (~20 variants); the
*infrastructure* is fully general.

## Why typed-op over closure-Vec, decided up front

Closures hide their shape from the type system — you can't ask a
closure "what op are you, what buffers do you read/write?"
without re-encoding that information separately. Building closures
first then refactoring to typed Ops eats the design twice. Going
straight to typed Ops means:

1. **Inspection works day one.** `Graph::dump()` prints a real
   IR. Debugging diff-vs-llama.cpp becomes pattern-match.
2. **Backend trait is real.** A second backend means writing
   `Op::encode_coreml(&self, …)` for each variant — no refactor
   of producer code.
3. **Buffer lifetime / aliasing analysis becomes possible** later
   without restructuring producers. Not landing in this session,
   but the shape supports it.
4. **One fewer migration.** Mike has been clear: do it right
   once.

The trade-off is more upfront design surface in this session.
That's fine — session 6 shipped 5 commits + bench + 4 memos in
one sitting. Capacity is there.

## Architectural sketch

### Core types (new module: `crates/moeflux/src/riir/graph.rs`)

```rust
/// Identifier into the per-Graph buffer pool. Backend-agnostic;
/// the pool maps it to a concrete buffer at encode time.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct BufId(pub u32);

/// Identifier into the per-Graph constants pool (small u32/f32
/// scalars referenced by ops — dim, eps, bits, etc.).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct ConstId(pub u32);

/// Reference to a weight tensor by (offset, dtype-tag) tuple. The
/// weight file is a single MtlWeightBuf shared across the chunk;
/// only offsets change per op.
#[derive(Copy, Clone, Debug)]
pub struct WeightRef {
    pub w_off: u64,
    pub s_off: u64,
    pub b_off: u64,
    pub bits: u32,
}

/// The op set. Bounded by the models we run (Qwen3-A3B GQA +
/// linear-attn + Cogito-V2 MLA). ~20 variants. Each carries
/// everything it needs to encode — no captured `&` references.
pub enum Op {
    /// Input or post-attn RMS norm, n_tokens batched, bf16 weight.
    RmsNormBf16NTokens {
        label: &'static str,
        x: BufId,
        weight_off: u64,   // offset into the shared MtlWeightBuf
        out: BufId,
        dim: u32,
        n_tokens: u32,
        eps: f32,
    },

    /// Per-head Q/K RMS norm (linear-attn + future full-attn GPU
    /// q/k norm). Operates in-place on `x`.
    RmsNormQk { label: &'static str, x: BufId, n_heads: u32, head_dim: u32 },

    /// Residual add over [n_tokens, dim].
    ResidualAddNTokens { label: &'static str, a: BufId, b: BufId, out: BufId, n_tokens: u32, dim: u32 },

    /// Quantized matvec over n_tokens. 4-bit or 8-bit selected by
    /// weight.bits.
    MatvecNTokens {
        label: &'static str,
        weight: WeightRef,
        input: BufId,
        input_off: u64,
        output: BufId,
        output_off: u64,
        in_dim: u32,
        out_dim: u32,
        n_tokens: u32,
    },

    /// SwiGLU element-wise: out = silu(gate) * up.
    SwigluFusedBatched { label: &'static str, gate: BufId, up: BufId, out: BufId, total: u32 },

    /// Batched tiled causal SDPA.
    SdpaCausalTiled { label: &'static str, q: BufId, k: BufId, v: BufId, attn_out: BufId,
                      running_max: BufId, running_denom: BufId, v_partial: BufId,
                      n_tokens: u32, num_heads: u32, heads_per_kv: u32, head_dim: u32,
                      kv_dim: u32, kv_start: u32, kv_len_total: u32, softmax_scale: f32 },

    /// MoE softmax + selection-sort top-K (the GPU router).
    MoeSoftmaxTopK { label: &'static str, logits: BufId, indices_out: BufId, weights_out: BufId, n_tokens: u32, n_experts: u32, k: u32 },
    MoeNormalizeWeights { label: &'static str, weights: BufId, n_tokens: u32, k: u32 },

    /// MoE permute-fuse — the bucket-driven path. Buckets are
    /// pre-built on CPU and uploaded as part of the Op's fields.
    MoeBatchedPermuteFuse {
        label: &'static str,
        expert_refs: Vec<(BufId, u64)>,  // per-bucket (blob, offset)
        bucket_input: BufId, bucket_gate: BufId, bucket_up: BufId,
        bucket_act: BufId, bucket_out: BufId,
        bucket_token_idx: BufId, bucket_weights: BufId,
        out_sum: BufId,
        buckets: ExpertBuckets,  // existing struct
    },

    /// MoE combine + residual.
    MoeCombineResidualNTokens {
        label: &'static str,
        h_mid: BufId, moe_sum: BufId, shared_out: BufId,
        shared_gate: BufId, hidden_out: BufId,
        n_tokens: u32, dim: u32,
    },

    /// Linear-attn recurrent kernels (looped over N tokens).
    Conv1dStep      { … },
    ComputeDecayBeta { … },
    DeltaNetStep    { … },
    GatedRmsNorm    { … },

    /// MLA path (cogito-v2). Not all wired in session 7 — the
    /// GQA path is the priority. Reserve the variant slots.
    MlaQPrime4Bit { … },
    MlaSdpaTileAccumulate { … },
    MlaSdpaTileFinalize { … },

    /// Final norm + lm_head for chunk-end logits.
    LmHead { label: &'static str, hidden: BufId, last_token_row: u32, logits_out: BufId },
}

/// The buffer pool. Keyed by BufId, values are owned metal::Buffer
/// (refcounted Objective-C handles — cheap to clone).
pub struct BufferPool {
    buffers: Vec<metal::Buffer>,
}

impl BufferPool {
    pub fn new() -> Self { Self { buffers: vec![] } }
    pub fn register(&mut self, buf: metal::Buffer) -> BufId {
        let id = BufId(self.buffers.len() as u32);
        self.buffers.push(buf);
        id
    }
    pub fn get(&self, id: BufId) -> &metal::Buffer { &self.buffers[id.0 as usize] }
}

/// The graph itself. Ops are appended in dispatch order. Backend
/// owns the cmdbuf and decides how to slice / commit / parallelise.
pub struct Graph {
    pub ops: Vec<Op>,
    pub pool: BufferPool,
}

impl Graph {
    pub fn new() -> Self { Self { ops: vec![], pool: BufferPool::new() } }
    pub fn push(&mut self, op: Op) { self.ops.push(op); }
    pub fn register_buf(&mut self, buf: metal::Buffer) -> BufId {
        self.pool.register(buf)
    }
    pub fn labels(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.ops.iter().map(|op| op.label())
    }
    pub fn dump(&self) -> String { … }  // for debugging
}

impl Op {
    pub fn label(&self) -> &'static str { … }
}

/// Backend trait. Per-op encoding lives in the impl.
pub trait Backend {
    fn encode_op(&mut self, op: &Op, pool: &BufferPool, cmdbuf: &metal::CommandBufferRef);
    fn encode_graph_into(&mut self, graph: &Graph, cmdbuf: &metal::CommandBufferRef) {
        for op in &graph.ops {
            self.encode_op(op, &graph.pool, cmdbuf);
        }
    }
}

pub struct MetalBackendCtx<'a> {
    pub metal: &'a mut MetalBackend,
    pub wf_buf: &'a MtlWeightBuf,
    // Cached pipeline states fetched once per chunk.
    pub matvec_pipes: MatvecPipelines,
    pub rms_n_pipe: RmsNormBf16FusedNTokensPipeline,
    pub router_pipes: MoeRouterPipelines,
    pub residual_add_n_pso: ComputePipelineState,
    pub combine_pso: ComputePipelineState,
    pub swiglu_pso: ComputePipelineState,
    pub sdpa_pipes: BatchedSdpaPipelines,
    // … the other PSOs the model needs.
}

impl<'a> Backend for MetalBackendCtx<'a> {
    fn encode_op(&mut self, op: &Op, pool: &BufferPool, cmdbuf: &metal::CommandBufferRef) {
        match op {
            Op::RmsNormBf16NTokens { x, weight_off, out, dim, n_tokens, eps, .. } => {
                encode_rms_norm_bf16_fused_n_tokens(
                    cmdbuf, &self.rms_n_pipe,
                    pool.get(*x), self.wf_buf.buffer(), *weight_off,
                    pool.get(*out), *dim, *n_tokens, *eps,
                );
            }
            Op::MatvecNTokens { weight, input, input_off, output, output_off, in_dim, out_dim, n_tokens, .. } => {
                encode_matvec_n_tokens(
                    cmdbuf, &self.matvec_pipes,
                    self.wf_buf.buffer(),
                    weight.w_off, weight.s_off, weight.b_off,
                    pool.get(*input), *input_off,
                    pool.get(*output), *output_off,
                    *in_dim, *out_dim, *n_tokens, weight.bits,
                );
            }
            // … each variant calls the existing `encode_X_into` helper.
        }
    }
}
```

### Producer side: layer-forward becomes graph-builder

`batched_full_attn_layer_forward` and
`batched_linear_attn_layer_forward` change from "imperatively
encode into cmdbuf" to "push Ops into Graph." Function signature
changes from taking buffer refs to taking BufIds:

```rust
pub(super) fn batched_linear_attn_layer_forward(
    graph: &mut Graph,
    wf: &WeightFile,
    layer_cache: &LayerWeightCache,
    layer_idx: usize,
    n_tokens: usize,
    k_active: usize,
    expert_files: &ExpertFiles,
    hidden_in_id: BufId,
    hidden_out_id: BufId,
    // … etc, no more direct buffer args
) -> Result<(), LayerForwardError> {
    // Register intermediates with the pool, push Ops, return.
    let normed_id = graph.register_buf(MtlBuffer::with_len(…).into_buffer());
    graph.push(Op::RmsNormBf16NTokens {
        label: "input_rms_norm",
        x: hidden_in_id,
        weight_off: layer_cache.input_layernorm_w,
        out: normed_id,
        dim: hidden_dim as u32,
        n_tokens: n_tokens as u32,
        eps: RMS_NORM_EPS,
    });
    // … 1b, 1c, 1d, 1e, 1f, 1g, combine — all push Ops.
}
```

The orchestrator builds the full graph across all 40 layers, then
hands it to the backend, which encodes into cmdbufs and commits:

```rust
fn step_internal_batched_gqa(&mut self, tokens: &[i32], …) {
    let mut graph = Graph::new();
    let hidden_a = graph.register_buf(…initial embeddings…);
    let mut hidden_b = graph.register_buf(…empty…);
    let mut in_id = hidden_a;
    let mut out_id = hidden_b;

    for layer in 0..40 {
        batched_X_layer_forward(&mut graph, …, in_id, out_id, …);
        std::mem::swap(&mut in_id, &mut out_id);
    }

    // (Optional) graph.dump() for debugging.

    let mut backend_ctx = MetalBackendCtx::new(metal, wf_buf, …);
    let cmdbuf = queue.new_command_buffer();
    backend_ctx.encode_graph_into(&graph, cmdbuf);
    cmdbuf.commit();
    cmdbuf.wait_until_completed();

    // Final norm + lm_head reads `in_id`'s last token row.
}
```

**Open design question for session 7:** the routing readback + CPU
bucket build still happens *mid-graph* in the current shape. Two
options:

(A) **Two-phase graph.** Build a pre-MoE graph, encode + commit
    + wait, read back routing, CPU bucket build, build a MoE
    graph that uses the buckets, encode + commit. Two cmdbufs
    per chunk.

(B) **Backend interrupt.** The backend's `encode_graph_into`
    learns to detect a `RoutingReadbackBarrier` Op and split
    cmdbufs there. Same effect but the producer side stays as
    one Graph build.

Recommend **(A)** for session 7 — cleanest. (B) can come later
once we have GPU bucket build (S7-η).

### What ships in session 7

Minimum bar for the session to "succeed":

1. `graph.rs` exists with `BufId`, `BufferPool`, `Op` enum (~20
   variants), `Graph`, `Backend` trait.
2. `MetalBackendCtx` implements `Backend` for every Op variant
   the GQA path needs — each variant's encode is a one-line call
   to the existing `encode_X_into` helper.
3. Both `batched_*_layer_forward` rewritten as Graph builders
   (no more direct cmdbuf encoding inside).
4. `step_internal_batched_gqa` rewritten to build → encode →
   commit (two cmdbufs: pre-MoE and MoE).
5. **Canary 9/9 cosine = 1.0.**
6. `Graph::dump()` works — useful for the next session's debug
   work.

Stretch goals (nice if they fit, fine to defer):

7. GPU q/k norm + RoPE + KV append + sigmoid_gate kernels — gets
   full-attn fully into the pre-MoE graph.
8. Parallel encode via rayon / dispatch_apply equivalent.

## Order of operations

Build bottom-up. Each step is a checkpoint with canary.

### S7-1: Op enum + BufferPool + Graph type (no Backend yet)

Just the types. Write enough variants for the GQA path
(RmsNormBf16NTokens, RmsNormQk, ResidualAddNTokens, MatvecNTokens,
SwigluFusedBatched, SdpaCausalTiled, MoeSoftmaxTopK,
MoeNormalizeWeights, MoeBatchedPermuteFuse, MoeCombineResidualNTokens,
Conv1dStep, ComputeDecayBeta, DeltaNetStep, GatedRmsNorm, LmHead).

Unit tests:
- Round-trip: push N Ops, iterate `graph.ops`, recover labels.
- `dump()` prints sensibly.
- `BufferPool::register` returns sequential IDs.

**Checkpoint:** compiles, unit tests green.

### S7-2: `Backend` trait + `MetalBackendCtx`

`encode_op` match arms call the existing helpers. Heavy but
mechanical.

Unit test (synthetic): build a tiny graph (one RmsNorm + one
Matvec), run it through the backend on synthetic buffers, verify
output matches a direct call.

**Checkpoint:** unit tests green.

### S7-3: Rewrite `batched_linear_attn_layer_forward` as graph
builder

Linear-attn first because it's the simpler shape (no q/k norm /
RoPE / sigmoid_gate complications). Builds Ops for 1a-1e
(pre-MoE) and 1f-1g + combine (MoE).

The MoE half stays in a *separate* Graph that the orchestrator
builds *after* the routing readback.

**Checkpoint:** canary 9/9 with linear-attn path going through
Graph + Backend. Full-attn still uses the old imperative path.

This is the highest-risk step. The function signature change
ripples to the orchestrator. Plan to spend the largest chunk of
session 7 here.

### S7-4: Rewrite `batched_full_attn_layer_forward` same way

Full-attn has the q/k norm + RoPE + KV append + sigmoid_gate CPU
host bounces. Two options:

(a) **Keep the host bounces inline** — the layer-forward Graph
    builder *runs* the CPU steps as it goes (push Op, do CPU
    work, push next Op). Graph still builds linearly; the
    cmdbuf gets split at the host-bounce points by the
    Backend. Equivalent to today's commit shape but with the
    Op-typed dispatch.

(b) **GPU port** q/k norm + RoPE + KV append + sigmoid_gate
    first. Eliminates the bounces. New kernels (S7-δ/ε from
    the earlier plan draft).

Recommend **(a) first** — session 7 is about shape. (b) becomes
session 8 (GPU port of the 3-4 small full-attn-specific kernels).

**Checkpoint:** canary 9/9 with both paths through Graph.

### S7-5: Refactor `step_internal_batched_gqa`

Two-phase graph: pre-MoE → commit → readback routing → CPU
bucket-build → MoE+combine → commit.

This is where commits drop from ~100/chunk to ~2/chunk for
linear-attn layers, ~5/chunk for full-attn (because of host
bounces — until S7-δ/ε).

**Checkpoint:** canary 9/9 + warm bench. Expect modest perf bump
from cross-layer pipelining.

### S7-6: Graph::dump() polish + commit

Make `dump()` produce something useful for diffing builds. Probably
just one line per op with label + key arg summary. Commit the
session's work as a single coherent diff (or several smaller
commits per checkpoint above).

### S7-7 (if time): GPU full-attn ops (S7-δ/ε from earlier draft)

`rms_norm_per_head_n_tokens` + `rope_apply_n_tokens` +
`kv_cache_append_n_tokens` + `sigmoid_gate_n_tokens`. Each ~30
lines of Metal. Full-attn collapses into the pre-MoE graph.

### S7-8 (stretch): Parallel encode

`Backend::encode_graph_partitioned(graph, queue, n_cb=2)`.
Cmdbufs enqueued in order, single commit_and_wait at the end.

## Bounded op vocabulary (locked)

Stay model-driven. Today's needs:

**Qwen3-A3B (GQA + linear-attn):**
- RmsNormBf16NTokens, RmsNormQk, ResidualAddNTokens
- MatvecNTokens (4-bit/8-bit)
- SwigluFusedBatched, SdpaCausalTiled
- MoeSoftmaxTopK, MoeNormalizeWeights
- MoeBatchedPermuteFuse, MoeCombineResidualNTokens
- Conv1dStep, ComputeDecayBeta, DeltaNetStep, GatedRmsNorm
- (full-attn S7-7): RmsNormPerHeadNTokens, RopeApplyNTokens,
  KvCacheAppendNTokens, SigmoidGateNTokens
- LmHead

**Cogito-V2 (MLA):**
- MlaQPrime4Bit, MlaSdpaTileAccumulate, MlaSdpaTileFinalize
- NoauxTcRouter (DeepSeek-V3 routing — reuses MoeSoftmaxTopK +
  group-mask + scale, may need an extra variant)

That's ~22 ops. Each variant maps to one Metal kernel that already
exists or will exist by session-7 close. No general-purpose op
fusion, no shape inference, no constant folding — just the
*dispatch list*.

## What is *not* in scope this session

- **CoreML or CUDA backend impl.** The Backend trait is ready;
  no second impl yet.
- **In-place buffer reuse / lifetime analysis.** Each Op gets
  distinct BufIds; the pool grows monotonically. Later pass once
  profile shows allocation churn.
- **GPU bucket build** (S7-η). Still requires the chunk_a / chunk_b
  split.
- **MLA path full conversion.** Stays on tokenwise oracle for now;
  reserve the variant slots in the Op enum but don't wire callers.

## Risks

- **Op enum fields explosion.** Some ops have a lot of args (SDPA,
  MoeBatchedPermuteFuse). If a variant ends up with 20+ fields,
  consider grouping into a sub-struct (`SdpaCausalTiledArgs`). Not
  a refactor blocker.
- **Buffer pool growth.** Each layer allocates ~15 intermediates.
  40 layers × 15 = 600 BufIds per chunk. Vec<metal::Buffer>
  storage is fine; cmdbuf encoding indirection through pool.get()
  is a Vec index — sub-nanosecond.
- **Ownership of the metal::Buffer in the pool.** The pool owns
  them. Layers register and return BufIds; the orchestrator drops
  the whole Graph at chunk end, releasing all buffers. Need to
  verify this doesn't conflict with persistent state (KV cache —
  that's owned by `LayerState`, not registered in the pool).
- **Test surface change.** The synthetic diff tests in
  `batched_diff_oracle.rs` call the old `encode_X_into` helpers
  directly. They keep working as kernel-level diff tests. The
  full-forward canaries in `diff_oracle.rs` exercise the new
  Graph path.
- **No half-states allowed.** If session-7 ships partial (e.g.
  linear-attn through Graph but full-attn still imperative), the
  orchestrator has to dispatch differently. That's fine
  temporarily but not durably. Commit boundaries: S7-3 (linear-
  attn graph) and S7-4 (full-attn graph) are landed together if
  possible, or behind a feature flag if not.

## Verification protocol

Canary battery after each checkpoint. Bench at session close
(post-reboot per `feedback_bench_discipline.md`).

```bash
cd ~/Projects/moeflux
cargo test --release -p moeflux --features model-qwen3-6-35b-a3b \
  --test diff_oracle -- --ignored --nocapture --test-threads=1 \
  state_round_trip_rust eval_token_matches_c_single_step \
  eval_prompt_matches_per_token_oracle slot_reuse_race_regression_rust \
  state_load_rust_from_c_save eval_prompt_matches_c_multi_token \
  eval_prompt_chunked_matches_eval_prompt_whole_prompt \
  prompt_cache_start_pos_nonzero_matches diag_b2_eval_prompt_chunk_1

cd ~/Projects/drama_llama
./bench.py --model a3b --prompt-file prefill_prompt.txt --max-tokens 1 -n 3
./bench.py --model a3b --prompt-file prefill_prompt_long.txt --max-tokens 1 -n 3
```

Pre-session-7 baseline (warm, n=1):
- 992 prefill: 74.66 tok/s (S7-2 post)
- 16k prefill: 42.66 tok/s (S7-1a post; S7-2 not re-benched on 16k)

Target post-session-7:
- **Canary 9/9 cosine = 1.0** (load-bearing — the perf number is
  secondary).
- 992 prefill: any movement is gravy. Mike's hypothesis is
  modest improvement from cross-layer pipelining. Could be flat;
  the architectural win is the point.

## Forward look (post-session 7)

- **Session 8:** S7-7 (GPU full-attn ops) + S7-8 (parallel encode).
- **Session 9+:** GPU bucket build (S7-η) if profile flags
  readback as the new pole. Otherwise kernel-efficiency work
  (FlashAttention SDPA, persistent chunk buffers).
- **Session N (later):** when CoreML/CUDA contact happens, add a
  second `Backend` impl. The producer side doesn't change at all
  — that's the entire point.

## Files where context lives

- This memo (rewritten 2026-05-14 to lock the general-shape
  decision): `qwen_graph_mode_session7_plan.md`.
- Session-6 outcome: `qwen_graph_mode_session6_partB_precursors_landed.md`.
- GPU saturation observation: `qwen_prefill_gpu_saturation_signal.md`.
- llama.cpp insulation motivation: `project_llama_cpp_insulation`
  (in auto-memory, not in-repo).
- llama.cpp reference (read at session start):
  - `~/Projects/llama-cpp-sys/external/llama.cpp/src/llama-graph.cpp:1305..1700`
    — `build_moe_ffn` reference for MoE routing shape.
  - `~/Projects/llama-cpp-sys/external/llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m:438..550`
    — Metal scheduler / dispatch_apply pattern for session 8's
    parallel encode work.
