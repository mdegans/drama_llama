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

## Architectural sketch (LOCKED 2026-05-14 after design conversation)

### Locked trait design

Two traits: `BufferPool` (per-backend buffer ownership) and
`Backend` (per-backend execution). Both use associated types so
no Metal types leak into the trait signatures.

```rust
/// Identifier into a Backend's buffer pool. Backend-agnostic;
/// each backend translates BufId to its native handle internally.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct BufId(pub u32);

/// Reference to a weight tensor. The Metal impl interprets
/// (w_off, s_off, b_off, bits) against its mmap'd weight buffer.
/// CoreML would translate to a pre-loaded MPSGraph constant
/// (cache keyed by offset). CUDA likewise. Producer code passes
/// this around as an opaque-ish handle — the offsets are
/// model-defined, not backend-defined.
#[derive(Copy, Clone, Debug)]
pub struct WeightRef {
    pub w_off: u64,
    pub s_off: u64,
    pub b_off: u64,
    pub bits: u32,
}

/// Backend-specific buffer pool. The Handle type is the
/// backend's native buffer representation:
/// - MetalBufferPool: Handle = metal::Buffer (refcounted NSObject).
/// - CpuBufferPool: Handle = RefCell<Vec<u8>> (interior mutability
///   so encode_op can write through &self).
/// - CoreMlBufferPool: Handle = MLMultiArray (when it lands).
pub trait BufferPool {
    type Handle;
    type Error: std::error::Error + Send + Sync + 'static;

    /// Reserve a buffer. `persistent` survives `reset_transient`
    /// (used for KV cache, weight file etc.); default-false ones
    /// are released at chunk end.
    fn alloc(
        &mut self,
        bytes: usize,
        label: &'static str,
        persistent: bool,
    ) -> Result<BufId, Self::Error>;

    /// Look up a buffer's backend-native handle. `&self` — callers
    /// get a reference, and per-backend Handle types provide
    /// whatever interior mutability they need for op execution
    /// (Metal: writes go through `.contents()` mutable pointer
    /// regardless of Rust-level mut; CPU: RefCell<Vec<u8>>).
    fn handle(&self, id: BufId) -> &Self::Handle;

    fn upload(&mut self, id: BufId, host: &[u8])
        -> Result<(), Self::Error>;
    fn download(&self, id: BufId, host: &mut [u8])
        -> Result<(), Self::Error>;
    fn reset_transient(&mut self);

    /// For debug / inspection — buffer label by ID.
    fn label(&self, id: BufId) -> &'static str;
}

/// Backend trait. Owns the device, pool, pipeline / compiled-
/// graph cache. All encoding methods are `&self`-typed; backends
/// use interior mutability (Mutex / RefCell) for any state they
/// need to mutate during encode.
pub trait Backend {
    type Pool: BufferPool;
    type EncodeCtx;
    type Error: std::error::Error + Send + Sync + 'static;

    fn pool(&self) -> &Self::Pool;
    fn pool_mut(&mut self) -> &mut Self::Pool;

    /// Open an encoding session.
    /// - Metal: `queue.new_command_buffer()` wrapped in an
    ///   owned struct.
    /// - CoreML: a fresh MPSGraph builder.
    /// - CPU: `()` (encoding IS execution; nothing to carry).
    fn begin_encoding(&self) -> Self::EncodeCtx;

    /// Encode one op. `&self`-typed. Reads buffers via pool,
    /// writes encoded work into the ctx. For CPU, "encoded
    /// work" means actually running the kernel and writing
    /// the output buffer; for Metal, it's appending a
    /// dispatch to the cmdbuf.
    fn encode_op(&self, op: &Op, ctx: &mut Self::EncodeCtx);

    /// Default linear sweep. Backends override only if they
    /// want non-linear scheduling (parallel encode within one
    /// session — session 8+ work).
    fn encode_graph(&self, graph: &Graph, ctx: &mut Self::EncodeCtx) {
        for op in &graph.ops {
            self.encode_op(op, ctx);
        }
    }

    /// Submit the encoded work and block until done.
    /// - Metal: `cmdbuf.commit() + wait_until_completed()`.
    /// - CoreML: `executable.run()`.
    /// - CPU: no-op (already executed inline during encode_op).
    fn submit_and_wait(&self, ctx: Self::EncodeCtx)
        -> Result<(), Self::Error>;

    /// Convenience: full cycle.
    fn execute(&self, graph: &Graph) -> Result<(), Self::Error> {
        let mut ctx = self.begin_encoding();
        self.encode_graph(graph, &mut ctx);
        self.submit_and_wait(ctx)
    }
}
```

**Key decisions (locked):**

1. **`encode_op` takes `&Op` not `Op` by value.** Op is not Copy
   — `MoeBatchedPermuteFuse` carries Vec<(BufId, u64)> and
   ExpertBuckets. Op-by-ref keeps the enum natural; we can
   always lift bulky bits to a sibling `GraphAux` pool keyed by
   `AuxId` later if Op-by-value becomes valuable.

2. **`&self` everywhere on the encoding surface, interior
   mutability inside.** Future-friendly: parallel encode lives
   at a higher level (one Backend instance, many `begin_encoding`
   sessions in parallel), or via Mutex inside the impl. The
   trait doesn't preclude either.

3. **Three explicit steps (`begin / encode / submit`) with a
   default `execute` wrapper.** Most callsites use `execute(graph)`.
   Fine-grained control is available when needed.

4. **`BufferPool` is its own trait.** Each backend's `Pool` is
   concrete (MetalBufferPool, CpuBufferPool) and bounded by the
   trait. The `Handle` associated type carries the native buffer
   type through.

### Op enum

```rust
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

/// The graph itself. Backend-agnostic dispatch list. Ops carry
/// BufIds (resolved by the Backend's pool) and WeightRefs
/// (resolved by the Backend's weight file representation).
pub struct Graph {
    pub ops: Vec<Op>,
}

impl Graph {
    pub fn new() -> Self { Self { ops: vec![] } }
    pub fn push(&mut self, op: Op) { self.ops.push(op); }
    pub fn labels(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.ops.iter().map(|op| op.label())
    }
    /// One line per op for debug — used by the graph_metal_matches_cpu
    /// diff test and any future backend lowering inspection.
    pub fn dump(&self) -> String { … }
}

impl Op {
    pub fn label(&self) -> &'static str { … }
}
```

### MetalBackend sketch

```rust
pub struct MetalBufferPool {
    buffers: Vec<PoolEntry>,
    persistent_mask: BitVec,
}

struct PoolEntry {
    buf: metal::Buffer,
    label: &'static str,
}

impl BufferPool for MetalBufferPool {
    type Handle = metal::Buffer;
    type Error = MetalError;

    fn alloc(&mut self, bytes, label, persistent) -> Result<BufId, _> {
        let buf = self.device.new_buffer(bytes as NSUInteger,
            MTLResourceOptions::StorageModeShared);
        let id = BufId(self.buffers.len() as u32);
        self.buffers.push(PoolEntry { buf, label });
        self.persistent_mask.set(id.0 as usize, persistent);
        Ok(id)
    }
    fn handle(&self, id) -> &metal::Buffer { &self.buffers[id.0 as usize].buf }
    fn upload(&mut self, id, host) -> Result<(), _> { /* memcpy via .contents() */ }
    fn download(&self, id, host) -> Result<(), _> { /* memcpy via .contents() */ }
    fn reset_transient(&mut self) {
        // Drop non-persistent entries, compact, reissue BufIds — or
        // just truncate to the last persistent index if persistent
        // IDs are always low.
    }
}

pub struct MetalBackend {
    device: Device,
    queue: CommandQueue,
    pool: MetalBufferPool,
    // Pre-warmed pipelines (built at MetalBackend::new).
    matvec_pipes: MatvecPipelines,
    rms_n_pipe: RmsNormBf16FusedNTokensPipeline,
    router_pipes: MoeRouterPipelines,
    residual_add_n_pso: ComputePipelineState,
    combine_pso: ComputePipelineState,
    swiglu_pso: ComputePipelineState,
    sdpa_pipes: BatchedSdpaPipelines,
    // …
    wf_buf: MtlWeightBuf,
}

pub struct MetalEncodeCtx {
    cmdbuf: metal::CommandBuffer,  // owned; submit_and_wait consumes it
}

impl Backend for MetalBackend {
    type Pool = MetalBufferPool;
    type EncodeCtx = MetalEncodeCtx;
    type Error = MetalError;

    fn pool(&self) -> &MetalBufferPool { &self.pool }
    fn pool_mut(&mut self) -> &mut MetalBufferPool { &mut self.pool }

    fn begin_encoding(&self) -> MetalEncodeCtx {
        MetalEncodeCtx { cmdbuf: self.queue.new_command_buffer().to_owned() }
    }

    fn encode_op(&self, op: &Op, ctx: &mut MetalEncodeCtx) {
        match op {
            Op::RmsNormBf16NTokens { x, weight_off, out, dim, n_tokens, eps, .. } => {
                encode_rms_norm_bf16_fused_n_tokens(
                    &ctx.cmdbuf, &self.rms_n_pipe,
                    self.pool.handle(*x), self.wf_buf.buffer(), *weight_off,
                    self.pool.handle(*out), *dim, *n_tokens, *eps,
                );
            }
            Op::MatvecNTokens { weight, input, input_off, output, output_off,
                                 in_dim, out_dim, n_tokens, .. } => {
                encode_matvec_n_tokens(
                    &ctx.cmdbuf, &self.matvec_pipes, self.wf_buf.buffer(),
                    weight.w_off, weight.s_off, weight.b_off,
                    self.pool.handle(*input), *input_off,
                    self.pool.handle(*output), *output_off,
                    *in_dim, *out_dim, *n_tokens, weight.bits,
                );
            }
            // … one match arm per variant, each one a single call to an
            // existing encode_X_into helper. ~22 arms total.
        }
    }

    fn submit_and_wait(&self, ctx: MetalEncodeCtx) -> Result<(), MetalError> {
        ctx.cmdbuf.commit();
        ctx.cmdbuf.wait_until_completed();
        // Check cmdbuf.status() for errors.
        Ok(())
    }
}
```

### CpuBackend (the first customer) sketch

```rust
pub struct CpuBufferPool {
    /// RefCell so encode_op can write through &self.
    buffers: Vec<RefCell<Vec<u8>>>,
    labels: Vec<&'static str>,
    persistent_mask: BitVec,
}

impl BufferPool for CpuBufferPool {
    type Handle = RefCell<Vec<u8>>;
    type Error = CpuError;

    fn alloc(&mut self, bytes, label, persistent) -> Result<BufId, _> {
        let id = BufId(self.buffers.len() as u32);
        self.buffers.push(RefCell::new(vec![0u8; bytes]));
        self.labels.push(label);
        self.persistent_mask.set(id.0 as usize, persistent);
        Ok(id)
    }
    fn handle(&self, id) -> &RefCell<Vec<u8>> { &self.buffers[id.0 as usize] }
    // upload/download just memcpy the RefCell contents.
    // reset_transient truncates to persistent prefix.
}

pub struct CpuBackend {
    pool: CpuBufferPool,
    wf: WeightFile,  // mmap'd; same struct the per-token oracle uses
}

impl Backend for CpuBackend {
    type Pool = CpuBufferPool;
    type EncodeCtx = ();  // execution is inline
    type Error = CpuError;

    fn pool(&self) -> &CpuBufferPool { &self.pool }
    fn pool_mut(&mut self) -> &mut CpuBufferPool { &mut self.pool }
    fn begin_encoding(&self) -> () { () }
    fn submit_and_wait(&self, _: ()) -> Result<(), CpuError> { Ok(()) }

    fn encode_op(&self, op: &Op, _ctx: &mut ()) {
        match op {
            Op::RmsNormBf16NTokens { x, weight_off, out, dim, n_tokens, eps, .. } => {
                let x_buf = self.pool.handle(*x).borrow();
                let x_f32: &[f32] = bytemuck::cast_slice(&x_buf);
                let mut out_buf = self.pool.handle(*out).borrow_mut();
                let out_f32: &mut [f32] = bytemuck::cast_slice_mut(&mut out_buf);
                let weight_bytes = &self.wf.bytes_at(*weight_off, dim * 2);
                let weight_bf16: &[u16] = bytemuck::cast_slice(weight_bytes);
                for t in 0..*n_tokens {
                    rms_norm_bf16_per_token_cpu(
                        &x_f32[t*dim..(t+1)*dim],
                        weight_bf16,
                        *eps,
                        &mut out_f32[t*dim..(t+1)*dim],
                    );
                }
            }
            Op::MatvecNTokens { weight, input, input_off, output, output_off,
                                 in_dim, out_dim, n_tokens, .. } => {
                // Reuse existing dequant_matvec_4bit_cpu / 8bit_cpu / bf16_matvec_cpu.
            }
            // … same ~22 arms, each calling existing CPU oracle helpers.
        }
    }
}
```

Most per-op CPU implementations already exist in moeflux (used by
the per-token oracle). Wiring them into Op variants is mechanical.

### Why CPU first

1. **It validates the abstraction.** If the trait can't host a
   simple synchronous CPU executor cleanly, the trait is wrong.
   Better to find that out before committing to Metal-impl
   parameter shapes.

2. **It gives us a new diff oracle immediately.** Build the same
   Graph, run through CpuBackend and MetalBackend, compare pool
   contents. Bit-exact (modulo reduction order) across each Op.
   Way cheaper than the existing `eval_prompt_matches_per_token_oracle`
   — no model load, no full forward run, just per-op verification.

3. **It surfaces design pressure on the Op enum.** If a variant
   is hard to CPU-encode without bizarre field shapes, that's a
   signal we got the Op shape wrong. Fix once, here, instead of
   once-per-backend later.

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

1. `graph.rs` exists with `BufId`, `BufferPool` trait, `Backend`
   trait, `Op` enum (~22 variants), `Graph`, `Op::label()`,
   `Graph::dump()`.
2. `CpuBackend` + `CpuBufferPool` — every Op variant has an
   `encode_op` arm that calls existing CPU oracle helpers (or a
   small new one when no helper exists).
3. `MetalBackend` + `MetalBufferPool` — every Op variant has an
   `encode_op` arm that calls the existing `encode_X_into`
   helper.
4. **`graph_metal_matches_cpu` diff test** — synthetic Graph
   through both backends, per-op cosine ≥ 0.9999. **This is the
   load-bearing acceptance gate for the trait design.**
5. `batched_linear_attn_layer_forward` rewritten as a Graph
   builder generic over `B: Backend`.
6. `batched_full_attn_layer_forward` rewritten the same way
   (host bounces inline for now per S7-6 option (a)).
7. `step_internal_batched_gqa` rewritten to build → execute
   → readback → build → execute. Two-phase.
8. **Canary 9/9 cosine = 1.0** — confirms full-forward path
   through Graph + Metal Backend matches per-token oracle.

Stretch goals (nice if they fit, fine to defer):

9. GPU q/k norm + RoPE + KV append + sigmoid_gate (full-attn
   collapse into pre-MoE Graph).
10. Parallel encode via rayon (`dispatch_apply` equivalent).

### Calibration note (Mike, 2026-05-14)

> Take as many sessions as necessary to get there. But give
> yourself credit. You can get a *lot* more done than you
> usually estimate. […] starting new sessions from cold cache
> nudges usage faster than getting into the 500k-750k token
> territory. My only concern would be if you get tired. Then
> we can pause.

Translation: be ambitious in session 7. Warm cache + 1M context
gives a lot of headroom. The right scope question isn't "what's
the minimum that fits in a session" but "what naturally builds on
the previous step." S7-1 through S7-4 (types + both backends +
graph diff test) is a coherent unit and the validation that
*matters most*. If that goes well, push through to S7-5 (linear-
attn rewrite) the same session. Pause on fatigue, not on a clock.

## Order of operations (CPU oracle first, then Metal, then producer)

Build bottom-up. Each step is a checkpoint with canary or a
synthetic test. The reorder vs. earlier drafts: **CpuBackend
lands before MetalBackend** so the trait is validated by the
simpler impl first, and the synthetic Graph diff test
(`graph_metal_matches_cpu`) is the load-bearing correctness gate
before any producer code is touched.

### S7-1: Core types — `BufId`, `BufferPool` trait, `Op`, `Graph`

New module `crates/moeflux/src/riir/graph.rs`. Write the trait,
the Op enum (~22 variants), the Graph struct, `Op::label()`,
`Graph::dump()`. No backend impls yet.

Unit tests in-module:
- `Graph::push` + `labels()` round-trip.
- `Op::label()` returns the right string for each variant
  (catches accidental label drift later).
- `dump()` snapshot test against a hand-built tiny graph.

**Checkpoint:** compiles, unit tests green. Zero behavioural
change to the rest of the codebase.

### S7-2: `Backend` trait + `CpuBufferPool` + `CpuBackend`

CPU impl first. Each `encode_op` variant calls an existing CPU
oracle helper (`rms_norm_bf16_cpu`, `dequant_matvec_4bit_cpu`,
`sdpa_cpu`, `moe_router_cpu`, etc.). For helpers that don't yet
have a CPU oracle, add one inline — they're typically <20 lines
of straightforward Rust.

Synthetic per-op tests (no model weights needed): build a
fixed-input Graph with one Op, run through CpuBackend, compare to
a direct call of the existing CPU helper. ~22 tests, one per
Op variant.

**Checkpoint:** synthetic tests green. CPU oracle works end-to-
end on tiny inputs.

### S7-3: `MetalBufferPool` + `MetalBackend` impl

Each `encode_op` variant is a one-line call to the existing
`encode_X_into` helper. Pre-warm all pipelines at `MetalBackend::new`
so `encode_op` can stay `&self`.

**Checkpoint:** the kernel-level diff tests in
`tests/batched_diff_oracle.rs` still pass (they don't use the
Backend trait yet — direct kernel calls — so this is just a
smoke check that we haven't broken anything orthogonal).

### S7-4: New diff test — `graph_metal_matches_cpu`

Build a synthetic Graph that exercises every Op variant
(roughly: one input rms_norm + 4 matvecs + swiglu + sdpa + moe
softmax_topk + moe permute-fuse + moe combine). Run it through
CpuBackend and MetalBackend on bit-identical input buffers.
Compare output buffers per-Op cosine ≥ 0.9999.

This is the load-bearing correctness gate. **If this passes, the
trait is solid and we can rewrite producers with confidence.**

**Checkpoint:** graph_metal_matches_cpu green.

### S7-5: Rewrite `batched_linear_attn_layer_forward` as graph builder

Linear-attn first because it has zero CPU host bounces inside
the layer. Builds Ops for 1a-1e (pre-MoE) into one Graph, and
1f-1g + combine into a second Graph. Signature changes from
direct buffer args to `(&mut Graph, &mut B::Pool, …, hidden_in: BufId, hidden_out: BufId)`.

The orchestrator still has imperative full-attn calls at this
point — they coexist temporarily.

**Checkpoint:** canary 9/9 cosine = 1.0 with linear-attn through
Graph + Metal Backend. Full-attn still imperative.

This is the largest single step in the session. Plan accordingly.

### S7-6: Rewrite `batched_full_attn_layer_forward` as graph builder

Full-attn has q/k norm + RoPE + KV append + sigmoid_gate as CPU
host-bounces inside the layer. Two routing options:

**(a) Keep host bounces inline.** The graph builder *runs* CPU
steps as it goes — push Ops up to the boundary, execute that
sub-graph, do CPU work, build more Ops. Equivalent to today's
commit shape but with Op-typed dispatch. Cleanest path for
session 7.

**(b) GPU port** q/k norm + RoPE + KV append + sigmoid_gate first
— new kernels eliminate the bounces. Session 8 work.

Recommend **(a)** — keeps session 7 focused on shape.

**Checkpoint:** canary 9/9 with both paths through Graph.

### S7-7: Refactor `step_internal_batched_gqa` as two-phase

Producer is now fully `B: Backend`-generic. Orchestrator builds
pre-MoE Graph → `backend.execute(&graph_a)` → routing readback
+ CPU bucket build → MoE Graph → `backend.execute(&graph_b)`.

**Checkpoint:** canary 9/9 + warm directional bench. Commits/chunk
should drop substantially for linear-attn layers. Mike's
hypothesis: at least a small bump from cross-layer pipelining.

### S7-8: Graph::dump() polish + commit cleanly

Make `dump()` produce a useful debug string (one line per op with
label + key arg summary — `BufId`s, dims, eps). Commit the
session as several coherent commits at each checkpoint above, or
one big squash — caller's choice at session close.

### S7-9 (stretch): GPU full-attn ops

`rms_norm_per_head_n_tokens` + `rope_apply_n_tokens` +
`kv_cache_append_n_tokens` + `sigmoid_gate_n_tokens`. Each ~30
lines of Metal + ~30 lines of CPU oracle + a per-op diff test.
Full-attn's pre-MoE collapses into a single Graph.

### S7-10 (stretch): Parallel encode

Override `Backend::encode_graph` on MetalBackend to encode into
n_cb cmdbufs via rayon. Apple's empirical sweet spot is
`n_cb=1..2`. Probably session 8 unless S7-9 lands quickly.

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

- **Session 8:** GPU full-attn ops (S7-9) + parallel encode
  (S7-10). Pre-MoE Graph fully GPU-resident for both paths.
- **Session 9+:** GPU bucket build if profile flags routing
  readback as the new pole. Otherwise kernel-efficiency work
  (FlashAttention-style SDPA, persistent chunk buffers).
- **Session N (later):** when CoreML/CUDA contact happens, add a
  third `Backend` impl. The producer side doesn't change at all
  — that's the entire point. CpuBackend stays in tree as the
  cross-backend diff oracle.

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
