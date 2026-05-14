# NPU (Apple Neural Engine) roadmap

**Status:** speculative roadmap, no commitments. Triggering
condition: M5 Studio arrives + Mike has time to explore.

## Why this exists

Mike's `project_backend_rework` memory called out "long-term: rip
llama.cpp, target NPU" as a stated direction. Pre-2024, my mental
model was "ANE doesn't help for Q4 LLM inference because it wants
fp16 and Q4 dequant eats memory bandwidth." Apple's CoreML docs
(referenced by Mike on 2026-05-14) updated that model
substantially:

> Linear quantization: Refers to approximating weights with a
> quantization function. Core ML supports INT4 and INT8
> quantization options for weights and INT8 for activations.

And:

> INT4 `per-block` quantization of weights can work really well
> for models using the GPU on a Mac.
>
> 8-bit activation plus weight quantization, also referred to as
> the W8A8 mode, can lead to considerable latency benefits on the
> Neural Engine by leveraging the faster int8-int8 compute path
> supported in newer hardware (A17 pro, M4).

This memo captures the post-update plan.

## Where we stand today

- moeflux uses W4A16 (4-bit weights, fp16 activations) on Apple
  Silicon GPU via custom Metal kernels.
- This is **exactly what Apple recommends** for Mac GPU per their
  own docs. Our current path is not architecturally behind for
  the Mac GPU target.
- ANE on M2 Max (current hardware) is a step behind M4+'s
  int8-int8 path. M5 Studio (incoming) will likely improve it
  further.
- llama.cpp's Metal backend is also targeting GPU INT4 — same
  hardware target, different implementation. The 26-32× prefill
  gap we measured against llama.cpp on 2026-05-14 is GPU-vs-GPU,
  not ANE-vs-GPU; closing it is session 6's job.

## Hardware path

| Generation | Path | Notes |
|---|---|---|
| **M2 Max (today)** | Metal GPU + W4A16 | Apple's recommendation. Session 6 graph-mode refactor brings us close to llama.cpp here. |
| **M5 Studio (incoming)** | Same as M2 + W8A8 ANE experiments | M5 likely extends M4's int8-int8 path. W8A8 weight artifact ships alongside W4A16. |
| **Future iPhone/iPad** | CoreML + W8A8 on ANE | Memory pressure means mmap-via-CoreML's weight constants is essential. |

## Architectural steps for the ANE/CoreML port

These compose on top of session 6 landing. Each step is its own
session-or-more.

### Step 1 — Backend trait extraction (post session 6)

Session 6 Phase B ships a closure-Vec graph
(`Vec<Box<dyn FnOnce(&CommandBufferRef) + Send>>`). Closures can't
be lowered to non-Metal backends — they capture Metal-specific
binds. The first step toward a portable backend is refactoring
closures into a typed-enum `Op`:

```rust
pub enum Op<'a> {
    InputRmsNormStack { input: &'a Buffer, output: &'a Buffer, ... },
    Matvec4bitN { weight_buf: &'a Buffer, w_off: u64, ..., bits: u32 },
    SoftmaxTopK { logits: &'a Buffer, indices_out: &'a Buffer, k: u32, ... },
    MoePermuteFuse { ... },
    Combine { ... },
    // ~30 variants total covering qwen3moe / qwen3.6-a3b / mla / etc.
}

pub trait Backend {
    fn encode(&mut self, op: &Op<'_>);
    fn submit_and_wait(&mut self);
}
```

Each `Op` variant carries shape+dtype data sufficient for any
backend to lower. The Metal backend (today) implements `encode`
by dispatching to the existing `encode_*_into` functions. A
future CoreML backend implements `encode` by appending an
`MLProgram` node to a builder.

**Cost:** ~500 LOC for the enum + Metal lowering. Pure refactor;
zero new functionality. Justified only when there's a second
backend in sight (i.e., M5 has arrived and CoreML port is the
next concrete task).

### Step 2 — W8A8 quantization output

CoreML's faster path on M4+ is W8A8 (INT8 weights, INT8
activations). Our current W4A16 packs differently and can't be
fed directly. The model-prep tool needs a new artifact:

- `packed_experts_w8a8/layer_NN.bin` — INT8 expert weights,
  per-tensor scales, alongside the existing
  `packed_experts/layer_NN.bin` (Q4).
- `model_weights_w8a8.bin` — INT8 non-expert weights (q_proj,
  k_proj, v_proj, etc.), alongside `model_weights.bin` (Q4).
- Symmetric quantization, per-row or per-block. Apple's docs
  recommend `per-block` granularity for Q4; W8A8 likely wants
  per-row or per-tensor.

Tool: extend the existing `moeflux-prep` (or whatever the
model-conversion tool is named) with a `--quant w8a8` flag. ~1
session.

Disk cost: ~2× the W4A16 footprint per model (8-bit vs 4-bit).
A3B's ~32 GB → ~64 GB W8A8. Cogito-V2's ~340 GB → ~680 GB W8A8.
Cogito-V2 W8A8 wouldn't fit on most disks; W8A8 is for the
smaller variants where ANE win matters more than disk size.

### Step 3 — CoreML backend implementation

Implement the `Backend` trait against Apple's CoreML APIs. Two
sub-options:

- **MPSGraph (`MetalPerformanceShadersGraph`)**: low-level,
  Metal-native. Lets us build a graph at runtime, hint compute
  device (`computeDeviceType = .neuralEngine`), and submit. More
  control, less abstraction.
- **CoreML `MLProgram`**: higher-level, compile-time graph
  embedded in an `mlpackage`. Apple's runtime auto-routes across
  ANE/GPU/CPU. Less control, more "trust the runtime."

Recommendation: **MPSGraph for the moeflux port**. Reasons:
- We need control over weight residency (the streaming-larger-
  than-RAM use case). `MLProgram` compiles weights as constants
  — fights our use case.
- MPSGraph lets us mix and match: ANE for matmuls,
  GPU for the routing kernel, CPU for the bucket-build readback.

Bindings: hand-rolled FFI via `objc2` crates. Estimated ~500-800
LOC plus the operator-by-operator mapping. The `Op` enum from
Step 1 gives each variant a place to define its MPSGraph
emission.

**Estimated effort:** 2-3 sessions.

### Step 4 — Benchmark on M5

The plan only ships if M5 + ANE actually win on our workload.
Apple's docs warn that compute-unit routing varies per-shape and
per-compiler; benchmarking is the only honest answer.

Critical workloads:

- 992 prefill — primarily ANE-friendly (large matmul, fixed
  shapes).
- 16k prefill — chunked; same shape per chunk.
- Decode N=1 — likely *not* ANE-friendly (memory-bound, small
  matmul). Apple's docs are explicit that compressed kernels'
  decompression strategy varies; on ANE the fully-decompressed
  weights might overwhelm the cache.
- Cogito-V2 671B streaming — this is moeflux's specialization.
  Per-layer working set is bigger than ANE's preferred
  characteristics. Unlikely to benefit.

**Decision matrix:**

| Workload | If ANE wins ≥1.5× | If ANE loses | Action |
|---|---|---|---|
| Prefill (any N) | Ship CoreML path | Stay Metal | — |
| Decode N=1 | Ship CoreML path | Stay Metal | (expected: stay Metal) |
| Cogito-V2 671B | (unlikely to win) | Stay Metal | (expected) |

### Step 5 — Palettization exploration (optional)

Apple's docs call out **palettization** (lookup-table-based
quantization) as the format that "typically works the best on
the Neural Engine for runtime memory and latency gains." Bits
options: `{1, 2, 3, 4, 6, 8}`.

This is a different quantization scheme from linear quant. Some
upstream models ship palettized weights directly (Apple-published
models, some research projects). If a future model arch arrives
with palettized weights as the canonical format, ANE is the
natural backend — no conversion needed.

For our existing models (Qwen3.5/3.6, Cogito-V2), palettization
would require re-quantizing from upstream sources. Not a current
priority; revisit when a palettized model is on the roadmap.

## Constraints + risks

- **Working-set-larger-than-RAM** is moeflux's reason for
  existing. CoreML's `MLProgram` compiles weights as constants
  and won't stream — that's the reason MPSGraph is the
  recommended sub-option in Step 3. Need to verify MPSGraph
  lets us bind external buffers without eager-loading.
- **ANE's compute-unit routing is opaque.** Apple's runtime
  decides per-op whether ANE/GPU/CPU runs. We can hint with
  `MPSGraphCompilationDescriptor.computeDevice`, but the runtime
  can ignore hints. Benchmarking is the only confirmation.
- **W8A8 accuracy.** Going from W4 to W8 should be *better*
  accuracy (more precision), but some upstream training-time
  quantization assumes Q4 specifically. Verify perplexity stays
  within 1-2% of the W4A16 baseline before shipping.
- **CoreML model file size.** If we compile through `MLProgram`,
  the `mlpackage` is large (weights baked in). Per-model
  packaging vs. weight-streaming is a deployment-shape
  conversation.

## Timing

Triggering events, in order:

1. Session 6 lands (graph-mode on Metal). **Necessary precondition**
   for any of this — the closure-Vec → enum-Op refactor needs
   session 6's graph to exist.
2. M5 Studio arrives + is set up.
3. Mike has interest + time to explore.
4. Step 1 (Backend trait) — 1 session.
5. Step 2 (W8A8 model-prep) — 1 session.
6. Step 3 (CoreML backend) — 2-3 sessions.
7. Step 4 (bench + decision) — 1 session.

Total: ~5-7 sessions after triggering events, assuming the bench
indicates ANE actually wins on M5. If it doesn't, the work stops
at Step 4 with a documented negative result and we stay on Metal.

## What this means for session 6

**Nothing changes in the immediate plan.** Phase A (GPU router),
Phase B (closure-Vec graph), Phase C (parallel encoding), Phase D
(cleanup) are all Metal-side. The NPU roadmap is downstream and
conditional.

The one tiny session 6 nod toward this roadmap: include a
`label: &'static str` parameter on `Graph::push` (cheap, helps
debugging, is the foundation for later enum-Op metadata).

## References

- Apple CoreML compression docs (link Mike shared 2026-05-14):
  https://apple.github.io/coremltools/docs-guides/source/opt-overview.html
- WWDC 2024 session 10159 "What's new in CoreML": LLM-specific
  optimization features.
- `~/Projects/llama-cpp-sys/external/llama.cpp/ggml/src/ggml-metal/`
  — Metal backend reference for graph-mode submission patterns.
  Not directly relevant for ANE but useful for the Metal-side
  Phase B work.
