# Phase 4 A3B 8-bit dequant — landed (partial)

Session 2026-04-26. Picked up from `plan_phase4_moeflux_a3b_debug.md`'s
"Real fix (recommended)" branch: add native 8-bit support to moeflux
rather than working around via extract-time re-quantization.

## What shipped

### moeflux commit `f47c0b3` — 8-bit dequant

- `metal_infer/shaders.metal`: new `dequant_matvec_8bit_v3` kernel.
  Same threadgroup layout as the 4-bit v3 (ROWS_PER_TG=8, 256 threads,
  SIMD reduction, 4096-float x_shared cache). Unrolls 4 byte-unpacks
  per uint32 instead of 8 nibble-unpacks. Handles `out_dim=1` (the
  shared_expert_gate case) via the same out-of-bounds early-return.
- `metal_infer/infer.m`:
  - `BatchMatvecSpec` gains `int bits`; 0 is treated as 4 in all
    dispatchers so the field is legacy-safe.
  - `cpu_dequant_matvec` parameterized on bits with mask
    `(1u<<bits)-1u` — 2/4/8-bit all decode correctly.
  - `gpu_batch_matvec` / `gpu_encode_batch_matvec` pick the 8-bit
    pipeline when `spec.bits == 8 && ctx->matvec_8bit_v3` is live.
  - `LayerWeightCache` caches per-tensor `gate_bits` / `seg_bits`
    resolved from the manifest at load. The fused-path moe_specs
    (5123) and non-fused moe_specs (5172) use `lc->gate_bits` /
    `lc->seg_bits`. The prompt-eval path (2828) resolves via
    `get_tensor_info` since it doesn't use LayerWeightCache.
  - `TensorInfo` gains `int bits`, populated by `load_manifest`
    (defaults to 4 for U32 tensors without the field — backward
    compatible with older extractions).
  - `MetalCtx` gains `matvec_8bit_v3` pipeline; `metal_setup`
    compiles it with WARN-not-fatal on failure (falls back to CPU
    path for 8-bit if the Metal kernel is absent — covers the case
    where someone runs with an older shaders.metal).
- `metal_infer/extract_weights.py`:
  - Reads the HF config `quantization` block. Resolves per-tensor
    overrides (80 on A3B; 0 on A17B) to integer `bits` and emits
    that in the manifest for every U32 tensor.
  - Emits `default_bits` / `default_group_size` in the config
    block for diagnostics.
  - **Promotes `linear_attn.A_log` BF16 -> F32 at extract time.**
    A17B stores A_log as F32; A3B as BF16. moeflux reads it as
    `float *` unconditionally so A3B's delta-net decay was reading
    two bf16 values as one float garbage. Shift bf16 -> f32 via
    `uint32 << 16` is bit-exact; adds 64 extra bytes per layer.

### drama_llama-side: no code changes

cross_backend test is still the right shape. Ran it post-fix:

```
argmax agreement: 0/32 -> 1/32  (3.1%)
top-20 Jaccard step 0: 0.081 -> 0.086  (noise-level)
```

Same 3% as the prior 4-bit-reqant workaround noted in the debug doc,
but via the clean native path with full 8-bit precision. Not yet at
the 95% threshold. Test stays `#[ignore]`.

## What the output actually looks like now

Smoke with `"The quick brown fox"`, greedy, 20 tokens:

Before 8-bit fix: `"ro a . The rome . The rome ..."` (degenerate
low-id cycle — routing gate picking wrong experts through 40 layers).

After 8-bit fix: `"es are the quick brown foxes are the quick brown
foxes..."` (coherent English, loops on prompt echo). Same output as
the prior 4-bit-reqant workaround.

MLX reference on same tokens: `"jumps over the lazy dog.\n\n\`\`\`python..."`
(pangram, per the prior session's ground-truth run).

So we're now producing **valid** output, conditioned on the prompt,
but diverging from MLX's argmax at the first new token.

## Layout verified against MLX

This session I added a numpy-only cross-check (no commit, ran
one-shot). It dequants A3B layer 0's `mlp.gate.weight` two ways:

1. `mx.dequantize(W, S, B, group_size=64, bits=8)` — MLX reference.
2. Pure-numpy unpack assuming 4 LSB-first bytes per uint32 (what
   our Metal kernel and CPU path do).

Result: mean abs diff 2.3e-05, max abs diff 4.9e-04. Matches the
BF16 scale-rounding ceiling. So the **bytes on disk are being read
correctly by our new code paths** — this is not a layout/packing bug.

Script is gone (ran directly via `uv run`). Key formula if re-
deriving: MLX's 8-bit packing stores consecutive 4 values in a
uint32 LSB-first: `packed = v0 | (v1<<8) | (v2<<16) | (v3<<24)`,
group-affine with per-group (scale, bias) in BF16.

## Where the remaining divergence must live

Not the 8-bit gate dequant. Not A_log. Not a fundamental routing
bug (output is coherent, top-10 token candidates overlap with
llama.cpp's).

Candidates for next session, in rough order of likelihood:

1. **Cumulative FP16/BF16 precision drift** across 40 layers. The
   debug doc noted A3B has only 2048 HIDDEN_DIM (vs A17B's 4096),
   which gives less numerical headroom per layer; moeflux's
   `MTLMathModeFast` may cost us more on A3B than it does on A17B.
   Try switching to `MTLMathModeSafe` as a one-line A-B test.

2. **Qwen3.6-specific architectural detail we haven't surfaced.**
   moeflux was built around A17B. A3B is same family but not
   identical. The debug doc already ruled out shape constants,
   embedding dequant, attention gate split, mRoPE config, layer-
   type pattern, and 8-bit dequant. Gated attention's exact scale
   factor? GPT-2 vs. something else in tokenizer bytes-to-unicode?

3. **Bug #4 from the debug doc** — the `gpu_linear_sentinel`
   fallback read. Non-fused path, shouldn't be hit on normal A3B
   runs. But worth confirming the fused path is actually the one
   being used end-to-end (a printf at layer 0 would answer).

## moeflux repo state

- Branch `main`, 6 commits ahead of origin.
- Stash `stash@{0}` from prior session (dense-F32 gate workaround +
  already-landed A17B-literal refactor + entangled sentinel fix)
  **dropped** (stash 335a6e69) — confirmed obsolete per debug doc's
  own guidance after 8-bit landed. Sentinel fix was ~15 LOC, easy to
  re-derive from the debug doc's description if ever needed.

## Files and paths unchanged since last session

Same paths as `plan_phase4_moeflux_a3b_debug.md` — the "Handy paths"
section there is still accurate. The A3B artifacts dir was re-extracted
this session (same MLX source, updated manifest with `bits` field +
F32 A_log).

## Key commands for next session

```bash
# Canonical A17B regression smoke (no change expected)
cd /Volumes/Temp\ Backup/models/moeflux/qwen3-5-a17b-artifacts && \
  MOEFLUX_SHADERS_PATH=/Users/mdegans/Projects/moeflux/metal_infer/shaders.metal \
  /Users/mdegans/Projects/moeflux/metal_infer/infer \
  --model /Volumes/Temp\ Backup/models/moeflux/qwen3-5-a17b-root \
  --weights model_weights.bin --manifest model_weights.json --vocab vocab.bin \
  --prompt "The quick brown fox" --tokens 12 --k 10
# Expect: "jumps over the lazy dog."

# A3B current state
cd /Volumes/Temp\ Backup/models/moeflux/qwen3-6-35b-a3b-artifacts && \
  MOEFLUX_SHADERS_PATH=/Users/mdegans/Projects/moeflux/metal_infer/shaders.metal \
  /Users/mdegans/Projects/moeflux/metal_infer/infer \
  --model /Volumes/Temp\ Backup/models/moeflux/qwen3-6-35b-a3b-root \
  --weights model_weights.bin --manifest model_weights.json --vocab vocab.bin \
  --prompt "The quick brown fox" --tokens 20 --k 8
# Expect: "es are the quick brown foxes..." loop.
# Observe: layer 0 reports "gate_bits=8, seg_bits=8" in the cache line.

# drama_llama cross-backend regression (still FAILS threshold)
cargo test --test cross_backend \
  --features "llama-cpp,moeflux,moeflux-model-qwen3-6-35b-a3b" \
  -- --ignored
```

## Rebuild recipe if kernels are touched

```bash
cd /Users/mdegans/Projects/moeflux/metal_infer
make clean
make MODEL=qwen3-6-35b-a3b infer
# or: MODEL=qwen3-5-a17b infer
```

Shader is compiled at runtime (`MOEFLUX_SHADERS_PATH` points at the
.metal source), so editing the kernel only needs the `infer` re-run,
not a rebuild.
