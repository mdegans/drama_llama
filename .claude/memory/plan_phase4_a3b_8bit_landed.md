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

**Additional rule-outs this session (don't re-test):**

- **`MTLMathModeSafe`** produces bit-identical output to
  `MTLMathModeFast` on A3B for this prompt. So FP fast-math
  reassociation across 40 layers is **not** the source of the
  divergence. Reverted to Fast for performance.
- **8-bit dispatch is actually happening.** A diagnostic counter
  (temporarily added, then reverted) showed 80 dispatches per
  token = 40 layers × 2 tensors (gate + seg) × 1 token via
  `ctx->matvec_8bit_v3`. 100% hit rate; no silent 4-bit fallback.
  First dispatch is in_dim=2048 out_dim=256 (routing gate) as
  expected.
- **No other dtype divergences** between A3B and A17B manifests
  beyond A_log. I diffed all non-quantized tensors; they're BF16
  on both variants with parallel structure.

**Additional rule-outs 2026-04-27 (don't re-test):**

- **Metal 8-bit kernel numerically matches CPU 8-bit reference to
  ~1e-5 on every one of the 80 8-bit dispatches per token (40 layers
  × 2 tensors: `mlp.gate` + `shared_expert_gate`).** argmax matches
  on all 80 dispatches; max_abs_diff stays in [0, 2.4e-5] across all
  layers. So `dequant_matvec_8bit_v3` is effectively correct
  (within fp32 ULP noise across ~2048 accumulations). The kernel is
  not the bug.

- **All A3B shape macros in `model_variant.h` match HF
  `text_config` bit-for-bit** (fresh verify): `hidden_size=2048`,
  `num_hidden_layers=40`, `num_attention_heads=16`,
  `num_key_value_heads=2`, `head_dim=256`, `vocab_size=248320`,
  `num_experts=256`, `num_experts_per_tok=8`,
  `moe_intermediate_size=512`, `shared_expert_intermediate_size=512`,
  `full_attention_interval=4`, `linear_num_value_heads=32`,
  `linear_num_key_heads=16`, `linear_{key,value}_head_dim=128`,
  `linear_conv_kernel_dim=4`, `partial_rotary_factor=0.25`,
  `rms_norm_eps=1e-6`.

- **RoPE params match A17B**: `rope_theta=10000000`,
  `mrope_interleaved=True`, `mrope_section=[11, 11, 10]`,
  `partial_rotary_factor=0.25`. Already ruled out last session; this
  session re-verified.

- **No inference-impacting config deltas A3B vs A17B**:
  `tie_word_embeddings=False` on both (A3B declares explicitly,
  A17B defaults). A17B-only `mlp_only_layers=[]` is empty (no
  effect). A3B-only `output_router_logits` is a training flag.
  A3B-only `bos_token_id`/`pad_token_id` affect tokenization
  only, not forward pass. The remaining A3B/A17B differences are
  all shape scalings (all already macro-driven and macros verified
  correct).

  *How we tested:* Added a `MOEFLUX_DIFF_8BIT` env-gated diagnostic
  to `gpu_flush_batch_results` (in `infer.m` after the memcpy loop).
  For each 8-bit spec, compute `cpu_dequant_matvec` against
  `[buf_input contents]` and diff vs `s->out_cpu` (the just-flushed
  Metal result). At flush time, `buf_input` still contains Enc 4's
  rms_norm output — the exact input Metal's 8-bit dispatch consumed
  during cmdbuf execution — so the comparison is sound. Reverted
  after measuring; pattern is simple to re-add if needed.

- **`MOEFLUX_FORCE_CPU_8BIT` env-var approach (make 8-bit specs
  bypass Metal and compute on CPU in-place) does NOT work as a
  single-call rewrite.** Inside `gpu_encode_batch_matvec`, at
  encode time, `buf_input` is stale — the preceding `rms_norm_apply`
  encode (Enc 4 in the fused-layer flow) hasn't executed yet, so
  the CPU dequant sees old hidden state. Output looked suspiciously
  coherent ("The quick brown fox is the slow fox. <EOS>") because
  stale-input garbage happened to be plausible English, which almost
  misled me. **Do not reach for this trick again** without
  breaking the fused cmdbuf: you'd have to commit+wait before each
  8-bit spec, which defeats the fused-path batching.

**Still not tested** (order of decreasing likelihood):

1. **Qwen3.6-specific architectural detail we haven't surfaced.**
   Remaining suspects: gated attention's scale factor, some detail
   in delta-net's beta gating, shared-expert combine weights.
   Rather than chase individually, the highest-signal next move is
   to **numerically compare moeflux's layer-0 output (or attention
   output) against MLX's layer-0 equivalent** on the same 4 prompt
   tokens. If they match, attention is correct → bug is in MLP/MoE
   post-attention. If they differ, localize within attention /
   linear-attn / RoPE.

2. **Top-K + softmax post-gate logic.** Gate logits are correct
   (we just ruled that out). But top-K selection (K=8 of 256) and
   any per-expert scaling (softmax across top-K, renorm) is
   A3B-same-as-A17B code. Worth a printf at layer 0: which 8
   expert indices + weights does moeflux pick? MLX should pick the
   same set to ~argmax precision.

3. **Bug #4 from the debug doc** — `gpu_linear_sentinel` fallback.
   Should not be hit on fused-path A3B runs. A one-line `fprintf`
   at the non-fused else branch at layer 0 would confirm.

4. **Re-examine `model_variant.h` A3B macros one more time with
   fresh eyes** — there's still some chance an off-by-2 or half-
   size constant slipped past. Specifically, anything involving
   `NUM_ATTN_HEADS`, `HEAD_DIM`, `LINEAR_NUM_V_HEADS`, or
   `LINEAR_VALUE_DIM` that flows into attention math is worth a
   paranoid cross-check against HF `text_config`.

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
