# moeflux A3B debug — full findings

Session 2026-04-25. Started at "A3B produces degenerate output on
moeflux"; ended with bug categorized, partial workarounds proven, and
clean upstream hygiene fixes landed. Enough to pick up the real fix
later without re-deriving.

## Bugs found (in order of discovery)

### 1. A17B-hardcoded buffer literals in delta-net init
**Status: FIXED upstream** (moeflux commit `09d07a0`,
`infer: Replace A17B literals with model-variant macros`).

`infer.m:928-942, 1158-1180, reset_delta_net_state, snapshot/restore`
used hardcoded `64*128*128`, `8192`, `12288`, `64`, `NUM_LINEAR_LAYERS
45`. For A17B these match the shape macros; for any variant where
`LINEAR_NUM_V_HEADS`, `LINEAR_TOTAL_VALUE`, or `LINEAR_CONV_DIM`
differ, buffers were oversized and the array count was off. Now
flows from `LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM * LINEAR_KEY_DIM`,
`(CONV_KERNEL_SIZE - 1) * LINEAR_CONV_DIM`, etc.

Verified: A17B still produces canonical "jumps over the lazy dog"
output end-to-end, same 1.3 tok/s. No regression.

### 2. `mlp.gate` + `shared_expert_gate` are 8-bit on A3B, but moeflux reads as 4-bit
**Status: ROOT CAUSE CONFIRMED, real fix needs native 8-bit dequant in
moeflux's Metal + CPU matmul kernels. Not yet landed upstream.**

A3B's HF config declares per-layer quantization overrides:

```json
"language_model.model.layers.N.mlp.gate":         { "group_size": 64, "bits": 8 }
"language_model.model.layers.N.mlp.shared_expert_gate": { "group_size": 64, "bits": 8 }
```

All other weights are 4-bit. A17B has no such override — everything
is 4-bit.

Moeflux's `fast_batch_matvec` / `gpu_encode_batch_matvec` /
`cpu_dequant_matvec` all assume 4-bit packing: 8 nibbles per uint32.
On A3B's 8-bit data this reads completely wrong values (each uint32
contains 4 bytes = 4 values, not 8 nibbles). Result: wrong routing
gate scores → wrong top-K expert selection → garbage MoE output →
garbage propagated through 40 layers.

**Confirmed ground truth** via MLX reference inference on A3B:
`mlx_lm` generate with the same 4 prompt tokens `[760, 3841, 13477,
37550]` produces `"jumps over the lazy dog.\n\n```python..."`. So
A3B knows the pangram, moeflux was just mis-reading the gate.

### 3. A_log stored as BF16 on A3B vs F32 on A17B
**Status: PARTIAL FIX (python patcher). Moeflux still reads A_log as
`float *` unconditionally.**

Per `model_weights.json`:
- A17B: `model.layers.N.linear_attn.A_log` shape `[64]` dtype F32
- A3B:  `model.layers.N.linear_attn.A_log` shape `[32]` dtype BF16

Moeflux loads `float *A_log = get_tensor_ptr(...)`. For A3B this reads
BF16 bytes as F32 — wrong values in the delta-net decay path. Python
patcher confirmed fix (promote BF16→F32 at extract time), but the
proper fix is either (a) extract_weights.py promotes at extract time
unconditionally, or (b) moeflux reads dtype from manifest and
converts at load time.

### 4. Non-fused fallback path has latent `gpu_linear_sentinel` bug
**Status: NOT FIXED. Exposed when I forced non-fused path to bypass
the 8-bit gate issue.**

`infer.m:5183` in the fused-fallback else branch calls
`fast_dequant_matvec(oproj_w, ..., attn_out_for_oproj, ...)`. When
the layer used `gpu_linear_attn` (i.e. GPU computed the linear-attn
output into `batch_out[6]`), `attn_out_for_oproj` was set to
`&gpu_linear_sentinel` (a single `static float`). The fused path
handles this correctly by reading `batch_out[6]`; the non-fused else
does not — it runs `fast_dequant_matvec` against 4 bytes of sentinel
as if it were a full `LINEAR_TOTAL_VALUE`-sized vector. Reads garbage
past the sentinel. A17B never hits this path because the fused
conditions always succeed on real A17B runs.

Untested fix (stashed in moeflux, not committed): memcpy from
`batch_out[6]` to a local buffer when `gpu_linear_attn` is set, before
calling `fast_dequant_matvec`.

## What we've confirmed as NOT the bug

- **Shape constants in `model_variant.h`** — every A3B macro matches
  HF `text_config` exactly.
- **Embedding dequant** — manually decoded embedding rows for tokens
  760, 3841, 298, 0, 220 have sensible mean/std/range; match MLX's
  dequant output to float precision.
- **Gated attention (`attn_output_gate=True`) split layout** — matches
  transformers `modeling_qwen3_5_moe.py` per-head
  `[head_dim queries | head_dim gate]`.
- **mRoPE config difference** — A17B has the same
  `mrope_interleaved=True, mrope_section=[11,11,10]` and works with
  moeflux's standard RoPE, so this isn't the A3B-specific issue.
- **Our 8-bit dequantization is correct** — comparison against
  `mx.dequantize(W, S, B, group_size=64, bits=8)` shows
  mean diff `3.3e-08`, max abs diff `4.9e-04` (BF16 scale rounding
  only).
- **Layer-type pattern** — A3B's `text_config.layer_types` list
  matches moeflux's `(i+1) % FULL_ATTN_INTERVAL == 0` formula exactly.
- **Manifest tensor shapes** — every A3B tensor has 2× or 1× of A17B
  dimensions in predictable ways tied to `HIDDEN_DIM` / `NUM_EXPERTS`
  / `LINEAR_NUM_V_HEADS`. Only `mlp.gate.weight` cols don't halve
  (stays 512 U32), because of the 4→8 bit quantization override.

## Workarounds attempted

1. **4-bit re-quantization of 8-bit gates** (`artifacts-patched/`):
   dequant 8-bit → re-quant 4-bit at extract time. Output becomes
   "The quick brown foxes are the quick brown foxes..." — coherent
   English but not canonical. 14.6% mean relative weight error on
   the gate; enough to shift greedy's argmax.
2. **Dense F32 gate + moeflux dispatch path fix** (`artifacts-densegate/`):
   dequant 8-bit → store as dense F32. Add `cpu_dense_f32_matvec` and
   route gate through it when manifest dtype is F32. Forces non-fused
   fallback path, which hits bug #4 → degenerate output again. If
   bug #4 were also fixed, this path would work (pending confirmation).

## Real fix (recommended)

**Add native 8-bit dequant support in moeflux.** Scope:
- `metal_infer/shaders.metal`: new kernel variant or parameterized
  bits for `cpu_dequant_matvec` / `matvec_v3` / `matvec_fast`. Loop
  4 values per uint32 instead of 8 when bits=8.
- `infer.m`: add `uint32_t bits` field to `BatchMatvecSpec`, thread
  it through the dispatch, default 4 for backward compat. Read the
  per-tensor dtype / config quantization override from the manifest
  at layer-cache build to set the right value.
- `extract_weights.py`: include the quantization-override dictionary
  in the manifest's `config` block so `infer.m` can read it without
  parsing HF config.json.

Estimated ~300 LOC including Metal kernel duplication. A day's work.

## Alternative fix (simpler, extract-time)

If we don't want 8-bit in moeflux, the extractor can pre-dequantize
the 8-bit gates to dense F32 and emit them that way. Combined with a
small F32 matvec path in moeflux (already prototyped in the stashed
change) and the `gpu_linear_sentinel` fix, this works. Code size is
similar; the tradeoff is ~90 MB extra per A3B deployment (dense F32
gates are 4× the packed 8-bit size) vs a cleaner moeflux codebase.

## drama_llama side

**Unchanged from prior entry in `plan_phase4_gate4_findings.md`.**
Implementation complete, cross-backend test committed as `#[ignore]`
(current argmax agreement 3% with 4-bit-patched artifacts — up from
0% with raw artifacts, but below the 95% threshold because of the
re-quant precision loss). Once moeflux handles 8-bit properly, the
cross-backend test should pass without needing Python artifact
patching.

## Handy paths

- moeflux repo: `/Users/mdegans/Projects/moeflux`
- A17B raw MLX: `/Volumes/Temp Backup/models/mlx-community/Qwen3.5-397B-A17B-4bit`
- A17B extracted: `/Volumes/Temp Backup/models/moeflux/qwen3-5-a17b-{artifacts,root,packed_experts}`
- A3B raw MLX: `/Volumes/Temp Backup/models/moeflux/qwen3-6-35b-a3b-mlx-4bit`
- A3B extracted (original): `/Volumes/Temp Backup/models/moeflux/qwen3-6-35b-a3b-artifacts`
- A3B 4-bit-regate patch: `/Volumes/Temp Backup/models/moeflux/qwen3-6-35b-a3b-artifacts-patched`
- A3B F32-dense-gate patch: `/Volumes/Temp Backup/models/moeflux/qwen3-6-35b-a3b-artifacts-densegate`
- Extract script: `/tmp/extract_a17b.sh` (re-runnable, idempotent)
- A17B HF config fetched in session — available at
  `mlx-community/Qwen3.5-397B-A17B-4bit` on HuggingFace

## Loose ends in moeflux

Stashed experimental changes (`git stash show stash@{0}`):
- `cpu_dense_f32_matvec` helper + `is_tensor_dense_f32` predicate
- `LayerWeightCache.gate_dense` / `.seg_dense` fields
- Dense-F32 dispatch in `moe_forward` and `fused_layer_forward`'s else
- `gpu_linear_sentinel` fallback fix

Drop the stash once the real 8-bit fix lands — most of this code
becomes obsolete.

## Key commands for next session

```bash
# Test A17B works (canonical):
cd /Volumes/Temp\ Backup/models/moeflux/qwen3-5-a17b-artifacts && \
  MOEFLUX_SHADERS_PATH=/Users/mdegans/Projects/moeflux/metal_infer/shaders.metal \
  /Users/mdegans/Projects/moeflux/metal_infer/infer \
  --model /Volumes/Temp\ Backup/models/moeflux/qwen3-5-a17b-root \
  --weights model_weights.bin --manifest model_weights.json --vocab vocab.bin \
  --prompt "The quick brown fox" --tokens 16 --k 10

# Test A3B (currently broken):
cd /Volumes/Temp\ Backup/models/moeflux/qwen3-6-35b-a3b-artifacts && \
  MOEFLUX_SHADERS_PATH=/Users/mdegans/Projects/moeflux/metal_infer/shaders.metal \
  /Users/mdegans/Projects/moeflux/metal_infer/infer \
  --model /Volumes/Temp\ Backup/models/moeflux/qwen3-6-35b-a3b-root \
  --weights model_weights.bin --manifest model_weights.json --vocab vocab.bin \
  --prompt "The quick brown fox" --tokens 16 --k 8

# MLX ground truth (needs uv):
uv run --with mlx --with mlx-lm python3 -c "
from mlx_lm import load, generate
m, t = load('/Volumes/Temp Backup/models/moeflux/qwen3-6-35b-a3b-mlx-4bit')
ids = t.encode('The quick brown fox', add_special_tokens=False)
print(generate(m, t, prompt=ids, max_tokens=16, sampler=None, verbose=False))
"
```
