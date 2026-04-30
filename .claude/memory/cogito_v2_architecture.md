# Cogito-V2-Preview-671B / DeepSeek-V3 architecture audit

Captured 2026-04-30 during the moeflux Cogito-V2 enablement work.
Source: `mlx-community/cogito-v2-preview-deepseek-671B-MoE-4bit`'s
`config.json`, `modeling_deepseek.py`, and tensor index. This is
durable reference for kernel work — kept in-repo per
`feedback_in_repo_memory.md`.

## Provenance

- **Cogito-V2-Preview** is a fine-tune of DeepSeek-V3-base by Deep
  Cogito. Architecture is unchanged from DeepSeek-V3.
- DeepSeek-V4-Pro 1.6T inherits the same V3 architecture family —
  the kernel work for Cogito-V2 stages V4 directly.
- MLX 4-bit quant: `group_size=64`, `.weight + .scales + .biases`
  triplet per matrix. Same MLX format moeflux already handles for
  Qwen-MoE, so the dequant primitives carry over unchanged.

## Key dimensions

| param | value | notes |
|---|---:|---|
| `num_hidden_layers` | 61 | layers 0-2 dense MLP, 3-60 MoE |
| `hidden_size` | 7168 | model width |
| `vocab_size` | 128815 | DeepSeek tokenizer (BPE) |
| `num_attention_heads` | 128 | per-layer Q heads |
| `qk_nope_head_dim` | 128 | non-rotated Q/K split |
| `qk_rope_head_dim` | 64 | rotated Q/K split (RoPE only here) |
| `qk_head_dim` | 192 | = nope + rope, full Q/K head |
| `v_head_dim` | 128 | V head dim (no rope split) |
| `q_lora_rank` | 1536 | Q LoRA bottleneck |
| `kv_lora_rank` | 512 | KV latent dim cached per token |
| `n_routed_experts` | 256 | MoE: 32 per group × 8 groups |
| `n_shared_experts` | 1 | always-on shared expert |
| `num_experts_per_tok` | 8 | top-K |
| `n_group` | 8 | router groups |
| `topk_group` | 4 | groups selected per token |
| `first_k_dense_replace` | 3 | first 3 layers are dense |
| `dense_intermediate_size` | 18432 | dense MLP width |
| `moe_intermediate_size` | 2048 | per-expert FFN width |
| `routed_scaling_factor` | 2.5 | post-renorm weight multiplier |
| `rope_theta` | 1e4 | YaRN base (Qwen used 1e7) |
| YaRN factor / orig_max | 40 / 4096 | extends to 163840 ctx |
| YaRN beta_fast / beta_slow | 32 / 1 | smooth-ramp bounds |
| YaRN mscale / mscale_all_dim | 1.0 / 1.0 | dual-application factors |

## What's new vs Qwen-MoE (moeflux's existing path)

### Multi-head Latent Attention (MLA) — load-bearing

Per-layer tensor names:
```
self_attn.q_a_proj           [7168, 1536]    Q LoRA down
self_attn.q_a_layernorm      [1536]          RMSNorm on Q latent
self_attn.q_b_proj           [1536, 24576]   Q LoRA up (128 heads × 192)
self_attn.kv_a_proj_with_mqa [7168, 576]     KV down + RoPE-K
self_attn.kv_a_layernorm     [512]           RMSNorm on KV latent
self_attn.kv_b_proj          [512, 32768]    KV up (128 heads × 256)
self_attn.o_proj             [16384, 7168]   output proj (128 heads × 128 V)
```

Forward (single-token decode, position `pos`):
```
q_lat   = q_a_proj @ h                        # [1536]
q_lat   = q_a_layernorm(q_lat)
q       = q_b_proj @ q_lat                    # [128 heads, 192]
q_nope, q_pe = split(q, [128, 64])

kv_pre  = kv_a_proj_with_mqa @ h              # [576]
kv_lat, k_pe = split(kv_pre, [512, 64])       # k_pe: shared rope-K
kv_lat  = kv_a_layernorm(kv_lat)
k_pe    = yarn_rope(k_pe, pos)                # [64], broadcast to all heads
q_pe    = yarn_rope(q_pe, pos)                # per-head [64]

# Append (kv_lat, k_pe) to MlaKvCache @ position pos.

# For attention against cached positions [0, pos]:
#   For each cached j: kv_lat_j -> kv_b_proj -> [128 heads, k_nope_j 128 + v_j 128]
#                      k_j = [k_nope_j | broadcast(k_pe_j)]   # [128 heads, 192]
#   score = Σ_j  softmax( q · k_j  *  (1/sqrt(192)) * yarn_mscale²  ) · v_j
o_pre   = concat_heads(score)                 # [16384]
out     = o_proj @ o_pre                      # [7168]
```

KV cache shape: `MlaKvCache { latent_cache[T, 512], rope_k_cache[T, 64] }`.
576 floats per token total — **~28× compression** vs GQA's full K/V
(128 heads × 128 dim × 2 = 32768).

### noaux_tc routing

Per MoE layer (≥ layer 3):
```
gate.weight                  [256, 7168]
gate.e_score_correction_bias [256]
```

Algorithm:
```
s_orig    = sigmoid(h @ gate.weight.T)        # [256]
s_biased  = s_orig + e_score_correction_bias  # [256]

# Group-limit selection (n_group=8, topk_group=4):
group_score = sum(top_2(reshape(s_biased, [8, 32]))) # [8] sum-of-top-2 per group
selected_groups = top_4(group_score)
mask    = scatter_zero_outside_groups(selected_groups)  # [256] zeros non-selected
s_masked = s_biased * mask

# Global top-K from masked space:
indices  = top_8(s_masked)                    # 8 expert IDs
weights  = s_orig.gather(indices)             # ORIGINAL (non-biased) sigmoid scores
weights /= sum(weights)                       # renormalize
weights *= 2.5                                # routed_scaling_factor
```

Critical: `e_score_correction_bias` is added ONLY for selection;
final renormalized weights use the un-biased sigmoid output.

### Shared expert composition

Per MoE layer:
```
shared_experts.{gate,up,down}_proj            # one always-on expert
mlp.switch_mlp.{0..255}.{gate,up,down}_proj   # 256 routed experts (MLX-stacked)
```

```
routed_out = Σ_i  weights[i] * expert[indices[i]](h)   # 8 routed experts
shared_out = shared_experts(h)                          # 1 expert, always
output     = routed_out + shared_out                   # SIMPLE SUM, no gate
```

Differs from Qwen-MoE which scales `shared_out` by
`sigmoid(shared_gate_score)`. moeflux's
`SharedExpertGate::Unscaled` variant gates this for Cogito.

### Dense-first-K MLP

Layers 0-2 use a single dense SwiGLU MLP with `intermediate=18432`
(no expert routing, no shared expert). Tensor names:
```
mlp.gate_proj                [7168, 18432]
mlp.up_proj                  [7168, 18432]
mlp.down_proj                [18432, 7168]
```

Same SwiGLU shape as moeflux's existing per-expert FFN, just larger.
Reuse the existing kernel with shape parameters.

### YaRN RoPE

Math (pre-cached at `RsCtx::open`):
```
freq_extra = 1 / base^(2i/dim)                # original freqs
freq_inter = 1 / (factor * base)^(2i/dim)     # scaled freqs

low, high  = yarn_find_correction_range(beta_fast=32, beta_slow=1,
                                        dim=64, base=1e4, max=4096)
ramp       = yarn_linear_ramp_mask(low, high, dim/2)   # smooth interpolation
inv_freq   = freq_inter * (1 - ramp) + freq_extra * ramp

mscale     = yarn_get_mscale(factor, mscale)           # ≈ 0.1*log(factor)+1
mscale_div = yarn_get_mscale(factor, mscale_all_dim)
_mscale    = mscale / mscale_div                        # final cos/sin scale

cos_table  = cos(pos * inv_freq) * _mscale             # cached
sin_table  = sin(pos * inv_freq) * _mscale
```

Applied as `apply_rotary_emb(q_pe, k_pe, cos, sin)` at use time.
Additionally, the attention softmax scale gets `* _mscale²`:
```
softmax_scale = (1 / sqrt(192)) * _mscale * _mscale
```

`yarn_get_mscale(scale, mscale)`:
```
if scale <= 1: return 1.0
else: return 0.1 * mscale * ln(scale) + 1.0
```

### MTP head (ignored)

Config has `num_nextn_predict_layers: 1` — DeepSeek-V3's
multi-token-prediction head used during training. For inference,
the model class only uses `lm_head`. moeflux ignores the MTP
weights entirely.

## What carries over from Qwen-MoE (zero rewrite)

- 4-bit MLX dequant primitives — same group_size=64 layout
- RMSNorm kernel (CPU + GPU)
- Embedding lookup, lm_head linear projection
- SwiGLU FFN (gate × up → SiLU → down) — used for dense, shared,
  and routed experts at different shapes
- MoE-block dispatch shape (router → routed expert weighted sum
  → add shared); only the inner formulas change
- Metal pipeline cache, command-buffer machinery
- Expert-streaming infrastructure (per-layer .bin files,
  prefetch state machine, deferred ring) — load layout extends
  cleanly since first 3 layers just have empty expert files
  (or skip layer files entirely for layers `< first_k_dense_replace`)

## Tokenizer specials

```
bos = 0   (<｜begin▁of▁sentence｜>)
eos = 1   (<｜end▁of▁sentence｜>)
pad = 2   (<｜▁pad▁｜>)
<｜User｜>     = 128803
<｜Assistant｜> = 128804
<｜tool▁calls▁begin｜>  = 128806
... and 11 more <｜tool▁*｜> tokens through 128814
```

No dedicated `<think>` token — chat template uses inline
`<think>\n` text (ASCII), which the BPE tokenizer splits into
multiple regular tokens. moeflux variant config sets
`think_start_token = think_end_token = -1`.

## On-disk conversion targets

Per `moeflux_disk_convention.md`, target layout under
`/Volumes/Temp Backup/models/blallama/cogito-v2-671b/`:

```
parent/
├── mlx/
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   ├── chat_template.jinja
│   └── config.json
├── artifacts/
│   ├── model_weights.bin     # MLA-attn + dense-MLP + shared-expert
│   │                            + gate (with e_score_correction_bias)
│   │                            + embedding + lm_head + final norm
│   └── model_weights.json    # tensor-name → offset/size manifest
└── root/
    └── packed_experts/
        ├── layer_03.bin      # 256 routed experts × 9 components
        ├── layer_04.bin      # (skip 0-2 — dense, no experts)
        ├── ...
        ├── layer_60.bin
        └── layout.json
```

## Validation strategy (no oracle)

The 671B model is too large to fit alongside any PyTorch / MLX
reference on 96 GB. Validation is layered:

1. **Synthetic noaux_tc unit test** — small router input vs NumPy
   reference computed inline.
2. **YaRN sanity test** — `factor=1` collapses to vanilla RoPE.
3. **Static-asserts on the variant struct** — catch dimensional
   mistakes at compile time.
4. **Output coherence** — model card sample prompts; first ~20
   tokens of greedy generation should be sensible English.
5. **Future**: DeepSeek-V2-Lite (16B, MLA-shaped, fits in 96 GB)
   as a kernel-level oracle for the MLA path. Parked as
   future work — the 671B run can ship without this, but the
   correctness sweep should circle back when convenient.

## Pointers

- `modeling_deepseek.py` (in the model dir): canonical reference
  for MLA forward, noaux_tc routing, YaRN math
- `~/Projects/drama_llama/.claude/memory/moeflux_disk_convention.md`:
  on-disk layout convention
- `~/Projects/drama_llama/.claude/memory/riir_moeflux_strategy.md`:
  the parent RIIR strategy this builds on
- `~/.claude/plans/squishy-wibbling-milner.md`: the approved
  end-to-end plan for this session's Cogito-V2 enablement
