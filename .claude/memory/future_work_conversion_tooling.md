# Future work — conversion tooling consolidation

Captured 2026-04-30 during Cogito-V2 enablement.

## Current state

Three Python scripts live in `~/Projects/moeflux/`, with overlapping
purposes:

- `crates/moeflux-sys/metal_infer/extract_weights.py` — original
  Qwen-MoE-flavored artifacts extractor. Hard-codes Qwen-specific
  config keys (`linear_num_value_heads`, `linear_key_head_dim`,
  etc.) that don't exist in DeepSeek-V3 configs.
- `tools/gen_expert_index.py` — Qwen-flavored expert-index builder.
  Doesn't handle dense-first-K layers (no `switch_mlp` tensors for
  layers 0..first_k_dense_replace).
- `repack_experts.py` — variant-agnostic. Reads `expert_index.json`
  produced by `gen_expert_index.py` and packs per-layer `.bin`
  files. Already tolerates missing layers.
- `tools/convert_cogito_v2.py` (new this session) — purpose-built
  end-to-end DeepSeek-V3 / Cogito-V2 converter. Self-contained,
  doesn't reuse the existing scripts.

## What "flexible" looks like

A single `tools/convert.py` that reads any MLX-converted MoE model
and emits the canonical `parent/{mlx,artifacts,root}/` layout, by:

1. **Architecture detection** — read `config.json::architectures[0]`
   (e.g. `Qwen3MoeForCausalLM`, `DeepseekV3ForCausalLM`). Map to a
   converter class with the right config-key knowledge.
2. **Generic tensor categorization** — same as
   `convert_cogito_v2.py:categorize_tensors`: walk MLX safetensors,
   classify by regex into experts vs artifacts.
3. **Per-arch config block** — each arch knows which of its config
   keys to surface in `model_weights.json::config`.
4. **Variant string** — emit a `variant` field that maps directly
   to a moeflux feature flag.

## Why not now

`convert_cogito_v2.py` is straightforward to write and lets us
move on to the kernel work. Generalization is best done after
*one* additional architecture lands (DeepSeek-V4-Pro 1.6T, or any
other future MoE), at which point the abstraction has two real
consumers and we can extract the right shape.

Right now the cost of a unified converter would land before we
have evidence about what the abstraction needs to look like.

## Steps when ready

1. Move shared helpers (`parse_safetensors_header`,
   `EXPERT_TENSOR_RE`, `categorize_tensors`) into
   `tools/_common.py`.
2. Convert `extract_weights.py` and `convert_cogito_v2.py` into
   thin wrappers over `tools/convert.py --variant <slug>`.
3. Delete `gen_expert_index.py` (subsumed by inline categorization
   in `convert.py`).
4. Update `repack_experts.py`'s docstring to point at the unified
   tool, or fold its functionality in.

Estimated size: half a session.
