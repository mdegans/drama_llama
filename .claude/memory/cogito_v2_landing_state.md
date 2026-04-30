# Cogito-V2 671B landing — continuation state

Captured 2026-04-30 EOS. Mid-arc through the session-plan at
`~/.claude/plans/squishy-wibbling-milner.md`.

## What's landed (commits)

**moeflux** (`~/Projects/moeflux`, branch `main`):
- `de2ba40` `variants: add Cogito-V2-Preview-671B feature flag`
  — extended Variant struct (AttnKind, RouterKind, RopeKind,
  SharedExpertGate, MlpKind enums; ~15 new fields). VARIANT block
  for `model-cogito-v2-671b`. Static asserts gated on attn_kind.
  moeflux-sys's build.rs now skips C compile gracefully when only
  Rust-side variants are enabled. C-oracle test files file-level
  cfg-gated. 3 variant unit tests green.
- `80375ae` `tools: convert_cogito_v2.py — DeepSeek-V3 / Cogito-V2
  weight converter` — self-contained MLX-4bit → moeflux on-disk
  converter emitting the canonical
  `parent/{mlx,artifacts,root/packed_experts}/` layout.
- `c8d74ce` `moe_router + rope: noaux_tc routing + YaRN math for
  DeepSeek-V3` — `noaux_tc_router_cpu` + 6 unit tests across both
  modules; YaRN helpers (`yarn_get_mscale`,
  `yarn_find_correction_range`, `compute_yarn_inv_freq`,
  `apply_rotary_emb_yarn`) covering the math half of MLA's RoPE.
  25 lib tests green on both Cogito and Qwen variants.

**drama_llama** (`~/Projects/drama_llama`, branch `v0.8.0`):
- `a7fc646` `Cargo: add moeflux-model-cogito-v2-671b feature;
  path-dep on local moeflux` — feature forwarding + architecture
  audit memo.
- `3b6391d` `memory: future-work note for unified MLX -> moeflux
  converter`.

## On-disk artifacts

Conversion ran with `tools/convert_cogito_v2.py` against
`/Volumes/HF Models/models/hf/mlx-community/cogito-v2-preview-deepseek-671B-MoE-4bit/`.
Output at `/Volumes/Temp Backup/models/blallama/cogito-v2-671b/`:
- `mlx/` — tokenizer.json, tokenizer_config.json,
  special_tokens_map.json, chat_template.jinja, config.json
- `artifacts/model_weights.{bin,json}` — non-expert tensors with
  the manifest `weight_file.rs` reads (1831 tensors, ~10 GB)
- `root/packed_experts/layer_NN.bin` — 58 MoE layers (3..60),
  ~6 GB each, ~340 GB total. layer_00.bin through layer_02.bin
  do NOT exist (those layers are dense MLPs whose weights live in
  the artifacts blob).

Conversion was running at session end; if you find the directory
incomplete (missing layer_60.bin, partial last file), re-run the
converter with `--skip-artifacts` to redo the experts only.

## What's pending

The remaining work (in suggested order):

### 1. MLA forward kernel (Phase C)

Stub at `~/Projects/moeflux/crates/moeflux/src/riir/mla_attn_cpu.rs`
returns `MlaForwardError::NotImplemented`. The forward shape is
documented in the module-level docstring and in
`~/Projects/drama_llama/.claude/memory/cogito_v2_architecture.md`.

Concrete subtasks:

**1a. CPU 4-bit dequant primitive.** moeflux's existing kernels are
GPU-side (`gpu_matvec.rs` etc.). The CPU MLA path needs a function
that takes `(u32_packed_weights, bf16_scales, bf16_biases, shape)
-> Vec<f32>` so we can materialize the projection matrices. Reuse
the `dequantize_*` logic from `expert_forward.rs` if it has a CPU
path; otherwise port the MLX 4-bit dequant: groups of 64 weights
share one BF16 scale + bias.

**1b. CPU matmul/matvec.** `f32_vec @ f32_matrix` for the
projections (q_a, q_b, kv_a, kv_b, o_proj). Naive triple loop is
fine; perf isn't load-bearing on CPU MLA.

**1c. MLA inner SDPA.** Per the docstring's pseudocode. Critical
detail: the softmax_scale is `(1/sqrt(qk_head_dim)) * mscale²`
where `mscale = yarn_get_mscale_full(factor, mscale, mscale_all_dim)`.
For `factor=1` mscale is 1.0 and this collapses to vanilla SDPA.

**1d. Wire MlaKvCache append + read.** `state.rs::MlaKvCache` is
defined; integrate it into `LayerState` (currently just `FullAttn` /
`LinearAttn`) — add `MlaAttn(MlaKvCache)` variant and update
`alloc_layer_states`, `truncate`, `clear_all`, `is_full`,
`pos_max`. **This will churn many match arms** — budget for the
spread.

### 2. Dense MLP path (Phase E)

For Cogito-V2 layers 0-2, run a single SwiGLU MLP with
`intermediate=18432` (no routing, no shared expert). Existing
expert-FFN kernel can be repurposed with a different shape parameter.

Tensor names: `model.layers.{0..2}.mlp.{gate,up,down}_proj.{weight,scales,biases}`.

### 3. MoE block composition (Phase E)

`post_attention_tail` in `linear_attn_forward.rs:~400+` does the
composition for Qwen. For DeepSeek:
- Use `noaux_tc_router_cpu` (already landed) instead of
  `moe_router_cpu`.
- Read `mlp.gate.weight` and `mlp.gate.e_score_correction_bias`
  from artifacts.
- Weighted sum over selected routed experts (read from
  `root/packed_experts/layer_NN.bin`).
- Add `shared_experts(hidden)` UNCONDITIONALLY (no sigmoid gate —
  contrast with Qwen's gated path; selected via
  `Variant::shared_expert_gate == Unscaled`).

### 4. Forward orchestration (Phase G)

`step_internal` in `mod.rs:~1045` dispatches to GPU full-attn or
linear-attn. For first-run Cogito, the simplest landing is a
parallel CPU-only entry point that bypasses the GPU pipeline
entirely:

```rust
#[cfg(feature = "model-cogito-v2-671b")]
fn step_internal_mla_cpu(&mut self, token: i32, pos: i32) -> Vec<f32> {
    // 1. embed
    // 2. for each layer: pre-attn norm → mla_attn_layer_forward_cpu
    //    → residual → post-attn norm → mlp_forward (dense or MoE)
    //    → residual
    // 3. final norm → lm_head
    // 4. return logits
}
```

Then in the public eval entry, dispatch on `VARIANT.attn_kind`:
- `Gqa` → existing `step_internal`
- `Mla` → new CPU path

This is the "first run cleanly separated, don't touch the GPU
pipeline" approach. GPU port for MLA is a separate slice.

### 5. Tokenizer + chat template (Phase G)

drama_llama's blallama already loads `mlx/tokenizer.json` via the
`tokenizers` crate. Verify special-token IDs match the variant
config (`bos=0`, `eos=1`). Apply `mlx/chat_template.jinja` via
misanthropic's renderer; spot-check a hello prompt renders
non-empty.

### 6. First forward pass (Phase H)

```bash
cargo build --bin blallama --features \
    "webchat,cli,moeflux,moeflux-model-cogito-v2-671b"

blallama --backend moeflux \
         --model-path /Volumes/Temp\ Backup/models/blallama/
```

Send "Hello. How are you?" via `/v1/messages`. Watch for NaN/Inf;
check first ~20 tokens are coherent English. Expected throughput
~1 tok/s warm.

If the output is garbage, bisect order: tokenizer → embed → first
layer's MLA output (compare to a tiny PyTorch / `modeling_deepseek.py`
reference at single-layer scale) → router → final logits.

## Validation strategy reminder

No oracle for the full 671B. Methodology:
- noaux_tc unit test (landed, 2 tests)
- YaRN sanity tests (landed, 4 tests)
- Output vibes on first forward
- Future: DeepSeek-V2-Lite (16B, MLA-shaped) as a small-model MLA
  oracle; parked for a follow-up correctness sweep, not blocking
  first run.

## Pointers

- Architecture audit (load-bearing for kernel work):
  `~/Projects/drama_llama/.claude/memory/cogito_v2_architecture.md`
- Approved session plan:
  `~/.claude/plans/squishy-wibbling-milner.md`
- moeflux RIIR strategy (parent context):
  `~/Projects/drama_llama/.claude/memory/riir_moeflux_strategy.md`
- DeepSeek reference impl (canonical math):
  `/Volumes/HF Models/models/hf/mlx-community/cogito-v2-preview-deepseek-671B-MoE-4bit/modeling_deepseek.py`

## Dependency / path notes

drama_llama's Cargo.toml is path-pointed at `~/Projects/moeflux/crates/moeflux`
while the work is in flight. Once the variant ships in moeflux
0.1.0-pre.3, switch back to `version = "=0.1.0-pre.3"` (per Mike's
"we'll publish when I get home" note this session).
