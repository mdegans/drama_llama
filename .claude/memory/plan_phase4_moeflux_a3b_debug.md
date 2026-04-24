# moeflux A3B degenerate-output debug notes

Session 2026-04-25 continuation. The drama_llama cross-backend test
found moeflux's Qwen3.6-35B-A3B output is garbage; this file captures
what we've ruled in/out so the next session (or the next human) doesn't
re-derive.

## The failure, reproduced upstream

moeflux's own `metal_infer/infer` binary (pre-API, reference path)
produces the **exact same degenerate cycle** on the same A3B
artifacts + same prompt "The quick brown fox":

```
[prompt] 4 tokens: 760 3841 13477 37550
ro a . The  rome . The  rome . The  rome . The ...
```

So the bug is **not** in drama_llama's wiring, **not** in our
`mf_eval_*` API wrappers, **not** in HF-vs-bpe tokenizer mismatch.
It's in moeflux's core inference path or A3B weight extraction.

## Historical receipts (important context)

`~/Projects/moeflux/results.tsv` has dozens of entries with **coherent
A17B output** — "Classic Coin Analogy…", "math(255=15*17 CORRECT)",
"to use a teapot." — from before the A3B port landed. So moeflux's
full pipeline has produced correct text historically, for A17B. **Zero
A3B entries** in the log. The A3B port was gated only by "smoke test
PASS" which in smoke.c means "logits aren't NaN and change between
steps" (synthetic tokens `[1,100,500,1000]` fed in).

## What we've checked (ruled out)

- **Shape constants.** Every macro in `model_variant.h`'s A3B block
  (`HIDDEN_DIM=2048`, `NUM_LAYERS=40`, `NUM_ATTN_HEADS=16`,
  `NUM_KV_HEADS=2`, `HEAD_DIM=256`, `NUM_EXPERTS=256`,
  `NUM_EXPERTS_PER_TOK=8`, `MOE_INTERMEDIATE=512`,
  `SHARED_INTERMEDIATE=512`, `FULL_ATTN_INTERVAL=4`,
  `LINEAR_NUM_V_HEADS=32`, `LINEAR_NUM_K_HEADS=16`) matches
  `config.json.text_config` exactly.
- **Layer-type pattern.** moeflux's `(i+1) % FULL_ATTN_INTERVAL == 0`
  matches HF's `text_config.layer_types` list for A3B — full-attn at
  indices 3,7,11,15,19,23,27,31,35,39; everything else linear.
- **Manifest tensor shapes + sizes** match safetensors headers — no
  extraction corruption. 1397 tensors, including 10× `self_attn.*`,
  30× `linear_attn.*`, all 40× `mlp.gate/shared_expert*` +
  `input_layernorm`, `post_attention_layernorm`. `model.norm.weight`,
  `embed_tokens.*`, `lm_head.*` all present.
- **Embedding dequant math correct.** Manually dequantized embeddings
  for tokens 760, 3841, 298, 0, 220 — all look sane: mean ≈ 0, std ≈
  0.01, no NaN, different tokens → different values. Shape `[248320,
  256]` U32 with `[248320, 32]` BF16 scales+biases makes sense
  (group_size=64, 32 groups × 64 = 2048 dims).
- **Gated attention (`attn_output_gate=True`) split layout** matches
  transformers `modeling_qwen3_5_moe.py:661-663`. q_proj out shape
  `[8192, 256]` = 2 × num_heads × head_dim. Per-head layout `[head_dim
  queries | head_dim gate]` — `infer.m:2280-2284` splits exactly this
  way (`src = q_proj_out + h*(2*HEAD_DIM); memcpy(q, src, HEAD_DIM);
  memcpy(q_gate, src+HEAD_DIM, HEAD_DIM)`).
- **Gate application timing** matches reference — sigmoid applied to
  `attn_out` before `o_proj` (`infer.m:2377-2394`,
  modeling_qwen3_5_moe.py:691-694).
- **mRoPE hypothesis** ruled out. A17B config **also** has
  `rope_parameters.mrope_interleaved=True, mrope_section=[11,11,10]`
  and A17B works with moeflux's standard-RoPE implementation, so this
  isn't A3B-specific.

## What we haven't checked (hypothesis space remaining)

1. **Linear-attention path with A3B's `LINEAR_NUM_V_HEADS=32` vs
   A17B's 64.** Half the value-head count. Moeflux compiles against
   the macro so the per-layer buffer sizes *should* flow through, but
   any hand-coded loop bound or Metal shader threadgroup size that
   assumed 64 would silently corrupt. The output signature ("low-id
   byte-level cycle") is consistent with attention producing zero
   output and the model falling back to the byte-level prior — linear
   attention corruption on 30/40 layers is a plausible cause.

2. **Metal shader kernels (`shaders.metal`).** Not audited. Shaders
   take shape constants through preprocessor defines; if any kernel
   has shape literals (grid sizes, threadgroup dims) set for A17B
   that don't scale to A3B, the arithmetic would silently corrupt on
   the GPU path.

3. **Shared expert / MoE gate shapes.** `mlp.shared_expert_gate.weight
   shape=[1, 512] U32` was surprising (1 output row? U32?). Need to
   confirm this matches A17B's pattern and that moeflux handles it
   correctly. `shared_expert_intermediate_size=512` for A3B vs 1024
   for A17B — halved.

4. **`extract_weights.py` tensor ordering.** It sorts alphabetically;
   A17B might have had the same order. If `repack_experts.py` writes
   per-layer expert files indexed by a different ordering, routing
   could pick the wrong expert. `expert_index_qwen3_6_35b_a3b.json`
   at the repo root governs this — untrusted until verified.

## Empirical next cuts (in order of cost/benefit)

**A. Enable `do_debug=1` in `infer.m:2206` and run one layer of
   full-attention** (e.g., layer 3). Dump hidden_rms, normed_rms,
   q_proj_first5, q_gate_sigmoid_mean, attn_out_rms. Compare numbers
   against an MLX reference forward pass on the same token. First
   layer where values diverge is the bug site.

   Cost: recompile infer with the flag on, run. 5 minutes.

**B. Also enable the analogous debug in the linear-attn path.** The
   `A_log`, `dt_bias`, `conv1d.weight` code path — if the values
   propagating through are zero or way off-scale for A3B specifically,
   linear attention is our smoking gun.

**C. Run MLX reference inference on the same MLX dir** (`pip install
   mlx-lm`; `mlx_lm.generate --model ... --prompt "The quick brown
   fox"`). If MLX produces " jumps over the lazy dog", the weights are
   correct and moeflux's Metal pipeline is the bug. If MLX also fails,
   the MLX source conversion from HF is broken and we re-convert.

**D. A17B smoke from scratch.** Download A17B artifacts (Mike started
   this in a prior session; may already be mid-download), run infer
   on them, confirm A17B still produces sane output post-refactor. If
   A17B is *also* broken now, the A3B port broke A17B too and git
   bisect is the next step. If A17B works, A3B-specific hypothesis
   narrows further.

## Drama_llama-side status

**Implementation complete, gate 4 FAILS due to upstream.** Commits not
yet made:
- `src/moeflux/{mod,decoder,model,engine}.rs` — MoefluxDecoder,
  MoefluxModel (HF tokenizers crate), MoefluxEngine type alias.
- `Cargo.toml` — moeflux + moeflux-model-* features.
- `tests/moeflux_smoke.rs` — single-engine smoke, PASSES.
- `tests/cross_backend.rs` — both-backends agreement gate, FAILS
  (0/32 argmax, 0.129 Jaccard vs 0.95/0.80 thresholds).

Recommend committing all of the above as `#[ignore]`-gated so the
harness is ready the moment moeflux's A3B output is fixed.
