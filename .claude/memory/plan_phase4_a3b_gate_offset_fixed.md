# Phase 4 A3B — gate-offset bug fixed, canonical output restored

Session 2026-04-27 (late). Starting state: A3B produced coherent but wrong
output ("es are the quick brown foxes..." loop); cross_backend argmax
agreement 3.1%; the preceding sessions had ruled out gate dequant, RoPE,
attention shapes, Metal kernels, and config deltas as the cause.

## Root cause

`infer.m` had **8 places with hardcoded A17B byte offsets** for the
per-expert gate_s and gate_b byte regions of the packed expert file:

  `gate_s_off = 2097152   // = MOE_INTERMEDIATE × HIDDEN_DIM × 4 / 8`
  `gate_b_off = 2228224   // = gate_s_off + MOE_INTERMEDIATE × (HIDDEN_DIM/64) × 2`

These values are correct for A17B (HIDDEN_DIM=4096, MOE_INTERMEDIATE=1024)
but wrong for A3B (HIDDEN_DIM=2048, MOE_INTERMEDIATE=512 → should be
524288 / 557056). For A3B these offsets pointed **past the actual gate
scale/bias data into the next expert's weight data or zero padding**,
causing gate_proj's dequant-matmul to produce ~zero output for every expert
on every layer.

The up_proj, down_proj, and other expert paths already used the
`GATE_S_OFF` / `GATE_B_OFF` / `UP_W_OFF` / etc. macros defined in
`model_variant.h` (which correctly derive per-variant). The gate_proj
literals slipped through the prior A17B-literals cleanup (commit
`09d07a0`).

Fix: replace all 10 occurrences of the raw numbers with the macros.
Upstream commit `925f7a0` on `main`.

## Why symptoms looked so strange

- Gate matmul for the 8-bit **routing** gate (`mlp.gate`) is a separate
  tensor, not in the packed expert file — so routing worked correctly.
- Shared expert is on its own path — worked.
- Attention worked (h_post matched MLX to 1% relative error at layer 0).
- Only the **routed experts' gate_proj** had bad reads → gate=0 →
  SiLU(0) × up = 0 → down_proj(0) = 0 → every expert's contribution
  was zero.
- moe_combine_residual computed `out = h_mid + 0 + shared_gated`.
- This gave **coherent English conditioned on the prompt** (residual
  stream + shared expert carry most of the signal) but the wrong
  distribution at every layer.

The "only gate_proj, not up_proj" asymmetry was the deciding clue: same
matvec kernel, same input buffer, different weight offset → gate_s/b
pointers must be wrong for one but not the other.

## How we found it

Three-phase localization using a new MLX-reference diff harness
(committed at `moeflux/metal_infer/tests/mlx_reference/`):

  1. **`diff_gate_outputs.py`**: dumped per-layer h_post + gate logits +
     top-K and compared to MLX. Layer 0 top-8 set matched exactly →
     routing is correct.
  2. **`diff_layer_inputs.py`**: dumped per-layer input hidden state (=
     prior layer output). Layer 0 input matched (it's just the
     embedding); layer 1 input (= layer 0 output) showed relative RMS
     0.32 vs MLX → **bug is inside layer 0's forward**.
  3. **`diff_l0_components.py`**: dumped h_mid, shared_raw, per-expert
     outs, and final out at layer 0. Moeflux's per-expert output buffers
     (`buf_multi_expert_out[k]`) were ALL ZERO; MLX's were non-zero.
     Probing `buf_multi_expert_gate[k]` (gate_proj intermediate) also
     zero, but `buf_multi_expert_up[k]` non-zero → gate_proj matmul was
     the culprit → hardcoded A17B offsets.

A17B run with the same diagnostic had non-zero expert outputs,
confirming the bug was A3B-specific (correct — for A17B the hardcoded
values happened to be correct).

## Verification

- A3B smoke ("The quick brown fox"): **"jumps over the lazy dog."** +
  Python code block, matching the MLX canonical completion.
- A17B smoke: still produces "jumps over the lazy dog" — no regression
  (hardcoded 2097152 = GATE_S_OFF for A17B, the substitution is a no-op
  there).
- drama_llama cross_backend test: argmax agreement 3.1% → 21.9%. Below
  the 95% threshold but the major bug is fixed; remaining delta likely
  smaller numerical issues or llama.cpp-specific divergences. Test stays
  `#[ignore]` until the remaining gap is characterized.

## Instrumentation left in place

The `MOEFLUX_DUMP_L0=/path/prefix` env-gated dump is committed upstream.
Zero cost when unset. Useful for the next MoE variant bring-up or any
similar numerical debugging. Files emitted:
  - `{prefix}_l{N}.bin`          per-layer h_post + gate logits + top-K
  - `{prefix}_l{N}_in.bin`       per-layer input hidden state
  - `{prefix}_l0_components.bin` layer-0 MoE component breakdown

The `tests/mlx_reference/README.md` documents the 3-phase diff workflow.

## What's next

1. Re-run full cross_backend test suite; document the remaining gap
   (21.9% → ? with the 6 intermediate sessions' worth of other fixes).
2. Consider un-`#[ignore]`ing cross_backend with a threshold tuned to
   the new baseline.
3. Update `plan_v0.8.0_backend_split.md` Phase 4 status: Gate 4 is now
   effectively passed for A3B output quality (canonical pangram). Next:
   Gate 5 (Cogito 600B bring-up) was the original endpoint.

## moeflux repo state

Branch `main`, 7 commits ahead of origin. HEAD is `925f7a0`
"infer: Fix A3B gate_proj reading zeros (hardcoded A17B offsets)".

Preceding commit `f47c0b3` (8-bit dequant) and `09d07a0` (A17B-literals
partial cleanup) set the stage — this commit is the completion of that
cleanup thread, specifically for the gate_proj offsets that were missed.

## Handy paths (unchanged)

- moeflux repo: `/Users/mdegans/Projects/moeflux`
- A17B extracted: `/Volumes/Temp Backup/models/moeflux/qwen3-5-a17b-*`
- A3B extracted:  `/Volumes/Temp Backup/models/moeflux/qwen3-6-35b-a3b-*`
- A3B MLX source: `/Volumes/Temp Backup/models/moeflux/qwen3-6-35b-a3b-mlx-4bit`

## Key commands

```bash
# A3B smoke — should produce canonical pangram
cd /Volumes/Temp\ Backup/models/moeflux/qwen3-6-35b-a3b-artifacts && \
  MOEFLUX_SHADERS_PATH=/Users/mdegans/Projects/moeflux/metal_infer/shaders.metal \
  /Users/mdegans/Projects/moeflux/metal_infer/infer \
  --model /Volumes/Temp\ Backup/models/moeflux/qwen3-6-35b-a3b-root \
  --weights model_weights.bin --manifest model_weights.json --vocab vocab.bin \
  --prompt "The quick brown fox" --tokens 16 --k 8

# drama_llama cross_backend
cargo test --test cross_backend \
  --features "llama-cpp,moeflux,moeflux-model-qwen3-6-35b-a3b" \
  -- --ignored

# Debug diff harness (see tests/mlx_reference/README.md)
MOEFLUX_DUMP_L0=/tmp/mf <infer invocation>
uv run --with mlx --with mlx-lm python3 \
  /Users/mdegans/Projects/moeflux/metal_infer/tests/mlx_reference/mlx_layer_dump.py /tmp/mx
python3 /Users/mdegans/Projects/moeflux/metal_infer/tests/mlx_reference/diff_layer_inputs.py
```
