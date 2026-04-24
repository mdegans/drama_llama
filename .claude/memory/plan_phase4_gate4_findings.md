# Phase 4 gate 4 — implementation done, gate FAILED on moeflux output

Session 2026-04-25 (continuation of 2026-04-24's gate-1/2/3 pass).

## What landed in drama_llama

All four implementation tasks complete:

- `src/moeflux/{mod,decoder,model,engine}.rs` — `MoefluxDecoder`,
  `MoefluxModel`, `MoefluxEngine` type alias, `from_paths()`
  constructor.
- `Cargo.toml` — `moeflux` feature gating `dep:moeflux` +
  `dep:tokenizers` (HF tokenizers crate, BPE via `onig`).
  `moeflux-model-qwen3-5-a17b` and `moeflux-model-qwen3-6-35b-a3b`
  forward to moeflux's variant features.
- `tests/moeflux_smoke.rs` — single-engine load + 4-step greedy
  decode. **PASSES.** Loads in ~2s, runs to completion. Confirmed
  `n_vocab=248320`, `n_ctx=1048576` reported correctly.
- `tests/cross_backend.rs` — both backends loaded in one binary,
  same 4-token prompt, 32-step greedy each, asserts argmax
  agreement ≥95% + top-20 Jaccard ≥80%. **FAILS** the agreement
  gate — see below.

drama_llama-side smoke gate is green. The trait extraction held;
nothing in `predictor.rs` / `sample.rs` / `candidates.rs` needed
unpinning beyond what the prior session had already shipped.
`MoefluxModel: Sync` works for the rayon grammar fan-out path.

## What the cross-backend run found

Same prompt (`"The quick brown fox"`, no specials, both backends),
same model (`Qwen3.6-35B-A3B`; GGUF Q4_K_M for llama.cpp, MLX 4-bit
+ moeflux-extracted weights for moeflux). Tokenization agreed
exactly: both backends saw `[760, 3841, 13477, 37550]`.

**llama.cpp produces the canonical pangram completion** —
`[" jumps", " over", " the", " lazy", " dog", ".", "\n\n",
"The", " quick", " brown", " fox", ...]` (cycles, since greedy +
no stop). Sane.

**moeflux produces a low-id degenerate cycle** —
`["ro", " a", ".", " The", " ", "rome", ".", " The", " ", "rome",
".", " The", ...]`. Top-10 logits at step 0 are dominated by IDs
under 400 (single-byte / short-fragment tokens). Pattern is the
classic "input embedding isn't conditioning the forward pass"
signature.

Numbers:
- argmax agreement: 0/32 (threshold 0.95)
- top-20 Jaccard step 0: 0.081
- top-20 Jaccard step N: 0.176
- mean overlap: 0.129 (threshold 0.80)

## Root cause hypothesis

Not a drama_llama-side bug. The harness is correctly constructed:
- HF tokenizer agrees with llama.cpp's BPE on the prompt token IDs.
- Logit buffer is sized to `mf_n_vocab() = 248320`, matching
  `lm_head.shape[0]` in the manifest.
- moeflux compile-time shape constants match HF config.json
  (HIDDEN_DIM=2048, NUM_LAYERS=40, NUM_EXPERTS=256,
  NUM_EXPERTS_PER_TOK=8 vs `text_config.num_experts_per_tok=8`).
- `experts_per_tok=8` passed in matches both compile-time max and
  HF config.

The C smoke (`tests/smoke.c`) feeds *fabricated* tokens
`[1, 100, 500, 1000]` and only checks "logits aren't all zero or
NaN" — it never validated correctness. **Our cross-backend test is
the first real correctness check on moeflux.** It found a real bug.

Most likely candidates (ordered by my prior, not by evidence —
none of these have been verified):

1. **MLX → moeflux weight extraction is wrong.** Either the
   embedding table, the lm_head, or a per-layer projection got
   mis-shaped / mis-strided / mis-quantized. The
   "low-id degenerate cycle" pattern strongly suggests the input
   embedding lookup isn't projecting to the right hidden state, so
   subsequent attention sees garbage and outputs the byte-level
   bigram prior.
2. **Quantization mismatch.** MLX uses group-affine quantization
   with per-tensor zero-point + scale; moeflux's Metal kernels
   might assume something different (e.g. symmetric, or different
   group-size). A miscalibrated dequant on lm_head alone would
   produce exactly this output shape.
3. **Position-encoding off-by-something.** RoPE base or theta
   doesn't match what the model was trained with. Less likely to
   produce *this specific* degeneration but possible.

## Suggested next moves (for Mike to decide)

A. **Dump intermediate activations** in moeflux at the input
   embedding stage and compare against an MLX reference forward
   pass on the same token IDs. If the embeddings already disagree,
   it's `extract_weights.py` or the dequant; if they agree but
   logits diverge, it's a layer-internal bug.

B. **Try Qwen3.5-A17B** (the original flash-moe target) to see
   whether the degeneration is variant-specific to A3B. If A17B
   produces sane output, the conditional-compile scaffold or the
   per-variant weight repack is what broke. If A17B is *also*
   garbage, the bug is in moeflux's core pipeline, not the
   variant addition.

C. **Skip cross-backend gate** for now and document it as
   blocked-on-moeflux. The drama_llama-side work is done; gate 4's
   "loads and runs the regression harness" half is satisfied
   (loads, runs, produces output, harness compares). The
   "tolerance bar" half can't pass until moeflux is correct.

D. **Open question for Mike**: should the regression test be
   committed as `#[ignore]` so it's there to catch the eventual
   fix, or should it stay out of tree until moeflux is producing
   non-pathological output? Leaning toward (a) commit-as-ignored
   — it documents the bar and won't run by default.

## What to load on the next session

Read in order:
1. `MEMORY.md`'s `plan_v0.8.0_backend_split` pointer.
2. This file (gate 4 findings).
3. `plan_phase4_gate4_handoff.md` (now fully consumed; the
   "what's missing" list is all done).
4. `git log --oneline v0.8.0` since this session for the
   commits that added the moeflux module + the cross-backend
   test.

Files to look at first:
- `src/moeflux/decoder.rs:127` — `MoefluxDecoder::Decoder` impl,
  if any further trait debug is needed.
- `src/moeflux/model.rs:158` — `MoefluxModel::n_vocab` resolution
  (was the first non-obvious bug — config.json nests `vocab_size`
  under `text_config` for Qwen3 multimodal configs).
- `tests/cross_backend.rs:204` — the failing assertion, with
  full diagnostic dump above it.
