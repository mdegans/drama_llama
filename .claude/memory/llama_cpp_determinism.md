# llama.cpp logits are not reproducible across KV layouts (open; explore later)

Tracked in #126.

**Measured 2026-09-30** on an M2 Max (Metal), mainly Qwen3.6-35B-A3B (Q4_K_S, hybrid MoE), with cogito-32b (dense) as a control.
The probe is [`examples/logits_determinism.rs`](../../examples/logits_determinism.rs).

**Why it matters.** Many tests print a seed so a failure can be reproduced by setting it explicitly. That only works if the same seed and the same prompt produce the same logits.
- They do **not** when anything else shares the KV cache: other `--cache-slots` sequences, or a different prefill chunking.
- So a seeded failure seen under blallama load may not reproduce in isolation. This hurts debugging, and it also makes warm-vs-cold equivalence checks diverge late in fluent text.
- It is a correctness issue (same input, different output), not only a performance one.

## What is and isn't deterministic

- **Bit-identical:** the same prompt with the same decode schedule (fresh or reused context, after `seq_rm` / `memory_clear`). `restore_to` from a snapshot is also bit-identical, so the prefix cache is lossless.
- **Changes every later logit:**
  - any ubatch boundary that is off the `n_ubatch` grid (max |Δ| ≈ 0.44 with flash attention, ≈ 0.22 without), even if later ubatches realign. Token-by-token decoding differs from batch prefill.
  - **other sequences in a unified KV** (`kv_unified`, `--cache-slots N`): ≈ 0.46. A single slot removes it.
- The magnitudes are large for floating-point noise. MoE routing amplifies the tiny differences: a near-tied expert choice flips.

## Root cause (neighbor effect)

Masking is exact. The neighbor effect comes from two places:
1. **Reduction order keyed on the absolute cell index.** `find_slot` places a sequence after its neighbors, so the same tokens sit at different cells, and the attention reduction sums in a different order.
2. **The Metal FA vec kernel picks `nsg` from `n_kv`** (1/2/4 at 2048/4096). A neighbor that grows the KV view changes the kernel's split.

Prefill is invariant to cell offsets that are multiples of 64. Decode is invariant to multiples of 512 at nsg 1, and of 4096 at nsg 4. Offsets that are symmetries of the reduction tree cost nothing.

## Experiments (on the mdegans/llama.cpp fork; not upstream)

llama.cpp fork (`~/Projects/llama-cpp-sys/external/llama.cpp`, fork remote `mdegans/llama.cpp`):
- [`determinism/kv-align`](https://github.com/mdegans/llama.cpp/tree/determinism/kv-align) @25ea5d074 — `GGML_METAL_FA_VEC_NSG` pins the FA vec nsg.
- [`determinism/sparse-decode`](https://github.com/mdegans/llama.cpp/tree/determinism/sparse-decode) @333a88e72, on top of kv-align:
  - backports upstream #27530 (K/V cleanup after a failed restore);
  - `LLAMA_KV_SPARSE_DECODE` passes a sparse FA bound for single-sequence decode;
  - `GGML_METAL_FA_SPARSE_NOCAP`;
  - `LLAMA_KV_CONGRUENT=S` places cells position-congruent mod S, with wrap-around and fallback.

To try it, add an **untracked** `.cargo/config.toml` in a drama_llama worktree:
```toml
[patch.crates-io]
llama-cpp-sys-3 = { path = "/Users/mdegans/Projects/llama-cpp-sys" }
```
and check the submodule out at the branch. Never commit this patch.

Results:
- **Sparse decode + congruent placement + pinned nsg:** NEIGHBORS and KV-VIEW logits become bit-identical on dense and hybrid models. Decode is 15% slower at 8k and 34–49% slower at 32k (worse at 128k).
- **Pinned nsg alone:** costs about 0–5%, but doesn't remove the cell-order effect.
- **S=4096 congruence:** needs about 16k free cells. Wrap-around reverses block order, and only about 45% of tokens were fully independent. It isn't usable at the Agora config (128k context, ~1 slot on 96 GB).

Conclusion for now: production stays on stock llama.cpp. If you need reproducibility, use `--cache-slots 1`, or run the probe with a matched schedule.

## Upstream context

Batch invariance was declined upstream (#16016, #23335). There is a related sparse-FA hint (#28098 / #27970).
Michael files upstream himself: their guidelines ban AI-written issue and PR text. Nothing has been filed.

## Next steps (when there is time)

- An `--isolate-slots` opt-in (`kv_unified=false`, `n_ctx` per slot), approved; only practical with more memory.
- Cheaper variants: dense path with page-congruent placement plus pinned nsg (≈ stock speed, ~12% KV waste).
- Seed-repro hygiene: when a seeded test fails, record `n_ubatch`, slot count and occupancy alongside the seed.
- Related: #117 (seeds 2k and 2k+1 alias).
