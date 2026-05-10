---
name: Future work — Qwen full-attn SDPA on GPU
description: Qwen a3b's full-attn layers run scaled-dot-product-attention on the CPU; this dominates long-prompt prefill. Mirror Cogito-V2's GPU MLA SDPA approach in `mla_attn_forward` / `gpu_mla` to land a `sdpa_gpu` on Qwen's full-attn path.
type: project
---

# Qwen full-attn SDPA on GPU

Discovered 2026-05-10 during the 8k-prompt cold-prefill investigation
(see `qwen3_a3b_prefill_baseline.md`). 76% of stack samples in
`moeflux::riir::sdpa::sdpa_cpu` called from
`full_attn_forward::full_attn_layer_forward`. **Prefill bottleneck for
Qwen a3b on long prompts.**

## What's missing

- Only `sdpa_cpu` exists in moeflux: `crates/moeflux/src/riir/sdpa.rs:63`.
- Hardcoded call site for Qwen full-attn path:
  `crates/moeflux/src/riir/full_attn_forward.rs:362`.
- `sdpa.rs` exports only `sdpa_cpu` and `SdpaError`
  (`crates/moeflux/src/riir/mod.rs:97`).

GPU SDPA exists in moeflux but only as **MLA-folded-form Metal kernels**
in `mla_attn_forward` / `gpu_mla`. Different attention mechanism
(MLA vs Qwen's GQA + GatedDeltaNet linear-attn split), separate
codepath.

## Why it hurts prefill but not decode

Asymmetry is structural:

- **Prefill** SDPA: Q[N tokens] × K[N tokens] → N×N scores per head per layer = **O(N²)**.
  At 7k context, ~50M dot products per head per layer, on CPU.
- **Decode** SDPA: Q[1 token] × K[N tokens] → N dot products per head = **O(N)**.
  At 7k context, ~7k dot products per head. CPU is fine.

That's why historical short-prompt decode benchmarks (a3b at 17.6 tok/s
grammar-path, ~20 tok/s ceiling) never surfaced this issue, but a 7k
cold prefill takes 25-75 min on CPU.

## Approach

Mirror Cogito-V2's GPU MLA SDPA (Phase 3-4 of the GPU MLA landing).
The kernel shape is even *simpler* for Qwen's full-attn path:

- No K/V latent decomposition (no kv_b_K / kv_b_V folding).
- No YaRN per-position scaling.
- Just standard SDPA: `softmax(Q @ K^T / sqrt(d)) @ V`.

Reuse machinery from the MLA implementation:

- `sdpa inner` Metal kernel from `mla_attn_forward.rs::sdpa_inner` —
  adapt for non-folded shape.
- Tile sizing already validated at 32 KB threadgroup memory cap
  (see `cogito_v2_full_gpu_session2_landed.md`).
- Diff oracle: `sdpa_cpu` is the byte-exact reference. Same testing
  approach as MLA's CPU-vs-GPU cosine validation.

## File pointers

- Add: `crates/moeflux/src/riir/sdpa.rs::sdpa_gpu` (signature mirrors
  `sdpa_cpu`).
- Add: `crates/moeflux/src/riir/sdpa_metal.metal` (or extend existing
  metal source with a `sdpa_qwen` kernel).
- Modify: `crates/moeflux/src/riir/full_attn_forward.rs:362` to
  dispatch GPU when buffers are GPU-resident, fall back to CPU
  otherwise (same pattern as MLA's CPU/GPU split).
- Update: `crates/moeflux/src/riir/mod.rs:97` to export `sdpa_gpu`.

## Expected impact

Cold prefill for 8k Qwen a3b: ~25 min → seconds (estimate based on
MLA's CPU-vs-GPU speedup ratio in `cogito_v2_full_gpu_session2_landed.md`).
Removes "first turn is unusably slow" as a user-visible Agora pain
point. Decode rates unaffected by this work.

## Dependencies

- None blocking. The MLA GPU kernels demonstrate every primitive
  needed.
- Optional: snapshot v2 (already landed) covers KV-cache state
  ser/de for breakpoint cache, so GPU-resident KV doesn't break
  the prefix-reuse path.
