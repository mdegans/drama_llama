# Qwen batched prefill — session 2 outcome (Phases 1+2+3 GPU primitives)

Plan-of-record was [`qwen_batched_prefill_session2_gpu_plan.md`](qwen_batched_prefill_session2_gpu_plan.md). Session-1 outcome (the discovery that pivoted us): [`qwen_batched_prefill_session1_landed.md`](qwen_batched_prefill_session1_landed.md).

**Headline:** Three phases planned, three phases landed. Originally Phase 3 (batched flash-attn) was sized as a session of its own — it landed in this session because Phases 1+2 went faster than estimated and the MLA tiled-SDPA pattern transferred cleanly.

## What landed (4 commits on moeflux main)

1. **`0bb8375` — tests: factor diff helpers to tests/common/diff_helpers.rs**
   - Phase 0 from session 1, finally committed to main. 122 lines removed from `tests/diff_oracle.rs`; 140 LOC new shared module.

2. **`8a2cdad` — batched-prefill: bf16_matmul_n_tokens (Phase 1)**
   - New Metal kernel + Rust encoder. Applies one BF16 weight matrix to N stacked token activations in one dispatch.
   - Diff target: N calls of `bf16_matvec_cpu`. **Measured cosine = 1.000000000, max_abs_diff ≈ 5×10⁻⁶** on N=4, in_dim=2048, out_dim=512.
   - N=1 bit-exact vs `encode_bf16_matvec` (same per-row arithmetic, by construction).

3. **`e3c31c9` — batched-prefill: dequant_matvec_4bit_n_tokens (Phase 2)**
   - Two new kernels: `v3_n_tokens` (in_dim ≤ 4096, cached x_shared, ROWS_PER_TG=8) and `fast_n_tokens` (in_dim > 4096, one TG per row).
   - `encode_matvec_n_tokens` selects on the same in_dim ≤ 4096 threshold as `encode_matvec`.
   - **Both paths: cosine = 1.000000000** vs `dequant_matvec_4bit_cpu`. N=1 bit-exact on the v3 path. max_abs_diff ≈ 1.5×10⁻⁴ on fast path (in_dim=8192, more accumulation).
   - 8-bit batched skipped: only `mlp.gate.weight` is 8-bit on A3B and it's a small projection (`hidden_dim → num_experts`). Per-token dispatch is cheap there — revisit if profiling flags it.

4. **`6bb87f4` — batched-prefill: causal-masked tiled SDPA (Phase 3)**
   - Three new kernels: `attn_sdpa_causal_init_running` + `_tile_accumulate` + `_tile_finalize`. Tiled online-softmax mirroring the `mla_sdpa_tile_*` pattern, adapted for standard SDPA shape `[N, H, head_dim]`, GQA mapping (kv_h = h / heads_per_kv), and per-query causal cutoff (kv_max = start_pos + q + 1).
   - `encode_sdpa_causal_tiled` orchestrates init → per-tile accumulate loop → finalize.
   - **Solves two problems in one kernel:**
     - The GPU_KV_SEQ=8192 cliff (no fixed-size scores buffer; tile state lives in threadgroup memory).
     - Batched causal attention with N queries against [start_pos + N] keys/values.
   - Three diff tests, all **cosine = 1.000000000, max_abs_diff = 0**:
     - N=1 single-tile (kv_len=64)
     - N=1 multi-tile (kv_len=5000, exercises online-softmax merge)
     - N=4 per-query causal cutoff (start_pos=4, kv_len=8, kv_max(q) in {5,6,7,8})
   - Gate-free output by design; `sigmoid_gate` stays as a separate post-attn kernel matching the per-token `attn_values_batched` pipeline. Tests use q_gate=1000.0 (sigmoid==1.0 exactly in f32) to align CPU oracle with the gate-free GPU output.

## What's left for session 3 (Phases 4-6)

- **Phase 4 (GPU MoE permute-and-fuse)** — the prefill I/O win. Bucket tokens by expert; for each non-empty expert, read blob once + run batched matmul (using Phase 2's primitive) over the bucket. Existing per-token MoE dispatch is the diff target (same expert blobs, same routing, same arithmetic — only dispatch ordering changes). Cosine ≥ 0.9999 (atomic-add noise floor for accumulation order).
- **Phase 5 (step_internal_batched orchestrator)** — wire Phases 1-4 + KV cache batched append + per-layer RoPE batched loop. Returns last-token logits. Public surface stays `pub(crate)` until session 3+1 wires it through the predictor / chat / Agora reactor.
- **Phase 6 (end-to-end diff against C tokenwise)** — capstone validation. Prime KV via C tokenwise for `start_pos` positions, `state_save`, `state_load` to Rust side, run Rust `step_internal_batched` for N tokens, compare last-token logits (cosine ≥ 0.99, MoE-atomic-noise floor).

## Architecture conventions established this session

- **`*_n_tokens` naming** for batched-over-tokens kernels, to avoid collision with the existing `dequant_matvec_4bit_batched` kernel which is batched-over-experts (different axis).
- **N=1 bit-exact + N>1 cosine ≥ 0.9999** as the two-tier diff oracle for every new batched primitive. The N=1 test catches dispatch/indexing bugs sharply; the CPU-diff test catches semantic correctness.
- **Encoders take `&Buffer` + offsets** (not `&MtlWeightBuf`). Matches `encode_bf16_matvec`/`encode_bf16_matmul_n_tokens`. Makes synthetic-data tests trivial; production callers already have offsets handy.
- **`MatvecPipelines` grows by field, not struct rewrite.** `v3_4bit_n` / `fast_4bit_n` added next to existing fields. No constructor outside the impl, so additive.
- **Test harness lives in `tests/batched_diff_oracle.rs`**, gated `#![cfg(target_os = "macos")]` only — model feature not required because tests are synthetic-data. Uses the Phase 0 `tests/common/diff_helpers.rs` (its first consumer).

## What's load-bearing for session 3

- **Phases 4-6 from `qwen_batched_prefill_session2_gpu_plan.md`** carry forward unchanged.
- **Phase 4 entry point**: the per-token MoE dispatch path is in `crates/moeflux/src/riir/expert_forward.rs`. Build per-expert assignment CSR offsets + (src_row, slot, weight) tuples on host, then for each non-empty expert: read blob once + batched matmul (Phase 2 primitive!) over the bucket + gather-weighted into out. Caller-CPU bookkeeping, GPU heavy lifting.
- **Phase 4 diff target**: tokenwise loop of the existing per-token GPU MoE dispatch. Same kernels, different ordering. Cosine ≥ 0.9999 — atomic-add noise floor only catches kernel-internal bugs, not dispatch-order ones (but dispatch-order bugs are easier to spot anyway).
- **Phase 5 wiring**: `step_internal_batched` in `crates/moeflux/src/riir/mod.rs`. Pseudo-sketch in plan-of-record. Needs batched KV cache append (extend existing `mla_kv_cache_append` pattern), per-layer RoPE batched (small new kernel; not on critical path), then sequence Phase 1-4 primitives.
- **Phase 6 oracle gotcha**: same-Ctx C tokenwise prefill (no `memory_clear` mid-prefill) + cross-Ctx `state_save`/`state_load` to seed Rust deterministically. Floors loosen to cosine ≥ 0.99 because of Metal MoE atomic noise.
- **Phase 3 cleanup post-validation**: per-token attention currently uses `attn_scores_batched` + `attn_softmax_batched` + `attn_values_batched` with the `GPU_KV_SEQ=8192` cap. Tiled SDPA from Phase 3 doesn't need that cap. After session 3's Phase 6 validation, the per-token fast path can either route through the new tiled kernel (n_tokens=1 special case) or stay as-is. Defer the decision until measurement.

## Numbers worth quoting

| Test | in_dim/kv_len | out_dim/head_dim | N | cosine | max_abs_diff |
|------|--------------:|-----------------:|--:|-------:|-------------:|
| bf16_matmul_n_tokens vs CPU | 2048 | 512 | 4 | 1.000000000 | 5×10⁻⁶ |
| bf16_matmul_n_tokens N=1 vs single | 1024 | 256 | 1 | bit-exact | 0 |
| dequant_4bit_v3_n_tokens vs CPU | 2048 | 512 | 4 | 1.000000000 | 3×10⁻⁵ |
| dequant_4bit_fast_n_tokens vs CPU | 8192 | 256 | 4 | 1.000000000 | 1.5×10⁻⁴ |
| dequant_4bit_v3_n_tokens N=1 vs single | 1024 | 256 | 1 | bit-exact | 0 |
| sdpa_causal_tiled N=1 single-tile | kv=64 | 256 | 1 | 1.000000000 | 0 |
| sdpa_causal_tiled N=1 multi-tile | kv=5000 | 256 | 1 | 1.000000000 | 0 |
| sdpa_causal_tiled N=4 causal mask | kv=8 | 256 | 4 | 1.000000000 | 0 |

All 8 tests run in 0.10-0.26s total. The harness is fast enough that any future per-kernel batched primitive can adopt the same N=1-bit-exact + N=4-CPU-diff pattern without budget worry.

## Verification commands

```bash
cd ~/Projects/moeflux

cargo test --release -p moeflux --features model-qwen3-6-35b-a3b \
  --test batched_diff_oracle -- --ignored --nocapture --test-threads=1

# Expected: 5 tests, all pass, cosine = 1.000000000 on the CPU-diff
# tests, bit-exact on the N=1 tests.
```

## Calibration

- Session moved at full pace once Mike toggled "no clarifying questions". The pivot-on-discovery discipline held: when the encoder signature would have required `MtlWeightBuf` (which needs a `WeightFile` we don't want for synthetic tests), I refactored the encoder to `&Buffer + offsets` mid-session rather than papering over it. That's the right discipline — same as session 1's CPU-forward pivot, much smaller scope.
- Plan estimate ("Phase 1 ~1-2 hours, Phase 2 ~45-60 min") held — both phases landed comfortably with time for memo. Phase 3 (originally projected as a session-of-its-own) also landed because the MLA tiled-SDPA pattern transferred cleanly and the per-query causal mask via `-INFINITY` scores avoided the off-by-one trap that was the highest-risk piece in the plan.
- **My Phase 3 diff-oracle prediction was right** (the calibration note in the previous version of this memo correctly anticipated cosine ≥ 0.9999). Reality came in tighter than the floor — cosine literally == 1.0 on all three Phase 3 tests, max_abs_diff == 0.0. The pre-init-running-state kernel turned out to be the load-bearing decision: it makes the "first tile is entirely masked" edge case impossible to hit, removing the NaN-from-exp(-INF - -INF) hazard.
- **Context budget**: Mike framed this explicitly mid-session — 1M is a lot, small sessions have warm-up overhead that costs more, /context output is the trustworthy gauge (not my self-estimate). After session 2 ended with Phases 1+2+3 landed, we were still nowhere near the threshold. The "don't pre-emptively wrap" calibration carries forward to future sessions: stop on the work, not on context anxiety.
