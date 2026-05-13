# Qwen batched prefill — session 2 plan (GPU-direct)

Plan-of-record for the next batched-prefill session. Folds in the session-1 discovery (no per-token CPU forward exists for Qwen3.6-A3B; CPU scaffolding as oracle was unworkable). Outcome memo for session 1: [`qwen_batched_prefill_session1_landed.md`](qwen_batched_prefill_session1_landed.md).

## Premise

Eventual goal: GPU batched prefill for Qwen3.6-A3B in moeflux. Eliminates the GPU_KV_SEQ=8192 cliff for Agora's 40-60k workloads + amortizes per-token expert-blob SSD fetches across the batch.

**Session 2 ships the GPU batched implementation directly, diffed against C tokenwise.** No CPU intermediate (per session-1 finding). Smaller-fixture validation per new GPU primitive substitutes for the missing per-kernel CPU diff.

## Strategy: incremental, each step has its own validation

Don't land "GPU batched forward" in one shot. Land each new primitive with its own bit-exact-against-something test, then compose.

### Phase 0 — `BatchedDiffBackend` trait + harness (precursor)

New `tests/batched_diff_oracle.rs`. Trait surface session 2 needs:

```rust
pub trait BatchedDiffBackend {
    fn open() -> Self;
    fn reset(&mut self);
    fn state_save(&self) -> Vec<u8>;
    fn state_load(&mut self, state: &[u8]);

    /// Process tokens at positions start_pos..start_pos+N.
    /// Per-implementation: C loops mf_eval_token; Rust calls
    /// the new step_internal_batched (this session's main deliverable).
    fn prefill(&mut self, tokens: &[i32], start_pos: i32);

    /// Capture per-layer hidden output at one position, one layer.
    /// For C: mf_layer_forward_dump. For Rust batched: the new
    /// batched_layer_forward_dump that takes positions[] and returns
    /// outputs[].
    fn layer_dump(&mut self, layer_idx: i32, pos: i32, hidden_in: &[f32]) -> Vec<f32>;
}
```

CBackend impl: tokenwise loop over `mf_eval_token` for `prefill`; single-pos `mf_layer_forward_dump` for `layer_dump`. Already exists in spirit at `tests/common/c_backend.rs`.

RsBackend impl: the new batched forward (Phase 4 below).

Use `tests/common/diff_helpers.rs` (landed session 1) for `cosine_sim`, tolerances, `default_a3b_paths`.

### Phase 1 — `batched_bf16_matvec` Metal kernel + diff against per-row CPU

Simplest GPU batched primitive. Existing `bf16_matvec` kernel (`shaders/shaders.metal`, registered in `gpu_matvec.rs::BfMatvecPipelines`) operates on single-vector input. Extend to multi-vector:
- New shader: `bf16_matmul_batched` — kernel parameterized by `batch_size`. Each threadgroup handles one (row, batch) pair.
- New Rust wrapper: `encode_bf16_matmul_batched(cmdbuf, pipe, x: [N, in_dim], w, out: [N, out_dim], ...)`.
- Diff: feed N rows, compare against N single-row `bf16_matvec_cpu` calls. Cosine = 1.0 (no MoE atomic noise here, just matvec). Tolerance: bit-exact-mod-fp-reorder.

**This is the simplest first step.** Lands one new kernel + its validation. ~1-2 hours of work. Sets the pattern for everything else.

### Phase 2 — `batched_dequant_matvec_4bit` Metal kernel (for batched Q/K/V proj + o_proj)

Same shape as Phase 1 but for the 4-bit MLX quantized weights. Single-row exists at `gpu_matvec.rs::encode_dequant_matvec_4bit` (or similar). Batch parameter added. Diff against N calls of `dequant_matvec_4bit_cpu` (`cpu_matvec.rs:83`).

### Phase 3 — `batched_sdpa_causal_flash_attn` Metal kernel (the cliff fix + batched attention)

The biggest new piece. Two problems solved in one:
- The GPU_KV_SEQ=8192 cliff (current `attn_scores_batched` + `attn_softmax_batched` + `attn_values_batched` allocates `scores[H, GPU_KV_SEQ]` device memory; tiled flash-attention keeps scores in threadgroup memory, no cap).
- Batched attention with causal mask (N queries against [start_pos + N] keys/values with per-query causal cutoff).

Reference: `mla_sdpa_tile_accumulate` + `mla_sdpa_tile_finalize` (`gpu_mla.rs:202`, `shaders/shaders.metal`). The MLA tile pattern is the architectural template. Adaptations:
- GQA shape (Qwen3-A3B has `num_attn_heads / num_kv_heads = heads_per_kv`, e.g., 16:1). Per-head `kv_h = h / heads_per_kv` lookup like the existing `attn_scores_batched`.
- Causal mask: per-query `kv_max = start_pos + q_idx + 1`. Tile-internal: `if (p >= kv_max) score = -inf` before softmax-online-merge.
- Output shape `[N, H, head_dim]` instead of `[H, kv_lora_rank]`.
- Sigmoid gate stays as a separate kernel (post-attn).

Diff target — **the hardest part of this session**:
- Cannot diff against a CPU port of flash-attention (no CPU equivalent in moeflux).
- Diff against tokenwise loop of `sdpa_cpu` (`sdpa.rs:63`) — one CPU SDPA call per query position with growing kv_len. ULP-bounded; cosine ≥ 0.9999 expected.
- Smaller fixtures: synthetic Q/K/V with known structure (e.g., Q=K=V=identity-ish, expected output = V[last]); validates the kernel mechanics in isolation.

### Phase 4 — GPU MoE batched dispatch (permute-and-fuse)

The I/O amortization win. Per-token Qwen3 MoE currently: per-token, K active experts × (read 24 MB blob + matvec). Batched: bucket tokens by expert, for each non-empty bucket read blob once + matvec K times.

GPU shape:
- Build per-expert assignment list on host (small — N×k_active integers); pass to GPU as a `[num_experts+1]` CSR offsets + `[N×k_active]` (src_row, slot, weight) tuples.
- For each non-empty expert: encode one cmdbuf with (read-blob → batched-matmul-N over the bucket → gather-weighted into out).
- The host-side bookkeeping for "non-empty experts" stays CPU; the heavy lifting goes GPU.

Diff target: tokenwise loop of the existing per-token GPU MoE dispatch. Same expert blobs, same routing, same arithmetic — only the dispatch ordering changes. Cosine ≥ 0.9999 (atomic-add noise floor for accumulation order).

### Phase 5 — `step_internal_batched` orchestrator

Public Engine entry point. Wires Phase 1-4 + KV cache batched append + per-layer RoPE batched loop. Returns last-token logits.

Diff target: this is where session 3 wires in (separately).

### Phase 6 — end-to-end diff against C tokenwise

The capstone validation. New test: prime KV via C tokenwise for `start_pos` positions, `state_save`, `state_load` to Rust side, run Rust `step_internal_batched` for N tokens, compare last-token logits (cosine ≥ 0.99 per the per-token logits floor — wider because end-to-end accumulation goes through MoE atomic noise).

## What NOT to do this session

- **No CPU port of Qwen3 layer forward.** Session-1 finding: too much scope. Keep CPU diff oracles to existing single-row primitives (sdpa_cpu, dequant_matvec_4bit_cpu, etc.) — these are the per-kernel diff targets.
- **No public Engine API wiring.** `step_internal_batched` stays `pub(crate)` until session 3 wires it through the predictor / chat / Agora reactor.
- **No linear-attn batched** (chunkwise state recurrence is its own algorithm; 27/36 layers in A3B). Phase 6 tests gate on full-attn layers only. Session 3 picks this up.
- **No perf benchmarks.** Bench protocol depends on the cold-token vs warm-token state being stable; benching mid-implementation churns numbers. Bench after Phase 6 validates correctness, in a separate session.

## Risks

- **MoE atomic-noise floor at end-to-end diff.** Existing `diff_oracle.rs` finding: end-to-end logits are non-deterministic across `memory_clear` due to Metal MoE atomic ops. Mitigation: Phase 6 uses **same-Ctx C tokenwise** prefill (no `memory_clear` mid-prefill) + cross-Ctx `state_save`/`state_load` to seed Rust deterministically. Floors loosen to per-position cosine ≥ 0.99 (not 0.9999).
- **No CPU oracle for permute-and-fuse MoE.** Mitigated by Phase 4's "diff against tokenwise GPU MoE" — same kernels, different dispatch order. But this only catches dispatch-order bugs, not kernel-internal bugs.
- **Flash-attention causal mask off-by-one.** Mitigation: N=1 degenerate case must match single-shot SDPA exactly (cosine ≥ 0.9999); N=2 with q_idx=1 attending to positions [0,1] (not [0,1,2]) must produce kv_max=2 internally.
- **GPU_KV_SEQ buffer removal causes regressions.** The current per-token GPU full-attn fast path uses `scores[H, GPU_KV_SEQ]` device memory. Tiled flash-attention doesn't need it. Keep the per-token fast path working until the tiled version is fully validated; remove the buffer in a separate cleanup PR after Phase 6.

## Files modified (anticipated)

- `crates/moeflux/shaders/shaders.metal` — new kernels: `bf16_matmul_batched`, `dequant_matvec_4bit_batched`, `attn_sdpa_tile_accumulate_causal`, `attn_sdpa_tile_finalize_batched`, possibly `moe_batched_dispatch` (more likely host-side orchestration of existing per-expert kernels).
- `crates/moeflux/src/riir/gpu_matvec.rs` — `encode_bf16_matmul_batched`, `encode_dequant_matvec_4bit_batched`.
- `crates/moeflux/src/riir/gpu_attn.rs` (or new `gpu_attn_batched.rs`) — `encode_attn_sdpa_causal_tiled` + `MlaPipelines`-style pipeline registry.
- `crates/moeflux/src/riir/expert_forward.rs` — `gpu_batched_experts_encode_permuted` (per-expert dispatch over a CSR-sorted assignment list).
- `crates/moeflux/src/riir/mod.rs` — `step_internal_batched`.
- `crates/moeflux/tests/batched_diff_oracle.rs` — new test binary, Phase 0+6.
- Plus the per-phase fixture / smoke tests (one per new kernel).

## Run commands (canonical)

```bash
cd ~/Projects/moeflux

cargo build --release --features model-qwen3-6-35b-a3b

# Per-phase tests as they land:
cargo test -p moeflux --features model-qwen3-6-35b-a3b --release \
  --test batched_diff_oracle -- --ignored --nocapture --test-threads=1
```

## Pointers (durable)

- Session-1 outcome: `qwen_batched_prefill_session1_landed.md`
- Original plan-of-record (now historical): `/Users/mdegans/.claude/plans/async-toasting-mitten.md`
- llama.cpp reference (production-fast Metal): `~/Projects/llama-cpp-sys/llama.cpp/` — `ggml_mul_mat_id`, `ggml_flash_attn_ext`, batched KV append via index tensors, same graph for prefill and decode.
- MLA tiled SDPA (architectural template for Phase 3): `crates/moeflux/src/riir/gpu_mla.rs:202`, shaders at `shaders/shaders.metal` (`mla_sdpa_tile_accumulate`, `mla_sdpa_tile_finalize`).
- Bench discipline for after Phase 6: high-perf power, n≥3, reboot between revisions (`feedback_bench_discipline.md`).
