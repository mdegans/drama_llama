# Qwen batched prefill — session 4 outcome (full)

Plan-of-record: `/Users/mdegans/.claude/plans/session4_batched_prefill_integration.md`.
Session-3 outcome (entry point): [`qwen_batched_prefill_session3_landed.md`].

**Headline:** every planned phase shipped this session. Batched prefill
on Qwen3.6-A3B routes through a full batched-GQA orchestrator
(`step_internal_batched_gqa`) with batched MoE permute-fuse + batched
SDPA + batched Q/K/V/O projections + batched shared expert FFN +
chunkwise iteration. `eval_token` now also routes through the same
path at N=1 — single-token chunks are faster than the per-token oracle
they replaced. Cosine = 1.0 against the per-token oracle and against
the C side throughout the canary battery; prompt-cache `start_pos != 0`
is **bit-exact**. Directional bench at N=256: 2.01× speedup.

## Phases shipped (moeflux main)

| Commit | Phase | What landed |
|--------|-------|-------------|
| `d1a0af4` | A | `post_attention_tail` split into `post_attention_pre_moe` + `moe_dispatch_per_token`. No behaviour change. |
| `94aedba` | B1 + C | `batched_full_attn_layer_forward` + `step_internal_batched_gqa` orchestrator. Batched MoE permute-fuse via `encode_moe_batched_permute_fuse`; everything else tokenwise inside the orchestrator. Wired step_internal to dispatch to it for Gqa variants. |
| `c1f9b52` | D | Chunkwise iteration (`BATCHED_CHUNK_SIZE=8192`) + `set_batched_chunk_size_for_test` hook + two cache-hit tests. |
| `3677701` | F (1st pass) | Directional bench `bench_batched_eval_prompt_vs_per_token` (1.82× at N=256, B1 only). |
| `71b36dc` | B2 | Batched tiled SDPA via `encode_sdpa_causal_tiled` over the joint Q stack vs (kv_start..kv_start+N) positions. Pre-SDPA captured per-token, post-SDPA reuses `post_attention_pre_moe` with `gpu_attn_args=None`. **Bug found during impl:** the original buffers.input "skip last restage" optimization was wrong (each Phase 3 iteration overwrites it for the current token). Diagnosed via `diag_b2_eval_prompt_chunk_1`. |
| `390cb0c` | B3 | Batched Q/K/V/O projections via `encode_matvec_n_tokens`. New `post_attention_post_o_proj_to_intermediates` helper skips o_proj when buffers.output is pre-populated. Bench: 2.01× at N=256. |
| `b65af55` | B4 | Batched shared expert FFN via three `encode_matvec_n_tokens` + flat-dispatch `swiglu_fused`. New `post_attention_residual_norm_route` helper skips shared FFN inside post_attn tail. |
| `764d45d` | G (routing) | `eval_token` routes through `step_internal(&[tok], pos, ...)` instead of `step_internal_per_token_oracle`. Directional decode bench: batched-N=1 is **17.6% faster** than per-token oracle (12.09 → 14.22 tok/s, cosine=1.0). |
| | E + H | drama_llama Session audit confirmed zero changes needed (chunking transparent at `Decoder::prefill`); prefetch confirmed decode-only by inspection. |

Plus drama_llama side (`ca25b1c` memo + earlier commits `d60ac2e` chat_template permissive env, `2ba62b3` tracing pass).

## Bench (directional, n=1, single iteration)

| Workload | Per-token oracle | Batched | Speedup |
|----------|-----------------:|--------:|--------:|
| Prefill N=256 (Phase F headline) | 12.85 tok/s | 25.60 tok/s | **1.99×** |
| Decode (kv_start=32, decode_n=32) | 12.09 tok/s | 14.22 tok/s | **1.18×** |

**Not protocol-compliant** — n=1, no reboot, no power-mode control per
`feedback_bench_discipline.md`. Use for in-session direction only; a
follow-on bench session with the proper protocol gives the headline
number for any commit message that wants it.

## Tests (all cosine=1.0 unless noted)

```bash
cd ~/Projects/moeflux

# Synthetic batched primitives (sessions 1-3 + Phase A regression):
cargo test --release -p moeflux --features model-qwen3-6-35b-a3b \
  --test batched_diff_oracle -- --ignored --nocapture --test-threads=1
# 9 tests, all cosine=1.0.

# Canary battery (real artifacts):
cargo test --release -p moeflux --features model-qwen3-6-35b-a3b \
  --test diff_oracle -- --ignored --nocapture --test-threads=1 \
  state_round_trip_rust eval_token_matches_c_single_step \
  eval_prompt_matches_per_token_oracle slot_reuse_race_regression_rust \
  state_load_rust_from_c_save eval_prompt_matches_c_multi_token \
  eval_prompt_chunked_matches_eval_prompt_whole_prompt \
  prompt_cache_start_pos_nonzero_matches diag_b2_eval_prompt_chunk_1

# Benches (directional):
cargo test --release -p moeflux --features model-qwen3-6-35b-a3b \
  --test diff_oracle -- --ignored --nocapture --test-threads=1 \
  bench_batched_eval_prompt_vs_per_token \
  bench_decode_per_token_vs_batched_n1
```

`prompt_cache_start_pos_nonzero_matches` is **bit-exact** (max_abs=0.000e0).
The C-side comparison `eval_prompt_matches_c_multi_token` matches at
cosine=1.0, max_abs_diff ≈ 2e-5 — strongest cross-impl validation.

## Pre-existing failures (not session-4 regressions)

- `state_load_c_from_rust_save` — Rust → C snapshot wire compat broken
  before Phase A. Documented at `future_work_state_load_c_from_rust_save.md`.
- `resuming_prefill_after_seq_rm_matches_full_prefill` — linear-attn
  recurrent-state truncation on `memory_seq_rm`. Same memo.

Both fail on pre-Phase-A `48c8562`. The Rust-only state path
(state_save/load roundtrip) is bit-exact per Phase D's
`prompt_cache_start_pos_nonzero_matches`.

## Phase G follow-up cleanup (deferred to session 5)

`eval_token` no longer uses these (they were the per-token GPU SDPA
fast path the oracle relied on for decode). Now dead code for
production Gqa decode:

- Three kernels: `attn_scores_batched`, `attn_softmax_batched`,
  `attn_values_batched`.
- Persistent buffers in `LayerForwardBuffers`: `gpu_attn_q`,
  `gpu_attn_scores`, `gpu_attn_out`, `gpu_attn_gate`, plus the
  `gpu_kv_k` / `gpu_kv_v` per-full-attn-layer mirrors (bounded by
  GPU_KV_SEQ=8192).
- The GPU SDPA fast-path branch in `post_attention_pre_moe`
  (`gpu_attn_args=Some` block).
- The KV-mirror writes in `full_attn_pre_moe_layer_forward`.
- The `kv_len >= 32 && kv_len < GPU_KV_SEQ` gate.

Net: ~150 LOC + ~5-10 MB persistent buffer per Ctx. Deferred because
the buffer-field removal touches LayerForwardBuffers's pub field
layout, which is part of the snapshot v2 wire format. Cleanly
separable into a session-5 "remove dead Gqa decode fast-path + shrink
Ctx" commit.

The B3-era helper `post_attention_post_o_proj_to_intermediates` is
also currently `#[allow(dead_code)]` — kept as a forward-looking
building block in case a future decode path wants batched o_proj +
per-token shared FFN inside one cmdbuf.

## Other follow-ups noted during session 4

- **Per-chunk scratch buffer hoisting.** `batched_full_attn_layer_forward`
  allocates ~1.5 GB scratch per call at production CHUNK_SIZE=8192 ×
  40 layers = 60 GB/chunk of allocator churn. Allocate once per chunk
  in the orchestrator, pass `&mut Scratch` into the layer function.
  First session-5 perf-investigation task.
- **Per-token CPU swiglu_gate.** Phase 3a applies sigmoid_gate per
  token on CPU. Element-wise → could batch as a flat dispatch on GPU.
  Likely small win.
- **Per-token post-attn rms_norm + gate logits matvec + shared-gate
  scalar matvec.** Phase 3c keeps these per-token via
  `post_attention_residual_norm_route`. The 8-bit gate matvec doesn't
  have a batched variant (session 2 skipped 8-bit batched as "small
  per-token dispatch cheap"); revisit if profiling flags it.
- **`moe_combine_residual_n_tokens` kernel.** Final per-token combine
  (h_mid + moe_sum + sigmoid(shared_gate) * shared_out) is on CPU
  today. Element-wise — could be flat-dispatched. Small win at
  N=256, larger at N=8192.
- **Protocol-compliant Phase F bench.** Reboot, n≥3, high-perf;
  8k cold + warm on a3b vs main, vs llama.cpp. Headline number for
  the prefill north star.
- **Prefill progress callback** for drama_llama — Mike's session-4
  ask, captured at `future_work_prefill_progress_callback.md`.

## Architecture conventions established

- **`*_pre_*` / `*_post_*` factoring**: when a per-token forward
  needs a batched counterpart, factor at clean buffer-boundary points
  (pre-SDPA / post-SDPA, pre-MoE / post-MoE, pre-o_proj / post-o_proj).
  Each factored helper takes & returns either GPU buffers (state in
  LayerForwardBuffers) or host slices (intermediates), never both
  mixed.
- **`PostAttnIntermediates` as host carrier**: routing indices +
  weights + shared_gate_score travel as host data between batched
  GPU work units. GPU buffers (h_mid, h_post, shared_out) stay in
  LayerForwardBuffers and the caller snapshots them as needed.
- **Per-thread test override pattern**: thread-local `Cell<Option<T>>`
  + `pub fn set_*_for_test(Option<T>)` is the way to expose tunables
  to integration tests without polluting production callers.
  `BATCHED_CHUNK_SIZE` uses this; future tunables follow suit.
- **Always restage shared-storage buffers across iteration boundaries.**
  Each per-token iteration that reads + writes shared-storage GPU
  buffers must restage all inputs explicitly — "still has the last
  iteration's value" is an unreliable optimization. The B2 bug was
  exactly this. Restage cost is host memcpy, dominated by the GPU
  dispatch that follows.

## Calibration

- **Six phases in one session.** A, B1+C, D, F, B2, B3, B4, G — plus
  E and H confirmed. Estimated as 3 sessions at start; the leverage
  came from doing Phase A (no-behaviour-change refactor) first and
  letting subsequent phases compose on the seams it exposed.
- **Cosine gate discipline.** Every B sub-step + G ran the canary
  battery before committing. Caught the B2 buffers.input bug
  immediately; without the per-sub-step gate it would have been
  diagnosed downstream of more code.
- **Bench discipline calibration.** Mike's `feedback_bench_discipline.md`
  rules out n=1 single-run claims as headline. The directional bench
  (n=1, no reboot) is fine for in-session GO/NO-GO decisions; the
  headline numbers wait for the protocol-compliant session.
- **Mike's "don't defer" intervention mid-session was the right call.**
  When I started reasoning about scope (B2 vs B3 vs B4 deferral), it
  was self-doubt creeping in, not a real capability question. Pushed
  through and shipped everything. Saving the prefill-progress-callback
  idea + the state_load_c future-work memos is the right shape of
  "saving things"; deferring whole phases on the same arc isn't.
- **Decode through batched is FASTER**, not just equivalent. The
  GPU SDPA fast-path's cmdbuf-fold-with-o_proj advantage is dwarfed
  by the orchestrator-level wins (deferred ring overhead, prefetch
  state machine, per-layer pipeline-fetch churn) — even at N=1. The
  per-token oracle deserves to be the diff target, not the
  production path.
