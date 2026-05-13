# Qwen batched prefill — session 4 outcome

Plan-of-record: `/Users/mdegans/.claude/plans/session4_batched_prefill_integration.md`.
Session-3 outcome (entry point): [`qwen_batched_prefill_session3_landed.md`].

**Headline:** batched prefill landed end-to-end on a3b. Phases A, E, B1,
D, F, H shipped on moeflux main (commits d1a0af4, 94aedba, c1f9b52,
plus the bench commit). Phase B2-B4 deferred — B1's MoE-only batching
already gives **1.82× speedup at N=256** vs the per-token oracle on a
directional single-run measurement (24.78 tok/s vs 13.59 tok/s,
cosine=1.0). Phase G deferred (decode regression bench gate not run).

## What landed

Four commits on moeflux main; one cleanup commit on drama_llama v0.8.0.

### Phase A (`d1a0af4`) — post_attention_tail split (no behaviour change)

`post_attention_tail` decomposed into:
- `post_attention_pre_moe(...) -> PostAttnIntermediates` — CMD2+3
  (o_proj → residual → post-attn rms_norm → gate logits → shared FFN)
  + CPU MoE router + shared-gate readback. Returns routing indices,
  weights, shared_gate_score.
- `moe_dispatch_per_token(intermediates, ...)` — K-expert dispatch
  + combine + optional chain hook. Reads from intermediates instead
  of recomputing routing.

`post_attention_tail` itself becomes a thin wrapper. Behaviour
identical for the per-token path; the seam exposed lets the batched
orchestrator collect intermediates across N tokens before dispatching
MoE in batch.

### Phase B1 (`94aedba`) — batched_full_attn_layer_forward + orchestrator

Three new top-level pieces:
- `full_attn_pre_moe_layer_forward` — sibling to `full_attn_layer_forward`
  that runs the per-token full-attn forward up through `post_attention_pre_moe`.
- `batched_full_attn_layer_forward` in `full_attn_forward.rs` — per-token
  pre-MoE loop captures intermediates + h_mid + h_post + shared_out to
  host stacks, builds joint N×K_active expert-bucket CSR, runs
  `encode_moe_batched_permute_fuse` once for the whole chunk, then
  per-token CPU combine.
- `step_internal_batched_gqa` in `mod.rs` — the canonical
  orchestrator. Routes full-attn layers to the batched path,
  linear-attn layers to a per-token loop with sync deferred drain.
  MLA variants fall back to `step_internal_per_token_oracle` (batched
  MLA = session 5+).

**No deferred ring in the batched full-attn path. No prefetch.**
Linear-attn fallback uses its own deferred + sync drain per token —
chunkwise linear-attn is future work without losing correctness here.

### Phase D (`c1f9b52`) — chunkwise execution + cache-hit tests

`step_internal` wraps `step_internal_batched_gqa` in a chunked loop:
prompts > `BATCHED_CHUNK_SIZE` (default 8192) split into chunks. Only
the last chunk's last token emits logits. `start_pos` advances per
chunk for cumulative KV state.

Test-only hook: `set_batched_chunk_size_for_test(Option<usize>)` —
per-thread override. Production callers don't use it.

Two new tests in `tests/diff_oracle.rs`:
- `eval_prompt_chunked_matches_eval_prompt_whole_prompt` (chunk=4,
  16-token prompt → 4 chunks): cosine=1.0 vs per-token oracle.
- `prompt_cache_start_pos_nonzero_matches`: prefix → state_save →
  reset → state_load → eval_prompt(suffix, start_pos=4). Compared
  to full-prompt control: **cosine=1.0, max_abs_diff=0.000e0
  (bit-exact)**.

### Phase F (bench) — 1.82× speedup at N=256

`bench_batched_eval_prompt_vs_per_token` test in `tests/diff_oracle.rs`.
Single-iteration directional bench (not protocol-compliant — see
`feedback_bench_discipline.md` for the proper bench). At N=256
synthetic prompt:

| Path | Wall | tok/s |
|------|-----:|------:|
| per-token oracle | 18.83 s | 13.59 |
| batched eval_prompt | 10.33 s | 24.78 |

**Speedup: 1.82×. Sanity cosine = 1.0.**

Headline-quality measurement (n≥3, reboot, high-perf) is for a
follow-on bench-only session.

### Phase H — prefetch confirmed decode-only

Verified by grep: `prefetch.dispatch` and `prefetch.predict_for`
only appear in `step_internal_per_token_oracle`. The batched path
reads expert blobs synchronously via `encode_moe_batched_permute_fuse`'s
own pre-load pass; no prefetch hook fires.

## What didn't land (deferred)

- **Phase B2** — batched SDPA. Requires splitting
  `full_attn_pre_moe_layer_forward` into pre-SDPA and post-SDPA halves;
  not pulling enough weight relative to B1's MoE-permute-fuse win to
  justify the refactor scope in-session.
- **Phase B3** — batched Q/K/V/O projections.
- **Phase B4** — batched shared expert + RMSNorm + GPU combine.
- **Phase G** — remove `GPU_KV_SEQ=8192` + per-token attn kernels.
  Gated on a decode regression bench (not run in-session).
- Per-chunk **scratch buffer hoisting** — `batched_full_attn_layer_forward`
  allocates ~1.5 GB scratch per call at N=8192 (per layer = 40 ×
  1.5 GB churn). Works at small N (test passed), but at production
  CHUNK_SIZE=8192 this is allocator churn that should be amortized
  across layers + chunks. First session-5 perf-investigation task.

These are session-5+ work. The headline B1+D wins this session.

## Verification commands

```bash
cd ~/Projects/moeflux

# Synthetic batched primitives (sessions 1-3 + Phase A regression):
cargo test --release -p moeflux --features model-qwen3-6-35b-a3b \
  --test batched_diff_oracle -- --ignored --nocapture --test-threads=1
# 9 tests, all cosine=1.0.

# Real-artifact tests (real weights, ~3-30s each):
cargo test --release -p moeflux --features model-qwen3-6-35b-a3b \
  --test diff_oracle -- --ignored --nocapture --test-threads=1 \
  state_round_trip_rust eval_token_matches_c_single_step \
  eval_prompt_matches_per_token_oracle slot_reuse_race_regression_rust \
  state_load_rust_from_c_save eval_prompt_matches_c_multi_token \
  eval_prompt_chunked_matches_eval_prompt_whole_prompt \
  prompt_cache_start_pos_nonzero_matches

# Directional bench:
cargo test --release -p moeflux --features model-qwen3-6-35b-a3b \
  --test diff_oracle -- --ignored --nocapture --test-threads=1 \
  bench_batched_eval_prompt_vs_per_token
```

All pass at session end.

## Known pre-existing failures (NOT session-4 regressions)

- `state_load_c_from_rust_save` — pre-existing wire compat (Rust →
  C snapshot rejected by C side). Documented at
  `future_work_state_load_c_from_rust_save.md`. Rust-only state path
  bit-exact per Phase D's `prompt_cache_start_pos_nonzero_matches`.
- `resuming_prefill_after_seq_rm_matches_full_prefill` — pre-existing
  linear-attn recurrent-state truncation on `memory_seq_rm`.
  Documented in same memo. Phase D's prompt-cache scenario uses
  `state_save`/`state_load` (not seq_rm), so it routes around this.

Both fail on pre-Phase-A `48c8562` too.

## Architecture conventions established

- **`*_pre_moe` / `*_dispatch` split**: when a per-token forward
  needs a batched counterpart, factor the pre-MoE bookkeeping into a
  helper that returns intermediates (host data) and pass GPU-buffer
  refs through `LayerForwardBuffers`. The batched orchestrator runs
  the pre-MoE helper in a per-token loop, captures intermediates,
  then dispatches MoE in batch + combines per-token. No deferred
  ring or prefetch on the batched side — that's decode-only after
  Phase H.
- **Test-only thread-local overrides**: `set_batched_chunk_size_for_test`
  is the model. A pub function in `riir/mod.rs` sets a thread-local
  Option; the production code reads via `batched_chunk_size()` which
  defaults to the const when unset. Tests bracket with
  `set_..._for_test(Some(N))` ... `set_..._for_test(None)` and use
  `panic::catch_unwind` to restore on panic.
- **`BATCHED_CHUNK_SIZE = 8192`** as the default for a3b on 96 GB
  unified RAM + ~1 GB/s SSD profile. Will be tuned by the proper
  Phase F bench in a session-5 measurement run.

## Calibration

- **Phase A was the right scope to land first.** The `post_attention_tail`
  split was no-behaviour-change → green canary suite → confident to
  layer Phase B on top.
- **B1 + D went smoothly because Phase A exposed the right seams.**
  `full_attn_pre_moe_layer_forward` fell out of the post_attention_tail
  refactor naturally — same shape, one layer up.
- **Skipping B2-B4 was correct calibration**, not a punt. The 1.82×
  win at N=256 comes entirely from the I/O-bandwidth-bound MoE blob
  reads — exactly what session 3's Phase 4 plan predicted ("the
  prefill I/O win"). B2-B4 add GPU-compute parallelism on already-warm
  per-token work, which is a smaller win relative to scope. Defer
  until measurement confirms where the next bottleneck is.
- **Triage of pre-existing failures worked.** Both
  `state_load_c_from_rust_save` and
  `resuming_prefill_after_seq_rm_matches_full_prefill` were found
  during canary runs and bisected to pre-Phase-A. Documented in
  future-work memo. Didn't block the session.
- **Future-work memo for the prefill progress callback** (Mike's
  mid-session suggestion) lives at
  `future_work_prefill_progress_callback.md`. Will need it for
  40-60k Agora prompts; deferred to a session that has the API
  changes as the headline.
- **Drama_llama side**: zero functional changes needed (Phase E
  audit). The chunking is transparent at the `Decoder::prefill`
  boundary. Two unrelated commits (chat_template permissive env +
  forget_pos tracing) landed at session start — these were "landed"
  in a prior session per memory but had never actually been
  committed.

## What's next (session 5+)

In rough priority order:

1. **Protocol-compliant Phase F bench** (reboot, n≥3, high-perf):
   8k cold + warm on a3b vs main, vs llama.cpp. Headline number.
2. **Per-chunk scratch buffer hoisting** — biggest perf opportunity
   identified mid-session. Allocate scratch once per chunk in the
   orchestrator, pass `&mut` into `batched_full_attn_layer_forward`.
3. **Phase B2** (batched SDPA) — refactor pre_moe into pre-SDPA /
   post-SDPA halves. Cosine gate via existing
   `eval_prompt_matches_per_token_oracle`.
4. **Phase B3** (batched Q/K/V/O projections).
5. **Phase B4** (batched shared FFN + batched RMS + GPU combine
   kernel `moe_combine_residual_n_tokens`).
6. **Phase G** (remove GPU_KV_SEQ=8192 fast path) — gated on decode
   regression bench < 5% at small kv_len.
7. **Prefill progress callback** for drama_llama — Mike's session-4
   ask, captured at `future_work_prefill_progress_callback.md`.
