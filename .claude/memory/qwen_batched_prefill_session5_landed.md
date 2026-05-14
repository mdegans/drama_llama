# Qwen batched prefill — session 5 outcome

Entry-point: [`qwen_batched_prefill_session5_plan`].

**Headline.** Phase 1 (batched linear-attn) shipped the biggest
per-call win of the session — directional **3.42× prefill speedup
at N=256**, **~36 prefill tok/s** on the 992-token real workload
(was ~21 post-session-4). Phase 2 (mmap env-toggle) landed
neutral but kept as a low-RAM/iPhone path. Phase 3 (prefetch in
batched orchestrator) shipped plumbing-only — the `eval_token`
re-route stayed reverted on discovery that batched-N=1 is still
~10% slower than the per-token oracle even with prefetch
(cross-layer deferred-K pipelining gap, not prefetch). Phase 4
(scratch hoist) and Phase 5 (dead-code cleanup) deferred after
head-to-head profile showed allocator churn isn't the bottleneck
and Phase 5's premise (eval_token re-routed) didn't fully hold.

**Capstone discovery.** Head-to-head bench against llama.cpp's
recent checkout (the one bundled with `llama-cpp-sys`) revealed a
**26-32× prefill gap** that motivates session 6's graph-mode
refactor. See [`qwen_graph_mode_session6_plan`] for the plan.

## Phases shipped (moeflux main)

| Commit | Phase | What landed |
|--------|-------|-------------|
| `6a3ef1a` | 1 | `batched_linear_attn_layer_forward` (N×5 recurrent kernels in 1 cmdbuf, batched projections, batched o_proj, batched shared FFN, batched MoE permute-fuse). 3.42× directional prefill speedup at N=256. Cosine=1.0 across 9-canary battery. |
| `91d3fd8` | 2 | `MOEFLUX_EXPERT_IO=pread|mmap` env toggle. mmap mode: per-layer mmap + `newBufferWithBytesNoCopy` Metal buffer. Neutral on M2 Max (page cache saturates pread), kept for low-RAM device cases. `encode_moe_batched_permute_fuse` signature changed to `&[ExpertRef] = &[(&Buffer, u64)]`. |
| `e0a4031` | 3 (partial) | Prefetch state machine plumbed into `step_internal_batched_gqa` and both batched layer forwards. `PrefetchEnv<'a>` struct, set-based hit matching, `record_outcome`/`record_actual`. Gated to N=1 in pread mode. Cosine=1.0. `eval_token` re-route reverted on discovery (see below). |
| `81c8b71` | 3 (follow-up) | `MOEFLUX_EVAL_TOKEN=oracle|batched` env toggle to keep both paths available. Default `oracle`. |

## Discovery — why eval_token stayed on oracle

In-process A/B (`bench_decode_per_token_vs_batched_n1`) post-Phase
3: batched-N=1 still 10.5% slower than oracle (7.63 vs 8.53
tok/s, decode_hit=0.395 on both — prefetch is firing correctly).

The remaining gap is the **oracle's cross-layer deferred-K-expert
pipelining**: layer N's async K-expert dispatch overlaps with
layer N+1's CMD1+CMD2+3 GPU work via the depth-2 deferred ring.
The batched permute-fuse path commits-and-waits synchronously per
layer at N=1, so there's no compute to overlap the K-expert wait
with. Prefetch alone is necessary but not sufficient.

Per `feedback_pivot_on_discovery`: stop and re-scope instead of
push through. The plumbing stays — it works at N=1, is cosine=1.0
across canaries, and is ready for the day cross-layer pipelining
lands inside the batched path. Session 6's graph-mode refactor is
that day.

## Head-to-head — 2026-05-14 baseline

Same M2 Max, same prompt, both at 4-bit (moeflux internal 4-bit;
llama.cpp `Q4_K_S`, file
`~/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf`):

| Workload | moeflux | llama.cpp | gap |
|---|---:|---:|---:|
| **992 prefill** (n=3) | 36.8 prefill_tok/s | 970 prefill_tok/s | **26×** |
| **16k prefill** (n=1) | 27.1 prefill_tok/s | 857 prefill_tok/s | **32×** |
| **essay+128 mixed** (n=3) | 8.23 tok/s | 22.78 tok/s | **2.77×** |

**No 8k wall on moeflux** — chunking + tiled SDPA work clean past
8k. Mike's GPU monitor: 90% sustained on 16k prefill.

bench.py `--backend llama-cpp` plumbing added locally (gitignored
script; useful for cross-impl A/B).

## Architectural diagnosis (for session 6)

llama.cpp's recent Metal scheduler at
`ggml/src/ggml-metal/ggml-metal-context.m:438..550`:

1. Full forward built as a single GGML compute graph
   (`build_qwen3moe` in `src/models/qwen3moe.cpp`).
2. Graph splits into ≤8 MTLCommandBuffers per chunk
   (`GGML_METAL_MAX_COMMAND_BUFFERS`; typically 1-2 on Apple Silicon).
3. Parallel encoding via `dispatch_apply` across `n_cb` threads.
4. Single `commit_and_wait` at chunk end.

moeflux's session-5 batched path: ~200+ `commit_and_wait`s per
chunk (40 layers × ~5 phases). At ~50–200 μs CPU↔GPU toggle per
commit, that's ~400–1600 ms of pure sync overhead per chunk.

**Mike on the architectural shift (2026-05-14):** "this crate was
mostly to run models larger than could fit in memory at once but I
suspect these architectural wins will be good for regardless."

Phase 4 (scratch hoist) becomes mostly moot under graph mode —
allocator churn isn't the bottleneck; commit overhead is. Phase 5
(dead-code cleanup) needs eval_token re-routed first.

## Tests (all cosine=1.0 unless noted)

```bash
cd ~/Projects/moeflux

# Canary battery — green in oracle mode (default) and batched mode:
cargo test --release -p moeflux --features model-qwen3-6-35b-a3b \
  --test diff_oracle -- --ignored --nocapture --test-threads=1 \
  state_round_trip_rust eval_token_matches_c_single_step \
  eval_prompt_matches_per_token_oracle slot_reuse_race_regression_rust \
  state_load_rust_from_c_save eval_prompt_matches_c_multi_token \
  eval_prompt_chunked_matches_eval_prompt_whole_prompt \
  prompt_cache_start_pos_nonzero_matches diag_b2_eval_prompt_chunk_1
# 9/9 cosine=1.0

# Synthetic batched primitives:
cargo test --release -p moeflux --features model-qwen3-6-35b-a3b \
  --test batched_diff_oracle -- --ignored --nocapture --test-threads=1
# 9/9 cosine=1.0
```

`prompt_cache_start_pos_nonzero_matches` bit-exact (max_abs=0).
C-side `eval_prompt_matches_c_multi_token` cosine=1.0,
max_abs_diff ≈ 2e-5.

## Pre-existing failures (not session-5 regressions)

- `state_load_c_from_rust_save` — Rust → C snapshot wire compat;
  pre-existing pre-Phase-A on moeflux main. See
  `future_work_state_load_c_from_rust_save`.

## Architecture conventions established this session

- **`PrefetchEnv<'a>` borrow-bundle pattern.** When an optional
  per-call subsystem needs multiple borrows from RsCtx, bundle
  them in a struct held as `Option<PrefetchEnv<'_>>` argument
  rather than threading 3+ optional parameters. Tested at the
  N=1 prefetch site; carry over for cross-layer pipelining in S6.
- **Env-flag toggles via `std::sync::OnceLock`.** For
  `MOEFLUX_EVAL_TOKEN` and `MOEFLUX_EXPERT_IO` — read once at
  first call, cached. Sibling pattern: `ExpertIoMode::from_env()`
  read at `ExpertFiles::open`. Both work, OnceLock is simpler
  when there's no natural init site.
- **Stacked-buffer + per-token-offset binding** in batched
  recurrent kernels. The kernel encoders gained
  `qkv_in_off: u64` / `alpha_in_off: u64` / `z_off: u64` /
  `output_off: u64` parameters; the per-token oracle still passes
  `0`. Metal serialises encoder order within one cmdbuf so the
  conv_state / delta_state recurrence chain is correct without
  per-token commit. **This is the Phase 1 win pattern** — same
  shape applies to any per-token sequential dependency.

## Calibration

- **Plan estimate: all 5 phases in 1 session.** Actual: 3 phases
  + 2 deferred + head-to-head bench. The discoveries (Phase 3's
  decode regression, Phase 5's invalid premise after the revert)
  ate the rest of the session.
- **`feedback_pivot_on_discovery` paid off twice this session.**
  Phase 3 partial (don't push through wrong premise) and Phase 5
  skip (don't accrete trivial cleanup when the premise changed).
  Both decisions saved context for the head-to-head bench, which
  was the actual high-leverage finding.
- **bench.py `--backend llama-cpp` plumbing** is the artifact that
  makes session 6's per-phase A/B measurable. Worth ~30 min of
  build/wire work for the rest of the refactor's measurement
  protocol.
- **Mike's "Profile now (skipping Phase 5)" call** with the 16k
  prompt request — surfaced the 32× gap that motivated session 6.
  Without that diversion the session would have shipped a small
  cleanup commit and missed the strategic insight entirely.

## Forward-work pointers

- [Session 6 plan](qwen_graph_mode_session6_plan.md) — graph-mode
  refactor.
- Phase 4 (scratch hoist) — deferred; reconsider only if graph
  mode doesn't naturally eliminate allocator churn.
- Phase 5 dead-code (per-token GPU SDPA fast path) — folds into
  Phase D of the session 6 plan, after eval_token re-route lands.
