# Session 8 handoff — notes from session-7-me to session-8-me

**Entry:** [`qwen_graph_mode_session7_partA_landed.md`](qwen_graph_mode_session7_partA_landed.md)
**Locked design:** [`qwen_graph_mode_session7_plan.md`](qwen_graph_mode_session7_plan.md)

## Top 5 things to know

### 1. Write the diff test BEFORE you trust a kernel arm

The single most useful catch this session: `Op::SwigluFusedBatched`
was missing the `K` arg in the Metal dispatch. The shape error was
invisible from reading the code — the diff test reported cos=0.0
in 0.07s and I fixed it in 60s. **Pattern**: for every new
`encode_op` arm in S7-6/S7-7, add the per-Op diff test entry to
`graph_diff_oracle.rs` BEFORE assuming the wiring is right.
Synthetic-input diff tests are cheap and they catch arg-count /
arg-order / buffer-binding bugs that compile fine.

### 2. The 7 `todo!()` arms have named deferral targets

`graph.rs:1640..1670` (approx). When you hit them in producer
rewrites, the choice of S7-6 vs S7-7 is in the comment string.
Don't blanket-wire them — each one wants context the producer
site naturally provides (workspace BufIds for LmHead, per-token
loop structure for the linear-attn variants, multi-pipeline
composition for MoeBatchedPermuteFuse).

### 3. `gated_delta_recurrence` needs splitting

Known dirty spot from S7-2. The existing helper fuses
ComputeDecayBeta math with the recurrence. Our Op vocabulary
separates them. CpuBackend's `Op::GatedDeltaNetStepNTokens` arm
currently passes dummy `a_log`/`dt_bias` because the recurrence-
only API doesn't exist. **Fix in S7-6**: add
`gated_delta_recurrence_supplied(g_decay, beta_gate, q, k, v,
v_heads, k_heads, key_dim, value_dim, ssm_state, out_values)` to
`linear_attn.rs`, refactor existing `gated_delta_recurrence` to
call it. CPU arm switches to the new helper.

### 4. `graph.rs` will need splitting after S7-5

Currently ~2070 LOC. S7-5 adds ~250 (lifetime coloring). At
~2300 LOC, splitting into a directory module becomes worthwhile:

```
crates/moeflux/src/riir/graph/
├── mod.rs       (types: BufId, WeightRef, Op, Graph, traits)
├── cpu.rs       (CpuBufferPool + CpuBackend + byte-variant helpers)
├── metal.rs     (MetalBufferPool + MetalBackend)
└── lifetime.rs  (analyze_lifetimes + greedy_color)
```

The split is purely mechanical (`git mv` then update `pub mod`
declarations). Do it as part of the S7-5 commit OR as a small
precursor commit before S7-5.

### 5. WeightFile resolution: bytes_at lives, name-based helpers stay

The session added `WeightFile::bytes_at(offset, len)` for the
graph compiler's WeightRef-based path. Existing CPU helpers
(`rms_norm_cpu`, `lm_head_cpu`, `rms_norm_gated`) still take
tensor names — they're called by the per-token oracle path. I
added byte-variant inline helpers in graph.rs (`rms_norm_bf16_n_tokens_cpu`,
`gated_rms_norm_n_tokens_cpu`, etc.) instead of refactoring all
callers. **In S7-7**, when LmHead's encode_op arm gets wired
(needs workspace BufId in the Op shape), promote those inline
helpers to a `cpu_byte_helpers.rs` sibling module if they grow
much further. Don't refactor existing name-based helpers — they
have other users.

## Verification commands (paste-ready)

```bash
cd ~/Projects/moeflux

# Quick sanity (lib + graph unit tests):
cargo build -p moeflux --features model-qwen3-6-35b-a3b
cargo test -p moeflux --features model-qwen3-6-35b-a3b --lib graph::
# Expected: 13 passed (8 unit + 5 CpuBufferPool)

# S7-4 acceptance gate:
cargo test --release -p moeflux --features model-qwen3-6-35b-a3b \
  --test graph_diff_oracle -- --ignored --nocapture --test-threads=1
# Expected: 3 passed, all cos=1.000000000

# Canary 9/9 (run after producer rewrites land):
cargo test --release -p moeflux --features model-qwen3-6-35b-a3b \
  --test diff_oracle -- --ignored --nocapture --test-threads=1 \
  state_round_trip_rust eval_token_matches_c_single_step \
  eval_prompt_matches_per_token_oracle slot_reuse_race_regression_rust \
  state_load_rust_from_c_save eval_prompt_matches_c_multi_token \
  eval_prompt_chunked_matches_eval_prompt_whole_prompt \
  prompt_cache_start_pos_nonzero_matches diag_b2_eval_prompt_chunk_1
```

## What I'd start with

Per the executable plan, S7-5 is next: lifetime analysis +
interval-coloring pool optimization. Concrete starting move:

1. Run `cargo build` + `cargo test --lib graph::` to confirm
   clean state from session 7.
2. Decide on graph.rs split timing (probably do it FIRST as a
   precursor commit, so S7-5 lands in `graph/lifetime.rs`
   cleanly — keeps the S7-5 commit focused on the algorithm
   rather than a big move).
3. Implement `analyze_lifetimes(&graph) -> LivenessMap` using
   the `Op::reads()` / `writes()` hooks already in place.
4. Implement `greedy_color(&liveness) -> ColoringMap` (linear-
   scan register-allocation-style).
5. Wire the coloring map into both pools via an `Option<
   ColoringMap>` field + `commit_plan(&graph)` method.
6. New test `graph_metal_matches_cpu_colored` — same as S7-4
   tests but with coloring on. Per-Op outputs must still pass
   cosine ≥ 0.9999, AND `pool.physical_buffer_count() <
   bufid_count` (real aliasing happened).

If S7-5 lands in ~1 session-third, push through to S7-6 (linear-
attn producer rewrite) the same session — the linear-attn path
has 5 of the 7 deferred `todo!()` arms and rewriting it forces
the wire-up. That naturally produces a 50% bump in MetalBackend
coverage.

## What I'd skip

- Don't rush to wire ALL 7 todo!() arms before producer
  rewrites. The deferral is intentional — wire them at the
  producer site where you can see the calling context.
- Don't run canary 9/9 between graph compiler S7-5 steps. The
  graph code is still not on the active forward path; canary
  is unchanged. Save it for after S7-6/S7-7 when producer
  rewrites land.

## Calibration check

Session 7 shipped 6 commits and 2845 LOC at ~1/3 context (333k/
1m). The "be ambitious, pause on fatigue" calibration held: I
was offered `/context` once and used it to confirm we had
budget. Don't pre-emptively wrap on perceived context pressure
when you have 67% free.
