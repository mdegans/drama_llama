# Qwen batched prefill — session 1 outcome (discovery + Phase 0)

Plan-of-record was [async-toasting-mitten](/Users/mdegans/.claude/plans/async-toasting-mitten.md). Session goal: CPU batched primitives + diff oracle as scaffolding for GPU batched prefill (session 2). **What actually landed: Phase 0 refactor + a load-bearing architectural finding that invalidates the rest of the plan.** Pivoted mid-session.

## What landed

### Phase 0 — diff helpers factored to `tests/common/diff_helpers.rs`

- `argmax`, `topk`, `jaccard`, `cosine_sim`, `assert_logits_close`, `TOPK_K`, `TOPK_JACCARD_MIN`, `COSINE_SIM_MIN`, `COSINE_FLOOR`, `REL_DIFF_FLOOR` moved from `tests/diff_oracle.rs` (lines 1141–1233) to a new `tests/common/diff_helpers.rs`.
- Path resolution (`artifacts_dir`, `root_dir`) and the new `A3BPaths` struct + `default_a3b_paths()` helper also live in `common/diff_helpers.rs`.
- `open_backend<B: DiffBackend>` in `diff_oracle.rs` now consumes `default_a3b_paths()` instead of inlining path logic.
- `tests/common/mod.rs` re-exports `pub mod diff_helpers;`.
- `#![allow(dead_code)]` on the diff_helpers module because per-test inline `const COSINE_FLOOR` redeclarations shadow the module constants — keeping them is intentional minimal-touch.

Diff stat: 2 files changed, 16 insertions(+), 122 deletions(-). One new file (~140 LOC). Zero production code touched. Existing diff oracle tests must still pass identically (refactor is structurally non-semantic — `cargo check --tests` clean with only pre-existing warnings).

### Discovery — Qwen3.6-A3B has no per-token CPU layer forward

The plan-of-record assumed a CPU batched layer forward could be built by extending an existing per-token CPU layer forward. **This holds for the Cogito-V2 (DeepSeek-V3 / MLA) path** — `mla_attn_layer_forward_cpu` (`mla_attn_cpu.rs:129`), `deepseek_moe_cpu` (`moe_cpu.rs:65`), `dense_mlp_swiglu_cpu` (`mlp_cpu.rs:47`), `shared_expert_swiglu_cpu` (`mlp_cpu.rs:62`) all exist and compose to a per-token CPU forward via `step_internal_mla_cpu` (`mod.rs:1244`).

**It does not hold for Qwen3.6-A3B.** The per-token Qwen3 path is `full_attn_layer_forward` (`full_attn_forward.rs:80`) and `linear_attn_layer_forward` (`linear_attn_forward.rs:421`), both of which dispatch to GPU. There is no `step_internal_cpu` equivalent for Qwen3; `step_internal` goes straight to the GPU primitives via deferred-ring orchestration.

Available CPU primitives for Qwen3 are lower-level (per the `DiffBackend` trait at `tests/diff_oracle.rs:84`): `rms_norm_cpu`, `rms_norm_per_head_cpu`, `apply_rotary_emb`, `sdpa_cpu`, `lm_head_cpu`, `moe_router_cpu`, `gated_delta_recurrence_cpu`, `conv1d_step_cpu`, `rms_norm_bare_cpu`, `rms_norm_gated_cpu`. No composed layer forward. Composing them into a per-token Qwen3 CPU forward from scratch is a multi-session effort (each layer kind — full-attn + linear-attn — needs its own composer with proper KV cache state handling).

### Why this matters

The plan's CPU scaffolding was load-bearing for two reasons:
1. **Bit-exact CPU reference for GPU kernels.** Session 2 would diff each new GPU batched kernel against its CPU counterpart, kernel by kernel. Without the CPU baseline, GPU diff falls back to "GPU batched vs C tokenwise loop" — a wider blast radius when something breaks.
2. **Permute-and-fuse MoE algorithm validation.** The new MoE permute-and-fuse dispatch is the algorithmically interesting piece. Without a CPU implementation, we can't validate the permute logic in isolation; we'd only see end-to-end output mismatch.

For Cogito-V2, the plan as written would have worked — primitives compose into a per-token CPU forward we can extend. For Qwen3.6-A3B, the plan would have required first building the per-token CPU forward (multi-session) before any batched work.

## Why pivot mid-session was the right call

Three options on discovery:
- **A. Build per-token Qwen3 CPU forward from scratch this session**, then batch in session 2+. Multi-session scope. Pushes batched prefill ~3 sessions further out.
- **B. Switch target to Cogito-V2** (which has CPU primitives). Drops Qwen3.6-A3B which is the Agora prefill target.
- **C. Drop CPU scaffolding; go straight to GPU batched against C tokenwise oracle.** No CPU intermediate. Matches Mike's stated GPU-first preference (`if we can, going straight to GPU would be ideal`).

Picked **C**. Lands Phase 0's real cleanup, captures the discovery, sets up session 2 for direct GPU batched work. No half-finished CPU code (which is the historical failure mode the plan called out as "load-bearing").

## What's load-bearing for next session

Session 2 plan: [`qwen_batched_prefill_session2_gpu_plan.md`](qwen_batched_prefill_session2_gpu_plan.md).

Key inputs to that plan:
- **C tokenwise oracle, not CPU intermediate.** Diff GPU batched output against C tokenwise loop. Per-layer hooks via existing `mf_layer_forward_dump` + state save/load for cross-backend KV priming.
- **MoE permute-and-fuse is GPU-side new code without a CPU diff target.** Mitigation: smaller per-component fixtures (N=2 with known routing); compare against analytical expected output; ULP-bounded against a simple CPU loop *of the GPU kernel's behavior* (not a CPU port of the layer). This is weaker than the original plan's per-primitive CPU diff but achievable.
- **Phase 0 cleanup unlocks any new test binary.** `tests/common/diff_helpers.rs` is the shared place for new diff infra.

## Files modified

- `crates/moeflux/tests/common/mod.rs` — `pub mod diff_helpers;` line
- `crates/moeflux/tests/diff_oracle.rs` — replace inline helpers with imports
- `crates/moeflux/tests/common/diff_helpers.rs` — new file (~140 LOC)

## Verification

```bash
cd ~/Projects/moeflux

# Phase 0 refactor regression check:
cargo test -p moeflux --features model-qwen3-6-35b-a3b --release \
  --test diff_oracle layer_forward_dump_close_c_vs_rust \
  -- --ignored --nocapture --test-threads=1

# Scope-creep guard — only tests/ touched:
git -C ~/Projects/moeflux status --short
# Expected: M tests/common/mod.rs, M tests/diff_oracle.rs, ?? tests/common/diff_helpers.rs.
# No src/ entries.

cargo check -p moeflux --features model-qwen3-6-35b-a3b --tests
# Expected: clean except pre-existing warnings in state_snapshot.rs / others
# unrelated to this session.
```

## Calibration for Mike's next-session frame

- Pivot-on-discovery was the right discipline call but ate the implementation budget. Phase 0 is a real delivery (sharable diff helpers) but smaller than originally promised.
- The discovery itself is the headline. The plan-of-record had a wrong premise for Qwen3.6-A3B; we now know why and what to do instead.
- Session 2's GPU batched path has weaker per-kernel oracles than originally planned (no CPU intermediate). Expect more time spent on small-fixture validation per kernel and less on end-to-end "GPU matches C" expectations. Counter-balancing: the GPU work itself becomes the deliverable, not the CPU scaffolding.
