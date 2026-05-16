# Session 8 Part A — graph compiler lifetime coloring landed

**Date:** 2026-05-14
**Branch:** moeflux `main`
**Commits:**
- `7b7dbff` — S7-4.5: split `graph.rs` into directory module
- `63e12cd` — S7-5a: `graph/lifetime.rs` (analyzer + greedy linear-scan colorer)
- `7fd0284` — S7-5b: pool aliasing via `BufferPool::commit_plan`
- `687cd2b` — S7-5c: `graph_metal_matches_cpu_colored` diff oracle

**Entry:** [`qwen_graph_mode_session8_handoff.md`](qwen_graph_mode_session8_handoff.md)
**Executable plan:** `/Users/mdegans/.claude/plans/synchronous-orbiting-gizmo.md`

## Headline result

**Load-bearing acceptance gate passed.** The new
`graph_metal_matches_cpu_colored` test runs a 10-op residual chain
through both backends after invoking `commit_plan`:

```
[s7-5 colored] N=4 dim=32 chain=10 bufid_count=12
               cpu_phys=4 gpu_phys=4
               cos=1.000000000 max_abs=0.000e0
```

12 logical BufIds (2 inputs + 10 transients) compress to 4 physical
buffers (2 inputs + 2 colors). CPU and Metal pools agree exactly on
the physical layout (the coloring is deterministic and both
backends see identical Graphs). The output is bit-exact across
backends.

## What's in scope and shipped

### S7-4.5 — graph.rs directory split (precursor)

`graph.rs` (2571 LOC) → `graph/{mod,cpu,metal}.rs`. Mechanical move
+ `pub use` re-exports preserve the public API surface at
`crate::riir::graph::...`. 13 in-module tests + 3 diff oracle tests
pass unchanged. The split happened as a separate commit before
S7-5 so the algorithm work landed in a clean module rather than at
the bottom of a 2800-LOC monolith.

### S7-5a — `graph/lifetime.rs`

**Types:**
- `Interval { first_write_op, last_read_op }` — live range in op-idx
  coordinates.
- `Lifetimes { intervals: HashMap<BufId, Interval> }` — only
  colorable BufIds present; pure-input BufIds (read but never
  written) are absent.
- `ColorId = u32`, `ColoringMap { bufid_to_color, color_count }`.

**Functions:**
- `analyze_lifetimes(&Graph) -> Lifetimes` — single linear walk of
  `Op::reads()/writes()`. RMW handled correctly (single-point
  interval at the RMW op).
- `greedy_color(&Lifetimes) -> ColoringMap` — register-allocator
  style. Sort intervals by start (BufId tiebreak for determinism);
  sweep; reuse lowest free color, allocate new only when no gap
  exists below `next_color`.

10 unit tests cover analyzer + colorer behavior including empty
graph, pure-input absence, residual chain shape, RMW single-point,
disjoint reuse, overlap forcing 2 colors, 10-op ping-pong ≤ 2
colors, 5-residual chain = 2 colors, determinism across insertion
order.

**Gap-detection bug caught and fixed in S7-5a**: when `active` was
empty after retain(), my "find lowest free color" loop stayed at
`expected=0` and chosen=None, then the allocator returned
`next_color` (which had been incremented from previous allocs)
instead of reusing freed color 0. The disjoint-intervals test
reported `color_count=2` instead of 1, surfacing the bug in 60s.
Fix: in the chosen=None branch, check `expected < next_color` and
reuse `expected` (a freed color) rather than allocating new.

### S7-5b — pool aliasing via `commit_plan`

**Trait change** in `graph/mod.rs`: new `BufferPool::commit_plan`
method with default no-op impl (backwards-compatible).

**Concrete impls** in `graph/cpu.rs` and `graph/metal.rs` (parallel
structure, near-symmetric implementations):

Each pool gains two new fields:
- `byte_sizes: Vec<usize>` — per-BufId allocation size.
- `bufid_to_physical: Vec<u32>` — per-BufId index into `buffers`.
  Identity by default; rewritten by `commit_plan`.

`handle`/`upload`/`download` route through `bufid_to_physical`.
Size checks use `byte_sizes[id]` (not `buffers[physical].len()`)
because aliased physical buffers are sized to `max(group)` and may
exceed any individual BufId's expected size.

`reset_transient` walks `persistent` flags in BufId space (existing
producer convention) and truncates `buffers` by `max(bufid_to_physical) + 1`
to drop unused physical buffers. Preserves the invariant that
persistent BufIds keep their original physical position after
`commit_plan`.

`commit_plan` algorithm:
1. Run `analyze_lifetimes` + `greedy_color`.
2. Filter coloring to exclude persistent BufIds (they must keep
   dedicated buffers for content preservation across
   `reset_transient`).
3. Phase 1: place non-aliasable BufIds (persistents + non-colorable
   transients) in the new physical layout via `mem::replace` swap
   — preserves their content.
4. Phase 2: allocate one new physical buffer per color, sized to
   `max(byte_sizes)` among the color's BufIds. Map all aliasable
   BufIds in that color to it.

### S7-5c — `graph_metal_matches_cpu_colored`

Test in `crates/moeflux/tests/graph_diff_oracle.rs`. A 10-op
residual chain `tmp_i = tmp_{i-1} + b` forces 10 colorable BufIds
overlapping at one op each → 2 colors. Total BufIds = 12; physical
after `commit_plan` = 4 (2 inputs + 2 colors).

Assertions:
1. Cosine ≥ 0.9999 between CPU and Metal outputs (correctness).
2. `pool.physical_buffer_count() < 12` on both pools (aliasing
   occurred).
3. `cpu_phys == gpu_phys` (deterministic coloring).

All three assertions pass; bit-exact match (`max_abs=0`) because
the chain is pure float additions.

## What's NOT in scope this session

- Producer rewrites (S7-6 linear-attn, S7-7 full-attn).
- Wiring the 7 deferred `todo!()` arms in MetalBackend.
- Orchestrator two-phase (S7-8).
- `Graph::dump` polish (S7-9).

The S7-5 work is purely additive on the graph module and adds zero
load to the active forward path.

## Stats

| File | LOC delta |
|---|---:|
| `graph.rs` → `graph/mod.rs` + `graph/cpu.rs` + `graph/metal.rs` | +11 net |
| `graph/lifetime.rs` (new) | +397 |
| `graph/cpu.rs` + `graph/metal.rs` + `graph/mod.rs` (pool integration) | +230 |
| `graph_diff_oracle.rs` (colored test) | +157 |
| **Session total** | **~795 net LOC** |

## Verification commands (paste-ready)

```bash
cd ~/Projects/moeflux

# Lib tests (23 = 8 generic + 5 cpu_pool + 10 lifetime):
cargo test -p moeflux --features model-qwen3-6-35b-a3b --lib graph::

# Diff oracle (4 = residual_add + swiglu + router/normalize + colored):
cargo test --release -p moeflux --features model-qwen3-6-35b-a3b \
  --test graph_diff_oracle -- --ignored --nocapture --test-threads=1
```

## Forward look — session 9 priorities

Per the locked plan, in order:

1. **S7-6 — Linear-attn producer rewrite + wire 5 deferred linear-
   attn `todo!()` arms** (`RmsNormQkNTokens`, `Conv1dStepNTokens`,
   `ComputeDecayBetaNTokens`, `GatedDeltaNetStepNTokens`,
   `GatedRmsNormNTokens` + the `MoeBatchedPermuteFuse` composite).
   Canary 9/9 at the checkpoint.

2. **S7-7 — Full-attn producer rewrite** + wire `SdpaCausalTiled` +
   `LmHead` with proper workspace BufIds.

3. **S7-8 — Per-layer two-phase orchestrator**, canary 9/9, warm
   directional bench.

4. **S7-9 — Polish + reboot bench + session-10 plan**.

S7-6 should call `commit_plan` once per Graph after building the
op list. The colored test is a working pattern.

## Known dirty spots (carried forward from S7-2)

- `Op::GatedDeltaNetStepNTokens` CpuBackend arm still calls
  `gated_delta_recurrence` with dummy `a_log`/`dt_bias`. The
  recurrence-only split (split `gated_delta_recurrence` into a
  function taking pre-computed g_decay/beta_gate) is still
  pending. Address at S7-6 producer rewrite time.

## Calibration note

Session started at ~333k/1m used (from session 7). Budget
remained ample. Mike's pre-session note that ample context
remained was correct; no need to wrap on context anxiety. The
plan was sized at "1 step + 4 commits"; the actual delivery was
exactly that.

The pattern from session 7 — write the diff test BEFORE trusting
the algorithm — caught the gap-detection bug in greedy_color
within 60 seconds. Same pattern caught SwiGLU in S7-3. Two-for-
two on diff-tests-first paying off.
