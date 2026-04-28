---
name: Plan — slice 5d-7 GPU full-attention fast path
description: Pre-implementation plan for the GPU `gpu_attn_fuse`-equivalent slice. Bridges 5d-6 (parallel + speculative pread) and the next session's plan-mode implementation. Captures C-side scout (with line refs from the 2026-04-28 morning Explore), riir-side current state, proposed design, diff strategy, and open decisions.
type: project
---

# Slice 5d-7: GPU full-attention fast path — implementation plan

**Status**: planning, not started.
**Authored**: 2026-04-28, end of 5d-6 session (cache-warm pre-plan to jump-start the next session — same pattern as `plan_5d6_async_pread.md` did for this one).
**Branch when implementing**: `riir` in `~/Projects/moeflux`.
**Diff baseline**: 6.75 cold / 7.53 warm tok/s (after 5d-6).
**Target**: close the remaining warm gap to C (8.70 warm). Estimate range: 7.53 → ~8.0–8.5 warm if the win materializes; could be smaller if my framing of the win below is wrong (it's the first time this analysis has been written down).

---

## 1. Why this slice

After 5d-6, blallama A3B essay runs at 6.75 cold / 7.53 warm vs the C path's 10.13 / 8.70. Warm gap shrunk from 1.62× (post-5d-5) to 1.16× — most of the closure came from 5d-6. The remaining warm gap, **per the strategy doc's "Suggested next-session order"**, is dominantly the GPU full-attn fast path: 10 full-attn layers / token (out of 40 on A3B) currently run their SDPA + sigmoid gate on the host CPU instead of the GPU.

**Reframing the perf hypothesis.** The strategy doc previously claimed GPU full-attn would "eliminate per-layer host transfers" of `q_proj_out` / `k_out` / `v_out`. The 2026-04-28 scout (this file's §3) shows that's **not what the C path does** — C still runs per-head Q/K rms_norm + RoPE on CPU and still memcpy's the projections back to GPU mirror buffers before SDPA. The actual win in C's `gpu_attn_fuse` is structural:

- GPU SDPA + sigmoid gate is encoded into the *same* CMD2 cmdbuf as post-attention work (o_proj, residual, post-attn rms_norm). CMD2 already commits-and-waits; SDPA goes along for the ride, costing zero additional commit-waits.
- CPU is freed during the SDPA window (which is GPU-bound on the C side).

So the riir win is "free up CPU time during CMD2's GPU work" + "match C's per-token CPU/GPU duty cycle better". Magnitude: TBD; the strategy doc called it "second-biggest remaining lever" but didn't pin a number.

**Bias check before committing**: bench between subsequent slices to confirm the win materialized; if 5d-7 lands flat, the next slice candidate is "chained CMD3 → next layer normed" (the original 4f-perf, ~5%).

---

## 2. Goal and scope

### In scope

- Port the four C-side GPU kernels: `attn_scores_batched`, `attn_softmax_batched`, `attn_values_batched`, `sigmoid_gate`. Each gets a slice-9-shape per-kernel diff test (bit-exact or ULP-bounded) before production wire-up.
- Add **per-layer GPU KV mirror buffers** (`buf_kv_k`, `buf_kv_v`) on `RsCtx`. ~16.78 MB each per full-attn layer × 2 (K + V) × ~10 layers on A3B ≈ 336 MB extra; lazy-init via `ensure_linear_resources`. (A17B: 5 full-attn × 2 × 16.78 ≈ 168 MB.)
- Wire the host KV append to also memcpy into the GPU mirrors (mirror C's pattern at `infer.m:4796-4802`).
- Wire the GPU SDPA dispatch into `full_attn_forward.rs`, gated by `kv_len >= 32 && kv_len < GPU_KV_SEQ`. Below 32 stays on CPU SDPA (existing path).
- Encode the four kernels into `full_attn_layer_forward`'s CMD1 or CMD2 (TBD in plan-mode — see §8).

### Out of scope

- Porting per-head Q/K rms_norm or RoPE to GPU. C keeps them on CPU; we keep them on CPU. A future slice could fuse them into a GPU kernel for further wins.
- Changing the host KV-cache canonicality contract. Host KV stays canonical for `state_save`/`state_load` (already wire-format-stable per slice 4g); GPU mirror is one-way-synced from host on append, and on snapshot restore.
- The `kv_len < 32` slow path stays CPU. C does the same.
- LZ4 / 2-bit / expert-cache work (slice 9f, deferred).

### Explicit non-goal: do not break wire format

`state_save` / `state_load` output bytes must remain bit-identical to the 4g-locked format. The GPU mirror is derived state, not part of the snapshot.

---

## 3. C-side reference (from the morning scout, 2026-04-28)

### `gpu_attn_fuse` host-side dispatch (`infer.m:5051-5163`)

Gate predicate: `kv_len >= 32 && kv_len < GPU_KV_SEQ` (where `GPU_KV_SEQ = 8192`, `model_variant.h:154`). Computed at `infer.m:5054`; entry block at `infer.m:5091-5163`.

The dispatch encodes **four kernels** into the CMD2 cmdbuf (`cmd_fused`, `infer.m:5088`), each in its own encoder block (Enc A1–A4):

| # | Kernel | Encoder line | Pipeline state |
|---|---|---|---|
| A1 | `attn_scores_batched` | 5102 | `attn_scores_pipe` (line 5105) |
| A2 | `attn_softmax_batched` | 5121 | `attn_softmax_pipe` (line 5124) |
| A3 | `attn_values_batched` | 5132 | `attn_values_pipe` (line 5135) |
| A4 | `sigmoid_gate` | 5151 | `sigmoid_gate_pipe` (line 5154) |

After A4, CMD2 continues with the existing o_proj + residual + post-attn rms_norm encoders (`infer.m:5165+`). Single commit-wait for the whole CMD2.

Buffer bindings:

| Buffer | Source | Shape (per layer) | Allocation |
|---|---|---|---|
| `buf_attn_q` | host memcpy from `q_norm + RoPE` | `NUM_ATTN_HEADS * HEAD_DIM` floats | `infer.m:1261` |
| `buf_kv_k[fa_idx]` | host memcpy from `k_proj + per_head_norm + RoPE` | `GPU_KV_SEQ * kv_dim` floats (16.78 MB) | `infer.m:1251-1252` |
| `buf_kv_v[fa_idx]` | host memcpy from `v_proj` | same, 16.78 MB | `infer.m:1253-1254` |
| `buf_attn_scores` | scratch | `NUM_ATTN_HEADS * GPU_KV_SEQ` floats | `infer.m:1263` |
| `buf_attn_out` | A3 writes here | `NUM_ATTN_HEADS * HEAD_DIM` floats | `infer.m:1265` |
| `buf_attn_gate` | host memcpy from `q_gate` | same | `infer.m:1267` |

`kv_dim = NUM_KV_HEADS * HEAD_DIM` (A3B: 2×256 = 512). `buf_kv_k` and `buf_kv_v` are persistent per full-attn layer, allocated once at init.

### The four kernels (`shaders.metal`)

| Kernel | Lines | Threadgroups | Threads | Reductions | Atomics |
|---|---|---|---|---|---|
| `attn_scores_batched` | 924-967 | `(seq_len * NUM_ATTN_HEADS,)` | 256 | `simd_sum` over head_dim | none |
| `attn_softmax_batched` | 974-1032 | `(NUM_ATTN_HEADS,)` | 256 | `simd_max`, `simd_sum` | none |
| `attn_values_batched` | 1043-1065 | `((HEAD_DIM*NUM_ATTN_HEADS+255)/256,)` | 256 | none (per-thread loop) | none |
| `sigmoid_gate` | (find on scout) | per-output | 256 | none | none |

**No atomics anywhere** → expect bit-exact per-PSO under diff (slice 9 finding holds). `simd_sum` / `simd_max` are deterministic per pipeline-state object on Apple Silicon.

GQA logic: `attn_scores_batched` and `attn_values_batched` index keys/values via `kv_h = h / heads_per_kv` (`shaders.metal:943, 1057`).

### KV append (`infer.m:4796-4802`)

After CPU per-head norm + RoPE (`infer.m:4766-4789`) on `k_out` and `v_out`:

```c
int cache_pos = kv->len;
memcpy(kv->k_cache + cache_pos * kv_dim, k_out, kv_dim * sizeof(float));
memcpy(kv->v_cache + cache_pos * kv_dim, v_out, kv_dim * sizeof(float));

if (g_metal && g_metal->attn_scores_pipe && fa_idx >= 0) {
    memcpy((float *)[g_metal->buf_kv_k[fa_idx] contents] + cache_pos * kv_dim,
           k_out, kv_dim * sizeof(float));
    memcpy((float *)[g_metal->buf_kv_v[fa_idx] contents] + cache_pos * kv_dim,
           v_out, kv_dim * sizeof(float));
}
```

**No Metal blit encoder, no separate kernel** — just memcpy via the shared-storage `[contents]` pointer. Synchronous, non-pipelined.

### State save/load (`infer.m:6404, 6570-6577`)

Host KV is canonical. `state_save` reads `kv_caches[i]->k_cache` / `v_cache` (`infer.m:6404-6405`). `state_load` writes host KV first, then re-syncs GPU mirror via memcpy (`infer.m:6570, 6576-6577`).

### Slow path (`infer.m:4823-4845`)

When the gate fails (kv_len < 32, GPU unavailable, or kv_len >= GPU_KV_SEQ), C runs CPU SDPA per-head — **identical numerics** to the riir port's current `sdpa_cpu` path. No port-of-the-slow-path needed.

---

## 4. Riir-side current state (inventory, 2026-04-28)

### `full_attn_forward.rs` (the file most affected)

CPU SDPA happens in the body of `full_attn_layer_forward` — between the CMD1 projections (q_proj, k_proj, v_proj) and the hand-off to `post_attention_tail`. Specifically:

- Per-head Q/K rms_norm: CPU (uses slice 8a's `rms_norm_per_head_cpu`).
- RoPE: CPU (uses RoPE slice's `apply_rotary_emb`).
- KV append: writes into `LayerState::FullAttn::KvCache` (host-side `Vec<f32>` per layer).
- SDPA: CPU `sdpa_cpu` (slice 7d3963a).
- Sigmoid gate: CPU loop over `q_dim`.
- Stage `attn_out` into `buffers.batch_out[6]` for `post_attention_tail` to consume.

After 5d-7, the section between "CPU per-head norm + RoPE" and "stage attn_out into batch_out[6]" gets replaced with:
1. memcpy q (post-norm + RoPE) to `buf_attn_q`.
2. memcpy q_gate to `buf_attn_gate`.
3. memcpy k_out (post-norm + RoPE) to host KV + GPU KV mirror at `cache_pos`.
4. memcpy v_out to host KV + GPU KV mirror at `cache_pos`.
5. Encode 4 attn kernels into the **next** cmdbuf (CMD2's). Output lands in `buf_attn_out`.
6. Readback `buf_attn_out` to host scratch + memcpy into `buffers.batch_out[6]` (or — better — pass the GPU buffer directly to `post_attention_tail`'s o_proj input via the existing `&BufferRef` plumbing from slices 5d-3 / 5d-4).

Item 6 is the structural alignment win — if we pass `buf_attn_out` directly as a GPU buffer to o_proj, we eliminate the readback. Worth pursuing in this slice.

### `state.rs::KvCache`

Host-side `k_cache: Vec<f32>` + `v_cache: Vec<f32>`, sized `MAX_SEQ_LEN * kv_dim`. Stays canonical. Append paths add a parallel write to the GPU mirror.

### `mod.rs::RsCtx`

After 5d-6, has `metal`, `moe_buffers`, `prefetch`, `io_pool`, `linear_buffers`, `lm_head_gpu`, etc. Add a new field `attn_buffers: Option<AttnBuffers>` (or fold into `linear_buffers`; see §8 — naming TBD). `ensure_linear_resources` is the natural lazy-init point (analogous to slice 4c's `LinearAttnBuffers`).

### `state_snapshot.rs`

`state_save` reads host KV (`LayerState::FullAttn::KvCache.k_cache` / `v_cache`). No change needed.
`state_load` writes host KV. **Adds a step to also memcpy host KV → GPU mirror after the host write** (mirrors C's `infer.m:6576-6577`). Drain semantics same as today (drain deferred + prefetch first).

### Diff oracle hooks today

No GPU-attn kernels are under diff yet. Slice 5d-7 adds 4 new per-kernel `mf_<name>` C hooks + `RsCtx::<name>` methods + per-kernel diff tests in `tests/diff_oracle.rs`. Same pattern as slice 9a / 9b / 9e.

---

## 5. Proposed design — phased

The full slice is large. Split into two commits, mirroring the 5d-6a / 5d-6b shape:

### 5d-7a: per-kernel diff coverage for the 4 attn kernels
**Hypothesis**: the 4 kernels are bit-exact per-PSO (slice 9 finding holds for SIMD-only / threadgroup-only kernels with no atomics). Land them under diff one at a time; production path stays on CPU SDPA.

**Changes**:
- Add a `gpu_attn` module (or extend `gpu_norm`): RAII pipeline-state cache for the 4 kernels, encoder helpers `encode_attn_scores_into`, `encode_attn_softmax_into`, `encode_attn_values_into`, `encode_sigmoid_gate_into`.
- Add `mf_attn_scores_batched` / `mf_attn_softmax_batched` / `mf_attn_values_batched` / `mf_sigmoid_gate` C hooks in `moeflux.h` exposing the static C primitives (or wrapping the existing `gpu_attn_fuse` decomposed). May need to refactor C side to expose individual kernels separately from the fused dispatch — investigate at plan-mode time.
- Add `RsCtx::attn_scores_batched`, etc., that allocate per-call scratch buffers and run the kernel with synthetic + real inputs.
- Four new diff tests in `tests/diff_oracle.rs`: `attn_scores_close_c_vs_rust`, `attn_softmax_close_c_vs_rust`, `attn_values_close_c_vs_rust`, `sigmoid_gate_close_c_vs_rust`. Tight floors: bit-exact for sigmoid_gate; cosine ≥ 0.9999 + max_abs_diff ≤ 1e-3 × max_abs_out for the SIMD-reduction kernels (hopefully bit-exact, but reserve cosine floor as defensible placeholder per slice 9 doc).

**Expected perf**: zero. This is correctness-only.

**Risks**: low. No production path touched. If a kernel turns out NOT to be bit-exact (e.g. unexpected compiler-driven reordering), the cosine floor catches it as a smaller drift than wrong answers.

### 5d-7b: production wire-up (KV mirrors + GPU SDPA)
**Hypothesis**: porting the SDPA work into CMD2 frees CPU time during CMD2's GPU compute, closing 5-15% of the remaining warm gap to C. Magnitude uncertain; bench will tell.

**Changes**:
1. **`AttnBuffers` struct** (or fold into `LayerForwardBuffers`): per-full-attn-layer KV mirrors (`buf_kv_k[fa_idx]` + `buf_kv_v[fa_idx]`), plus shared `buf_attn_q`, `buf_attn_scores`, `buf_attn_out`, `buf_attn_gate`. Allocated lazily via `ensure_linear_resources`. Sizing per scout §3.

2. **KV append**: in `full_attn_forward.rs`, after host per-head norm + RoPE on `k_out` / `v_out`:
   ```rust
   // Host KV (canonical, for state_save).
   kv_state.append(&k_out, &v_out);
   // GPU mirror (for fast-path SDPA).
   if attn_buffers.gpu_attn_ready() {
       attn_buffers.append_kv(fa_idx, cache_pos, &k_out, &v_out);
   }
   ```
   `append_kv` is a thin shared-storage memcpy via `&mut [f32]` borrowed from the `MtlBuffer<f32>` (analogous to `data_synced_slot_mut` from slice 5d-5).

3. **Gate predicate**: `let gpu_attn_ready = kv_state.len + 1 >= 32 && kv_state.len + 1 < GPU_KV_SEQ;`. Match C exactly.

4. **GPU SDPA encode**: when `gpu_attn_ready`, encode the 4 kernels into the CMD2 cmdbuf (the one `post_attention_tail` opens). The encode happens **inside** `full_attn_layer_forward`, right before the `post_attention_tail` hand-off. `attn_out` lives on `buf_attn_out`; pass that as a `&BufferRef` to `post_attention_tail`'s o_proj input.

5. **Slow-path fallback** (`kv_len < 32`): keep the existing CPU SDPA path. No code deletion — the predicate just skips the GPU encode.

6. **`state_load` GPU sync**: after host KV write, copy host KV → GPU mirror for each full-attn layer's current `kv->len` window. Mirror C `infer.m:6576-6577`.

7. **`state_save` unchanged**: reads host KV only.

8. **`memory_clear`**: zero the GPU mirrors alongside host KV (host clears are existing). Same shape as slice 4f-3's `bufs.reset_recurrence()` in `memory_clear`.

**Expected perf**: 7.53 → 8.0–8.5 warm tok/s (5–13%). Could be smaller if GPU is already saturated on CMD2; could be larger if bench reveals other surprises.

**Risks**:
- **Numerical drift**: each new GPU kernel introduces drift vs CPU SDPA (per slice 8b's RoPE finding, `f32` scalar reductions on CPU vs Metal SIMD reductions can ULP-drift). End-to-end `eval_token_matches_c_single_step` may need a slightly looser cosine floor (currently 0.9999); plan-mode should pre-baseline by running without the slow-path (force GPU at all kv_len) and measuring drift. Probably stays at 0.9999.
- **Race**: GPU SDPA reads `buf_kv_k[fa_idx]` / `buf_kv_v[fa_idx]`. The host appends to the same buffers via memcpy. Sequencing: append happens BEFORE encode within the same `full_attn_layer_forward` call; encode commits at the end of CMD2. The append's bytes are visible to GPU at commit time (Apple UMA shared-storage semantics). No race within a layer. Across layers: layer N's KV append for THIS token doesn't touch layer N+1's KV mirror; safe.
- **Allocation footprint**: 168–336 MB extra for KV mirrors. Memory-budget acceptable on M2 Max 32 GB.

---

## 6. Diff test strategy

### 5d-7a (per-kernel tests)
Mirror slice 9's pattern:
- `attn_scores_close_c_vs_rust`: synthetic Q (random f32), synthetic K (random f32), various `kv_len ∈ {32, 64, 128, 512}`. Compare `buf_attn_scores` per-element.
- `attn_softmax_close_c_vs_rust`: synthetic scores, various kv_len. Compare per-element.
- `attn_values_close_c_vs_rust`: synthetic softmax probs + V, various kv_len.
- `sigmoid_gate_close_c_vs_rust`: synthetic input. Bit-exact (per-thread, no reductions).

C hooks: each `mf_<name>` takes input buffers, runs ONE kernel, returns output. May require refactoring C `gpu_attn_fuse` to expose individual kernels — investigate.

### 5d-7b (production end-to-end)
Existing tests should stay green:
- `eval_token_matches_c_single_step` — both backends now run GPU SDPA (when kv_len >= 32). Numerics should align bit-exactly per-PSO; cosine floor 0.9999 already trivially satisfied.
- `eval_prompt_matches_c_multi_token` — exercises GPU path naturally as kv_len grows past 32 mid-prompt.
- `state_round_trip_rust` — verifies state_load's GPU-sync path doesn't break the round-trip.
- `layer_forward_dump_close_c_vs_rust_full_attn` — single full-attn layer at pos=0, kv_len=1. **Below the gate**, so stays on CPU SDPA. Need an additional dump test at kv_len ≥ 32 to exercise the GPU path through this hook.

New end-to-end test:
- `layer_forward_dump_close_c_vs_rust_full_attn_gpu_path`: dump-hook at a position with kv_len ≥ 32 (e.g., dump layer 3 at pos=64). Exercises GPU SDPA path; existing tight-floors apply.

---

## 7. Sequencing / commit plan

**Commit A — slice 5d-7a (per-kernel diff coverage)**:
- `gpu_attn` module + 4 kernel encoders.
- 4 C hooks `mf_attn_*`.
- 4 `RsCtx::attn_*` methods.
- 4 diff tests.
- Run diff suite, commit.

**Commit B — slice 5d-7b (production wire-up)**:
- `AttnBuffers` struct + lazy-init.
- KV mirror append in `full_attn_layer_forward`.
- Gate predicate + GPU SDPA encode + CMD2 wire-up.
- `state_load` GPU sync.
- `memory_clear` GPU mirror reset.
- New end-to-end diff test for kv_len ≥ 32 dump hook.
- Run diff suite, run blallama A3B essay perf bench, commit.

If 5d-7a's per-kernel tests pass at bit-exact, expect 5d-7b to land cleanly. If any kernel needs cosine floor (atomics/reordering surprises), 5d-7b's end-to-end drift is bounded by per-kernel drift — diff oracle catches regressions early.

---

## 8. Open decisions for next-session plan-mode

These are the design choices I'd surface for sign-off when entering plan mode:

1. **Where does `AttnBuffers` live?**
   - (a) New struct alongside `MoeBuffers` and `LinearAttnBuffers` on `RsCtx`. Cleanest separation; matches the per-domain pattern.
   - (b) Folded into `LayerForwardBuffers` (which is per-call). Wrong — KV mirrors are persistent across calls.
   - **Lean: (a).** Persistent, lazy-init alongside other GPU resources.

2. **Per-layer or shared scratch for `buf_attn_scores`?**
   C uses one shared `buf_attn_scores` buffer across all layers (each layer overwrites it; SDPA is layer-sequential). Riir can do the same. Saves ~64 MB on A3B vs per-layer.
   - **Lean: shared.** Match C.

3. **C-side hook refactor: do we need to expose individual kernels?**
   The C path's `gpu_attn_fuse` is monolithic — runs all 4 kernels back-to-back. For per-kernel diff tests we need to run ONE at a time. Two options:
   - (a) Refactor C's `gpu_attn_fuse` into 4 functions; expose each via `mf_<name>`. Touches C side; FIXME-territory.
   - (b) Write fresh C primitives in `moeflux-sys/oracle/` that don't go through `gpu_attn_fuse` — they're test-only oracle entry points. Mirrors the slice 9 pattern (oracle vs production split).
   - **Lean: (b).** Don't touch the C production path; keep it bit-exact for the existing diff tests. Test-only oracle hooks.

4. **Where in `full_attn_layer_forward` does GPU SDPA encode?**
   - (a) New cmdbuf, commit-wait, then `post_attention_tail`. Adds a commit-wait. Bad.
   - (b) Encode into CMD2 — but CMD2 is opened by `post_attention_tail`. Need to either (i) plumb the cmdbuf out of `post_attention_tail` to `full_attn_forward` for pre-population, or (ii) move the SDPA encoding into the start of `post_attention_tail` (with a flag). C does (b/ii) effectively (`gpu_attn_fuse` is invoked from inside the post-attn cmdbuf flow at `infer.m:5091-5163`).
   - **Lean: (b/ii).** Pass an `Option<GpuAttnInputs>` to `post_attention_tail`, which encodes the 4 attn kernels at the top of CMD2 if Some.

5. **Pass `buf_attn_out` directly as `&BufferRef` to `post_attention_tail`'s o_proj input?**
   - Yes. This is the structural alignment with slice 5d-4's `_buf` plumbing — eliminate the host stage of `attn_out` into `batch_out[6]`. Saves another HIDDEN_DIM readback per full-attn layer.
   - **Lean: yes.** No extra cost; cleanest.

6. **Do we replicate C's `kv_len < GPU_KV_SEQ` upper-bound check?**
   - GPU_KV_SEQ = 8192. If kv_len exceeds 8192, GPU mirror runs out of slots. C falls back to CPU SDPA. Riir should match.
   - **Lean: match C.** Same gate predicate.

7. **First-token / dump-hook handling.**
   - Dump-hook tests run at kv_len = 1 (first position). Below the gate. Stays on CPU SDPA. No special-case needed.

---

## 9. References

- C scout report: this file's §3, generated by the morning Explore agent reading `metal_infer/infer.m` + `shaders.metal`. Lines refs preserved.
- Riir code as of commit `f9d8ff3` (slice 5d-6b).
- 5d-6 perf log: `riir_moeflux_strategy.md` § Phase 5 progress (cold 6.75, warm 7.53).
- Strategy doc's "Suggested next-session order" entry on GPU full-attn fast path (now superseded by this plan).
- `metal_infer/infer.m` line refs:
  - `gpu_attn_fuse` gate + dispatch: 5051-5163
  - Attn buffer allocation: 1251-1272
  - KV append (host + GPU mirror): 4796-4802
  - CPU SDPA slow path: 4823-4845
  - State save (reads host): 6404-6405
  - State load (writes host + GPU sync): 6570-6577
- `metal_infer/shaders.metal`:
  - `attn_scores_batched`: 924-967
  - `attn_softmax_batched`: 974-1032
  - `attn_values_batched`: 1043-1065
  - `sigmoid_gate`: search by name (scout didn't pin the line range)

---

## 10. What success looks like

After 5d-7 lands:
- 5d-7a: 4 new per-kernel diff tests green at bit-exact (or cosine ≥ 0.9999 if ULP-bounded). Existing 35 tests stay green.
- 5d-7b: full diff suite (39+ tests) green at the existing tight floors. blallama A3B essay perf in the 8.0–8.5 warm tok/s range — target: parity with C's 8.70.
- Activity Monitor shows GPU utilization on A3B closer to C's ~40%. (5d-6 didn't capture this; would be informative to compare 5d-6 baseline vs 5d-7 result.)
- Cold tok/s improves modestly (less pipeline-warmup CPU work to do per first-token-batch).

If the warm number lands flat (no measurable improvement), the framing in §1 was wrong and the actual remaining gap is elsewhere (chained CMD3 → next layer normed; or something further downstream like sampling-side cost). 5d-7's correctness work is still load-bearing for a future GPU-side norm/RoPE fusion (which IS the strategy doc's "eliminate host transfers" claim — but that's a slice further out).
