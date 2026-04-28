---
name: Plan — slice 5d-8 chained CMD3 → next-layer normed
description: Pre-implementation plan for the chained CMD3 + rms_norm slice (the original 4f-perf). Bridges 5d-7 (GPU full-attn) and the next session's plan-mode implementation. Captures C-side scout (with line refs from the 2026-04-28 session), riir-side current state, proposed design, and open decisions.
type: project
---

# Slice 5d-8: chained CMD3 → next-layer normed — implementation plan

**Status**: planning, not started.
**Authored**: 2026-04-28, end of 5d-7 session (cache-warm pre-plan to jump-start the next session — same pattern as `plan_5d6_async_pread.md` and `plan_5d7_gpu_full_attn.md` did).
**Branch when implementing**: `riir` in `~/Projects/moeflux`, base commit `5cfb521` (after 5d-7b).
**Diff baseline**: cold 7.45 / warm 7.36 tok/s (after 5d-7).
**Target**: close another ~5% of the warm gap to C (8.70 warm). Estimate range: 7.36 → ~7.7–8.0 warm.

---

## 1. Why this slice

5d-7 came in flat on warm (7.36 vs 5d-6's 7.53, within noise). The diagnostic was clean: GPU utilization climbed from C's ~40% to ~50% — the GPU SDPA fast path IS firing — but warm tok/s didn't move because the CPU's critical path was already dominated by per-head Q/K rms_norm + RoPE + memcpys, not by SDPA. We freed GPU's CPU-side helpers from one job (SDPA) but the CPU was busy elsewhere on the critical path the whole time.

The next lever, per the strategy doc and the 5d-7 bias check, is **chained CMD3 → next-layer normed** — the original 4f-perf. Today riir's `complete_deferred_experts_into` does a GPU→host→GPU dance to hand off between layers:

1. Layer N's CMD3 (K-expert dispatch + `moe_combine_residual`) commits async, writes to `bufs.moe_hidden`.
2. Layer N+1's iteration starts with `complete_deferred_experts_into(deferred, bufs, &mut host_slice)`:
   - waits for layer N's cmdbuf
   - copies `bufs.moe_hidden` to host scratch
   - copies host scratch to `linear_buffers.input` (a different GPU shared-storage buffer)
3. Layer N+1's CMD1 starts: GPU rms_norm reads `linear_buffers.input`, writes `linear_buffers.normed`.

C's path skips all of this: layer N's CMD3 writes the *already-normalized* next-layer input directly to `buf_input` via 3 extra dispatches appended to the K-expert cmdbuf:
- Enc C1: `moe_combine_residual` → `buf_moe_hidden`
- Enc C2: `rms_norm_sum_sq` reads `buf_moe_hidden` → `buf_cmd3_sum_sq`
- Enc C3: `rms_norm_apply_bf16` reads `buf_moe_hidden` + layer N+1's `input_norm_w` → `buf_input`

Layer N+1's CMD1 then *skips* its own input-norm prelude entirely. The deferred-wait at the layer boundary is just a `wait_until_completed` — no readback, no GPU→host→GPU dance.

**Win shape**: per-layer (60 layers on A3B) we eliminate (a) one HIDDEN_DIM readback, (b) one HIDDEN_DIM host→GPU copy, (c) one CMD1 rms_norm dispatch pair. Each is small but they multiply by 60 and gate the next layer's CMD1 commit. Rough estimate ~5%.

**Bias check**: bench between this slice and the next. If 5d-8 lands flat, the next candidate is GPU per-head Q/K rms_norm + RoPE fusion (now more attractive after 5d-7 confirmed the GPU has headroom).

---

## 2. Goal and scope

### In scope

- Reroute `moe_combine_residual` to write into `linear_buffers.input` (the next layer's input buffer) instead of `bufs.moe_hidden` for non-last-layer dispatches.
- Append the rms_norm chain (`rms_norm_sum_sq` → `rms_norm_apply_bf16`) into the same K-expert cmdbuf, reading layer N+1's `input_layernorm.weight` from the cache, writing the normed output to `linear_buffers.normed`.
- Modify the layer N+1 path to skip the existing CMD1 rms_norm prelude when the previous layer chained.
- Modify `complete_deferred_experts_into` to skip the readback when the chain wrote directly to GPU buffers (just `wait_until_completed`).
- Last-layer handling: layer 59 on A3B has no next layer; chain disables, existing host-readback flow applies (the orchestrator's post-loop `model.norm` rms_norm + LM head consume the host snapshot).
- Dump-hook test path: `layer_forward_dump` runs one layer; chaining doesn't apply. Forced disable.

### Out of scope

- Changing the CPU-combine path (`gpu_combine = false` in `DeferredMode::Cpu`). CPU combine doesn't go through `moe_combine_residual`; the chain has nothing to attach to.
- Porting per-head Q/K rms_norm + RoPE to GPU (the obvious *next* slice if 5d-8 still leaves headroom).
- Touching the wire format or `state_save` / `state_load`.

### Explicit non-goal: don't break existing diff coverage

The chain produces the same numerical output as the unchained path modulo per-PSO drift (slice 9e established `rms_norm_sum_sq` + `rms_norm_apply_bf16` is bit-exact per-PSO). All existing diff tests should stay green at the existing tight floors (cosine ≥ 0.9999, rel ≤ 1e-3).

---

## 3. C-side reference (from the 2026-04-28 scout)

### `gpu_combine` predicate (`infer.m:5670..5675`)

```c
int gpu_combine = (g_metal->moe_combine_residual &&
                   g_metal->rms_norm_sum &&
                   g_metal->rms_norm_apply_bf16 &&
                   g_metal->wf_buf &&
                   layer_idx < NUM_LAYERS - 1 &&
                   layer_cache[layer_idx + 1].input_norm_w != NULL);
```

Three gates: pipelines available, *not the last layer*, next layer's input-norm weight is cached. The third is critical — without it, the CPU side has no weight pointer to bind. Layer-weight-cache hits are guaranteed by `LayerWeightCache::build_all` in riir, so the analogue check simplifies to `layer_idx + 1 < num_layers`.

### Chained dispatches (`infer.m:5677..5750`)

| # | Kernel | Buffer in | Buffer out | Notes |
|---|---|---|---|---|
| C1 | `moe_combine_residual` | `buf_h_mid` + `buf_shared_out` + 16 expert outs + `buf_combine_params` | `buf_moe_hidden` | unchanged from non-chained path |
| C2 | `rms_norm_sum_sq` | `buf_moe_hidden` | `buf_cmd3_sum_sq` | single threadgroup, 256 threads |
| C3 | `rms_norm_apply_bf16` | `buf_moe_hidden` + `wf_buf[next_norm_off]` + `buf_cmd3_sum_sq` | `buf_input` | per-element, `(dim+255)/256` threadgroups |

After C3, `buf_input` is layer N+1's normalized input. Layer N+1's CMD1 starts directly with the projection matvecs, skipping its own input-norm prelude.

### Per-layer state for the chain (`infer.m:5760..5775`)

After committing the cmdbuf, C saves:
```c
g_deferred.gpu_combined = gpu_combine;  // 1 if chained, 0 if not
g_deferred.cmd_experts = cmd_experts;
// ... actual_K, shared_gate_score, hidden, layer_idx, expert_weights, valid ...
if (!gpu_combine) {
    memcpy(g_deferred.h_mid, h_mid, HIDDEN_DIM * sizeof(float));
}
```

`h_mid` is only saved when chaining is OFF — because the only consumer is the CPU-combine fallback path inside `complete_deferred_experts`, which only runs when `gpu_combined == 0`. When chaining is ON, the GPU already wrote the final hidden into `buf_input`; the deferred-complete path just waits and returns.

### Buffer roles in C

- `buf_h_mid` = residual source (post-attention, pre-experts hidden state). Filled by CMD2's `residual_add`. Only consumed by CMD3's `moe_combine_residual` (which adds it back into the combine output).
- `buf_moe_hidden` = unnormalized combine output. Intermediate scratch for the chain (read by C2 + C3). When chaining OFF, also the readback target.
- `buf_input` = layer N's input AND (after chaining) layer N+1's normalized input. C overwrites it via Enc C3.

The riir buffer mapping isn't 1:1 — see §4.

---

## 4. Riir-side current state (inventory, 2026-04-28 post-5d-7b)

### Buffer landscape on `LayerForwardBuffers`

Slice 5d-2 collapsed riir's buffer layout: `buffers.input` plays a dual role as (a) input to CMD1's GPU rms_norm prelude and (b) residual source for CMD2's `encode_residual_add`. There is no separate `h_mid`-analogue feeding the combine — the combine reads `buffers.h_mid` (CMD2's `residual_add` output, the post-attention hidden state).

Mapping vs C:
- C `buf_h_mid` ≈ riir `buffers.h_mid` — combine input (residual added back inside CMD3).
- C `buf_moe_hidden` ≈ riir `bufs.moe_hidden()` (on `MoeBuffers`) — combine output / readback target.
- C `buf_input` ≈ riir `buffers.input` — layer's input / residual source for the layer's own CMD2.

So riir's `buffers.input` plays the same role as C's `buf_input` for layer N's *own* CMD2. The chain in C writes the *next layer's* `buf_input` — which on the riir side means writing to the next layer's `buffers.input`. Since `linear_buffers` is shared across layers (single `LayerForwardBuffers` reused per token), "next layer's input" is *the same buffer* as "this layer's input" — they're the same `linear_buffers.input` slot, advanced through time.

This simplifies the design: the chain writes to `linear_buffers.input` (which is layer N+1's input on the next iteration). No per-layer buffer indexing.

### `gpu_batched_experts_encode_pre_staged` (`expert_forward.rs:~700`, slice 5d-5)

Encodes:
1. Per-expert FFN dispatches (gate / up / SwiGLU / down) writing to `bufs.out[k]` for each routed slot.
2. `moe_combine_residual` reading `bufs.h_mid_buf` + `bufs.shared_out_buf` + `bufs.out[0..16]` + `bufs.combine_params`, writing to `bufs.moe_hidden`.

After the chain lands, this function gains:
3. `rms_norm_sum_sq` reading `linear_buffers.input` (or wherever combine output lands), writing `linear_buffers.sum_sq`.
4. `rms_norm_apply_bf16` reading the same + `wf_buf[next_norm_off]` + `linear_buffers.sum_sq`, writing `linear_buffers.normed`.

Pre-existing encoder helpers in `gpu_norm.rs`: `encode_rms_norm_bf16_into` does both 3+4 in one call. **Reusable**.

### `complete_deferred_experts_into` (`deferred.rs:251`)

Today:
```rust
let Some(state) = slot.take() else { return Ok(()); };
state.cmd_buffer.wait_until_completed();
match state.mode {
    DeferredMode::Gpu => {
        hidden_out.copy_from_slice(&bufs.moe_hidden().to_vec());
    }
    DeferredMode::Cpu { ... } => { cpu_combine(...); }
}
```

When chained, the GPU has already written the normalized data to `linear_buffers.normed` and the unnormalized data to `linear_buffers.input`. No host slice needed. The function should:
- Still wait on the cmdbuf.
- Skip the `to_vec` + `copy_from_slice` readback when chained.

API shape options (open decision §8.1):
- **(a)** New variant `DeferredMode::GpuChained { /* no payload */ }`. `complete_deferred_experts_into` matches and skips readback. Layer-loop callers in `step_internal` skip the `&mut host_slice` arg (or pass an unused dummy).
- **(b)** New free function `complete_deferred_experts_chained(slot, bufs)` that just waits + clears state. Layer loop calls one or the other based on whether layer N chained.

(b) is cleaner — no enum churn, callsite explicitly says "I expected the chain". Lean: (b).

### Layer N+1's CMD1 input-norm prelude

In both `linear_attn_layer_forward` and `full_attn_layer_forward`, CMD1 starts with:
```rust
encode_rms_norm_bf16_into(
    cmdbuf, &rms_pipes,
    &buffers.input,
    wf_buf.buffer(),
    layer_cache.input_layernorm_w,
    &buffers.sum_sq,
    &buffers.normed,
    v.hidden_dim as u32,
    super::variants::RMS_NORM_EPS,
);
```

When the previous layer chained, `buffers.normed` is *already* populated and this dispatch is redundant. The fix: skip when `prev_layer_chained` is true. Plumbing options (open decision §8.2):
- **(a)** Pass a `prev_layer_chained: bool` arg through `linear_attn_layer_forward` / `full_attn_layer_forward`.
- **(b)** Add a `chain_state: ChainState { Fresh, NormedReady }` field to `LayerForwardBuffers`; layer N+1 reads + resets it.

(a) is explicit, (b) hides a coupling. Lean: (a).

### Layer-loop orchestration in `step_internal` (`mod.rs:~985..1100`)

Today (paraphrased):
```rust
for layer_idx in 0..v.num_layers {
    if layer_idx > 0 {
        complete_deferred_experts_into(deferred, moe_buffers, buf_input_slice);
    }
    if is_full {
        full_attn_layer_forward(..., gpu_combine: bool);
    } else {
        linear_attn_layer_forward(..., gpu_combine: bool);
    }
}
// post-loop: drain final layer, model.norm + LM head
```

After 5d-8: each layer's forward decides whether *its own* CMD3 will chain (based on `layer_idx + 1 < num_layers && gpu_combine`). The next-iteration drain calls `complete_deferred_experts_chained` (no readback) or the existing readback variant. The next-iteration forward gets `prev_layer_chained = true` and skips its own input-norm prelude.

### Layer-weight cache lookahead

Layer N's `post_attention_tail` needs layer N+1's `input_layernorm_w` weight offset. `LayerWeightCache::build_all` already builds the cache for all layers up front; access is `layer_caches[layer_idx + 1].input_layernorm_w`. Pass this in alongside the existing `gpu_combine` arg.

The layer cache lookahead is gated by `layer_idx + 1 < v.num_layers` — at the last layer, pass `None` for the next-layer-norm offset, no chain encoded, no `prev_layer_chained` for the next iteration (because there is none).

### `layer_forward_dump` interaction

`layer_forward_dump` runs one layer with deferred-state bracketed by `discard_deferred_experts_in` on entry and `complete_deferred_experts_into` on exit (drains into `linear_buffers.input` for the readback path). The dump hook expects the *unnormalized* combine output as its returned hidden state — same shape as `mf_layer_forward_dump`'s C-side contract.

When chaining, `linear_buffers.normed` would hold the *normalized* output (which the dump hook does NOT want). To avoid breaking the dump-hook diff:
- Force `gpu_combine = false`-style chain disable in `layer_forward_dump_inner` (always pass `next_layer_norm_w = None` regardless of whether layer + 1 < num_layers).
- The existing readback flow stays intact: `complete_deferred_experts_into(slot, bufs, host_slice)` reads from `bufs.moe_hidden` (unnormalized). ✓

Cleanest design: a `disable_chain: bool` flag on `gpu_batched_experts_begin_pre_staged`, defaulting to false in production. `layer_forward_dump_inner` sets it to true.

OR (simpler): don't pass `next_layer_norm_w` from the dump hook → chain auto-disables. Cleaner.

---

## 5. Proposed design

Single commit. The changes are tightly coupled; phasing them would either land a dead-code path (encoder gains chain support but no caller uses it) or break diff tests in between (chain wired but layer N+1 still runs its own rms_norm).

### A. New helper in `gpu_norm.rs`: `encode_chained_combine_norm_into`

Wraps the C path's Enc C2 + Enc C3 (the two-dispatch rms_norm chain that produces normalized output for the next layer). Signature:

```rust
pub fn encode_chained_combine_norm_into(
    cmdbuf: &CommandBufferRef,
    pipes: &RmsNormBf16Pipelines,
    combine_out: &BufferRef,   // bufs.moe_hidden (combine output, unnormalized)
    weight_buf: &BufferRef,    // wf_buf
    next_norm_off: u64,        // layer_caches[layer_idx + 1].input_layernorm_w
    sum_sq: &BufferRef,        // linear_buffers.sum_sq (scratch)
    normed_out: &BufferRef,    // linear_buffers.normed (target)
    dim: u32,
    eps: f32,
)
```

Reuses `RmsNormBf16Pipelines` (already cached + fetched per-layer in 5d-2). Body is structurally identical to `encode_rms_norm_bf16_into` — a thin renaming for clarity at the callsite. (Could just reuse `encode_rms_norm_bf16_into` directly without a new helper; lean toward the rename for readability. See §8.5.)

### B. Encoder changes

`gpu_batched_experts_encode_pre_staged` (and `_begin_pre_staged`) gain a new optional arg:

```rust
pub struct ChainNorm {
    pub next_norm_off: u64,    // layer N+1's input_layernorm.weight offset
}
```

Then:
```rust
fn gpu_batched_experts_encode_pre_staged(
    ...,
    chain_norm: Option<ChainNorm>,
    chain_targets: Option<ChainTargets>,  // refs to linear_buffers.{input,sum_sq,normed}
) -> Result<...>
```

Where `ChainTargets` carries `&BufferRef` for the three targets (input is rebound as combine output; sum_sq + normed are written by the chain).

Wait — combine output target. Let me think.

C path: combine writes to `buf_moe_hidden`, chain reads `buf_moe_hidden` and writes to `buf_input`.

Riir option **(a)**: combine writes to `bufs.moe_hidden()` (unchanged), chain reads `bufs.moe_hidden()` and writes to `linear_buffers.normed`. **`linear_buffers.input` stays as the residual source for layer N+1's CMD2** — the chain doesn't touch it because the residual source is layer N's `buffers.h_mid`-analogue (which the combine already adds back). Hmm but layer N+1's CMD2 residual_add reads `buffers.input` — and layer N+1's `buffers.input` should be the post-experts hidden state of layer N (the combine output, unnormalized). Where does it come from?

This is the subtle bit I flagged earlier. Three sub-options:

- **(a-i)** `complete_deferred_experts_chained` does a GPU→GPU copy of `bufs.moe_hidden()` → `linear_buffers.input`. Eliminates the host roundtrip (save) but adds a memcpy. On Apple UMA, shared-storage memcpy is fast — but it's still a memcpy.
- **(a-ii)** Combine writes directly to `linear_buffers.input` (rebind C1's output target). The chain reads `linear_buffers.input` and writes `linear_buffers.normed`. No memcpy needed. **Cleaner**, but requires `gpu_batched_experts_encode_pre_staged` to take `linear_buffers.input` as the combine output target — currently it's hardcoded to `bufs.moe_hidden()`.
- **(a-iii)** Combine writes to `bufs.moe_hidden()` (unchanged), chain writes `linear_buffers.normed`. Layer N+1's CMD2 reads `bufs.moe_hidden()` for residual instead of `buffers.input`. Requires plumbing `bufs.moe_hidden` to layer N+1's `encode_residual_add` callsite, breaking the slice-5d-2 invariant that `buffers.input` is the residual source.

Lean: **(a-ii)**. Combine output goes directly into `linear_buffers.input`. The non-chained path keeps writing to `bufs.moe_hidden()` (for readback). So the encoder's combine target is parameterized per-call.

### C. `complete_deferred_experts_chained` — new free function

```rust
pub(crate) fn complete_deferred_experts_chained(
    slot: &mut Option<DeferredState>,
) -> Result<(), DeferredError>
```

Just `wait_until_completed` + clear. No buffer reads. Used by the layer loop on iterations where the previous layer chained.

### D. `step_internal` orchestration

Track `prev_layer_chained: bool` across iterations:
```rust
let mut prev_layer_chained = false;
for layer_idx in 0..v.num_layers {
    if layer_idx > 0 {
        if prev_layer_chained {
            deferred::complete_deferred_experts_chained(deferred)?;
        } else {
            deferred::complete_deferred_experts_into(deferred, moe_buffers, &mut buf_input_slice)?;
        }
    }
    let chain_next = gpu_combine && layer_idx + 1 < v.num_layers;
    let next_norm_off = if chain_next {
        Some(layer_caches[layer_idx + 1].input_layernorm_w)
    } else {
        None
    };
    if is_full {
        full_attn_layer_forward(..., gpu_combine, prev_layer_chained, next_norm_off);
    } else {
        linear_attn_layer_forward(..., gpu_combine, prev_layer_chained, next_norm_off);
    }
    prev_layer_chained = chain_next;
}
// post-loop: prev_layer_chained is true ⇒ final hidden state lives in linear_buffers.input
// (not bufs.moe_hidden). Drain via _chained, then read linear_buffers.input to host scratch
// for model.norm + LM head.
```

The post-loop drain after the *last* layer needs care: the last layer NEVER chains (gate is `layer_idx + 1 < num_layers`). So the last layer's combine still writes to `bufs.moe_hidden`, and the post-loop drain reads `bufs.moe_hidden` to a host scratch (existing path). ✓ No change needed.

### E. CMD1 input-norm skip

In `linear_attn_layer_forward` / `full_attn_layer_forward`, the existing input-norm prelude:
```rust
encode_rms_norm_bf16_into(cmdbuf, &rms_pipes, &buffers.input, ..., &buffers.normed, ...);
```
becomes conditional:
```rust
if !prev_layer_chained {
    encode_rms_norm_bf16_into(cmdbuf, &rms_pipes, &buffers.input, ..., &buffers.normed, ...);
}
```

`buffers.normed` is already populated by layer N's chain when `prev_layer_chained = true`.

### F. `layer_forward_dump` chain disable

`layer_forward_dump_inner` always passes `gpu_combine = ??? (existing)`, `prev_layer_chained = false`, `next_norm_off = None`. The forward functions accept these args — chain auto-disabled.

### G. Bias-check note

If 5d-8 lands flat too, we've hit a deeper bottleneck. Likely candidates: per-head Q/K rms_norm + RoPE on CPU; pread bandwidth; MoE routing CPU-side. The 5d-7 reprofile would tell us.

---

## 6. Diff test strategy

### Existing tests that must stay green

- `eval_token_matches_c_single_step` / `eval_prompt_matches_c_multi_token` — exercise the chain naturally (every non-last layer chains).
- `state_round_trip_rust` / `state_load_*` — the chain doesn't touch the wire format. Should be unaffected.
- `layer_forward_dump_close_c_vs_rust_*` — runs single layer; chain disabled by `layer_forward_dump_inner`. Unchanged behavior.
- `layer_forward_dump_close_c_vs_rust_full_attn_gpu_path` (slice 5d-7b) — same; chain disabled in dump hook.
- `cpu_combine_path_matches_c` (slice 4f-4) — chain doesn't apply to CPU combine. Unchanged.
- All 4 attn-kernel diff tests — unaffected.

### Regression risk

If the chain has a numerical drift bug (e.g., wrong `next_norm_off` lookup), `eval_token` / `eval_prompt` would catch it: layer N's chained-norm output flows into layer N+1's projections, drift compounds visibly across 60 layers.

### New tests (optional)

The chain is structurally a refactor of an existing kernel pair (rms_norm) into a new cmdbuf. The existing per-kernel diff for `rms_norm_sum_sq` + `rms_norm_apply_bf16` (slice 9e) already covers the kernels' correctness. The integration is covered by the existing end-to-end tests.

If a new test feels valuable, candidates:
- `chained_norm_matches_unchained_norm` — opens two RsCtx instances, one with chain enabled and one with chain disabled (via a debug toggle), runs the same prompt through both, compares logits. Bit-exact expected. **Probably overkill** — the existing eval_token / eval_prompt tests catch the same regression.

Lean: skip new tests. If something unexpected surfaces during plan-mode, add a chained-vs-unchained toggle test then.

---

## 7. Sequencing / commit plan

**Single commit** — `riir: Phase 5d-8 — chained CMD3 → next-layer normed`:

- New helper `encode_chained_combine_norm_into` in `gpu_norm.rs` (or just reuse `encode_rms_norm_bf16_into` — see §8.5).
- `gpu_batched_experts_encode_pre_staged` + `_begin_pre_staged` gain `chain_norm: Option<ChainNorm>` + `chain_targets: Option<ChainTargets>`.
- New `complete_deferred_experts_chained` in `deferred.rs`.
- `linear_attn_layer_forward` + `full_attn_layer_forward` + `post_attention_tail` accept `prev_layer_chained: bool` + `next_norm_off: Option<u64>`.
- `step_internal` threads `prev_layer_chained` through the layer loop.
- `layer_forward_dump_inner` always passes `prev_layer_chained = false`, `next_norm_off = None`.
- Run full diff suite. Run blallama A3B essay perf bench (Apollo × 3, --seed 42, max_tokens 512) — capture cold/warm tok/s.

If the diff suite passes and the perf bench shows the expected ~5%, commit. If perf is flat, still commit (the architectural alignment matters even if perf is noise) and note in the perf log.

---

## 8. Open decisions for next-session plan-mode

1. **`complete_deferred_experts` API: enum variant vs separate function?**
   - (a) `DeferredMode::GpuChained` variant + match in `complete_deferred_experts_into`.
   - (b) New `complete_deferred_experts_chained(slot)` free function; layer loop picks based on `prev_layer_chained`.
   - **Lean: (b).** No enum churn, callsite is explicit.

2. **`prev_layer_chained` plumbing: arg vs buffer-state field?**
   - (a) Pass `prev_layer_chained: bool` through layer-forward signatures.
   - (b) Add `LayerForwardBuffers::chain_state: ChainState` field; set/reset by encoder + CMD1.
   - **Lean: (a).** Explicit is better than stateful.

3. **Combine output target: stay on `bufs.moe_hidden` or rebind to `linear_buffers.input`?**
   - (a-i) Keep combine on `bufs.moe_hidden`, GPU→GPU copy to `linear_buffers.input` in chained-complete.
   - (a-ii) Combine writes directly to `linear_buffers.input` when chained. Encoder takes the output target as a param.
   - (a-iii) Combine on `bufs.moe_hidden`; layer N+1's CMD2 residual_add reads `bufs.moe_hidden` instead of `buffers.input`.
   - **Lean: (a-ii).** No extra memcpy, doesn't break the slice-5d-2 invariant.

4. **Chain enabled by default in production?**
   - Yes — same as C. The chain is on whenever `gpu_combine && layer_idx + 1 < num_layers`. No runtime toggle needed.

5. **New `encode_chained_combine_norm_into` helper, or reuse `encode_rms_norm_bf16_into`?**
   - The body is identical to `encode_rms_norm_bf16_into` — same kernel pair, same bindings, just different buffer references. A new helper is just a rename for callsite readability.
   - **Lean: reuse `encode_rms_norm_bf16_into`.** Less surface area; the call site already explains what it's doing via the buffer names.

6. **CPU-combine path interaction.**
   - `gpu_combine = false` (CPU-combine, slice 4f-4) doesn't dispatch `moe_combine_residual` — so there's nothing to chain off of. CPU-combine path skips the chain entirely. The orchestrator's `prev_layer_chained` should be set to `false` whenever `gpu_combine == false` for the current layer.
   - No code changes beyond the predicate gate. Mention in commit message.

7. **Last-layer behavior.**
   - Last layer never chains (gate `layer_idx + 1 < num_layers`). The post-loop drain stays as the existing readback path. No special-case code beyond the gate.

---

## 9. References

- 5d-7 perf log: `riir_moeflux_strategy.md` § blallama perf log (cold 7.45, warm 7.36).
- C-side scout (this file's §3): `~/Projects/moeflux/metal_infer/infer.m:5670..5775`.
- Riir code as of commit `5cfb521` (slice 5d-7b).
- Slice 5d-2 (`9153d95`) — established that `buffers.input` is the residual source AND the input-norm input.
- Slice 4f-3 (`fca7fab`) — wired `post_attention_tail` through deferred experts.
- Slice 4f-4 (`13d7b35`) — added the CPU-combine path; `DeferredMode::Cpu`.
- Slice 5d-5 (`de47fa3`) — `gpu_batched_experts_encode_pre_staged` (the function this slice modifies).
- Slice 9e (`6dfb51d`) — bit-exact per-PSO finding for `rms_norm_sum_sq` + `rms_norm_apply_bf16`.

---

## 10. What success looks like

After 5d-8 lands:
- Full diff suite (40+ tests) green at the existing tight floors.
- blallama A3B essay perf in the 7.7–8.0 warm tok/s range — closes another ~5% of the gap to C's 8.70.
- Activity Monitor: GPU utilization roughly stable at ~50% (fewer host roundtrips means GPU dispatches issue back-to-back faster, but per-kernel work is the same).
- Cold tok/s improves modestly (one fewer cmdbuf commit per layer × first-token).

If the warm number lands flat: the framing was still wrong and the actual remaining bottleneck is per-head Q/K rms_norm + RoPE on CPU, OR pread bandwidth, OR sampling-side cost. 5d-8's correctness work is still load-bearing for any future reorganization (the chain is a structural alignment with C), so commit anyway and update the perf log honestly.

If the warm number jumps significantly (>10%): the framing was *more* right than estimated — possibly the per-layer cmdbuf commit overhead was higher than expected, or the host roundtrip was on the critical path more than the profile suggested.

Either way, the next slice candidate after 5d-8 is **GPU per-head Q/K rms_norm + RoPE fusion** (now more attractive after 5d-7 confirmed the GPU has headroom).
