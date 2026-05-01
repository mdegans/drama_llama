# Qwen cmdbuf consolidation plan (next session)

Plan-of-record for the next moeflux optimization session, drafted at
the close of the prefetch-set-based-matching session
(2026-05-01). Pulls from the cogito-v2 session 2 architectural
foundation (`cogito_v2_full_gpu_session2_landed.md`) and applies the
equivalent consolidation work to Qwen3.5-397B-A17B's forward path.

## Premise

Per the cogito-v2 session 2 memo, the architectural foundation for
cmdbuf reduction (GPU residual stream, ChainToNormed, the depth-2
deferred ring) is already on the floor. Qwen has the same wins
available, plus more — full-attn layers run an extra synchronous CMD1
commit-wait between input-norm/projections and the post-attention
tail that doesn't need to be there, and the K-expert dispatch is
already its own commit (correctly so — it's the ring boundary).

After the set-based prefetch matching landed (commit f86fd93 in
moeflux, d59aa93 in drama_llama), `samply` shows ~60% main-thread
inclusive in `MTLCommandBuffer::waitUntilCompleted`. That sample
includes the GPU's actual compute time, which is unavoidable. What
we're targeting is the **submit + wait setup overhead per cmdbuf**,
plus the *gap on the GPU side between cmdbufs* (driver-side
scheduling latency, ~30–80 µs per cmdbuf empirically on M-series
UMA). With ~36 layers × ~3 commit-waits per layer = ~108 cmdbuf
boundaries per token. Reducing to ~36 boundaries per token (one per
layer) is the target.

## Phase 0 — Per-cmdbuf instrumentation (precursor)

**Concrete change.** Add `MetalBackend::record_cmdbuf_submit(label:
&'static str)` and `record_cmdbuf_wait_done(label, cpu_wait_ns,
gpu_runtime_ns)`. Use `metal::CommandBuffer::add_completed_handler`
to capture `gpu_start_time()` / `gpu_end_time()` (Metal exposes these
as f64 host-clock seconds; convert to ns). Aggregate per-label
totals into `cmdbuf_stats: AtomicU64` map (mirror the
`prefetch_stats` pattern at `mod.rs:377` / `prefetch.rs:198`).
Expose `pub fn cmdbuf_stats(&self) -> Vec<(&'static str, u64 count,
u64 cpu_wait_ns, u64 gpu_ns)>`.

**Diff oracle.** None for the instrumentation itself. Smoke-run
confirms counts match the static analysis below.

**Risk.** Read-only instrumentation. The `add_completed_handler`
allocation per cmdbuf is ~100 ns; ignorable.

## Phase 1 — Audit baseline (current cmdbuf count per layer)

**Linear-attn layer** (75% of layers):
1. `linear_attn_forward.rs:540–667` — CMD1: input rms_norm + 4
   projections + 5 linear-attn fused kernels. **commit + wait at
   line 666–667**.
2. `linear_attn_forward.rs:838–1039` (`post_attention_tail`) —
   CMD2+3 fused: o_proj + residual + post-attn-norm + gate +
   shared-FFN gate/up/swiglu/down. **commit + wait at line
   1037–1038**.
3. `expert_forward.rs:800–816`
   (`gpu_batched_experts_encode_pre_staged`) — CMD3b: K-expert FFN +
   combine + chained next-layer-norm. **commit, no wait** (deferred
   ring drains at next iteration's top).

= **3 cmdbufs/layer** for linear-attn. The first two synchronously
commit-and-wait before the K-expert. Only the K-expert is deferred.

**Full-attn layer** (every 4th, 25% of layers):
1. `full_attn_forward.rs:174–231` — CMD1: input rms_norm + 3
   projections (q/k/v). **commit + wait at line 229–230**. Followed
   by host readback (`read_buffer_to_vec` × 3 at lines 234–236),
   per-head Q/K rms_norm + RoPE + KV append + GPU SDPA stage.
2. `linear_attn_forward.rs:838–1039` (shared `post_attention_tail`)
   — CMD2+3: 4 GPU-attn kernels (when fast path active) + o_proj +
   residual + post-attn-norm + gate + shared-FFN. **commit + wait**.
3. `expert_forward.rs:800–816` — CMD3b: K-expert + combine + chain.
   **commit, no wait**.

= **3 cmdbufs/layer** for full-attn, but the host-bounce between
CMD1 and CMD2 (q_proj_out / k_out / v_out readback) blocks GPU
pipelining harder than the linear-attn case.

The "6+ per layer" framing from the cogito memo applied to the MLA
path before Phase 5 landed. Qwen post-5d-3 is already at 3.

## Phase 2 — Fold linear-attn CMD1 into CMD2+3

**Concrete change.** Linear-attn CMD1 work has no host-side
dependency between it and CMD2+3. Refactor `post_attention_tail` to
accept `cmdbuf: &CommandBufferRef` instead of creating one. Linear-
attn caller creates the cmdbuf, encodes CMD1 dispatches, calls
`post_attention_tail(cmdbuf, …)` which encodes CMD2+3 work into the
same cmdbuf, returns. Caller commits-without-wait — K-expert
encoder picks up from a fresh cmdbuf.

Drop `linear_attn_forward.rs:666–667` commit+wait entirely.

Net per-layer: **3 → 2 cmdbufs** for linear-attn.

**Diff oracle.** Existing layer_forward_dump_diff (CPU vs Rust GPU).
Expected drift: zero — encoders within one cmdbuf already serialize.

**Risk.** Borrow surgery on `post_attention_tail` rippling to both
callers (`linear_attn_layer_forward` line 676,
`full_attn_layer_forward` line 397). Pipeline lookups dedupe to one
fetch site at the top of the merged forward.

## Phase 3 — Fold full-attn CMD1 into CMD2+3 (host-bounce removal)

**Concrete change.** Today full-attn does q/k/v projections on GPU,
reads them back to host (`full_attn_forward.rs:234–236`), runs
per-head Q/K rms_norm + RoPE on CPU, copies host data back into
gpu_attn_q/kv_k/kv_v (lines 308–352), then encodes CMD2+3.

**Recommended: 3b — Move per-head Q/K rms_norm + RoPE to GPU.**
Kernels exist for MLA in `gpu_mla.rs`; per-head rms_norm has
precedent in `gpu_norm::encode_rms_norm_bf16_into`. Implement
`encode_qk_rms_norm_per_head` and `encode_apply_rotary_emb` (port
`rope.rs` → kernel). Full-attn collapses to **1 cmdbuf** for the
whole prefix (CMD1 + CMD2 + CMD3a/b non-K-expert), matching
linear-attn after Phase 2.

**Diff oracle.** Per-stage tests:
- `tests/rms_norm_per_head_gpu.rs` (new) — bit-exact vs
  `rms_norm_per_head_cpu`. Tolerance: cosine ≥ 0.9999, max_abs_diff ≤
  4 ULP.
- `tests/rope_gpu.rs` (new) — vs `apply_rotary_emb`. Tolerance: ≤ 4
  ULP per channel.
- End-to-end `layer_forward_dump_diff` for full-attn layers.

**Risk.** RoPE on GPU needs the position-dependent sin/cos table
(`rope.rs` constructs from theta_base). Persistent buffer at RsCtx
init (mirrors `LmHeadGpu` pattern).

## Phase 4 — Cross-layer cmdbuf chaining

**Concrete change.** With Phases 2+3 landed, layer N emits 2
cmdbufs (unified pre-K-expert + K-expert). The deferred-ring's
K-expert is already chained into the next-layer-norm via
`ChainToNormed`. The remaining per-layer commit-wait is the *first*
cmdbuf.

- **4a — limit:** Can't fully remove the wait — `moe_router_cpu`
  reads `batch_out[4]` (gate logits) and `batch_out[5]` (shared-gate
  score) which the K-expert dispatch needs.
- **4b — reorder for overlap:** Issue the gate-logits readback non-
  blockingly. After the unified cmdbuf is committed, CPU does other
  work (drain older deferred entry, fire next-layer prefetch) while
  waiting. Today `complete_deferred_experts_chained` runs *before*
  this layer's commit (`mod.rs:1771`). Re-order: kick off pre-K-
  expert cmdbuf commit *first*, then deferred-drain CPU work, *then*
  wait for cmdbuf's gate logits.

**Diff oracle.** End-to-end `layer_forward_dump_diff` + smoke.
Pure scheduling, no mathematical change.

**Risk.** Re-borrow surgery in `step_internal` (mod.rs:1769–1875).
Reordering needs a small state-machine to track the layer's
pre-K-expert cmdbuf as a future. Pattern matches the deferred ring;
reuse the same shape.

## Phase 5 — Full-layer single-cmdbuf (stretch)

**Concrete change.** Move `moe_router_cpu` to GPU (`encode_moe_router`:
top-K + softmax + normalize on `batch_out[4]`, output `routed_indices[k]`
and `routed_weights[k]` into device buffers). Then expert-data `pread`
becomes the only CPU work — reads `routed_indices` from a shared-storage
buffer after unified cmdbuf completes, and only that one wait remains
per layer.

Per-layer reduces to **1 unified cmdbuf + 1 K-expert cmdbuf = 2
cmdbufs/layer** (already the linear-attn count after Phase 2, but
now with full-attn parity and the host-bounce gone).

**Diff oracle.** New `tests/moe_router_gpu.rs`. Expert selection is
integer; tolerance is exact match on `routed_indices`, ≤ 4 ULP on
`routed_weights`.

**Risk.** Top-K on GPU is fiddly for K=4..8. Simple repeated-argmax
kernel for K ≤ 8 (Qwen 397B uses k_active in this range).
4-iteration sequential kernel inside the unified cmdbuf — still
serializes but no commit-wait.

## Estimate the win

Static count: 3 → 2 cmdbufs/layer (Phase 2) for linear-attn = 27
layers × 1 = 27 cmdbufs/token saved. + Phase 3b: full-attn 3 → 1 =
9 layers × 2 = 18 cmdbufs/token saved. Total Phase 2+3b: **~45
cmdbufs/token eliminated** out of ~108. Phase 4b's reorder reclaims
overlap on the remaining ~36 commits.

Submit+driver-scheduling overhead per cmdbuf on M-series UMA is
empirically 30–80 µs (call it 50 µs avg). 45 × 50 µs = **~2.25
ms/token** of pure overhead removed.

The bigger expected win is **GPU pipeline gap reduction**. Each
commit-wait is a ~150 µs GPU stall. 45 × 150 µs = ~6.75 ms/token.
With Phase 4b overlap, realistic recoverable fraction of the 60%
Metal-wait sample is **15–25%** — main-thread wait drops 60% →
45–50% inclusive. Phase 5's router consolidation adds another 3–5%.

## Critical files for implementation

- `crates/moeflux/src/riir/linear_attn_forward.rs` — primary refactor
  target (CMD1+CMD2+3 fold, `post_attention_tail` signature change to
  take `cmdbuf` ref).
- `crates/moeflux/src/riir/full_attn_forward.rs` — Phase 3
  host-bounce removal (per-head Q/K norm + RoPE + KV append GPU port).
- `crates/moeflux/src/riir/expert_forward.rs` — `emit_batched_experts`
  entry-point shape stays the same. K-expert stays its own cmdbuf
  (ring boundary, correct).
- `crates/moeflux/src/riir/mod.rs` — `step_internal` loop (1769–1875),
  Phase 4b CPU/GPU overlap reorder.
- `crates/moeflux/src/riir/metal.rs` — Phase 0 cmdbuf instrumentation.

## Run commands (canonical for next session)

```bash
# Build:
cd ~/Projects/drama_llama
cargo build --release --bin blallama \
  --features axum,cli,toml,moeflux-model-qwen3-5-a17b

# Clean baseline (matches what the 1.95 tok/s pre-change baseline
# was measured at — max_tokens=512):
./target/release/blallama "/Volumes/Temp Backup/models/blallama" \
  --backend moeflux --port 11435 --seed 42

# Profile:
./profile.py --model a17b --duration 60 --max-tokens 128 \
  --prompt "Write a long, detailed essay …"
```

## Calibration note

Use **max_tokens=512** for tok/s benchmark comparisons — that's what
the original 1.95 tok/s baseline was measured at. Hello-prompt /
26-token runs are fast smoke tests but the prefill amortizes
poorly. Long essay + 512 tokens is closer to steady-state long-form
generation and matches the comparison surface.
