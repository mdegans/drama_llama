# RIIR moeflux: strategy

Captured 2026-04-27. Decision: rewrite moeflux's host-side dispatch
in Rust. Triggered by bisect findings (see
`blallama_session_state_pollution.md`) — the bugs found are exactly
the kind that disappear in a Rust rewrite (process-globals with raw
pointers; semantic invariants the C API silently violates).

## Scope

In:
- `metal_infer/infer.m` (8002 lines of host-side Objective-C
  dispatch) → idiomatic Rust.
- `metal_infer/model_variant.h` → `mod variants` enum.
- `metal_infer/tokenizer.{h,bin}` → `mod tokenizer`.
- `crates/moeflux-sys/` → deleted at cutover.

Out (stays as-is):
- `metal_infer/shaders.metal` (1385 lines) — Metal kernels keep
  their license attribution (Mike wrote them) and don't get
  rewritten.
- Python conversion scripts (`extract_weights.py`,
  `repack_experts.py`, `export_*.py`) — Python is the right tool
  for the model-prep pipeline.
- `metal_infer/main.m` (1847 lines, CLI driver) — drama_llama is
  the only consumer; no need.

## Ground rules

- **No `Arc`.** Single `&mut Ctx` for inference. Deferred-experts
  state is a field on `Ctx`, not a global. Metal command buffers
  RAII'd within method scopes.
- **`metal-rs`** for Metal bindings (mature; used by candle,
  mistral.rs).
- **Compile-time variant selection** kept 1:1 during the port (the
  existing `MOEFLUX_MODEL_*` Cargo features) so differential tests
  are apples-to-apples. Runtime-variant rearchitecture is post-port.
- **Bit-for-bit faithful port first; fix bugs as separate commits.**
  When the C path has a known bug or surprising semantic, port it
  faithfully so the diff tests stay load-bearing — a failing diff
  unambiguously means a porting error. Leave a `// FIXME(riir):`
  pinned to `infer.m:LINE` describing the bug, and queue the fix as
  a Phase 7 (post-cutover) commit. Fixing while porting is asking
  for trouble: a failing diff could be either a porting mistake or
  an intentional fix, and you can't tell which without re-reading
  the patch. The bisect findings (cross-Ctx NaN, partial-truncate
  divergence) are exactly this shape — they get faithful ports plus
  FIXMEs, and the typed `Result<(), CannotTruncateLinear>` lands as
  a separate post-cutover slice.
- **No new build systems.** `cargo build`, `cargo test`. Shader
  compile goes through `build.rs` or runtime `MTLLibrary::with_source`.

## Branch & cutover

- Work in `~/Projects/moeflux` on a `riir` branch off main.
- Single squashed cutover commit at Phase 6.
- drama_llama keeps using `moeflux::*` throughout — no churn there
  until cutover.

## Phases

| # | Phase | Approx |
|---|-------|--------|
| 0 | Differential harness (`tests/diff/`, `DiffBackend` trait, comparison helpers; Rust side panics until Phase 1) | 1–2h |
| 1 | Pure-Rust foundations: `mod variants`, `mod weight_file`, `mod vocab`, `mod tokenizer`, `mod quant` | 4–8h |
| 2 | Metal infrastructure: `metal-rs` device/queue/library/pipeline cache, RAII buffers | 4–8h |
| 3 | Forward pass bottom-up: embedding, RMSNorm, RoPE, full attn, linear attn, MoE router, MoE dispatch, LM head — diff-tested per kernel | 8–16h |
| 4 | Top-level: `eval_prompt`, `eval_token`, `memory_*`, `state_save`/`load` | 4–8h |
| 5 | API stabilization, drama_llama full test run | 2–4h |
| 6 | Cutover: delete C, move shaders, squash | 1–2h |
| 7 | Post-cutover (separate PRs): typed `memory_seq_rm`, multi-Ctx (now: `g_deferred` + `layer_cache` together), runtime variant dispatch, expanded coverage | 4–8h |

Total: 28–48h focused work.

## Differential harness shape

`crates/moeflux/tests/diff_oracle.rs` exposes a `DiffBackend` trait
that both the existing C path (via `moeflux-sys`) and the new Rust
path implement.

**Originally planned**: end-to-end logits comparison with argmax
match + top-K Jaccard ≥ 0.95 + cosine ≥ 0.99.

**Refined (Phase 0 finding)**: end-to-end logits are NOT a useful
oracle. The C path itself is non-deterministic across
`memory_clear` for the same prompt (cosine ≈ 0.65–0.76 between
two C-side runs), so we can't expect Rust-vs-C to be tighter than
that. The original bisect's `memory_clear_*` tests were green only
because they used argmax + trajectory equality, which both pass
trivially when greedy decoding lands in the same attractor (a
runaway 5073 token in our synthetic-prompt case).

**Real diff strategy (Phase 3+)**: intermediate-tensor checkpoints.
Both backends expose hooks that dump per-layer outputs (post-RMSNorm,
post-attention, post-MoE, etc.). Compare layer-by-layer where Metal
nondeterminism has had less chance to accumulate. The earliest
divergence between C and Rust pinpoints the exact kernel that's
been mis-ported.

The Phase 0 harness in place today has the trait + impls + helpers
+ a scaffold-validation test (`harness_loads`). The intermediate-
checkpoint hooks come in Phase 3 when there's actually something to
compare.

## Phase 3 progress (in-flight)

Bottom-up kernel ports, diff-tested per kernel against the C path.
Each landed kernel is bit-exact (deterministic CPU work) or within
the cosine/Jaccard floors (Metal nondeterminism territory).

| Kernel | Landed | Diff signal | Notes |
|--------|--------|-------------|-------|
| embedding | 2026-04-26 (4216e2f) | bit-exact, 8 tok × 2048 elem | First per-kernel hook; CPU 4-bit dequant. |
| RMSNorm (CPU) | 2026-04-26 (a7866cc) | bit-exact, 4 tok × 3 norms × 2048 elem | model.norm + layer-0 input_layernorm + post_attention_layernorm. Reduction-order-matched. |
| RoPE | 2026-04-26 (250f5e8) | ULP-bounded ≤128 (observed 34 max), 5 positions × q+k | First non-bit-exact kernel. See "Tolerance regimes" below. |
| RMSNorm per-head (CPU) | 2026-04-27 (5adabc5) | bit-exact, Q (16×256) + K (2×256) at first FA layer | Per-head Q/K norm extracted from full_attention_forward. Same arithmetic shape as whole-vector RMSNorm. |
| SDPA core | 2026-04-27 (7d3963a) | cosine = 1.000000, max_abs_diff ≤ 1.5e-8 across kv_len ∈ {1, 8, 64, 512} | Q·K^T scores + softmax + V weighted sum + sigmoid gate. ULP-bounded territory (2× expf per output element). kv_len=1 is bit-exact (softmax(s)=1 skips expf). Cosine ≥ 0.9999 + max_abs_diff ≤ 1e-3 × max_abs_out floors. |
| LM head (CPU) | 2026-04-27 (slice 6) | bit-exact, 248320 logits × 2 inputs (synth + real-derived hidden) | 4-bit dequant matvec, full vocabulary projection. Required `mul_add` to match clang's FMA contraction (`acc += (val*scale+bias)*x[i]` fuses into 2 fmadd on AArch64 at `-O3` with default `-ffp-contract=on`). Without `mul_add`: cosine still 1.0 but ~3.5e-7 relative drift from unfused multiply-then-add — the same FMA gap rope.rs noted. New diff hook bypasses `fast_dequant_matvec`'s GPU dispatch via direct `cpu_dequant_matvec` call. |
| MoE router | 2026-04-27 (slice 7) | bit-exact, max_ulp=0 across 2 score patterns (clear-winner + mild-spread) | softmax → top-K (selection-sort slot order) → normalize, on NUM_EXPERTS=256 logits. Predicted ULP-bounded (libm `expf`); turned out bit-exact because clang doesn't auto-vectorize the softmax loop (sequential `sum +=` reduction blocks `vexpf` substitution). Rust scalar `f32::exp()` and clang scalar `expf` produce identical bytes. New `mf_moe_router_cpu` C hook composes `cpu_softmax` + `cpu_topk` + `cpu_normalize_weights`. |
| linear-attn primitives (8a) | 2026-04-27 (slice 8a) | bit-exact, max_ulp=0 across all three sub-kernels | Three small CPU helpers from `linear_attention_forward`: `rms_norm_bare` (no weight; LINEAR_KEY_DIM=128), `conv1d_step` (depthwise 1D conv + SiLU; LINEAR_CONV_DIM=8192 channels × CONV_KERNEL_SIZE=4 — used `mul_add` proactively at FMA sites per the LM head finding), and `rms_norm_gated` (RMSNorm × SiLU × weight; LINEAR_VALUE_DIM=128). All three use the layer-0 real `linear_attn.*` weight tensors. Predicted: bare bit-exact, conv ULP-bounded (SiLU `expf`), gated ULP-bounded (SiLU `expf`). Landed: all three bit-exact — same pattern as MoE router, scalar `expf` in element-wise loops with no shared cross-iter dependency stays scalar on both sides. |
| linear-attn recurrence (8b) | 2026-04-27 (slice 8b) | state bit-exact (0/524288); out_values ULP-bounded max_ulp=12, max_abs_diff=1.9e-6 / max_abs_out=8.6 (~2.2e-7 relative) | Per-v-head decay → kv_mem → delta → state update → output. Standalone `cpu_gated_delta_recurrence` C helper (parallel to the inline production loop, not refactored from it) keeps prod codegen unchanged while exposing the test surface. All FMA contraction sites use `mul_add` per LM head findings. State mutations land bit-exact (element-wise updates, no cross-iter dependency). The per-head output read-out `sum += S[ki] * q[ki]` lands ULP-bounded — same dot-product-reduction-vectorization gap as SDPA, where clang's NEON horizontal-sum reorders the reduction tree vs Rust's strictly sequential mul_add chain. Curious that `kv_mem` reduction (same loop shape) stays bit-exact — probably because its result feeds the inner state-update loop and clang prefers scalar there to keep the FP register pipelined. |
| MoE dispatch — 9a single-expert GPU FFN | 2026-04-27 (slice 9a) | bit-exact — cosine=1.000000, max_abs_diff=0.0 across HIDDEN_DIM | First GPU kernel under diff. Four dispatches per call: gate matvec (`dequant_matvec_4bit_v3`) → up matvec → `swiglu_fused` → down matvec, all in one MTLCommandBuffer. Fresh `MtlBuffer` allocation per call (persistent buffers come in 9b). Lazy `MetalBackend` on `RsCtx`. C-side hook `mf_gpu_expert_forward` wraps the existing internal `gpu_expert_forward(MetalCtx*)` — passes `g_metal`, not the `mf_ctx` (file-scope MetalCtx owns the pipelines + buffers). Diff harness uses synthetic 4-bit blob (PRNG nibbles + BF16 0x3C00 scales / 0 biases) — same bytes both sides. Empirical surprise: the cosine/Jaccard regime I reserved for "Metal kernels" was not needed here — `simd_sum` reductions are deterministic per pipeline-state object, atomic ops live downstream (which I'd assumed was 9b). 4-bit only; FIXME-noted for 2-bit at Phase 7. |
| MoE dispatch — 9b batched K-expert + combine | 2026-04-27 (slice 9b) | bit-exact — cosine=1.000000, max_abs_diff=0.0 across HIDDEN_DIM with K=4 experts | Persistent `MoeBuffers` (16 slots × {data, gate, up, act, out} + h_mid + shared_out + moe_hidden + 18-float combine_params) lazily allocated on `RsCtx` via `metal_and_moe_mut`. ~28 MB total on A3B. Hook stages K expert blobs, runs `gpu_encode_experts_batched` (2K encoders) + `moe_combine_residual` in one cmdbuf, reads back. Combine kernel pattern: `hidden = h_mid + Σ weights[k] × expert_out[k] + sigmoid(gate) × shared_out`. Output magnitude was meaningful (8.22e-4, not near-zero) so the bit-exact result is real — second hypothesis disproved: `moe_combine_residual` is ALSO atomic-op-free; per-thread reads K expert buffers and computes its own sequential sum. The cosine/Jaccard regime I reserved for "where atomic ops stack" doesn't materialize in this kernel either. Genuinely-nondeterministic territory probably starts in 9e (the `rms_norm_sum_sq` threadgroup reduction) or 9c (async pread + cache eviction ordering). |
| RMSNorm (Metal) | — | — | Cosine/Jaccard tolerance — fast_math + Metal reduction order diverge. |
| MoE dispatch — 9c expert I/O subsystem | 2026-04-27 (slice 9c) | byte-exact across 6 (layer, expert) probes covering full-attn / linear-attn / first-block / last-block; 1.77 MB per blob | `ExpertFiles` RAII struct on `RsCtx` opens `experts_dir/packed_experts/layer_NN.bin` for every layer eagerly at `RsCtx::open`. Missing files leave the slot at `None` per the C tolerance semantics. `read_expert(layer, expert, out)` uses `std::os::unix::fs::FileExt::read_at` (i.e. `pread64`) at `expert_idx * EXPERT_SIZE`. Async pread thread pool, mmap, LRU caches, malloc-cache, LZ4 decompression all stay out of scope (slice 9f if needed). C-side hook `mf_load_expert_bytes` calls `pread(ctx->layer_fds[i], ...)` directly — same syscall both sides. Byte-equality is exactly what we'd expect; the test is a sanity guard against indexing bugs (wrong subdir, wrong offset arithmetic, wrong file-naming format) rather than numerical concerns. |
| MoE dispatch — 9d deferred experts state | — | — | `g_deferred` → `Ctx`-owned struct. Cross-Ctx NaN bug source — faithful port keeps the lossy semantic but lifetime-binds it; FIXME for the typed `CannotTruncateLinear` Phase 7 fix. Best landed alongside Phase 4 integration since the diff oracle pattern doesn't naturally exercise async sequencing. |
| MoE dispatch — 9e GPU rms_norm fused | 2026-04-27 (slice 9e) | bit-exact — cosine=1.000000, max_abs_diff=0.0 across HIDDEN_DIM (max_abs_out=4.737, real magnitude) | `rms_norm_sum_sq` + `rms_norm_apply_bf16` chained in one cmdbuf. **First kernel under diff using threadgroup-shared memory across SIMD groups** (256 threads → simd_sum → 32-element threadgroup-shared array → second-stage simd_sum → 1 scalar). Per-call scratch buffer alloc on both sides; no dependency on the production CMD3 fast-path's deferred state machine. C+Rust hooks take bf16 weight bytes directly so the test doesn't depend on tensor-name lookup. Real `model.norm.weight` bytes used (read via `WeightFile::tensor_bytes`). The threadgroup-shared reduction also lands bit-exact. |
| MoE dispatch — 9f LZ4 + 2-bit + caches | — | — | Exotic quantization paths + the LRU/malloc expert caches. May be deferable to Phase 7 if the basic path is enough for cutover (unlikely — caches matter for tok/s). |

## Phase 4 progress (in-flight, 2026-04-27)

| # | Slice | Landed | Diff signal | Notes |
|---|-------|--------|-------------|-------|
| 4a | state structs + memory ops | 2026-04-27 (`9a1d60d`) | structural pos_max equivalence on empty state | KvCache + LinearAttnState + LayerState in `riir::state`, allocated per-layer in `RsCtx::open` (~40 GB lazy-committed virtual address space for KV on A3B; matches C `calloc`). Faithful port of the lossy partial-linear truncation; FIXME for the typed `Result<(), CannotTruncateLinear>` Phase 7 fix. |
| 4b | layer-output dump hook | 2026-04-27 | C-side sanity (finite output) | `mf_layer_forward_dump(ctx, layer_idx, pos, hidden_in, hidden_out)` brackets `fused_layer_forward` with `discard_deferred_experts` / `complete_deferred_experts` so `g_deferred.active` is 0 on entry and exit. `RsCtx::layer_forward_dump` stub'd for 4c/4d. The first non-kernel hook to call into the production forward path — surfaced a third cross-Ctx state-pollution bug (the file-scope `layer_cache` weight-pointer cache); see the Phase 7 list. |
| 4c | linear-attn fused_layer_forward | 2026-04-27 (`f44fc9c`) | cosine=1.0000000 max_abs_diff=4.1e-8 (effectively bit-exact) | Full GPU production path through linear-attn layers via `metal-rs` encoders. Six new modules: `mtl_weight_buf` (wraps mmap as MTLBuffer), `layer_weight_cache` (per-Ctx tensor offsets, fixes the 4b cross-Ctx bug class on the Rust side), `gpu_matvec` (4/8-bit dequant matvec encoder), `gpu_linear_attn` (5 linear-attn kernels), `linear_attn_forward` (composer + ~63 MB persistent buffer set on A3B). One-line defensive fix in C `mf_free_model` (reset `layer_cache_built=0`) lets the diff suite run end-to-end without cross-Ctx pollution; full Phase 7 cleanup still pending. K=experts_per_tok was the load-bearing bug — the Rust port had hardcoded `VARIANT.num_experts_per_tok` (architectural max, 8 for A3B) instead of using the runtime arg (4 in the test); finding it required wiring an `*_intermediates` diff hook and walking the per-stage diffs until the routing checkpoint flagged different top-K. The diagnostic infra stays in tree as a 4d debugging tool. |
| 4d-pre | extract `post_attention_tail` (refactor) | 2026-04-27 (`621e134`) | 4c stays green | `LinearAttnBuffers` → `LayerForwardBuffers` (+ 3 new fields for full-attn projection outputs: `q_proj_out`/`k_out`/`v_out`). `LinearAttnForwardError` → `LayerForwardError`. Both old names kept as `pub type` aliases. `linear_attn_layer_forward` shrinks to input rms_norm + CMD1 (projections + 5 fused linear-attn kernels) and hands off to `post_attention_tail` (CMD2 + CMD3 + MoE) via an `OProj` adapter that names the right o_proj weights/in_dim. Sets up 4d to add `full_attn_layer_forward` calling the same tail. |
| 4d | full-attn fused_layer_forward | 2026-04-27 (`44879b2`) | cosine=1.0000000 max_abs_diff=3.576e-7 max_abs_out=2.772 rel=1.290e-7 | New module `full_attn_forward.rs`. Pipeline: CPU input rms_norm + 3-matvec CMD1 (q/k/v) + per-head split into q+gate + per-head Q/K rms_norm + RoPE + KV append (host) + CPU SDPA + stage attn_out into `batch_out[6]` + `post_attention_tail`. `RsCtx::layer_forward_dump` now branches on `(layer_idx + 1) % full_attn_interval == 0` and dispatches the right forward / extracts the right `LayerState` variant. GPU attention fast path (`gpu_attn_fuse`, gated on `kv_len >= 32`) and GPU KV-cache mirror buffers are out of scope — FIXMEs in place; the dump-hook test at pos=0 never engages either. Per-stage drift composition predicted; observed even better than predicted thanks to RoPE + SDPA hitting kv_len=1 trivial cases. |
| 4e | deferred-expert state machine (was 9d) | 2026-04-27 (`be80c56`) | bit-exact — cosine=1.0000000 max_abs_diff=0.0 across HIDDEN_DIM on both new tests | Smallest standalone version per the strategy doc's "Suggested next-session order". `DeferredState` struct on `RsCtx` (`Option<DeferredState>` field; `Some` ↔ C `g_deferred.active=1`); three methods `begin_deferred_experts` / `complete_deferred_experts` / `discard_deferred_experts`. The synchronous `gpu_batched_experts_forward` is now a thin wrapper around a new `gpu_batched_experts_encode` helper; the deferred path uses the same encode helper, commits async, stashes the owned cmdbuf. C side: matching `mf_begin` / `mf_complete` / `mf_discard` FFI hooks; static `oracle_batched_experts_encode` shared between sync + deferred. Cross-Ctx NaN bug **structurally absent** in the Rust port — `complete_deferred_experts` writes through caller-supplied `&mut [f32]`, never a stored raw pointer; documented as `NOTE(riir-bugfix):` not FIXME, since a single-Ctx diff test cannot distinguish faithful-with-bug from faithful-without-bug and the bug only manifests across two `Ctx`s. Out of scope (4f): `gpu_combined=false` CPU-combine path; fast/slow split in `fused_layer_forward`; rewiring `post_attention_tail` to call `begin` instead of the synchronous variant. Two new diff tests (`deferred_experts_begin_complete_close_c_vs_rust` + `deferred_experts_discard_clears_state_c_vs_rust`) bit-exact; full diff suite 24/24 green in 228s. |
| 4f-1 | Variant::layer_kind + LayerKind enum | 2026-04-27 (`fd63c0a`) | refactor; existing tests stay green | Replaces inline `(layer_idx + 1) % full_attn_interval == 0` modulo with `Variant::layer_kind(i) -> LayerKind { LinearAttn, FullAttn }`. Five callsites updated. New unit test asserts kind sequence agrees with legacy modulo for every layer in active variant. Trivial today; load-bearing for DeepSeek-V3 (first N dense, rest MoE+MLA). |
| 4f-2 | LayerWeightCache nested-attn refactor | 2026-04-27 (`d496b05`) | refactor; existing tests stay green | Splits flat `Option<u64>` struct into `LayerWeightCache { input_layernorm_w: u64, post_attention_layernorm_w: u64, attn: LayerAttnW, gate: GateW, shared: SharedExpertW }` where `LayerAttnW` is a tagged enum `{ LinearAttn(LinearAttnW), FullAttn(FullAttnW) }`. `build()` errors via `MtlWeightBufError::MissingTensor` at build-time; `~60 LOC` of `require()` ladder in `post_attention_tail` + per-layer-forward goes away. Strategy doc's flat-enum proposal overruled because `post_attention_tail` extraction (4d-pre) made common/attn separation cleaner. |
| 4f-3 | post_attention_tail rewired through deferred experts | 2026-04-27 (`fca7fab`) | layer-forward diff stays at cosine 1.0; new back-to-back parity test bit-identical | Replaces synchronous `gpu_batched_experts_forward` with `gpu_batched_experts_begin` (slice 4e). Three free functions in `deferred.rs` (`gpu_batched_experts_begin`, `complete_deferred_experts_into`, `discard_deferred_experts_in`) take disjoint borrows; `RsCtx` methods are thin wrappers. `post_attention_tail` signature gains `deferred: &mut Option<DeferredState>`. `RsCtx::layer_forward_dump` brackets pre-call with `discard_*` (defensive) and drains post-call into `linear_buffers.input` (reconstitutes synchronous single-step contract). **Unrelated bug fixed:** `RsCtx::memory_clear` was only resetting host-side `LayerState`, not GPU-side linear-attn recurrence. C side stores recurrence host-resident and pushes to GPU each call (host alone suffices). Rust port treats GPU as canonical (kernels mutate in place). Without GPU reset, back-to-back forwards after `memory_clear` see stale recurrence. New test `layer_forward_dump_back_to_back_no_deferred_leak` asserts bit-identical outputs across 5 iterations with `memory_clear` between. |
| 4f-4 | CPU-combine path (`gpu_combined=false`) | 2026-04-27 (`13d7b35`) | new `cpu_combine_path_matches_c` cosine 1.0000000 max_abs_diff 4.098e-8 | Ports C's `gpu_combine = 0` finalize (infer.m:4106..4129). New `cpu_ops` module (`cpu_vec_madd` + `cpu_sigmoid_scalar`, FMA via `mul_add`). `DeferredMode { Gpu, Cpu { h_mid, shared_out, expert_weights, shared_gate_score } }` enum on `DeferredState`. `gpu_batched_experts_encode/begin` accept `gpu_combine: bool`; `complete_deferred_experts_into` matches mode and runs CPU-combine when `Cpu`. New `RsCtx::layer_forward_dump_with_gpu_combine` test entry. Production callers all pass `true` — CPU branch reached only via the test. Slice 4f-perf will plumb the C-mirrored `should_gpu_combine` predicate. |
| 4f-5 | step_internal + eval_prompt + eval_token | 2026-04-27 (`0267bda`) | preserves layer-forward cosine 1.0 | Replaces `todo!()` stubs at mod.rs:723..741. `step_internal(token, pos, logits_out: Option<&mut [f32]>)` mirrors C `mf_step_internal` shape: embed → per-layer loop with drain-then-forward → final drain (or discard) → CPU `model.norm` rms_norm → CPU `lm_head_cpu` → write logits. `gpu_combine = true` for every layer (preserves 4f-3 default; slice 4f-perf gates per-layer). Field-disjoint borrow pattern same as `layer_forward_dump_inner`. Empty `tokens.len()` returns `Ok(())`. |
| 4f-6 | end-to-end eval_prompt / eval_token diff | 2026-04-27 (`ceaa3ba`) | `eval_token`: argmax c=17 rs=17 cosine=1.0000000 max_abs_diff=2.670e-5 rel=1.958e-6 jaccard=1.0000; `eval_prompt(8tok)`: argmax c=198 rs=198 cosine=1.0000000 max_abs_diff=2.241e-5 rel=1.703e-6 jaccard=1.0000 | Two end-to-end tests against C. Fresh-Ctx-per-side (sidesteps the file-level "memory_clear non-determinism" warning — that's intra-Ctx). Floors set at cosine ≥ 0.9999; both pass with substantial margin. Confirms slice 9 per-PSO bit-exactness composes end-to-end across 40 layers + final RMSNorm + LM head. Full diff oracle suite 28/28 green in 273.6s. |
| 4f-perf | fast/slow split (post-correctness) | — | bit-exact vs slow-path | Deferred from 4f per the plan. GPU-side CMD3 combine + chain into next layer's CMD1 input via `should_gpu_combine` (mirrors `infer.m:5668..5673`). ~50ms/token expected (~0.83ms/layer × 60). Bit-exactness vs slow path verifies correctness — same arithmetic, different scheduling. |
| 4g | state_save / state_load | 2026-04-27 (`037f74c`) | byte-identical state_size; round-trip max_abs_diff=0; bidirectional wire compat cosine 1.0000000 | New `state_snapshot.rs` module ports `mf_state_save` / `mf_state_load` / `mf_state_size` (infer.m:8485..8700). Wire format: 8×u32 header (magic 'MFLX' + version + 6 shape constants) then per-layer body (full-attn = i32 len + K + V; linear-attn = conv_state + ssm_state). GPU recurrence read back from / written into `linear_buffers.{conv_state,delta_state}` (Metal canonical store). state_load is two-pass: header + per-layer-length preflight before mutation. Drains pending deferred dispatch (moeflux.h:481 contract). RsCtx methods are thin wrappers. Four diff tests: state_size match, Rust↔Rust round-trip (bit-exact), Rust→C wire compat (cosine 1.0000000 max_abs_diff 2.575e-5), C→Rust wire compat (cosine 1.0000000 max_abs_diff 2.718e-5). 66 MB snapshot at 4-token prefill on A3B (mostly the 45-layer GatedDeltaNet ssm_state, fixed-size). Unlocks the disk-cache use case Mike flagged: agent runner saving common-and-expensive prefixes (system+tools) across process restarts. |

## Phase 5 progress (2026-04-27)

Phase 5 = API stabilization + drama_llama integration + perf parity
to the C path. Slices land here, not under Phase 4.

| # | Slice | Landed | Notes |
|---|-------|--------|-------|
| 5a | public Ctx/Error → Rust port under `riir-port` | 2026-04-27 (moeflux `ee27663` + drama_llama `7fced10`) | `mod imp` promoted to `pub mod imp` so the diff oracle can pin-import the C path. `RsError` gains `StateBufferTooSmall { have, need }` for parity with `imp::Error`. drama_llama's `moeflux` feature now propagates `moeflux/riir-port`. drama_llama lib `cargo check` clean against the Rust impl. drama_llama tests have 33 pre-existing E0034/E0107/E0277/E0282/E0599 errors (verified pre-existing via stash-and-check); those are 5b territory. |
| 5d-1 | GPU LM head | 2026-04-27 (moeflux `68b4964`) | Profile-driven. The 2026-04-27 samply trace of the riir production path showed `lm_head_cpu` at 59% self-time per token. New `gpu_lm_head.rs` routes the 2048×248320 4-bit dequant matvec through the existing `dequant_matvec_4bit_v3` pipeline (one Metal dispatch, ~31040 threadgroups × 256 threads). Persistent shared-storage input + logits buffers on `RsCtx`, lazy-init via `ensure_linear_resources`. End-to-end diff tests stay green; blallama A3B essay perf jumps 1.24 → 4.90 tok/s cold, 1.25 → 4.58 tok/s warm (3.7-4×). |
| 5d-2 | GPU input rms_norm | 2026-04-27 (moeflux `9153d95`) | Replaces per-layer CPU `rms_norm_cpu` + 4 host↔GPU memcopies with 2-dispatch GPU prelude inside CMD1 of both `linear_attn_layer_forward` and `full_attn_layer_forward`. `buffers.input` is now read-only within the layer and serves both as rms_norm source and (later) as residual source for `encode_residual_add`. Reusable `gpu_norm::encode_rms_norm_bf16_into` + `RmsNormBf16Pipelines` so the helper is shareable for future chained-CMD3 work. Diff tests pass at the existing tight floors. **Perf-neutral** (cold 4.90 → 4.81, warm 4.58 → 4.54): we were already GPU-bound (cvwait 41.7%); moving CPU work that ran during GPU wait windows just turns CPU samples into wait samples. Code is correct, cleaner, GPU-first. |
| 5d-3 | fuse post-attn + shared-FFN cmdbuf | 2026-04-27 (moeflux `8655396`) | Collapses CMD2 + CMD3a + CMD3a-b commit+wait sequence into one cmdbuf. Replaces the CPU shared-FFN swiglu (was at `infer.m:2977 cpu_swiglu`) with GPU `swiglu_fused`. Net: 4 → 2 commit+waits per layer (matches C `cmd_fused` shape minus the K-expert deferred which is still async). Kernel `swiglu_fused` is bit-exact per-PSO (slice 9a); drift against C's CPU swiglu stays well within `cosine ≥ 0.9999`. New helper `encode_swiglu_buf` for `&Buffer` (vs `MtlBuffer<f32>` in expert_forward.rs). Diff tests green at tight 4e-8 floors. **+3-5% perf** (cold 4.81 → 5.04, warm 4.54 → 4.67) — fewer than expected because per-sync overhead is ~0.1-0.3ms not 1ms. |
| 5d-4 | GPU-buf inputs to K-expert dispatch | 2026-04-27 (moeflux `a27388b`) | Eliminates the 3 × HIDDEN_DIM GPU↔host readbacks at the tail of `post_attention_tail` and the matching host→GPU memcpys inside `gpu_batched_experts_encode`. Production fast path passes `LayerForwardBuffers.{normed, h_mid, shared_out}` directly. New `gpu_batched_experts_encode_buf` + `_begin_buf`. Encoder helpers `encode_matvec_into` / `encode_swiglu_into_buf` now take `&BufferRef`. **+2-5%** (cold 5.04 → 5.16, warm 4.67 → 4.88) — modest because the eliminated memcopies overlapped GPU work. |
| 5d-5 | pread experts directly into shared-storage | 2026-04-27 (moeflux `de47fa3`) | The 5d-4 re-profile showed `memmove` STILL at 9.9% — the elephant was the `expert_data: Vec<u8>` of K × EXPERT_SIZE that production allocated and copied per layer (~7 MB / layer). New `MoeBuffers::data_slot_mut(slot)` lets production `pread` directly into shared-storage. New `gpu_batched_experts_encode_pre_staged` skips the K-blob memcpy. Slot reuse is sound: every dispatch is waited at the top of the next layer. The 5d-4 `_buf` stepping stones removed. **+8-10%** (cold 5.16 → 5.58, warm 4.88 → 5.38) — second-biggest win after 5d-1. |

## blallama perf log (A3B, M2 Max)

| Configuration | Cold (512 tok) | Warm (~138 tok) | Notes |
|---|---|---|---|
| C path (baseline) | 10.13 tok/s | 8.70 tok/s | reference; `moeflux` feature with `riir-port` off |
| riir before 5d-1 | 1.24 | 1.25 | CPU lm_head dominated |
| riir after 5d-1 | 4.90 | 4.58 | GPU lm_head; ~half the gap to C closed |
| riir after 5d-2 | 4.81 | 4.54 | GPU input rms_norm; perf-neutral but GPU-first refactor |
| riir after 5d-3 | 5.04 | 4.67 | post-attn + shared-FFN cmdbuf merge; +3-5% |
| riir after 5d-4 | 5.16 | 4.88 | GPU-buf K-expert inputs; +2-5% |
| riir after 5d-5 | 5.58 | 5.38 | pread direct into shared-storage; +8-10% |

Re-profile after 5d-1 (samply, 26s of generation):
- `__psynch_cvwait` 41.7% — *good*: GPU is the wait, not the work
- `pread` 30.6% — streaming-experts disk reads; same on C path
- `partial_sort` 9.6% — drama_llama sampling; same on C path
- `_platform_memmove` 9.3% — per-layer host↔GPU staging churn
- `__bzero` 3.6% — scratch allocator
- `sdpa_cpu` 0.9% — confirms full-attn SDPA is genuinely cheap
- `rms_norm_cpu` 0.1% — tiny

The remaining gap to C is dominantly host↔GPU staging cost, NOT
specific kernels left on CPU. Two natural next slices:

## Suggested next-session order

- **5d-6: async / parallel pread (BIG one)** — Mike's CPU 60 / GPU
  20 observation at 5d-4 (Activity Monitor) turned out to be the
  load-bearing architectural gap. **Reference: C path runs
  approximately 40% GPU use on A3B / 60% on A17B**, so even C
  isn't saturating GPU here — but riir at ~20% GPU is half of C's
  already-low utilization. C path (per `metal_infer/infer.m:23,
  39, 991-992`) runs the K-expert disk reads in **4 parallel
  pthreads** on a **double-buffered** set ("set A" feeds GPU
  compute while "set B" is being prefetched in the background).
  The pread cost (30% of CPU samples ≈ 70 ms / token at our
  current rate) is hidden under GPU compute. In riir today preads
  are sequential and on the critical path. Implementing this:
  - Add a second `data_b: [MtlBuffer<u8>; MAX_K]` set on
    `MoeBuffers` with a "current" index alternating per layer.
  - Spawn (or thread-pool) a worker that preads layer N+1's
    experts into the alternate set while layer N's K-expert
    dispatch is in flight on the active set. Need to fan the K
    reads across 4 pthreads — `std::thread::scope` or `rayon`.
  - Layer N+1's expert indices are known after CMD3a's gate-logits
    matvec completes (mid-layer). Question: does C overlap the
    pread with CMD3b (its OWN K-expert dispatch on set A) or only
    with cross-layer compute? Worth reading `infer.m` carefully
    before designing the riir version. Read `g_pread_pool` /
    `InferPreadTask` setup at `infer.m:154+` and the consumer
    sites at `pread` callouts.
  - Estimated ROI: the pread time becomes free (overlapped under
    GPU compute), so wall-clock drops by ~30%. **5.4 → ~7-8 tok/s**
    expected, possibly more if it also lets the GPU saturate.
  - Diff coverage: existing `eval_token` end-to-end diff still
    holds (deterministic dispatch order across both backends).
    Add a regression test for the slot-reuse race window.
- **GPU full-attn fast path** — `gpu_attn_fuse` (kv_len ≥ 32) was
  cut from slice 4d's scope. Today every full-attn layer's SDPA
  runs on CPU (10 layers / token at growing kv_len). Profile shows
  `sdpa_cpu` at only 0.9% even at 100-token kv_len, so the SDPA
  arithmetic isn't expensive — but the surrounding work IS:
  per-head Q/K rms_norm + RoPE + KV append all run host-side, with
  matching readbacks of `q_proj_out` + `k_out` + `v_out`. Porting
  to GPU eliminates these per-layer host transfers. Bigger surgery
  — needs GPU KV-cache mirror buffers + the 3 `attn_*_batched`
  kernels wired up. Probably second-biggest remaining lever.
- **Chained CMD3 → next layer normed (the original 4f-perf)** —
  C path infer.m:5668-5764: CMD3 emits `moe_combine_residual` +
  `rms_norm_sum_sq` + `rms_norm_apply_bf16` so the next layer's
  input is GPU-resident pre-normalized. In riir today
  `complete_deferred_experts_into` does a GPU→host→GPU dance to
  hand off between layers; chaining makes it a no-op. Slice 5d-2
  already structurally enabled this (input rms_norm reads
  buffers.input directly). Estimated ~5%.
- **9f (LZ4 + 2-bit + expert caches)** — pread is 30.6% of CPU.
  An LRU expert cache would lift absolute throughput on both
  sides. Probably do this as a Phase 6/7 unification.

Phase 4 numerical correctness is **fully done** as of 2026-04-27.
Slice 4f-1 through 4f-6 (commits `fd63c0a` → `ceaa3ba`) wired the
public eval API; slice 4g (`037f74c`) added wire-compatible state
snapshots. Full diff oracle suite **32/32 green in 345s**.

Then Phase 6 (cutover), Phase 7 (post-cutover; see list below).

## Phase 7 follow-ups (post-cutover)

These are tracked here so they don't get lost. Once the C is gone
and we're the only consumer, we can break wire compat freely.

- **Typed `memory_seq_rm`** — return `Result<(),
  CannotTruncateLinear>` instead of silently resetting linear-attn
  layers to empty. Bisect's silent-truncate finding made this the
  motivating Phase 7 item.
- **Ctx-owned `g_deferred` + `layer_cache`** — already structurally
  fixed in the Rust port (`RsCtx::deferred` field, per-Ctx
  `layer_caches`). Phase 7 deletes the C-side equivalents and
  removes the cross-Ctx hang reproducer.
- **Runtime variant dispatch** — replace the `cfg(feature = "model-
  ...")` `VARIANT` const with a runtime config (drama_llama
  `project_moeflux_runtime_variant.md`). Touches every place that
  reads `VARIANT`.
- **Snapshot format simplification** (added 2026-04-27 after Mike
  flagged that the C-driven format has redundant fields):
  - Drop `linear_conv_bytes` / `linear_ssm_bytes` from the header
    (derivable from the other shape constants; only present
    because C lacks const-fn arithmetic in headers). Collapse to
    magic + version + shape-hash u64.
  - Replace per-full-attn-layer `i32 len` prefix with a single
    `u32 len` at the top — moeflux is single-stream so all KV
    caches share the same length per snapshot. Body becomes flat.
  - bf16 K/V — halves the KV portion. Already what's stored on
    the GPU side per slice 9; currently we round-trip through f32
    for the host snapshot. f32 on the host is wasted bytes.
- **Expanded coverage** — 9f (LZ4 + 2-bit + expert caches) is the
  next code-path to bring under diff. Performance + coverage work,
  not a numerical-correctness slice; deferable until a benchmark
  shows it's load-bearing.

**Cross-Ctx pollution as a Phase 7 reproducer:** the diff oracle suite
is now a useful (intermittent) reproducer for the cross-Ctx state-
pollution bug class. First run on 2026-04-27 hung in
`mf_layer_forward_dump → [_MTLCommandBuffer waitUntilCompleted] →
pthread_cond_wait` after ~25 min serial test execution; second run
finished cleanly in 176.6s. Classic non-determinism — depends on OS
page reclamation timing, GPU scheduling, allocator state. The 4c
`mf_free_model` defensive `layer_cache_built=0` reset covers the
common case but doesn't address `g_deferred` cross-Ctx fully. Phase
7 lifetime-binds both into Ctx-owned state, and the bug class becomes
uncompilable. Until then, "test hangs once in N runs" is the expected
shape; the suite is not deterministically broken.

Bisect's silent-truncate bug for partial linear-attn truncation is
NOT being fixed during the port (per the bug-fix policy above);
the Rust port replicates the C reset-to-empty semantic and the
typed `Result<(), CannotTruncateLinear>` lands as a Phase 7
post-cutover slice.

9f (LZ4 + 2-bit + expert caches) — performance + coverage work, not
a numerical-correctness slice. Deferable to Phase 7 unless a
benchmark on real prompts shows the basic path tops out below the
17.6 tok/s grammar-path target.

## Empirical finding: GPU kernels are bit-exact per-PSO

Through 2026-04-27, every GPU kernel landed in slices 9a / 9b / 9e
(`dequant_matvec_4bit_v3`, `swiglu_fused`, `moe_combine_residual`,
`rms_norm_sum_sq`, `rms_norm_apply_bf16`) is **bit-exact** between
two `MTLComputePipelineState` instances built from the same source
in the same process on the same device. This includes kernels using:

- SIMD-group operations (`simd_sum`)
- Threadgroup-shared memory writes/reads behind
  `threadgroup_barrier(mem_threadgroup)`
- Multi-buffer K-expert weighted accumulation
  (`moe_combine_residual`'s per-thread `Σ_k weights[k] *
  expert_out_k[tid]`)

The cosine/Jaccard tolerance regime that the strategy doc reserved
for "Metal kernels" has not engaged anywhere. It applies to:

- Kernels using true atomic ops (`atomic_*` — moeflux's kernels
  don't use these)
- Different shader sources (e.g. `fast_math` toggle producing
  divergent codegen — not the case here)
- Cross-device or cross-driver-version comparisons (out of scope)

Diff tests still set `cosine ≥ 0.9999` / `rel ≤ 1e-3` floors as
defensible placeholders. Tightening to bit-exact is a future
defensive option — would catch sub-1e-3 porting drift but bets on
Metal's per-PSO determinism being stable across driver updates.
Current floors trivially pass; raising them is a Phase 7 polish.

## Tolerance regimes (set by RoPE slice)

Three diff signals, picked per kernel based on what compiler choices
will and won't preserve:

- **Bit-exact** (per-element `to_bits` equality): only for kernels
  that are pure integer arithmetic + sequential f32 reduction +
  bf16-as-shift. Embedding and CPU RMSNorm sit here.
- **ULP-bounded** (per-element `ulp_diff` ≤ N, N small): kernels
  involving trig or other libm-precision calls. Catches porting
  bugs (which produce thousands of ULPs / NaN / sign flips) without
  chasing two different compiler-choice artifacts:
  1. Apple clang `-O3` auto-vectorizes scalar `cosf` / `sinf` via
     Apple's libm vector variants. Rust extern-`"C"` calls don't.
  2. Apple clang `-ffp-contract=on` (default) fuses `a*b ± c*d`
     into FMA instructions. Rust plain `*` / `-` don't.
  Even with extern libm bindings on the Rust side, those two
  compiler choices produce ≤ ~30 ULPs of drift on RoPE — bounded
  per call, not growing unboundedly with position. RoPE uses
  `MAX_ULP_DRIFT = 128`; this is the working budget for trig
  kernels.
- **Cosine / Jaccard floors** (vector-level similarity): for
  Metal kernels and full-pipeline checkpoints, where MoE atomic-op
  nondeterminism stacks. Helpers + thresholds already in
  `diff_oracle.rs`. Reserved for layer-boundary checkpoints, not
  per-element kernel diffs.

The `mul_add` retrofit on the rotation step would shrink RoPE's
drift (matches clang's likely FMA pattern) but doesn't change the
methodology; left as a future micro-optimization if any consumer
benefits from tighter agreement.

**Empirical confirmation (slice 6, LM head)**: matching clang's FMA
contraction on the Rust side via `mul_add` did move the matvec from
"1.0 cosine + 3.5e-7 relative drift" to fully bit-exact across
248320 vocabulary logits. So the FMA hypothesis isn't speculation —
it's the load-bearing reason a literal port of `acc += a*b + c`
patterns drifts on AArch64 release builds. Worth applying
proactively in future kernels where the inner loop has the
`acc + T*x` shape (linear attention will have plenty of these).

**Diff-oracle hook pattern**: each kernel gets a `mf_<kernel>` accessor
in `moeflux.h` that exposes the static C primitive. `RsCtx::open`
populates only the state the landed kernels need — `WeightFile`
today, `MetalBackend` + per-layer state as needed. Tests in
`crates/moeflux/tests/diff_oracle.rs`.

**RsCtx incremental init**: `open` no longer panics — it loads the
`WeightFile`. Methods that need unported kernels still `todo!()`
with a phase tag in the panic message. Each kernel landing flips
0..N methods to real impls.

## What the bisect tests become

- `consecutive_eval_prompt.rs` (in moeflux): keep as-is during
  port. Two of its four sub-tests fail today; they're expected to
  pass after Phase 7's typed `memory_seq_rm` and multi-Ctx work.
- `moeflux_session_pollution.rs` (in drama_llama): currently
  passes; serves as a regression guard the Rust port must keep
  passing throughout.

## Why this fixes the bisect findings

- **Cross-Ctx NaN bug**: `g_deferred` is a file-scope global with a
  raw pointer that outlives its referent. In Rust, that becomes a
  field on `Ctx` (or part of an MoE dispatcher owned by the Ctx).
  Lifetime-checked. The class of bug doesn't compile.
- **Partial-truncate divergence**: in Rust, `memory_seq_rm` returns
  `Result<(), CannotTruncateLinear>`. Callers must explicitly
  handle the case where linear-attn state can't be unwound — the
  silent-lossy behavior of the C API becomes a typed error.

## Future-arch openings (informal — not a plan)

Captured 2026-04-27 during 4d. Mike's decided multi-model is a
post-RIIR effort, target architecture #2 is **Cogito-V2 671B
(DeepSeek-V3 base)** — needed for the Council's emergency
deliberation path. Cogito-V2-70B (Llama base) is the easier port
but distillation tax kills it for the high-stakes reasoning use
case; not worth doing as a stepping stone.

Two small choices to make during 4f's `eval_prompt` integration that
keep the DeepSeek port from being uglier than it needs to be — both
land at no extra cost during the work that's happening anyway:

- **`LayerWeightCache` → `LayerKind`-tagged enum.** Today the struct
  has `linear_attn.*` and `self_attn.*` tensor slots side-by-side,
  unused half left `None`. Works for two layer types from the same
  family. For MLA's `kv_a_proj_with_mqa` / `kv_b_proj` /
  `q_a_proj` / `q_b_proj` (plus the rotary-vs-nope split), the
  "every-tensor-slot-on-every-layer" shape gets ugly. Same pattern
  as `LayerState::{LinearAttn, FullAttn}` — turn it into
  `LayerWeightCache::{LinearAttn(LinearAttnW), FullAttn(FullAttnW)}`
  ahead of needing the third `Mla(MlaW)` variant.
- **Layer-kind dispatch from `Variant`, not `(layer_idx + 1) %
  full_attn_interval == 0`.** The modulo predicate is qwen3-family-
  specific; DeepSeek-V3's first 3 layers are dense FFN, the rest are
  MoE+MLA, no modulo describes that. When 4f writes the per-layer
  loop, look the kind up via `VARIANT.layer_kind(layer_idx)` (or
  similar). Trivial today; load-bearing once a second arch lands.

Other DeepSeek-specific things that *do* need real work later
(estimate 2–4 weeks for a focused port):

- New attention pipeline (MLA: KV down/up projections, rotary on a
  partial head split, materialized K/V matmul + standard SDPA).
- New router (group-limited top-K of K experts within top-K_groups
  groups; sigmoid not softmax for gate scoring).
- New shaders for MLA (different head_dim layout from qwen3's 256).
- Dense FFN dispatcher for the first N layers (no MoE).

Kernels that survive verbatim: RMSNorm (CPU + GPU), `dequant_matvec`
flavors, SwiGLU, the MoE base infrastructure. The `post_attention_tail`
in `linear_attn_forward.rs` is largely reusable too — DeepSeek's
post-attn shape is similar (residual + post-attn-norm + MoE).
