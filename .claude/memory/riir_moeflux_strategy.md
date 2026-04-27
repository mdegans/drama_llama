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
| 4d | full-attn fused_layer_forward | — | same | Adds RoPE + KV append + SDPA on top of the linear-attn shape. |
| 4e | deferred-expert state machine (old 9d) | — | end-to-end via 4f | `g_deferred` → field on `RsCtx`. FIXME for the cross-Ctx NaN bug. |
| 4f | mf_step_internal + eval_prompt/eval_token | — | end-to-end logits cosine/Jaccard | Wires the per-layer forward into the public eval API. |
| 4g | state_save / state_load | — | byte-identical snapshots | Last Phase 4 slice; the state-snapshot binary format. |

## Suggested next-session order

Phase 3 numerical-correctness work is essentially done as of
2026-04-27. **4/6 MoE-dispatch sub-slices landed** (9a / 9b / 9c /
9e), all bit/byte-exact. 18/18 diff oracle tests green. Remaining:

- **9d (deferred experts state machine)** — `g_deferred` →
  `Ctx`-owned struct. The cross-Ctx NaN bug source; faithful port +
  FIXME for the Phase 7 typed-`memory_seq_rm` fix. *Best landed
  alongside Phase 4* — the diff oracle pattern doesn't naturally
  exercise async sequencing, so the testing only really makes sense
  when there's an end-to-end forward pass to integrate it into.
- **9f (LZ4 + 2-bit + expert caches)** — performance + coverage
  work, not a numerical-correctness slice. Per-blob LZ4
  decompression, the LRU Metal-buffer cache, the malloc cache, the
  2-bit quantization pipeline. *Deferable to Phase 7* unless a
  benchmark on real prompts shows the basic path tops out below the
  17.6 tok/s grammar-path target.

Implication: **Phase 3 can effectively be declared closed** with 4
sub-slices instead of 6. Phase 4 (top-level forward-pass
orchestration: `eval_prompt`, `eval_token`, `memory_*`, state
save/load) opens next session, with 9d folded in as one of the
Phase 4 integration tasks and 9f deferred until a real benchmark
motivates it.

If a future session prefers to land 9d in isolation first (before
Phase 4), the smallest version is: add a `DeferredState` struct on
`RsCtx` mirroring the C `g_deferred` fields; add a "begin-deferred"
variant of `gpu_batched_experts_forward` that commits without
waiting; add a "complete-deferred" call that waits + reads back.
Diff: a paired test that calls begin then complete on both sides
and compares the final hidden state. The hard part is the diff
shape, not the code.

Bisect's silent-truncate bug for partial linear-attn truncation is
NOT being fixed during the port (per the bug-fix policy above);
the Rust port replicates the C reset-to-empty semantic and the
typed `Result<(), CannotTruncateLinear>` lands as a Phase 7
post-cutover slice.

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
