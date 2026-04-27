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
| 7 | Post-cutover (separate PRs): typed `memory_seq_rm`, multi-Ctx, runtime variant dispatch, expanded coverage | 4–8h |

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
| RMSNorm (Metal) | — | — | Cosine/Jaccard tolerance — fast_math + Metal reduction order diverge. |
| linear-attn recurrence (8b) | — | — | The novel part of GatedDeltaNet: per-v-head `state *= g; kv_mem = sum(state * k); delta = (v - kv_mem) * beta_gate; state += outer(delta, k); out = sum(state * q)`. Has libm `expf`/`logf`/`softplus`/`sigmoid` for per-head decay/beta, plus a 4D-shaped state update with FMA contractions. Use `mul_add` proactively. C path has the documented partial-truncate semantic (bisect finding #3); port faithfully with `// FIXME(riir):` note pointing at the typed-Result fix in Phase 7. |
| MoE dispatch | — | — | GPU-heavy expert-forward orchestration. Probably last; depends on Metal infrastructure. |

## Suggested next-session order

linear attention → MoE dispatch. Linear attention is the hard one (bisect's
silent-truncate bug gets fixed in passing via typed
`Result<(), CannotTruncateLinear>`); MoE dispatch is the GPU-orchestration
finish. After both, Phase 3 closes and Phase 4 (top-level forward pass)
opens.

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
