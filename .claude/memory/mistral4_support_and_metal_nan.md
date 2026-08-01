# Mistral Small 4 support, and the Metal f16-overflow NaN

**Status (2026-07-28, Opus 5): WORKING.** Support landed (`53a07d3`,
`2c4a538`, `74cf8da`) and `just test session_mistral4` is **7/7 on the
real model** — tool calls, thinking, prefix cache, and the repo's first
end-to-end rung-2 witness. The Metal all-NaN decode is root-caused to
an upstream f16-overflow bug (below) and worked around with
`n_ubatch = 31`.

Measured on device: the model's unforced call spelling is **`Spaced`**
(`json.dumps` style), so the owned template renders arguments with
`json_dumps`; and it never volunteers a `[CALL_ID]`, confirming
`CallIdPosition::None`.

**Open — one unreproduced tool-call loop.** A forced-tool-choice turn
emitted the same call 26x until the budget truncated it mid-string.
**Replayed unchanged and passed**, so it is stochastic and was never
diagnosed — no fix was applied, and nothing here is established.

Correcting a wrong first read (Mike caught it): I called the default
sampler "very wide". It is not. `SamplerConfig::default()` is TopK 1024
-> locally-typical p=0.5, and our own code documents the TopK as "a
pre-cut, not a sampler ... behaviorally invisible" — so the effective
sampler is locally-typical alone, which is **tight**. That inverts the
causal story: cool/greedy decoding is *more* loop-prone, so if sampling
contributes it is by being too cold, and the remedy is temperature, not
tightening.

Better-supported candidate, unverified: `constrained_regions = true` is
on and is documented as "what breaks small-model loops inside always-on
tool-call grammars" — but `ignored_categories = [English, Json,
Punctuation]` means those tokens are never penalized, and a
`[TOOL_CALLS]name[ARGS]{...}` sequence is almost entirely English
identifiers plus JSON punctuation. The anti-loop mechanism is enabled
with nearly nothing to act on.

Contributing dialect property either way: `call_separator` is empty and
parallel calls are grammar-legal, so another `[TOOL_CALLS]` is always a
legal continuation and only EOS ends the turn — no structural "you may
stop now" signal, unlike dialects with a closing tag.

The e2e tests pin **no seed**, so run-to-run divergence is expected by
construction; see [[qwen_flakes_correlate_with_heat]] for the
pin-the-seed-and-replay discipline this wants before anyone tunes
anything. The GGUF advertises no `general.sampling.*`, so a tuned
sidecar is still worth having on its own merits.

Read this before re-diagnosing a NaN on this model, and before
re-testing flash attention or the quantization — both are already
ruled out, with evidence, below.

## What landed

Model: `models/Mistral-Small-4-119B-2603-UD-IQ3_S.gguf` (+ pixtral
mmproj sidecar). Arch `mistral4`, 36 blocks, MLA (`key_length=320`,
`value_length=256`, `kv_lora_rank=256`), MoE 128 experts / 4 active +
1 shared, 1M ctx, YaRN x128. bos `<s>`, eos `</s>`, **no eot** — so
`</s>` alone is EOG (do not rebuild it, see [[eog_is_not_eos_plus_eot]]).

**The dialect needed no code.** The call format is
`[TOOL_CALLS]name[ARGS]{…}` — function name *outside* the JSON, no
wrapper object, one `[TOOL_CALLS]` per call — and the differential
analyzer derives it whole as `Family::TagWithJson` (`per_call_start =
"[TOOL_CALLS]"`, `function.name_suffix = "[ARGS]"`, empty close, empty
separator). No new `Family` variant, no `parse.rs` arm, no `emit.rs`
arm, no `sniff_hand_built` entry. Every marker is a single special
token in the vocab (`[TOOL_CALLS]`=9, `[ARGS]`=32, `[CALL_ID]`=33,
`[THINK]`=34, `[/THINK]`=35, `[IMG]`=10, …). This is the
"code accepts families, data accepts models" principle paying out —
see [[plan_template_ownership]].

**One `PATCHES` entry** (`dialect/analyzer.rs`): the reasoning probes
see the `[THINK]` markers but cannot see *which side owns them*, and
the derived default `InlineThink` would route thoughts into `content`,
where our template renders none. Keyed on the analyzed `[THINK]`
marker plus a `reasoning_content` source guard — deliberately narrow,
because the Qwen templates also mention `reasoning_content` and are
correctly `InlineThink`.

**Owned template** `templates/mistral4-cache-stable.jinja`, five
changes vs stock, each pinned in `tests/dialect_roundtrip.rs`:

1. `</s>` emitted **per message**. Stock emits it unconditionally and
   has *no `add_generation_prompt` branch at all*, so it cannot render
   an open assistant turn and the generation prompt is never a byte
   prefix of the follow-up.
2. Reasoning round-trips as `[THINK]…[/THINK]` from
   `reasoning_content`, gated by `preserve_thinking` when aged. Stock
   accepts a thought **only** as a `thinking`-typed content chunk, so a
   `ReasoningReingest::Field` transcript trips stock's own
   `raise_exception` (pinned: `mistral4_stock_cannot_render_field_reasoning`).
3. Pre-call prose in emission order (`content_pre` / `content_post`).
4. **The Unsloth date preamble is deleted.** Its default system message
   interpolated today's *and* yesterday's date into the prompt
   **prefix** — a session spanning midnight lost its entire cache.
   Behaviour change only for callers that send no system message and
   relied on the vendor Le Chat default; Agora always sends one, so
   stock discarded the default there anyway.
5. Role-alternation `raise_exception` dropped; mid-conversation system
   turns render in `[SYSTEM_PROMPT]` framing rather than raising.

6. Argument interiors render with `| json_dumps`, not stock's
   `| tojson`. **Measured** once decode worked
   (`probe_unforced_habit::mistral4_unforced_call_spelling`, greedy, no
   grammar): the unforced emission is
   `[TOOL_CALLS]create_post[ARGS]{"community": "debate", "title": ...}`
   — uniform `": "` and `", "`, including between array elements. So
   the habit is `JsonSpacing::Spaced` and stock's compact spelling would
   have forced the model off it to stay round-trip stable (#85's
   lesson). The analyzer measures the swap as `Spaced` and the grammar
   prelude plus `render_reference` follow. Same one-liner cogito took.
   The same probe answered the `[CALL_ID]` question: the model never
   volunteers one, so `CallIdPosition::None` is correct and the
   re-render has nothing it cannot reproduce.

**Vision needs no template or dialect work.** Images never reach the
template layer: `Session` renders a per-call random sentinel, splits
the render on it, and hands mtmd the image chunks out-of-band, so
pixtral's `[IMG_END]` framing comes from mtmd. The mmproj auto-loads
from the `<model>.mmproj.gguf` sidecar. **Untested on this model**, but
vision tests DO exist — I claimed otherwise and was wrong (Mike caught
it): five of them in `src/llama_cpp/mtmd.rs`'s `#[cfg(test)]` module
(`tokenize_chunk_structure`, `prefill_image_smoke`,
`segment_tokenize_differential`, `eval_loop_differential_vs_helper`,
`mrope_kv_semantics_probe`). A subagent grepped `tests/` only and I
generalized its miss into "none anywhere" — vision coverage lives in
`src/`, not `tests/`.

They are **single-model by construction**: `local_vision_paths()`
(`mtmd.rs:954`) resolves `models/model.gguf`, the symlink, so they only
ever cover whatever it points at (Qwen3.6 today). Making them
multi-model wants the `session_mistral4.rs` pattern — env var →
conventional path → loud skip, never substitute. Mike is taking that in
a parallel coverage session.

**Cannot run in CI**: Mistral Small 4 is 44 GB (IQ3_S) / 74 GB (Q4) and
will not fit the self-hosted runner's 3090 (24 GB). Its vision coverage
is M2-Max-only. See [[plan_ci_self_hosted_runner]].

`tests/session_mistral4.rs` is the **first test in the repo that
observes rung 2 end-to-end on a real model**: it deliberately installs
no sidecar and asserts the baked dialect, closing the gap #88 phase 1
left open. It hard-asserts that no `*.template.jinja` sidecar exists,
so it can never quietly downgrade to a rung-1 test.

## The blocker: NaN logits on Metal above 32 tokens

**Symptom.** `mistral4` on Metal returns an entirely NaN vocabulary for
any prefill of >=32 tokens. Below that, correct logits. Deterministic,
bit-identical across separate processes.

| prefill tokens | Metal | CPU-only |
|---|---|---|
| 15 | clean (max 21.29) | clean |
| 25 | clean (max 25.36) | clean (max 25.14) |
| 35 | **131072/131072 NaN** | clean (max 21.26) |
| 65 – 2055 | **all NaN** | clean (max 21.68) |

**Ruled out — do not re-test these:**

- **Flash attention.** FA-on and FA-off agree to two decimals below the
  threshold and fail identically above it. Also: Metal *does* have a
  `DK=320` FA kernel (`ggml-metal-device.m:1232`), so the obvious
  "unsupported head dim" story is wrong.
- **Flaky Metal state** ([[feedback_reboot_on_gpu_weirdness]] does not
  apply). Bit-identical across processes and rebuilds, with a hard
  boundary. Not weather.
- **Quantization, including IQ2_S.** Qwen3.6-A3B — MoE, works daily on
  this box — uses **IQ3_S** experts through the same kernel. And
  upstream `test-backend-ops -o MUL_MAT_ID -b MTL0` passes
  `type_a=iq2_s … m=512,n=32,k=256` against the CPU reference. (Note
  the backend is named `MTL0`, not `Metal`; `-b Metal` silently skips
  everything and prints a green OK.)
- **Our code.** Reproduces on a bare `[INST]…[/INST]` filler prompt via
  `Engine::predict_candidates` — no tools, no grammar, no `Session`.

- **The llama.cpp update b9754 -> b10156** (sys crate 0.8.2). Tested
  via a local `[patch.crates-io]`. Not a fix, and *conclusively* so:
  the sub-threshold logits are **bit-identical** across the two trees
  (`max=25.355751` both), so the DeepSeek V4 / fused-hyper-connection
  work that landed in `src/models/deepseek2.cpp` never touched
  mistral4's execution path. The Metal MoE code is byte-identical
  between the trees. (The bump itself is safe — the `unsafe impl Sync`
  preconditions in `Cargo.toml:30-48` were re-verified at b10156: still
  `const llama_model & model`, still no `mutable` members, still no
  `llama_opt_init` binding.)
- **`mul_mm_id` at this model's REAL geometry.** `test-backend-ops`
  patched with nine cases at `n_mats=128, n_used=4` and the actual
  tensor shapes (`ffn_gate_up_exps` = `(k=4096, m=4096, 128)` IQ2_S;
  `ffn_down_exps` = `(k=2048, m=4096, 128)` IQ3_S), `n` in
  {16, 32, 64}: **all OK vs the CPU reference**, and the logs confirm
  the `kernel_mul_mm_id_*` pipelines actually ran at n>=32. So the MoE
  matmul kernel is correct at the exact shapes/quants/expert-count this
  model uses. IQ4_XS at the same geometry also passes, which is what
  finally rules the **quantization** out at the real shapes and makes a
  4-bit re-download pointless.
- **Four Metal runtime knobs**, each A/B'd on the real model with the
  boundary unchanged: `GGML_OP_OFFLOAD_MIN_BATCH=1000000`,
  `GGML_METAL_FUSION_DISABLE`, `GGML_METAL_GRAPH_OPTIMIZE_DISABLE`,
  `GGML_METAL_CONCURRENCY_DISABLE`. Op offload, kernel fusion, graph
  optimization and concurrency are all exonerated.

**The threshold is EXACTLY 32**, measured at 1-token resolution: 31
tokens clean, 32 tokens all-NaN. That is a real signal, not an
approximation — but note there are **two** independent 32s in the Metal
backend and *both* are now eliminated as mechanisms:
`ne21_mm_id_min = 32` (`ggml-metal-ops.cpp:2382`, the
`mul_mv_id` -> `mul_mm_id` switch) and
`op_offload_min_batch_size = 32` (`ggml-metal-device.m:848`). Grep for
*all* the 32s before anchoring on one; anchoring on the first cost a
`test-backend-ops` build.

## ROOT CAUSE (found) — f16 overflow in Metal's `mul_mm_id`

`llama-eval-callback` (built in a scratchpad clone) named the first
non-finite tensor in the graph:

    ffn_moe_down-32 = MUL_MAT_ID(blk.32.ffn_down_exps.weight{2048,4096,128},
                                 ffn_moe_swiglu-32{2048,4,41}) = {4096,4,41}

Layers 0..31 are finite through the *identical* kernel at the same
token count; 32..35 are poisoned downstream. So it is **data**, not
shape or path — which is why the isolated op test (random values in
±1) passes while the model fails.

The data: **layer 32's activations are ~1000x every other layer's**
(`ffn_moe_gate-32` sums to -311493; other layers are in the hundreds).
The Metal MMA path carries its operands in **half**
(`simdgroup_half8x8`), and f16 tops out at 65504. Outliers overflow to
inf, and inf arithmetic yields NaN. The `mul_mv_id` path below 32
tokens carries the same values in f32 and is correct.

**Proved by construction:** at `n_ubatch = 31` there are zero NaNs and
that same `ffn_moe_down-32` comes out **-138943** — enormous but
finite. That is the value f16 cannot hold.

**Workaround, shipped:** `LlamaCppOptions::with_n_ubatch(31)` keeps
prefill on `mul_mv_id`. Costs prefill parallelism, nothing else;
`n_batch` stays large so prompts still submit in one call.
`tests/session_mistral4.rs` sets it (7/7 on device with it, all dead
without it). Drop it when upstream fixes the kernel.

### Fixed upstream — PR ggml-org/llama.cpp#26223 (2026-07-28)

Branch `fix/metal-mul-mm-id-f16-overflow` in the llama.cpp submodule
(pushed to the `mdegans/llama.cpp` fork). Two commits: a reproducer,
then the fix.

**Reproducer.** This class of bug was *untestable* upstream:
`init_mul_mat_id_tensors` inits uniform [-1, 1], so no case could drive
an operand out of f16 range. `test_mul_mat_id` gains an `amax`
parameter (default 1.0f = historical behaviour) scaling only the f32
activations. Six cases; `n=16` controls stay green, `n=32/64` failed.
Reproduces at *minimal* size (q8_0, 8 experts / 2 active, 512x256), so
it is not model- or geometry-specific.

**Fix.** Rescale src1 by a power of two on load, undo it on the f32
accumulator at the store. Exact, not approximate: the dot product is
linear so one tensor-wide factor commutes through accumulation, and a
power-of-two factor is exact in binary FP. When amax already fits the
factor is 1.0 and output is **bit-identical** — which is what makes it
defensible to perf-conscious maintainers.

**Perf matters here and nearly sank it.** The first version dispatched
the reduction as a *single threadgroup* scanning all of src1: **+38% to
+451%** on prefill. Rewritten two-stage (256 threadgroups → partials →
one folding pass) it is **+1.14% median overall**, +1.3–4% on prefill,
~0% on decode (the reduction is dispatched only on the mm path). Mike's
prompt — "upstream might accept broken code if it's faster" — is why
this got measured at all. **Always benchmark a kernel change here.**

**Review round 2 landed (2026-08-01).** ggerganov's inline suggestions
were applied plus his concurrency question — stack `map0` with
`amax_part` (wide dispatch hides map0's token-scaling work) instead of
with the 32-thread `amax` reduce. Measured before adopting: A/B on
op-level `test-backend-ops perf`, 16 interleaved order-counterbalanced
runs per side, per-case medians, mv-path cases as a no-change control
(±0.5%). All 18 mm-path cases favored the reorder: **mean −2.6%**,
growing with batch (−2.3% at n=512 → −3.5% at n=2048; the n>512 cases
were a *local* test-file edit, deliberately not committed — upstream's
perf lists stop at 512 and 56 extra cases ≈ +1 min per sweep).
Cold-chip and steady-state batches agree. Branch rebased on master
(798/798 post-rebase) and force-pushed 2026-08-01; commit message is
Mike's, with a generative-AI-disclosure Co-Authored-By. Awaiting
merge. Protocol lesson that earned the result: the naive back-to-back
perf comparison showed ±20% swings *in the control group* (cold-start
+ slot-order bias) — interleave and counterbalance or the numbers are
fiction. When the PR merges into a tagged release, bump llama-cpp-sys
and retire `with_n_ubatch(31)`.

Upstream issue **#25722** (open) is the same bug: mistral4, Metal,
empty output above ~300 tokens, FA on *and* off, generation degenerating
to a single control token — argmax over an all-NaN distribution. Their
"~300" is an un-bisected 32. **#20668** (closed, "repetitive/empty
output", same token-31 spam) is plausibly the same defect misattributed
to a corrupt GGUF.

Not covered by the PR: `kernel_mul_mm` (dense) has the identical
narrowing and should fail the same way; and the scale could be
per-output-column rather than per-tensor for better precision when one
token is the hot one.

### Trap: `[patch.crates-io]` is silently ignored if Cargo.lock disagrees

Cost a wrong conclusion reported to Mike. Patching drama_llama's
`Cargo.toml` to point at the local `llama-cpp-sys` does **nothing** if
`Cargo.lock` pins the old version — cargo prints
`warning: patch ... was not used in the crate graph` and builds the
*published* crate anyway. The earlier "does b10156 fix it?" test was
therefore the same 0.8.1 binary twice, and its headline evidence
("sub-threshold logits bit-identical between trees") was worthless —
identical code trivially gives identical output. The conclusion
survived only because the defective conversion is visibly present in
b10156.

**Always run `cargo update -p <crate>` after adding a path patch, and
check the build actually recompiles.** A 0.3s "build" is the tell.

### `llama-cpp-sys-3` 0.8.2 is NOT a drop-in bump

llama.cpp b10156 added a `text_len` field to `mtmd_input_text`, so
`src/llama_cpp/mtmd.rs` fails to compile with `E0063` until it is set
(`text_c.as_bytes().len()` — excludes the NUL, which is what upstream
wants since it reads `text_len` bytes rather than scanning). Fix is
written but uncommitted, pending the bump.

`llama_model_mistral4` is literally `llama_model_deepseek2` — same
hparam loader, same tensor loader, same graph, one-line
`build_arch_graph` override (`models.h:1275` at b10156). Note this does
**NOT** implicate Cogito/DeepSeek work here: that runs on the moeflux
engine, not llama.cpp (Mike, 2026-07-28).

Tools left in the scratchpad pattern, cheap to recreate: a GGUF
tensor-type dumper (expert quant types without loading weights) and a
`LlamaCppOptions` A/B harness (`cpu_only()` / `with_flash_attention`)
that reports NaN counts from `predict_candidates` — both of those
knobs are documented as diagnostics in `llama_cpp/options.rs` and this
is what they are for.

## `DecodeError::NonFinite` (landed, `2c4a538`)

NaN logits used to surface as `partial_cmp(…).unwrap()` panicking in
`Candidates::sort` — a decode failure that reads like a sampler bug.
Now caught at both decode exits (`prefill`, `step`) via
`logits_checked`, KV-dirty, with the pure predicate split out as
`first_non_finite` so it pins model-free.

**Open, and Mike's call — the typed error does not yet reach a
caller.** `predictor.rs` has three `.expect()` sites (`:350`, `:392`,
`:443`) that turn every `DecodeError` into a panic, and `Session` is
built on the predictor, so a NaN decode still **kills the process**
rather than failing one request. That matters for blallama/Agora,
which serve. Fixing it is an API-shape change on a public interface
with downstream consumers (Weave uses `Engine` directly):
  - (a) `Item = Result<Candidates, DecodeError>` — breaking;
  - (b) store the error, end iteration, expose `take_error()` —
    non-breaking, and the shape streaming iterators usually take.
(b) is the recommendation. Not done unilaterally: it is a design
change to the crate's core prediction API, not a bugfix.

## Related

- [[plan_template_ownership]] — the #88 arc this extends; rung 2, the
  probe-before-own discipline, and the canonical-bytes rule.
- [[llama_cpp_ffi_audit]] — same class of defect (an upstream failure
  surfacing as something that looks like our bug).
- [[eog_is_not_eos_plus_eot]] — why `</s>`-only EOG here is fine.
