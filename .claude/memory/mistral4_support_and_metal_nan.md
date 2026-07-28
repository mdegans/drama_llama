# Mistral Small 4 support, and the Metal NaN blocker

**Status (2026-07-28, Opus 5):** template/dialect/registry support is
**landed and green** (`53a07d3`, `2c4a538`). Generation on Metal is
**blocked** by an upstream NaN bug that is not ours and not the
template's. Mike took the llama.cpp update in a parallel session.

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

Argument interiors stay `tojson`-**compact**, matching stock, because
the model's unforced habit is **not yet measured** — the probe is
written (`probe_unforced_habit::mistral4_unforced_call_spelling`) but
cannot run until decode works. If it comes back uniformly spaced, the
fix is the cogito one-liner: `| tojson` -> `| json_dumps`.

**Vision needs no template or dialect work.** Images never reach the
template layer: `Session` renders a per-call random sentinel, splits
the render on it, and hands mtmd the image chunks out-of-band, so
pixtral's `[IMG_END]` framing comes from mtmd. The mmproj auto-loads
from the `<model>.mmproj.gguf` sidecar. Untested end-to-end (no vision
test exists anywhere in `tests/` to model one on — a genuine gap).

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

**Residue:** something else in the deepseek2 graph mistral4 inherits —
MLA attention math, the fused `gate_up` split, MoE routing
(`noaux_tc` / `expert_weights_norm`) — or an op whose Metal path
happens to change at 32 that neither grep found. The next step is
`llama-eval-callback` (built in a scratchpad clone, not Mike's
checkout) to name the first tensor that goes non-finite; that is the
one thing that turns this into a filable upstream issue.

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
