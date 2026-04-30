# Cogito-V2 671B landing — first forward green

Captured 2026-04-30 (continuation session). The CPU MLA + MoE
forward path landed end-to-end and produced coherent English on
the first run. **Status: Phase H of the original session-plan
(`~/.claude/plans/squishy-wibbling-milner.md`) reached green;
GPU MLA is the next slice.**

## First-run result

```text
prompt:  "Hello. How are you?"   (11 input tokens)
output:  "I'm doing well, thanks! How"  (8 tokens, max_tokens cap)
stop:    max_tokens
```

Per-token /probe telemetry showed sensible distributions: token 3
(" well") at p=0.90, entropy 0.51 — well-calibrated. No NaN, no
word-salad, no thrashing.

Throughput: ~12 s / token warm on M-series CPU at 8 P-cores,
800% rayon utilization. Acceptable for debug; GPU MLA is a
follow-up. Mike's call: "won't tell us if that holds over a long
generation but it's a very good sign."

## What landed this session (commits in flight)

**moeflux** (`~/Projects/moeflux`, branch `main`, two commits):
1. `LayerState::Mla` plumbing — variant added to enum, all match
   arms across `state.rs` / `state_snapshot.rs` / `mod.rs` updated.
   Snapshot v1 wire format doesn't encode MLA caches (typed
   `MlaUnsupported` error; v2 wire format is post-cutover work).
2. CPU forward path —
   - `cpu_matvec.rs` (new): fused 4-bit dequant + matvec primitive
     `dequant_matvec_4bit_cpu` (rayon-parallel over output rows),
     bytes-input variant for packed-expert blobs, BF16 matvec for
     the router gate. MLX 4-bit format (group-of-64 BF16 scale +
     bias).
   - `mla_attn_cpu.rs`: `mla_attn_layer_forward_cpu` impl. Naive
     form — per-token decompresses `kv_b_proj @ latent[j]` for
     every cached j. Folded form (precompute
     `q_nope @ kv_b_proj_K`, `kv_b_proj_V @ V_combine`) is the
     follow-up perf slice.
   - `mlp_cpu.rs` (new): `dense_mlp_swiglu_cpu` (layers 0-2,
     intermediate=18432) + `shared_expert_swiglu_cpu`
     (intermediate=2048).
   - `moe_cpu.rs` (new): `deepseek_moe_cpu` — BF16 router gate +
     `noaux_tc_router_cpu` + per-expert SwiGLU on packed blobs from
     `root/packed_experts/layer_NN.bin` + unconditional
     shared-expert add (no gate, contrasts with Qwen).
   - `mod.rs`: `ensure_mla_resources` (slim init, skips
     `LayerWeightCache::build_all` which requires GQA-only
     `q_proj`/`k_proj`/`v_proj` names) + `step_internal_mla_cpu`
     (full host pipeline) + dispatch branch in `step_internal`.
   - `tests/cogito_v2_smoke.rs` (new): end-to-end `Ctx::open` +
     `eval_token` smoke against the real 671B weights.

**drama_llama** (this repo, branch `v0.8.0`, one commit):
- This memo replacing the previous "continuation state" with the
  green outcome.

## Test coverage added

All `#[ignore]` smoke tests against the on-disk 671B; non-ignored
unit tests are pure-arithmetic and run in <1 s:

- `cpu_matvec::tests` (5 unit + 1 ignored): all-ones, zero-input,
  bias-only, slice mismatch, alignment check, BF16 identity.
  Ignored: `q_a_proj_smoke_against_real_weights`.
- `mla_attn_cpu::tests`: ignored `mla_layer0_pos0_smoke`.
- `mlp_cpu::tests`: ignored `dense_mlp_layer0_smoke`.
- `moe_cpu::tests`: ignored `moe_layer3_smoke`.
- `tests/cogito_v2_smoke.rs`: ignored
  `cogito_v2_eval_token_smoke` (full single-token e2e via the
  public `Ctx` API; ~12 s).

## On-disk artifacts

`/Volumes/Temp Backup/models/blallama/cogito-v2-671b/`:
- `mlx/` — tokenizer.json, chat_template.jinja, config.json
- `artifacts/model_weights.{bin,json}` — 1831 tensors, ~10 GB
- `root/packed_experts/layer_NN.bin` — 58 files (3..60),
  ~340 GB total

352 GB on disk; verified end-to-end this session.

## Known follow-ups (next sessions)

In suggested order:

### 1. GPU MLA (next session — Mike already greenlit)

The CPU baseline is the diff oracle. Plan: write a Metal kernel for
the MLA forward (per-head SDPA over decompressed K/V), reuse the
existing `dequant_matvec_4bit_v3` pipeline for the projections.
Validate against the CPU path token-by-token (the `/probe`
telemetry gives per-token logit snapshots — bit-equal isn't the
target since GPU softmax / reduction order will diverge, but
top-1 and top-3 set equality at low temperature should hold).

### 2. Folded MLA form (perf slice)

Naive decompresses `kv_b_proj @ latent[j]` per cached position;
folded form precomputes `q' = q_nope @ kv_b_proj_K_per_head`
(shape `[num_heads, kv_lora_rank]`) once per token, then per
cached j just does `q' · latent[j]`. ~60× speedup at long context.
For first run we deliberately landed naive for clarity. Land
this either before or alongside the GPU port.

### 3. YaRN inv_freq caching

Currently `compute_yarn_inv_freq` runs per-token in
`step_internal_mla_cpu`. Trivial to cache on `RsCtx` — microseconds
on the wall clock but reduces per-token allocations.

### 4. Long-generation stability check

First run was 8 tokens. Mike flagged: "won't tell us if that holds
over a long generation." Run a 200+ token generation, watch for
degradation patterns (repetition, thrashing, magnitude drift on
the residual stream as `cache_len` grows). Fold-vs-naive results
should match if the math is right.

### 5. Tokenizer / chat-template verification

Worked on first try — but the special-token IDs in the variant
config (`bos=0`, `eos=1`) were taken from the architecture audit,
not empirically verified against the `tokenizer.json`. Sanity-check
that `</s>` decodes back to id=1 and BPE round-trips a known
prompt.

### 6. Snapshot v2 wire format

The existing `state_snapshot` v1 doesn't encode MLA's compressed
latent + rope-K caches. Add a v2 format with MLA support. Not
blocking blallama (which doesn't use snapshots for the main path)
but will be needed once the Council reactor uses Cogito-V2.

### 7. Publish moeflux 0.1.0-pre.3

drama_llama is path-pointed at local moeflux while the work is in
flight. After all in-flight changes land + tests stay green,
publish moeflux and flip drama_llama back to a version dep.
