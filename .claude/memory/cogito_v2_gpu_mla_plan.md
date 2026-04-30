# Cogito-V2 — GPU MLA forward, plan-of-record

Companion to `cogito_v2_landing_state.md` (CPU forward green).
Drafted at the close of the CPU-landing session while context was
loaded — concrete enough that next-session Claude can start coding
without re-exploring. Mike's call: "next session we'll just skip
straight to the GPU version."

## Goal & success criteria

Drive Cogito-V2 671B inference through the existing Metal pipeline
so MLA layers behave like GQA layers do today: GPU-side projections,
GPU-side SDPA, GPU-side cache append. lm_head stays GPU. MLP / MoE
stay on the existing GPU expert-forward path.

**Success = same coherent output as the CPU path at order-of-
magnitude faster wall-time.** Concrete bars:

1. blallama returns coherent English to "Hello. How are you?"
   (same prompt as CPU smoke), token-1 and top-3 set equal to the
   CPU path at low temperature (T=0 ideally).
2. Throughput improves substantially over CPU's ~12 s / token.
   Order-of-magnitude estimate: SSD-bound at ~1 tok/s warm per
   the original pre-RIIR memo; we'll know after first run.
3. 200+ token generation stays coherent (the open question Mike
   flagged at CPU landing).

## CPU baseline as diff oracle

`step_internal_mla_cpu` (mod.rs, this session) is the source of
truth. Keep it callable — either as a public alternative entry
point or behind a `MOEFLUX_FORCE_CPU_MLA=1` env override — so the
GPU path can diff against it on the same prompt.

Per-layer dump points worth preserving for bisect:

- `linear_buffers.input` after embed (host-side already on CPU
  path; needs a GPU readback hook).
- Per-layer post-MLA hidden (residual contribution before add).
- Per-layer post-MoE / post-MLP hidden.
- Final `hidden_normed` before lm_head.
- Final logits.

The /probe SSE stream already gives per-token logit snapshots
(top-K, entropy) — those are the natural validation lever for
top-1 / top-3 agreement across the two paths.

## Architectural decisions already made

1. **Folded MLA form on GPU, not naive.** Naive (per-cached-j
   `kv_b_proj @ latent[j]`) is O(len × 16M) per token. Folded
   (precompute `q' = q_nope @ kv_b_proj_K_per_head` once;
   per-cached-j `q' · latent[j] + q_pe · k_pe`) is O(16M + len ×
   130K). 60× speedup at long context. Naive made sense for CPU
   first-run because it maps line-for-line to
   `modeling_deepseek.py`. On GPU the folded form is the only
   one worth implementing — naive would just bottleneck on
   bandwidth for no debug value (the CPU path already exists as
   an oracle with the naive math).
2. **MLA cache lives in Metal buffers, not host `Box<[f32]>`.**
   `MlaKvCache` (the host-side struct landed this session) stays
   for the CPU path; add `MlaKvCacheGpu` with MTLBuffer fields
   for `latent_cache`, `rope_k_cache`, plus the `len: i32`
   counter. Allocate to `MAX_SEQ_LEN`; macOS lazy-commit handles
   the virtual reservation (existing GQA path already does the
   same trick at similar magnitudes).
3. **MLA dispatches inside the existing FullAttn branch.** The
   per-layer loop in `step_internal` (mod.rs:1180-ish) already
   branches `LayerKind::FullAttn` vs `LayerKind::LinearAttn`.
   For Cogito-V2 every layer is FullAttn (`full_attn_interval =
   1`); we add a second-level `attn_kind` branch inside FullAttn:
   `Gqa → full_attn_layer_forward (existing)` or
   `Mla → mla_attn_layer_forward_gpu (new)`. Same contract
   (writes to `linear_buffers.normed`, supports `chain_next_norm
   _off`), so the deferred-ring / prefetch / chained-MoE
   pipeline outside the branch is unchanged.
4. **Existing MoE GPU path is reused as-is.** `gpu_batched
   _experts_forward` doesn't care about attention flavor;
   it consumes a hidden state and produces a hidden state. The
   shared-expert-add (unconditional, no gate, contrasting Qwen)
   needs a small composition tweak — but only because the
   existing tail `post_attention_tail` does the gated path. Land
   a sibling `post_attention_tail_unscaled` or feature-toggle.
5. **YaRN inv_freq + mscale precomputed once at engine init.**
   Currently `step_internal_mla_cpu` computes them per token
   (microseconds, but pointless). Cache on `RsCtx` as
   `mla_yarn: Option<MlaYarnTables>` lazily built from VARIANT.
   GPU kernel reads the inv_freq table from a small constant
   buffer.

## Phases

### Phase 1 — GPU plumbing (state + LayerWeightCache + buffers)

Files: `state.rs`, `layer_weight_cache.rs`, `mod.rs`,
new `mla_attn_buffers.rs`.

Deliverables:

- `MlaKvCacheGpu { latent_cache: Buffer, rope_k_cache: Buffer,
  len: i32 }`. Sized at `MAX_SEQ_LEN * (kv_lora_rank +
  qk_rope_head_dim) * 4` bytes per layer.
- `LayerState::Mla` variant becomes `Mla(MlaKvCacheGpu)` for
  GPU builds, `MlaCpu(MlaKvCache)` if we keep both paths
  selectable. Cleaner: keep both variants and decide at
  alloc time based on a runtime config flag (default GPU).
  Affected match arms: `truncate`, `clear_all`, `pos_max`,
  `is_full`, snapshot stubs (still MlaUnsupported in v1).
- `LayerWeightCache` extension: `LayerAttnW::Mla(MlaAttnW
  { q_a_w/s/b, q_a_layernorm_w, q_b_w/s/b, kv_a_w/s/b,
  kv_a_layernorm_w, kv_b_w/s/b, o_proj_w/s/b })`. Resolution
  in `LayerWeightCache::build` gated on `VARIANT.attn_kind`.
- `ensure_mla_gpu_resources` — extends `ensure_linear_resources`
  to also build the per-layer `MlaKvCacheGpu`. Per-layer
  scratch buffers for q_full / q_pe / kv_pre / decoded etc.
  go in a new `MlaAttnBuffers` (analogue of the existing
  `LayerForwardBuffers`).
- `MlaYarnTables { inv_freq: Buffer, mscale: f32 }` lazily
  built and cached on `RsCtx`.

Compile-check both Qwen and Cogito-V2 stay green after
plumbing.

### Phase 2 — YaRN RoPE Metal kernel

File: `crates/moeflux/shaders/shaders.metal` (new function),
`gpu_rope.rs` (new dispatcher) or extend `rope.rs`.

Mirror `apply_rotary_emb_yarn` (rope.rs:306). The kernel takes
a per-head `[num_heads, rotary_dim]` buffer, the precomputed
`inv_freq[half]` constant buffer, the position scalar, the
mscale scalar; rotates in place. One thread per `(head, i)`
pair where `i ∈ [0, half)`. Dispatch shape `(num_heads, half,
1)`.

Validation: bit-tolerant diff (≤ 4 ULP) against
`apply_rotary_emb_yarn` on a small synthetic input. The
existing rope.rs has a `yarn_rope_at_pos_zero_mscale_one_is
_identity` test as a starting fixture — extend with a
non-trivial pos.

### Phase 3 — MLA attention Metal kernels (folded form)

This is the load-bearing slice. Three sub-kernels, each
mirrors a step in the folded math:

**3a. q' = q_nope @ kv_b_proj_K_per_head** — produces a
`[num_heads, kv_lora_rank=512]` buffer. Per head h:
`q'_h = q_nope[h] @ kv_b_proj_K[h]` where `kv_b_proj_K[h]` is
a `[qk_nope_head_dim=128, kv_lora_rank=512]` slice of
`kv_b_proj`.

This is a per-head matvec on a 4-bit weight matrix.
`dequant_matvec_4bit_v3` already handles the matvec shape;
we'd dispatch it `num_heads` times with per-head weight
offsets, OR write a fused kernel that walks all heads in one
dispatch. The fused version is cleaner — write a new
`dequant_per_head_matvec_4bit` that takes head_idx as a
threadgroup index.

Note: `kv_b_proj` is stored as a `[num_heads * (nope+v_head),
kv_lora_rank]` matrix in the manifest. Strided per-head reads
into the K-portion (rows `h*256 .. h*256 + 128`).

**3b. SDPA scoring + softmax + V_combine** — per token,
per head:

```
scores[j] = q'_h · latent_cache[j] + q_pe[h] · rope_k_cache[j]
            for j in 0..len
softmax(scores) in place
V_combine_h[c] = sum_j scores[j] * latent_cache[j][c]
                 for c in 0..kv_lora_rank
```

This is the tight inner loop. Geometry: `(num_heads,) ×
(num_threads_per_head,)` threads. Per head, threads cooperate
on the dot products + softmax reduction + weighted sum.

The latent_cache rows are 512 floats — fits comfortably in
threadgroup memory at low cache-len. For long contexts
(len > ~256 at 32KB tg-mem cap), tile.

**3c. out = V_combine @ kv_b_proj_V_per_head** — produces a
`[num_heads, v_head_dim=128]` buffer. Per head h:
`out[h] = V_combine[h] @ kv_b_proj_V[h]` where
`kv_b_proj_V[h]` is the `[v_head_dim, kv_lora_rank]` half of
`kv_b_proj` (rows `h*256+128 .. h*256+256`).

Same kernel shape as 3a, different stride.

**Validation:** diff against `step_internal_mla_cpu` at the
post-MLA hidden state (post-o_proj, pre-residual-add). Top-1
identity at first decode step, ≤ ULP-bounded drift on the
Cogito-V2 logits. The CPU path is the source of truth; any
disagreement is the GPU kernel's bug.

### Phase 4 — Forward integration

Files: `mla_attn_forward.rs` (new, analogous to
`full_attn_forward.rs`), `mod.rs`.

Deliverables:

- `mla_attn_layer_forward(metal, wf, wf_buf, layer_cache,
  linear_buffers, mla_buffers, deferred, layer_idx, pos,
  k_active, experts, io_pool, prefetch, prefetch_set,
  mla_kv_state, gpu_combine, prev_layer_chained,
  chain_next_norm_off)` — same contract as
  `full_attn_layer_forward`. Internally:
  1. Pre-attn rms_norm fused (existing `gpu_rms_norm_fused`).
  2. Q chain: `dequant_matvec_4bit_v3` for q_a_proj;
     per-head fused rms_norm; `dequant_matvec_4bit_v3` for
     q_b_proj.
  3. KV chain: `dequant_matvec_4bit_v3` for
     kv_a_proj_with_mqa; rms_norm on the latent half.
  4. YaRN RoPE on q_pe and k_pe (Phase 2 kernel).
  5. Cache append: copy latent + rope_k into the GPU cache
     at offset `len * stride`; bump `len`.
  6. Folded MLA attention (Phase 3 kernels).
  7. o_proj via `dequant_matvec_4bit_v3`.
  8. Hand off to MoE / dense MLP path (Phase 5).

- `step_internal` dispatch: inside the `LayerKind::FullAttn`
  arm, branch on `VARIANT.attn_kind` to call either
  `full_attn_layer_forward` or `mla_attn_layer_forward`.
  Remove the GPU-MLA-rejection arms added in this session
  (they become unreachable once dispatch lands).

- The existing dense MLP path doesn't apply to Qwen (no
  dense layers); add a `dense_mlp_layer_forward_gpu` for
  Cogito-V2 layers 0-2. Or for first GPU pass, run dense
  MLP CPU-side (via the existing `dense_mlp_swiglu_cpu`)
  with a GPU↔CPU sync — only 3 layers per token, < 1 % of
  total time. Decide based on perf measurement.

### Phase 5 — MoE composition tweak (unconditional shared expert)

The existing `post_attention_tail` (linear_attn_forward.rs)
applies `sigmoid_gate * shared_expert + routed_sum`. For
DeepSeek-V3 the shared expert is added unconditionally.

Land a sibling `post_attention_tail_unscaled` selected by
`VARIANT.shared_expert_gate`. Same kernel except the gate
path drops out.

### Phase 6 — Validation

- Unit: `cargo test -p moeflux --features
  model-cogito-v2-671b` (existing CPU smokes stay green;
  add GPU-vs-CPU diff smokes per kernel).
- Integration: `tests/cogito_v2_smoke.rs` adds a GPU
  variant — same prompt, assert top-1 token ID matches.
  Then a GPU long-generation smoke (200 tokens, watch for
  entropy drift / repetition).
- E2E: blallama with Cogito-V2 features, send "Hello. How
  are you?", expect same `"I'm doing well, thanks! How"`
  (exact match at T=0; close at T=0.7).

## Critical files (existing — to read first)

- `crates/moeflux/src/riir/mod.rs:1037..1306` — step_internal
  + the new step_internal_mla_cpu landed this session. The
  GPU MLA dispatch slots in alongside.
- `crates/moeflux/src/riir/full_attn_forward.rs` — contract
  template for `mla_attn_layer_forward` (signature, buffer
  ownership, deferred-ring usage, prefetch hand-off).
- `crates/moeflux/src/riir/linear_attn_forward.rs` — the
  `post_attention_tail` is here; this is what needs the
  unconditional-shared-expert sibling.
- `crates/moeflux/src/riir/gpu_attn.rs` — existing GQA
  scoring / softmax / values kernels. Won't reuse directly
  (MLA geometry differs) but the dispatch idioms are
  identical.
- `crates/moeflux/src/riir/gpu_matvec.rs` — `dequant_matvec
  _4bit_v3` is the workhorse for the Q / KV / O projections.
- `crates/moeflux/src/riir/gpu_norm.rs` — `gpu_rms_norm_fused`
  for the pre/post-attn norms.
- `crates/moeflux/src/riir/rope.rs` — CPU YaRN reference;
  Phase 2 ports this to Metal.
- `crates/moeflux/src/riir/mla_attn_cpu.rs` (this session) —
  full forward shape with comments mapping to
  `modeling_deepseek.py`. The GPU kernel mirrors this.
- `crates/moeflux/src/riir/layer_weight_cache.rs:170..200` —
  `FullAttn` arm currently resolves GQA tensor names; Phase 1
  adds an MLA arm.

## Open questions for Mike (decide in-session)

1. **Keep CPU MLA path callable, or delete after GPU lands?**
   Recommended: keep, as `pub fn eval_token_cpu` or behind
   `MOEFLUX_FORCE_CPU_MLA=1`. Dual purpose: diff oracle for
   future kernel work + last-resort fallback if GPU breaks.
2. **Dense MLP layers 0-2 — GPU or CPU?** Recommended: CPU
   first run (already implemented + tested), measure, port to
   GPU only if it's a meaningful fraction of per-token time.
3. **Diff oracle granularity.** Per-kernel diffs (separate
   fixtures for q_a_proj, q_b_proj, etc.) vs end-to-end logit
   diff only? Per-kernel is more debug-friendly but more code.
   Recommended: end-to-end logit diff for first GPU run; add
   per-kernel diffs only if logits disagree.

## Estimated session shape

Phases 1-2 (plumbing + YaRN kernel) are mechanical — half a
session. Phase 3 (MLA attention kernels) is the real work —
probably the second half. Phase 4 (forward integration) +
Phase 5 (MoE tweak) + Phase 6 (validation) likely spill into
session 2 unless Phase 3 lands cleanly first try.

Realistic outcome for one session: GPU MLA forward runs
end-to-end, produces the same coherent output as CPU on the
"Hello" prompt, throughput dramatically better. Long-
generation stability + perf tuning are session-3 material.

## Pointers (durable)

- This session's CPU landing: `cogito_v2_landing_state.md`
- DeepSeek-V3 architecture audit: `cogito_v2_architecture.md`
- moeflux RIIR strategy: `riir_moeflux_strategy.md`
- Reference impl (canonical math):
  `/Volumes/HF Models/models/hf/mlx-community/cogito-v2-preview-deepseek-671B-MoE-4bit/modeling_deepseek.py`
