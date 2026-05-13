# Qwen batched prefill — session 3 outcome

Plan-of-record was [`qwen_batched_prefill_session2_gpu_plan.md`](qwen_batched_prefill_session2_gpu_plan.md); session-2 outcome (what we built on) is [`qwen_batched_prefill_session2_landed.md`](qwen_batched_prefill_session2_landed.md); this session's plan file is `/Users/mdegans/.claude/plans/sunny-mixing-scott.md`.

**Headline:** Phase 4 (MoE permute-and-fuse) landed end-to-end with the synthetic-data diff test at cosine = 1.000000000 across all 4 tokens. Phase 5 / Phase 6 *were* scoped to ship a full batched layer forward + capstone test, but mid-session Mike caught a real scope issue with the parallel `_batched` API I was drafting and we pivoted: instead of carrying two public surfaces, this session ships the rename that sets up session 4 to swap the canonical `step_internal` body for batched primitives without further public-API churn. The renamed per-token implementation stays as the diff oracle.

## What landed

All changes live on moeflux main; commit format follows session 1/2 (one commit per phase).

### Phase 4: MoE permute-and-fuse + bucketing helper + bucket-accumulate kernel

The actual prefill I/O win primitive. Replaces per-token MoE dispatch (which re-reads each expert's blob K times per token) with a bucketed dispatch (read each blob once per non-empty bucket and run a batched matmul over the bucket via Phase 2's `encode_matvec_n_tokens`).

Files touched (one logical commit):
- `crates/moeflux/src/riir/moe_router.rs` — `pub struct ExpertBuckets` (CSR view: `expert_ids`, `offsets`, `token_idx`, `weights`) + `pub fn build_expert_buckets(...)`. Four unit tests: round-trip, distinct-tokens-within-bucket (the no-atomics invariant), empty-bucket skipping, single-expert degenerate.
- `crates/moeflux/shaders/shaders.metal` — `kernel void moe_bucket_accumulate(...)` (15-line scatter-weighted-add, atomic-free because top-K returns distinct experts per token so `token_idx[b]` is unique within one bucket; cross-bucket sequencing handled by Metal's encoder ordering).
- `crates/moeflux/src/riir/gpu_matvec.rs` — `encode_matvec_n_tokens` signature extended with `input_off: u64, output_off: u64` (byte offsets into packed buffers; replaces the implicit zero binding). One additive change, four call-site updates (all in `tests/batched_diff_oracle.rs`).
- `crates/moeflux/src/riir/expert_forward.rs` — `pub fn encode_moe_batched_permute_fuse(...)`. Per-non-empty-bucket it issues: gate matvec → up matvec → swiglu (flat-dispatched over `bucket_size * MOE_INTERMEDIATE` because `swiglu_fused` is element-wise) → down matvec → `moe_bucket_accumulate`. Caller stages: pre-loaded expert blobs (parallel to `buckets.expert_ids`), packed `bucket_input` (host gather), zeroed `out_sum`.
- `crates/moeflux/tests/batched_diff_oracle.rs` — `moe_permute_fuse_n_tokens_matches_tokenwise` test. Fixture: N=4 tokens, k_active=4, num_experts=12 (4 empty by construction — exercises the empty-bucket skip), each bucket size 2 (forces multi-token-per-bucket arithmetic). Reference: tokenwise loop of `gpu_expert_forward` per (token, slot) pair, weighted-summed per token.

Diff result, all 4 tokens:

| Token | cosine | max_abs_diff |
|-------|--------|--------------|
| 0 | 1.000000000 | 3.052×10⁻⁵ |
| 1 | 1.000000000 | 7.629×10⁻⁶ |
| 2 | 1.000000000 | 6.104×10⁻⁵ |
| 3 | 1.000000000 | 6.104×10⁻⁵ |

The plan predicted "cosine ≥ 0.9999, FP-reorder envelope only" — reality came in tighter (cosine literally == 1.0). The per-bucket vs per-slot accumulation order difference doesn't disturb the dot-product magnitude.

### Phase 5 / Phase 6: scope pivot — canonical-vs-oracle rename

Mid-session I caught myself adding a `pub fn eval_prompt_batched` + `pub(crate) fn step_internal_batched` parallel API as a tokenwise wrapper, then walking back from making it "really batched" because the per-layer integration needs a `post_attention_tail` refactor (~520 lines, too tightly coupled with the deferred ring + prefetch + chained RMSNorm for one session's scope). Mike pushed back on the parallel API: **"If we don't need two versions of a thing, we should replace the old one with the new. Having a CPU and a GPU or an internal version of a thing is helpful as an oracle but not in the public API."**

The right shape: one canonical public API, the older implementation kept as a diff oracle with restricted visibility. Settled rename:

- `RsCtx::step_internal(token: i32, pos: i32, ...)` → `pub(crate) fn step_internal_per_token_oracle(token, pos, ...)`. Doc-tagged as oracle. Visible inside the crate so `eval_token` (decode) and the new `step_internal` (slice-taking) can call it.
- New `pub(crate) fn step_internal(tokens: &[i32], start_pos: i32, ...)` is the canonical multi-token forward. Session 3 body: per-token loop calling `step_internal_per_token_oracle`. Session 4 body: GPU batched primitives.
- `pub fn eval_prompt(...)` body now calls `self.step_internal(tokens, start_pos as i32, Some(&mut logits[..]))`. Public-API name unchanged. No parallel `_batched` API.
- `pub fn eval_token(...)` body calls `step_internal_per_token_oracle` directly (single-token decode doesn't benefit from batching).

Phase 6 scaffold test (`tests/diff_oracle.rs::eval_prompt_matches_per_token_oracle`):
- Reference: per-token `eval_token` loop (routes through the oracle).
- Test: canonical `eval_prompt`.
- Compares end-of-prompt logits + post-prompt continuation logits. Currently trivially passes (cosine = 1.0, max_abs_diff = 0) because the canonical path *is* the oracle loop. Becomes a real test in session 4 when the canonical body swaps in batched primitives.

The N=16 prompt prefill + 1 continuation token completes in 3.1s on the 1.39 GB A3B artifact. Fast enough to leave on by default.

## Verification commands

```bash
cd ~/Projects/moeflux

# Phase 4 (synthetic, fast — ~0.2s):
cargo test --release -p moeflux --features model-qwen3-6-35b-a3b \
  --test batched_diff_oracle -- --ignored --nocapture --test-threads=1 \
  moe_permute_fuse_n_tokens_matches_tokenwise

# Phase 4 + 1/2/3 regression (all batched primitives, ~0.15s):
cargo test --release -p moeflux --features model-qwen3-6-35b-a3b \
  --test batched_diff_oracle -- --ignored --nocapture --test-threads=1

# Phase 6 scaffold (real artifacts, ~3s):
cargo test --release -p moeflux --features model-qwen3-6-35b-a3b \
  --test diff_oracle -- --ignored --nocapture --test-threads=1 \
  eval_prompt_matches_per_token_oracle

# Rename regression sanity (real artifacts, ~15s):
cargo test --release -p moeflux --features model-qwen3-6-35b-a3b \
  --test diff_oracle -- --ignored --nocapture --test-threads=1 \
  state_round_trip_rust slot_reuse_race_regression_rust \
  eval_token_matches_c_single_step

# In-crate unit tests (includes build_expert_buckets coverage):
cargo test --release -p moeflux --features model-qwen3-6-35b-a3b --lib
```

All passed at session end.

## Architecture conventions established / refined

- **Canonical-vs-oracle pattern**: when a session ships a new path that supersedes an old one, the old becomes the in-crate oracle with a descriptive `_oracle`-suffixed name and `pub(crate)` (or stricter) visibility. The public API surface stays at one name per concern. Diff tests bridge canonical and oracle via existing public methods (e.g., `eval_token`-per-token-loop hits the oracle indirectly).
- **`*_n_tokens` axis vs `*_batched` axis** (refined from session 2): `_n_tokens` for batched-over-tokens (the prefill axis); `_batched` for the original batched-over-experts axis. The new `encode_moe_batched_permute_fuse` is *both* — bucketed over experts AND batched over tokens within each bucket — so it doesn't fit the binary; the name leads with the algorithmic shape ("permute_fuse").
- **`MtlBuffer::buffer()` vs `.raw()`**: production callers that need `&Buffer` (for `set_buffer` taking an owned-buffer lifetime, or for the new `encode_matvec_n_tokens(...input_off, ...)` signature) use `.buffer()`. `&BufferRef` callers use `.raw()`.
- **Atomic-free scatter via uniqueness invariant**: `moe_bucket_accumulate` doesn't need `atomic_add` because top-K's distinct-experts-per-token guarantees `token_idx[b]` is unique within one bucket. Cross-bucket accumulation serializes through Metal's encoder ordering inside a single cmdbuf. Documented in the kernel comment + `build_expert_buckets` doc.

## Session 4 plan of record

Mike's note: "In the next session, I'd like to cover all of the out of scope work." Concretely:

1. **Refactor `post_attention_tail`** (`linear_attn_forward.rs:739`) to expose a "stop before K-expert dispatch" boundary. The function currently does post-attn RMSNorm + routing + shared expert FFN + K-expert dispatch + combine + (optional) chain-RMSNorm all in one body. Split into:
   - `post_attention_pre_moe(...) -> PostAttnIntermediates { h_mid, h_post, shared_out, routing_indices, routing_weights, sigmoid_gate }`. Runs everything up to and including the shared expert FFN.
   - `moe_combine_residual_into_buffer(...)` for the per-token combine step.
   - This is the load-bearing scope item for session 4. Risk: the deferred-ring + prefetch + chained-RMSNorm interaction needs careful handling.

2. **Replace `step_internal`'s body** (currently the per-token loop) with the real batched layer forward:
   - For each layer: per-token `post_attention_pre_moe` (loop, captures intermediates into `[N, hidden_dim]` host buffers) → batched MoE permute-fuse over the joint N×k_active routing CSR → per-token combine.
   - For full-attn layers, also batch the SDPA call via Phase 3's `encode_sdpa_causal_tiled`. Linear-attn layers stay tokenwise (recurrent state).
   - Optional batched Q/K/V/O projections (using Phase 1's `bf16_matmul_n_tokens` for BF16 weights / Phase 2's `encode_matvec_n_tokens` for 4-bit) — measure first, then decide.

3. **Phase 6 graduates from scaffold to real signal**: `eval_prompt_matches_per_token_oracle` will compare canonical batched vs oracle tokenwise. The expected envelope is cosine ≥ 0.9999 (per-bucket vs per-slot accumulation order — same as the standalone Phase 4 diff). Failure modes to watch: KV append off-by-N in batched, MoE routing CSR construction bug for k_active=8 (Qwen3 production setting; session-3 fixture used k_active=4).

4. **Bench**: high-perf-power, reboot between revisions, n≥3. Headline: prefill rate on a3b (cold and warm) vs current main. Expected ~6× on the SSD-bandwidth pole with N=64 chunks.

## Risks for session 4

- **`post_attention_tail` refactor surface**: ~520 lines, owns deferred-ring state, owns prefetch interaction, owns chained-RMSNorm. Splitting cleanly is non-trivial. Suggested approach: introduce the split as a refactor-only PR (no behaviour change), verify against existing diff_oracle tests, then layer batched MoE on top.
- **Shared expert handling**: shared experts are 4-bit and run per-token in the current path. For batched: either keep per-token (loop) or batch via Phase 2's `encode_matvec_n_tokens`. Per-token is correctness-first; batched is a measurement question.
- **GPU mirror KV + SDPA fast-path interaction**: the per-token full-attn path maintains a GPU mirror of the KV cache up to `GPU_KV_SEQ=8192` for the fast-path `attn_*_batched` kernels. Phase 3's tiled SDPA doesn't need the mirror. When we route `step_internal` (batched) through tiled SDPA, the per-token GPU mirror code becomes dead for prefill but is still needed for decode. Either remove for prefill (and let decode keep its mirror) or leave the mirror code in place. Defer until measurement.
- **Phase 4 only validates against synthetic data**: real-weight diff happens for the first time in session 4 via the end-to-end test. The synthetic test caught cosine = 1.0 but production routing distributions might have different bucket-size statistics. Worth watching the per-token cosine in the end-to-end test.

## Calibration

- **Scope catch from Mike (mid-session)**: I drafted a parallel `pub fn eval_prompt_batched` + `pub(crate) fn step_internal_batched` API as a tokenwise wrapper, telling myself it "ships the API surface for downstream callers." Mike: "If we don't need two versions of a thing, we should replace the old one with the new." Correct call. Lesson: don't ship the API of a thing before the thing itself exists; if the thing is identical to an existing thing, the API rename happens at integration time, not as scaffolding.
- **Widen-the-frame call I made before Mike's catch**: I was cycling between scope options (heavy refactor vs scope down vs tokenwise wrapper) and asked Mike to choose between three options via AskUserQuestion. He rejected the question and clarified instead — which was the right move. The lesson per Mike's CLAUDE.md note: "stop and ask for help" when spinning. I did, and the answer was a *rename*, not a scope choice.
- **Plan-mode scout agents (3 parallel Explore agents)** turned out to be excellent ROI for this session — they came back with concrete file:line pointers + signatures that I trusted into the plan file. I never needed to re-scout the same files. The "load-bearing files" tally (full_attn_forward, post_attention_tail, expert_forward, mod.rs, state_snapshot, diff_oracle, batched_diff_oracle) was complete from the scouts.
- **Context discipline**: at session end, /context reported 24% used (243k / 1M). The "Don't wrap on context anxiety" memory was directly applicable — Mike explicitly told me twice not to pre-emptively scope down for context budget. Trust the budget, trust the process; the harness re-paginates if needed.
- **Pivot landing**: even though session 3 didn't ship "all of Phase 4-6 as planned", it shipped the load-bearing primitive (Phase 4) cleanly + a sound architectural prep for session 4. That's a real-shaped session. The original plan's Phase 5/6 sizing was off — `batched_full_attn_layer_forward` from scratch is two sessions of work, not one. Session 4 has a concrete entry point (refactor `post_attention_tail`) and a known end-state (canonical `step_internal` body uses batched primitives).
