# Qwen cmdbuf consolidation — Phase 0+2 landed

Outcome memo for 2026-05-13 session. Plan-of-record was
`qwen_cmdbuf_consolidation_plan.md` (still durable — Phases 3, 4, 5
remain as forward-looking work). Commit: moeflux `d5d7676`
("linear_attn: fold CMD1 into post_attention_tail's cmdbuf").

## What landed

### Phase 0 — Per-cmdbuf instrumentation

`MetalBackend` gains:
- `commit_and_wait_labeled(&self, cmdbuf, label)` — wraps commit+wait
  with `std::time::Instant` timing, accumulates per-label CPU wait stats.
- `cmdbuf_stats()` getter (sorted by label), `reset_cmdbuf_stats()`.
- `queue_clone()` — returns owned `CommandQueue` so the caller can
  allocate cmdbufs without holding a borrow on `*metal`. Without this
  the cmdbuf borrow conflicts with the `&mut metal` arg to
  `post_attention_tail`.

GPU runtime stats (gpu_start_time / gpu_end_time) are stubbed at zero —
metal-rs 0.32 doesn't surface those ObjC selectors. Wall-clock CPU wait
is sufficient for the metric we care about; surface them later if a
metal-rs upgrade lands them.

### Phase 2 — Fold linear-attn CMD1 into the same cmdbuf as CMD2+3

`post_attention_tail` signature gains `cmdbuf: &CommandBufferRef`
as a caller-owned parameter. The inner `metal.queue().new_command_buffer()`
is gone; the closing `commit + wait_until_completed` is now
`metal.commit_and_wait_labeled(cmdbuf, "post_attn_tail.cmd2_3")`.

`linear_attn_layer_forward` hoists the cmdbuf to function scope (above
the CMD1 encode block), encodes CMD1 work into it, and passes the
same cmdbuf into `post_attention_tail` which appends CMD2+3 onto it.
The CMD1 commit+wait is **deleted**. Encoders within one cmdbuf
serialize on the GPU side, preserving the rms_norm → projections →
linear-attn → o_proj → residual_add → post-attn-norm → gate →
shared-FFN data flow.

Net per-layer cmdbuf count:
- **Linear-attn (27/36 layers): 3 → 2** ✓
- Full-attn (9/36 layers): 3 unchanged. The host-bounce (q/k/v
  readback + CPU per-head norm + RoPE + KV append) sits between CMD1
  and CMD2+3 and can't fold until Phase 3b moves the host work to GPU.
  Full-attn's CMD2+3 is still wired through the same
  `post_attention_tail` though — caller creates a fresh cmdbuf after
  the host-bounce, passes it in.

## Measured win

Bench protocol: high-perf power management, n=3, 992-token prefill via
`./bench.py --binary blallama-{prefold,postfold} --max-tokens 4`. Both
binaries built from the same drama_llama checkout, differing only by a
moeflux stash-pop between them.

| | Pre-fold | Post-fold | Δ |
|---|---|---|---|
| Iter 1 | 125.12s | 108.40s | -16.7s |
| Iter 2 | 125.35s | 98.44s | -26.9s |
| Iter 3 | 129.43s | 104.54s | -24.9s |
| **mean** | **126.6 ± 2.5s** | **103.8 ± 5.1s** | **-22.8s (-18%)** |
| tok/s | 7.84 | 9.56 | +22% |

Signal-to-noise: ~4σ above the combined stdev. Real win, not variance.

Bit-exactness preserved at the diff oracle:
- `layer_forward_dump_close_c_vs_rust` (linear-attn layer 0):
  cosine=1.0000000, max_abs_diff=9.2e-7
- `_cpu_combine`: cosine=1.0, max_abs_diff=8.9e-7
- `_full_attn` (full-attn layer 3, CPU SDPA): cosine=1.0, max_abs_diff=1.9e-5
- `_full_attn_gpu_path` (full-attn layer 3, GPU SDPA at kv_len=33):
  cosine=1.0, max_abs_diff=7.2e-6

## Bench-protocol learning (load-bearing)

**Default macOS power management thermally couples consecutive bench
iterations on the M2 Max.** First iter after a reboot runs at peak
speed; subsequent iters slow ~20% even with fans not engaging and the
external TB enclosure (which has pad + heatsink + fan) cool to the
touch. Cause unidentified — could be Mac SoC silent throttling
(macOS throttles before fans engage), Metal driver state, scheduler
QoS biasing, or memory-pressure-related page cache shuffling. Not
external-NVMe-thermal (enclosure cool, fan engaged); not chip-thermal
in the obvious sense (chassis warm but fans never spin up).

**High-perf power management flattens the variance** from ~12s stdev to
~2.5s stdev across iters within a run. Absolute numbers go *up* a bit
(no cold-chip opportunism), but reproducibility is dramatically better.

**A/B protocol**: enable high-perf, run pre + post back-to-back, both
n=3. Reboots between binaries are NOT needed — high-perf maintains a
stable thermal state. Compare means.

**Anti-pattern**: comparing a cold-chip first iter against a
mid-session warm-chip iter. That's what produced the spurious "8%
regression" earlier in this session — pre-fold profile (chip cold)
came in at 93.5s, post-fold profile (chip already warm from pre-fold)
at 101.2s. Both numbers are real, but they describe different thermal
states, not different code paths.

## Forward-looking from here

The plan-of-record (`qwen_cmdbuf_consolidation_plan.md`) still has:

- **Phase 3** — fold full-attn CMD1 into CMD2+3 by moving per-head Q/K
  rms_norm + RoPE to GPU (host-bounce removal). Reuses existing
  `yarn_rope_apply` kernel with a plain inv_freq table (no new RoPE
  shader needed). New `rms_norm_per_head_bf16` kernel needed (~30
  Metal LOC). Profile justified deferring this session — CPU norm/RoPE
  combined was <0.5% of main thread — but the cmdbuf-fold win it
  unlocks (full-attn 3 → 1 + linear-attn 2 → 2 = average ~1.25
  cmdbufs/layer) is the bigger structural improvement.

- **Phase 4b** — cross-layer cmdbuf chaining / overlap reorder. The
  remaining commit-wait per layer is on `post_attn_tail.cmd2_3` and
  blocks moe_router_cpu reading gate_logits. Phase 4b kicks off the
  cmdbuf, lets CPU drain older deferred work while it runs, *then*
  waits. Builds on a small in-flight-future state machine. Probably
  one session of work.

- **Phase 5** — GPU MoE router. `moe_router_cpu` shows 0.1% inclusive
  in the profile. Not worth the refactor weight at this stage.

**The bigger fish — and probably the next session's primary target —
is the `GPU_KV_SEQ = 8192` cliff.** Past kv_len=8192, full-attn falls
back to `sdpa_cpu` per token per full-attn layer, and that cost scales
O(kv_len). For Agora's 40-60k-token workloads this is the dominant
cost past the system+tools prefill cap. Tiled SDPA exists for the MLA
path (`mla_sdpa_tile_accumulate` + `mla_sdpa_tile_finalize`,
flash-attention online-softmax style). Porting the same shape to
full-attn GQA is the path: new `attn_scores_tile_accumulate` /
`attn_values_tile_finalize` kernels, replacing the existing
GPU-attn-fast-path encoder with a tiled wrapper.

## Plumbing notes

- `bench.py` had a stale model name (`qwen3-6-35b-a3b` → `qwen3-6-a3b`).
  Fixed locally. Same fix landed in `profile.py` earlier in the session.
  Both files are gitignored, so the fix is workspace-local.
- `bench.py` now logs the prompt length on each iter — added because
  `/tmp/prefill_prompt.txt` got nuked on reboot and the silent
  empty-prompt fallback wasted a bench run. The log line catches
  the failure mode loudly.
- `prefill_prompt.txt` lives at the repo root now (gitignored) so it
  survives reboots. Used by both `bench.py --prompt` and
  `profile.py --prompt` via shell expansion: `--prompt "$(cat ./prefill_prompt.txt)"`.
- `drama_llama/.gitignore` got a new line: `/prefill_prompt.txt`.
  Uncommitted from this session (Mike has unrelated WIP in `src/`).
  Fold into a Mike commit when convenient.
- Binaries `target/release/blallama-{prefold,postfold}` are sitting at
  16 MB each. `rm` whenever; they were the A/B fixture.

## Run commands (canonical for follow-up)

```bash
# Build:
cd ~/Projects/drama_llama
cargo build --release --bin blallama \
  --features axum,cli,toml,moeflux-model-qwen3-6-35b-a3b

# Bench (post-fold steady state ~104s on a3b, 992-token prefill):
./bench.py --model a3b --no-build \
  --max-tokens 4 \
  --prompt "$(cat ./prefill_prompt.txt)" \
  -n 3

# Diff oracle (correctness — runs against the C-side moeflux-sys):
cd ~/Projects/moeflux
cargo test -p moeflux --features model-qwen3-6-35b-a3b --release \
  --test diff_oracle layer_forward_dump_close_c_vs_rust \
  -- --ignored --nocapture --test-threads=1

# Profile (samply main-thread inclusive — confirms where remaining
# wait sits):
cd ~/Projects/drama_llama
./profile.py --model a3b --no-build --duration 300 --max-tokens 2 \
  --prompt "$(cat ./prefill_prompt.txt)"
```
