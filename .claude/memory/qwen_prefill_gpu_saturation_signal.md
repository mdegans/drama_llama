# Post-S7-1a GPU saturation signal

**Date:** 2026-05-14
**Observed by:** Mike, during the long-prompt bench on a3b after
S7-1a (commit `6628eaf`).

## What was seen

While the 16k-prefill bench runs:
- **CPU usage: ~0% sustained** on the main thread.
- **GPU: saturated** (no measurable idle).

Pre-S7-1a (and pre-B-0a/b/c/d) the shape was the inverse: CPU
busy in `commit_and_wait` polling, GPU bursty/idle between phases.
The session-5 profile (`profile_post_5d8.md`) showed 76% of
main-thread time in `__psynch_cvwait` — that's gone now.

## What this means for the remaining gap

The remaining ~13× gap to llama.cpp on 992 prefill (75 vs 970
tok/s) is **no longer commit-overhead bound**. It's GPU-throughput
bound. Future wins come from a different set of levers than the
session-6 work attacked:

- **Kernel efficiency**: FlashAttention-style SDPA fusion, better
  matvec tiling, fewer redundant memory passes. The current 4-bit
  matvec spends real GPU cycles on dequantization that a smarter
  kernel can hide.
- **Buffer alloc / copy churn**: `MtlBuffer::<f32>::with_data` and
  `MtlBuffer::<f32>::with_len` are still called per layer-forward.
  Persistent per-chunk buffers would skip the alloc + initial host
  upload.
- **Multi-cmdbuf parallelism** (parent plan Phase C): split the
  graph across n_cb cmdbufs encoded in parallel via dispatch_apply.
  Apple's documented sweet spot is n_cb=1-2 — modest but real
  speedup on top of saturated single-cmdbuf encoding.

## What this rules out

- **CPU work refactor**: pointless until we have a CPU-bound
  workload again. The remaining CPU steps (q/k norm + RoPE,
  sigmoid_gate, bucket build) happen in negligible time relative
  to GPU compute.
- **Reducing commit count further**: under saturation, even one
  commit per chunk is ~free in wall-clock terms. The S7-2/S7-3
  plan items (cross-layer cmdbuf fusion) are nice-to-have but
  won't move tok/s much until kernel-side wins land.

## How to verify this stays true

Re-profile via `profile.py --model a3b --prompt-file
prefill_prompt_long.txt --max-tokens 1 --duration 60 --top 30`.
If `__psynch_cvwait` is back to ≥30% of main-thread time, we've
regressed on the saturation shape and the next-session priorities
flip back to commit-count work.

## Strategic implication for session 7

The session 7 plan currently leads with S7-1b (GPU q/k norm +
RoPE + KV append) and S7-2/S7-3 (orchestrator-level cmdbuf
fusion). Given GPU saturation, those are now lower priority than
**kernel-efficiency work** (FlashAttention-style SDPA, persistent
chunk buffers).

Recommend reshuffle for session 7:
1. **First**: a sample profile post-S7-1a to characterize the new
   pole — is it 4-bit matvec? SDPA? KV ops? somewhere else?
2. **Second**: kernel work targeted at the actual pole.
3. **Third**: the planned orchestrator refactor, only if profile
   shows residual cmdbuf overhead.

Don't bury this insight under the existing S7 plan — open
session 7 with `profile.py` first, decide priorities from data.
