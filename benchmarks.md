# benchmarks.md

Recorded `blallama` benchmark numbers, newest first per model.

How to reproduce a row — **use `--prompt-file`, not `--prompt`** (the
latter takes a literal string; passing a filename to it benches the
tokenized filename, not the file's contents):

```bash
# reboot first, then let the machine settle — wait for `uptime`'s
# 1-min load average back to ~1. Page-cache state AND CPU contention
# from boot churn both swing tok/s on this hardware; benching 3 min
# post-reboot under load avg 20 gave incoherent numbers once.
# bench.py always builds (cargo no-op-rebuilds in ~1s).
./bench.py --model a3b -n 3 --prompt-file prefill_prompt.txt --max-tokens 1
./bench.py --model a3b -n 3 --prompt-file prefill_prompt_long.txt --max-tokens 1
```

`prefill_tok/s` = `input_tokens / elapsed`. With `--max-tokens 1` the
single-token decode tail is negligible, so it reads as a clean prefill
rate. Iteration 1 is often a cold-start outlier (model load + cold page
cache); the warm iterations are the number that matters — that's why we
run `-n 3`.

Row format: `- [<commit>] <date>: <metric> — <notes>`

---

## qwen3-6-a3b (moeflux backend, M2 Max)

Prompts: `prefill_prompt.txt` ≈ 992 tokens, `prefill_prompt_long.txt`
≈ 15692 tokens. seed=42, temperature=0.0.

- [a236a0e] 2026-05-19: **prefill ≈ 254 tok/s** — Tier 1 (chunkwise
  DeltaNet phases 3 + 5-GEMM2 → simdgroup_matrix). 992-tok: 247.81 /
  254.82 / 254.03; 15692-tok: 256.04 / 256.58 / 253.57. Dirty
  (uptime ~33 min, quiet machine — no reboot A/B). Directional +1.6%
  over f0f8edb on both prompts; inside the ±0.5 tok/s noise band but
  consistent. Correctness: chunkwise diff oracle cos=1.0, canary 12/12.
- [f0f8edb] 2026-05-19: **prefill ≈ 250 tok/s** — 992-tok: 128.85 (cold) /
  252.22 / 249.31; 15692-tok: 251.26 / 251.49 / 250.84 (stdev 0.000).
  Reboot, uptime ~10 min. First correctly-flagged bench since the
  graph-mode arc — prefill rate is now flat across context length.
  Supersedes the session-12 ~43.5 tok/s @ 15.7k figure (5.8× faster).

### Historical (from `.claude/memory/`, pre-`benchmarks.md`)

- session 12 (~2026-05-16): prefill ≈ 43.5 tok/s @ 15.7k tokens.
- session 11 (~2026-05-15): +10.3% over the session-4 baseline.
- session 6 Part B precursors (~2026-05-14): 36.8 → 74.66 tok/s @ 992
  (directional warm bench) — prior high-water mark on the 992 prompt.
- session 4 (~2026-05-13): 10.54 tok/s essay+512; ~21 tok/s @ 992
  single-chunk.
