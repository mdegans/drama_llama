---
name: Reboot before perf benches
description: Page-cache / mmap state has measurable variance on this hardware; reboot before any A/B comparison and treat single-run numbers as approximate.
type: feedback
---

When measuring blallama tok/s for any A/B comparison (before/after a
code change, head-to-head between commits, etc.), reboot first.

**Why:** On 2026-05-01 we chased what looked like a 9% regression
(1.96 → 1.78 tok/s on Qwen3.5-A17B max_tokens=512) for an entire
arc of the session — gating F_RDAHEAD=0 to cogito-only, setting up
worktree bisect, building old code at the 1.96-baseline commit. The
old code on the current machine state benched at **1.7535 tok/s** —
slightly slower than the current code. The "regression" was
entirely machine state. Mike noted: "I haven't rebooted in a very
long time." Anecdotally the same hardware has hit 2.5 tok/s in
prior sessions, suggesting cache state has ±0.5+ tok/s of variance.

**How to apply:**
- Before any important perf comparison, reboot. Don't trust
  same-session A/B unless we're talking about big jumps (the 2×
  set-based-matching win was robust under any state).
- `bench.py` prints `uptime` alongside results so the bench log
  carries machine-state context. Use it.
- For tiny perf claims (<5%), a single bench is not signal — run
  `-n 3` or higher and look at the stdev. If stdev is comparable
  to the difference, say so.
- If a code change unexpectedly looks like a regression but the
  rationale is "this should be at worst neutral," default to
  suspecting machine state before suspecting the code. The
  worktree-bisect dance is cheap and definitive.
