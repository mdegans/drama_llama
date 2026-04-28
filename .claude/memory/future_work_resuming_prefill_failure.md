---
name: Investigate resuming-prefill failure post-5d-8
description: Pre-existing failure in `resuming_prefill_after_seq_rm_matches_full_prefill` reproduces at moeflux 5cfb521 (5d-7b) — the riir baseline before 5d-8. Worth investigating once 5d-8 is committed.
type: project
---

# `resuming_prefill_after_seq_rm_matches_full_prefill` fails at baseline

**Discovered**: 2026-04-28, mid-5d-8 implementation.
**Repro**: bisected to baseline `5cfb521` (5d-7b) — fails identically
without any 5d-8 changes applied. Not a 5d-8 regression.

## Symptom

Test in `crates/moeflux/tests/consecutive_eval_prompt.rs:419`. The
flow is:

1. Open RsCtx, run `eval_prompt(P_full)` → capture baseline argmax.
2. Open fresh RsCtx, run `eval_prompt(P_prefix)` (first 8 tokens of
   P_full).
3. `memory_seq_rm` to truncate the KV down to some position k.
4. Re-run `eval_prompt(remaining tokens of P_full)` — the "resume".
5. Assert: resume's argmax matches baseline's argmax.

Observed at 5cfb521:
```
[resume] baseline traj: [248046, 198, 248045, 74455, 198, 248068, 271,
                         248069, 271, 760, 4496, 173976, 323, 4000,
                         5000, 6000]
[resume] resume traj:   [62, 332, 271, 760, 88461, 4549, 63, 1654, 369,
                         279, 6007, 314, 279, 1510, 3833, 19557]
[resume] top-20 jaccard=0.290 traj_match=0/16
```

argmax mismatches on the very first token after resume.

## Why it's worth investigating

This points at a real bug in either:

- riir's `memory_seq_rm` + resume path on linear-attn layers (the
  lossy-partial-truncate bug class — `blallama_session_state_pollution.md`
  bisect already flagged this on the C side).
- Some divergence in how `step_internal` handles `pos` tracking after
  truncation.

The bisect-findings memory notes that "lossy partial-truncate of
linear-attn" was identified as one of two real upstream bugs. This
test may be exercising the riir-side analogue. If the riir port is
"differential against C as oracle" but C also fails this test, then
both are wrong — and 5d-8 didn't make it worse.

## How to investigate

1. Run the same test against the C path (CBackend) — does C also
   fail? (Likely yes, per the bisect memory.)
2. If both fail, the test itself is asserting against an expectation
   the implementation never satisfied — re-classify as a known-
   failing test or add `#[should_panic]` until the truncate-resume
   bug is properly fixed.
3. If C passes and riir fails, the riir port has a real divergence
   in either `memory_seq_rm` or resume-path `pos` handling.

## Don't conflate with 5d-8

5d-8's diff suite passed at the canonical end-to-end checks
(`eval_token_matches_c_single_step`, `eval_prompt_matches_c_multi_token`)
with cosine 1.0000000. The chained CMD3 → next-layer normed slice is
correct.

Followup work, not a blocker.
