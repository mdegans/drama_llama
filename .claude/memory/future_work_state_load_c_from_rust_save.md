# Future work — state_load_c_from_rust_save pre-existing failure

## Status (2026-05-13)

`state_load_c_from_rust_save` fails on moeflux main (commit 48c8562
"batched-prefill: canonical eval_prompt + per-token oracle rename")
before Phase A's refactor. The failure is in the C side: `state_load`
rejects the bytes Rust produced.

## How surfaced

While verifying the Phase A `post_attention_tail` split, I ran the
canary battery from the session-4 plan. `state_load_c_from_rust_save`
failed both before AND after the refactor — a `git stash` + retest
on main confirmed it's pre-existing, not a Phase A regression.

The sister test `state_load_rust_from_c_save` (C saves, Rust loads)
passes. So C → Rust is fine; the regression is Rust → C wire
direction only.

## Failure shape

```
thread 'state_load_c_from_rust_save' panicked at .../diff_oracle.rs:3896:27:
FAILED
```

Line 3896 is `c.0.state_load(&snap).expect("C state_load(rust_snap)");`
— the `expect` itself fires, meaning `state_load` returned `Err`.
Not a cosine drift; the bytes aren't accepted.

## Likely cause (hypothesis)

Snapshot wire format on the Rust side drifted from C's expectations
during the Phase 4 batched-prefill work, *or* the rename in commit
48c8562 changed how state is laid out at `state_save` time. The
canonical `eval_prompt` calls `step_internal` which is now a loop
over `step_internal_per_token_oracle` — same per-token state
machinery, but maybe a subtle difference in what gets snapshotted
(e.g., deferred ring state, prefetch state, layer cache hot bits).

Snapshot v2 (per `cogito_v2_full_gpu_session2_landed.md`) added MLA
plumbing but should have been bit-exact roundtrip on the test suite.
This may be specific to a3b's snapshot fields.

## Pure-Rust state-save/load: works correctly

For the avoidance of doubt, the Rust-only state path is **fully
functional**:

- `state_round_trip_rust` (Rust → Rust): cosine=1.0
- `state_load_rust_from_c_save` (C → Rust): passes
- `prompt_cache_start_pos_nonzero_matches` (Phase D, 2026-05-13):
  cosine=1.0, **max_abs_diff=0.000e0 (bit-exact)** — eval_prompt(prefix,0)
  → state_save → reset → state_load → eval_prompt(suffix, prefix.len())
  produces the SAME bytes as a fresh full-prompt forward.

The only broken direction is Rust → C, and C is deprecated.
Drama_llama's prefix-reuse layer (the production consumer of this
path) is Rust-only end-to-end, so this failure has no downstream
impact.

## Triage

**Insignificant for session 4 scope.** The Phase A refactor is
behavior-preserving (the 9 batched diff tests + `state_round_trip_rust`
+ `eval_token_matches_c_single_step` + `slot_reuse_race_regression_rust`
+ `eval_prompt_matches_per_token_oracle` all pass cosine = 1.0). The
session-4 goals don't depend on Rust → C wire compat; we're
prefilling on Rust and decoding on Rust, with C only used as a
generation oracle (which the *other-direction* test
`state_load_rust_from_c_save` covers, and it passes).

## Sister failure (2026-05-13)

`resuming_prefill_after_seq_rm_matches_full_prefill` in
`tests/consecutive_eval_prompt.rs` also fails on moeflux main (both
Phase A and B1). Different test, related cause: `memory_seq_rm` +
resuming-prefill produces different argmax (`248046` baseline vs `62`
resume). This is the linear-attn recurrent-state truncation issue
documented at length in `qwen3_a3b_llama_cpp_rewind_diagnosis.md` —
moeflux's linear-attn layers lose recurrent state on partial-end
truncate.

Phase D's planned `prompt_cache_start_pos_nonzero_matches` test uses
state_save/state_load (not seq_rm) so it routes around this — but
worth confirming during Phase D implementation that the state path
is actually clean.

## To investigate later

- Bisect across the session 1-3 batched-prefill commits + the
  session 4 Phase A refactor to find which one regressed it. Likely
  candidate: 48c8562 (the rename) or one of the snapshot v2
  changes.
- Compare what Rust's `state_save` writes vs what C's `state_load`
  expects at the byte level. The `state_round_trip_rust` test
  passing means the Rust round-trip is internally consistent, so
  the drift is C-vs-Rust schema.
- If wire compat is needed for downstream use (e.g., Council
  workflow that saves on Rust, decodes on C), this becomes load-
  bearing. Until then, accept the failure and gate downstream
  callers to single-implementation snapshots.
