# Plan: grammar canonicity — fuzz out masking-forced splits

**Status: designed, not started** (2026-07-28, follow-up to #96; issue
filed the same day). Read before touching the GBNF emitter
(`schema_to_gbnf`, the dialect grammar builders) or before widening any
cache-lookup tolerance to "help" tool-call turns hit.

## Why this is the lever now

#96's fix made `slot_l_hit` take the max of its two lookups, so the
post-generation tip anchors whenever the emission **re-tokenizes
canonically**. Both lookups refuse non-canonical emission and must keep
doing so:

- the LCP walk stops at the first divergent entry;
- the hash path refuses disagreeing coordinates (#91 — restore
  addresses the KV's tokenization, the suffix indexes the new one;
  reuse across a count mismatch skips real tokens. **Do not weaken.**)

So the residual tool-call-turn miss rate is exactly the rate at which
grammars force non-canonical emission. Mike's policy line (#96
session): a grammar that forces non-canonical emission is a **grammar
bug** — the grammar should constrain generation such that emission is
canonical, accepting we can never hit 100%.

## The two causes; only one is ours

1. **Model-chosen splits** — rare under greedy, not our bug.
2. **Masking-forced splits** — the canonical merged token would
   overshoot into bytes the grammar forbids, gets masked, and the model
   is forced into a split the tokenizer reads differently. #91's
   measured case: a schema grammar forcing bare `"` where the tokenizer
   merges `"word` (2322 bytes, 616 cached entries vs 613 re-read).
   Mechanically discoverable; emitter-fixable per construct.

## Harness: a canonicity oracle on the existing fuzzer

`examples/grammar_fuzz.rs` has the chassis (schema gen, grammar walker,
oracle classes 1–8, corpus + replay) and today no tokenizer notion at
all. Add:

- walk grammar → accepted byte string `s` (existing machinery);
- `T = tokenize(s)` against a real vocab (tokenizer only, CPU, no
  inference);
- replay `T` step-by-step through the grammar's token filter; each step
  whose canonical next token is **masked** is a finding — a boundary
  where this grammar forces non-canonical emission on this vocab.

Caveat from the fuzzer runbook: the walker is ASCII-only (multi-byte
UTF-8 first-byte bug in the matcher's pending buffer) — a canonicity
oracle over real text needs that scoped in or worked around.

Second finder, zero new code: the #96 per-model suite
(`tests/common/tip.rs`, tool-rounds scenario) is the model-as-fuzzer.
Any red there is a canonicity finding by policy, not a flake.

## Success metric

Tool-call-turn tip hit rate in seed runs: `probe_tip.py` exit code
(agora-agents), blallama's `prefix-reuse:` debug lines — #96 added the
"tip lost the pick" line, so losses are finally visible in logs.

## Links

- #96 (composition fix, regression suite, policy), #91 (refusal
  semantics), #30 (the emitter this hardens).
- [[grammar_fuzzer_runbook]] — existing fuzzer operation.
- [[plan_template_ownership]] — the other half of hit rate: byte-stable
  re-renders. Mostly landed; rung-2 witness in `session_mistral4.rs`.
