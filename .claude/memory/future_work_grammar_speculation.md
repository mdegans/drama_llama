---
name: Future work — grammar speculation
description: Overlap Metal decode with CPU grammar filter by speculatively decoding the top-1 token and rolling back the KV cache on rejection. Leverages the ~99% greedy acceptance rate observed in practice.
type: project
---

Status: unscheduled, captured for long-term. Deferred past v0.7 tag.

## Motivating observation

Empirically (user's observation, 2026-04-22): without grammar,
~99% of the time the greedy top-1 token is exactly the one the
grammar would accept anyway. But that 1% breaks the whole
generation — an invalid token produces malformed JSON /
malformed tool call, and downstream parsing fails.

This asymmetry — "mostly right, rare but catastrophic on miss"
— is *exactly* the shape where speculation pays off. Do the
fast thing optimistically, verify, roll back on miss.

## Mechanism

Current loop (synchronous):

```
loop {
    decode()                       // ~50 ms on M2 Max
    candidates = get_logits()
    filtered   = grammar_filter()  // ~40 ms post-DFA on JSON
    token      = sample(filtered)
    commit(token)                  // decode next
}
```

Speculative loop:

```
loop {
    candidates = get_logits()
    top1       = sample(candidates)              // without grammar

    // fire these two in parallel —
    let decode_next  = spawn(decode(top1))       // GPU: ~50 ms
    let filter_fut   = spawn(grammar_filter())   // CPU: ~40 ms
    let (decode_done, filtered) = join(decode_next, filter_fut)

    if filtered.contains(top1) {
        commit(top1)                  // decode was free
    } else {
        rollback_kv(n_cur - 1)        // cheap: one memory_seq_rm
        token = sample(filtered)
        decode(token)                 // now pays decode cost
        commit(token)
    }
}
```

GPU decode (Metal) and CPU filter (rayon) are independent
resources today — they'd genuinely overlap.

## Expected payoff

If acceptance rate p ≈ 0.99 as the observation suggests, and
decode and filter cost roughly D and F respectively:

- Current cost per token: D + F ≈ 90 ms
- Speculative cost per token: max(D, F) + (1 - p) × (rollback + D)
  ≈ 50 ms + 0.01 × 52 ms ≈ 50.5 ms
- Steady-state gain: ~43% on the JSON phase

Combined with shipped phase-split (phase 1 unconstrained), the
aggregate whodunit number could approach the unconstrained
ceiling (~20 tok/s).

At lower acceptance rates (say p ≈ 0.8 for a tight schema with
many structural branches) the win is smaller but still positive:
~25%.

## Feasibility concerns

1. **Loop inversion.** Today's `CandidatePredictor::next` is
   synchronous: decode → candidates. Speculation needs
   `spawn decode → concurrently filter → join`. Options:
   - Tokio task pair (drags tokio into core — unwanted).
   - Rayon thread pool (awkward for Metal calls but works).
   - Dedicated 1-thread-per-resource with a mpsc channel.
   None are disasters; all are real surgery to the predictor
   stack. Would likely land as a new `SpeculativePredictor`
   alongside `TokenPredictor` rather than replacing it.

2. **KV rollback cost.** `Engine::memory_seq_rm(seq_id, n..n+1)`
   is cheap in llama.cpp — a metadata update, not a recompute.
   Confirmed by the prefix-cache work already in v0.7.

3. **Measurement first.** Before any code: instrument what the
   actual greedy acceptance rate is on real workloads. The 99%
   figure is an anecdotal observation; if the true rate is 70%
   on a tight schema, the gain calculation changes. Could add a
   stat counter to `grammar_filter` that records
   "would-have-picked top-1" separately from "top-1 passes
   filter".

4. **Correctness.** Speculation must not change observable
   output relative to non-speculative grammar-filtered sampling.
   The filter's decision is authoritative; the speculation is
   purely a scheduling optimization. Test with deterministic
   seeds to confirm identical token streams speculative vs
   serial.

## Why defer

- 17.6 tok/s already past the 15 tok/s target for Agora reactor.
- Loop-inversion surgery is invasive; touches every Predictor
  layer.
- Measure acceptance rate first — the gain is entirely gated on
  it.

Revisit when: a downstream consumer is pushing for >20 tok/s, or
we observe on real Agora reactor traces that `grammar_filter`
time is dominating a specific workload despite phase-split being
on.
