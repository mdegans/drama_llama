# Future work: benchmark, profile, and optimize structured generation

**Date:** 2026-07-22. **Status:** open, deliberately deferred past
publish. Mike: "I would like to benchmark that, profile, and optimize,
but later."

## The gap

**`bench.py` does not exercise the grammar path at all.** It sends
`model` / `messages` / `max_tokens` and nothing else (`bench.py:181`),
so every tok/s number this repo has recorded is the *unconstrained*
path. Structured output — the thing Agora actually runs, and the thing
`whodunit` / `vote_intent` / `few_shot_triage` demonstrate — has no
repeatable benchmark.

Not *entirely* unmeasured: the phase-split + DFA arc got the grammar
path to ~17.6 tok/s (see `project_tok_s_targets` in personal memory).
But that was a one-off during optimization, not a harness anyone can
re-run to detect a regression. There is no structured equivalent of
`bench.py --model a3b`.

## The lead worth chasing: ~10% lazy-grammar miss rate

Mike, 2026-07-22: the lazy-grammar stats "sometimes indicate something
like 10% miss rate, which makes me wonder about other things."

Why that number matters. [#28] made grammar checking lazy: sample
first, verify the one sampled token (`O(piece)`), and only on rejection
fall back to a full `O(vocab)` mask. The whole optimization is
predicated on rejections being *rare*. At a 10% miss rate, one token in
ten pays the full masked path — filter plus a re-run of the sampler
chain — so the effective cost is nothing like the fast-path cost, and
"lazy grammar is fast" stops being the right mental model.

**Grammar is now the slow part of sampling** (Mike, same conversation)
— "but much less than before". Note this is *after* the `TopK`-ahead-of
-`LocallyTypical` fix; do not confuse the two. See
[`future_work_seeded_vs_tuned_sampling.md`](future_work_seeded_vs_tuned_sampling.md)
for why `LocallyTypical` is no longer the cost it once was.

## The counters already exist — no instrumentation work needed

Collection is **env-gated**, off by default:

- **`DRAMA_LLAMA_GRAMMAR_STATS`** — set to anything non-empty and not
  `"0"` to enable (`grammar.rs:1849`).
- **`DRAMA_LLAMA_DFA_CACHE=0`** — disables the lazy-DFA cache and falls
  back to per-candidate clone-and-walk (`grammar.rs:1490`). Not a stats
  knob but an **experiment axis**: this is the A/B for "is the DFA worth
  it on `.+`-heavy JSON", which Agora has reason to care about.

Two traps for a harness, both the silent kind:

1. **Both flags are `OnceLock`-cached at first access.** Setting or
   changing them mid-process does nothing. They must be in the
   environment before the run starts — and `bench.py` launches blallama
   as a subprocess, so the harness has to pass them through, not just
   set them for itself.
2. **`grammar_stats_snapshot()` returns zeros when collection is
   disabled**, not an error. A harness that forgets the env var reports
   a clean sheet of zeros that looks like "no fallbacks ever" rather
   than "not measured". Assert `calls > 0` before believing any of it.

`GrammarStats` (`src/sample/grammar.rs:1745`), behind
`grammar_stats_enabled()` / `grammar_stats_reset()` /
`grammar_stats_snapshot()`:

| question | counters |
|---|---|
| miss rate | `lazy_fallbacks / lazy_checks` (and `lazy_hits`) |
| cost split: verify vs. fallback-filter | `check_us_sum` vs `filter_us_sum` |
| worst-case stall | `check_us_max`, `filter_us_max` |
| is the bitmap prefilter earning its keep | `candidates_in` → `candidates_bitmap_pass` → `candidates_final_pass` |
| DFA cache health | `dfa_transition_hits/misses`, `dfa_bitmap_hits/misses`, `dfa_states` |
| grammar complexity | `stacks_in_max`, `depth_max_max` |

In lazy mode the filter counters measure **fallback invocations only**,
which is exactly the decomposition this needs.

## "Makes me wonder about other things"

Mike's phrasing, and worth preserving as the open part. He did not say
what the other things are; the following are **my hypotheses, not his**,
offered as things to check rather than conclusions:

- **A miss rate should be bursty, not uniform.** Rejections ought to
  cluster at structural boundaries (right after `{`, at a key→value
  transition) where the model wants prose and the grammar wants syntax,
  and fall to ~0 inside permissive free regions where nearly everything
  is legal. If the 10% turns out to be *evenly spread*, the cost model
  is wrong somewhere and that is the interesting finding.
- **A high miss rate is a correctness smell, not only a cost.** It means
  the model persistently wants to emit something the grammar forbids —
  which can mean the grammar over-constrains relative to the dialect, or
  the schema does not match what the model was trained to produce. That
  is the same failure shape as the reserved-token loop
  ([`grammar_reserved_token_loop.md`](grammar_reserved_token_loop.md)),
  found from the other direction.
- **Check interaction with the modes ahead of the grammar.** `Deny`,
  `banned_specials`, and a hard `TopK{20}` all narrow the set before the
  grammar sees it. A tight `TopK` could plausibly *raise* the miss rate
  by discarding the legal continuations before the check runs — the
  seeded chain (k=20) prunes far harder than the old default (k=1024),
  and the miss-rate observation post-dates that change.

That last one is cheap to falsify and would be worth doing first: run
the same structured prompt under both chains and compare
`lazy_fallbacks / lazy_checks`.

## Shape of the work

1. A structured mode for `bench.py` (a fixed schema + prompt, sampling
   fields still unsent so the sidecar stays the knob), reporting tok/s
   *and* a `GrammarStats` snapshot — with `DRAMA_LLAMA_GRAMMAR_STATS`
   passed into the blallama subprocess and a `calls > 0` assertion so a
   misconfigured run fails loudly instead of reporting zeros. Record the
   chain with the number, same discipline as `bench.py:47-62`.
   `DRAMA_LLAMA_DFA_CACHE=0` is a second run, not a second column.
2. Profile it. Do not assume the answer; the last two sampler-perf
   assumptions in this repo were both wrong in the same direction
   (something believed slow was already fine).
3. Only then optimize.

Deliberately not before publish: this is perf, not correctness, and the
structured path is *correct* today — the grammar suites and
`grammar_fuzz` cover that.

[#28]: https://github.com/mdegans/drama_llama/issues/28
