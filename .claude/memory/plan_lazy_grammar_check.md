---
name: Plan — lazy grammar check (sample-then-check)
description: Replace per-token O(vocab) grammar masking with an O(1) legality check of the sampled token, falling back to the full mask only on rejection. Supersedes the speculative-decode idea in future_work_grammar_speculation.md.
type: project
---

Status: plan-of-record, approved 2026-07-03. Not yet scheduled.
Canonical copy: filed as a GitHub issue (see link at bottom) so the
plan isn't tied to one machine's checkout.

Decisions (user, 2026-07-03):

- Grammar-first masking semantics are **not** load-bearing.
- **Determinism is** load-bearing: fixed seed → identical token
  stream, every run.
- Prefer removing CPU work from the critical path over overlapping
  it ("whether the CPU is going BRRRrrr" matters — thermals, power,
  and blallama multi-session contention).

## Summary

Today every constrained token pays `grammar_filter` /
`json_filter`: walk all surviving candidates' pieces through the
matcher (rayon `par_iter` + first-byte bitmap prefilter + lazy-DFA
memoization). ~40 ms/token post-DFA on JSON on M2 Max, vs ~50 ms
decode. Numbers differ on balerion (amd64 + CUDA), but the point is
architecture-independent: the filter is CPU work on the generation
critical path regardless of what device decodes.

The insight that kills the old speculative-overlap plan:
**verification ≠ mask computation.** To verify a sampled token you
don't need the legal set — you need one membership test, and that
API already exists and is non-mutating:

- `GrammarState::accepts_bytes(&self, &[u8]) -> bool`
  (`src/sample/grammar.rs:794`)
- `JsonState::accepts_bytes` (`src/sample/json.rs:89`)

New flow: sample **without** the grammar modes in the chain, then
check just the chosen token's piece with `accepts_bytes` — O(chosen
token's bytes), microseconds. Legal (~99% empirically, to be
measured)? Advance matcher state and commit. Illegal? Restore
RNG + mu snapshots and rerun **exactly today's masked path** from a
pre-fold clone of the candidates.

Per-token cost: `D + µs` common case, vs `D + F` today. No threads,
no loop inversion, no KV rewind — the entire change lives inside
`sample_token` (`src/sample.rs`). In particular it is immune to the
one-token `memory_seq_rm` unreliability documented in
`qwen3_a3b_llama_cpp_rewind_diagnosis.md` (llama.cpp rejects partial
seq_rm for recurrent/hybrid models) and
`blallama_session_state_pollution.md` (moeflux C partial truncation
is lossy by design).

## Why not the original speculative overlap

The 2026-04-22 sketch (`future_work_grammar_speculation.md`)
speculatively decoded the unconstrained token and rolled back the KV
cache on rejection. Ruled out / dominated because:

1. **Rewind is not reliable across backends.** Discovered after the
   sketch was written: llama.cpp hard-rejects partial `seq_rm` for
   models with a recurrent component (Qwen3.5-A3B), and moeflux's C
   implementation loses linear-attention state on partial truncation
   by design. Only pure transformers on llama.cpp and the
   moeflux-Rust `checkpoint_pos`/`restore_to` path rewind safely.
2. **Worse cost even when rewind works**: `max(D, F) + miss × (rewind
   + D)` with real predictor-loop surgery, vs `D + µs` with none.
3. Under temperature sampling, verify-by-token-equality misses more
   often than "token is illegal": any grammar-removed probability
   mass ahead of the drawn point in the CDF remaps the same draw to
   a different token.

A middle option ("Design C": precompute the next position's mask on
rayon while decode runs, apply on arrival) preserves *exact*
grammar-first semantics at `max(D, F)` cost with no rewind. Kept in
reserve if bit-compat with the masked distribution ever becomes
load-bearing; not scheduled.

## Semantics change (deliberate)

Accept-if-legal ≠ grammar-first masking:

- Masked-first: illegal mass is removed before truncation samplers
  run, then renormalized.
- Lazy: the unconstrained winner keeps its seat if legal; only on
  rejection do we resample from the masked distribution.

Formally `P_lazy(t) = P_unc(t)·[t legal] + P_unc(illegal set) ·
P_masked(t)`. llama.cpp ships this same accept-if-legal shape as its
standard grammar optimization. Fully deterministic: fixed seed →
same draw sequence → same stream. But lazy streams **differ from
masked-first streams by design** — any test asserting equality
across the two modes is invalid. New invariants:

1. Output is always grammar-legal / parses.
2. Same seed, two runs → byte-identical output.
3. `lazy_grammar = false` reproduces current v0.8.0 streams
   bit-exact (the masked path is untouched — it *is* the fallback).

## Mechanism

All in `sample_token` (`src/sample.rs:1128–1240` on v0.8.0), gated
by new `SampleOptions::lazy_grammar: bool` (serde default `false`):

1. Repetition penalty unchanged — runs once, mutates `freq_map`,
   *before* any cloning.
2. If `lazy` and `modes` contains `Grammar`/`Json`: snapshot `rng`
   (`Xoroshiro128` is `Copy` — `[u64; 2]`) and `mu`
   (`Option<f32>`), and clone the post-repetition `Candidates`.
   Requires making `Candidates` `Clone` — fields are
   `Sorted` + `Option<NonZeroUsize>` + `Vec<TokenData>`, all
   trivially cloneable (`src/candidates.rs:199`). ~3 MB memcpy at
   250k vocab; negligible vs decode. Clone only on this path.
3. **Fast path**: run the fold skipping the `Grammar`/`Json` arms.
   `Deny`, truncation samplers, mirostat all still run — the
   reserved-token protection (`grammar_reserved_token_loop.md`) is
   unchanged. `choose_candidate` as today.
4. **Check**: `token_to_piece_ref(chosen)` once; for every
   `Grammar`/`Json` mode in the chain, `accepts_bytes(piece)`. All
   accept → `json::advance_all` + `grammar::advance_all` (they
   iterate `opts.modes`, which still contains the grammar modes —
   unchanged, `src/sample.rs:1232-1233`) → return.
   Empty pieces are trivially accepted — same per-candidate behavior
   as `grammar_filter` today.
5. **Fallback** (any rejection): restore `rng` + `mu` from
   snapshots, rerun the full fold over the cloned candidates with
   *all* modes in original order — literally today's path, including
   `grammar_filter`'s empty-kept → `is_complete` → reset →
   forced-EOS logic (`src/sample/grammar.rs:1872–1882`). Then
   `advance_all`, return.
6. **EOS / termination**: when the grammar completes and nothing
   legal remains, the model's pick gets rejected → fallback forces
   EOS exactly as masked mode does. Costs one `F` at stream end.
   If a model's EOS piece is empty, the fast path accepts it
   directly — also consistent with masked mode, which keeps
   empty-piece tokens.
7. **RNG accounting**: the fallback restores the pre-fold RNG, so
   exactly one draw is consumed per emitted token on either path —
   the same `u` decides both attempts. Keeps the draw stream aligned
   and the whole thing deterministic.

Deferred grammar (thought/JSON phase-split) composes untouched:
promotion pushes the grammar mode into `opts.modes` mid-stream
(`src/predictor.rs:668–704`); lazy logic sees it on the next token.
**No predictor changes at all.**

## Stats / measurement

Extend `GrammarStats` (`src/sample/grammar.rs:1552`, env
`DRAMA_LLAMA_GRAMMAR_STATS`): `lazy_checks`, `lazy_hits`,
`lazy_fallbacks`, `check_us_sum/max`. The existing filter counters
then measure fallback-only invocations. This finally turns the
anecdotal "~99% acceptance" (2026-04-22 observation) into a measured
number — in lazy mode, the measurement *is* the feature's hit rate.

## Phases

1. `Candidates: Clone` + `sample_token` restructure + flag + stats.
   Unit tests (toy grammar): legal-token fast path advances matcher
   state identically to the masked path; fallback resamples from the
   masked set; rng/mu restored exactly; one draw per token both
   paths.
2. Validation on a real model (whodunit example already has
   `--no-grammar` baseline): (a) JSON parses across N seeds;
   (b) determinism — same seed twice → identical bytes; (c)
   `lazy = false` unchanged vs v0.8.0; (d) tok/s + stats comparison
   masked vs lazy, on mac (Metal) and balerion (CUDA).
3. Flip default to `lazy = true` once validated; document the
   semantics difference on `SampleOptions`; keep the masked path
   selectable indefinitely (it is the fallback implementation — zero
   dead code).

## Expected payoff

M2 Max JSON phase: ~90 ms → ~50 ms/token (~43%, matching the old
speculation estimate — but with no threads, no rewind, and the CPU
idle instead of pegged). Frees the rayon pool every token, which
matters for blallama multi-session serving and for power/thermal
headroom. On balerion the absolute numbers differ but `F` drops off
the critical path identically.

## Files

- `src/sample.rs` — `SampleOptions` field, `sample_token`
  restructure (essentially the whole change)
- `src/candidates.rs` — derive/impl `Clone`
- `src/sample/grammar.rs`, `src/sample/json.rs` — stats counters
- `src/predictor.rs` — none

## Links

- Supersedes: `future_work_grammar_speculation.md`
- Rewind unreliability: `qwen3_a3b_llama_cpp_rewind_diagnosis.md`,
  `blallama_session_state_pollution.md`
- Reserved-token Deny motivation: `grammar_reserved_token_loop.md`
- GitHub issue: https://github.com/mdegans/drama_llama/issues/28
