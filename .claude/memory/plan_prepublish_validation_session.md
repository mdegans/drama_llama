# Pre-publish validation session (dedicated, next on this repo)

2026-06-12: v0.8.0 publish is technically unblocked (#24 fixed &
smoke-verified, #25 fixed, dry-run passes) but Mike wants a dedicated
validation session before `cargo publish` — long-unrun tests, multiple
models, blallama. Expected to balloon; that's fine, it's the session's
whole job.

## Checklist

- [ ] Full `cargo test` (default features) — quick re-green.
- [ ] Long-running tests: `cargo test -- --include-ignored` with
      `models/model.gguf` valid. These haven't run in a while.
- [ ] **At least three models** through the examples / ignored tests.
      Candidates on hand: Qwen3.6-35B-A3B (gguf + moeflux artifacts),
      whatever `models/model.gguf` points at, cogito — pick per
      what's mounted; record which were used.
- [ ] `cross_backend.rs` #[ignore] run (both backends + 35B
      artifacts) — newly compile-fixed (e0d2d34), hasn't *run* since
      before the backend split.
- [ ] **`blallama.rs`** — must test. (Binary/serving path.)
- [ ] Misanthropic examples as a test harness against the local
      server: the non-streaming ones SHOULD run as-is; the
      streaming-API ones are expected to fail until issue #26
      (stream `misanthropic::stream::Event`) lands in 0.9. Don't
      chase streaming failures — they're known-out-of-scope.
- [ ] Examples sweep: strawberry, whodunit, chat_repl, grammar_fuzz,
      dump_template, inspect_prompt — each with required features.
- [ ] Pre-publish hygiene: README, version metadata, `cargo package`
      file list, doc build zero-warnings.
- [ ] Then: `cargo publish` on Mike's go, tag.

## Status 2026-06-12 evening (mid-session)

- 1a + full serial ignored sweep (1b3): **green** — 277 lib tests +
  all integration suites, after fixing 14 stale/parallelism failures
  (see commits db42d1f, 7092a98, dbf5751, e3c9277, 0b28469).
- moeflux+llama-cpp combo bitrot fixed (`LlamaCppSession` alias,
  misanthropic-alpha.2 port of moeflux_session_pollution) — 707af24.
- moeflux smoke + three_consecutive: green.
- **`partial_hit_output_matches_fresh_session`: RED, real bug, in
  moeflux not drama_llama.** Batched prefill never populates the
  gpu_kv mirrors that oracle decode's GPU SDPA reads at kv_len ≥ 32 —
  fresh sessions decode against a zeroed prompt region on full-attn
  layers. Checkpoint/restore exonerated. Full chain + acceptance
  tests: moeflux f63352f,
  `~/Projects/moeflux/.claude/memory/gpu_kv_mirror_not_populated_by_batched_prefill.md`.
- moeflux_coherence + cross_backend deferred: results are moot until
  the mirror fix (moeflux pre.4); a17b run likewise.
- Recommendation: publish decision is unaffected on the llama-cpp
  side (default features fully green). moeflux feature ships pinned
  to pre.3 which already has the bug; fix lands as pre.4 + a
  drama_llama 0.8.1 pin bump.

## Process notes

- Model runs from Mike's terminal, not the Bash tool
  (feedback_gpu_launch_from_claude_code). Batch the commands so Mike
  can run them in sequence and paste results.
- Delegate boring/mechanical legs (log triage, checklist sweeps) to
  subagents — Haiku/Sonnet fine, Mike's standing permission
  (CLAUDE.md, Claude's Notes, 2026-06-12).
- Triage discipline applies: every failure classified (fix /
  explicit-ignore / pre-existing memo), never silently filed.
