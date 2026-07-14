# Pre-publish validation session (dedicated, next on this repo)

2026-06-12: v0.8.0 publish is technically unblocked (#24 fixed &
smoke-verified, #25 fixed, dry-run passes) but Mike wants a dedicated
validation session before `cargo publish` — long-unrun tests, multiple
models, blallama. Expected to balloon; that's fine, it's the session's
whole job.

## Checklist

**2026-07-14: feature freeze.** Mike: stop adding to 0.8.0 and publish.
Interface tweaks and condensing the examples are 0.9→1.0 work; no new
model support planned (but "you never know").

- [x] Full `cargo test` (default features) — 363/363 unignored green
      (`just test`).
- [x] Long-running tests: the whole `#[ignore]`'d sweep (`just test
      full`) green, incl. the seven that were red at session start (see
      `eog_is_not_eos_plus_eot`).
- [x] **At least three models.** Used: Qwen3.6-35B-A3B (GGUF +
      moeflux MLX artifacts), gpt-oss-20b-UD-Q8_K_XL, gemma-4-31B-it-qat.
- [x] `cross_backend.rs` — RUN, and rewritten (teacher-forced; the old
      self-driven-trajectory metric was measuring chaos). 29/29 decisive
      argmax, worst-step mass recall 0.979. Also added `just test
      moeflux`, without which nothing could *reach* these suites —
      neither `just test` nor `just test full` sees them.
- [x] **The June blocker is CLOSED.** `partial_hit_output_matches_
      fresh_session` (the moeflux batched-prefill GPU-KV-mirror bug) now
      PASSES on the `=0.1.0-pre.4` pin. No 0.8.1 pin-bump caveat needed;
      the moeflux feature can ship.
- [ ] **`blallama.rs`** — must test. (Binary/serving path.) Still the
      biggest untested surface, and the one users actually hit.
- [ ] Misanthropic examples as a test harness against the local
      server: the non-streaming ones SHOULD run as-is; the
      streaming-API ones are expected to fail until issue #26
      (stream `misanthropic::stream::Event`) lands in 0.9. Don't
      chase streaming failures — they're known-out-of-scope.
- [ ] Examples sweep: strawberry, chat_repl, grammar_fuzz,
      dump_template, inspect_prompt (`just example NAME` runs each with
      the right features + target dir). `whodunit` ran green — but took
      ~15 min, see the perf note below.
- [ ] Pre-publish hygiene: README, version metadata, `cargo package`
      file list, doc build zero-warnings, and date the CHANGELOG's
      `## [0.8.0] — Unreleased`.
- [ ] Then: `cargo publish` on Mike's go, tag.

## Known-and-accepted going into publish

- **`whodunit` takes ~15 min** (Qwen3.6-35B, JSON grammar, n_ctx 8192).
  NOT runaway generation — Mike localized the time to *sort*. Suspect a
  full 248k-candidate sort per token, possibly re-sorted per chain stage
  or per grammar-rejection fallback. Deliberately deferred: it is a perf
  issue, not a correctness one, and it wants its own session. Do not
  block publish on it; do not "fix" it by lowering `n_ctx`.
- Two warnings under the `moeflux` feature (`log_moeflux_prefetch` dead,
  `WorkerOutput.gather_id` unread). Left alone on purpose — the first
  looks like disconnected instrumentation, and deleting telemetry to
  silence a warning loses information. Ask before touching.

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
