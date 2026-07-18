# Council cache debugging — landed state + next two sessions

2026-07-18 PM. In-repo copy of the approved plan (Mike: temporary plan
files get copied here on handoff), updated to landed state. Delete
sections as they resolve (delete-resolved-memos convention).

## Landed today (this session)

- **Root cause of the v2 council's zero cache reads**: the prefix cache
  was never enabled. `from_path_with_cache_slots` only shapes the
  multi-seq KV pool; the enable was a side effect of
  `SessionTransport::new` that the v2 rewrite (direct
  `complete_response`) silently lost. Fixed in the examples helper
  (`session_with_cache_slots` → `.with_prefix_cache(true)`).
- **Honest cache stats** (closes #40): `cache_creation_input_tokens`
  had been hardcoded `Some(0)` since the original caching commit
  (468f32d, 2026-04-19 — it never worked; the "used to work" memory
  was reads via the transport). Now `creation = input − read` (every
  newly decoded prompt token lands in tip/breakpoint snapshots);
  cache-off reports both counters `None`. `input_tokens` stays the
  full prompt (`read + creation == input` when cache on; switch at
  `make_usage` if API-billing-style input is ever needed).
  `complete_response` usage is now the same `Usage` recorded as
  `last_usage`, carried through `CallOutcome` (dedup — do NOT add
  `record_usage` to `complete_response_id`; `run_call` records).
- Constructor-default repetition penalty now ignores specials
  (`from_engine` injection; the sidecar/`with_*` paths already did).
- Council example: 1-hour tail markers (5-minute ephemeral TTL expired
  while other seats ran), 32k default `--n-ctx` (clap `mut_arg`, shows
  in `--help`), full per-turn counters printed, TEMPORARY
  `assert_cache_hit` (panics on zero-read follow-up — remove when
  payroll makes sense), doc trimmed. Default sampler chain gained
  top-k 1024 pre-cut before locally-typical; model.sampling.toml was
  deleted so the next load writes the new default (its old hardlink
  twin `Qwen3.6-*.sampling.toml` still holds stale contents — re-link
  or delete after eyeballing).

## Durable findings (keep)

- Two paths report `read == Some(0)` on a *genuine* hit-then-reset:
  (a) perfect-prefix identical re-send with no lower breakpoint →
  slot reset + full re-prefill (session/mod.rs, empty-suffix guard);
  (b) `restore_to` failure (checkpoint lost to LRU) → wipe. Both
  honestly re-prefill, but both would trip the council assert /
  downstream panic-on-miss. Council prompts always grow, so (a)
  shouldn't fire there.
- The cache tripwire (`DRAMA_LLAMA_CACHE_TRIPWIRE=1`) only evaluates
  *inside* cache selection — cache-off or slot-evicted states never
  reach it, so a quiet tripwire does not mean hits.
- Sweep results: moeflux `context_size()` silent-0 fallback (#41),
  moeflux silent no-op seq ops (#42). Everything else in response
  assembly is honest (stop_reason/sequence/read real; thinking-token
  details honestly `None`).

## Next session A — confirm/fix any remaining cache miss

Run the council (Mike's terminal). Expectations with honest stats:
round-1 filings `w == full prompt, r == 0`; every follow-up `r > 0`
(assert stays quiet). If the assert fires: suspects are the two
zero-report paths above, TTL (markers are 1h now — a steward pause
> 1h fires it spuriously), or a render byte-instability breaking the
prefix hash (round-trip byte-stability is the cache invariant —
check `DRAMA_LLAMA_CACHE_TRIPWIRE=1` and the tripwire's drift
diagnostics once the cache is actually on). Model-backed tests to
run first: `cargo test --features toml -- --ignored test_make_usage
test_ttl_expiry_evicts test_usage_counters_across_append_only_calls`.
When the numbers make sense: remove the temporary council assert.

## Next session B — free-region repetition penalty (#43)

After A. Region-aware suspension instead of all-or-nothing: penalty
(and stats ingestion) live only inside permissive regions (JSON string
values, until() spans) while a grammar is active. Evidence: artist
looped a paragraph verbatim ~10× inside a forced call's `analysis`
string, burned 4096 tokens to max_tokens → GrammarViolation (#36 is
the misreport half). Care points in the issue: exit delimiters stay
unpenalized, `SamplerState::step` coherence across cache resumes,
structural tokens never seed stats. Pairs with the region-aware emit
ban (#37 / future_work_region_aware_emit_ban.md).
