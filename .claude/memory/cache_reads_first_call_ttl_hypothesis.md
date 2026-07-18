# Cache-reads tripwire: first-call snapshots get the 5-minute default TTL

**2026-07-18, during the exit-interview sidequest.** Evidence and a
strong hypothesis for the queued payroll/cache-reads diagnosis. The
council tripwire (`assert_cache_hit`) fired once in two runs; this
memo is the state of the investigation so the dedicated session
doesn't start cold.

## The event (run 1, car-wash petition, one round)

Reaction-phase reads were exact full-prefix reuses for three advisors
and zero for the fourth:

| seat        | filing (in+out) | reaction `cache r` | expected |
|-------------|-----------------|--------------------|----------|
| artist      | 536+547 = 1083  | 1082 ✓             | ~1083    |
| philosopher | 544+2435 = 2979 | 2978 ✓             | ~2979    |
| engineer    | 532+597 = 1129  | 1128 ✓             | ~1129    |
| lawyer      | 529+543 = 1072  | **0** ☠            | ~1072    |

Lawyer is the last reactor: its idle gap (filing end → reaction
start) spanned the jester rebuttal (753 out over a 4578-token prefill)
plus three reactions (2129+1327+1190 out) — roughly 8–12 minutes.
Engineer's gap was only ~40 s of decode shorter and survived. Run 2
had much shorter filings (everyone's gaps shrank) and completed clean
with lawyer reading 880 ≈ 529+352. Longest-idle-seat-only failure,
moving with the timing, is a TTL cliff, not corruption.

## The hypothesis (plumbing traced, not yet instrumented)

- `Slot::expired` (`src/session/mod.rs` ~603): a breakpoint expires
  when `now - last_used > ttl_duration(bp.ttl)`; `last_used`
  refreshes on read/write.
- The end-of-generation tip snapshot's TTL comes from `tip_ttl`
  (~1059): **max of the call's rendered marker TTLs, defaulting to
  FiveMinutes on a markerless call.**
- Council's `file_call` applies `cache_windowed_with(2, one_hour())`
  only *after* a completion returns — so every seat's **first** call
  renders markerless, and its tip snapshot carries the 5-minute
  default. The one-hour intent lands one call too late to protect the
  first idle window. A markerless first-call slot holds *only* that
  tip, so expiry evicts the slot wholesale (`sweep_expired`) →
  `cache r: 0`, full re-prefill, tripwire.

## Candidate fixes (decide in the diagnosis session)

1. **Council-side (one-liner):** mark the seat's prompt with a
   one-hour breakpoint at construction (before the first call), so
   the first render carries a marker and `tip_ttl` resolves OneHour.
   Same for the judge.
2. **Library-side:** make the markerless-call tip TTL configurable
   (session-level default TTL), or inherit the *session's* longest
   recently-seen TTL. More surface, but protects every caller with
   the same first-call pattern — the `Chat` driver's quirk path marks
   after the first assistant turn too.

Verification once fixed: rerun the one-round car-wash petition with a
long jester/reaction phase (or insert an artificial delay before the
lawyer's reaction) and watch the payroll's reaction-row reads.

Related: the exit-interview flow (`council --dump` → `chat --load`)
only archives on *graceful* adjournment; a tripwire panic loses the
dump. If interviews-after-failures become important, dump-on-error is
a follow-up (convert the tripwire assert to a typed error, or dump in
a scope guard).
