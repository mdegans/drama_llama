# Future work — CallOutcome could re-use `misanthropic::response::Message`

**Captured 2026-04-28** during Phase 7 prefix-cache rework, when
`Session::run_call` was already returning `CallOutcome { blocks,
prompt_tokens, cache_read_tokens, generated_tokens, stop_reason,
stop_sequence }`.

## Idea

`CallOutcome` is a drama_llama-internal aggregate built up by
`Session::run_call`. Most of its fields map 1:1 onto
`misanthropic::response::Message`:

- `blocks` → `Message::content` (already a `Content`/`Vec<Block>`).
- `cache_read_tokens` / `prompt_tokens` / `generated_tokens` →
  `Message::usage` (already a `Usage` shape we populate elsewhere via
  `Self::make_usage`).
- `stop_reason` / `stop_sequence` → `Message::stop_reason` /
  `Message::stop_sequence`.

If we adopt `misanthropic::response::Message` as the internal carrier
instead of `CallOutcome`:
- `complete_response` becomes a near-identity transform.
- `complete` / `complete_blocks` / `complete_text` keep their
  shape-narrowed views with cheaper conversions.
- One fewer drama_llama-internal type to maintain.

## Why now (or rather, why not now)

Mike flagged this in passing during the Phase 7 implementation. Not
in scope for the prefix-cache fix — `CallOutcome` works as-is and
the API churn would distract from landing the lossless-rewind
behavior. Worth picking up after Phase 7 lands and the rest of the
moeflux RIIR settles.

## What to check before doing it

- Whether `misanthropic::response::Message` already carries
  everything we need or if we'd lose a field.
- Public API impact: `complete_response` returns the
  misanthropic Message directly today — does the conversion at the
  Session boundary still feel clean if `CallOutcome` becomes the
  same type?
- Whether downstream consumers (Weave, Agora reactor) destructure
  CallOutcome anywhere that would break.
