# Future work: blallama as misanthropic's local test server

Mike's idea (2026-07-17, during the Transport arc): point misanthropic's
ignored integration tests at blallama instead of the live API, so the wire
contract gets exercised on every run rather than ~monthly at ~$1/run. "Not a
perfect test. It *does* test blallama, however."

## Honest coverage assessment

Tests well (both directions of the contract):
- request serialization → response deserialization over real HTTP
- turn-order/wire-legality, usage fields incl. cache accounting,
  tool-call round-trips, `Message::id` correlation
- **blallama's Anthropic-API emulation itself** — which serves Agora, so
  drift here is a real bug this would catch

Cannot cover (Anthropic-hosted behavior blallama doesn't emulate):
- server tools (web_search/web_fetch/code_execution/PTC), batch API,
  thinking signatures, refusals/StopDetails

So: complements, never replaces, the live-key run (misanthropic#137 has the
CI-cadence context: weekly/dispatch live leg is affordable).

## Practical shape

- Hosted CI is a poor fit as-is (needs a GGUF). Start as a local `just`
  target in misanthropic: boot blallama, run the client integration suite
  against `localhost`.
- Cheap middle path: a **deterministic canned/echo mode** in blallama (no
  model load) that still validates the entire wire layer — that *could* run
  in hosted CI.
- Bonus: Quirks-contract tests — assert blallama's measured behavior matches
  the profile `SessionTransport::quirks()` / agentkit's
  `EndpointVariant::Blallama` advertise (`breakpoint_after_assistant`,
  `output_config_cache_safe`). Today that mapping is asserted by hand in
  three places and verified in none.

Related: [[plan_transport_chat_examples_arc]] (the arc that produced
SessionTransport and the quirks profile).
