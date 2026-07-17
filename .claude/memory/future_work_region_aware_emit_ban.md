# Future work: region-aware emit ban (specials inside tool-call arguments)

**Status:** filed as a GitHub issue; deferred. Until it lands, tools that
relay model text between sessions (swarm mail, council docket) should
guard at `send` time: reject/bounce bodies whose text re-tokenizes
(`parse_special = true`) to any declared special, or resample with a new
seed. Example-local, removable once this fix ships.

## The incident (swarm example, 2026-07-17)

`wasp` emitted special token 248058 (`<tool_call>`) inside the JSON
string body of a `mail` tool call. The letter was delivered into `ant`'s
prompt as `Block::Text`; ingest (`check_no_special_injection`) correctly
rejected it as possible injection — and the error killed ant's `Chat`
loop. One poisoned letter = one dead agent.

## Root cause (traced 2026-07-17)

The emit ban **already runs unconditionally** on every sampled token
(`src/sample.rs:1108-1134`), grammar region or not. The gap is the ban
set's *contents*, not its timing:

- `emit_ban_set` (`src/session/mod.rs:1921-1985`) exempts any special
  whose piece appears in a dialect framing marker (`:1979`), because the
  session must emit `<tool_call>` legitimately as the frame. So the
  frame token is *never* banned, anywhere.
- The grammar can't catch it either: the matcher checks decoded piece
  **bytes**, not token identity (`accepts_chosen`,
  `src/sample/state.rs:407-450`). `<tool_call>` is legal JSON-string
  content byte-wise (`string`/`dstring`/`until` rules,
  `src/grammar_compile.rs:817-818`, `:434-437`, `:709`).
- Grammar-legal + ban-exempt ⇒ sampled and committed as the real
  special id inside an argument value.
- The lazy-grammar path is innocent: the ban block runs after the lazy
  check, always (`src/sample.rs:1054-1134`).

## Fix sketch

Ban the frame specials **only inside permissive constraint regions**
(JSON-string bodies, `until` bodies) — frames are emitted as grammar
*literals* (`src/dialect/emit.rs:208`), so this costs nothing
legitimate:

1. New predicate on the matcher stack, e.g.
   `StackState::in_permissive_region()`
   (`src/sample/grammar.rs:732`/`921`): true inside `char*` / `until`
   bodies, false at literal-frame positions.
2. Second, stricter ban set (all `special_tokens()` minus EOG, **no**
   marker exemption) built in `emit_ban_set`, plumbed as e.g.
   `SamplerConfig::banned_specials_constrained` (`src/sample.rs:95`,
   wired in `predict_options_for`, `src/session/mod.rs:1811-1817`).
3. `sample_token` (`src/sample.rs:1108-1134`) selects the stricter set
   when the active matcher is in a permissive region.

**Subtlety:** do NOT blanket-ban specials whenever
`constrained_incomplete()` — banning at frame positions forces the frame
to be emitted as multi-token bytes, which destabilizes the prefix cache
(existing warning at `src/session/mod.rs:2246-2268`). The region
predicate is the load-bearing piece.

**Repetition-penalty parallel (Mike):** rep-penalty is already switched
off under similar structural rules — that machinery may be reusable for
the region signal. Check before building a new predicate from scratch.

## Path B: byte-spelled markers (unfixable at the sampler)

Even with the id ban, a model can spell `<tool_call>` as ordinary
multi-token bytes inside a string. Ingest detects by *re-tokenizing
surfaced text* with `parse_special = true`
(`find_injected_special_in_prompt`, `src/session/mod.rs:711-742`), so
the spelled form collapses back to the special id and rejects
identically. We have encountered byte-spelling before. Consequences:

- The sampler fix reduces frequency (path A) but cannot eliminate the
  class; relay boundaries (tools passing text between sessions) need
  their own policy regardless.
- Worth considering alongside the fix: should ingest offer a
  *sanitize/escape* mode for relay use-cases, so a hostile or clumsy
  letter degrades instead of killing the recipient's loop? Severity
  amplifier today is fatal-on-ingest, not the emission itself.

## Regression tests to write with the fix

- Grammar-constrained tool call cannot emit any declared special id
  inside a string argument (per dialect: Hermes/Qwen `<tool_call>`,
  others' frames).
- Frame literals still emit as single tokens (prefix-cache stability —
  assert on the committed token ids, not just the text).
- Lazy-grammar fast path and full-mask fallback both respect the
  stricter set.
- `with_emit_specials_ban(false)` still opts out of everything
  (Qwen-VL grounding markers).
