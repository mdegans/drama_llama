# Baked chat templates

Shipped artifacts, not test fixtures: `src/baked.rs` embeds these via
`include_str!` and `Session` applies them by the loading ladder
documented there (sidecar → baked → embedded-with-warning; issue #88).
Each supported model contributes a pair — the exact stock template
dumped from the GGUF we validated against (the byte-equality detection
key) and drama_llama's cache-stable replacement. Round-trip pins live
in `tests/dialect_roundtrip.rs`; a change to any file here must keep
that suite green.

`gemma4-gguf.jinja` is dumped from the Gemma-4 31B IT Unsloth GGUF
(`tokenizer.chat_template`). It is a lightly patched superset of the
upstream-vendored `google-gemma-4-31B-it.jinja` (content-parts
support, a `has_content` turn-close fix); the tool-call rendering
path is byte-identical, so upstream's pinned expectations
(`tests/test-chat.cpp`, "Google Gemma 4" section) apply to both.

`gemma4-cache-stable.jinja` is drama_llama's cache-stability patch of
`gemma4-gguf.jinja`: model turns re-render the thinking channel the
model actually generated against (real reasoning, gated by
`preserve_thinking` for aged turns; the empty
`<|channel>thought\n<channel|>` scaffold otherwise), so the KV cache
stays a byte prefix of the next render across tool turns. Everything
outside the thinking-channel block is byte-identical to
`gemma4-gguf.jinja`.

`gptoss-gguf.jinja` is dumped from the gpt-oss-20b Unsloth GGUF
(`tokenizer.chat_template`, Apache 2.0 per its own footer) — the
Harmony template we actually serve.

`gptoss-cache-stable.jinja` is drama_llama's cache-stability patch of
`gptoss-gguf.jinja` (#30 Phase G): the macro section (system /
developer / TypeScript tool namespace) is byte-identical to stock;
the message loop is rewritten so `render(parse(emission))` reproduces
the emission — analysis (CoT) renders on every reasoning turn (gated
by `preserve_thinking`, drama_llama's default), tool calls render in
the model's trained channel-header shape
(`<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>`)
for ALL `tool_calls` (stock renders only the first, in the role-header
re-ingest shape), pre-call prose renders as a causal commentary
preamble, and tool responses render by forward-scan with
`tool_call_id`-resolved names.
The `<|return|>`/`<|end|>` re-ingest rewrite (upstream issue #15417)
costs nothing: the sampled EOG is never committed to KV, and the
session's auto-tip records the CANONICAL close token from the
byte-stable re-render (`compute_tip_extension`), so the next call's
LCP walks through the rewritten `<|end|>` and splices at the tip.

`cogito-gguf.jinja` is dumped from the cogito-32b GGUF
(`tokenizer.chat_template`, via `scripts/gguf_template.py`) —
byte-identical to the 14B fixture
(`tests/fixtures/cogito_14b_template.jinja`), so results transfer
across both sizes.

`cogito-cache-stable.jinja` is drama_llama's cache-stability patch of
`cogito-gguf.jinja` (#88 phase 2), and the smallest of the set: one
filter swap, `tool_call.arguments | tojson` → `| json_dumps`. Stock
`tojson` re-renders arguments compact while the model's unforced habit
is uniform `json.dumps` spacing (measured greedy with no grammar,
`tests/probe_unforced_habit.rs`), so under stock bytes the #85 fix had
to pin generation *off* the model's habit to keep the round-trip
stable. The dialect analyzer measures this template's spacing as
`Spaced` and pins the grammar and `render_reference` to the same
spelling, so the model now generates its natural bytes and the
re-render reproduces them. The `enable_thinking` front-rewrite
(issue #86 interaction) is deliberately untouched: partial-render
thinking flags are `render_partial`'s bug to fix, not the template's.

`mistral4-gguf.jinja` is dumped from the Mistral-Small-4-119B-2603
Unsloth GGUF (`tokenizer.chat_template`, arch `mistral4`). Its call
format is `[TOOL_CALLS]name[ARGS]{…}` — function name outside the
JSON, no wrapper object, one `[TOOL_CALLS]` per call — which the
dialect analyzer derives whole as `Family::TagWithJson`; every marker
in it is a single special token in the model's vocab.

`mistral4-cache-stable.jinja` is drama_llama's cache-stability patch
of it (#88). Five changes, and nothing else — the call-rendering path
is byte-identical:

1. The assistant turn close (`</s>`) is emitted per *message*. Stock
   emits it unconditionally and has no `add_generation_prompt` branch
   at all, so it cannot render an open assistant turn and the
   generation-prompt render is never a byte prefix of the follow-up.
2. Reasoning round-trips as `[THINK]…[/THINK]` from the
   `reasoning`/`reasoning_content` field, gated by `preserve_thinking`
   for aged turns. Stock accepts a thought only as a `thinking`-typed
   content chunk, so the analyzer measures `ReasoningMode::None`, the
   channel is invisible to grammar/parser/re-render, and a
   `ReasoningReingest::Field` transcript trips stock's own
   `raise_exception` (pinned: `mistral4_stock_cannot_render_field_reasoning`).
3. Pre-call prose renders in emission order (`content_pre` before the
   calls, `content_post` after) rather than merged into one slot.
4. The 140-line Unsloth date-arithmetic preamble and the default Le
   Chat system message are removed. That block injected today's *and*
   yesterday's date into the prompt **prefix**, so a session spanning
   midnight lost its entire cache. Persona and dates are app content
   under the 0.7 boundary — supply them in your own system prompt.
   Note this is a behaviour change for callers that sent no system
   message at all and relied on the vendor default.
5. The `raise_exception` role-alternation guard is dropped;
   mid-conversation system turns render in the format's own
   `[SYSTEM_PROMPT]` framing instead of raising.

Argument interiors stay `tojson`-compact, matching stock, because the
model's unforced habit has not been measured yet — the cogito
precedent (`json_dumps`) is a one-filter swap once
`tests/probe_unforced_habit.rs` has run against this model.
`[MODEL_SETTINGS]{"reasoning_effort": …}` is driven by
`enable_thinking`; it sits in the prefix, so toggling thinking
mid-conversation invalidates the cache — inherent to the format, the
same way Qwen's `enable_thinking` front-rewrite is.

A `<model>.template.jinja` sidecar next to the GGUF still overrides
any of these — baked templates removed the *need* for sidecar
deployment on recognized models, not the mechanism.

## completion-scaffold.jinja

Not a baked pair and not detection-keyed: the completion scaffold for
*base* models (issue #88, Phase 6 / rung 4b). Renders the prompt as a
bare, never-closed JSON array of records — the kind of scraped data
file pretraining is full of — with **no special tokens** and no chat
framing. Assistant turns are the records; user turns render as zero
bytes (turn-order ballast only); optional system text sits above the
array. Byte layout documented in the file. Deployed today as a
`<model>.template.jinja` sidecar next to a base GGUF (rung 1);
`examples/soul_forge.rs` is the driving consumer. Graduates into the
ladder proper when rung 4b lands in `src/baked.rs`.
