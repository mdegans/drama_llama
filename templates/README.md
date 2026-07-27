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

A `<model>.template.jinja` sidecar next to the GGUF still overrides
any of these — baked templates removed the *need* for sidecar
deployment on recognized models, not the mechanism.
