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
stays a byte prefix of the next render across tool turns. A model
turn's answer also renders verbatim: stock `| trim`s it, so an answer
the model ended in whitespace, or began with a newline after its
thought, re-rendered shorter than it was generated (found by the
fleet sweep, `fleet_bakes_round_trip_every_admitted_shape`; not yet
seen live). A model turn renders from the `chunks` drama_llama
supplies (its blocks in emission order, see the gpt-oss notes below):
the first thought opens the turn as before, and a second thought —
`<|channel>thought\nA\n<channel|><|channel>thought\nB\n<channel|>` —
renders as its own channel block where the model wrote it instead of
merging into one `reasoning` (the fleet sweep pins it). Everything
else outside the thinking-channel block is byte-identical to
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
A turn renders from the `chunks` drama_llama supplies on every
assistant message — its blocks in emission order — so two analysis
blocks render as two (they used to merge into one) and a thought after
the preamble renders after it. Each text block is its own chunk, and in
a turn without calls every text but the last renders as a commentary
preamble: a preamble then a final come back from the parse as two text
blocks, and only the last is the final. Without `chunks` the turn is
rebuilt from the merged fields as before.

A final answer renders under the header the model wrote:
`<|channel|>final <|constrain|>json<|message|>` when the thinking chunk
before it carries `constrain` (the content type the model declared,
which the parser records in the signature of the analysis block right
before the final — `drama_llama:tail;constrain=json`), plain otherwise;
an empty final renders from the thought alone. That content
type is what gpt-oss writes for structured output, unforced (every JSON
final in the 2026-10-01 Agora run carried it); stock renders every
final channel plain, so each one re-rendered a constraint short and
lost its tip (470..1111 tokens a turn, live). The content's shape
cannot stand in for the header — `[1, 2, 3]` may be prose, and a
structured answer whose schema root is a string or a number is not
`{…}` — so both spellings round-trip, and after another block the
`output_config` grammar leaves the choice to the model; opening the
turn it admits only the plain header. Irreducible, pinned in
`gptoss_cache_stable_round_trips_json_final` and
`gptoss_cache_stable_keeps_a_preamble_apart_from_its_final`: a
constrained final with no thought right before it — none, or a
preamble between — has nowhere to record its header, and re-renders
plain. Under `output_config` the grammar refuses that preamble after
the analysis, and a turn that opens with one is a schema violation. Harmony does not document a final-channel content
type (its guide shows `<|constrain|>` only on commentary calls); this
follows the model.
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

A second change (2026-10-01): the **prose-to-call gap**. Stock prints
`\n` before *every* call, the first included. The model's habit is to
end its prose in whitespace and open the call (`Checking.\n\n<tool_call>`);
the parser leaves that gap in the prose, so stock re-rendered
`Checking.\n\n\n<tool_call>` — one newline more than was generated —
and every cogito tool turn after prose lost its tip (19 live events).
The bake prints the gap before the first call only when the prose does
not already end in whitespace, the same rule as the Qwen bakes' point 3
below. A content-less turn and trimmed client prose render exactly as
before (`cogito_cache_stable_aged_turn_renders_as_stock`). Between
calls the `\n` stays: the analyzer now measures it as the dialect's
`call_separator`, so the grammar forces the same byte (it used to force
calls back to back). Round-trip pins:
`session::tests::cogito_cache_stable_round_trips`. Irreducible, as for
Qwen: prose run straight into the call (`Checking.<tool_call>`) gets the
template's `\n`, and whitespace after the last call is dropped.

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
2. Reasoning round-trips as `[THINK]…[/THINK]`, one block per thought,
   from the
   `reasoning`/`reasoning_content` field, gated by `preserve_thinking`
   for aged turns. Stock accepts a thought only as a `thinking`-typed
   content chunk, so the analyzer measures `ReasoningMode::None`, the
   channel is invisible to grammar/parser/re-render, and a
   `ReasoningReingest::Field` transcript trips stock's own
   `raise_exception` (pinned: `mistral4_stock_cannot_render_field_reasoning`).
3. The turn renders in emission order from the `chunks` drama_llama
   supplies on every assistant message — stock Mistral's own chunk
   shape (`text` / `thinking`), plus a `tool_calls` marker where the
   first call sat — rather than merging prose into one slot and
   thoughts into one block. Back-to-back thoughts
   (`…[/THINK][THINK]…`, live 2026-10-01: ~1.6k tokens lost when they
   merged) keep their markers, and prose between thoughts stays where
   the model wrote it. Without `chunks` (the analyzer's probes) the
   turn renders from `reasoning` / `reasoning_content` and
   `content_pre` / `content_post`.
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

`qwen3.6-gguf.jinja` is dumped from the Qwen3.6-35B-A3B Unsloth GGUF
(`tokenizer.chat_template`) — byte-identical, modulo the trailing
newline, across the `UD-Q4_K_S` and `UD-IQ4_XS` quants we serve.
`qwen3.8-gguf.jinja` is dumped from Qwen3.8-27B `UD-Q8_K_XL`. Both
moved here from `tests/fixtures/templates/` when they became detection
keys.

`qwen3.6-cache-stable.jinja` / `qwen3.8-cache-stable.jinja` are
drama_llama's cache-stability patches of those. The deviation from
stock, and why: stock `|trim`s an assistant turn's content and its
reasoning (3.6 also `lstrip`/`rstrip`s the halves it splits on
`</think>`), then prints a fixed `\n\n` after the close and before the
first call. The model does not always write exactly that — an answer
ending in `\n` or a space, one starting with `\n`, a thought closed on
a blank line, a single `\n` after `</think>` or before `<tool_call>` —
and every such turn re-rendered differently than it was generated, so
the next request lost the turn's whole KV (#88's round-trip invariant;
the 2026-09-30 Qwen3.6 run lost a 7364-token tip this way). The patch:

1. The **assistant** turn renders verbatim — no trim of the answer or
   the reasoning, and the 3.6 `</think>` split is exact (drama_llama
   inlines a thought as `<think>…</think>` with no padding, so the
   split recovers the parsed blocks byte-for-byte). 3.6 splits on the
   *first* `</think>`, the one closing the inlined thought; stock keeps
   only what follows the last, so a second `</think>` in the answer
   dropped all the prose before it (the aged branch below follows the
   bake here too — the one place it parts from stock). The flip side: the
   split no longer normalizes a *padded* `<think>\n…\n</think>` a
   client inlined into a Text block itself — its padding renders
   inside the thought, doubling the template's own. drama_llama never
   writes one (`append_block_text` inlines unpadded), so its own
   turns are unaffected.
2. The gap after `</think>` is carried by the parsed answer (the
   parser leaves it there). The template prints its canonical `\n\n`
   only when there is no thought (the gap then belongs to the
   thinking-off generation prompt's closed scaffold) or when the
   answer starts with no whitespace (a client that trimmed it).
3. Likewise the prose-to-call gap: `\n\n` before the first
   `<tool_call>` only when the prose does not already end in
   whitespace.
4. An aged turn rendered with `preserve_thinking` off drops its
   thought and cannot be byte-stable anyway; it renders exactly as
   stock, 3.6's `lstrip('\n')` of the answer included
   (`qwen_cache_stable_aged_turn_renders_as_stock`).
5. A **system turn after the leading run** renders as its own
   `<|im_start|>system\n…<|im_end|>` block, content trimmed like the
   leading system text, where it was seated. Stock 3.8 raises
   ("System message must be at the beginning.", live on
   Qwen3.8-27B 2026-09-30) and stock 3.6 drops it silently — so a
   misanthropic `Chat` in-conversation System note either failed the
   request or never reached the model. Anthropic's Messages API seats
   such turns on some models; Mistral 4's bake made the same call
   (its point 5). Pinned, with both stock behaviours as controls:
   `qwen_cache_stable_renders_mid_conversation_system`.
6. Qwen3.6 only: every **non-string tool-call argument** renders with
   `tojson`. Stock 3.6 `tojson`s only mappings and sequences and
   prints every other value `| string`, which minijinja spells
   Python-style — `null` as `none`, and from 2.24 booleans as
   `True`/`False` (drama_llama#120). The grammar has the model write
   JSON, and the parser types a non-string parameter's value from it,
   so a `<parameter=detail>\nnull\n</parameter>` the model wrote
   re-rendered as `none` and the next request lost the turn's KV
   (live, Qwen3.6 on Agora, 2026-10-01: 359..6909 tokens a turn).
   Strings are untouched (`| string`, raw — a string-typed value that
   merely looks like `null`, `true` or `5` stays the string it is);
   integers and floats spell the same either way (`5`, `1.0`). This is
   stock 3.8's own rule, so the 3.8 bake needs no change here. Pinned,
   stock 3.6 as the control:
   `session::tests::qwen_cache_stable_round_trips_scalar_args`.
7. A turn that **reasons again after its prose**
   (`…</think>\n\nChecking.<think>\nMore.\n</think>…`) renders from
   the `chunks` drama_llama supplies: the first thought opens the turn
   as before, and each later one renders inline where the model wrote
   it, `<think>\n…\n</think>`. The merged fields cannot place it — 3.6
   inlined it unpadded, 3.8 joined both thoughts in `reasoning_content`
   — so the turn lost its tip
   (`session::tests::qwen_cache_stable_round_trips_a_second_thought`).
   Aged out with `preserve_thinking` off, such a turn renders its prose
   alone, every thought dropped as stock drops the merged ones, and a
   client's text after the calls renders before them with the rest
   (`session::tests::qwen_cache_stable_chunks_age_and_keep_late_text`).

The grammar keeps the same split, so no template change is needed for
it: a string argument is generated raw, and so is one drawn from a
finite set — a string `enum` or `const`, nullable or not
(`<parameter=detail>\nfull\n</parameter>`, never `"full"`). Before,
such a member was generated JSON-quoted and re-rendered without its
quotes (Agora's `detail`, 2026-10-01), a tip miss on every such call.
A set whose raw spellings would collide (`"1"` beside `1`) stays
JSON. Pinned in the same test, the quoted spelling as the control.

The leading system/tools header, user and tool turns, the
reasoning-effort block (3.8), tool declarations, the rest of the
tool-call bodies and the generation prompt are byte-identical to
stock, and the dialect analyzer measures the same `CallSyntax` for
each pair (`qwen_cache_stable_analyzes_like_stock`).
Round-trip pins: `session::tests::qwen_cache_stable_round_trips` and
`qwen_cache_stable_round_trips_scalar_args`.

Irreducible, and pinned there so an improvement flips them
deliberately. The first three because no block can record a byte the
model *didn't* write; the next two because the template's own
structure drops bytes, as stock's does; the last because a parsed
number keeps no spelling:

- A gap after the close the model omitted (`…\n</think>Ada`)
  re-renders as the canonical `\n\n`.
- An empty thought followed by a lone `\n` (thinking on,
  `\n</think>\nAda`): an empty thought is the thinking-off scaffold,
  whose `\n\n` the template supplies, so the `\n` stays in the answer
  and renders after it — `\n</think>\n\n\nAda`.
- A thought closed without its newline (`Thought.</think>\n\nAda`)
  gets the canonical `\n</think>` back: the parser strips that `\n`
  when present but cannot record its absence.
- Whitespace after the last call (`…</tool_call>\n`) is dropped: the
  template closes the turn right after `</tool_call>`, and no block
  carries the tail. No constrained call turn writes it: the tool
  grammar reads a newline after a call as the separator to the next
  (`qwen_whitespace_after_the_last_call_is_unreachable`).
- Qwen3.6 only: a thought containing a literal `<think>` loses
  everything before it. The inlined thought is recovered with
  `split('<think>')[-1]`; 3.8 reads `reasoning_content` and
  round-trips it.
- A number argument in a non-canonical spelling (`1.50`, `1e3`)
  re-renders canonically (`1.5`, `1000.0`): the parsed value keeps no
  spelling. The grammar's `number` rule admits both forms.

Byte-stable is not token-stable at the prompt seam. An emission that
*starts* with `\n` was generated as its own token after the generation
prompt's trailing newline, but when the next request re-tokenizes the
turn, BPE merges the two (`…\n\n` + `\n` → one `\n\n\n` token). The
bytes match; the token sequences part at the seam, so the LCP walk
stops there and the turn re-prefills. Only a tip reached by the
hash-keyed lookup (`hash_keyed_l_hit`, render-hash equality, which
reaches past BPE boundaries) survives it. Outside the template's reach
— a tokenizer property, not a rendering one.

A `<model>.template.jinja` sidecar next to the GGUF still overrides
any of these — baked templates removed the *need* for sidecar
deployment on recognized models, not the mechanism. A sidecar that is
a byte-identical copy of an *old* bake holds back every fix since, so
`src/baked.rs` keeps the SHA-256 of every superseded replacement
(`SUPERSEDED`) and a match logs `stale_template_sidecar` at `WARN` on
load. **When you change a replacement here, add the hash of the version
it replaces to that list.**

## Framing no block carries: the thought tail

Anthropic never returns a whitespace-only text block and rejects one on
ingest, so the parse never yields one — yet the whitespace a model
writes between a thought and its call (`[/THINK]\n[TOOL_CALLS]`,
`<channel|>\n<|tool_call>`, `</think>\n\n<tool_call>`) is in the KV.
It rides in the closed thought's `signature`
(`drama_llama:tail;gap=%0A`), and the renderer (`chat_template.rs`,
before any template sees the turn) puts it back as the text it was, so
every template here renders the turn exactly as if it were a block.
The same tail carries the content type of a Harmony final right after
the thought (`;constrain=json`, above). Whitespace with no thought
before it — a turn's first bytes before a call, after the last call —
has nothing to ride and is dropped (`blank_text_is_never_returned`).

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
