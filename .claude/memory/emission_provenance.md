---
name: emission-provenance
description: 2026-10-01, branch fix/content-literal-specials — why the dialect parser now reads a reserved piece the model *spelled* as text, how (marker swap on the emission + real-token-only trigger), which vocabularies it protects (measured), and the limits left open
metadata:
  type: project
---

# Emission provenance — read before touching the parse paths in `Session`, `StreamParser`, or the deferred-trigger scan

## Why it exists

Content literals (`0a33029`) made prompt content that spells a reserved
piece read as text instead of a 400. That opened a hole the old 400
had been hiding: the model can now *read* `<tool_call>{…}</tool_call>`
in an Agora post and *copy* it into its own output as ordinary tokens.
The dialect parser works on text, so it seated the copy as a real
`ToolUse` — tool-call injection from content. The lazy grammar's
trigger scan was byte-based too, so the copy also armed the call
grammar (the old class (c) in `truncated_call_containment.md`).

Required property: **only framing the model emitted as the real
reserved token id is structure.** The same bytes as ordinary tokens are
text.

## How

- `dialect::Provenance` (crate-private, `src/dialect/provenance.rs`):
  fed `(piece, token)` per emitted token. A reserved token whose piece
  ends the reassembled piece passes verbatim; ordinary bytes accumulate
  in `pending`, and every reserved piece found there is replaced by the
  render's own literal marker `<{sentinel}:t{id}>` (fresh sentinel per
  call). A tail that could still grow into a piece
  (`LiteralNeutralizer::could_grow`, sorted-index binary search) is held
  back, so the marked text only grows by appending — required by the
  streaming diff. A real token ends the ordinary run.
- The parser never changed. Session parses the *marked* text and
  restores the blocks (`restore_block` / `restore_open`, partial JSON
  gets JSON-escaped pieces). `StreamParser::with_provenance` +
  `push_token` does the same per tick and never cuts a prose delta
  inside a marker (`cut_before_marker`).
- Wired into `run_call`, `complete_stream`/`BlockStream`, the
  `StopFilter` parser on every path, and `complete_text`'s stop cut.
  That cut is searched in the *restored* bytes, each prefix parsed as
  the marked text it restores from (`Provenance::marked_prefix`): a
  stop can start inside a spelled piece (`_call` in a spelled
  `<tool_call>`), where no marked prefix ends, so a marked-coordinate
  search found no cut and kept the stop (caught in review; regression
  test `a_stop_inside_a_spelled_piece_is_cut`).
- Trigger: `PiecePredictor::with_reserved` → `TokenPredictor` records
  byte spans of real reserved tokens; a trigger occurrence is accepted
  only if each reserved piece *inside the trigger* is exactly a real
  span. Scan now iterates all occurrences, so a refused spelled one
  does not hide a later real one.
- Containment reads the unrestored parse: any reserved piece left in
  its free text is a real token (spelled ones are markers). Exact now;
  the old count-based `real_specials_in_free_text(blocks, raw, emitted)`
  is gone. On a stop turn it reads only the parse of what the cut keeps
  (`stop::marked_stop_cut`, shared with `complete_text`): the stop is
  seen late while provenance holds a growable tail (`<`), and a real
  `</tool_call>` past the cut used to reject a turn whose output was
  clean.
- Duplicate-text specials: `LiteralTable::build` keeps only the id a
  piece tokenizes back to; any other special with the same text is an
  *alias* (`LiteralNeutralizer::with_aliases` / `emitted_piece`) — real
  framing when emitted, for provenance and the trigger scan. Content
  marking never uses aliases. Without them a call opened by the alias
  came back as text (fail-closed).
- Auto-tip: `byte_stable` compares restored bytes, which cannot see a
  real reserved token in content (the render spells it). `run_call`
  stores no tip hash when the marked parse holds one — reachable only
  with `with_emit_specials_ban(false)`, since containment rejects it
  otherwise.

## Coverage — measured, per vocabulary (vocab-only loads, 2026-10-01)

Protection is per *vocabulary*, not per dialect: it exists where the
framing piece is a reserved special.

- Protected: Qwen 3.6 / 3.8 `<tool_call>` `</tool_call>` `<think>`
  `</think>`; all gpt-oss Harmony header tokens — but the recipient is
  text, and `harmony_block` accepts a header starting with a plain
  ` to=…` at the turn's start and after every real `<|end|>`, so the
  only real token gating a gpt-oss call is `<|message|>` (`<|call|>`
  optional). Inherent to Harmony; Gemma 4 `<|tool_call>`
  `<tool_call|>` `<|"|>` channel markers; Mistral 4 `[TOOL_CALLS]`
  `[ARGS]` `[THINK]` `[/THINK]`; cogito-32b `<tool_call>`.
- **Not** protected: cogito-32b `<think>`/`</think>` (plain text in Qwen
  2.5's vocab — its reasoning split is still by bytes); Llama 3.1 bare
  JSON; Hermes on a vocab without a `<tool_call>` special; plain-text
  markup inside a real call (Qwen XML `<function=`, `<parameter=`).
  Defense there: grammar + `tool_choice`.

## Limits left open (deliberately)

- **A copy can be real.** Provenance separates a spelling from a real
  token, not a copy from an intent: a model that read spelled
  `<tool_call>…` in a post can emit the *real* ids when it repeats it,
  and nothing at the parse level can tell that from a genuine call.
  Defense: `tool_choice`, grammar, model tier.
- `BlockStream` runs no emission containment (a stream cannot take back
  what it yielded). Pre-dates provenance: a spelled opener with a real
  close streams as Text holding the real `</tool_call>`, where the
  batch path returns `EmittedSpecialToken`. The next ingest reads it as
  text, so it is a KV/containment gap, not a structure one.
- **The grammar is byte-level.** Under an armed grammar the model can
  spell a piece the grammar requires. Parse reads it as text: a forced
  call with a spelled opener → `GrammarViolation` (warm retry); a real
  opener with a spelled close → the call still seats (balanced JSON)
  with the close left as text, or degrades and is contained. Never a
  call the model did not open with the real token. Rare — models emit
  their own framing as the token. A token-level grammar literal would
  close it; that is the #97 canonicity arc, not this.
- Public `dialect::parse_text` / `StreamParser::push` see only text and
  stay provenance-free.
- **Model tier not run for this change** (GPU runs are Mike's). The
  unignored tier + scripted-decoder Session tests (`session::literal`)
  and dialect tests (`dialect::provenance`) are green; mutation-checked
  (disable marking → 6 reds; disable trigger provenance → 2 reds). The
  real-model risk to watch: a model that habitually *spells* its own
  framing would stop seating calls — none of the fleet should, but only
  `just test ignored` says so.
