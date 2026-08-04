# Plan of record: owned chat templates, loading ladder, base-model fallback

> **STATUS: PHASES 0–2, 4, 5b LANDED. 5a CLOSED as not-to-do.**
> (phases 4 + 5b: 2026-07-27, session 3, Opus 5.) Mike confirmed the
> full-commitment direction in-session ("Cool. Your plan.") after the
> #85 arc; scope details delegated.
> GitHub twin: [issue #88](https://github.com/mdegans/drama_llama/issues/88).
> Supersedes the "own templates on measured need" stance argued
> earlier the same session, and the sidecar-only distribution model.
>
> **Phase 3 is the remaining work**, and it is smaller and more
> conditional than written — see session 3's progress note. Two of
> this plan's phase premises were wrong and were corrected by
> measurement; if you are about to implement a phase from its bullet
> below, read session 3 first.

## Progress (2026-07-27, session 3, Opus 5) — Phases 5b + 4

Headline: **two phase premises in this plan were wrong.** Both were
written from a doc comment rather than from the code, and both were
corrected by reading the code before implementing. The general lesson
for this arc: *the phase bullets below are hypotheses, not
specifications.*

### Phase 5b LANDED (`83fb5d3`) — but not the bug the plan described

The plan said `compute_tip_extension` "creates a tip only in the
stop-sequence branch", so grammar-complete and max-tokens endings got
none. **False.** The discriminator is arithmetic —
`generated_tokens.len() == kv_generated_count + 1` — and because the
predictor decodes lazily (the token sampled on iteration *k* is
committed by iteration *k+1*'s `decoder.step`), the last sampled token
is uncommitted for **every** ending. All three produced a tip already.
The method's own doc comment asserted the opposite and is what
misled the plan; it is now corrected in place.

The real defect was one level down, in *what the tip predicted*:

- **stop-sequence ending** — the terminal token's piece is dropped from
  `raw_text` by `eos_pieces`, so `canonical_close` (the render tail
  after `raw_text`) correctly **replaces** it. Coherent.
- **grammar-complete / max-tokens** — that piece **is** in `raw_text`
  (pushed before the `grammar_complete()` break). The close is the tail
  *after* it, but the code still replaced the uncommitted token,
  **dropping a real content token from the prediction**. Stored
  `prev_entries` said `…t₁…t_{k-1}, close`; the next turn renders
  `…t₁…t_{k-1}, t_k, close`. The LCP died exactly *at* the tip entry,
  `compute_l_hit`'s `safe = lcp - 1` landed one short, tip
  disqualified.

So the tip survived grammar-complete turns **on the hash path alone** —
which is why `hash_cache_smoke` was green throughout (byte-stable
render ⇒ hash present). Every turn whose hash missed lost it, and
**grammar-complete is the normal ending for a tool call**.

Fix is one unifying rule, smaller than the plan's: *the entries past
the KV head are the re-render's own tokenization of everything past
the KV head.* `uncommitted_bytes` tracks the final loop iteration's
contribution to `raw_text`; tail = `close` after a stop, `piece +
close` otherwise, tokenized **jointly** (the content→close seam is
exactly where BPE merges). The truncate-then-extend already
implemented "replace the past-KV region" — only the caller's notion of
where the tail began was wrong.

Pure core extracted to a free `tip_extension` so it is fast-tier
testable without a model. Six pins; the grammar-complete one asserts
the *old* close-only tail loses the tip against the same next-call
entries, so it cannot silently regress.

**Known, unchanged, deliberately tip-less**: the UTF-8 flush ending.
`PiecePredictor` yields a piece with no new token
(`predictor.rs`, stream-end flush), both session loops re-push
`last_token()`, the count comes out one high, and the arithmetic check
fails. Rare (needs a codepoint split at stream end) and safe. It also
inflates `generated_count` — i.e. reported `output_tokens` — by one in
that case. Not worth machinery; noted so it is not re-diagnosed.

### Phase 5a CLOSED as not-to-do — no re-render contract exists

The plan wanted structured output moved to the canonical prelude, "same
latent bug when a JSON response replays as history". Measured: **there
is no such bug.** A structured-output answer becomes a `Block::Text`,
and `Block::Text` renders **verbatim** (`chat_template.rs`,
`out.push_str(text)` → `content` as a plain Jinja string). Nothing
re-serializes it — `json_canon`/`json_dumps`/`render_reference` are all
tool-call-only paths. The byte-stability gate therefore holds for a
structured-output turn *whatever whitespace spelling the model chose*.
(This also independently re-confirms #85's "not `output_config`"
finding.)

Pinning it would repeat the `grammar_for_tool_choice` regression
exactly: bytes forced where nothing re-serializes them, which the #85
collateral measured as *degraded generation* — see
[[cogito_tool_turn_cache_loss]]'s collateral section. **Pin bytes only
where a re-render contract exists.** Structured output has none.

### Measured follow-up (same session, Mike's prompt: "so structured output *will* still round-trip then?")

**Yes — the bytes. No — the token ids.** This distinction is the whole
finding, and it took a real measurement to see, because the analysis
above is correct and still predicts the wrong practical outcome.

New model-backed pin `structured_output_round_trips_as_history`
(`tests/output_config.rs`). Run first with the assistant turn
**unmarked**, instrumented at `select_slot`:

    prev_len=581 new_len=605 lcp=327
    tip=Some(EntryPos { entry: 578, pos: 578 }) tip_hashed=true
    n_bp_hashes=0 hit_entry=0     →  cache_read = 0

Read that carefully: the tip exists, sits at the right position, and
**its hash is present** — which means `render_extended` reproduced the
emission byte-for-byte and the byte-stability gate passed. The
round-trip claim is confirmed, not refuted. And reuse was still
**zero**, because the prompt was 325 entries and the LCP died at 327 —
*two tokens into a 254-token emission*.

**Cause: grammar-constrained generation emits a NON-canonical BPE
segmentation.** The grammar masks a longer merged token whenever it
would overshoot the allowed next characters, so the model's token
sequence is not what the tokenizer produces from the same bytes.
`prev_entries` holds the emitted ids; `new_entries` holds the
re-tokenized render; `compute_l_hit` compares ids. Identical bytes,
different segmentation, dead walk.

So for any grammar-constrained turn — structured output **and tool
calls** — the hash path is not an optimization, it is the only path
that works, and it needs a breakpoint at the tip position: the caller
must mark the assistant turn's last block. That is exactly what
`hash_cache_smoke`'s `mark_last_block` does, and why that suite has
always passed while an unmarked equivalent gets nothing. Marking it
here took the same test to `hit_entry=593, cache_read=593`.

**Consequence for Agora / blallama** (flagged for Mike, not acted on):
if the seed runner does not put `cache_control` on the assistant turn
it replays, structured and tool turns will not reuse — regardless of
everything else in this plan. Unconstrained prose is unaffected (its
segmentation is canonical), which is why plain chat reuses happily
without marking and hid this.

**Open design question**: the tip carries a hash, but
`hash_keyed_l_hit` only compares it against breakpoint hashes the *new
call declares*. We could instead hash the new call's prefix at the
tip's own entry and compare directly, making tip reuse survive BPE
drift with no marking required. A real improvement and a real design
change — Mike's call, not a bugfix to slip in.

**…and chasing that turned up a live silent-corruption bug:
[issue #91](https://github.com/mdegans/drama_llama/issues/91).**
`hash_keyed_l_hit` returns a position computed against `prev_entries`
and the caller indexes `new_entries` with it, with no translation. Its
doc justifies this by claiming a hash match makes the prefix entries
"identical in the new list" — true for bytes, false for segmentation,
and its signature (a bare set of hashes, no entry indices) cannot
express the new-space answer anyway. Measured on Qwen3.6: the same
**2322 bytes** occupied **616 entries in `prev`, 613 in `new`**, so
reuse at 616 skipped `new_entries[613..616]` — three tokens of the new
user message, never decoded. Cause visible at the LCP end: the grammar
forces a bare `"` (token 1) where the tokenizer merges the quote into
the following word (token 39441). Fires on grammar-constrained turns
replayed WITH `cache_control` on the assistant turn — i.e. the
configuration that otherwise works. Pinned model-free by
`hash_keyed_l_hit_result_is_prev_space_issue_91`. Fix direction in the
issue; option (1) there is the same change this design question wants,
so they want designing together.

Note the shape of the near-miss: the earlier probe
(`tests/retokenization_drift.rs`, since deleted) tokenized the emission
**in isolation** and found zero drift, which would have exonerated the
whole area. Drift only appears against the *whole-render* tokenization
with the real schema grammar. Isolating the suspect changed its
behavior — measure the thing in situ.

---

On the whitespace question specifically: **the JSON serialization
round-trips unconditionally, because we never re-serialize it.** The
whitespace *around* it is a separate matter, and there are two seams on
such a turn that are not the JSON body:

1. **A thought prefix** (`allow_thought` ⇒ `root ::= thought? ws
   output_schema`) *is* normalized: `parse_thought` strips a leading
   `\n`, `trim_end()`s the body and swallows one `\n` after the close
   (`parse.rs:609-618`), while the renderer re-emits bare
   `<think>`/`</think>` and lets the template lay out its own newlines
   (`chat_template.rs:1114-1117`). Shared with every prose turn; a
   prelude swap does not address it.
2. **Template-level `|trim`.** `ws ::= [ \t\n\r]?`
   (`grammar_compile.rs:878`) permits one whitespace char before the
   JSON, and e.g. Qwen3.6's template does
   `render_content(message.content, true)|trim`. If the model takes
   that character, emission and render disagree by one byte, the
   byte-stability gate fails, and the turn falls to the LCP walk.

If (2) ever bites, the fix is *not* this phase: pin that single leading
`ws` to empty (no trained habit is being fought there) rather than
swapping the whole prelude, which would also pin the JSON interior
where the model DOES have a habit — the exact mistake #85's collateral
measured. A prelude swap would additionally over-pin framing
whitespace, since these roots reference `ws` directly and
`json_grammar_canonical` rewrites `ws ::= ""`; they would need moving
to `fws` first, as `dialect/emit.rs` already does.

Unmeasured oddity noticed in passing, pre-existing and unrelated: with
`allow_thought` that same `ws` sits between `</think>` and the JSON and
admits only ONE whitespace char, so the `\n\n` a thinking model is
trained to emit there is not grammar-legal today.

### Phase 4 LANDED (`85027b3`) — drift alarm; owned Qwen template NOT needed

- **Drift alarm**: `baked::nearest_stock` analyzes an unrecognized
  embedded template and compares its `CallSyntax` against every
  registry entry's **stock** dialect — stock-vs-stock deliberately,
  because a replacement diverges from its stock *by design* (Cogito's
  spacing swap), so comparing against replacements would report drift
  on every healthy model. A hit means "same dialect, other bytes" and
  the rung-3 warning now names the family and says a second detection
  key would restore rung 2. Advisory by construction: analysis failure
  ⇒ `None`, and full `CallSyntax` equality is strict, so a miss
  degrades to the plain warning — the only failure mode it can add is
  silence, never a wrong name.
  - This is the mechanism for the incoming **gpt-oss 120b** case
    already flagged in session 1: if its template does not byte-match
    the Unsloth 20b dump, the load warning will now say so *and* name
    gptoss, instead of a generic "unrecognized".
- **Owned Qwen template: not needed for the stated reason.** The plan
  filed aged-thinking continuity as "a Phase 4 owned-template
  decision". Measured first (probe-before-own): the Qwen3.6 template
  **already honours `preserve_thinking`**, and `Session::from_engine`
  already sets it on every render. New fast-tier pin
  `qwen36_aged_thinking_continuity` builds a real `Block::Thought`,
  ages it with a **real user query** (the case `last_query_index`
  reacts to — the existing pin ages with a tool response, which the
  pre-scan deliberately skips) and passes on stock. Its control half
  pins that dropping the flag *does* strip the thought, so the flag is
  load-bearing and the pin will fire if upstream changes that.
  - Residual, genuinely-owned-template-shaped reasons remain if we ever
    want one: the stock template `|trim`s content and reasoning
    irreversibly, splits on `</think>` exactly once (a second thought
    block or a literal `</think>` in content loses the middle), and we
    depend on an *Unsloth patch* for `preserve_thinking` — a stock
    Qwen dump from elsewhere may lack it. None of these is urgent and
    none is aged thinking.

### Exit gate

`just test ignored` — **121/121 on the real models**, run separately
for each commit (5b's gate ran before the phase-4 edits existed, so
attribution is clean). Fast tier 512/512.

## Progress (2026-07-27, session 2, Fable) — Phase 2, the Cogito pilot

- **Probe first, as planned**: `tests/probe_unforced_habit.rs` (new,
  `#[ignore]`d, cogito-gated, skip-not-substitute) renders a
  tools-bearing prompt and predicts greedily through the raw Engine
  path — no grammar, no Session. Measured emission: **uniform
  `json.dumps` spacing**, exactly as predicted — `": "` after every
  key (envelope AND interior), `", "` between fields and array
  elements, no brace padding, raw apostrophes,
  `<tool_call>\n{...}\n</tool_call>` framing. The #85 compact-interior
  canonical was confirmed off-habit. (Probe lesson: raw-Engine stops
  are opt-in — `PredictOptions::add_model_stops` — without it greedy
  runs straight through `<|im_end|>`.)
- **Mechanism — spacing is measured data, not a hardcode.** The chain:
  the *active template's* probe render → analyzer
  (`detect_json_spacing`, both separator positions must agree, mixed
  ⇒ Compact) → `ArgumentsSyntax::json_spacing: JsonSpacing`
  (`Compact` default = pre-existing behavior everywhere) → both the
  grammar prelude and `render_reference` key off it. Three views, one
  byte string, single source (the template bytes).
  - `src/json_canon.rs` (new): `JsonSpacing` + the `json.dumps`-exact
    serializer (`SpacedFormatter`); separator accessors shared with
    the grammar prelude so they cannot drift.
  - `JSON_GRAMMAR` separators are now **position-aware named
    productions** (`kv_sep` / `elem_sep` / `pad` — one generic `ws`
    could never express "space after `:` but not inside braces");
    `schema_to_gbnf` emits references to them, staying
    prelude-agnostic. `json_grammar_canonical(spacing)` pins all
    three per profile. Fast pins updated:
    `canonical_json_grammar_pins_separators`,
    `canonical_prelude_admits_exactly_one_spelling` (supersede the
    `..._pins_ws` / `..._admits_only_compact_json` names cited in
    older memos).
  - `json_dumps` minijinja filter (chat_template.rs) +
    `register_template_filters` shared by ChatTemplate's two envs
    **and the analyzer's probe env** — closing a latent gap: the
    analyzer env previously used builtin (HTML-escaping!) `tojson`,
    masked only by clean-ASCII sentinels.
- **Owned template**: `templates/cogito-gguf.jinja` (detection key,
  dumped from the 32b GGUF, byte-identical to the 14b fixture) +
  `templates/cogito-cache-stable.jinja` = stock with ONE line changed
  (`tool_call.arguments | tojson` → `| json_dumps`). `baked::COGITO`
  registered. Deliberately minimal: re-ingest semantics untouched
  (Phase 3), `enable_thinking` front-rewrite untouched (#86 is
  render_partial's bug; owned template keeps stock structure).
- **Pins**: `render_reference_matches_cache_stable_template_render`,
  `reconstruct_cogito_cache_stable` (adversarial payload sweep),
  `cogito_cache_stable_prefix_continuity` (continuity refactored into
  `assert_prefix_continuity`, shared with stock-cogito and qwen36
  pins). `dialect_analyzer`: `json_dumps_template_measures_spaced` +
  Compact pin on hermes. All fast-tier, model-free.
- **Deprecated `ToolChoiceOptions` path stays Compact** (explicit
  arg) — it has no analyzer to measure with, and adding a pub field
  would break 0.8 struct literals.
- **Latent, noted not fixed** (pre-existing, out of scope; filed
  2026-07-27 at Mike's request — slated for end-of-arc or just
  after, mid-arc only if they block):
  (1) [issue #89](https://github.com/mdegans/drama_llama/issues/89)
  `fun_name_is_key` JsonNative: grammar envelope (`KV_SEP`) vs
  compact render disagree under Compact — no current model hits it;
  comment at the render site.
  (2) [issue #90](https://github.com/mdegans/drama_llama/issues/90)
  Container-valued `enum:`/`const:` schema literals embed compact
  bytes in rules, so they'd mismatch a Spaced dialect —
  schemars-derived tools only produce scalar literals; documented on
  `json_grammar_canonical`.
- **Verified on the real model** (M2 Max, cogito-32b): cogito
  hash_cache_smoke n=3, deterministic — round 2 `cache_read = 196`
  vs round 1 `input_tokens = 170`, tip alive. Plus a **rung-2
  witness** now in the test itself:
  `session.dialect().arguments.json_spacing == Spaced` asserted for
  the cogito variant — the tip surviving alone does NOT discriminate
  stock from baked (both are round-trip stable post-#85); the
  measured spacing does. This closes session 1's "no test observes
  rung 2 end-to-end" gap for cogito; gemma/gptoss still get theirs
  via the Phase 3 e2e flip.
- **Phase 2 residue for Mike**: the seed-runner A/B on post-body
  quality (spaced-canonical vs compact-canonical, same seeds) — the
  distributional worry from the Why chain, checkable only on the
  Agora side. Mike (end of session 2): deliberately deferred until
  **Phase 3 at least** — task-switching cost outweighs the likely
  small change; don't re-raise it before then.

## Progress (2026-07-27, session 1, Fable)

- **Phase 0 LANDED** (`ce39ebe` + `c12ec4b`): the #85 acceptance pin
  (`render_reference_matches_template_tool_call_render`) was still
  `#[ignore]`d post-fix — un-ignored, doc rewritten to pin the fix.
  `assert_reconstruction` now sweeps every fixture with a second
  **adversarial payload** (`'`, `&`, `<`, `>`, `"`, `→`, embedded
  newline, `", "` inside a string) — all 11 fixtures pass both.
  Cogito joined the harness (`reconstruct_cogito`,
  `cogito_prefix_continuity` — the #85 cache property FFI-free), and
  Qwen3.6 got the same continuity pin (non-thinking; aged-thinking
  continuity is a Phase 4 owned-template decision). NOTE the payload
  sweep found **zero** live bugs — the #85 fixes hold across every
  family, including gemma4's dict quoting and Qwen raw values.
- **Phase 1 LANDED** (`5b447e0`): `src/baked.rs` registry
  (`BakedTemplate { name, stock, replacement }`, `detect()` by
  trailing-whitespace-insensitive **byte-equality** with the stock
  dump — never fuzzy; unit-pinned incl. near-miss fall-through and
  replacements-never-keys). Templates moved to crate-root
  `templates/` (shipped via `include_str!`; provenance in
  `templates/README.md`). `Session::from_engine` runs rungs 2–3:
  baked replacement applies through `set_template_source` (dialect
  re-analyzes in lockstep), unrecognized templates warn as the
  best-effort tier. Sidecar appliers run after in `from_path_with`,
  so rung 1 still wins. e2e suites keep installing sidecars —
  deliberately, they now exercise rung 1 over identical baked bytes.
  - **Lockstep gap found and fixed**: `with_dialect` and
    `set_template_source` refreshed `thought_reingest` but left
    `render_opts.reasoning_start` stale — harmless so far only
    because every current override keeps the template's reasoning
    markers. Both now refresh it.
  - **Detection keys verified against the real GGUFs** (scratchpad
    GGUF-metadata reader, no model load): gemma-4-31B and
    gpt-oss-20b embedded templates byte-match `templates/*-gguf
    .jinja`, and cogito-32b matches the 14b fixture. Rung 2 fires
    for both fleet models in production.
  - **Open verification**: no test yet loads a real model and
    observes rung 2 end-to-end (gemma/gptoss e2e pin rung 1). The
    Phase 3 e2e flip (drop the sidecar installs, assert baked
    covers them) closes this on the model boxes.
  - The GGUF-metadata reader graduated to
    `scripts/gguf_template.py` (`--compare` mirrors
    `baked::detect`'s trailing-whitespace-insensitive equality).
- **Incoming: gpt-oss 120b** (Mike, end of session 1) — quantizing
  from **OpenAI's safetensors** with our llama.cpp checkout, landing
  in `models/`. Expectation is "same template as 20b", but our
  detection key is the *Unsloth* 20b dump and Unsloth patches
  templates — so before assuming rung 2 fires, run
  `python3 scripts/gguf_template.py models/<120b>.gguf --compare
  templates/gptoss-gguf.jinja`. On MISMATCH: diff the dumped
  `.embedded`; if semantically identical, add a second detection
  key (another `stock` mapping to the same `GPTOSS.replacement`) —
  by design, NOT a bug.

## Why (the short causal chain)

1. #85 measured that with a stock template, **the canonical byte
   string is dictated to us**: Cogito's template hardcodes a spaced
   envelope (`{"name": "`, `", "arguments": `) and delegates the
   interior to `tojson` (compact). The grammar must agree with the
   re-render, so we had zero degrees of freedom and pinned the model
   into that *mixed* shape — which matches no natural serializer and
   is not the model's habit (it emits uniform `json.dumps`-style
   spacing, per the captured trace). See
   [[cogito_tool_turn_cache_loss]].
2. Mike's worry (2026-07-27): forcing off-habit structural bytes may
   bias free-text quality (Agora posts live inside those argument
   strings). Grammar-wise string interiors are untouched free regions,
   so the only channel is distributional — small, but real in
   direction, and **unfixable under a stock template**.
3. Ownership makes canonical bytes a *design variable*: derive them
   from the **model's unforced emission habit** (probe the model, not
   just the template), render history identically, and the
   grammar-forcing distance approaches zero.
4. Full commitment also deletes code: `ReasoningReingest::Thinking`'s
   content-withholding contortions, stock-template KNOWN-QUIRK
   accommodations and their per-turn LCP fallbacks, and the per-family
   re-ingest variance (owned templates all read `reasoning_content`,
   keep aged thinking, render shape-C causal ordering).
5. Distribution: **bake templates into the crate** (`include_str!`,
   selected by detection) — consumers like blallama are self-contained
   binaries (Mike's call). The existing `<model>.template.jinja`
   sidecar remains as the per-install override.

## Design principles (the anti-spaghetti constitution)

- **Code accepts families, data accepts models.** New model support
  adds templates/fixtures/CallSyntax data and (rarely) a parser
  family arm — never sampler branches. The sampler stays family-blind:
  predicates on grammar state only. (Verified clean 2026-07-27:
  every model name in `sample.rs`/`predictor.rs` is comment
  provenance, not a conditional.)
- **Canonical bytes derive from the model's unforced habit**, measured
  (greedy, no grammar), not from the stock template's serialization.
- **Detection is tight, never fuzzy.** An unrecognized template/vocab
  falls through to the next rung with a loud warning. Gemma 5 must
  never silently receive Gemma 4's template.
- **The stock path is code-frozen.** It keeps working (best-effort,
  warning, no cache guarantee) but never again grows quirk
  accommodations. That freeze is what prevents dual-path rot.
- **Template owns framing; app owns content** (the 0.7 boundary).
  Scaffold exemplars, persona content, prompt engineering live in the
  consumer (seed runner), not in `.jinja`.

## Template loading ladder (Mike's design, 2026-07-27)

1. **Sidecar** `<model>.template.jinja` / `parent/template.jinja` —
   wins if present (existing mechanism, unchanged; dialect sidecar
   still overrides analysis afterward, lockstep invariant holds).
2. **Detected model** → baked owned template. Detection = embedded
   template-source sniff (primary, what works today) + vocab
   special-token cross-check (confirmation). Mechanically the
   metadata template is *read* before this rung decides — the ladder
   governs which template renders, not read order.
3. **Embedded metadata template** → used as-is + `tracing::warn`:
   best-effort tier, no cache guarantee.
4. **No template** (today: hard `ChatTemplateError::NoTemplate`):
   a. vocab special-token fingerprint matches a known format
      (converted GGUFs often strip `chat_template` but keep
      `<|im_start|>` marked special) → matching baked template +
      warning. Evidence-based guess, not vibes.
   b. true base model → **completion-scaffold fallback** (below).
   Rung 4b must ALSO be explicitly selectable (builder / named
   template): many "base" GGUFs carry a vestigial converted template,
   so detection alone can't be the only door.

## Phases

Ordered so the boring load-bearing work lands before the fun rung.

- **Phase 0 — harness backbone.** Generalize
  `tests/template_rendering.rs` beyond Cogito: per-owned-template
  round-trip pins (`render_reference` vs template render, payloads
  with `'`/`&` deliberately) + prefix-continuity pins (the
  `gemma4_cache_stable_prefix_continuity` pattern), fast tier. Under
  ownership these ARE the product contract — every cache guarantee
  rests on `.jinja` files we wrote.
- **Phase 1 — baked registry + ladder.** `include_str!` template
  registry keyed by detection; ladder rungs 1–3 wired (4 comes last);
  stock-path warning + code-freeze documented in code.
- **Phase 2 — Cogito pilot.** Probe the model's unforced call
  spelling (expected: uniform `json.dumps` spacing — space after `:`
  and `,`); pin THAT as canonical: position-aware `ws` rules in the
  emitter (the compact pin was a prelude swap; habit-matching needs
  separator-aware rule references) + a small custom
  `serde_json::ser::Formatter`; owned Cogito template renders history
  with the same serializer. Check the #86 interaction
  (`enable_thinking` rewrites the prompt *front* on this template).
  Measure: `hash_cache_smoke` n≥3, and a seed-runner A/B on post-body
  quality (the distributional worry is checkable — same seeds,
  spaced-canonical vs compact-canonical, diff the bodies).
- **Phase 3 — promote existing sidecars. PARTLY DONE; residue is a
  decision, not a deletion.** The promotion itself landed back in
  Phase 1: `GEMMA4` and `GPTOSS` are already in `baked::ALL`. What the
  bullet still describes as mechanical cleanup is not, per session 3's
  survey:
  - **`ReasoningReingest::Thinking` is not dead code.** The entire
    behavioural delta between it and `Field` is ~14 lines in
    `chat_template.rs` (`tool_call_message`'s content-withholding
    arm). Under the *owned* gpt-oss template, switching to `Field` is
    byte-neutral. But rung 3 still exists, and a gpt-oss GGUF whose
    template does **not** byte-match ours (e.g. the incoming 120b)
    would then render merged `content` as an analysis block via the
    stock template. So deleting it trades a real rung-3 behaviour for
    tidiness. **Mike's call, not a mechanical cleanup** — and worth
    deciding *after* the 120b's template is dumped and compared.
  - **"KNOWN QUIRK" greps to zero in `src/` and `tests/`** — the
    phrase lives only in memory docs. The gpt-oss owned template's
    remaining `<|return|>`/`<|end|>` quirk is an *owned-template*
    quirk, paid for by `compute_tip_extension`'s canonical tail; it is
    not deletable and should not be confused for a stock accommodation.
  - **The e2e sidecar flip is safe and small**: `session_gemma4.rs` and
    `session_gptoss.rs` copy the same crate-root `templates/*.jinja`
    bytes that `baked` `include_str!`s, verified byte-identical on
    disk. Dropping `install_template_sidecar()` exercises rung 2
    end-to-end and closes session 1's "no test observes rung 2 on a
    real model" gap. Reword the "(is the template sidecar installed?)"
    panic messages to name rung 2 when doing it.
  **Exit gate: `just test ignored`, full tier.** Phase 2's session
  learned this the hard way — the #85 compact pin sat broken for a
  whole session because the fast tier can't see grammar-vs-model
  interaction, and the tell only surfaced when the full ignored tier
  finally ran. Template/re-ingest/grammar changes get the model tier
  before "done", every phase. (Session 3 held to this for both of its
  commits.)
- **Phase 4 — drift alarm. DONE (`85027b3`), see session 3.** Landed as
  `baked::nearest_stock` + a richer rung-3 warning. The "Qwen owned
  template" half was **retired on measurement**: stock already honours
  `preserve_thinking`, which is the only thing aged-thinking continuity
  needed. (Analyzer's second job — bootstrap for new models — is
  unchanged and still available.)
- **Phase 5 — finish the #85 class.**
  (a) Structured output → canonical prelude: **CLOSED as not-to-do,
  session 3.** `Block::Text` renders verbatim; there is no re-render
  contract, so pinning would repeat the `grammar_for_tool_choice`
  regression. Do not reopen without new evidence of a re-serializing
  path.
  (b) Tip on grammar-complete / max-tokens endings: **DONE
  (`83fb5d3`)** — though the diagnosis in gap (2) of
  [[cogito_tool_turn_cache_loss]] was wrong about the mechanism. Those
  endings always *made* a tip; what was broken was the tip's
  prediction. Session 3 has the correction. Rung 4b still depends on
  this being right, and now is.
- **Phase 6 — rung 4 + base-model completion mode (the fun one,
  deliberately last).** Driving use case: **SOUL documents** —
  personality metadata for the Agora seed runner, generated on *base*
  models because open-weights instruct tunes are distillation-
  collapsed into Claude/GPT voice. Division of labor: **grammar
  supplies form, pretraining supplies voice.**
  - Completion-scaffold template: plain text, no special tokens,
    renders Prompt exchanges as document records (interview,
    archive-of-profiles — exact framing is app content). A Jinja
    template can do this; nothing about `ChatTemplate` assumes
    specials.
  - **EOG at end of messages needs design** (Mike's flag). Base
    models have no turn-trained EOG; three stop paths:
    1. grammar-complete (the SOUL case; eager-only grammar — there
       is no trained trigger for lazy mode, so `Auto` degrades to
       eager here) — force-EOS machinery exists;
    2. stop-strings on the scaffold's next-record separator
       (`PredictOptions::{stop_strings, regex_stop_sequences}`
       exist);
    3. pretraining EOS (`<|endoftext|>` at perceived document end) —
       honor as stop.
    Cache note: the "canonical close" becomes multi-token *text*; the
    post-#85 `compute_tip_extension` design (canonical close tokens
    from the re-render, cap 8) accommodates that, but the
    emission→separator seam makes the BPE-boundary case the common
    case — pins must cover it.
  - **Generator + test shape = `examples/few_shot_triage.rs`**
    (Mike's pick): `Prompt::add_examples` seeds exemplar exchanges
    AND the schema so the constraint can't drift from the exemplars;
    under the completion template those exchanges render as document
    records. Same Prompt API, same `.json()` decode — only the
    render changes. e2e test mirrors the example with a base GGUF.
  - Candidate pairing: opt-in generative `minItems` enforcement
    ([[future_work_min_items_for_creative]]) — SOUL docs are the
    first real consumer of the generative direction.
  - Honest caution: recent pretraining contains assistant
    transcripts, so an AI-chat-smelling scaffold re-summons the
    attractor. Scaffold content (human-authored-looking character
    material) matters as much as mechanism; that content is
    seed-runner-side.

## Risks / watch-fors

- **We inherit vendor scaffolding.** Owned templates must reproduce
  load-bearing stock behavior (thinking toggles, date injection,
  pre-opened `<think>` tails). Phase 0's pins are the mitigation.
- **Version skew** is the standing risk ownership creates; the drift
  alarm (Phase 4) plus never-fuzzy detection is the answer.
- **Forced-span injection** (deferred idea from the dialect arc)
  remains compatible follow-up, with a precise caveat: it removes
  *sampler interference* on structural bytes but NOT the
  distributional worry — forced bytes still condition the model
  either way. Habit-derived canonical bytes are the remedy for the
  latter; the two compose.
- Rung 4b quality control: base models ramble. The grammar bounds
  structure; content length wants schema-level caps (maxLength /
  item caps) rather than sampler hacks.

## Related

- [[cogito_tool_turn_cache_loss]] — the #85 findings this plan
  builds on (mechanism, governing requirement, canonical-form facts).
- [[plan_tool_dialects]] — the completed dialect arc (#30); this plan
  extends its probe-first philosophy from templates to *models*.
- Issue #86 — `render_partial` drops `prompt.thinking`; interacts
  with the Cogito template front-rewrite (Phase 2 checks it).
- `examples/few_shot_triage.rs` — the generator/test shape for
  Phase 6.

## Progress (2026-08-04, Fable) — Phase 6 / rung 4b: SOUL forge on a real base model, example-side

Headline: **rung 4b works today with ZERO library changes**, because the
ladder already covers it. Qwen3.5-35B-A3B-Base (converted this session:
`convert_hf_to_gguf.py --outtype q8_0`, `models/Qwen3.5-35B-A3B-Base-Q8_0
.gguf`; the `blk.40 unused tensor` load warnings are the exported MTP
layer, benign) ships a **vestigial VL-instruct `chat_template`** — the
exact case the ladder bullet predicted — so `NoTemplate` never fires and
a rung-1 sidecar carries the whole design. The library rung-4b work
(explicit selection + the no-template rescue in `baked.rs`) remains open;
what landed is the consumer:

- **`templates/completion-scaffold.jinja`** (+ copy as the model's
  `.template.jinja` sidecar): Mike's design — the render is a **bare,
  never-closed JSON array of records**, the kind of scraped data file
  pretraining is full of. No specials, no chat framing, no roles in the
  bytes. Assistant turns are records (`content + ",\n  "`), user turns
  render as ZERO bytes (turn-order ballast for the prompt API), optional
  system text above the `[`. Generation always resumes at an open record
  slot; canonical close is `",\n  "`.
- **`examples/soul_forge.rs`**: exemplar SOUL.json files → 
  `Prompt::add_examples` (schema seeded from the same exemplars, mirror
  `Soul` struct WITHOUT `evolution_log` — the seed runner owns history)
  → one grammar-locked completion per soul, `--n` loop feeding each
  emission back verbatim + `.cache()` breakpoint. Per-record generation
  chosen over a single `Vec<Soul>` completion deliberately: the grammar
  compiler enforces `minItems` only as non-emptiness (filler-entry
  rationale, `schema_constraint_keywords_decision.md`), so a one-shot
  array cannot pin the count; the loop guarantees exactly n while every
  element stays schema-guaranteed. `OutputConfigOptions { allow_thought:
  false, phase_split: false }` is load-bearing — the default optional
  `<think>` limb is a trap on a base model — exposed via new
  `TransportBuilder::output_config_opts` in `examples/utils/args.rs`.

### The copy attractor (measured, and the mechanism matters)

First run (3 exemplars, 2 of them same-y edgy-troll souls, `tactic`
last): the model emitted a **byte-verbatim copy of the `tactic`
exemplar, twice**. Mechanism, not bad luck:

1. Induction: in a record-series document whose records are
   near-duplicates of each other, the copy token carries ~0.99.
2. Default chain is TopK 1024 → **LocallyTypical p=0.5** (Mike confirmed
   these defaults): any token above 0.5 mass leaves the typical cut a
   **singleton** — generation is deterministic regardless of seed.
3. The usual counterweapon can't reach: `constrained_regions` applies
   repetition penalties inside grammar free regions with a **call-local
   accumulator** — prompt-history n-grams are invisible during a
   grammar-constrained emission, so cross-record copying is never
   penalized there. (Window/decay tuning in a sampling sidecar does NOT
   fix this; don't try.)

**Fix is document-side and it worked completely**: 8 genuinely diverse
exemplars (archivist / ethicist / economist / satirist / philosopher /
muckraker / narrative / troll) → two novel souls ("empath", "narrative"),
novel names, coherent field content, valid JSON, clean parse, no sampler
changes. The exemplar set defines the file's internal pattern; a same-y
set legitimately predicts near-duplicate records. Residual sameness
("I am an AI agent who…" openings) is inherited from the exemplar corpus
(instruct-generated, distillation-collapsed) — exactly the problem the
base-model program exists to fix; expect it to fade as forge output
replaces instruct output in the exemplar pool.

### Open / next

- Library rung 4b proper: explicit template selection + vocab-fingerprint
  rescue in `baked.rs`; the scaffold graduates from sidecar to baked.
- Cache: `.cache()` on each fed-back assistant turn should give hash-path
  reuse across the `--n` loop (phase-5b marking rule); not yet measured —
  the example now prints `response.usage` under `--verbose` for exactly
  this.
- e2e pin mirroring the example with a base GGUF (the plan's Phase 6
  test shape) not yet written.
- Sampling for creative diversity (typical p=0.5 is conservative for this
  workload) deliberately untouched — defaults confirmed by Mike; revisit
  only on evidence.

### Round 2 (same session): agentkit as source of truth; --names measured

- `Soul` now comes from **agora-agentkit 0.7.1** (crates.io; optional
  dep, `seed` feature; public path `reactor::seed::Soul` — the `agent`
  module is private). Its misanthropic req ^1.0.0-alpha.7 unifies with
  our alpha.12; schemars same major. The implicit `agora-agentkit`
  feature gates the example and was added to `scripts/test.py`'s
  AGNOSTIC group so the permutation gate keeps compiling it. Exemplar
  `evolution_log`s are cleared before prompting; generations come back
  with `"evolution_log": []` (exemplar mimicry) and agentkit's
  `validate()` runs on each.
- **`--names` (const schema patch) works and is the diversity lever
  that bites**: `properties.name = {"const": …}` compiles to the exact
  literal, the grammar forbids name collisions by construction, and —
  name being the first-generated field — distinct names pulled whole
  records apart (quill → language-obsessive; ember → wild-ideas) from
  the SAME exemplar permutation. `--shuffle` covers the unpinned case.
  Sampling sidecar for the base model: locally_typical p=0.9
  (hand-written, models/ only — sidecars are per-install).
- **Cache measured**: record 2 read 2634/2637 input tokens from cache
  (3 created). The `.cache()` breakpoint on each fed-back assistant
  turn bridges grammar BPE drift exactly as phase 5b designed.
- **Optional fields get skipped**: both generations omitted
  `boundaries` even though every exemplar had it — the grammar's
  optional-property door. If it must be present, patch the schema to
  require it (same trick as --names).
- Shitposter provenance (Mike's question): run 1's troll output was a
  byte-verbatim exemplar copy — zero signal about the base model. Run 2
  (2/8 troll-ish exemplars) generated 0/2 trolls; the base model at
  typical sampling leans agreeable. Exemplar mix is the ~20%-troll
  lever, with --names as the per-record steering wheel.

### Issue #106 filed (same session): constrained-region penalty vs history

Mike's Agora symptom (near-identical sequential posts from similar
agents in one thread) traced to the same mechanism soul_forge hit:
regime (b)'s call-local accumulator means structured-output bodies see
ZERO repetition pressure from prompt history, and the prose fold
excludes tool results, so tool-shaped thread context is absent even
from the persistent corpus. Both by design (determinism linchpin;
digit-penalty case) — the gap is the all-content-in-JSON workload.
Fix direction in the issue: seed `constrained_ngram_stats` from the
folded prose corpus at generation start AFTER the last breakpoint
snapshot (determinism untouched), behind a serde-default-false
`RepetitionOptions` flag. Read #106 before touching
`constrained_ngram_stats`, the seeding fold, or breakpoint snapshots.
