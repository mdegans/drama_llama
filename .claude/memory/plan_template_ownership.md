# Plan of record: owned chat templates, loading ladder, base-model fallback

> **STATUS: PHASES 0–2 LANDED** (phase 2: 2026-07-27, session 2,
> Fable). Mike confirmed the full-commitment direction in-session
> ("Cool. Your plan.") after the #85 arc; scope details delegated.
> GitHub twin: [issue #88](https://github.com/mdegans/drama_llama/issues/88).
> Supersedes the "own templates on measured need" stance argued
> earlier the same session, and the sidecar-only distribution model.

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
  Agora side.

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
- **Phase 3 — promote existing sidecars.** `gemma4-cache-stable` and
  `gptoss-cache-stable` become baked; delete the stock accommodations
  that go dead (`ReasoningReingest::Thinking` contortions, KNOWN
  QUIRK LCP paths); unify re-ingest on `reasoning_content`.
- **Phase 4 — Qwen owned template + drift alarm.** Smallest template
  delta. Analyzer repurposed: at load, analyze the *embedded stock*
  template, diff its dialect against our owned one, warn on mismatch
  — this is how we hear that upstream moved. (Analyzer's second job:
  bootstrap for new models.)
- **Phase 5 — finish the #85 class.** (a) Structured output moves to
  the canonical prelude (the "phase 4" deferred in `86c9fe4` — same
  latent bug when a JSON response replays as history; under ownership
  that serialization is ours too). (b) **Tip creation on
  grammar-complete and max-tokens endings** — gap (2) in
  [[cogito_tool_turn_cache_loss]]: `compute_tip_extension` only fires
  in the stop-sequence branch, violating the governing requirement
  (resume identical down to RNG position) for every model. Rung 4b
  makes this load-bearing (grammar-complete IS its normal ending).
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
