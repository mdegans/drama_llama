# Plan of record: tool-call dialects (template-derived CallSyntax)

Approved by Mike 2026-07-10. Canonical copy lives as a GitHub issue
(see repo issues, "Tool-call dialects" umbrella); this file is the
in-repo twin. Supersedes the sketch in issue #29 (absorbed as Phase
D/Qwen below) and the format-selection knobs in `ToolChoiceOptions`
(`wrap_tags` / `arguments_field` are a proto-dialect this plan
subsumes). Rolls issue #28 (lazy grammar check) into the sequence.

## Progress (2026-07-10, session 1)

- **Phase B LANDED** (`0810ea9` + fix `d7a8cd6`): lazy sample-then-
  check behind `SampleOptions::lazy_grammar` (default still `false`;
  flip is #28 phase 3, pending Mike's GPU tok/s comparison —
  invocation: `DRAMA_LLAMA_GRAMMAR_STATS=1 cargo test --features
  serde json_integration_lazy_grammar -- --ignored --nocapture`).
  Bonus find via the new stats: post-complete constraints kept
  empty-piece reserved tokens, so every constrained Qwen run burned
  the full `max_tokens` invisibly. Fixed in both modes (`d7a8cd6`).
- **Phase A LANDED** (`121aec6` A1, `530c6af` A2):
  `emit_until_rules` (KMP-DFA complement, exhaustive+fuzz tested);
  `DeferredGrammar::feed_trigger`; `deferred_grammar_for_prompt`
  (Auto/absent + tools ⇒ trigger-lazy grammar);
  `RootShape::Eager{thought_pre_opened}` — Qwen thinks under eager
  grammar now (Session detects the pre-opened `<think>` tail per
  render). Priority: eager tool grammar > output_config > auto-lazy.
- **Phase C LANDED** (`014bd17`): `CallSyntax` (serde/TOML),
  differential analyzer (FFI-free core, catch_unwind-guarded probes,
  patches-as-data), 6 vendored fixtures + qwen3.6-gguf dump, all 7
  pins pass (Qwen XML markers byte-exact vs upstream). Sidecar:
  `<model>.dialect.toml`. Vocab cross-check deferred to E.
- **Phase D LANDED** (`898f9cb`, absorbs #29): grammar_source() +
  render_reference() + validate_representable() (UnrepresentableValue
  per amendments) in dialect/emit.rs; re-parse-per-tick lenient
  envelope parser in dialect/parse.rs (pre_opened_reasoning = the
  #27 fix). Reconstruction harness green on all 6 fixtures.
  NOTE deliberate divergence: args emitted SORTED BY KEY (minijinja
  alphabetizes re-renders; also closes the duplicate-optional hole).
- **Phase E LANDED (code)** (2026-07-10, session 2, Fable): Session
  owns `dialect: CallSyntax` — analyzed at `from_engine` (never
  gates a load; `Family::None` fallback = hermes_json for enforcement
  until Phase F), `<model>.dialect.toml` / `parent/dialect.toml`
  sidecar override at `from_path*`, `with_dialect` builder,
  `vocab_cross_check` advisory (outer markers must be single tokens;
  suspects → tracing::debug). `resolve_grammar` goes through
  `dialect::grammar_source` (Method/Any eager w/ pre-opened anchor;
  Auto → lazy deferred on `syntax.trigger()`; parallel gated on
  `!disable_parallel_tool_use && per_call_start non-empty` — section-
  only dialects stay single-call). `BlockParser` DELETED
  (`session/parse.rs` gone, exports removed); batch path parses once
  via `parse_text(Final)`, streaming via new
  `dialect::StreamParser` (re-parse-per-tick + prose byte-deltas;
  holdback covers open AND close markers — a partial `</tool_call>`
  degrading to prose then reclassifying was caught by the
  chunking-invariance test). Canonicalization gate: tip_hash stored
  only when `render_extended(prompt+blocks)` == rendered_prompt ++
  raw emission (byte prefix); else LCP-only fallback.
  `ToolChoiceOptions`/`with_tool_choice_opts` deprecated (shim maps
  to CallSyntax). `SessionError::Dialect` added. All non-ignored
  tests green (294 lib + integration), fmt clean.
- **Phase E e2e GREEN — 9/9 on Qwen3.6-35B-A3B, RTX 3090**
  (2026-07-10, session 2 cont.; Mike okayed GPU runs on this box —
  the GPU-runs-are-Mike's rule was about macOS/Metal instability,
  Nvidia is fine). First runs failed 5/9 with one shared signature:
  sampled generation died at the second parameter. Greedy top-k
  tracing (`Session::top_k_trace` + a manual predictor-loop example)
  isolated THREE stacked causes, all fixed in `821639e`:
  1. empty-piece reserved tokens legal mid-parse → livelock fuel
     (extends d7a8cd6 to active constraints);
  2. EOG passed byte-acceptance inside until() regions (`<|im_end|>`
     literal bytes are legal raw-value content!) → predictor stop
     mid-call. Fix: reject EOG **by id** while constraint incomplete
     (llama.cpp grammar-sampler rule; `<|return|>` stays viable for
     Phase G because Harmony's grammar completes first);
  3. **repetition penalty vs delimiter exit**: the until() exit needs
     exact bytes `\n</parameter>\n` built from tokens the call
     already used 5-10× ("\n", "</", "parameter", ">") — penalty
     crushed exactly those, sampling thrashing 1024 tokens of
     near-misses. Fix: penalty suspended while any byte-constraint in
     the chain is incomplete (constrained spans also no longer seed
     penalty stats). Greedy was immune to all three — which is why
     Phase A–D testing (greedy + FFI-free) never saw them.
  Also relaxed the lazy-vs-masked RNG-stream-equality test (kept-set
  can now be a singleton → no draw; token equality + fallback
  bit-exactness are the real contracts). Test-side fixes: round-trip
  must mirror session render opts (no stray `enable_thinking`);
  thinking e2e must set `prompt.thinking` to get a pre-opened tag.
  #27 + #29 closed. Behavior change: streaming yields reasoning as
  one `Thought` on close (pre-#27 it streamed mislabeled as `Text`);
  thought-delta streaming is #26 territory — and per Mike, #26
  should yield `misanthropic::stream::Event` + reuse its `StreamExt`
  machinery rather than a bespoke enum.
- **Follow-up (not blocking)**: argument *fidelity* under sampling —
  with the penalty suspended the structure completes, but
  locally-typical can still pick a low-quality value (observed: empty
  `string` param mid-debug; final green runs filled "strawberry"
  correctly). If tool-arg quality regresses in blallama, consider
  greedy-inside-constraint or a constrained-span sampler profile.
- Gemma 4 + gpt-oss GGUFs downloaded to `models/`. NOTE: it's Gemma
  **4** (not 3) — llama.cpp gives it a hand-built handler, so Phase F
  may become "native weird template" rather than (or in addition to)
  `Instructed`. Re-scope F when its template gets probed.
- **Phase F RE-SCOPED + LANDED** (2026-07-10, session 3, Fable; Mike
  okayed "native only, Instructed deferred — play it by ear"). Probing
  the Gemma 4 GGUF template killed the Instructed premise: it has FULL
  native tool support, hand-built upstream
  (`common_chat_params_init_gemma4`). Format:
  `<|tool_call>call:name{key:value,…}<tool_call|>` — brace dict with
  BARE keys, strings quoted by the special token `<|"|>` (no in-band
  escaping possible → recursive `UnrepresentableValue`), values
  otherwise JSON-ish; `<|channel>thought\n…\n<channel|>` reasoning
  (renders ONLY on tool-call turns after last user =
  `ReasoningMode::ToolsOnly`); asymmetric `<|x>`/`<x|>` markers, all
  single vocab tokens (ids 46–106). Implementation:
  - `Family::TagWithDict` + `CallSyntax::gemma4()` baked constant +
    analyzer sniff patch (`<|tool_call>call:` && `<|"|>` in source →
    wholesale overwrite, upstream-style).
  - Dict-encoded schema compiler (`schema_to_dict_gbnf`) — sorted-
    in-place object layout (optionals before first required carry
    trailing comma; after, leading) matching `| dictsort` re-renders;
    `dict_encode_value` is the single byte-encoding source for
    grammar literals AND `render_reference`.
  - minijinja probe findings baked in: null renders `none` (grammar
    accepts null/none/None, parse coerces all → JSON null, render
    emits `none`); floats are ryu-shortest (serde_json matches:
    `1.5e10` → `15000000000.0`).
  - Parser: recursive dict-value reader; channel noise per upstream
    matrix (empty thoughts DROP, bare `<|channel>` and unmatched
    `<channel|>` swallowed — full test-matrix port from
    `tests/test-chat.cpp` "Google Gemma 4").
  - **Thought re-ingest convention** (`ReasoningReingest` on
    `ReasoningSyntax` → `RenderOptions::thought_reingest`, wired in
    `from_engine`/`with_dialect`): Gemma wants thoughts as the
    message `reasoning`/`reasoning_content` FIELDS, not inline
    `<think>` (which its template would NOT strip → content
    pollution). Also fixed alongside: `tool_call_message` now carries
    ALL ToolUse blocks (was first-only — parallel-call re-ingest was
    silently lossy for every dialect).
  - **Turn-exit discovery (e2e round 1, 6/7)**: after `<tool_call|>`
    Gemma's trained continuation is `<|tool_response>` (its template
    keeps the call turn OPEN awaiting in-turn responses). Masking it
    made the model loop identical calls to max_tokens. Fix:
    `CallSyntax::tool_response_start` — grammar REQUIRES it as turn
    exit (it's also the canonical re-render byte, so byte-stability
    improves), after which nothing is legal → sampler's complete-
    constraint logic forces EOG = deterministic stop; parser
    swallows it as envelope (batch + streaming holdback).
  - Reconstruction green through the REAL template incl. thought-via-
    field, null→none, nested dict re-sort, parallel calls.
  - KNOWN QUIRK 1 (accepted): Gemma's template renders assistant
    `content` AFTER tool_calls; an Auto-mode "prose then call"
    emission re-renders in the other order → canonicalization LCP
    fallback for that turn. Cache-safe, one re-prefill; not fixable
    at grammar level.
  - KNOWN QUIRK 2 (accepted, e2e round 2): the NON-thinking
    generation prompt ends with a pre-closed empty thought scaffold
    `<|channel>thought\n<channel|>` that re-ingested turns do NOT
    reproduce → same LCP fallback, shared prefix ends at
    `<|turn>model\n`, one-assistant-turn re-prefill per non-thinking
    turn (prefix_cache e2e still passes — the big prefix hits).
    Thinking mode has no scaffold → full byte-stability. Sibling of
    upstream's chat.cpp:1223 workaround.
  - Bonus fix: `trim_eos` now also trims the EOT piece — Gemma
    splits eos (`<eos>` id 1) from eot (`<turn|>` id 106) and
    `complete_text` was leaving a trailing `<turn|>` in the text
    (Qwen never showed it: its eot == eos piece).
  - `Instructed` dialect DEFERRED (no on-disk model needs it;
    `Family::None` keeps the hermes_json enforcement fallback).
  - e2e: `tests/session_gemma4.rs` (7 tests, `--features serde,cuda`
    — plain `serde` builds CPU-only and 31B crawls; that mistake cost
    round 1). GPU runs on this box are fine per Phase E note.
- **Phase F follow-up: cache-stable template sidecar** (2026-07-10,
  session 3 cont.; Mike: "willing to sacrifice context for reasoning
  quality" — keep aged thinking). Both known quirks above are now
  FIXED for deployments that install the sidecar; only quirk 1
  (content-after-calls ordering) remains accepted.
  - `tests/fixtures/templates/gemma4-cache-stable.jinja` — patch of
    the GGUF template: model messages ALWAYS render the thinking
    channel (real reasoning gated on `preserve_thinking` |
    last-turn; the empty `<|channel>thought\n<channel|>` scaffold
    otherwise). Rationale: the model's emission always begins with a
    thought block (it emits the empty scaffold itself when the
    prompt omits it — observed e2e), so the stock template dropping
    it on re-ingest is the template's bug. Note keeping aged
    thinking is the ONLY cache-stable option — the thought bytes are
    already in the KV; older Anthropic models dropped thinking too
    (and stopped).
  - New sidecar kind: `<model>.template.jinja` (GGUF) /
    `parent/template.jinja` (moeflux) — raw Jinja override of the
    embedded template. `sidecar::load_template_source` +
    `Session::set_template_source` (recompiles ChatTemplate AND
    re-analyzes the dialect from the SAME source — lockstep
    invariant); applied before the dialect sidecar so an explicit
    dialect override still wins. `sidecar` module un-gated from
    `toml` (template sidecar is toml-free; TOML items gated
    individually).
  - Pins: `gemma4_cache_stable_prefix_continuity` (FFI-free, the
    cache property itself: generation-prompt render is a byte PREFIX
    of the follow-up render — non-thinking scaffold, thinking
    byte-exact thought, aged thought preserved);
    `reconstruct_gemma4_cache_stable`; analyzer sniff pin covers the
    patched fixture. e2e installs the sidecar (copy fixture →
    `models/<model>.template.jinja`) and the round-trip test asserts
    the STRICT byte prefix (no LCP fallback — assistant prefill
    skippable).
  - blallama deployment note: ship `gemma4-cache-stable.jinja` as
    `<model>.template.jinja` next to the Gemma GGUF.

## Problem

The tool-call *format* a model is trained on is implied by its chat
template, and drama_llama currently hardcodes one format assumption in
three unrelated places that must agree and don't:

1. **Enforce** — `tool_choice.rs` grammar forces the Hermes envelope
   `<tool_call>{json}</tool_call>`.
2. **Parse** — `session/parse.rs` only understands fixed
   `<think>` / `<tool_call>`+JSON tags.
3. **Re-ingest** — the chat template renders assistant `tool_calls`
   in its *native* format.

Qwen3.6 (Unsloth template) trains an XML-ish shape
(`<tool_call>\n<function=NAME>\n<parameter=KEY>\nvalue\n</parameter>…`)
with raw unquoted parameter values. Forcing JSON there requires a
system-prompt retcon, is off-distribution for the model (argument-
fidelity cost, see #29), and — the sharpest edge — **breaks the prefix
cache**: emission (forced JSON) ≠ re-render (native XML), so every
tool-call turn invalidates the cache, and the hash-keyed tip path in
`session/mod.rs` can splice KV state whose bytes don't match the new
render (`compute_tip_hash` hashes the canonical re-render while the KV
holds raw emission).

Goal: Qwen3.6, Gemma 3, and gpt-oss all first-class in blallama —
native tool-call formats, prompt cache + breakpoints stable across
tool turns, no retcon.

## Design center

**Round-trip byte-stability**: `render(parse(emission))` must
reproduce `emission` byte-for-byte within the assistant span. This is
a *cache-correctness invariant*, not a nicety. It becomes a standing
property test (see "reconstruction harness" below; llama.cpp's
`expect_reconstruction()` in `tests/test-chat.cpp:1312-1361` is the
model).

**Single source of truth**: one per-model `CallSyntax` value drives
both the grammar emitter and the parser. Render stays the template's
job; `CallSyntax` *agrees* with the template because it is **derived
from** the template by differential probing.

## Upstream validation (llama.cpp @ b9754 / 52b3df00)

Read 2026-07-10 from `~/Projects/llama-cpp-sys/external/llama.cpp`
(the pin we build against). Independent convergence on the same
design, more radical than our sketch:

- Old ~20-format registry **deleted**. Everything is a PEG combinator
  tree — one IR that both parses output and emits GBNF
  (`common/peg-parser.cpp:1713` `build_grammar`). Parsers serialize to
  JSON (`common_peg_arena::save/load`). Dialects-as-data, shipping.
- **Probe-first**: `common/chat-diff-analyzer.cpp` renders the
  template with sentinel payloads (`FFF_FIRST_FUN_F`, `XXXX`/`YYYY`,
  0/1/2 calls, 0/1/2 args, `enable_thinking` flipped, differing call
  IDs) and diffs the renders to extract markers. Formats classify into
  three families: `JSON_NATIVE`, `TAG_WITH_JSON`, `TAG_WITH_TAGGED`.
  ~10 hand-built PEG handlers remain only for structurally weird
  templates (gpt-oss, Kimi K2, Gemma 4, DeepSeek V3.2 …), selected by
  source sniff-strings (`chat.cpp:2337-2420`), plus ~10 post-analysis
  patch lambdas that just assign analysis fields
  (`chat-diff-analyzer.cpp:36-177`).
- **Qwen3-Coder XML is not sniffed** — discovered empirically as
  `TAG_WITH_TAGGED` with markers (newlines significant):
  `per_call_start="<tool_call>\n"`, `function.name_prefix="<function="`,
  `name_suffix=">\n"`, `function.close="</function>\n"`,
  `arguments.name_prefix="<parameter="`, `name_suffix=">\n"`,
  `value_suffix="\n</parameter>\n"`, `per_call_end="</tool_call>"`.
  (Expectations pinned in `tests/test-chat-auto-parser.cpp:1358-1376`.)
- **Typed values**: per-param, if the schema `resolves_to_string` →
  raw text `until(value_suffix)`; else a JSON value schema-compiled to
  GBNF (`chat-auto-parser-generator.cpp:390-402`). Required params in
  declaration order; optionals trail, any order. Parse-side mapper
  coerces: raw → JSON-escaped string; `True`/`'…'` pythonisms
  normalized; bounded brace-healing on tool-close
  (`chat-peg-parser.cpp:83-185, 365-417`).
- **`until` compiles to GBNF as the complement of an Aho–Corasick
  DFA** (`gbnf_excluding_grammar`, `peg-parser.cpp:1609`, ref upstream
  PR #24839); the pinned commit adds the inclusion variant `ac()`.
  This is the one new grammar-engine construct we need.
- **Lazy grammar**: `grammar_lazy ⇔ tool_choice=auto`; trigger WORD =
  section/per-call start (`<tool_call>`); `required` → eager grammar
  from token 0 *including* an optional reasoning rule, anchored on the
  rendered generation-prompt tail (handles Qwen's pre-opened
  `<think>\n`). Activation is NOT gated on `</think>` — accepted
  trade-off. GBNF can't express lookahead; PEG peeks drop from the
  grammar, so grammar is strictly looser than the parser and the
  parser is the source of truth.
- **Streaming**: full re-parse of accumulated text per tick, lenient
  mode with partial-AST + `NEED_MORE_INPUT`; `atomic()` suppresses
  half-parsed nodes. No incremental parser state.
- **Generic handler for tool-less templates no longer exists**
  upstream — tools silently dropped. Our `Instructed` dialect (Gemma
  3) is beyond-parity capability, not ported behavior.
- **gpt-oss/Harmony**: hand-built grammar (`chat.cpp:1053-1210`),
  detected by `<|channel|>` substring; parsing 100% shared machinery.
  Warts (all documented in code): stray `<|channel|>commentary
  to=assistant` prefixes on 20b, `<|return|>`→`<|end|>` rewrite when
  re-ingesting a final turn (issue #15417), recipient legal in two
  header positions, `<|constrain|>json` optional, multiple analysis
  blocks per turn concatenate. **EOG for `<|return|>`/`<|call|>` is
  handled at vocab level inside libllama** (`llama-vocab.cpp:
  2762-2869`) — free for our llama.cpp backend via FFI; moeflux would
  need it reimplemented.
- Test corpus: `models/templates/*.jinja` (~50 real templates incl.
  Qwen3-Coder, Qwen3.5, Gemma 2/4, gpt-oss-120b) + pinned analysis
  expectations + reconstruction checks. MIT; we can vendor a subset.

## Decisions (Mike, 2026-07-10)

- Probe-first architecture; hardcoded dialects are *overrides*
  ("dialects as data", patches-as-data).
- `CallSyntax` struct (Serialize/Deserialize) adopting llama.cpp's
  field vocabulary; baked constants for known families; per-model
  sidecar override. **GBNF fragments as data** — the full grammar is
  generated per-request from tool schemas; `CallSyntax` parameterizes
  the emitter.
- **No PEG-arena port.** We keep our Rust GBNF engine; `CallSyntax`
  is the single source with two small compilers: (a) GBNF emitter,
  (b) generic envelope parser. Revisit only if Harmony strains it.
- Analyzer runs **at load** (`Session::from_path`, ~30 cheap minijinja
  renders) with optional sidecar override.
- Streaming parse: adopt **re-parse-per-tick lenient** design
  (replaces the incremental `BlockParser` state machine and its
  partial-tag-holdback FIXME).
- Issue #28 (lazy O(1) grammar check) rolls into this sequence — same
  sampling-path code area as trigger-lazy activation.

## Amendments (Mike + Claude, 2026-07-10 pre-implementation review)

- **Unrepresentable raw values → typed error at ingest.** Tagged
  dialects have no in-band escape the model was trained on (the
  template dumps raw bytes between markers), so a client-constructed
  string arg containing the dialect's close delimiter (e.g.
  `</parameter>`) cannot round-trip. Escaping is rejected — it would
  invent a private dialect the model has never seen. The library
  raises a typed, catchable error at message-conversion time
  (`UnrepresentableValue { tool, param, delimiter }`-shaped); the
  consuming app may catch and substitute (e.g. replace the input with
  a warning to the model). Policy stays in the app, per the 0.7
  Vocab-removal philosophy. JSON dialects are immune (native
  escaping). Distinct from *awkward-but-legal* values (trailing
  newlines, JSON-looking strings), which must round-trip and get
  Phase D harness fixtures, not errors.
- **Phase D harness: adversarial fixtures required** — trailing
  whitespace/newlines in raw values, embedded close-delimiters
  (expect typed error), values that look like JSON, unicode.
- **Phase C phrasing:** analyzer *core* is FFI-free (testable against
  .jinja fixtures alone); the vocab cross-check is an optional
  post-pass taking `&Model`, available at `Session::from_path` where
  the model is already loaded.
- **Re-parse-per-tick is deliberately O(n²)** over a generation
  (llama.cpp ships the same; outputs are small). Comment it as
  intentional so it isn't "optimized" back into an incremental state
  machine — that's the BlockParser we're deleting. Full re-parse
  yields a complete partial AST each tick, which is what #26
  (streaming Events, 0.9) needs; D is a quiet prerequisite for #26.
- **Phase ordering pick:** B before A's trigger-lazy half (both
  rewrite `sample_token`; write activation logic once against the
  post-#28 path shape). A's `until` emitter is independent and may
  land in either order / parallel.

## Cache-stability strategy

Wire constraint: raw emission formatting cannot survive the Anthropic
wire (clients echo back `input` as parsed JSON), so unlike llama.cpp's
test harness we cannot rely on preserving raw args end-to-end.
Two-layer strategy instead:

1. **Grammar forces canonical bytes** where active (Method/Any):
   emitter produces exactly the serialization the template re-render
   will produce (match minijinja `tojson` compactness / `| string`
   scalar forms). Drift ≈ 0 by construction.
2. **Post-generation canonicalization check** (Auto or residual
   drift): Session compares emission to the canonical re-render of
   the parsed blocks; on mismatch, invalidate/re-prefill just the
   divergent assistant-turn tail so KV always holds canonical bytes.
   Cheap (one message), and makes `compute_tip_hash`'s assumption
   true by construction.

The reconstruction property test guards both layers.

## Phases

Each phase ≈ one implementation session (Opus 4.8 / Sonnet 5 per the
planning split; elastic — merge or split as sessions allow). Every
phase lands green tests; `cargo test` stays the gate. GPU runs are
Mike's.

### Phase A — grammar engine: `until` + trigger-lazy
- `emit_until_rules(delim)` in `grammar_compile.rs`: GBNF complement-
  of-AC-DFA for "raw bytes not containing DELIM, terminated by DELIM"
  (multi-byte generalization of the `think_char` trick). Crib
  `gbnf_excluding_grammar` semantics; differential-fuzz vs a naive
  matcher (`grammar_fuzz.rs` extension).
- Generalize `DeferredGrammar` activation: substring WORD triggers
  (activate when trigger appears in generated bytes; trigger bytes
  feed into grammar state), per llama.cpp lazy-pattern semantics.
  `tool_choice=Auto` + tools ⇒ lazy grammar w/ dialect trigger.
- Eager (`Any`/`Method`) grammar root gains the optional-reasoning
  prefix anchored on the rendered tail — fixes the current
  Qwen-can't-think-under-grammar bug (template pre-opens `<think>\n`;
  our root demands `<think>` or the call as first bytes).

### Phase B — #28: lazy O(1) sample-then-check
Already planned (issue #28, `plan_lazy_grammar_check.md`). Sequenced
here because it touches the same `sample_token` path Phase A modifies.
Order A↔B flexible.

### Phase C — `CallSyntax` + template analyzer
- `CallSyntax` (serde): section/per-call markers, function
  open(name_prefix/suffix)/close, arg name/value markers, separator,
  args-wrapper, call-id position, value encoding, reasoning tags,
  content wrapping, trigger, preserved tokens. Baked constants:
  HermesJson, Llama31Json (+ QwenXml as the expected analyzer output).
- Analyzer (pure minijinja, no FFI): sentinel differential renders →
  diff-split → marker segmentation (`<…>`/`[…]` spans; strengthen
  with vocab cross-check: candidate markers that tokenize to a single
  special token are high-confidence) → family classification →
  marker extraction. Patches-as-data list post-analysis.
- Sidecar override: `<model>.dialect.toml` (same discovery convention
  as `sampling.toml`).
- Fixtures: vendor a subset of llama.cpp `models/templates/*.jinja`
  (MIT, attribute) + our Qwen3.6 GGUF template; pin expected
  `CallSyntax` per template (mirror `test-chat-auto-parser.cpp`).

### Phase D — emitter + parser from `CallSyntax` (absorbs #29)
- GBNF emitter: TAG_WITH_TAGGED (per-tool literal names; required
  args in declaration order; optionals trail any-order; string params
  = raw-until-close via Phase A; non-string = schema-compiled JSON) +
  TAG_WITH_JSON (subsumes current `tool_choice.rs` path) +
  JSON_NATIVE. Multi-call gated on parallel-tool-calls.
- Generic envelope parser driven by `CallSyntax`: re-parse-per-tick
  lenient; schema-guided coercion (raw → typed JSON); healing rules
  from llama.cpp mapper (escape raw strings, normalize pythonisms,
  bounded brace-close). Replaces fixed-tag `BlockParser`.
- Reconstruction harness: parse → append → re-render → assert byte
  prefix. Runs per dialect per fixture; also as `#[ignore]` e2e
  against real models.

### Phase E — Session integration + Qwen e2e
- `Session` resolves dialect at `from_path` (analyzer + sidecar);
  `resolve_grammar` and the parse path go through the dialect;
  `ToolChoiceOptions` deprecated/subsumed. Canonicalization check
  post-generation (cache-stability layer 2).
- blallama e2e on Qwen3.6-35B-A3B: native XML calls under
  Auto/Any/Method, thinking works under grammar, no retcon, prompt
  cache + breakpoints stable across tool turns (`#[ignore]` tests).

### Phase F — Gemma 3: `Instructed` dialect
- Template has no tool support → dialect owns: system-prompt
  injection (library-owned, no user retcon), **our** rendering of
  tool_use/tool_result turns (render-side hook in the
  `chat_template` message conversion for templates whose caps lack
  tool_calls), Hermes-JSON CallSyntax for emit/parse. Round-trip
  stable by construction (we control both sides).
- Prereq: Gemma 3 GGUF on disk (5G link — resumable download).

### Phase G — gpt-oss: Harmony dialect
- Hand-built dialect (data can't express channels): grammar + parser
  for `<|start|>assistant<|channel|>…<|message|>…` structure; both
  recipient positions; optional `<|constrain|>`; stray-commentary
  swallow rule; multiple analysis blocks → concatenated Thought;
  `<|return|>`→`<|end|>` re-ingest rewrite.
- Stop tokens: verify our Engine honors libllama's `special_eog_ids`
  (should be free via FFI); moeflux backend explicitly out of scope
  this phase.
- Prereq: gpt-oss GGUF on disk.

## Pruning (do alongside, not as its own session)

- #29: absorbed into Phase D — comment + close when D lands.
- #27 (Session::parse misses Qwen XML on unforced path): fixed by
  the Phase D/E generic envelope parser — comment + close alongside
  #29.
- `.claude/memory/qwen36_xml_tool_call_shape.md`: fold anything not
  already here into this doc, then delete.
- `future_work_grammar_speculation.md`: already superseded by #28;
  delete when #28 lands.
- `ToolChoiceOptions` docs/tests referencing `wrap_tags` as the
  extension point: update when Phase E deprecates it.

## Standing risks / watch-fors

- minijinja vs upstream jinja divergence in sentinel probing (e.g.
  `| string` on bools: minijinja `"true"` vs Python `"True"`). The
  reconstruction harness catches this; keep fixtures for both.
- Analyzer misdetection on unseen finetune templates → patches-as-
  data list, sidecar override, and (worst case) `Instructed`
  fallback. Never hard-fail a load on analysis failure; fall back to
  content-only + warning (diverges from upstream's hard error,
  deliberately).
- Optionals-any-order grammar admits duplicate optional args
  (upstream has same hole); parser dedups, validator rejects.
