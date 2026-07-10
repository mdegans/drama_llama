# Plan of record: tool-call dialects (template-derived CallSyntax)

Approved by Mike 2026-07-10. Canonical copy lives as a GitHub issue
(see repo issues, "Tool-call dialects" umbrella); this file is the
in-repo twin. Supersedes the sketch in issue #29 (absorbed as Phase
D/Qwen below) and the format-selection knobs in `ToolChoiceOptions`
(`wrap_tags` / `arguments_field` are a proto-dialect this plan
subsumes). Rolls issue #28 (lazy grammar check) into the sequence.

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
