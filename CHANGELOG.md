# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **`--n-ctx` is now a ceiling, capped at each model's window.** The
  served context is the smallest of `--n-ctx`, a `load.toml`'s
  `n_ctx` and the model's window, so a sidecar can lower a model's
  context but no longer raise it past `--n-ctx`. Before, `--n-ctx`
  was used as given for any model without a sidecar: at 262144,
  cogito (131072 trained, 256 KiB of KV per token) allocated 64 GiB
  of KV and ran Metal out of memory. A fleet that raised one model
  with a sidecar now launches with the larger `--n-ctx` and lowers
  the others (e.g. Mistral Small 4, whose window is 1M).

### Added

- **KV cache types: `--cache-type-k` / `--cache-type-v`, and
  `cache_type_k` / `cache_type_v` in `<model>.load.toml`** (which beat
  the flags). llama.cpp's set and names (`f16`, `q8_0`, `q4_0`,
  `iq4_nl`, …), as `KvCacheType` and `LlamaCppOptions::cache_type_*`.
  `q8_0` halves the KV cache, which is what bounds a long context on a
  dense model. A quantized V needs Flash Attention; llama.cpp turns it
  on under the default `Auto` and refuses the context if it is forced
  off.
- **`rope_scale` in `<model>.load.toml`**: a YaRN factor over the
  model's original window (its GGUF's
  `rope.scaling.original_context_length`, else its context length).
  It stretches the window to `original × rope_scale` (Qwen3.5+:
  `4` takes 262144 to 1M), and `/v1/models` advertises the stretched
  window. Opt-in per model: static YaRN costs some short-context
  quality, and a GGUF that ships scaling (gpt-oss, Mistral Small 4)
  was trained with its own factor.

### Fixed

- **A client that leaves mid-turn no longer frees the model for a
  second load.** Completions and `count_tokens` run as detached tasks
  that keep the session lock until the work ends; a retry meanwhile is
  answered 529. Before, the retry loaded another copy of the model
  beside the running generation, and Metal ran out of memory (live on
  2026-10-02 with Mistral Small 4, and on 2026-10-08 with gpt-oss).
- **A YaRN context over a shared model is refused**
  (`NewError::SharedModel`). llama.cpp rewrites a model's trained
  context when a context rescales it, so building one while another
  handle reads the model would race; the load path holds the only
  handle and is unaffected.
- **The escaped-closer repair reaches past escaped whitespace** ([#148]).
  It rolled back only to the newest backslash-bearing token, so a value
  ending `\"` + an escaped `\n` or `\t` + `}` (the escape's backslash
  newer than the quote's) fell back as `shape_mismatch`. The repair now
  keeps the last four backslash marks and rewinds to the one holding the
  quote's backslash; a longer escaped-whitespace tail still falls back.
- **The `escaped_closer_repair` event logs the text at the overrule**
  ([#148]) as `site`: the overrule's own `tail` is the turn's end, after
  the model wrote on, so it never showed what the shape check judged.

## [0.9.0] — 2026-10-05

### Breaking changes

API (from `cargo semver-checks` against `v0.8.3`, plus signature
changes it does not check):

- `IgnoreCategory` gains `Markdown` and `Numbers` (it is not
  `#[non_exhaustive]`): exhaustive matches break, `IgnoreCategory::ALL`
  has five entries, and `Punctuation`'s discriminant and `Ord` position
  move.
- `dialect::Leniency` gains `Clipped`.
- `SessionError::InjectedSpecialToken` carries `violations` in place of
  `token` / `piece`; `SessionError::GrammarViolation::partial_output` is
  a `Content`, not a `String`.
- `LlamaCppDecoder::{memory_clear, memory_seq_rm, memory_seq_cp,
  memory_seq_keep, memory_seq_add, memory_seq_div}` take `&mut self`.
- `FromPath` has a required method, `peek`.
- `grammar_for_output_config`, `compile_output_config`,
  `output_config::grammar_for_prompt` and `compile_prompt_output_config`
  take a third parameter (the schema limits).
- `RenderOptions` gains `efforts` and `literals`, `ToolChoiceOptions`
  gains `schema_limits`, and `OutputConfigOptions` gains
  `thought_separator`, `framing`, `thought_open`, `thought_close` and
  `schema_limits`: struct literals without `..Default::default()` break.
- `NewError` gains variants ahead of the old ones (their discriminants
  move) and is no longer `UnwindSafe` / `RefUnwindSafe`;
  `backend::MemoryRmError` is now `Copy`.
- `schema_to_gbnf` returns `Result<(), SchemaError>` and is
  `#[doc(hidden)]`.

Behavior:

- **Usage:** `input_tokens` is only the uncached tail, as on Anthropic;
  sum the three input counters for the prompt size.
- **An empty tool-use id is an error** on every model with reserved
  pieces; prompt content that spells a special token is read as text
  rather than refused, and `InjectedSpecialToken` now means a
  drama_llama bug (a 500 on blallama).
- **A forced `tool_choice` allows parallel calls**; set
  `disable_parallel_tool_use` for exactly one.
- **A turn cut by `max_tokens` or a stop sequence is a response**, its
  cut call truncated as Anthropic truncates it, not a
  `GrammarViolation`. Gate dispatch on `stop_reason: tool_use`.
- **Sliding-window models get a window-sized KV cache** by default
  (`--swa-full` / `LlamaCppOptions::swa_full` restores the old size).
- **The repetition penalty sees history** (three seeding flags, on by
  default, existing sidecars included), and its `window_size`, `decay`
  and `penalty_freq` defaults changed; an existing
  `<model>.sampling.toml` pins the old three until edited or deleted.
- **`CallSyntax::trigger()`** for the marker dialects is the bare
  special (`"<tool_call>"`, not `"<tool_call>\n"`).

### Changed

- **Usage is counted the way Anthropic counts it.** `input_tokens`,
  `cache_read_input_tokens` and `cache_creation_input_tokens` are
  disjoint and sum to `count_tokens`: the read is the restored prefix,
  creation runs from there to the last breakpoint, and `input_tokens` is
  only the tail. `input_tokens` used to be the whole prompt, so a client
  summing the three counted it twice. With the cache off it is still the
  whole prompt.
- **Tool-call JSON has one spelling, the template's ([#85], [#88]).**
  The grammar allowed many byte spellings of a call's arguments and the
  re-render wrote one, so a replayed call turn lost its cache tip. The
  analyzer now measures the template's spacing
  (`ArgumentsSyntax::json_spacing`) and the grammar and
  `render_reference` both follow it. `tojson` no longer HTML-escapes
  (`'`, `&`, `<`, `>`), and templates gain a `json_dumps` filter.
- **cogito's tools preamble allows up to three calls a turn**, and its
  tracked sidecar sets `max_tool_calls_per_turn = 3`. With parallel
  calls in its history it escalated to sixty calls a turn and loops that
  ran to `max_tokens`.
- **Dependencies:** `llama-cpp-sys-3` 0.8.4, `minijinja` 2.24 (which
  prints a bare bool or null Python-style; the bakes spell them
  themselves, [#120]), `misanthropic` 1.0.0-alpha.18 (through
  alpha.21), `agora-agentkit` 0.62 (the `soul_forge` example only).
- **An escaped closer is repaired in place instead of resampling the
  turn (#140).** When the model writes `\"}` where it means `"}` and
  reaches for the end of its turn, the grammar still holds the string
  open and the overrule used to reject the whole turn (blallama then
  redrew it from the start: minutes per draw on Mistral Small 4). The
  session now rolls back to the start of the token holding that
  backslash — KV, tokens, matchers and text — writes again what the
  token held before it, and redraws with backslash-led tokens banned
  for one step, so the model writes the `"` it meant. Once a turn,
  only on a model whose KV a truncate rewinds (dense), and on a stream
  only while nothing past the rollback point has been released (a tool
  call is; an answer's text is not). Anything else, or a redraw that
  overrules again, stands as the `GrammarViolation` it was. Each try is
  logged as `escaped_closer_repair` with its `outcome` (`repaired`,
  `repeat_overrule`, `unrestorable`, `shape_mismatch`, `streamed`) and
  the tokens `rolled_back`.
- **`BlockStream::stop_reason` returns
  `Option<(StopReason, Option<&str>)>`** — the reason is no longer an
  `Option`, because a finished turn always has one (below). (New in
  this release; this only affects code written against `dev`.)
- **`LlamaCppDecoder`'s `memory_clear`, `memory_seq_rm`,
  `memory_seq_cp`, `memory_seq_keep`, `memory_seq_add` and
  `memory_seq_div` take `&mut self` and keep the checkpoints in step.**
  They were `&self` raw FFI calls that shadowed the `Decoder` trait's
  methods at a call site, so `decoder.memory_clear()` — reachable from
  `Engine::vision_and_decoder` — skipped the checkpoint invalidation
  the trait methods do. On a sliding-window model that is silent
  corruption: llama.cpp's iSWA cache reports positions from the window
  cells alone, so after a raw clear a stale window checkpoint loaded
  over empty dense layers and the restore reported success. Every
  mutator now drops the partial checkpoints the change invalidates
  (`memory_clear` all of them; `memory_seq_add` and `memory_seq_div`
  from the lowest position a cell lands on, `p0 + delta` for a shift
  back and `p0 / d`, since cells move below `p0`), and the trait
  methods are thin wrappers. Only the raw context pointer still reaches
  the memory around them, and its docs say so.
- **`LlamaCppEngine::default_context_params` sets `swa_full = false`**,
  so `LlamaCppEngine::new(path, None, None, …)` sizes the
  sliding-window cache at the window, as `LlamaCppOptions` already did
  (Gemma 4 OOMs at 131k with the library's full-size one).
- **A sliding-window rewind accepts the window llama.cpp reads, not
  one cell more.** The truncate check asked for `pos_min <= pos -
  n_swa` (llama-server's margin), but the cell at `pos - n_swa` is
  masked for the token at `pos` and `find_slot` recycles it as soon as
  a neighbour needs a cell, so an idle slot's rewind to its own head
  went to a checkpoint load (≈ 800 MiB on Gemma 4) or, with the
  checkpoint evicted, failed. Now `pos_min <= pos - n_swa + 1`,
  `is_masked_swa`'s boundary; still fail-closed on any hole in the
  window.
- **BEHAVIOR CHANGE: sliding-window models get a window-sized SWA KV
  cache** (`LlamaCppOptions::swa_full`, unset = `false`; blallama
  `--swa-full` restores the old size). llama.cpp's library default
  sizes every sliding-window layer at the full context: Gemma 4 31B
  pays 880 KiB per token across its 50 window layers, ≈ 110 GiB of KV
  at 131k (an OOM, #125), and gpt-oss half its 72 KiB, ≈ 4.5 GiB. The
  window-sized cache is what llama.cpp's own server and CLI default
  to: at 131k / 4 slots, Gemma 4's KV drops to ≈ 13.5 GiB (10 GiB
  dense + 3.5 GiB window) and gpt-oss's to ≈ 4.5 GiB. The full-size
  cache bought nothing: llama.cpp recycles a window's masked cells
  whichever its size (below), so prefix reuse on these models now goes
  through window checkpoints either way. This is the one field where
  an unset `LlamaCppOptions` no longer means llama.cpp's library
  default. llama.h warns the small cache "can cause bad performance in
  some cases" with several sequences — unmeasured here; `--swa-full`
  is the way back if a workload shows it.
- **Hybrid models (Qwen3.6, Qwen3.8) checkpoint only their recurrent
  state.** A prefix-cache checkpoint was the whole sequence — the
  attention KV for the entire prefix again, ≈ 650 MiB at 30k tokens on
  Qwen3.6 and ≈ 2 GiB on Qwen3.8, up to 24 of them at four slots. It is
  now llama.cpp's partial state (`LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`,
  as its server's context checkpoints use): the recurrent layers only,
  ≈ 63 MiB (Qwen3.6) / ≈ 150 MiB (Qwen3.8) regardless of position,
  restored on top of the truncated attention KV. Because a partial
  checkpoint depends on the KV below it, every `Decoder::memory_*` call
  that changes that KV now drops the checkpoints above the change, and
  a checkpoint is only ever taken at the sequence's head.
  `LlamaCppDecoder::checkpointing` / `LlamaCppEngine::checkpointing`
  report the mode (`Checkpointing::{Off, Partial, Whole}`); dense models
  (cogito, Mistral 4) stay `Off` — truncation alone, unchanged.
- **BEHAVIOR CHANGE: prompt content that spells a special-token piece
  is no longer a 400 — the model reads it as text.** Every prepare
  path tokenized the whole render with special-token parsing on, so a
  reserved piece in *content* — Qwen's `<think>` / `<tool_call>`
  (`USER_DEFINED`, matched even with specials off), `<|im_start|>`,
  Mistral's `<s>` (which is also HTML strikethrough) — became the real
  control token, and `SessionError::InjectedSpecialToken` rejected the
  whole request. One Agora post quoting such a piece turned every
  reader's `/v1/messages` and `/count_tokens` into a 400, permanently,
  since the post stays in the transcript. Now the chat template
  replaces each reserved piece found in content with a per-call
  out-of-band marker (`LiteralNeutralizer`, via the new
  `RenderOptions::literals`), and `Session` tokenizes around it with
  specials off, spelling out byte by byte any piece the tokenizer
  still matches. Content surfaces covered: system text, user and
  assistant text (after joining blocks, so a piece split across two
  blocks is caught), thought bodies (inside our own `<think>`
  wrappers, which stay real), reasoning fields, tool results,
  tool-call input keys and values, tool descriptions and schemas, and
  a resumed open thought. The template's own framing is untouched.
  What changes for callers:
  - A prompt quoting a piece prepares; the model sees the spelled
    text. Clean prompts are byte-identical in render, tokens and cache
    hashes (tested), so existing caches stay warm.
  - `InjectedSpecialToken` is now a bug detector: the old guard still
    scans content the way the tokenizer reads it, and every piece it
    finds must have been neutralized at least as often; a shortfall
    fails the call loudly (and logs `literal_neutralization_bypassed`)
    as a drama_llama bug. blallama answers it with a 500 `api_error`
    instead of a 400, since the client's request is not at fault.
    `content_special_neutralized` is logged at debug on every call
    that neutralizes something.
  - Tool names and tool-use ids are *validated* rather than
    neutralized — the grammar and parser key on them: names must
    match Anthropic's `^[a-zA-Z0-9_-]{1,64}$`, ids
    `^[a-zA-Z0-9_-]+$`, else `ChatTemplateError::InvalidIdentifier`
    (a 400 on blallama). **An empty tool-use id is now an error** on
    every model with reserved pieces — every real one. A `ToolUse`
    built with `ToolUse::new` and no `.with_id(..)` rendered fine
    before and now fails; Anthropic rejects it too. Set an id on any
    call echoed back as history, and the same id on its result.
  - The dialect parser holds the names the *model* emits to the same
    pattern: a call named `foo.bar` (Harmony keeps whatever follows
    `functions.`), `get weather` or longer than 64 bytes degrades to
    text like any malformed call (or is withheld when cut) instead of
    seating a `ToolUse`. Seated, its name and `call_{n}_{name}` id
    would have failed every later request on the transcript once the
    client echoed it. The degraded text is then subject to containment
    like any other free text.
  - Templates no longer act on a piece that content spells: a Qwen
    template does not split assistant text at a literal `</think>`,
    nor treat user text starting with a literal `<tool_response>` as a
    tool response.
  - Emission containment (`EmittedSpecialToken`) now rejects only a
    piece the model emitted as the *real* token into free text; one it
    merely spelled (an agent quoting a post) passes, since the next
    ingest reads it as text.
  - **The model's output is parsed with emission provenance**, which
    closes the hole the change above opens. A transcript quoting
    `<tool_call>{…}</tool_call>` used to be a 400, so the model never
    read it; now it reads the spelled markup and can copy it into its
    own output, where a text-level parse would seat the copy as a real
    `ToolUse` (or a copied `<think>` as a `Thought`) — tool-call
    injection from content. `Session` knows the id behind every
    emitted piece, so every reserved piece the model *spelled* in
    ordinary tokens is swapped for an opaque marker before the dialect
    parser sees it and swapped back in the parsed blocks: only framing
    emitted as the real reserved token is structure. That covers
    `complete_blocks` / `complete` / `complete_response`,
    `complete_stream` (a piece spelled across several tokens is held
    back until it completes or cannot), the stop-sequence parse of all
    of them and `complete_text`'s. The lazy tool-call grammar arms
    only on a trigger whose reserved pieces are real tokens
    (`PiecePredictor::with_reserved`), so a spelled `<tool_call>` no
    longer drags the rest of the turn into call syntax; the same goes
    for `output_config`'s `</think>` trigger. A special whose text
    duplicates a reserved piece's (dropped from the reserved set, since
    the text tokenizes to the other id) is framing too when emitted,
    and the id-level bans (`ToolChoice::None`'s opener ban, #107's
    reasoning opener and closer bans) hold it with the reserved id.
    Containment reads the unrestored parse, so it is exact now rather
    than a count, and of a turn a stop sequence cut it reads only what
    the cut keeps. A real token the cut splits (a stop starting inside
    its piece) is not counted: what stays is the part of the piece
    before the stop, which is text, not the piece, and the caller sees
    it as text. A spelled piece re-renders byte-identically (it is a
    content literal on the next ingest), so the auto-tip is
    unaffected; a *real* reserved token left in content (only possible
    with `with_emit_specials_ban(false)`) re-renders spelled, so that
    turn stores no tip hash.

    **Protected** wherever the framing is a reserved special in the
    loaded vocabulary, which is per vocabulary, not per dialect.
    Measured 2026-10-01 (vocab-only loads): Qwen 3.6 / 3.8 call and
    reasoning markers (`<tool_call>`, `</tool_call>`, `<think>`,
    `</think>`); every gpt-oss Harmony header token (but Harmony's
    recipient is text — a header may start with a plain ` to=…` after
    a real `<|end|>` or at the turn's start — so what gates a gpt-oss
    call is the real `<|message|>` alone, `<|call|>` optional); Gemma 4's
    `<|tool_call>`, `<tool_call|>`, `<|"|>` and channel markers;
    Mistral 4's `[TOOL_CALLS]`, `[ARGS]`, `[THINK]`, `[/THINK]`; and
    cogito-32b's `<tool_call>` / `</tool_call>`. **Not protected**
    where framing is ordinary text: cogito-32b's `<think>` /
    `</think>` (plain text in its Qwen 2.5 vocabulary), trigger-less
    bare-JSON Llama 3.1, Hermes on a vocabulary without a `<tool_call>`
    special, and markup *inside* a real call that is plain text (Qwen
    XML's `<function=` / `<parameter=`, JSON's quotes), which reads by
    bytes — as the grammar reads it. There, grammar-constrained
    generation and `tool_choice` are the defense.
    Three limits remain. Provenance tells a spelling from a real
    token, not a copy from a call: a model that read spelled markup can
    still emit the *real* framing ids when it repeats it, and that is a
    call at every level of the parse. There, as for the unprotected
    vocabularies, `tool_choice`, the grammar and the model are the
    defense. The grammar is byte-level, so under an armed
    grammar the model *can* spell a piece the grammar requires, and the
    parse reads that piece as text: a forced call whose opener is
    spelled is a `GrammarViolation` (a retry is warm), and a real
    opener whose close is spelled either still parses, the close left
    as text, or degrades and is contained — never a call the model did
    not open with the real token. Models emit their own framing as the
    token, so this is rare. And
    `dialect::parse_text` / `StreamParser::push`, which see only text,
    stay provenance-free: a caller parsing emission itself gets the
    text reading.
  - Markers the model emits as real specials outside the dialect
    (Qwen-VL grounding's `<|box_start|>`, with
    `with_emit_specials_ban(false)`) re-ingest as text, not as ids.
  - `RenderOptions` gains a public field, `literals`; code building it
    with a struct literal and no `..Default::default()` must add it.
    `Session` sets it itself on every render, whatever
    `with_render_opts` was given.
  - moeflux: `tokenize_special` honors `add_special`, and
    `parse_special = false` now leaves added tokens marked `special`
    as text (an `encode_special_tokens` tokenizer clone), matching
    llama.cpp.
- **An unseeded cache resume reseeds the sampler rng.** A resumed
  call used to continue the snapshot's exact rng stream, and a prompt
  breakpoint's snapshot holds the *initial* rng of the call that made
  it, so a byte-identical retry replayed the byte-identical output:
  on 2026-09-12 six Agora agents replayed one bad tool-call sample
  every five minutes for fifteen sweeps with no chance of a different
  draw. `Session::build_initial_state` now reseeds the working rng
  from fresh entropy on the resume arm; `mu` and the n-gram stats
  still carry (they measure the corpus, not the draw). Bit-exact
  reproduction remains the fork arm's job (`with_seed`), and a
  directly restored `SamplerState` still round-trips its rng.
  `resume_at_breakpoint_is_deterministic` is replaced by
  `resume_at_breakpoint_resamples`.
- **The repetition penalty now sees history (#106) — three new
  `RepetitionOptions` seeding flags, all default ON.** For an
  all-structured workload (an agent whose every turn is tool calls
  and `output_config` JSON) the penalty was effectively disabled
  end-to-end: the prose fold excluded tool results and tool-call
  args, and the constrained-region accumulator started empty every
  call — near-identical sequential posts from similar agents were
  the symptom. Now `seed_tool_results` folds tool-result text (how
  agents read thread context), `seed_tool_args` folds the string
  values of prior tool-call arguments (how agents emit; keys,
  numbers and booleans stay excluded), and `seed_constrained_regions`
  clones the folded corpus into the constrained accumulator at
  generation start — after every breakpoint snapshot, step rebased —
  so grammar free regions feel prompt-history pressure from token
  one. Cold-prefill ≡ cache-resume is untouched (the seed re-derives
  from the prompt identically on both paths; oracle-pinned including
  the constrained fields). Escape hatches: `set_seed_tool_results` /
  `set_seed_tool_args` / `set_seed_constrained_regions`, all
  sidecar-settable per model.
- **Penalty defaults retuned for thread-corpus reach**: `window_size`
  256 → 2048, `decay` 0.95 → 0.99, `penalty_freq` 0.125 → 0.025. The
  decay/freq pair moves together, so the saturated additive cap
  (≈ 2.6) is unchanged — 5× the decayed-term reach at the same
  ceiling. Note existing sidecars serialize the full options table
  and therefore pin the old numbers; delete or edit
  `<model>.sampling.toml` to pick up the new defaults. The
  digit-echo pins (a `"3"` echoed from a tool result) stay green on
  all four model suites: short tool echoes seed nothing (blocks
  shorter than `ngram_max_size` produce no windows) and surgical
  mode gates single occurrences.
- **Tag-based reasoning dialects (Qwen, Gemma 4, Mistral 4…) keep a
  thought's own trailing whitespace.** A closed thought lost
  everything `trim_end` could take; now only the close marker's
  canonical leading whitespace comes off (the `\n` of `\n</think>` /
  `\n<channel|>`; Mistral 4's `[/THINK]` has none, so nothing does),
  so `Thought.\n\n</think>` parses to `Thought.\n` and re-renders
  exactly under a verbatim template — every baked one. Harmony already
  kept its analysis verbatim and is unchanged. An *empty* pre-opened
  thought now consumes the reasoning separator after it (it is the
  thinking-off scaffold to the re-render), and `CallSyntax::qwen_xml()`
  spells its close `\n</think>`, as the analyzer measures it. Visible
  to API clients: a `thinking` block's text can now end in the
  model's own blank lines (`"Thought.\n"` where it was `"Thought."`),
  including under an *unbaked* template that trims them away again on
  re-render.

### Fixed

- **blallama's `count_tokens` takes a body without `max_tokens`**, as
  Anthropic's does. misanthropic stopped sending it in 1.0.0-alpha.21,
  and blallama answered 400 (`missing field max_tokens`). `/v1/messages`
  still requires it.
- **A JSON string can no longer escape a C0 control ([#141]).** Raw
  controls were refused but any `\uXXXX` was legal, so cogito posted a
  comment of 3,000 `\u0010`. Both string grammars now refuse C0 escapes
  other than tab, LF, CR, `\b` and `\f`.
- **A cut turn keeps its repeated calls.** Dropping repeats from a
  `max_tokens` turn hid the identical-call loop blallama redraws on.
- **An effort-only `output_config` requests no grammar.** It failed
  every request with `UnsupportedFormat`; now its effort reaches the
  template and nothing else changes.
- **A render hash proves bytes, not segmentation ([#91]).** A hash hit
  restored a position measured in the cached tokenization and indexed
  the new one with it, skipping tokens of the new message.
- **The tip prediction keeps the last content token** when the grammar
  or the budget ended the turn, so the next call's walk reaches it.
- **The trained context is read on a vocab-only load**, so `/v1/models`
  caps a model's ceiling before it is loaded.
- **`/v1/models` lists each file once** (symlinks and hardlinks
  grouped), skips `mmproj-*.gguf` projectors, and advertises
  `image_input` only in an `mtmd` build.
- **Cogito thinks when asked to.** Any `thinking` but `disabled`
  already rendered cogito's `Enable deep thinking subroutine.`
  incantation, but the stock template never renders a thought: the
  analyzer measured no reasoning markers, so a `<think>` the model wrote
  stayed in the answer's text, and live cohort turns (`thinking:
  adaptive`) came back with no thought at all. The baked
  `cogito-cache-stable` template now pre-opens `<think>\n` on a
  thinking-on generation prompt and renders an assistant turn's thoughts
  as `<think>\n…\n</think>` in emission order. The analyzer measures
  the markers (`Field` reingest, `\n\n` separator), so the thought
  parses into a `Thought` block and the turn re-renders byte for byte
  for the prefix cache. Thinking off renders as stock.
- **A finished turn never reports `stop_reason: null`.** Generation
  that ended on its own read `null` when the turn was empty or ended
  inside a thought the model never closed; a client read that as "no
  action" and dropped the turn (live, Mistral Small 4). An empty turn
  is `end_turn`, as on Anthropic. A turn that ends inside an open
  thought is a `GrammarViolation` with the cache left warm — returned,
  the open thought would be rejected on the client's next request as
  an `UnrenderableOpenThought` — so blallama resamples it; the drained
  stream reports the same through `BlockStream::violation`. A thought
  the budget cuts is still `max_tokens`, open, for the client to
  continue.
- **EOG inside an open thought is steered to the closer.** Live on
  Mistral Small 4 the model ends its turn without closing its thought
  often, and every resample of the violation above did the same: all
  three draws failed and the client got an error. The sampler now
  writes the dialect's closer in EOG's slot while a thought is open
  (`ThoughtSpecials`, as for a nested opener), a pre-opened render
  included, so the model answers or ends a closed, thought-only turn
  (`end_turn`). Where the closer can't take the slot (not one token,
  banned, or refused by a constraint), EOG stands and the violation
  is the backstop. A budget cut is unchanged.
- **An image-end boundary checkpoints and rewinds again on M-RoPE
  models (Qwen3.6 with an image).** The checkpoint rules took a
  sequence's head as `pos_max + 1`, but every cell of an M-RoPE image
  carries the chunk's *start* position while the counter advances by
  `n_pos`, so a checkpoint at image-end was skipped as off-head and the
  restore there failed with `NoCheckpoint` (caught by
  `mtmd::tests::mrope_kv_semantics_probe`). The vision path now reports
  each media chunk's end to the decoder, the head comes from it while
  the chunk is the tip, and each checkpoint records the `pos_max` it
  was taken over so a restore checks it landed there. The report is
  dropped by anything that touches the cells below its end (a
  truncate, a checkpoint load, `memory_*`), so text decoded later at
  the image's old start never inherits its end. A pure-attention
  M-RoPE model now also rewinds to image-end by truncation. Verified
  on Qwen3.6: both a text-boundary and an image-end rewind re-extend
  to logits bit-identical to the cold prefill.
- **A turn the model wrote in a split its tokenizer would not produce
  keeps its tip ("trust the emission").** A model does not always emit
  the tokenizer's own segmentation of its text — a grammar forces a
  piece, sampling picks a rarer one — and the next call re-tokenized
  the byte-identical re-render, disagreed with the cached ids partway
  through the turn, and re-prefilled from the last anchor before it.
  The hash path saw the same bytes and refused them, rightly (#91:
  reuse positions would be measured in the wrong tokenization). Live
  on the cohort run (2026-10-01), a Qwen3.8 agent lost 19,340 tokens
  to one such turn (`hash_drift`, then `tip_diverged` with equal text
  on both sides), and eight `hash_drift` events cost 297–21,143 tokens
  each. Each call now reads its prompt in a cached slot's own ids as
  far as the two read alike, and in the tokenizer's from there on: it
  walks the slot's ids against its plain tokenization, through
  stretches where they agree (an id comparison each) and across those
  where they spell the same bytes in different tokens, up to the last
  token boundary both share before the first byte the client actually
  changed. A respelled stretch may hold only ordinary tokens — never a
  special, a duplicate of one, an empty piece or an image — so a span
  the model *spelled* is never read where the tokenizer would see a
  special. The tail is the plain tokenization's own ids, so there is
  no second tokenization, no BOS mid-stream and no split that neither
  list already has. It chains across turns (older turns in the
  model's split stay matched), survives an edit or a truncated or
  resent history (everything before the edit stays matched, where
  hash-matched records lost back to the earliest respelled span), and
  works across content literals and images, which compare as ids
  rather than as render bytes under a per-call sentinel. Breakpoint
  partials are checked against the plain tokenization as before, then
  placed in the spliced list; one ending inside a respelled stretch
  keeps the anchor the slot recorded for its hash. `count_tokens` reads
  the same splice, so the three usage counters still sum to it. The
  splice comes from the slot reaching furthest into the prompt; when
  that slot already reads it in the tokenizer's split, nothing is
  spliced. Safety is tightened, not just kept: the hash path now also
  requires the ids before its anchor to be equal. A turn the model
  wrote `a|bc` re-tokenizes `ab|c` — the same bytes in the same number
  of entries — and a hash hit there used to restore KV holding the
  model's ids while the slot recorded the tokenizer's (reachable with
  adoption off, and with it on wherever the walk stops short, e.g. at a
  duplicate special). Reuse now needs the token-id walk or a hash hit
  over equal ids, so the KV is the ids the slot records. The walk
  itself is the proof that the spliced ids read as the render;
  `DRAMA_LLAMA_CACHE_TRIPWIRE=1` re-checks it (the same bytes, the same
  specials and images in the same places) and panics on a walk bug.
  On a 96,628-token Qwen3.8 prompt with 833 respelled stretches the
  walk took 0.21–0.25 ms per slot against ~190 ms for the plain
  tokenization each call already pays; a contrived worst case, every
  multi-character token respelled, takes ~8 ms per slot. On by
  default; see `PrefixCacheConfig::adopt_emitted_tokens` below for the
  one behaviour change.
- **A rejected turn no longer logs cache diagnostics for itself.** A
  turn's `emission_not_byte_stable` and `tip_not_recorded` events are
  now logged only once the turn stands: one that #101 containment, the
  schema backstop or a grammar check rejects reaches neither the
  client nor the next request, and its re-render cost was noise that
  read as a cache bug in the operator log (cohort item 13).
- **Checkpoints have a host-RAM budget** (`CheckpointBudget`, default
  8 GiB in all and 4 GiB per slot; `LlamaCppOptions::{checkpoint_mib,
  checkpoint_slot_mib}`, blallama `--checkpoint-mib` /
  `--checkpoint-slot-mib`). The count cap alone let Gemma 4 hold 24
  window checkpoints of ≈ 800 MiB — ≈ 19 GiB of unified memory next to
  the weights and KV. Past either bound the least recently used
  checkpoint is evicted (a per-slot overrun only evicts that slot's),
  but each slot's first anchor (its lowest checkpoint — the system
  prompt and tools) goes last: restores refresh only the anchor they
  land on, so pure LRU evicted that one first. One larger than a bound
  on its own is not stored; each logs `snapshot_evicted` with the
  `bound` it hit, or `snapshot_over_budget`. The budget is per decoder,
  not per process, and peak host RAM is the budget plus one
  checkpoint, which is serialized before the store evicts to fit it.
- **An anchor restored by truncation is checkpointed on the way.** A
  window-sized cache recycles the window below an anchor soon after,
  so an anchor whose checkpoint was evicted (or never taken) restored
  once and then failed with `restore_failed`. `restore_to`'s truncate
  path now takes the missing partial checkpoint there, where the head
  is; a stored one is only refreshed, never copied again.
- **Checkpoint restores and invalidations are logged** (DEBUG, under
  `drama_llama::snapshot_store`): `checkpoint_restore` with
  `restore_via = truncate | checkpoint`, `bytes` and `ms`;
  `checkpoint_taken` with `bytes` and `ms`; `checkpoint_invalidated`
  with a `count` (a `memory_clear` included). `checkpoint_skipped` now says whether the position
  was off the head (`cause = off_head`) or llama.cpp failed to
  serialize it (`save_failed`), where it blamed the head for both.
- **Sliding-window models (gpt-oss, Gemma 4) restore their prefix
  cache from window checkpoints — and never from half a window.** A
  KV truncate rewinds a sliding-window layer only while the window
  below the anchor is still there, and llama.cpp recycles masked
  window cells of *any* sequence whenever it places a batch, even with
  a full-size SWA cache. These models were treated as dense, so
  nothing was checkpointed: live on gpt-oss (2026-10-01, twice) an idle
  slot's window under its 10,088-token anchor was recycled by its
  neighbours, `restore_failed` found no checkpoint, and the request
  re-prefilled all 14k tokens. Worse, the restore checked only that the
  sequence's head landed on the anchor, so a truncate that left *part*
  of the window would have been reported lossless and generated over a
  window with a hole in it (#91's class: KV that is not what the model
  should see). Now `checkpoint_pos` stores the window cells (gpt-oss
  ≈ 4.5 MiB, Gemma 4 ≈ 800 MiB per anchor), `restore_to` takes the
  truncate only when the head *and* the whole window below it survive,
  and loads the checkpoint otherwise; with neither, the anchor reports
  `NoCheckpoint` and the session's restore ladder falls to the next
  anchor below. The rules live in `llama_cpp::checkpoint`, FFI-free
  and pinned against a simulated cache
  (`a_recycled_window_restores_from_its_checkpoint`,
  `a_truncate_that_leaves_half_a_window_is_not_a_rewind`); the
  ladder's live shape
  (`a_hash_hit_without_a_checkpoint_falls_to_the_anchor_below`) and
  the invariant that every anchor a call leaves was checkpointed
  (`every_anchor_a_call_leaves_is_checkpointed`) through `Session`;
  real-weights checks in `tests/swa_checkpoint.rs` (`#[ignore]`d).
- **The baked Qwen3.6 template keeps the prose before a second
  `</think>`.** It recovered an inlined thought by splitting content on
  `</think>` and kept only what followed the *last* one, as stock does,
  so an answer containing a stray `</think>` re-rendered without the
  prose before it (live on Agora 2026-10-01, in the session's own
  round-trip check: a false `emission_not_byte_stable`; the rendered
  content of a real request has such a close neutralized to text). It
  now splits on the first, in the aged-turn branch too. Qwen3.8 reads
  `reasoning_content` and was never affected. Pinned in
  `qwen_cache_stable_round_trips` and
  `qwen_cache_stable_aged_turn_renders_as_stock`.
- **A gpt-oss final re-renders under the header the model wrote.**
  gpt-oss writes a structured answer under
  `<|channel|>final <|constrain|>json<|message|>`, unforced, but the
  baked template rendered every final channel plain, so each JSON final
  parted from the KV at the constraint and the next request
  re-prefilled the turn (470..1111 tokens, four times live on Agora
  2026-10-01). The header is framing, so the final's `Text` cannot say
  which spelling it had, and its content does not either (`[1, 2, 3]`
  may be prose; a structured answer whose schema root is a string is
  not `{…}`). The parser now records the content type the header
  declared in the signature of the analysis block before the final
  (`drama_llama:tail;constrain=json`, see the whitespace entry below),
  and the baked template renders exactly that — an empty final too,
  from the thought alone. The `output_config` grammar keeps the
  constraint optional after another block — both spellings are the
  model's, and both round-trip — and **forbids it on a final that opens
  the turn**, which has nowhere to record it and would re-render plain
  (the unified grammar sees the analysis; the deferred one now reads its
  trigger, `<|end|><|start|>assistant<|channel|>final` before the bare
  `<|channel|>final`). Pinned as irreducible: a constrained final with
  no thought right before it — none, or a commentary preamble between —
  re-renders plain; free generation can still write one (the
  `output_config` grammars refuse that preamble, see the preamble entry
  below). Pinned in
  `gptoss_cache_stable_round_trips_json_final` and
  `gptoss_cache_stable_keeps_a_preamble_apart_from_its_final`.
  **Deployments that
  copied the baked template to a `<model>.template.jinja` sidecar must
  delete it** — it wins over the bake; a sidecar that is a known old
  bake now logs a `WARN` at load (see *Added*).
- **Back-to-back thoughts stay two thoughts.** A model that closes a
  thought and opens another (`…[/THINK][THINK]…`, live on Mistral Small
  4, 2026-10-01, ~1.6k tokens lost) was parsed into two
  `Block::Thought`s, which the session then merged into one, and the
  template rendered one block — the markers between them were gone, the
  turn no longer matched the KV, and its tip was lost. Adjacent thoughts
  no longer merge in a response (Anthropic returns consecutive thinking
  blocks too; clients now see each one), and every assistant message
  reaches the template with **`chunks`**, its blocks in emission order
  (`{type: "text", text}`, `{type: "thinking", thinking}`, one
  `{type: "tool_calls"}` where the first call sat) — stock Mistral's
  own content-chunk shape. The baked Mistral 4, gpt-oss and Gemma 4
  templates render from it, so each `[THINK]` block, each analysis
  block and each Gemma thought channel renders as its own, with prose
  where the model wrote it; without `chunks` they render from the
  merged fields as before. Pinned in
  `mistral4_cache_stable_round_trips_back_to_back_thoughts`,
  `gptoss_cache_stable_round_trips_each_analysis_block` and the fleet
  sweep.
- **A gpt-oss preamble and final are two text blocks, and the answer is
  the final.** A turn with a commentary preamble and then a final, no
  call between (`…<|channel|>commentary<|message|>Hi.<|end|>
  <|start|>assistant<|channel|>final<|message|>{"a":1}`), parsed to one
  text, `Hi.{"a":1}`: not the visible answer, a value no `output_config`
  schema could accept (`SchemaViolation` on a well-formed answer), and
  a turn the bake re-rendered as one final, losing its tip. Each Harmony
  channel's body is now its own text block, the batch path no longer
  merges adjacent text (a parse merges its own prose, so side by side
  they are channels), `chunks` carries each text block as its own
  chunk, and the gpt-oss bake renders every text but the last of a turn
  without calls as a preamble. **A streamed turn cannot mark the
  boundary**: `BlockStream`'s text yields are deltas, so the preamble
  and the final stream as adjacent text. Both paths therefore treat
  them as one run where the client cannot tell them apart: a request
  stop sequence matches across the two on the batch path as on the
  stream (the batch keeps the blocks apart and cuts the one the match
  starts in), so both stop at the same text and report the same
  `stop_sequence`. An assistant message a client sends with several
  text blocks now renders on gpt-oss as preambles then a final, where
  it rendered one final.
- **Under `output_config` a gpt-oss answer is one text block.**
  Anthropic (and misanthropic's `json_items` and kin) promise a
  json_schema answer as exactly one text block holding the JSON. The
  output_config grammars now refuse whatever gpt-oss would write
  between its analysis and its final — a commentary preamble, a call, a
  second analysis — as the unified grammar always did: the deferred
  (phase-split) grammar's trigger is now the end of the turn's first
  block, `<|end|><|start|>assistant`, and its header rule admits only
  the final there. A preamble that *opens* the turn runs before any
  trigger and cannot be refused up front; such a turn (two text blocks)
  is a `SchemaViolation` on both paths, as is any turn whose answer is
  more than one text block — the batch judges its parse, a drained
  `BlockStream` the parse its yields cannot split — never a 200 with the
  preamble run into the JSON. Pinned in
  `harmony_output_config_refuses_a_detour` and
  `gptoss_cache_stable_keeps_a_preamble_apart_from_its_final`.
- **A Qwen turn that reasons again after its prose re-renders.**
  `…</think>\n\nChecking.<think>\nMore.\n</think>…` parsed to two
  thoughts, but the merged fields could not place the second: 3.6
  inlined it without its markers' newlines, 3.8 joined both thoughts in
  `reasoning_content`, and the turn lost its tip. The Qwen bakes render
  such a turn from `chunks`, each later thought where the model wrote
  it. Aged out with `preserve_thinking` off, such a turn drops every
  thought, the later ones too, and a client's text after the calls
  renders before them with the rest, as stock renders merged content.
  Pinned in `qwen_cache_stable_round_trips_a_second_thought`,
  `qwen_cache_stable_chunks_age_and_keep_late_text` and the fleet
  sweep. Pinned as unfixed, with the reason: whitespace after
  Qwen's last call (`…</tool_call>\n`) has no block to ride and is
  dropped, but no constrained call turn can write it — the tool grammar
  reads a newline after a call as the separator to the next
  (`qwen_whitespace_after_the_last_call_is_unreachable`).
- **An empty thought re-renders.** Mistral 4's `[THINK][/THINK]` and
  an empty gpt-oss analysis block were dropped by the parser, so the
  re-render lacked their markers and the turn lost its tip. A closed
  thought is now kept even when empty (Anthropic returns empty thinking
  blocks too), except on Gemma 4, whose empty channel is its own
  thinking-off scaffold, and on dialects that re-ingest thoughts
  inline, where it would put a bare `<think></think>` into content.
- **Whitespace around a Mistral 4 or Gemma 4 thought re-renders as
  written.** A fleet-wide sweep (`fleet_bakes_round_trip_every_admitted_shape`:
  every baked template through the session's parse and re-render, over
  the shapes its grammars admit — thought or none, two back to back,
  prose, a JSON answer, a call, prose then a call, parallel calls,
  answers with leading or trailing whitespace) found two more shapes
  that lost their tip. The parser swallowed one `\n` after a thought
  the model opened itself, and one at the start of its body, whatever
  the template renders there: right for Qwen's `<think>\n`, wrong for
  Mistral 4, whose `[THINK]…[/THINK]` has no layout newlines, so
  `[/THINK]\n{…}` (an `output_config` answer the grammar admits) came
  back without the `\n`. It now takes off exactly the whitespace the
  dialect's markers are spelled with and leaves the rest to the answer,
  as the pre-opened path already did. And the baked Gemma 4 template
  `| trim`med a model turn's answer, as stock does; it now renders it
  verbatim. `CallSyntax::qwen_xml()`'s reasoning start is now
  `"<think>\n"`, as the analyzer measures it. Neither shape has been
  seen live; Gemma 4 served no turns in the 2026-10-01 run.
- **No response carries a whitespace-only text block.** Anthropic
  never returns one and rejects one on ingest ("text content blocks
  must contain non-whitespace text"), so a transcript replayed from
  blallama to Anthropic failed on it — and the parser produced one for
  the whitespace between a thought and a call (`[/THINK]\n[TOOL_CALLS]`
  on Mistral 4, `<channel|>\n<|tool_call>` on Gemma 4,
  `</think>\n\n<tool_call>` on Qwen). That whitespace is in the KV and
  the next render needs it, so it now rides in the thought's
  `signature` (`drama_llama:tail;gap=%0A`, alongside the
  `OPEN_THOUGHT_SIGNATURE` overload) and the renderer puts it back where
  it sat: every template sees the turn exactly as before. A closed
  thought with nothing to record keeps the empty signature. Whitespace
  with no thought before it (the turn's first bytes before a call,
  after the last call) has nothing to ride and is dropped — the turn
  re-renders without it, losing its tip rather than the client's
  ingest; pinned in `blank_text_is_never_returned`. A stream withholds
  a thought until what follows it starts, and never yields a run of
  text that is only whitespace (the batch path drops what a stop
  sequence cut leaves the same way). The fleet sweep now also renders
  every shape followed by the next request (its tool results, or a
  user turn), aged with `preserve_thinking` off (only a thought — or
  Qwen's thinking-off scaffold, as stock — may be lost), streamed a
  char at a time against the batch parse, and every thought the budget
  can cut continued as an open thought.
- **blallama exits on every fatal fault, reliably, with a code saying
  which — and no longer serves on a failed backend.** Its policy is
  unchanged: when the process can no longer be trusted it exits rather
  than recover in-process, because llama.cpp does not promise its
  destructors clean up after a failure, and a supervisor restarts it.
  It was not reliable or complete:
  - Only a panic in a request's blocking session task exited. A panic
    in a handler, a hyper connection task, the probe writer or the
    catalog warm-up was caught by tokio and the server kept running —
    having unwound, and dropped, whatever that task owned (a `Session`
    between checkout and its blocking call included). A panic hook now
    covers every thread: it declares the process fatal and parks the
    panicking thread instead of unwinding it (a tokio worker hands its
    scheduler on first), so nothing it owns is dropped. Model loads and
    catalog reads are the exception the chat-template analyzer needs
    (it catches the panics minijinja raises on some templates); a panic
    that escapes one still exits, through its `JoinError`.
  - A failed prefill (`SessionError::Decode`) was "reusable", so a
    backend llama.cpp leaves broken kept serving: on 2026-10-01 one
    Metal out-of-memory command buffer left its context "in error state
    … recreate the backend", and every later request failed until a
    manual restart. Every decode error, and every error after which the
    session is not reusable (which used to be dropped and reloaded),
    now exits; the session is leaked, not dropped. The predictor's
    decode-failure panics (#92) count as backend failures too.
  - The exit was `std::process::exit(1)`, which runs C++ static
    destructors: ggml-metal's asserts every buffer was freed
    (`ggml_metal_rsets_free`), so with a model loaded the exit became a
    `SIGABRT`. It is now `_exit`, after the cause is logged as one
    `ERROR` line (`event: "fatal"`, `kind`, `exit_code`, `cause`) and
    flushed, and after a 500 ms grace in which requests in flight —
    and any that arrive — are answered 500 `api_error`, which SDKs
    retry. The exit runs on its own timer thread, so nothing waits on
    those answers.
  - Exit codes: **70** after a panic, **75** after a backend failure.
    The policy is documented in `bin/blallama` ("Run it under a
    supervisor") and the README, with a minimal restart loop in
    `scripts/blallama-supervise.sh` (keeps the arguments, backs off on a
    crash loop, logs each restart, stops on a clean exit). Tests run the
    test binary as a child process: a panic in the session task, a
    panic in a handler, a caught and an escaping panic in a model read,
    and a fatal decode error each exit with their code, log one fatal
    line and answer 500.

- **blallama exits on a model load that fails while allocating, and
  serves on after one that fails before.** A failed load used to be
  answered and served past, whatever the cause — including an
  out-of-memory load that died with the backend half-allocated, which
  is exactly the state the exit-and-restart policy exists not to serve
  from. llama.cpp answers every failed load with the same null, so
  `LlamaCppEngine` now finds the benign failures first: it opens the
  file (`NewError::Unreadable` — missing, a directory, unreadable),
  then reads it vocab-only, which allocates no backend memory
  (`NewError::Metadata` — not a GGUF, an unsupported architecture, a
  malformed vocabulary), and only then loads it in full. A full load
  that fails after both passed (`NewError::Model`), a failed context
  creation (`NewError::Context`: the KV cache and compute buffers), and
  a failed mmproj load are resource failures. The class is
  `NewError::is_resource`, `MoefluxEngineError::is_resource`
  (`mf_init_model`'s bare null is a resource failure; the tokenizer and
  config parse, a path C cannot take and weights for another variant
  are not) and `SessionError::is_resource`; where the backend does not
  say why, it answers `true`. blallama declares a resource failure
  fatal and exits **75** with its one `ERROR` line; a benign one is
  answered as before (and now logged, `event: "load_failed"`). The
  vocab-only pass adds tens of milliseconds to each load, and the
  catalog's metadata peek now fails as `Metadata`, not `Model`. Tests:
  real llama.cpp classification of a missing path, a directory and a
  garbage file; a child process whose load fails with a stand-in
  context-creation failure exits 75 with one fatal line and answers
  500; the real binary, serving a directory holding a garbage `.gguf`,
  answers the load with an error, a nonexistent id with a 404, and is
  still up and serving past the fatal grace period.

- **Non-ASCII model output can no longer panic a dialect parser.** The
  gpt-oss Harmony parser stepped one *byte* past a block boundary
  (`rest[1..]`) before looking for the next marker, so prose outside a
  channel block that began with a multibyte char (`"Über alles"`, an
  `é` after `<|end|>` or `<|start|>assistant`) sliced mid-char and
  panicked — in blallama, the blocking session task, so the process
  exited. Every scan that must make progress now steps one *char*. A
  fuzz over each dialect's markers, their prefixes and multibyte chars
  found two more: Gemma 4 read a turn-exit marker (`<|tool_response>`)
  ahead of a channel open as prose once the open arrived, re-cutting
  the streamed text under what it had already yielded (now consumed as
  envelope, as it is everywhere else); and the stream parser sliced its
  trailing text at the byte count it had yielded, which a re-cut of
  that block (also reachable when a call opens inside an open thought
  that a later close turns back into thought) puts mid-char — it now
  resumes at the next char boundary and logs a warning. Tests put é,
  🦀 and CJK at every char boundary of each dialect's reference turn,
  batch under every leniency and streamed in uneven pieces
  (`dialect::utf8_tests`), and through stop sequences and emission
  provenance.

- **A Qwen XML parameter whose JSON does not read is a malformed call,
  not a string.** A JSON-spelled parameter (a number, an object, a
  union, `{}`) whose text serde_json refused came back as the raw text,
  a *string* — silently retyped. The grammar admits two such values: a
  tree nested past serde's 127 levels through a recursive `$ref` (a
  back-reference costs no depth), and a number past `f64` (a 310-digit
  integer, `1e999`). Every other dialect refused those calls; Qwen
  returned `{"x": "999…"}`, which a union admitting strings even passed
  the schema check as. Now the call is malformed like theirs: it
  degrades to text, which the session rejects as a real `<tool_call>`
  in free text (or, under a forced call, as a grammar violation) — a
  resample. Kept lenient, and pinned: raw string parameters (any text,
  JSON-looking or not), a nullable string's bare `null`, an unlisted
  member of a string set, pythonisms (`True`, `None`, single quotes)
  and unclosed brackets in a JSON value. One behavior change for a
  model writing unconstrained: a JSON-spelled union that admits
  strings (`["number", "string"]`) no longer takes an unquoted word as
  a string; the grammar always quotes it.

- **Deeply nested output can no longer abort the process.** The Gemma
  4 dict-value reader and the readers of a call cut short (every
  dialect's streaming and clipped parse) recursed once per bracket, so
  an untyped parameter nested ~2,500 deep (Gemma 4, inside what its
  grammar admitted) or ~3,800 (the rest) overflowed a 2 MiB stack —
  tokio's, which blallama runs `Session` on — and aborted the server
  with every request on it. Each reader now refuses nesting past 127
  levels as malformed, exactly as serde_json (which the JSON dialects
  parse with) refuses it. And the grammars no longer admit it: an
  untyped value (`{}`, `{"type": "object"}` without properties, …)
  nests at most 32 levels of objects and arrays, in the JSON and dict
  encodings alike, and the schema around it at most 64
  (`SchemaLimits::max_depth`, see Added), so with the call envelope a
  constrained value stays under 127 and every dialect reads back what
  its grammar admits (`depth_budget_fits_the_parsers`). Recursion
  through `$ref`s is the one way past it: a tree schema still nests as
  deep as the model takes it, and a value past 127 levels is refused
  by the parse, as before.

- **A generation that floods its call trigger parses in linear time.**
  The dialect parser rescanned the rest of the text for each of its
  landmarks (reasoning open, call trigger, Gemma's channel markers and
  turn exit) from every block boundary, so 384 KB of Gemma 4's
  `<|tool_call>` — 32k tokens, each a malformed call — took 10 s to
  parse once (Qwen's flood 0.9 s), and the streaming parser re-parses
  on every token. A landmark's next occurrence is now remembered for
  the parse: the same floods take 160 ms.

- **`required` and `type` arrays count each name once.** A `required`
  listing an undeclared name twice was a key the grammar made the
  model write twice, and a `type` naming one type twice
  (`["string", "string"]`) compiled to any value, which the schema
  check then refused. The compilers, the width measure and the check
  now read each name once; the check gathers a schema's names once per
  check, where it rebuilt the whole list for every object it judged: a
  `required` naming one property 100,000 times (inside every limit)
  took 8.4 s to check over 128 KB of output, past the step budget no
  request inside the limits may reach.

- **A nullable Qwen XML parameter takes a bare `null`; a non-nullable
  one never comes back `null`.** A JSON-spelled parameter whose type
  is nullable through a `type` array (`["integer", "null"]`, schemars'
  `Option<T>` for a non-string `T`) compiled to its base type alone, so
  the `null` its schema allows was unwritable — a required `Option<i64>`
  could not be null at all. It now takes `T | null`, and `null` reads
  back JSON `null`. The rest, now pinned together
  (`qwen_xml_null_only_where_the_schema_allows_it`): a non-nullable
  integer, number, boolean, object, array or enum parameter has no
  `null` in its grammar; a plain string parameter is raw, so the text
  `null` is admitted and reads back as the *string* `"null"`; a
  nullable string and an `anyOf` with a `null` variant (Agora's
  `Option<DetailLevel>`) take a bare `null` as `null`, and so does a
  parameter that `$ref`s a nullable def (through an alias chain too),
  as its inline form does. Nested fields and the JSON dialects keep
  collapsing a nullable type to its base.

- **A matcher state over its caps refuses alternatives, not the
  close.** Past `MAX_STACKS` (4096) stacks a state kept a prefix of its
  sorted stacks, and the stack that closes a structure sorts after its
  alternatives: after the `{` of an object of more than 4096 optional
  members, the `}` was cut, so the object could not be empty, and a
  required member after them could not follow `{` alone. A capped state
  now keeps one stack per distinct grammar position first (the
  shallowest, shallowest first), then the rest in sorted order, so
  every way to go on survives the cut and what is dropped is extra
  derivations of a kept position or, past 4096 positions, the deepest
  ones: members, not the close. Below the caps nothing changes.
  (Keeping the shallowest stacks outright would not do: an ambiguous
  recursive schema doubles its stacks at every level, and keeping each
  level's closes first leaves no room to go deeper.)

- **Qwen's tagged-value classifier reads each `$def` once per tool,
  and the streaming parser classifies each tool once per
  generation.** Two thousand parameters naming one def whose `enum`
  holds a 100 KB string spelled that member two thousand times (200
  MB of `to_string`) and kept a copy per parameter (400 MB), in the
  emitter and again in the parser; and the streaming parser, which
  re-parses the whole generation on every token, re-classified the
  tool on every one. Now:
  - A def's class is computed once per tool (unless cut short by
    depth or budget) and each member spelled once; a parameter that
    `$ref`s it takes its members as shared ids, and parameters with
    the same member set share one `Choice` (`Arc`), which the grammar
    writes as one rule for all of them.
  - The per-tool budget is charged by member bytes spelled, plus a
    step per schema visited and per member taken from a def, at 2^20
    (about a mebibyte of member text per tool) where it was 2^16
    steps of any size.
  - `StreamParser` keeps each tool's spellings across re-parses, and a
    lookup clones a reference count, not the member list.
  Grammars and parses are identical to before on two random corpora
  (3,000 schemas, and 2,000 with `$defs`, `$ref`s and nullable refs),
  in every dialect.

- **The grammar matcher's memory is bounded however a hostile schema
  nests its output.** Measured on the hostile recheck's ambiguous
  recursive schema (`N1 = N2 = {"c": N1 | N2}`) and on `[` nested
  thousands deep, over a 75,000-token vocabulary:
  - A matcher state keeps at most 2^16 frames (`MAX_STATE_FRAMES`,
    was 2^18), and a lone stack deeper than that is no longer exempt:
    output nested past it (~32,000 levels of `[`) is refused through
    the usual violation path instead of copying an ever-deeper stack
    on every byte. 2^16 is `MAX_STACKS` stacks 16 frames deep; the
    widest real states (a 2,000-member `enum` three objects deep in a
    tool call) are under 9 frames a stack, so the cap still binds only
    where `MAX_STACKS` already does. (2^15 would over-restrict that
    `enum` in the Hermes dialect.)
  - The DFA cache's weight cap drops from 256 MiB to 32 MiB, and holds
    *within* a sampling step: `intern` refuses a state that would take
    the cache past twice the cap, or one heavier than a quarter of it,
    with `UNCACHED_STATE`, and the grammar filter and the repetition
    region guard walk the matcher for that input instead. Previously
    only the next step's base intern could clear it.
  - Spilled matcher stacks get power-of-two capacities: exact-length
    copies, one block size larger per nesting level, left freed blocks
    too small to reuse, and the process held over a gigabyte resident
    above ~50 MB live.
  Filtering every level of the ambiguous schema to 1,000 deep went
  from ~1.45 GB resident to ~580 MB at the same speed, and to 6,000
  deep holds ~800 MB resident (under 300 MB physical footprint) at
  10–25 ms a step; at 2^18 it held 1.3 GB filtering only every 50th
  level to 400.

- **A grammar with too many rules is `SchemaError::TooComplex`, like
  one with too many bytes.** The compiler stopped at 8 MiB of source,
  but `Grammar::parse`'s 2^18-rule limit was only met later, as a
  `GrammarError::TooLarge` that read like our bug: a 400,000-property
  object passes it long before 8 MiB. The compiler now counts the rules
  it writes (`rule_count`, which mirrors the parser: one per
  definition, string literal, group and `*`/`+`/`?`) and stops at the
  limit; the dialect emitters and strict `tool_choice` count across
  tools too, since many tools under the limit can pass it together.
  `SchemaError::TooComplex` gains `what` (`"bytes"` or `"rules"`).

- **Qwen's tagged dialect writes each `$def` once per tool, and
  classifies a tool's parameters once.** Each JSON-valued parameter
  had its own compiler, so a tool whose P parameters all `$ref` the
  head of a D-long chain of defs wrote all D defs P times: 800 × 800
  was 143 MB of grammar and 5 million rules. One compiler per tool
  (def rules `tool_<i>__def…`) writes each once. Alongside:
  - Every raw (string) parameter shares one until-rule, `val_raw`; a
    ~1.5 KB copy per parameter made a string-heavy tool's grammar
    mostly duplicates (6000 string parameters passed the 8 MiB limit).
  - The raw-vs-JSON spelling classifier resolves `$ref`s straight from
    the `$defs` table instead of building `Defs` (a strongly-connected-
    components pass over every def) per parameter, and the parser
    classifies a tool once per parse instead of once per parameter it
    reads. A tool's parameters share one step budget (2^16), in
    declaration order, for the emitter and parser alike, so a schema
    that makes every parameter walk a thousand-member union costs at
    most that: a parameter past it is spelled as JSON (always correct,
    merely not the raw spelling) by both sides. A finite set of more
    than 4096 members (the matcher's stack cap) is JSON too, so every
    set inside the schema width limit is written raw, as the template
    writes it; at the first cap, 1024, a string `enum` of 1025–2048
    members passed the measure and was quoted where the template writes
    it bare.

- **An all-optional object compiles to a linear grammar.** Both the
  JSON and the dict (Gemma 4) encodings wrote, for each property, the
  whole tail of later properties: quadratic, so 4000 optional
  properties were over 300 MB of grammar (now past the 8 MiB limit,
  but a legitimately wide object should not be anywhere near it). They
  now share suffix rules (`pick_k ::= member_k rest_{k+1} | pick_{k+1}`,
  `rest_k ::= ( sep pick_k )?`): each member written once, the
  separator matched once before the choice of the next member, and the
  accepted language unchanged — same fixed order, same separators,
  checked against the old encoding on random member sets.

- **An ambiguous recursive schema can no longer stall or exhaust the
  grammar matcher.** Two interchangeable recursive defs (`N1 = N2 =
  {"c": N1 | N2}`) double the matcher's live stacks at every nesting
  level — they differ in which rule each frame is in, so no dedup
  merges them: 18 levels held 393,216 stacks at over a second a byte,
  and a deeper document never finished. A matcher state now keeps at
  most 4096 stacks and 2^18 frames across them (a deterministic
  prefix of its sorted stacks, and always at least one). Dropping
  stacks only drops continuations, so the cap can over-restrict —
  surfacing as the existing grammar-violation path — but never admit
  a byte the full state would refuse. Real grammars peak under 40
  stacks. Alongside:
  - The epsilon walk is bounded by frames copied (2^22) and steps
    (2^20) instead of 4096 steps per starting stack, which cut wide
    alternations short: a 4000-property all-optional object refused
    even `{}`, and an `enum` past ~2000 members lost members
    unpredictably (now it keeps 4096).
  - The session-lifetime DFA cache also restarts cold past ~256 MiB of
    interned states, not only past 65,536 of them — capped states can
    still be megabytes each.

- **A schema that has no grammar is a 400, and compiling one is
  bounded.** A client's schema (any tool's `input_schema` in any
  dialect, a strict `tool_choice`, an `output_config` `json_schema`)
  could make the compiler build an arbitrarily large grammar before
  anything looked at its size, and `{"enum": []}` compiled to an empty
  rule body that failed as a GBNF syntax error (`compiled grammar is
  invalid: …`), reading as our bug. Now:
  - The compiler stops once the grammar passes 8 MiB and fails the
    schema as `SchemaError::TooComplex`; `Grammar::parse` refuses any
    source over 8 MiB or 2^18 rules (`GrammarError::TooLarge`) as a
    backstop. Real grammars are far smaller (a large tool set compiles
    to a few hundred KiB).
  - An empty `enum` admits no value: `SchemaError::EmptyEnum`, wherever
    in the schema it sits.
  - The errors surface as `DialectError::Schema`, `ToolChoiceError::Schema`
    and `OutputConfigError::Schema` (each naming the tool where there
    is one), all 400 `invalid_request_error` on blallama. Anthropic's
    own wording for these was not captured; the messages are plain.
  - `schema_to_gbnf` (doc-hidden, the fuzzer's entry) returns
    `Result<(), SchemaError>`.

- **A recursive `$ref` no longer aborts the server.** The schema
  compiler inlined every `$ref` it met, with no cycle guard, so a
  recursive schema — `{"$ref": "#/$defs/Node"}` whose `Node` has
  `children: {items: {"$ref": "#/$defs/Node"}}`, exactly what
  schemars emits for a tree type — recursed until `fatal runtime
  error: stack overflow, aborting`: an abort, not a panic, so one such
  tool or `output_config` schema took down the whole blallama process
  and every request in it. Every dialect was exposed (Hermes/JSON-
  native, Mistral, Gemma's dict encoding, gpt-oss Harmony, and Qwen
  XML since its per-parameter schemas began resolving the tool's
  `$defs`), as were strict `tool_choice` grammars and structured
  output. Each referenced def now compiles once, to its own named rule
  (`<root>__def<n>_<Name>`) that every `$ref` to it names, so recursion
  lives in the grammar; defs come off a worklist, not nested calls, and
  alias chains (`A → B → C`) resolve in a loop to their end. A
  reference that loops back before any byte of the value — a
  self-alias, an alias cycle, a left-recursive `anyOf` — would be left
  recursion, and is unconstrained instead (`value`), as it is to the
  schema check. That check, and the tagged-value classifier, follow
  the same rule; the check memoizes each def's verdict per value and
  caps its nesting (past it, a value passes rather than overflow), and
  the classifier visits each def once, so neither a chain thousands of
  defs long nor a diamond of `anyOf`s (`D_i = anyOf[D_{i+1}, …]`, 2^n
  paths) can overflow or stall them. Everything else the schema
  walkers recurse on is bounded by serde_json's 128-level nesting
  limit on the request. Pinned on every dialect with a tree, mutual
  recursion and alias chains (`recursive_refs_compile_on_every_dialect`)
  and on `output_config` and strict `tool_choice`; chain, diamond and
  depth bounds on small thread stacks in `grammar_compile`.

- **A nullable-string tool argument on a tagged (Qwen XML) dialect
  parses as the string the model wrote.** The grammar generates an
  `Option<String>` parameter (`"type": ["string", "null"]`) raw, like
  any string, but the parser JSON-parsed it: a `5` or `true` the model
  wrote as text came back a number or boolean, `None` came back
  `null`, and a `"quoted"` value or `{"a": 1}` lost its spelling, so
  the turn re-rendered differently than it was generated. It is now
  read raw, with JSON `null` its one non-string value. Enum-constrained
  parameters are unchanged. Pinned in
  `qwen_cache_stable_round_trips_scalar_args`.
- **A Qwen3.6 tool call with a `null` or boolean argument re-renders
  as written.** The grammar has the model write a non-string
  parameter as JSON and the parser types it from that, but stock 3.6
  re-renders every non-container value `| string`, which minijinja
  spells Python-style: a `<parameter=detail>\nnull\n</parameter>`
  came back as `none` (and from minijinja 2.24 booleans come back as
  `True`/`False`, drama_llama#120), so the next request lost the
  turn's KV — `emission_not_byte_stable` then `tip_diverged`, 359..6909
  tokens a turn, live on Agora 2026-10-01. The baked 3.6 template now
  renders every non-string argument with `tojson`, stock 3.8's own
  rule; strings still render raw, so a string-typed value that looks
  like `null`, `true` or `5` stays that string. The analyzed dialect is
  unchanged. Pinned for both templates with null, boolean, integer,
  float and literal-looking string arguments, stock 3.6 as the control
  (`qwen_cache_stable_round_trips_scalar_args`); a number in a
  non-canonical spelling (`1.50`, `1e3`) re-renders canonically and is
  pinned as irreducible (deviation 6 in `templates/README.md`).
- **gpt-oss structured output is constrained again — it never was.**
  The `output_config` grammar was dialect-blind: its phase-split
  trigger was a hardcoded `</think>`, which a Harmony model never
  writes, so with thinking on the JSON body ran entirely unconstrained,
  and with thinking off the unified grammar demanded `{` where gpt-oss
  writes its channel header. On 2026-10-01 an Agora consent question
  came back as `"soul_text":"", "$memory_note":""}` and as
  `"soul_text":"", ""}` — each a 200 `end_turn`, and a mis-read
  consent. `OutputConfigOptions::framing` (`ResponseFraming::Bare` |
  `Harmony`, filled by `Session` from the dialect on every call, like
  the thought separator) now puts the body in the final channel: the
  deferred grammar triggers on `<|channel|>final` and constrains the
  rest of the header (` <|constrain|>json` optional, as gpt-oss writes
  it) and the body; the unified one admits at most one analysis block
  before it. The token-level hypotheses — a multi-byte token such as
  `""` or `", "` judged without walking it, a reset after an empty
  string — are ruled out by
  `harmony_output_config_rejects_the_live_bodies_by_token`, which
  drives the sampler's own legality checks over gpt-oss's real
  tokenizer (`vocab_only`, `#[ignore]`d) and fails on the old framing.
- **Gemma 4 and Mistral 4 structured output is constrained with
  thinking on.** The same hole as gpt-oss's, one dialect over: the
  phase-split trigger was `</think>` for every non-Harmony dialect, and
  Gemma 4 closes its thought with `<channel|>`, Mistral 4 with
  `[/THINK]`, so their json_schema bodies ran unconstrained. The
  trigger is now the dialect's own closer, whitespace-trimmed as the
  parser reads it (`OutputConfigOptions::thought_open` /
  `thought_close`, filled by `Session` from the dialect's reasoning
  markers like the separator), and the unified grammar's optional
  thought is spelled in the same markers — its body now runs to the
  closer, so a thought may contain a `</` that is not one. Qwen and
  cogito keep `</think>`; a dialect that measured no reasoning markers
  keeps `<think>…</think>`. Thinking off needed no fix: neither format
  frames its content, so the unified grammar's `{` first is right.
  Reproduced at the byte level on the live gpt-oss bodies
  (`tagged_reasoning_output_config_constrains_the_body`, red on the old
  trigger).
- **A deferred output_config grammar that never activated is a grammar
  violation.** When the model never writes the trigger (it skipped the
  thought, or wrote a closer the grammar did not know), the answer ran
  unconstrained and was returned as if constrained. It is now
  `SessionError::GrammarViolation` — or `SchemaViolation` when the free
  body also breaks the schema — with the cache left warm, since no
  constraint ever started; blallama resamples it. The Auto tool-call
  lazy grammar keeps its exemption: never calling is legal. A render
  that already closed the turn's thought (a prefilled closed thought,
  thinking on) now gets the unified grammar, since its closer can never
  be written and the deferred one would never fire.
- **A structured answer written without a thought is steered, not
  refused (Gemma 4, Mistral 4, cogito).** With thinking on, these
  renders leave the thought optional, but the body still waited for the
  thought's closer — so a model that answered `{…}` straight away woke
  nothing, ran free, and drew the never-activated `GrammarViolation`
  above on every greedy or seeded draw. The body now defers only where
  the trigger is certain to come: a render that opened the thought
  (Qwen) or Harmony's final channel. Every other call gets the unified
  `( thought gap | ws ) body` grammar from the first token.
  `OutputConfigOptions::phase_split` documents the rule.
- **cogito's structured-output thought parses as a thought.** cogito's
  template has no reasoning markers, so its `output_config` grammar
  offers `<think>…</think>` (the model thinks in it when its template
  asks for deep thinking), but the parser read no thought at all: the
  thought stayed in the answer's text, and `</think>\n{…}` failed the
  schema on every draw. A call whose output_config grammar offers that
  fallback thought now parses with the same markers; other calls (no
  structured output, or a forced tool) still read cogito's `<think>` as
  text. Known cost: the parser trims whitespace beside the markers
  (`<think>\n…</think>\n\n{` re-renders as `<think>…</think>\n{`),
  so such a turn's tip is not byte-stable — a one-turn cache miss.
- **The gap after a thought is bounded whitespace.** Every
  output_config grammar admits the measured separator after a thought
  *and* any other run of up to two whitespace bytes
  (`OutputConfigOptions::thought_separator`, `THOUGHT_GAP_MAX`). A
  literal gap masked what models write — `[/THINK]\n{` and
  `[/THINK] {` against Mistral 4's measured empty gap, `<channel|>\n\n{`
  against Gemma 4's single byte — and where the gap rode in on the
  closer's own token (cogito's `>\n\n` after `</`, a text-spelled
  `]\n` after `[/THINK`) the deferred body woke on bytes it refused,
  which ends the turn. The bound keeps whitespace from running on.
- **A token that would wake a deferred grammar illegally is masked.**
  While a deferred grammar sleeps nothing masks the vocab, so a token
  that finished its trigger and carried bytes the grammar refuses was
  sampled, woke it, and the predictor ended the turn mid-structure. The
  sampler now checks such a token's tail against the sleeping grammar
  before accepting it and resamples without it, like any other
  grammar-illegal pick (`sample_token_in`, given the generated text so
  far; the public `Candidates::sample_token` judges what one piece
  spells). Pinned by the vocab-only token-level tests
  `qwen38_output_config_by_token`, `gemma4_output_config_by_token`,
  `mistral4_output_config_by_token` and `cogito_output_config_by_token`
  (`#[ignore]`d; `DRAMA_LLAMA_{QWEN38,GEMMA4,MISTRAL,COGITO}_MODEL`):
  the live invalid bodies are masked inside the body, the valid body
  with each natural gap is admitted, complete and EOS-legal and keeps
  the turn contract, and the trigger-crossing tokens are admitted or
  masked — never fatal.
- **A Qwen XML tool's string `enum` argument is written raw, as the
  template renders it.** The grammar wrote such a value as JSON
  (`"full"`, quoted) but the parser read it raw, so the tool got
  `"\"full\""` — and, for a `strict` tool, the schema backstop refused
  it on every draw and blallama answered 500. Qwen's template renders
  any string argument raw (`args_value | string`), so the model was
  trained on `<parameter=detail>\nfull\n</parameter>`, and a quoted
  value re-rendered without its quotes: a tip miss on every such call
  (Agora's `detail`). In a tagged dialect (`Family::TagWithTagged`;
  of the fleet, Qwen 3.6 and 3.8) a string value is now never
  JSON-quoted at the top of a parameter: a parameter that admits only
  finitely many values, at least one a string — `enum`, `const`,
  nullable or not, through `$ref`, `anyOf` or schemars' `oneOf` of
  `const`s (`Option<DetailLevel>`) — is generated as an alternation of
  its members' raw spellings (`full`, `null`) and read back by exact
  match, so the round trip is byte-for-byte and a member passes the
  `strict` backstop. A mixed set (`["a", 1, null]`) spells its strings
  raw and the rest as JSON, as the template renders each; a set whose
  spellings would collide (`"1"` beside `1`, `"null"` beside `null`) or
  whose member contains the close tag stays JSON. A union of a free
  string and `null` (`anyOf: [{"type": "string"}, {"type": "null"}]`)
  is now raw like `"type": ["string", "null"]` already was. A quoted
  member from an unconstrained model still reads as the member. JSON
  dialects (Hermes/cogito, Mistral, gpt-oss, Gemma) are unchanged.
  Also: a tagged parameter's `$ref` now resolves against the tool's
  `$defs` (it compiled to any JSON value before).
- **The schema backstop reads every number the grammar can write.** The
  grammar's `number` has an unbounded integer part, and serde_json
  refuses one too large for `f64` ("number out of range"), so such an
  answer was `NotJson` on every draw; it now reads as a number. (Lone
  surrogate escapes and three-digit exponents, serde_json's other
  refusals, are already unwritable under the grammar — now pinned.)
- **A real `<tool_call>` always arms the grammar (#101).** The marker
  dialects (Hermes, cogito, Qwen's XML) triggered the lazy tool-call
  grammar on the whole opener, `<tool_call>\n`, and the parser looked
  for the same bytes. A real `<tool_call>` the model followed with
  anything else — `{`, a space, `\r\n`, EOG — never armed the
  grammar, was not parsed as a call, and landed in prose, where
  containment rejected the turn (`EmittedSpecialToken`): ~10% of
  cogito-32b attempts in the live Agora cohort, each a resample.
  `CallSyntax::trigger` (and so `triggers`, and the legacy
  `deferred_grammar_for_prompt`) is now the marker with its trailing
  whitespace trimmed — the special itself, still a prefix of the
  opener; the grammar, which starts at the full opener, takes
  over one token earlier and forces the canonical newline, so a real
  opener is seated as a call or surfaces as a `GrammarViolation`,
  never as prose. The parser reads openers whitespace-tolerantly
  (`<tool_call>{…}` is a call), canonical shapes are unchanged
  byte-for-byte, and dialects whose trigger was already a bare marker
  (Gemma 4, Mistral 4's `[TOOL_CALLS]`) and Harmony's recipient
  headers are untouched. Behaviour change: `trigger()` returns
  `"<tool_call>"` where it returned `"<tool_call>\n"`; the untrimmed
  `section_start` / `per_call_start` remain the canonical bytes. A
  model naming its own opener in prose now gets a call forced — the
  special was rejected from free text anyway.
- **A tool call that repeats an earlier one in the same turn is
  dropped.** Arming on the bare special has one risky shape: a stray
  real opener after a *finished* call (`{call}\n<tool_call>`, then end
  of turn) gets a second call forced, and a model with nothing more to
  say fills it with the call it just made — a duplicate `create_post`
  or `vote` the client would dispatch twice. A call with the same name
  and an identical input (JSON value equality: member order aside,
  `1` and `1.0` differ) is now dropped from the turn, batch and
  streamed alike, and logged at `WARN` (`event = "tool_call_dropped"`,
  the tool name only). Identical parallel calls are not observed from
  Anthropic, so one is treated as never intended; a call that differs
  in anything is a parallel call and stays. Only a *complete* call is
  judged: a call cut by `max_tokens` or a stop sequence holds only the
  members that completed, so it could match a call it would not have,
  and it comes back cut (and, streamed, left open by
  `BlockStream::open_call_json`) as on Anthropic — a client runs no
  call of a turn that ended that way anyway. The stream judges a call
  before yielding it, and the parser releases calls only whole, so no
  part of a dropped one is ever seen. The batch turn leaves no
  auto-tip: its KV holds the dropped call, which no re-render of the
  returned turn reproduces.
- **The #101 containment log says where.** The `EmittedSpecialToken`
  error event now carries up to three `hits` (block index and kind,
  offset in the block and in the emission, ~96 bytes of context each
  side, and the 8 bytes after the special), whether a deferred grammar
  ever activated, `emission_bytes` and `generated_tokens`. Operator
  trace only; the error's `Display` still withholds the pieces.
- **Cogito tool turns keep their tip.** Cogito's stock template prints
  `\n` before every call, and the model ends its prose in whitespace
  (`…\n\n<tool_call>`), so a prose-then-call turn re-rendered one
  newline longer than it was generated and lost its tip on every
  request (19 live events). The cogito bake now prints the
  prose-to-call gap only when the prose carries none — as the Qwen
  bakes do — and renders content-less turns and trimmed client prose
  exactly as before. Parallel calls agree too: the analyzer read the
  inter-call `\n` of per-call JSON templates as no separator (the two
  probe calls share arguments, and the diff came back rotated), so the
  grammar forced `</tool_call><tool_call>` against the template's
  `</tool_call>\n<tool_call>`; it now measures `call_separator: "\n"`.
  Not cogito-only: the NousResearch-Hermes-3 and Qwen3-0.6B templates
  measure the same `"\n"` now, and their parallel calls change the
  same way.
- **Qwen3.6 and Qwen3.8 turns re-render byte-for-byte: both get a
  baked cache-stable template.** Their stock templates `|trim` an
  assistant turn's answer and thought (3.6 also `lstrip`/`rstrip`s the
  halves it splits on `</think>`) and print a fixed `\n\n` after the
  close, so a turn the model ended with whitespace, began with a
  newline, or whose thought closed on a blank line re-rendered shorter
  than it was generated, and the next request lost the whole turn's KV
  (the 2026-09-30 Qwen3.6 run's 7364-token tip). The embedded
  templates of every Qwen3.6 GGUF we serve (35B-A3B `UD-Q4_K_S`,
  `UD-IQ4_XS`) and of Qwen3.8-27B now detect and are replaced
  (`baked::QWEN36`, `baked::QWEN38`): the assistant turn renders
  verbatim, and the template supplies its `\n\n` after the thought and
  before the first call only where the content carries no whitespace
  there. Everything else is byte-identical to stock and analyzes to the
  same dialect. An aged turn (`preserve_thinking` off) still renders
  exactly as stock. Irreducible and pinned (listed in
  `templates/README.md`): a gap the model omitted entirely
  (`</think>Ada`) re-renders as the canonical one; an empty thought's
  lone `\n` answer gap renders after the scaffold's `\n\n`; a thought
  closed without its `\n` gets it back; whitespace after the last call
  is dropped; and (3.6 only) a thought containing a literal `<think>`
  loses everything before it. Byte-stable is not token-stable at the
  prompt seam: an emission *starting* with `\n` merges with the
  generation prompt's trailing newline token under BPE (`\n\n` + `\n`
  → `\n\n\n`), so such a turn still re-prefills unless its tip is
  reached by hash.
- **A mid-conversation system turn renders on Qwen3.6 and Qwen3.8.**
  misanthropic's `Chat` seats in-conversation System notes (Anthropic
  accepts them on some models), and stock Qwen3.8 raises on any system
  turn past the leading run — "chat template raised: System message
  must be at the beginning." (live, Qwen3.8-27B `UD-Q8_K_XL`,
  2026-09-30) failed the whole request. Stock Qwen3.6 accepted the
  same request only by dropping the note without a word, so the model
  never saw it. Both baked cache-stable templates now render such a
  turn as its own `<|im_start|>system\n…<|im_end|>` block where it was
  seated, content trimmed like the leading system text; the leading
  system/tools header is byte-identical to stock and the analyzed
  dialect is unchanged
  (`qwen_cache_stable_renders_mid_conversation_system`, deviation 5 in
  `templates/README.md`). blallama already answered a template raise
  with Anthropic's 400 `invalid_request_error`, not a 500; that is now
  pinned (`template_raise_is_anthropic_400_envelope`).
- **A forced Harmony call can no longer run to `max_tokens` in
  analysis and commentary.** The eager (`tool_choice` `any` / `tool`)
  gpt-oss grammar admitted any number of analysis and commentary
  blocks before the call, and under it EOG is illegal and `final` is
  not a channel — so gpt-oss-120b, forced to call again after it had
  already answered, alternated "Now final." / "pong" / "We need to
  stop." / "[END]" blocks for 16384 tokens. The root is now one
  optional analysis block, one optional commentary preamble, then the
  call: the one-thought-then-calls shape every other dialect's eager
  grammar already had. `tool_choice: none` was never the cause — on
  Harmony it resolves no grammar and bans nothing, so the final
  channel's `<|return|>` ends the turn as it does without tools.
- **A turn cut short is a response, not an error (#121).** When
  `max_tokens` (or the context window) ran out mid tool call, the
  batch path returned `SessionError::GrammarViolation`, blallama
  resampled twice — failing identically, the budget being the budget —
  and answered HTTP 500 `api_error`. Anthropic answers 200 with
  `stop_reason: max_tokens` and the partial turn, and clients key their
  clip handling on that stop reason. Now so does `Session`: the turn
  comes back `MaxTokens` with usage filled, and the incomplete call
  comes back **as Anthropic returns it**: a `ToolUse` whose input holds
  only the members that *completed* — the one being generated dropped
  whole, however far it got — and none of its bytes seated as prose;
  calls that closed before the cut stand unchanged. Captured
  2026-09-30 on claude-haiku-4-5, raw bytes (misanthropic's
  `misanthropic/test/data/stop/clip*.*`, requests in
  `misanthropic/test/data/requests/`): a `write_file` cut mid-`contents`
  came back `{"path":"hello.py"}`, and 140 output tokens into a
  200-word `contents` string still `{"path":"story.txt"}`; streamed,
  the `input_json_delta` chunks stop at the last completed member and
  the block never gets `content_block_stop`. `{}` when no member
  completed. Nested containers keep their completed members at every
  depth, the one in progress dropped (inferred — only the top level is
  captured). A call whose input closed but whose dialect close marker
  was cut keeps its whole input. A call cut before its *name* is whole
  has nothing to return and is left out. The same prompt must drive a
  client the same way on both backends. A cut
  outranks `ToolUse` in the stop reason, so a turn clipped mid-way
  through its second parallel call never reads as a finished call
  turn. **A `max_tokens` turn can therefore carry complete calls, and
  a cut one that looks complete: clients must gate dispatch on
  `stop_reason: tool_use`, never on the presence of a `ToolUse`
  block** — as on Anthropic. One exception in blallama, deliberately
  better than parity: a cut turn that had already repeated a call
  verbatim (same tool, same input — the identical-call loop the old
  grammar-violation check caught, plan Phase G) is resampled on the
  warm cache like the other unlucky draws, and answered `max_tokens`
  only if every draw loops. The mechanism is a third parse leniency,
  `dialect::Leniency::Clipped` (and `StreamParser::finish_clipped`):
  incomplete calls cut short, unclosed thoughts surfaced open as under
  `Final`. Bare-JSON dialects are exempt — any `{` is their call
  landmark, so a clipped structured output keeps its text. A forced
  call that finishes on the budget's last token still reports
  `ToolUse` — streamed too: `complete_stream` now halts on an exhausted
  grammar as the batch path does, and reads the ending by the same
  rule; `BlockStream::open_call_json` reports the cut call's input as
  the JSON an Anthropic stream leaves open (`{"path":"story.txt"`), for
  an SSE bridge that must not send its `content_block_stop`.
  `GrammarViolation` remains for a constraint that failed with budget
  to spare. A clipped turn whose KV no longer matches its output (a cut
  call, which re-renders closed, or a cut mid-constraint) records its
  prompt extent but no auto-tip, leaving the generated span to the
  next call's LCP walk — a cache miss on that turn, which two
  breakpoints at the end of the prompt contain. The JSON repair is a
  pure helper, `dialect::truncate_partial_object`.
- **A Mistral call at its `[ARGS]` marker is incomplete, not
  malformed.** `[TOOL_CALLS]name[ARGS]` with nothing after it (the
  arguments not yet generated) parsed as malformed, so the streaming
  parser yielded the frame as prose and a clip there seated it in a
  `Text` block. It now waits (streaming) or comes back with input `{}`
  (clipped). Under `Leniency::Clipped`, a call that the cut left
  unreadable and that runs to the end of input is withheld as the call
  in flight, Harmony blocks included.
- **Request `stop_sequences` stop generation (#122).** They were read
  only after the fact, to label a turn that happened to end on one.
  Generation now stops at the first match **in client-visible
  text**, the match is cut from the output, and the response reports
  `stop_reason: stop_sequence` with `stop_sequence` set — batch,
  `complete_text` and `complete_stream` alike (the stream holds back
  text that could still grow into a stop sequence). Matching goes
  through the dialect parser and sees only prose (`Block::Text`) and
  the string values of a tool call's input — the call still being
  generated included, so generation stops at the match, not when the
  call closes. A match in a call's input **cuts the call there**, as
  Anthropic does: the string is cut right before the match, the
  members before it stand, the JSON is closed, and nothing after the
  match exists — `stop_sequences: ["print("]` on a forced `write_file`
  gives `{"path":"hello.py","contents":"import datetime\n"}` under
  `stop_reason: stop_sequence`, as claude-haiku-4-5 did (captured
  2026-09-30; misanthropic's
  `misanthropic/test/data/stop/stop_sequence_tool.*` and
  `stop_sequence_text_tool.*`; streamed, with a normal
  `content_block_stop`). `complete_text`'s raw bytes then end inside the
  call, right before the match. Never matched: thinking, and framing —
  the dialect's markers (`<tool_call>`, `<function=…>`,
  `[TOOL_CALLS]`/`[ARGS]`, Harmony
  headers, EOG pieces) and the whitespace between prose and a
  structure (`"Sure, checking.\n\n<tool_call>"`, `"</think>\n\n"`).
  A stop of `"\n"` matched against raw bytes killed every Qwen call at
  its opener, and matched against the prose the parser seats before a
  call, it still did. Whitespace inside prose, or ending a turn that
  finished cleanly, is text and matches. (Anthropic's behavior inside
  `thinking` is uncaptured.) The
  predictor no longer carries a request's stops; its own stop-string
  window is now sized from the stop strings' byte lengths too — it was
  sized from token-sequence lengths alone and missed any stop string
  longer than a token. New: `TokenPredictor::stop_string` /
  `hit_token_limit` (and on `PiecePredictor`),
  `BlockStream::stop_reason` / `open_call_json`, `StreamParser: Clone`,
  `dialect::truncate_partial_object`.
- **blallama answers an undeserializable body with Anthropic's 400
  (#123).** `/v1/messages` and `/v1/messages/count_tokens` returned
  axum's plain-text 422 (or 415), which no Anthropic client parses; they
  now return `{"type":"error","error":{"type":"invalid_request_error",…}}`
  with status 400 (413 `request_too_large` for an oversized body). The
  body limit is raised from axum's 2 MB to Anthropic's 32 MB, which a
  single base64 image could exceed.
- **blallama rejects a whitespace-only stop sequence, as Anthropic
  does.** A `stop_sequences` entry with no non-whitespace character
  (`"\n"`, `" "`) is a 400 `invalid_request_error` on Anthropic,
  message `stop_sequences: each stop sequence must contain
  non-whitespace` (captured 2026-09-30 on claude-haiku-4-5 — see
  misanthropic's `misanthropic/test/data/stop/whitespace_stop.error.json`
  — for
  `stream: false` and `stream: true` alike — the streaming request gets
  the plain JSON body too). blallama accepted it and generated; it now
  answers the same envelope, status and message, on `/v1/messages` and
  (assumed, uncaptured) `/v1/messages/count_tokens`. The session's
  whitespace-framing rule for stop sequences (#122) is unchanged: it
  still governs library callers and stops that merely begin or end with
  whitespace (`"\nObservation:"`).
- **Docs:** `grammar_compile`'s claim that Anthropic hoists required
  properties ahead of optional ones is dropped; probed live, it keeps
  optionals in place, exactly as the grammar does (misanthropic
  `9be105f`).
- **Qwen3.8 thinking turns re-render byte-stable; the tip anchors
  (#112).** Two causes, both in how a dialect describes reasoning.
  (1) The analyzer never *measured* the thought re-ingest convention:
  every `<think>` dialect defaulted to `InlineThink`, but Qwen3.8's
  template dropped 3.6's `content.split('</think>')` and reads
  `reasoning_content` alone, so every thinking turn re-rendered its
  thought as content behind an empty `<think></think>`. The analyzer
  now renders both conventions and picks the one the template honours
  (subsuming the Mistral `[THINK]` source patch). (2) Under a grammar
  (forced tool call, `output_config`), the gap after `</think>` was a
  free `[ \t\n\r]?` — one byte at most — so Qwen's canonical
  `</think>\n\n` was unreachable. The analyzer now measures
  `ReasoningSyntax::separator` and the tool and JSON grammars spell it;
  unmeasured dialects keep the old gap. Also fixes the same gap on
  Qwen3.5/3.6 and pins Mistral 4's to empty.
  **API:** `OutputConfigOptions` gains `thought_separator` (Session
  fills it from the dialect); full struct literals need
  `..Default::default()`.

- **`Session::complete_text` halts on `grammar_complete()`**, exactly
  where `run_call` (and so `complete_blocks` / `complete_response` /
  blallama) always has. It used to run on to EOS or `max_tokens`, which
  showed not "raw" output but post-grammar drift: a forced Mistral call
  went `[TOOL_CALLS]` over `</s>` 26 times, and Qwen repeated the same
  call under greedy, until the budget cut one mid-JSON — behaviour no
  production path ever sees. The "two views of the same bytes" contract
  between `complete_text` and `complete_response` needs the same
  stopping rule on both sides; `top_k_trace` remains the way to look
  past the grammar. `multi_call_round_trips_under_greedy` now asserts
  the analyzed `call_separator` directly (#58's actual fix) and a
  single-call byte-exact round-trip — the multi-call emission it relied
  on was that loop.
- **A truncated last tool call no longer re-emits the complete calls
  before it.** `parse_calls` degraded from the *section* start on an
  incomplete or malformed call, so under `Leniency::Final` every call
  already parsed into a `ToolUse` block was pushed a second time inside
  the degraded `Text` — a 25-call Mistral `[TOOL_CALLS]` turn cut by
  `max_tokens` re-rendered as 50 calls, unreachable to any cache
  breakpoint. Degradation is now scoped to the call it happened in; the
  cut call itself is still unrepresentable (half a JSON object) and
  still degrades to text for `Session` to contain. Latent since #85;
  surfaced by `session_mistral4::emission_round_trips_through_parse_
  and_render` once the tool-call loop there started reproducing.
- **Cache-breakpoint partials now carry the request's `thinking`.**
  `render_partial` built its truncated prompt with `..Prompt::default()`,
  dropping `thinking`, so every partial rendered with
  `enable_thinking = false`. Mistral Small 4 writes that switch into
  the prompt prefix (`[MODEL_SETTINGS]{"reasoning_effort": ...}`), so
  under a thinking-on request no partial was a byte prefix of the full
  render and the entry-prefix check silently dropped every breakpoint:
  slots held only the 5-minute tip, new agents found no anchor in the
  shared system+tools prefix, and any turn whose re-render was not
  byte-stable re-prefilled from zero. Surfaced 2026-09-12 when the
  Agora runner began sending `thinking`; Qwen was unaffected because
  its switch only shapes the generation tail, which partials never
  render. Pinned model-free with a prefix-switch template.
- **No more double BOS on add_bos models (#93).** Mistral's template
  emits `<s>` itself and llama.cpp's pixtral/tekken vocab has
  `add_bos`, and every render was tokenized with `add_special = true`,
  so each prepared prompt (and every cache-breakpoint partial) began
  `BOS BOS` — the `check_double_bos_eos` warning that fired 3 to 5
  times per Mistral request. Gemma 4 and Llama 3 templates have the
  same shape. `tokenize_render` now tokenizes a render that already
  starts with the BOS piece with `add_special` off (llama.cpp's chat
  path strips the piece for the same reason); renders without it keep
  the auto-BOS. The three emit-ban sites tokenize their markers the
  same way, so BOS no longer lands in the ban sets by accident. Cached
  Mistral prefixes change by one token, so each agent pays one cold
  prefill after upgrade.
- **A model-emitted stray `</think>` is masked at the sampler whenever
  the render has already closed the turn's thought (#109).** The #107 opener
  ban had no closer counterpart: the standing emit ban exempts the
  closer unconditionally (it is the phase-split trigger), so under a
  thinking-off render (Qwen's `<think>\n\n</think>\n\n` stub, Gemma
  4's `<channel|>` stub, a prefilled closed thought) a thinking-native
  model that still wants to reason — observed on Qwen 3.6 after a
  `tool_result`, when the transcript pulls it toward a tool it was not
  given — reasons in the open and then closes the thought it never
  opened. #101's containment rightly rejected the bare closer, three
  identical attempts deep, and every retry of that prompt failed the
  same way: a deterministic 500 that wedged the calling agent for
  good. A per-call `reasoning_closer_ban` (the closer's specials, EOG
  excluded) is now unioned into `banned_specials` exactly when the
  render ends with a closed stub — never on a pre-opened render, where
  the closer is the model's job — so the reasoning simply continues as
  prose the model finishes. Verified by A/B against the pre-rebase
  llama.cpp: the failure predates the llama-cpp-sys update it was
  first blamed on. Callers who want the reasoning *as* a thought
  should enable `thinking` on the request; the stub is Anthropic's
  `thinking: None` semantics, not a template failure.
- **The eager output-config grammar no longer forces a duplicate
  `<think>` under a pre-opened render (#107).** Root cause, found by
  arm-by-arm config bisection against the raw predictor: with
  `prompt.thinking` unset, `compile_prompt_output_config` compiles the
  *unified* grammar whose root was `thought? ws output_schema` — an
  **optional `"<think>"` opener literal** — while the template (Qwen
  3.6 with the `enable_thinking` render extra) had already pre-opened
  the thought. At position 0 the grammar masked the model's actual
  preference (`"Here"`, p≈0.98 raw) as illegal and offered
  `{<think>, ws, {`}; the model picked the opener — deterministically,
  at every seed — and #101's containment then rightly refused the
  response (`EmittedSpecialToken(<think>)`, red since 2026-07-29,
  silently present since at least 2026-07-20). The model never wanted
  the duplicate: 0/9 seeds emit it through the raw predictor.
  `build_grammar_source` now takes `thought_pre_opened` (threaded from
  the render, same measurement the tool grammars use) and anchors the
  pre-opened root close-first — `think_body "</think>" ws
  output_schema`, no opener literal, thought mandatory, dominating
  `allow_thought = false` per the `EagerThoughtPreOpened` precedent.
  This also closes the early-stop hole on this path: the grammar is
  incomplete until the JSON exists, so EOG is illegal mid-thought. The
  deferred (phase-split) path is unchanged — its trigger is the
  closer, which is the model's job to emit either way.
- **A model-emitted duplicate `<think>` is masked at the sampler
  whenever the turn's opener is already supplied (#107,
  defense-in-depth).** The grammar fix above removes the *forcing*;
  this removes the single-token *path*: the standing emit ban's
  reasoning-opener exemption is now conditional. A per-call
  `reasoning_opener_ban` is unioned into `banned_specials` whenever
  the render shows the opener is spent — pre-opened thought, closed
  thinking-off stub (Qwen's `<think>\n\n</think>\n\n`, Gemma 4's
  default `<channel|>` stub), or a resumed open thought. The closer
  stays exempt (it is the phase-split trigger); self-opening dialects
  keep the exemption (the ban keys on the *render*, not the dialect —
  gpt-oss and thinking-on Gemma are untouched);
  `with_emit_specials_ban(false)` disables this set along with the
  other two. Id-level only, as with the rest of the ban family: a
  byte-spelled opener still lands in free text and containment keeps
  rejecting it.
- **The post-generation tip anchors continuations again (#96).** The
  prefix cache's two lookups composed as hash-keyed first, LCP only on
  a total hash miss — and every continuation re-renders its old
  `cache_control` markers to identical partials, so the hash path
  always matched something and capped reuse at the last explicit
  marker. The tip (the internal anchor at the KV head, past every
  marker) was never consulted unless the client happened to mark the
  re-ingested assistant turn. Every drama_llama release with hash-keyed
  reuse re-prefilled the entire final turn on every continuation.
  `slot_l_hit` now runs both lookups and takes the larger offer; both
  prove their prefix in both coordinate spaces, so the max is always
  sound. Two debug lines make future losses visible: one when a live
  tip loses the pick, one when `tip_extension` declines to build a tip.

### Added

- **Baked chat templates and a loading ladder ([#88]).** A model's
  template is resolved sidecar first, then a *baked* cache-stable
  replacement shipped in the crate (`baked`) when the embedded template
  is byte-for-byte one we validated, then the embedded template with a
  warning. Bakes ship for Qwen3.6, Qwen3.8, Gemma 4, gpt-oss (two
  detection keys: Unsloth's and OpenAI's upstream template, [#99]),
  Mistral Small 4 and cogito, each pinned by round-trip tests and a
  per-model suite. A template that misses detection but analyzes to a
  known stock dialect is named in the warning (`baked::nearest_stock`).
  `scripts/gguf_template.py` checks a GGUF's template against the keys.
- **Mistral Small 4 support ([#88]):** baked template,
  `[TOOL_CALLS]name[ARGS]{…}` calls, `[THINK]` reasoning, pixtral
  vision, and the `session_mistral4` suite. Needs `llama-cpp-sys-3`
  0.8.4, whose Metal `mul_mm_id` fix ends the all-NaN decode above a
  32-token micro-batch.
- **`DecodeError::NonFinite`.** A decode that returns NaN or infinite
  logits fails at the decode boundary, KV-dirty, instead of panicking
  later in the sampler's sort.
- **Per-model load sidecar, `<model>.load.toml` ([#118], [#138]).**
  `n_ctx` sets the model's context (capped at its trained window,
  overriding `--n-ctx`) and `n_ubatch` its micro-batch (clamped to
  `n_batch`; an explicit `LlamaCppOptions::n_ubatch` still wins).
  Unknown keys are a parse error; a sidecar that won't read is logged
  and ignored. `/v1/models` advertises the per-model ceiling before the
  model loads, and the load logs `event = "context_size"` with each
  value's source. `LoadSidecar`, `LlamaCppEngine::n_ubatch`.
- **Parallel tool calls.** A constrained turn halts when its grammar is
  exhausted, not at the first accepting state, and end-of-generation is
  legal by id at every accepting state, so between calls the model
  chooses another call or the end of its turn. This also ends Mistral
  Small 4's forced call-after-call loop. A forced `tool_choice` now
  means at least one call, as on Anthropic; ask for exactly one with
  `disable_parallel_tool_use`.
- **Known-id exemption ([#113]).** `RepetitionOptions::id_patterns`
  (`IdPattern`, regexes compiled at the sidecar door) name what an
  identifier looks like; a token that faithfully copies one the prompt
  holds is never penalized, multi-word ids (`September 22, 2026`)
  included. Thoughts are not searched for ids.
- **`IgnoreCategory::Numbers` ([#113])** (digits and the bare space
  that starts a number) and **`IgnoreCategory::Markdown`** (code-fence
  tokens), both on by default and in every tracked sidecar. A closing
  fence copied from the prompt was penalized like a word; Qwen3.6
  replaced it with `}` or a stray tag in 3 of 4 draws.
- **Emitted-special containment ([#38], [#101]).** A turn whose free
  text holds a reserved special the model emitted as the real token is
  `SessionError::EmittedSpecialToken`, with the cache left warm, instead
  of an answer that fails the client's next request. blallama redraws
  it (twice at most).
- **blallama answers with Anthropic's error types ([#119]).** A request
  that fails the same way on every retry is a 400
  `invalid_request_error`, a transient failure a 500 `api_error`; both
  used to be a 500 `unknown`. `Session::with_strict_context_fit` refuses
  a prompt plus `max_tokens` past `n_ctx` up front
  (`SessionError::ContextOverflow`, Anthropic's `prompt is too long`
  400 on blallama) instead of truncating. `Session::count_tokens`
  backs `POST /v1/messages/count_tokens`.
- **The walked divergence point is a restore rung ([#102]).** Where a
  KV truncate alone rewinds (`Decoder::truncate_restores`: llama.cpp on
  a dense model), a prompt that parts from a cached slot between two
  anchors restores to where it parts instead of the anchor below it
  (`source=walk` in the logs).
- **Logs an operator can follow:** the grammar-overrule line names the
  model, block kind, tool and a tail of the value ([#140]); blallama
  logs its shutdown stages and `/probe` opens ([#95]).
- **Examples:** `soul_forge` (base-model generation through a completion
  scaffold), `id_fidelity` (scores verbatim id copying per penalty
  variant), `longform` and `replay` (replay captured requests, with
  `--model` and `--only`), and `logits_determinism` (a raw decoder
  determinism probe).
- **The per-model sampling sidecars are versioned** in `models/`.

- **An id copy-lock** (#144, `RepetitionOptions::id_copy_lock`, sidecar
  `[repetition] id_copy_lock`, default `true`). Once a copy is eight
  characters into exactly one known hex id from the prompt (the
  `id_patterns` matches: a UUID), the sampler takes the model's
  favorite among the tokens that continue that id and are legal there
  (grammar, bans, framing), until the id is complete. cogito copied a
  `reply_to` UUID 28 characters right and then drifted, 404ing a
  comment that existed. Fewer than eight characters, a prefix two ids
  share, and ids whose first eight are not hex (dates, handles) or are
  one repeated digit (the `00000000-…-0001` system sender, a nil UUID)
  never lock; at exactly eight the copy may still end as a short label.
  With no legal continuation the step samples as before. A capped turn
  is the tool-call cap's. Logged at `DEBUG` (`event =
  "id_copy_lock"`, `outcome`, `id_len`, `matched`).
- **A hard per-turn cap on client tool calls** (`ToolCallCap`). The
  per-model sampling sidecar's `max_tool_calls_per_turn` sets it, as
  does `Session::with_max_tool_calls_per_turn`. A request's
  `tool_choice.disable_parallel_tool_use` caps the turn at one, and the
  tighter cap wins. The sampler counts every call the tool-call grammar
  completes, repeats included, so a loop of one call meets the cap too.
  Once the last allowed call completes, it ends the turn on the model's
  own end of generation, the likeliest one the grammar admits. The KV
  and the re-render then agree, and the turn reports `tool_use` on the
  batch and streaming paths alike. The cap is logged at `WARN`
  (`event = "tool_call_cap"`, `cap`, `source`). cogito's tracked
  sidecar sets 3: it escalated parallel calls into loops that ran to
  `max_tokens` (16k tokens, 31 minutes, 2026-10-04).
- **`PrefixCacheConfig::adopt_emitted_tokens`** (default `true`): the
  switch for the fix above. Its one behaviour change is that a warm
  call can read a turn in the model's own split where a cold session
  reads the tokenizer's, so warm and cold output can differ under
  greedy sampling; `cached_output_matches_uncached_output` turns it
  off for that comparison, and `adoption_reuses_at_least_the_canonical_split`
  runs the default on real output with the tripwire armed.
  `count_tokens` with the cache on now depends on slot state the way
  the call does: the same prompt can count a few tokens differently
  before and after the slot it continues is evicted (documented there
  and tested).
- **Cache logging an operator can act on.** A `cache_reuse` miss —
  nothing reused, the whole prompt prefilled — is now always `WARN`,
  with `lost_tokens`, `prompt_tokens` and `cold` (nothing was lost: no
  slot offered anything, and the call neither missed a tip it
  continues nor refused a hash for its split — a new conversation's
  first turn, or one whose slot is gone), so a filter on `cold=false`
  leaves the misses that lost something; a hit
  stays `DEBUG`. `hash_drift` carries `diverge_at` and the `shared` /
  `cached` / `new` text around the first entry where the two
  tokenizations part, like `tip_diverged` (the live events could not
  say *which* boundary retokenized). A tip miss now also names where
  the *text* parts (`text_diverge_at`, `text_cached`, `text_new`) and
  sets `resplit` when that is past where the ids part — a stretch
  spelled in different tokens before the real edit, which used to put
  the operator at a spot where both sides read the same. One whose new
  prompt reads every byte the slot holds to the tip is reported as
  `segmentation_drift` (with `in_turn`) instead of `tip_diverged` or
  `history_changed`: same text, different split — what adoption
  exists to prevent, so it means adoption was off. A prompt read in a
  slot's own split logs `cache_adopt` at `DEBUG`.
- **`CheckpointBudget`, `LlamaCppDecoder::set_checkpoint_budget` /
  `seq_snapshot_bytes`** (and the `LlamaCppEngine` forwards) — bound and
  read the host RAM a decoder's checkpoints hold.
- **`LlamaCppModel::{n_swa, is_hybrid, is_recurrent}`** and
  **`LlamaCppDecoder::memory_seq_pos_min`** — what llama.cpp reports
  about a model's sliding window and recurrent layers, readable from a
  `vocab_only` load, and the lowest position a sequence still holds.
- **`baked::superseded` and a `WARN` for a stale template sidecar.** A
  `<model>.template.jinja` sidecar wins over the baked template, so a
  copy of an old bake silently holds back every fix to it since — the
  gpt-oss and Gemma 4 sidecars of the 2026-10-01 cohort run were the
  2026-07-27 bakes, byte for byte. The registry now keeps the SHA-256 of
  every superseded replacement, and loading a sidecar that matches one
  logs `stale_template_sidecar` at `WARN`, naming the bake and saying to
  delete the file.
- **A request's schemas are measured before anything compiles them**
  (`SchemaLimits`, `Session::with_schema_limits`,
  `SessionError::SchemaBudget`, a 400 `invalid_request_error` on
  blallama). Every custom tool's `input_schema` and an `output_config`
  `json_schema` are walked once, iteratively, before rendering,
  classification or compilation, and the request is refused, naming
  the limit and where, when it has more than:

  | limit | default | largest measured |
  |---|---|---|
  | custom tools | 512 | 15 (Agora) |
  | top-level properties per tool | 512 | 5 |
  | JSON values across the request's schemas, as written and with each `$ref` at its target's size | 131,072 | 882 (2,690) |
  | `$defs` + `definitions` per schema | 1,024 | 5 |
  | bytes of one `enum` member / `const` | 16 KiB | 24 |
  | member bytes across the request, each `$ref` at its target's size | 1 MiB | ~49 KB |
  | width: ways one schema's grammar can go on at once | 2,048 | 613 |
  | depth: levels of objects and arrays a value nests, each `$ref` at its target's | 64 | 3 |

  Measured on Agora's seed-agent request (15 tools and the `Soul`
  output schema), misanthropic's captured request fixtures, Anthropic's
  documented tool examples and a heavier synthetic tool (a 600-member
  time-zone `enum` behind a `$ref` four parameters name); the largest
  is 3× inside the width and 21× inside every other limit, Agora's
  request at least 29× inside all. The `$ref`-expanded total is the work a
  per-parameter pipeline would do: a large `enum` behind a `$ref` two
  thousand parameters name is two thousand copies of it, and a `$ref`
  fan-out with no members at all (a doubling chain of defs ending in
  `{"type": "string"}`) is as many copies of its values. A member's
  bytes are its compact JSON's, control characters at their escaped
  length (`\u001f` is six). Every hostile
  shape the rechecks found is refused in milliseconds, while requests
  at the limits (512 parameters over shared defs, 512 tools, 2,000
  nested optional properties) compile in every dialect in under 110 ms.

  The width bounds the grammar matcher's stacks, which it caps at 4096
  by refusing the excess — over-restricting the output — so a request
  inside the limit never reaches that cap. It is counted from the
  shape: an `enum` member or `const` 1; `boolean`/`null` 4, `number`
  8, `string` 16, `integer` 24 (its 18 optional digits are a stack
  each) and an untyped value 16, each the most measured in any dialect
  plus margin; an object its properties + 4 plus its widest property;
  an array 4 plus its items; an `anyOf`/`oneOf` the *sum* of its
  variants plus one, since variants sharing a prefix (objects all
  opening `{"a":`) are alive at once and nested ones multiply (the one
  is the alternation's own step in the schema check below, so a chain
  of one-variant `anyOf`s costs its length there too); a `$ref` its
  target's width, each def once, a reference back into its own cycle
  as untyped. Every shape filled to the default (an `enum`, optional
  and required properties, `anyOf`s of objects and arrays alive through
  an integer, `anyOf`s nested to multiply), as a tool parameter in
  every dialect and as structured output, peaks at or under its count:
  an `enum` exactly, the `anyOf`s at 60–95%, so under 2,048 stacks.
  Ambiguity *through* recursion — two interchangeable recursive defs
  doubling at every level of output — is not bounded by the count, only
  by the matcher's cap, which keeps every way to go on (see Fixed).
  Agora's widest schema counts 70.

  Every entry point that takes client schemas measures them first.
  `Session` checks the prompt in every `complete*` call and in
  `count_tokens` (which renders the schemas, and on the tagged
  dialects classifies them) against `Session::with_schema_limits`, and
  the grammars it then compiles are held to those limits too. The
  public compilers measure against a new `schema_limits` field
  (default `SchemaLimits::default()`) on their options:
  `dialect::grammar_source` (`EmitOptions::schema_limits`,
  `DialectError::SchemaBudget`), `grammar_for_tool_choice` and
  `deferred_grammar_for_prompt` (`ToolChoiceOptions::schema_limits`,
  `ToolChoiceError::SchemaBudget`), and `grammar_for_output_config` /
  `compile_output_config` / `compile_prompt_output_config`
  (`OutputConfigOptions::schema_limits`,
  `OutputConfigError::SchemaBudget`); `SchemaLimits::unlimited()` opts
  out. **Breaking** for code that builds `ToolChoiceOptions` or
  `OutputConfigOptions` as a full struct literal: add the field or
  `..Default::default()`. The non-streaming paths now classify a
  tagged dialect's tools once per call: a stop-sequence cut parses the
  output prefix after prefix, and each parse used to classify every
  tool again.

  The pipelines keep their own caps for callers that skip the measure.
  blallama takes each as a flag (`--schema-max-tools`,
  `--schema-max-params`, `--schema-max-nodes`, `--schema-max-defs`,
  `--schema-max-member-bytes`, `--schema-max-total-member-bytes`,
  `--schema-max-width`, `--schema-max-depth`).

  The depth keeps every constrained value readable. serde_json — and
  every dialect parser with it — reads at most 127 levels of nesting,
  and a chain of `$ref`s, one `required` object per def, spells a
  value of any depth in a request a few levels deep: at 127 levels the
  schema compiled in every dialect to a grammar whose every value no
  parser could read back, so each draw failed and the request ended in
  resampling and a 500. It is counted from the shape: an object or
  array one level more than its deepest property or `items`, an
  `anyOf`/`oneOf` variant at its schema's level, an `enum` member or
  `const` at its own depth as JSON, a `$ref` at its target's depth, one
  back into its own cycle at none. With the 32 levels an untyped value
  may add and the call envelope's two, a value inside the default
  stays under 127.

- **Footprint guards for the schema pipelines and the matcher**
  (`footprint_guard_pipelines`, `footprint_guard_matcher`, both
  `#[ignore]`d): the hostile rechecks' probes, committed. Requests at
  the default schema limits and past them go through the measure,
  every dialect's compile and grammar parse, a 512-parameter Qwen
  call's parse, the schema check and the matcher over ~128 KB (~32k
  tokens) of output; and four adversarial byte streams of up to 128 KB
  (ambiguous recursion, nested brackets, an array of an `enum` at the
  width limit, 2,000 optional integers) are fed through the matcher
  with a filter step over a ~120k-piece synthetic vocabulary every 256
  bytes. Each step must take under 2 s and the process stay under
  1.2 GB resident (`ps`, no `unsafe`). They measure the whole process,
  so they run alone: under nextest in the nightly/GPU window, `cargo
  nextest run --run-ignored only -E 'test(footprint_guard)'` (or as
  part of `just test ignored`); CPU only, no model. Measured
  (2026-10-01, M-series, test profile): at most 52 ms a step but one
  (a 432 ms filter step 2,560 brackets deep, where the state is past
  the DFA cache's threshold and runs uncached), 434 MB peak.

- **Constrained output is checked against its schema before it is
  answered** (`SessionError::SchemaViolation`). A finished
  json_schema `output_config` answer must be exactly one JSON document
  across the turn's text, matching the schema; a `strict` tool call's
  input must match its tool's. Checked are the keywords the grammar
  compiler enforces (`type`, `properties`, `required`,
  `additionalProperties`, `enum`, `const`, `anyOf`, `items`, `$ref`,
  non-empty `minItems`) — never the validator-only ones it deliberately
  leaves to the model (`pattern`, `minLength`, `maximum`, …), which
  would turn every such request into a resample loop. A turn cut by
  `max_tokens` or a stop sequence is exempt (#121). The error leaves the
  cache warm, `Display` names the schema location but never the value,
  and blallama resamples it on the warm cache like a grammar violation,
  then answers 500 `api_error` — never a 200 carrying the invalid
  value. `SchemaMismatch` / `MismatchKind` are public.

  The check's work is bounded: a step per subschema judged and per
  `enum` member, distinct `required` name or distinct `type` compared,
  plus a step per entry of a schema's `required` and `type` arrays the
  once their names are gathered, at most 2^20 plus 2^14 per JSON value
  of the output. Inside the schema limits every
  way a value can be judged — `anyOf` variants tried in turn and nested
  to multiply, object variants that all declare the property, `enum`
  members, each alternation itself — is counted by the width (at most
  2,048), and each def is judged at most twice per value (memoized
  inside an `anyOf` and out), so a request inside them stays far under
  the budget: each worst shape filled to the width limit and checked
  over 2,000 values takes at most ~2,050 steps a value, under 75 ms.
  Past the budget (a schema past the limits, from a caller that skips
  the measure) the check stops judging, logs a `schema_check_budget`
  warning and passes the value, which the grammar already constrained,
  rather than turning valid output into a resample loop and a 500. A
  failing `anyOf` variant no longer copies the path or builds its
  message, and an object's undeclared keys look `required` up in a
  set.
- **`BlockStream::violation`**: once drained, a stream reports the
  `GrammarViolation` or `SchemaViolation` the batch path would have
  returned for the same turn, by the same rules (one `TurnContract`
  for both). The blocks are already out, so discarding them is the
  caller's call. The batch path's special-token containment
  (`EmittedSpecialToken`) is not checked there.

- **Automatic prompt caching, as on Anthropic.** A request-level
  `cache_control` (`Prompt::cache_control`, misanthropic's
  `auto_cache`) now places a breakpoint after the last cacheable block,
  walking back past thoughts, with its own TTL. An anchor an earlier
  call placed is read again when the new prompt reproduces everything
  before it (lookback), so each request reads back the previous one's
  prompt: a turn that does not round-trip costs only itself instead of
  everything back to the system marker. The lookback is not a copy of
  Anthropic's: it reads anchors Anthropic's would not, and misses some
  Anthropic's would hit. Anthropic reads an earlier request's entry
  only within 20 blocks of one of the new request's markers, while this
  reads the anchors the slot's last request placed at any distance; but
  Anthropic can also hit an older request's entry within those 20
  blocks, which the slot no longer keeps. A candidate whose snapshot
  is gone falls to the next anchor below it rather than to zero, and so
  does a hit covering the whole prompt, which used to consider only the
  new call's own lower markers. blallama answers Anthropic's 400s for
  the markers word for word, checked in Anthropic's order (captured
  2026-09-30 on claude-haiku-4-5): a fifth marker counting the
  automatic one — even on an already-marked block with the same TTL,
  which the docs call a no-op — answered even when the markers are
  also out of order; then a 1-hour marker after a 5-minute one, named
  by its path (`messages.0.content.1.cache_control.ttl: …`); then an
  automatic TTL that disagrees with the target block's marker; then a
  1-hour automatic marker after a 5-minute one. `check_cache_controls`
  and `MAX_CACHE_CONTROLS` are public.
- **The llama.cpp snapshot store holds a full set of anchors per
  sequence.** Its cap was a flat 16 shared by every cache slot; with
  automatic caching a hybrid model's slot holds up to six snapshots
  (four markers and two tips), so `--cache-slots 4` on Qwen3.6 sat at
  the cap and one slot's newest snapshot could evict another's system
  anchor. The cap is now six per sequence (`n_seq_max`), never below
  16. The cost is host RAM — a hybrid snapshot carries the sequence's
  attention KV too — and each eviction logs the dropped snapshot's
  size.
- **Every cache reuse decision is logged.** One `cache_reuse` event
  per call (`hit` with `source` = `tip` / `breakpoint` / `lookback` /
  `hash` and token counts, or `miss` with its `reason`), plus a
  `cache_degrade` or `cache_evict` event for everything that cost
  reuse: a tip the call continues past but cannot use (the first
  diverging entry and the decoded text on both sides), a turn whose
  emission does not re-render byte-for-byte, a breakpoint dropped for
  not being a token prefix, a #91 hash refusal, a failed restore, a
  snapshot evicted at the store's cap, and TTL, capacity, slot-thrash
  and error evictions. Hits log at `DEBUG`; misses and losses over 256
  tokens at `WARN`, the rest at `INFO` — a history change on the slot
  that shared an anchor with the request included, since that is
  plausibly the same conversation edited. Filter on the targets
  `drama_llama::session` and `drama_llama::snapshot_store`. The
  2026-09-30 Qwen3.6 run lost a 7364-token turn's tip with nothing
  but an ordinary stats line to show for it.
- **`output_config.effort` reaches the chat template as
  `reasoning_effort`.** It used to be dropped, so Qwen3.8 always
  rendered its `xhigh` default ("think carefully… validate key
  assumptions…") and thoughts ran 4–8k tokens. The analyzer now
  measures the levels a template accepts (`ReasoningSyntax::efforts`):
  each of `low`/`medium`/`high`/`xhigh`/`max` is probed thinking-on
  and accepted iff it renders; a template that also renders a nonsense
  value validates nothing and gets the trained `low`/`medium`/`high`
  (gpt-oss); one whose output never changes has no knob (the Mistral
  cache-stable template, which derives it from `enable_thinking`).
  Measured: Qwen3.8 `low`–`xhigh`, stock Mistral Small 4 `high`,
  gpt-oss `low`–`high`. For a thinking-enabled prompt the render maps
  the requested level onto that set — exact if accepted, else the
  nearest, the lower on a tie (Qwen3.8 `Max` → `xhigh`) — through the
  new `RenderOptions::efforts`, which `Session` fills from the dialect
  and `with_render_opts` forces like the re-ingest convention. Thinking
  off never sets it; a caller's `reasoning_effort` extra still wins.
  The effort is written into the prompt *prefix*, so `render_partial`
  now carries it into every truncated prompt — without it no partial
  would be a prefix of the full render and every cache breakpoint
  would be lost (the #93 `thinking` bug, again). Remaps and "no knob"
  are logged once at debug level.
- **`GET /v1/models` and `GET /v1/models/{id}` on blallama**, in
  misanthropic's `Models` / `ModelInfo` types, for every model on
  disk — loaded or not — with real metadata: `display_name` from
  GGUF `general.name` (new `Model::title`), token ceilings
  `min(served n_ctx, n_ctx_train)` (what llama.cpp reports as
  `n_ctx`), `created_at` from the file's mtime, and the capabilities
  a session can honor (`structured_outputs` always; `image_input` iff
  an mmproj sidecar is present; `thinking` iff the dialect has a
  reasoning syntax). `/api/tags` is derived from the same entries
  instead of hand-built with blank fields. Behind it: `Catalog<B>`
  (a directory plus a per-process metadata cache, keyed by file
  mtime+size, misses serialized through a peek lane so cached reads
  never wait), `FromPath::peek` (a `vocab_only` load — no weights,
  no GPU — walking the same template ladder as a load), and
  `Session::model_info`, all through one `Advertised → ModelInfo`
  mapping so a listing and a load never disagree (pinned by
  `peek_agrees_with_load`). `SessionTransport::models` now advertises
  the same rich entry instead of a basename stub. blallama warms the
  cache at startup off the request path, stage-logged
  (`read_model_metadata` / `model_metadata_read`).
- **#96 regression suite, per model.** A shared scenario
  (`tests/common/tip.rs`) runs the downstream agent shape — sliding
  marker window, forced tool-call turns, honest tool results — plus
  the issue's probe (a continuation adding no new `cache_control`
  anywhere), asserting each call resumes past the *entire* previous
  prompt: a read that far can only come from the tip. Wired into the
  Qwen (`session_cache`), Gemma 4, gpt-oss, and Mistral Small 4
  suites; green on all four. No golden text — CI runs smaller family
  members (`session_gptoss` gained `DRAMA_LLAMA_GPTOSS_MODEL` for
  that).

## [0.8.3] — 2026-07-25

### Added

- **`llama_cpp::gpu_device_names`** — the GPU-class compute devices ggml
  actually discovered, as opposed to the ones the build asked for. Those
  differ in a way nothing else reported: a `feature = "cuda"` build links
  CUDA and then silently runs every model on the CPU when the driver is
  unusable, and from inside the process that is indistinguishable from a
  box with no card.

- **A test asserting a `cuda` build found a GPU.** The model tier stays
  **green** through CPU fallback — correctness doesn't depend on the
  device — and timing doesn't give it away either, because only the
  generation-heavy tests slow down while load-dominated ones get *faster*
  (no VRAM upload, weights already in page cache). The suite's wall clock
  can therefore move in either direction, which makes an explicit device
  assertion the only reliable signal.

  Caught for real on 2026-07-25: an unattended upgrade of
  `nvidia-driver-580-server` (580.159.03 → 580.173.02) left the old kernel
  module loaded against new userspace libraries. `nvidia-smi` failed
  outright, `cuInit` returned 804 (`CUDA_ERROR_SYSTEM_DRIVER_MISMATCH`),
  ggml found zero CUDA devices, and the entire model tier passed on the
  CPU with nothing in CI saying so. The new test fails in 0.04 s and names
  the diagnosis.

  Set `DRAMA_LLAMA_ALLOW_CPU_FALLBACK=1` to exercise the CPU path
  deliberately — a legitimate thing to want; silently getting it is not.

  The test's `cuda` gate is on its **body**, not on `#[test]`, so it is
  listed in every configuration and `nextest list` returns the same count
  everywhere. Gating the attribute makes README.md's hand-counted numbers
  mean one thing under `llama-cpp` (605) and another under
  `llama-cpp-cpu` (604) — which is how this PR first went red. House rule
  now, and the reason `just check` can verify the badge against whichever
  target dir is warm instead of building the cpu one.

- **A `nvidia-smi` preflight on `test.py run`** for `cuda` configurations
  on Linux. Reaches the same verdict as the test above roughly 40 minutes
  earlier — before a cold cuda build rather than after it. It is not a
  replacement: a working `nvidia-smi` says nothing about a llama.cpp that
  cmake quietly configured without CUDA, so ggml's own device list stays
  the authority. Honours `DRAMA_LLAMA_ALLOW_CPU_FALLBACK=1`.

### Fixed

- **The pre-commit hook actually gates the README badge now.** 6d4a9e2
  put that gate in `scripts/test.py check` and claimed the hook was
  covered; it was not. The hook runs `just check`, whereas that function
  is `just permutations` — which the hook deliberately skips as too slow
  — so the gate never ran once, and a stale badge reached CI in #82. It
  lives in the `just check` recipe now.

- **CI's `gpu` step no longer swallows a failing `nvidia-smi`.** It pipes
  into `tee` under the default `bash -e`, which takes its status from
  `tee`, so the exact driver-mismatch state the step exists to record
  reported green. It sets `pipefail` now.

## [0.8.2] — 2026-07-24

### Fixed

- **A truncation sampler can no longer starve the deferred (lazy)
  tool-call grammar** ([#76]). An eager grammar — the forced
  `tool_choice` path — is prepended into `SamplerConfig::modes` by the
  session so it masks the full vocab *before* `LocallyTypical` / `TopP`
  / `TopK` narrow it. The deferred grammar, which is not in `modes`,
  was applied after the entire fold instead, so it only ever saw what
  truncation left behind — frequently a single token. When that lone
  survivor was illegal, `grammar_filter` force-EOS'd and generation
  ended mid-structure, surfacing as
  `SessionError::GrammarViolation`, even though thousands of legal
  tokens existed in the full candidate set.

  Because only `tool_choice: auto` builds a deferred grammar, this
  presented as *intermittent* tool-call failures on local models under
  auto while forced tool choice was unaffected, and it was independent
  of `max_tokens` (reproduced identically at 2048, 4096, 8192, and
  16384). `apply_modes` now runs the deferred grammar first, at the
  same point an eager grammar runs. Reproduced end-to-end on
  gpt-oss-20b via a multi-round agentic tool session, where it failed
  on the first acting round every time and passes 8/8 rounds after.

## [0.8.1] — 2026-07-24

### Fixed

- **Tip resume into a new assistant turn no longer forces an
  immediate EOS** (the 0-output-token second-round bug). A cached
  tip's `SamplerState` carries constraint-matcher positions on
  grammar identity — correct for assistant-prefill / partial-
  completion resume, but wrong when the new call seats the generated
  turn plus a tool result: the matcher arrived parked at
  tool-call-complete, where the only legal continuation is the turn
  terminator, so every second acting round of an agentic session
  generated zero tokens (reproduced on both Qwen 3.6 and gpt-oss —
  template-independent). `Session::build_initial_state` now resets
  matchers (eager, JSON, and deferred) to their grammar roots when
  the fold has messages past the resume cursor
  (`matcher_carry_valid`); the prose stream (rng, mirostat `mu`,
  n-gram stats) still carries. `hash_cache_smoke` — which exercised
  the exact failing shape but asserted only cache-read counters —
  now also asserts round 2 produces output: cache stats are not a
  proxy for generation.

## [0.8.0] — 2026-07-24

Backend split. The chat-style API (`Session`), the engine layer
(`Engine`), the Predictor family, and the binary (`blallama`) are
all generic over a single `Backend` parameter. drama_llama can now
drive either llama.cpp or moeflux's Metal MoE runtime through the
same surface. Runs Cogito-class MoE models on Apple Silicon without
the Anthropic API as a dependency.

Three further arcs land on top of the split:

- **Image input** via llama.cpp's mtmd ([#31]). A backend-agnostic
  `Vision<D>` trait, a safe `Mtmd` wrapper, and a cache-aware
  `Session` media path let vision models take `Block::Image` input.
  Images are rendered out-of-band through a per-call random sentinel
  — mtmd never sees prompt text — and the prefix cache accounts in
  M-RoPE cell space so an image mid-prompt doesn't invalidate the
  KV walk. Gated on `feature = "mtmd"` (or pure-Rust `feature =
  "media"`).
- **Per-model tool-call dialects** ([#30], absorbing [#29]). A
  `CallSyntax` derived by differentially analyzing each model's chat
  template drives both the GBNF grammar emitter and the response
  parser, so `Session` speaks a model's *native* tool-call format
  instead of one imposed shape. Qwen3.5/3.6 (XML-ish), Gemma 4
  (`TagWithDict`, causal announce-then-call), and gpt-oss (Harmony
  channels) ship as validated dialects. Round-trip byte-stability is
  the cache invariant.
- **Lazy grammar checking** ([#28]). Grammar-constrained sampling
  now samples first and checks the one sampled token
  (`GrammarState::accepts_bytes`, O(piece)), falling back to a full
  O(vocab) mask only on rejection — instead of masking the whole
  vocabulary every step.

### Added

- **The README is now the crate-level documentation** (#66), mounted
  with `#![doc = include_str!("../README.md")]`, so the front page of
  the repository and the front page of docs.rs cannot drift apart. It
  was rewritten from the v0.3-era text it still carried, which
  advertised a `bin/dittomancer` deleted in `57016dd` and a hardcoded
  n-gram blocklist removed in `c2d8579`. Its two `rust` blocks are
  doctests: a `no_run` structured-output walkthrough that is
  type-checked on every CI run, and an executed one driving the GBNF
  matcher standalone.
- **Coverage** (#70) — `scripts/test.py coverage`, `just coverage`,
  and a CI job uploading to Codecov. Runs the tests under
  cargo-llvm-cov once and re-reads that data per output format
  (`--lcov`, `--json`, `--html`, and a per-file table). Defaults to
  `-t all` because the fast tier alone reports every generation path
  as dead code. Read the headline percentage as an upper bound:
  llvm-cov filters by file, so `#[cfg(test)] mod tests` inside `src/`
  cannot be excluded from it.
- **`scripts/test.py doctest`** / `just doctest`, wired into `just
  check` and its own CI job. cargo-nextest has no doctest support
  (nextest-rs/nextest#16), so until now every recipe in the repo ran
  exactly zero of them — which stopped being acceptable once the
  README became a doctest. The pre-commit hook and `ci.yml` both
  skipped `.md`-only changes; both now treat `README.md` and
  `TERMS_OF_USE.md` as source, since `src/lib.rs` `include_str!`s
  them.
- **`FromPath::Options`** — an associated type carrying whatever a
  backend needs to be told at load time, so generic code can ask for
  a context size. `LlamaCppOptions` (`n_ctx`, `cache_slots`,
  `flash_attn`, `no_gpu`, `numa`) and `MoefluxOptions` (`use_2bit`);
  both `serde`-serializable, and `clap::Args`-flattenable under
  `feature = "cli"` for single-backend binaries. Every field unset
  means the backend's own default, so `from_path` is unchanged
  behaviour. `FromPath` gained `from_path_with` (the constructor),
  keeps `from_path` (default options) and `from_path_async` (the
  same on tokio's blocking pool) as provided methods, and is no
  longer `tokio`-gated.
- **`cli::BackendArgs`** — the union of every compiled-in backend's
  load knobs plus the `--backend` selector, for binaries that pick
  their backend at run time (clap flattens at compile time, so such
  a binary cannot flatten `B::Options` directly). Narrows to a
  concrete backend's options with `TryFrom`, returning
  `cli::UnsupportedOptions` when a flag names something that backend
  has no notion of. Also `cli::DEFAULT_N_CTX` (32768), the value
  this repo's own front-ends default to.
- **`Backend::set_log_callback` / `Backend::clear_log_callback`** —
  route a backend's native log stream wherever the application wants,
  returning `Result<(), NotImplemented>` with a default body that
  errors. Lets an application install a sink *before* loading a
  model, which is when llama.cpp is loudest and the only point at
  which the noise can be caught. `LogLevel` moved to
  `crate::backend` (its `Other` variant now carries `u32`) so it
  compiles with no backend feature enabled.
- **`blallama --n-ctx` / `--cache-slots` / `--use-2bit`**, via the
  flattened `BackendArgs`.

- **`SamplingMode::Deny { range: Range<Token> }`** — sample-time
  mask for forbidden token-id ranges. Constructor:
  `SamplingMode::deny_range(r)`. Filters candidates whose id falls
  in the range out of the set before any downstream mode runs;
  falls back to a single EOS if the range eats every candidate.
  Primary use case: tokenizer reserved/unused vocab tails (Qwen3:
  ~248088..248320). `Session` automatically prepends a Deny mode
  computed once at construction by scanning from the highest vocab
  id downward — empty-piece tokens trivially pass byte-stream
  grammar filters and would otherwise let the model land in a loop
  scattering reserved tokens after a structured response closes.
  See `.claude/memory/grammar_reserved_token_loop.md` for the
  full analysis.
- **`Model::eog_tokens()`** — trait method exposing the model's
  complete end-of-generation set, and the single authority for what
  stops a prediction. `LlamaCppModel` returns libllama's
  `special_eog_ids` verbatim (`llama_vocab_is_eog`); `MoefluxModel`
  composes it from the `eos_token_id` config array (Qwen3 declares
  `[<|im_end|>, <|endoftext|>]`, and the secondary decodes to an
  empty piece, so missing it means an invisible loop to
  `max_tokens`). Never derive a stop set from `eos()`/`eot()` — see
  *Changed* and *Fixed*.
- **`Model::display_name(&self) -> Option<String>`** — human-readable
  identifier on the trait. `LlamaCppModel` returns the GGUF basename;
  `MoefluxModel` returns the parent dir's basename (overridden
  by `MoefluxEngine::from_path` to match the discovery-dir name).
  Used by `Session::complete_response` for the `model` field of
  responses, and by `blallama` for model-name matching.
- **`backend::Backend` trait** bundling `type Decoder: Decoder + Send`
  and `type Model: Model + Send + Sync` as a single generic
  parameter. Compile-time monomorphization, no `dyn` indirection on
  the hot path. ZST tag impls: `LlamaCppBackend`, `MoefluxBackend`.
- **`MoefluxEngine::from_path(parent: &Path)`** — convention-based
  wrapper around `from_paths`. Expects `parent/{mlx,artifacts,root}/`
  with sane runtime defaults (`experts_per_tok = 8`, `use_2bit =
  false`). Symmetric with `LlamaCppEngine::from_path` so binaries
  can take a single `--model <path>` arg for either backend. The
  5-arg `from_paths` stays for callers needing explicit paths or
  non-default runtime params.
- **`blallama --backend {llama-cpp|moeflux}`** flag with cfg-gated
  variants. `main()` dispatches once at startup; each backend half
  monomorphizes independently. llama-cpp build accepts only
  `llama-cpp`; moeflux build accepts only `moeflux`; combined build
  accepts both.
- **`drama_llama::sidecar` module** (gated on `feature = "toml"`):
  per-model sampling-config TOML files colocated with the model on
  disk. `Session::from_path*` looks for the sidecar, applies it via
  `with_sample_options`, and writes a default if none exists so
  there's a starting point to edit.
  - **GGUF (llama-cpp)**: sibling `<model>.sampling.toml` next to
    the `.gguf`.
  - **Moeflux**: `parent/sampling.toml` alongside the
    `mlx`/`artifacts`/`root` symlinks.
  - Reset = `rm <sidecar>`; tweak = edit it.
  - `Json`/`Grammar`/`Deny` modes are excluded from sidecars on
    purpose — those are runtime per-request constraints, not
    per-model defaults.
- **`Session::with_sample_options(SamplerConfig)`** — wholesale
  setter used by sidecar loading. Sets the post-grammar sampling
  chain, repetition penalty, and any deferred grammar in one shot.
  Auto-extends `repetition.ignored` with the model's special
  tokens (matches `with_repetition` semantics) so a strong rep
  penalty can never lock out EOS / chat-template markers.
- **`Session::with_seed(Option<NonZeroU128>)`** — fixed RNG seed
  forwarded to every `predict_*` call. Makes tuning iteration
  meaningful: same prompt + same seed = same output, so a
  sidecar tweak shows up as a deliberate change rather than
  stochastic noise.
- **`RepetitionOptions::window_size: NonZeroU32`** (default 256)
  and **`RepetitionOptions::decay: f32`** (default 0.95).
  Together they bound the repetition-penalty additive contribution.
  Effective per-n-gram count is now
  `Σ decay^(current_step - position)` over occurrences inside the
  last `window_size` generation steps; bounded above by
  `1 / (1 - decay)`. Pre-fix the additive `count * penalty_freq`
  term grew linearly with generation length and dominated the
  model's natural logit gradient on long generations (~20 logits
  below baseline at 200 steps, ~60 at 600). With the fix the gap
  saturates once the window fills and stays put. See
  `.claude/memory/qwen3_long_form_degradation.md` for the analysis.
- **`NGramStats::evict_outside_window(current_step, window_size)`**
  and **`NGramData::windowed_decayed_count(current_step, decay)`** —
  the primitives backing the windowed-decay penalty path. Maintains
  the `count == positions.len()` invariant on each entry.
- **`blallama --no-penalty`** — force repetition penalty OFF, even
  when the per-model sidecar enables it. For probes, canary runs,
  and any "what does this model do with no penalty" diagnostic.
- **`blallama --seed <u128>`** — fixed RNG seed forwarded to every
  prediction. For tuning iteration where you want sidecar changes
  to show up as deliberate divergences rather than stochastic ones.
- **Probe mode** — `Engine::set_probe_hook` installs a per-token
  `ProbeHook` observer on the prediction loop, so a consumer (canary
  suite, batch evaluation, Weave) can watch each yielded token
  without forking the predictor iterator. `ProbeCtx` carries the
  sampled token, its position, and the effective sampler
  configuration; hooks self-declare their snapshot appetite via
  `ProbeHook::snapshot_opts` — the default `None` skips the per-token
  softmax/sort entirely, while `Some(SnapshotOpts)` populates a rich
  `Candidates` `Snapshot` (`Candidates::capture_snapshot`:
  pre-everything top-k probabilities and ranks, optional entropy)
  for cross-validating external behavior against internal
  disposition. `blallama` exposes both recorders: `--record-json
  <FILE>` appends one JSONL record per token, and `--probe-stream`
  mounts a `/probe` SSE endpoint streaming per-request
  `session_start` / `token` / `session_end` events.
- **`SessionTransport` / `LocalTransport`** ([#48]) — a
  `misanthropic::Transport` over a locally-owned `Session`, so
  anything generic over a transport (the upstream `Chat` driver
  included) drives local inference exactly as it would the API
  client. Completions run on tokio's blocking pool behind an async
  mutex, one at a time — cloning the transport clones the *handle*,
  and every clone serializes through the same session and KV cache,
  the honest shape of one local model on one GPU. `LocalTransport`
  is the erasure: `Arc<dyn LocalTransport>` serves both `Prompt` and
  `CachedPrompt` and carries `scan_text_for_specials` through the
  erased type, which is how the examples went backend-generic (see
  `.claude/memory/examples_erase_at_transport.md` for why the type
  is the transport, not a boxed `Session`).
- **Multi-slot prefix cache.** The v0.7 single-sequence prompt cache
  grew slots: up to `PrefixCacheConfig::max_slots` cached prefixes
  (clamped to the engine's sequence count), so a council of agents
  each keep a warm prefix instead of evicting one another. Enable
  with `Session::with_prefix_cache(true)`, tune with
  `Session::with_prefix_cache_config`. Slots evict
  least-recently-used against a cell budget
  (`PrefixCacheConfig::capacity_cells`, default the engine's
  `n_ctx`), and breakpoints honor Anthropic-style TTLs (5m default,
  1h opt-in) with refresh-on-read semantics: an idle breakpoint past
  its TTL loses its snapshot, and a fully-expired slot is evicted.
- **`Model::recommended_sampling()`**, with **`SamplingParams` /
  `Mirostat`** and **`apply_request_sampling`** — each backend
  reports the sampling settings the model recommends for itself
  (GGUF `general.sampling.*` for llama.cpp; returning empty is
  expected, not exceptional — gpt-oss carries no such keys). Seeds a
  fresh sampling sidecar, and serves as the fallback tier for
  per-request sampling: `apply_request_sampling` patches a single
  unambiguous knob into the existing chain, otherwise rebuilds a
  canonical `TopK → TopP → MinP → Temperature` chain from the
  request layered over the model's recommendation — keeping
  constraint modes as a prefix and never touching repetition or
  `banned_specials`, which are emission-side protocol integrity a
  remote client must not be able to switch off.
- **`ToolChoice::None` is enforced** ([#44]) — "the model must not
  use any tool" now bans the dialect's tool-call opener for that
  call alone (the standing emit-ban exempts the opener so
  Auto/Any/Method can call), leaving the tool definitions rendered
  and the cached prefix intact. Use cases: prefix-preserving
  interviews, and forcing prose out of a tool loop. The `chat`
  example grew `--tool-choice <auto|any|none|method:NAME>`.

#### Image input — mtmd ([#31])

- **`feature = "media"` and `feature = "mtmd"`** — two-tier gating.
  `media` is the pure-Rust image layer (decode via the `image` crate
  — never mtmd's bundled `stb_image`, a deliberate CVE-posture
  choice — plus the conversions into the frozen `Image` pixel
  record); it compiles without `llama-cpp`, so moeflux-only builds
  get typed "media unsupported" errors from `NoVision` rather than
  silent drops. `mtmd` adds the llama.cpp multimodal backend on top
  (libmtmd bindings + the safe `Mtmd` wrapper).
- **`backend::Vision<D: Decoder>` trait** — the backend-agnostic
  image-input capability, generic over the decoder. Placeholder-
  typed by design: `tokenize_image` takes an `ImageInfo` (dims +
  identity hash, no pixels), while `prefill_image` requires a full
  `Image` and the decoder — encoding a placeholder is unrepresentable
  in the type. `NoVision` is the uninhabited impl for backends
  without vision, so generic `Session` code compiles for every
  backend.
- **`backend::Image` / `ImageInfo` / `MediaSpan` / `MediaChunk`** —
  the frozen pixel record (`Image::from_rgb8`, sha256 identity via
  `Image::id`) and the placeholder / span types the `Vision` trait
  and the media-aware cache traffic in.
- **`llama_cpp::Mtmd`** (`feature = "mtmd"`) — safe wrapper owning
  the `mtmd_context`, implementing `Vision<LlamaCppDecoder>`.
  `MtmdParams` is the small stable construction subset; typed error
  ladder (`MtmdNewError` / `MtmdTokenizeError` / `MtmdPrefillError`
  / `MtmdError`), all `Send + Sync`. The media eval loop is Rust-
  owned (`EmbdBatch`, a pre-KV `NaN` guard, explicit M-RoPE position
  planes) rather than delegated to mtmd's C helper.
- **`Session` media path** — `Block::Image` is accepted at ingest,
  decoded through `media`, and rendered out-of-band via a per-call
  random sentinel so mtmd never tokenizes prompt text (injection-
  proof by construction). The prefix cache is media-aware end to
  end: `CacheEntry` (token vs. media sentinel), `EntryPos`
  (entry↔position translation), and cell-space accounting so an
  image's M-RoPE cell span participates in the longest-common-prefix
  KV walk instead of forcing a full reprefill.

#### Per-model tool-call dialects ([#30], absorbs [#29])

- **`drama_llama::dialect` module** — template-derived tool-call
  formats. `CallSyntax` (with `ReasoningSyntax` / `ContentSyntax` /
  `FunctionSyntax` / `ArgumentsSyntax` / `CallIdSyntax` /
  `JsonFields` and the `Family` / `ReasoningMode` / `ContentMode`
  axes) is the single description that drives both emission and
  parsing.
- **`analyze_template` / `vocab_cross_check`** — the differential
  analyzer that derives a `CallSyntax` from a model's chat template
  (probe-first, llama.cpp-validated), cross-checked against the
  model's vocab so emitted markers are real tokens.
- **`grammar_source` / `render_reference` / `validate_representable`
  / `EmitOptions` / `Anchor`** — the GBNF emitter half: a
  `CallSyntax` compiles to a grammar that constrains generation to
  the model's native call shape.
- **`parse_text` / `StreamParser` / `ParseStatus` / `Parsed` /
  `Leniency`** — the parser half: the model's emitted envelope is
  re-ingested back into typed tool-call blocks, byte-stable with
  what the grammar emitted (the cache invariant).
- **`emit_until_rules`** (in `grammar_compile`) — GBNF encoding of
  llama.cpp's `until()` combinator (KMP-DFA complement), the
  grammar-engine primitive dialects need to consume "everything up
  to the closing tag." Exhaustively differential-tested against a
  naive matcher.
- **`Session::with_dialect` / `Session::dialect`** — a dialect is
  analyzed once at load and thereafter drives grammar construction
  and response parsing. Shipped dialects: Qwen3.5/3.6 (XML-ish,
  native format from [#29]), Gemma 4 (`TagWithDict`, causal
  announce-then-call render), gpt-oss (Harmony channel format; see
  the `dialect::harmony` submodule).
- **Chat-template sidecar** — an optional `<model>.chat_template.jinja`
  sibling overrides the GGUF-embedded template (used to ship the
  cache-stable Gemma 4 / gpt-oss templates without patching the
  model file).

#### Lazy grammar checking ([#28])

- **Sample-then-check grammar constraint** — `SamplingMode::Grammar`
  / `Json` now sample a token from the *unmasked* distribution and
  validate just that token with `GrammarState::accepts_bytes`
  (O(piece)); only on rejection does the full O(vocab) mask-and-
  resample path run. Common case drops from per-step vocab masking
  to a single byte-run check.
- **`SamplerConfig::banned_specials: Vec<Token>`** — emit-side
  special-token mask applied before sampling, so a dialect's illegal
  control tokens never reach the candidate set (opt-out via
  `Session::with_emit_specials_ban(false)` for e.g. Qwen-VL
  grounding markers). Falls back to a resample when the ban would
  empty the set.
- **Exit interviews for agents** — the `council` example grew
  `--dump [DIR]`, archiving each seat's complete prompt to
  `<DIR>/<seat>.json` on adjournment, and the `chat_repl` example
  became `chat`: `--load` reseats a dumped prompt so you can
  interview the agent about the run. Loaded tools are kept verbatim
  (their schemas are debug context); every tool call is printed but
  only bash executes (`--add-bash`, Docker-sandboxed `RichBash`
  driven without a `ToolBox`); everything else is answered with a
  stub receipt so the transcript stays wire-legal. `--clear-tools`
  strips tools for a prose-only interview.

### Changed

- **Tool arguments render in schema declaration order** ([#60]).
  `serde_json/preserve_order` and `minijinja/preserve_order` are now
  enabled unconditionally, so JSON maps keep insertion order
  end-to-end: schemars-derived `properties` arrive in field
  declaration order, the GBNF grammar emits arguments in that order
  (letting a model condition later arguments on earlier,
  reasoning-ish ones), the parser preserves it, and minijinja's
  `tojson` re-renders it unchanged — matching llama.cpp. Required
  vs. optional is now classified by *membership* in `required:`,
  never by that array's order, and optionals sit in place in the
  grammar. The dict family (Gemma 4) is the deliberate exception:
  its model-shipped templates pipe arguments through `| dictsort`,
  so that family stays explicitly alphabetical. Behavior-breaking
  for caches: the canonical bytes of rendered tool definitions and
  tool calls change (previously alphabetized), so warm prefix caches
  from earlier 0.8.0 dev builds will not match. Enabling the
  features ourselves also closes a feature-unification hazard where
  a downstream crate enabling `serde_json/preserve_order` would have
  silently broken the old sorted-order assumptions. Note for tools
  served to both a local backend and the Anthropic API: Anthropic's
  structured outputs reorder required properties first, so declare
  required fields before optional ones if identical ordering across
  engines matters.

- **Pre-publish API hardening** (pre-crates.io review). Every public
  error enum, the growth-prone enums (`LogLevel`, `MediaChunk`,
  `Family` — whose docs already promise an `Instructed` variant —
  `ReasoningMode`, `ContentMode`, `CallIdPosition`,
  `ReasoningReingest`), and the options/config structs
  (`LlamaCppOptions`, `MoefluxOptions`, `BackendArgs`,
  `SamplerConfig`, `SamplingParams`, `PrefixCacheConfig`,
  `EmitOptions`, `CallSyntax` and its field-structs) are
  `#[non_exhaustive]`: adding a variant or field later must not be a
  breaking change. Construction is `Default` + pub-field mutation or
  the `with_*` builders (`BackendArgs` gained the `Default` its clap
  attributes already advertised; `MoefluxOptions::use_2bit()` is now
  `with_use_2bit(bool)`, matching the family). The `moeflux` module
  itself went private with curated re-exports, mirroring
  `llama_cpp`: `MoefluxDecoder::ctx()`/`ctx_mut()` leaked
  `moeflux::Ctx` — a third-party type pinned at a pre-release — into
  the public API, and the cmdbuf/`eos_raw` diagnostics were public.
  `MoefluxEngineError` and `PrefetchStats` joined the crate-root
  exports because public signatures name them.
- **`SampleOptions` split into `SamplerConfig` + `SamplerState`.**
  The old type conflated immutable configuration with live per-call
  run-state — grammar matcher positions behind `Arc<Mutex<_>>`, ~80
  lines of custom serde hooks (inconsistent between the `Json` and
  `Grammar` variants), and a ~113-line mutex-locking `PartialEq`.
  `SamplerConfig` is now a pure value: derive-serializable,
  `PartialEq`, no interior mutability, and `banned_specials` became a
  plain `Vec<Token>` in the same move. `SamplerState` gathers
  everything a run accumulates: matcher positions, mirostat `mu`, the
  n-gram stats, and the working RNG — now `rand_pcg::Pcg64Mcg`, whose
  single-`u128` state can actually be serialized (the previous
  `xorshift::Xoroshiro128` could not expose its state, which is the
  point).
  The effective config is the sole constructor of a fresh state
  (`SamplerConfig::init_state`), and `Engine::predict_tokens` /
  `predict_pieces` / `predict` (plus their `_resuming` variants) take
  a new `initial_state: Option<SamplerState>` parameter to resume a
  caller-owned state. `Session` caches the state at breakpoints
  alongside the KV snapshot, and the per-call seed default flipped to
  `None`, making the seed encode resume/fork/fresh: no seed + cache
  hit resumes the exact stream, no seed + miss draws a fresh random
  seed, an explicit seed forks deterministically. Serialize → restore
  → continue is bit-exact — `NGramStats` moved to `BTreeMap` so
  iteration order (and thus float accumulation) survives a round
  trip. See `.claude/memory/design_sampler_config_state_split.md`
  for the full design.
- **`tracing` is now a non-optional dependency** (with its `log`
  feature), and the crate's own diagnostics — sidecar read/write
  failures, chat-template dialect analysis failures — emit
  `tracing::warn!` instead of writing to stderr unconditionally.
  `RUST_LOG` now governs them on every backend.
- **`Session::from_path_sync` is now `FromPath::from_path`** (the
  trait must be in scope). `Session::from_path_with_n_ctx` and
  `LlamaCppEngine::from_path_with_n_ctx` remain as shorthand for the
  common case.

- **`Model::extra_eos_tokens` → `Model::eog_tokens`**, and it now
  returns the *whole* end-of-generation set (`eos` and `eot`
  included) rather than the extras beyond them. For the llama.cpp
  backend that set is libllama's `special_eog_ids` verbatim —
  `llama_vocab_is_eog`, quirks and per-family workarounds included.
  It is the single authority for both "does emitting this end the
  turn" and "may this token be masked while a constraint is open";
  callers must not derive a stop set from `eos()`/`eot()`, which are
  labels the vocab applies, not statements about behavior (see
  *Fixed*). Backends that have no `is_eog` oracle report their own
  truth: `MoefluxModel` composes it from the tokenizer config, where
  `eot` genuinely does terminate a turn.
- **`Session::run_call` now breaks generation on grammar accept**.
  When any active `SamplingMode::Grammar` / `SamplingMode::Json`
  matcher reaches its accept state, the call halts immediately
  instead of continuing to wait for EOS. Belt-and-suspenders with
  the Deny mask: Deny prevents reserved tokens from being sampled;
  break-on-accept terminates cleanly the moment the structured
  output is satisfied. Includes deferred-grammar phase-split paths
  (post-`</think>` JSON matchers terminate the same way once their
  root rule completes).
- **`Engine<D, M>` → `Engine<B: Backend>`.** Type aliases preserve
  the public names: `LlamaCppEngine = Engine<LlamaCppBackend>`,
  `MoefluxEngine = Engine<MoefluxBackend>`. Inherent-method blocks
  on the aliases (state ser/de, log callbacks, `from_path*`, etc.)
  unchanged.
- **Predictor family migrate the same way.** `CandidatePredictor`,
  `TokenPredictor`, `PiecePredictor`, `Predictor` all become
  `<'engine, B>` instead of `<'engine, D, M>`. Iterator-impl `M:
  Sync` bound collapses into Backend's trait-level requirement.
- **`Session<B: Backend>`.** Generic chat-style API. Backend-
  specific constructors (`Session::<LlamaCppBackend>::from_path*`
  with `quiet`; `Session::<MoefluxBackend>::from_path`) live in
  cfg-gated impl blocks. Generic methods (`from_engine`,
  `with_*`, `complete_*`, `engine`, `engine_mut`) live in
  `impl<B: Backend>`.
- **`ChatTemplate::from_model<M: Model>`** and
  `tokenize_with_breakpoints<M: Model>` generalize over the trait.
  `mod chat_template` is no longer gated on `feature = "llama-cpp"`.
- **`mod session` cfg gate** flips from `feature = "llama-cpp"` to
  `any(feature = "llama-cpp", all(feature = "moeflux", target_os
  = "macos"))`.
- **`unsafe impl Send for Engine`** dropped — auto-derive picks it
  up from `B::Decoder: Send` + `B::Model: Send` baked into the
  Backend trait.
- **`llama-cpp-sys-3` 0.7 → 0.8.1.** Picks up the upstream cmake
  `mtmd` target, libmtmd bindgen, and packaging that back the new
  `mtmd` feature.
- **`misanthropic` alpha.3 → alpha.12.** Adds the image content-block
  types `Session` needs to accept `Block::Image`; the `image` /
  `jpeg` / `png` sub-features are pulled in by drama_llama's `media`
  feature.
- **`Session` tool-call termination is constraint-owned.** With a
  dialect active, the emitted call's own close marker (not a raw
  sampled EOG) terminates the turn: EOG and empty-piece tokens are
  masked while a constraint is live, the repetition penalty is
  suspended across structural emission (region-aware within free-text
  spans — see below), and the recorded tip is the canonical close
  token rather than whatever EOG happened to be sampled — so the next
  turn's cache walk stays byte-stable.
- **Repetition penalty now applies inside grammar free-text regions.**
  The penalty was previously suspended for the entire span of any
  active byte-constraint — which also silenced it inside the free
  islands where the model writes prose (JSON string bodies, `until()`
  spans), letting small models loop a paragraph verbatim inside a
  forced tool-call argument. Suspension is now scoped to *structural*
  emission (delimiters, keys, tags); inside permissive regions the
  penalty runs against a call-local n-gram accumulator, with
  region-exit tokens (the closing quote, merged `",` pieces) left
  unpenalized so the model can always leave the region. Default-on;
  opt out with `RepetitionOptions::set_constrained_regions(false)`,
  which restores the pre-feature blanket suspension exactly. ([#43])
- **Default sampler chain prepends a top-k 1024 cut before
  locally-typical.** The stock `SamplerConfig` now applies a top-k
  1024 pre-cut ahead of the locally-typical stage (typical mass
  concentrates in the head, so the cut trims the tail cheaply).
  Output for streams pinned by seed against the previous default
  chain will differ.

### Fixed

- **`blallama` served every llama.cpp model at `n_ctx = 512`.** It is
  generic over the backend, so it could only reach `FromPath`, and
  `FromPath` carried nothing but a path — leaving llama.cpp's own
  512-token default in place with no way to override it. Its
  `session_ready` log line had been reporting this all along. Fixed
  by `FromPath::Options`; `--n-ctx` now defaults to 32768.
- **`--seed` did not exist on any example.** The field in
  `CommonArgs` was missing its `#[arg(long)]` attribute, so clap made
  it a positional argument instead of a flag.

- **Cache usage counters are honest now** ([#40]).
  `cache_creation_input_tokens` had been hardcoded `Some(0)` since the
  original caching commit; it now reports the prompt tokens newly
  decoded into the cache this call (`input − read`, per the Anthropic
  field semantics — every decoded token lands in the slot's
  tip/breakpoint snapshots). With the prefix cache **disabled**, both
  cache counters are now `None` ("not reported") instead of `Some(0)`,
  so consumers can finally distinguish cache-off from a healthy cold
  call. `input_tokens` stays the full prompt. Additionally,
  `complete_response`'s `Message.usage` is now the *same* `Usage` the
  session records as `last_usage` (carried through `CallOutcome`)
  instead of an identical second build.
- **The constructor-default repetition penalty no longer penalizes
  specials.** `from_engine` seeded `SamplerConfig::default()` without
  the specials injection the `with_repetition` / `with_sample_options`
  setters apply, and the per-call assembly discards
  `add_model_stops`' injection — so a session that never routed
  through those setters (no sidecar on disk, sidecar parse error,
  `from_engine` directly) penalized its own EOG/framing tokens,
  making every turn less likely to end than the last. Injected at
  construction now; all paths protected.
- **Harmony turns died at the end of their reasoning block —
  `eot` is not a stop token.** libllama auto-detects the EOT token
  *by text*, and `"<|end|>"` is on that match list, so gpt-oss's
  `eot()` is `<|end|>` — its in-stream *channel separator*. libllama
  then removes `<|end|>` from `special_eog_ids` precisely so the model
  can close an analysis channel and keep going, and leaves
  `special_eot_id` pointing at it; upstream stays consistent because
  its generation loop only ever asks `llama_vocab_is_eog`. drama_llama
  instead rebuilt the stop set by hand as `{eos} ∪ {eot} ∪ extras`, in
  seven places, dragging `<|end|>` back in. Unconstrained, a Harmony
  turn stopped dead after its analysis block (one lone `Block::Thought`
  came back — no answer, no tool call); under a tool grammar, where the
  same set is masked while the constraint is incomplete, the model
  could not emit the token that closes the channel and rambled to
  `max_tokens`. `<|end|>`'s piece was also being stripped from the
  surfaced text, which would have left the dialect parser with an
  unterminated reasoning block. Fixed by deleting the union: see
  `Model::eog_tokens` under *Changed*.
- **Qwen3 chat-template thinking-mode forced on by default.**
  `ChatTemplate::render_with` never consulted `prompt.thinking`,
  leaving the Jinja `enable_thinking` variable undefined. Templates
  that gate their `<think>` block on it (Qwen3 family) interpret
  undefined as "thinking on" and emit `<think>\n` after
  `<|im_start|>assistant\n`, forcing the model into thinking mode
  regardless of caller intent. Now derived from
  `prompt.thinking.is_some()` mirroring Anthropic's API semantics
  (`thinking: None` = disabled, `Some(_)` = enabled). Caller-set
  `RenderOptions::with_extra("enable_thinking", _)` continues to win
  for explicit overrides. ollama exhibits the same bug for the same
  reason. Coverage in `tests/template_rendering.rs`.
- **Reserved-token loop on grammar-constrained generation.**
  Tokenizers like Qwen3.5/3.6 carve out a reserved tail of the
  vocab (~248088..248320 for Qwen3) for special-token slots, only
  some of which have registered text content; the rest decode to
  empty strings. Empty-piece tokens contribute zero bytes to a
  byte-stream-driven grammar's matcher and are trivially accepted
  regardless of state, while EOS (`<|im_end|>`) decodes to
  non-empty text the grammar rejects. Result: post-JSON, the model
  could land in a loop scattering reserved tokens until
  `max_tokens` exhausted. Cross-backend testing (A3B on llama.cpp
  vs moeflux) confirms the issue lives at the model/grammar
  layer, not in either backend's decode path. Fixed via the
  `SamplingMode::Deny` mask + `Model::eog_tokens` plumbing
  + grammar-accept-state break described above.
- **Repetition-penalty additive growth on long generations.**
  The additive `count * penalty_freq + penalty_present` term grew
  unboundedly with generation length because `NGramStats` was a
  monotonic frequency map with no eviction. Past ~200 steps the
  additive contribution dominated the model's natural logit
  gradient and content prose collapsed into thesaurus chains or
  fragment loops (the dominant cause of the Qwen3 long-form
  degradation arc). Fixed by replacing the lifetime count with a
  windowed-decay structure: each n-gram tracks the positions of
  its occurrences inside the last `RepetitionOptions::window_size`
  generation steps; the effective count fed to the penalty math
  is `Σ decay^(current_step - position)`, bounded above by
  `1 / (1 - decay)`. With defaults (window=256, decay=0.95) the
  effective count saturates near 20 regardless of how long
  generation runs. With the growth bounded,
  `SamplerConfig::default()` now ships the penalty **on** — the
  unbounded additive term was the reason it had been off — with
  `SamplerConfig::greedy()`, the per-model sidecar, and blallama's
  `--no-penalty` as the opt-outs.
- **Special-token injection through prompt content.** `Session`
  rejects `Block::Text` content bearing chat-format control tokens
  (`<|im_end|>` and friends) or media markers at ingest via
  `check_no_special_injection` — a framing token inside content is
  an accident or an injection, never meaning, and letting it through
  desynchronizes the KV cache, the block parser, and the marker-
  count contract. This is format-integrity enforcement, not content
  filtering: `Session` owns it, `Engine`/the raw predictor stay
  permissive for callers deliberately hand-feeding control tokens.
  The guard scans `ToolUse` surfaces too — tool name, call id, every
  string leaf *and key* of the `input` (`ServerToolUse` included),
  and `ToolResult.tool_use_id` — because templates render all of
  them verbatim and ingest tokenizes with specials enabled, so a
  special piece in any of them became real control tokens; the
  [#37] relay scenario is exactly a tool-use-shaped payload.
- **Ingest injection guard no longer false-positives on `add_bos`
  vocabs.** The guard keyed off raw tokenization including a leading
  BOS the caller never wrote; on `add_bos` vocabs that flagged clean
  content. Now compared against the content's own token span.
- **Grammar exit-marker EOG exemption + until-delimiter trim.** A
  completed constraint whose close marker *is* an EOG-adjacent token
  no longer double-terminates or trims the closing delimiter out of
  the emitted text; incomplete-constraint violations surface as
  errors instead of silent truncation.
- **A safe-code use-after-free through `pub Engine.model`** ([#54]).
  `llama_context` keeps a reference to its model for its whole life,
  but nothing tied the decoder's lifetime to the model's — only
  `Engine`'s field declaration order kept drops sound, and
  `engine.model = other` through the `pub` field freed the weights
  under a live context. Two lines, all safe code. `LlamaCppModel` is
  now a cheap-clone refcounted handle (a private inner owns the
  pointer); the decoder and `Mtmd` each hold a clone, so the weights
  structurally outlive anything referencing them. `Engine.model` is
  private behind `Engine::model()` — now for coherence rather than
  safety: swapping the model would leave the tokenizer disagreeing
  with a KV cache built from the previous weights. See *Migration*.
- **The auto-tip was silently defeated on `add_bos` vocabs.** The
  canonical close token was tokenized with `add_special = true`, so
  on Gemma / Llama-3-style vocabs the stored tip ended `[.., BOS,
  close]`; the next call's longest-common-prefix walk stopped at the
  BOS and fell back to the last breakpoint on every such model.
- **Dialect analyzer / parser hardening from the pre-publish
  review.** The template diff-split and its common-prefix/suffix
  helpers now round byte-wise match lengths down to char boundaries
  in both strings — two renders diverging inside a multi-byte
  character (any non-English template) panicked `Session` load,
  outside the analyzer's `catch_unwind`. `heal_json` copied
  quoted-string bytes `as char` (a Latin-1 reinterpretation),
  mojibaking non-ASCII tool arguments into valid JSON that serde
  then accepted — silent corruption on every heal path.
  `parse_tagged_call` gained a per-iteration progress guard: a
  degenerate `CallSyntax` (reachable via the analyzer's fallback)
  consumed zero bytes per iteration forever; zero progress is now
  `Malformed`. The `Malformed`-degrade path and
  `extract_args_markers` no longer slice mid-character or chop real
  content on non-ASCII input.
- **An over-`CAPACITY` `ngram_max_size` panicked prompt seeding.**
  The prose-seeding fold now clamps to `NGram::CAPACITY` like the
  live penalty pass does; previously `with_repetition` with a large
  `ngram_max_size` plus any `complete_*` over a long-enough prose
  block hit an `unwrap` in the fold.

### Removed

- **`cli::Args`** and **`LlamaCppEngine::from_cli`** — superseded by
  `LlamaCppOptions`, which is the same idea with the missing knobs,
  serde support, and no CLI dependency. `regurgitater` wraps it in
  its own `Parser` struct.
- **`Session::from_path_cpu_only`**, **`Session::from_path_with_flash_attention`**,
  **`Session::from_path_with_cache_slots`**, and the matching
  `LlamaCppEngine::from_path_cpu_only` /
  `from_path_with_flash_attention` / `from_path_with_n_ctx_and_seqs`
  — all expressible as `from_path_with(path, options)`. The first two
  had no callers at all.

- **`blallama --repetition-penalty`** (the v0.7.x band-aid opt-in
  flag). Sampling configuration now comes from the per-model
  sidecar; for force-off probe runs use the new `--no-penalty`
  flag, which overrides the sidecar.
- **`StopWords` and the `*_stopwords` methods** — the pure-rename
  shims deprecated in 0.7.0 (`type StopWords = IgnoreCategory` and
  the four `ignored_stopwords`-family methods on
  `RepetitionOptions`). Live replacements have existed since 0.7.0,
  and 0.8.0 is a breaking release, so they go now rather than riding
  through another one. The TOML sidecar's legacy `ignored_stopwords`
  key keeps working — that back-compat lives in a serde alias on the
  field, not in the removed API.

### Migration

- Most callers see no change: `LlamaCppEngine`, `LlamaCppModel`,
  `MoefluxEngine`, etc., are preserved as type aliases / re-exports.
- Callers that explicitly spelled out generic parameters
  (`Engine<LlamaCppDecoder, LlamaCppModel>`) should switch to
  `Engine<LlamaCppBackend>` or just `LlamaCppEngine`.
- `Session` is now `Session<LlamaCppBackend>` (or `Session<MoefluxBackend>`).
  If you stored `Session` in a struct field, parameterize the field.
- `Session::engine()` returns `&Engine<B>` (was `&LlamaCppEngine`).
  For a `Session<LlamaCppBackend>` that's the same type — calls
  unchanged. For ergonomic surface unchanged uses, prefer
  `session.engine().model().display_name()` over the now-llama-cpp-
  only `session.engine().model().file_name()`.
- `Engine.model` is no longer a `pub` field ([#54], see *Fixed*);
  use the `Engine::model()` accessor.
- `Session::from_path_sync(p)` → `Session::from_path(p)`, with
  `use drama_llama::FromPath;` — it is a trait method now.
- The specialized constructors become one call with an options
  struct. `from_path_with_cache_slots(p, 4096, 3)` becomes:

  ```rust
  Session::from_path_with(
      p,
      LlamaCppOptions::default().with_n_ctx(4096).with_cache_slots(3),
  )
  ```

  Note `cache_slots: Some(1)` is still not the same as leaving it
  unset — it switches the KV cache to unified, exactly as the old
  three-argument constructor did.
- Loading in async code: `from_path_async(path, options)` runs the
  load on tokio's blocking pool. The old `FromPath::from_path` was
  async; the new one is sync.

### Notes

- **`blallama` takes its repetition-penalty setting from the
  per-model sidecar**, not from a flag. The v0.7.x defaults
  (`penalty_max_count=1`, `ngram_min_size=1`, `penalty_repeat=1.06`)
  were sized for small downstream models in Weave; on the larger MoE
  models drama_llama now drives they over-penalise common content
  tokens during long-form free-text generation and degrade output to
  thesaurus chains or sentence-fragment loops. `--no-penalty`
  overrides the sidecar to force the filter OFF for probe and canary
  runs; there is no flag to force it on, because the sidecar is where
  that decision now lives.

  (This bullet said "no longer enables the filter by default … new
  `--repetition-penalty` flag re-enables it", carried over from
  v0.7.x. That flag was removed in this same release — see *Removed*
  — and the sentence had been contradicting the entry three sections
  up ever since.)
- **Upstream moeflux MAX_K bump.** moeflux fork commit `d013a0b`
  raises `MAX_K` from 8 to 16 in `metal_infer/infer.m` (plus the
  combine-shader binding shifts). Without it, A17B (`K=10`) silently
  drops 2 of 10 routed experts per layer per token because the
  `actual_K = (K > MAX_K) ? MAX_K : K` clamp at line 5364 was a
  no-op for A3B (`K=8`) but truncated A17B unconditionally; the
  corresponding routing-weight normalisation already happened over
  the full K, so the dispatched MoE residual was also under-scaled.
  Not the dominant cause of the long-form-degeneration symptom we
  diagnosed (the repetition-penalty defaults above were), but a real
  correctness bug fixed in passing.
- Build matrix: `--no-default-features` (trait layer only),
  `--features llama-cpp,...` (default), `--features
  moeflux-model-qwen3-6-35b-a3b` (moeflux only on macOS), and both
  enabled together. All four combinations build clean.
- **Dev workflow: `justfile` + cargo-nextest** (`just setup` installs
  nextest). `just test` runs the fast suite GPU-accelerated,
  `just test ignored` runs only the long-running `#[ignore]`d
  GPU/model tests, `just test all` runs both, and `just test cpu`
  runs CPU-only — the recipes are thin wrappers over
  `scripts/test.py`, which owns the configuration × tier topology so
  the justfile, the hooks, and CI cannot drift ([#68]). CUDA is
  auto-enabled on Linux (Metal is automatic on macOS) — deliberately
  kept OUT of the crate's default features so a bare `cargo build`
  stays portable. A 30B-class model barely fits once on a 24GB card,
  so the model tests must run one-at-a-time; the nextest `full`
  profile caps `test-threads = 1` (see `.config/nextest.toml`) and
  `just test ignored` / `just test all` run under it, so the whole
  set is serialized without a fragile per-test filter. GPU vs. CPU
  builds use separate target dirs to avoid evicting each other's
  llama.cpp build.
- Send/Sync trade-offs: `B::Decoder` is required Send (not Sync) —
  `*mut llama_context` is internally mutable. `B::Model` is Send +
  Sync (Iterator impls hand `&Model` to grammar / sampling code
  that fans out across rayon).
- See `.claude/memory/moeflux_disk_convention.md` for the
  forward-looking on-disk layout `MoefluxEngine::from_path`
  expects, and the migration story for current artifacts.
- **Known issues** (tracked, not fixed in this release):
  - Context-full during generation ends the stream silently and, on
    the grammar path, misreports as a `GrammarViolation` ([#36]).
  - A tool call truncated by the token budget can seat frame-marker
    text in the transcript; containment (not prevention) is the
    planned fix ([#38]).
  - On the un-grammared lazy path, nothing forbids EOG while a
    thought block is open, so a model can end its turn mid-thought
    ([#64]).

## [0.7.0] — 2026-04-22

Major release. Prompt caching, structured output, grammar-perf finish
line, and a top-to-bottom cleanup pass on the prompt primitives. Requires
`llama-cpp-sys-3` `0.7`, tracking llama.cpp `b8882-5-g82d3f4d3b`.

### Added

- **Prompt caching** (KV-cache reuse across calls). `Engine::prefill` +
  `predict_*_resuming` resumes generation from a populated KV without
  re-decoding the prefix. `Session` tracks previous-turn tokens and
  breakpoints, computes longest-common-prefix `L_hit` with BPE-safety
  backoff, and narrows the KV window on partial reuse. `ChatTemplate`
  supports breakpoint-aware rendering (`render_with_breakpoints`).
  `response::Message` return shape surfaces token usage (input / output /
  cache_read) and a stop-reason. See the `chat_repl` example.
- **Structured output** via `Prompt::output_config`. New `output_config`
  module compiles a `misanthropic::OutputConfig` to a GBNF grammar and a
  `SamplingMode::Grammar`. The shared `grammar_compile` module handles
  `$ref`, `anyOf`, and `const` schema shapes (schemars-emitted schemas
  round-trip cleanly). `Session::complete_*` routes the compiled grammar
  through `SampleOptions::modes`. New `json-schema` feature adds typed
  helpers: `Prompt::structured_output::<T>()`, `OutputConfig::for_type::<T>()`.
- **Thought/JSON phase-split.** New `DeferredGrammar` and
  `SampleOptions::deferred_grammar` let a grammar stay suspended until a
  trigger byte sequence appears in the predictor's output, then get
  promoted into `modes`. `OutputConfigOptions::phase_split` (default `true`)
  compiles a JSON-only grammar triggered by `</think>` — grammar filtering
  is skipped entirely during the thought preamble. `CompiledOutputConfig::
  {Single, Deferred}` + `compile_output_config` / `compile_prompt_output_config`
  expose the phase-split-aware compiler. Legacy `grammar_for_output_config` /
  `grammar_for_prompt` remain as the unified-grammar path. `TokenPredictor`
  drives promotion; post-trigger tail bytes are fed through
  `GrammarState::advance_bytes` so the matcher lines up with the model.
- **Lazy-DFA grammar cache.** `DfaCache` interns canonical `StackState`
  values into `StateId`s and memoizes one-byte transitions + first-byte
  bitmaps. Hot path becomes a `DashMap` lookup; misses pay the current
  `feed_byte` + intern cost. Shared across clones of `GrammarState` via
  `Arc`. Default-on; disable via `DRAMA_LLAMA_DFA_CACHE=0`. Extended
  `GrammarStats` with `dfa_states` / `dfa_transition_hits|misses` /
  `dfa_bitmap_hits|misses`.
- **Grammar matcher profiling.** Opt-in per-call stats via
  `DRAMA_LLAMA_GRAMMAR_STATS=1`. `grammar_stats_snapshot()` /
  `grammar_stats_reset()` return cumulative counts of filter calls,
  candidate survival at each prefilter stage, stack depth, and wall-clock.
- **Tool-choice constrained generation.** `grammar_for_tool_choice`
  emits GBNF for `ToolChoice::{Auto, Any, Method}` with optional
  `wrap_tags` and an `allow_thought` preamble. Session priority is
  `tool_choice > output_config > none`.
- **`Session::from_path_with_n_ctx`** — construct a session with a custom
  KV context size without crafting unsafe FFI params.
- **`blallama` example** — small `/v1/messages` server.
- **Examples**: `whodunit` (structured output integration),
  `chat_repl` (prompt caching demo), `--no-grammar` and `--phase-split`
  flags on `whodunit` for baseline measurements.

### Changed

- **Prompt primitives are misanthropic-native.** `Message` / `Content` /
  `Block` / `Role` come from misanthropic and are aliased to `'static`;
  `Prompt` is a thin wrapper. `RenderOptions::with_extra<V: Serialize>` is
  now generic over serializable extras.
- **`ChatTemplate`** renders via minijinja + pycompat. Handles
  `raise_exception` and a `strftime_now` subset.
- **Sampling chain now applies grammar in parallel.** The per-candidate
  `grammar_filter` loop runs under Rayon (`3.5×` on complex grammars).
  Requires `unsafe impl Sync for Model` — post-load model state is
  immutable.
- **Grammar matcher** refactored for throughput: 256-bit first-byte
  acceptance bitmap prefilter; stack storage moved to
  `TinyVec<[Position; 8]>`; `StackState` split from `GrammarState` so the
  hot clone path doesn't bump the `Arc<Grammar>` refcount; fast-path
  `expand` skips alloc + sort + dedup when every stack is at a yield
  point; tail-call optimization in `expand` bounds stack depth for
  right-recursive rules like `.+`.
- **Repetition penalty rewrite (surgical/"B2")**. New `IgnoreCategory`
  variants for JSON / Punctuation; special tokens (EOS / EOT /
  ignored_stopwords) auto-added to the repetition ignore list. Ignored
  fields moved to `BTreeSet`.
- **`rocket::serde`** indirection dropped from the library.
- **`Session::complete_*` setup paths** polished — `complete_text` /
  `complete_stream` / `complete_blocks` / `complete` / `complete_response`
  all flow through the same prepare-call path.

### Removed

- **`Vocab` / `VocabKind` subsystem** and `data/banned.rs`. Content
  filtering belongs in the consuming app, not the library. See the note
  in `CLAUDE.md` — the Eric Hartford uncensored model check in
  `Model::from_file` stays.
- **`llama_params_fit` / `llama_memory_breakdown_print`** vanished from
  upstream llama.cpp between `b8809` and `b8882`; neither was exposed by
  this crate.

### Fixed

- `session: merge adjacent prose blocks on batch return`
  (`9b62626`).
- `example(whodunit): strip EOS piece from raw text before JSON parse`
  (`4361556`).
- `tool: add strict: None for new Method.strict field` (`c250d31`).

### Performance

On the `whodunit` workload (Qwen 3 8B Q8_0, structured output with
thought preamble):

| config                             | tok/s |
| ---------------------------------- | ----- |
| unconstrained (`--no-grammar`)     | ~20.0 |
| v0.6.2 grammar-constrained         | ~0.7  |
| v0.7.0 after bitmap + TCO + etc.   | 10.1  |
| v0.7.0 with DFA, no phase-split    | 8.9   |
| v0.7.0 with `--phase-split` + DFA  | **17.6** |

Phase-split on + DFA on: phase 1 thought runs at the unconstrained
ceiling (~21.5 tok/s, zero grammar filter calls) and phase 2 JSON at
~13.0 tok/s with 99.8% DFA transition hit rate. Workloads with wide
free-form `.+` regions inside JSON (some Agora reactor shapes) should
flip `DRAMA_LLAMA_DFA_CACHE=0`.

### Notes

- `cargo publish` for this crate is still gated on misanthropic 1.0
  landing on crates.io. Published as a git tag only.
- Known pre-existing test failures: `candidates::tests::test_apply_entropy`,
  `candidates::tests::test_sample_tail_free` are `todo!()` stubs;
  `model::tests::test_model`, `model::tests::test_model_desc` assume a
  Llama-family model and fail when `models/model.gguf` points at Qwen.
- egui 0.34 deprecation warnings (`clamp_range`, `id_source`) are left
  for a follow-up PR.

[0.8.0]: https://github.com/mdegans/drama_llama/releases/tag/v0.8.0
[0.7.0]: https://github.com/mdegans/drama_llama/releases/tag/v0.7.0
[#28]: https://github.com/mdegans/drama_llama/issues/28
[#36]: https://github.com/mdegans/drama_llama/issues/36
[#38]: https://github.com/mdegans/drama_llama/issues/38
[#64]: https://github.com/mdegans/drama_llama/issues/64
[#29]: https://github.com/mdegans/drama_llama/issues/29
[#30]: https://github.com/mdegans/drama_llama/issues/30
[#31]: https://github.com/mdegans/drama_llama/issues/31
[#37]: https://github.com/mdegans/drama_llama/issues/37
[#40]: https://github.com/mdegans/drama_llama/issues/40
[#43]: https://github.com/mdegans/drama_llama/issues/43
[#44]: https://github.com/mdegans/drama_llama/issues/44
[#48]: https://github.com/mdegans/drama_llama/issues/48
[#54]: https://github.com/mdegans/drama_llama/issues/54
[#60]: https://github.com/mdegans/drama_llama/issues/60
[#68]: https://github.com/mdegans/drama_llama/issues/68
[#76]: https://github.com/mdegans/drama_llama/issues/76
[#85]: https://github.com/mdegans/drama_llama/issues/85
[#88]: https://github.com/mdegans/drama_llama/issues/88
[#91]: https://github.com/mdegans/drama_llama/issues/91
[#95]: https://github.com/mdegans/drama_llama/issues/95
[#99]: https://github.com/mdegans/drama_llama/issues/99
[#101]: https://github.com/mdegans/drama_llama/issues/101
[#102]: https://github.com/mdegans/drama_llama/issues/102
[#113]: https://github.com/mdegans/drama_llama/issues/113
[#118]: https://github.com/mdegans/drama_llama/issues/118
[#119]: https://github.com/mdegans/drama_llama/issues/119
[#120]: https://github.com/mdegans/drama_llama/issues/120
[#138]: https://github.com/mdegans/drama_llama/issues/138
[#140]: https://github.com/mdegans/drama_llama/issues/140
[#141]: https://github.com/mdegans/drama_llama/issues/141
[#148]: https://github.com/mdegans/drama_llama/issues/148
