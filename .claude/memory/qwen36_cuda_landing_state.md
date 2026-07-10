# Landed: full suite green on Qwen3.6-35B-A3B / CUDA (2026-07-10)

First complete `--include-ignored` run in the crate's history: 298 lib
tests + every integration binary + doctests, zero failures, on
Qwen3.6-35B-A3B (UD-IQ4_XS) with CUDA 12.6 on the 3090 box. Prompt
caching verified end-to-end: Session-level (`tests/session_cache.rs`,
incl. cached-vs-fresh output parity), auto-tip across a tool-use round
trip (`hash_cache_smoke`), and over HTTP through blallama driven by
misanthropic's real client (`tests/blallama.rs`).

## Decisions made this session (with Mike, in-session)

- **`preserve_thinking` defaults to `true` at the Session level**
  (not ChatTemplate — fixtures/oracle stay neutral). Rationale:
  byte-stable transcripts are the prefix cache's contract on
  think-stripping templates (Qwen3.5/3.6 keep prior-turn reasoning
  only when this template var is set), and current Anthropic models
  (Opus 4.5+/Sonnet 4.6+/5) keep prior-turn thinking too. Mike:
  "I *do* want models to see their past thinking." Opt-out:
  `with_render_opts(...with_extra("preserve_thinking", false))`.
- **`minItems >= 1` grammar enforcement = non-emptiness only** —
  Anthropic parity (sanitizer passes 0|1). Addendum in
  `schema_constraint_keywords_decision.md`; fuzzer generates
  `minItems: 1` on a third of arrays.
- **llama.cpp thread default fixed**: ggml's library default is 4
  threads (llama.cpp itself marks it "TODO: better default");
  `LlamaCppEngine::default_context_params()` now uses
  `available_parallelism`. Every prior CPU run of this crate was
  4-threaded.
- **blallama emits the Anthropic error envelope**
  (`{"type":"error","error":{...}}`) — bare `AnthropicError` was
  unparseable by real clients (misanthropic's own client caught it).
  Upstream ask to expose `AnthropicErrorWrapper`: misanthropic #134.
- **Qwen3.6 tool-call shape is XML-ish, not JSON** — full analysis
  and plan in issue #29 + `qwen36_xml_tool_call_shape.md`. The
  round-trip test skips block comparison when it detects the shape.

## Gotchas that cost time (don't re-hit)

- HF throttles single-connection downloads (~1 MB/s); 12 parallel
  range requests hit ~32 MB/s. `scratchpad/parallel_fetch.sh` pattern.
- `cargo test --features cuda` without `/usr/local/cuda/bin` on PATH
  fails in the sys build script; cmake wants nvcc visible.
- Under `--features cuda`, non-`#[ignore]` model tests must load
  CPU-only or the parallel runner stacks models into VRAM
  (`load_test_model_cpu` in model.rs tests).
- Global `llama_state_get_size` is content-dependent: never assert a
  saved buffer matches the *current* size before `set_state` (bug
  fixed this session; per-seq API was already correct).
- Regression goldens are per-model: regenerate with
  `DRAMA_LLAMA_UPDATE_GOLDEN=1` after a model swap.

## Still open

- Issue #29: native XML-ish tool-call shape (emitter + parser +
  shape selection). Qwen is the #1 local family; priority next arc.
- Issue #28: lazy grammar check (this session's original plan-of-
  record; untouched today).
- misanthropic #134: provider-side constructors for non_exhaustive
  response types (drama_llama carries a serde workaround in
  `Session::empty_response_message`).
- `future_work_gguf_sampling_defaults.md`: seed sidecar from
  `general.sampling.*` GGUF keys.
- Disk-backed snapshot persistence for the 1h TTL
  (`llama_state_seq_save_file` already bound).
