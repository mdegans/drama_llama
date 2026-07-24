# 1.0 wishlist — deferred from the 0.8.0 pre-publish review (2026-07-23)

Items the review surfaced that are *not* 0.8.0 blockers but will be
breaking (or embarrassing) to do after 1.0. The `#[non_exhaustive]`
sweep, moeflux privatization, and `StopWords` removal already landed
in 0.8.0; this is what remains.

## Breaking-later, so decide before 1.0

- **`type Token = i32`** (`src/backend.rs`) carries its own TODO to
  become an associated type on `Decoder`/`Model`/`Backend`. Guaranteed
  1.0 break if done later; decide, or commit to `i32` explicitly.
- **Flat crate root (~100 re-exports) vs public modules.** The stale
  `// TODO: version 0.2.0` in lib.rs still points the other way.
  Moving items out of the root is breaking; at minimum stop *adding*
  to the root (`SamplingParams`, `apply_request_sampling`,
  `build_grammar_source_for_debug` were recent additions that could
  live in modules; the last is a `#[doc(hidden)]` candidate).
- **Two log-callback surfaces**: crate-root free fns
  (`set_log_callback` et al., llama-cpp-gated) vs the backend-agnostic
  `Backend::set_log_callback`. Internally coherent; a consumer sees
  two blessed ways. Consolidate toward the trait, demote the free fns.
- **Constructor conventions**: `MoefluxEngine::from_path(parent:
  &Path)` vs `LlamaCppEngine::from_path(path: PathBuf)` — same name,
  different path conventions and ownership.
- **`Backend::is_supported_model(&std::fs::Metadata)`** bakes
  filesystem assumptions into the trait — revisit when a non-file
  backend (NPU arc) appears.

## Quality, non-breaking, any time

- **Gate clippy in CI.** The sweep itself landed 2026-07-24 (commit
  `refactor(lint): clippy sweep`): `cargo clippy --all-targets`
  (default features) is warning-clean, resolved by fix-or-`#[allow]`-
  with-a-reason (no blanket crate-level allows). What remains is the
  *gate*: add `cargo clippy --all-targets -- -D warnings` as a
  `test.py` subcommand and wire it through the hook/justfile/CI chain
  so all three stay one source of truth. Mike asked for this
  explicitly ("Are we checking clippy in CI?" — answer was no,
  nowhere). Note the sweep was scoped to *default features*; the gate
  should decide whether to also cover the other feature configs
  (`permutations`-style) — those weren't swept and may still warn.
- `#![warn(missing_docs)]` + the ~10 undocumented pub items the review
  listed (notably the root-re-exported `Tool`/`ToolUse`/`ToolResult`/
  `MessageResponse` aliases in src/prompt.rs use `//` not `///`, so
  docs.rs renders nothing for them; also `SamplingMode`'s enum-level
  doc is a literal `// TODO`).
- Session polish from the review's minor list:
  `record_cache_miss_on_error` uses `.or()` where `.or_else()` was
  meant (eagerly clears `last_active`, costing a full wipe on
  post-hit errors); output-token accounting differs between
  `complete_text` (counts EOG pieces) and `run_call` (skips them);
  `top_k_trace` stops on `eos()` not `eog_tokens()` (diagnostic path;
  the exact eog-vs-eos trap `eog_is_not_eos_plus_eot.md` documents);
  a Rust-side per-image marker-count assert after template render
  (option-level check exists; the render itself is unverified);
  `Breakpoint::ttl` doc says TTL enforcement "lands with multi-slot
  bounds" — it landed; the doc block above `merge_adjacent_prose`
  absorbed `infer_stop_reason`'s intended doc (comment-run fusion).
- Dialect: `TagWithTagged` render silently drops non-object args
  (`emit.rs` — typed error or doc note); `StreamParser::new` takes
  `Vec<Tool>` vs `parse_text`'s `&[&Tool]`; Harmony lenient-path
  recipient parsing absorbs a constraint clause into the tool name;
  bare-JSON `JsonNative` streaming stalls all streamed output on an
  unbalanced prose `{` until finish() (length-bound it).
- minijinja `tojson` HTML-escapes `<`/`>`/`&` while the grammar lets
  the model emit them raw → canonicalization-gate cache miss on tool
  args containing URLs. Doc note, or custom non-escaping tojson.
  (#60 landed 2026-07-24 — declaration order via `preserve_order`;
  the escaping mismatch is unchanged by it. See
  `byte_exact_round_trip_invariant.md` for the landed state.)
- mtmd upstream findings: see the 2026-07-23 addendum in
  `llama_cpp_ffi_audit.md` (F1 wants an upstream issue).
