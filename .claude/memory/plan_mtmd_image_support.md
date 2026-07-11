# Image support via llama.cpp mtmd (drama_llama + llama-cpp-sys-3)

## Context

drama_llama's `Block::Image` (upstream misanthropic, `image`-crate decode)
is currently **silently dropped** at render time (`src/chat_template.rs:920`,
catch-all in `append_block_text`). Image input is the last feature on the
v0.8.0 publish list.

llama.cpp's multimodal support is `libmtmd` (`tools/mtmd/` in our pinned
submodule, commit `52b3df0`, 2026-06-21): C++ implementation (clip.cpp +
~35 per-model encoders), **clean C API** (`mtmd.h`, `mtmd-helper.h`,
`extern "C"`, opaque handles). Rolling our own vision stack is a
non-starter; binding the C API is the llama.h playbook we already run.

**Ecosystem check (2026-07-11)**: no standalone mtmd crate. The one quality
binding (llama-cpp-2/utilityai, MIT+Apache) hard-vendors its own llama.cpp —
unusable beside our submodule (duplicate ggml/llama symbols). Decision:
**`mtmd` feature in our own llama-cpp-sys-3**, version-locked to our pin.
Design reference for the safe layer: [llama-cpp-2's mtmd.rs](https://github.com/utilityai/llama-cpp-rs/blob/main/llama-cpp-2/src/mtmd.rs) (MIT/Apache).

**Decisions from discussion (Mike, 2026-07-11):**
- Upstream CMake build, not a cc::Build source glob.
- **Cache-aware from the start** — no "images force cache miss" interim.
- **Rust-owned chunk-eval loop from the start** (upstream helper is ~310
  lines C++; we differential-test against the bound helper). Enables
  pre-KV NaN detection.
- Validation: Qwen 3.6 (M-RoPE) + Gemma 4 (non-causal image attention);
  f16 mmproj sidecars installed locally. Fixture:
  `tests/data/images/samoyed.jpg` (committed).
- Out of scope: gpt-oss vision tunes, audio (no misanthropic Block variant;
  don't invent one), video (`MTMD_VIDEO=OFF`).
- CI: llama-cpp-sys build/link/bindgen smoke only, no inference.
  drama_llama CI: SmolVLM-256M plumbing smoke (logit-ordering or weak
  contains assertion). Breed-level ("samoyed") assertions in local
  `#[ignore]` tests with grammar-constrained answers.
- Local drama_llama test runs use `--features cuda` (recently dropped from
  defaults; CPU is painful).
- NaN guard: yes. NaN is maximally contagious in KV (one poisoned cell →
  all later logits NaN). Detect at encoder output (pre-KV) in our eval
  loop → typed error, caller decides; `record_cache_miss_on_error` wipe
  is the backstop. Decode `Block::Image` with the `image` crate only —
  never mtmd's bundled stb_image (CVE history; adversarial-input posture).

**Plan validated** by an adversarial review agent against the actual code
(session/mod.rs, predictor.rs, batch.rs, mtmd.h/mtmd-helper.cpp, llama.cpp
batch/KV internals). 10 design holes found and fixed below (marked ⚠ where
they change the obvious approach). Verdict on the shape: sound.

**Plan of record**: [issue #31](https://github.com/mdegans/drama_llama/issues/31)
(canonical; this file is the in-repo mirror). Implementation lands in
three sessions: A (llama-cpp-sys), B (safe layer), C+D (Session
integration, one big session). Planned 2026-07-11 (Fable 5 + adversarial
validation agent); decisions credited inline are Mike's.

---

## Phase A — llama-cpp-sys-3 `mtmd` feature (own session)

Repo: `~/Projects/llama-cpp-sys` (github.com/mdegans/llama-cpp-sys).
PR from fork; CI (win/mac/linux, ~10m) must pass; merge to main deploys.

1. Cargo feature `mtmd` (off by default).
2. `build.rs`, feature-gated:
   - `LLAMA_BUILD_COMMON=ON`, `LLAMA_BUILD_TOOLS=ON` (mtmd's
     add_subdirectory gated on both, upstream `CMakeLists.txt:217`),
     `MTMD_VIDEO=OFF`.
   - Build the `mtmd` **target**, not `install` (mtmd links only
     ggml+llama; upstream FATAL_ERRORs if it links llama-common — no tool
     executables get built). Keep the existing `install` build for the
     core libs; link `libmtmd.a` from `build/tools/mtmd/`.
   - `rustc-link-lib=static=mtmd` + search path.
   - bindgen: add `tools/mtmd/mtmd.h` + `mtmd-helper.h` to the existing
     builder, `allowlist_function/type("mtmd_.*")` (helper fns match the
     same pattern; we want them for differential testing).
3. **Packaging (hidden work)**: extend Cargo.toml `include` with
   `external/llama.cpp/tools/mtmd/**` **excluding test media** (there's a
   .mp4; crates.io cap 10MB), `tools/CMakeLists.txt`, and
   `external/llama.cpp/vendor/**` (stb_image/miniaudio — clip.cpp includes
   them unconditionally). Verify: `cargo package --list`, then build the
   packaged crate with `--features mtmd`.
4. Tests: link smoke only (`mtmd_default_marker()` non-null,
   `mtmd_context_params_default()` sane). No models.
5. CI: add `--features mtmd` build to the 3-OS matrix.

## Phase B — drama_llama safe layer (own session)

New module `src/llama_cpp/mtmd.rs`; feature
`mtmd = ["llama-cpp", "llama-cpp-sys-3/mtmd", "dep:image", "image/jpeg", ...]`
(+ misanthropic image features for `Block::Image` decode; fixture is .jpg).

House style: thiserror per-op error enums + `assert_impl_all!(Send, Sync)`,
`from_path*` constructors, consuming-self where state must stay consistent.

- `Mtmd` — owns `*mut mtmd_context`.
  `from_path(mmproj, &LlamaCppModel, MtmdParams) -> Result<Self, MtmdNewError>`;
  `supports_vision/audio()`, `marker()`,
  `tokenize(&self, text, &[BitmapRef], add_special) -> Result<Chunks, MtmdTokenizeError>`.
- `Bitmap` — owns `*mut mtmd_bitmap`. `from_rgb8(nx, ny, &[u8])`,
  `try_from_block(&Block)` (image-crate decode), `id()` = our sha256 of
  pixels (feeds cache identity; not mtmd's FNV). Dims validated by
  construction (nx·ny·3 == len; reject 0-dim).
  **Placeholder bitmaps are a separate type or typestate** (upstream:
  data==nullptr tokenizes/counts but must carry real nx/ny, and encode
  rejects them — enforce "can't encode a placeholder" at the type level).
- `Chunks`/`Chunk<'_>` — owns `*mut mtmd_input_chunks`;
  `Chunk::Text(&[Token])` | `Chunk::Media { id, n_tokens, n_pos }`;
  `n_tokens()`, `n_pos()` (M-RoPE-aware, ≠ n_tokens).
- ⚠ **Ownership**: `LlamaCppEngine` is `pub type … = Engine<LlamaCppBackend>`
  (`src/llama_cpp/engine.rs:22`) — a generic struct; it cannot own
  backend-specific state. `Option<Mtmd>` hangs off **`LlamaCppModel`**
  (or `LlamaCppDecoder`; either drops before the raw model per documented
  field order, `engine.rs:22-27`), surfaced through the `Model` trait as a
  defaulted accessor (`fn mtmd(&self) -> Option<&…> { None }`-shaped) so
  generic Session code can branch and moeflux gets the typed
  "media unsupported" error for free.
- Sidecar: `from_path*` auto-loads `<model>.mmproj.gguf` (extends
  `src/sidecar.rs` convention); explicit constructor for arbitrary paths.
- Interface refined in-session; this sketch was reviewed but not committed.

## Phase C+D — Session integration, cache-aware (one big session)

Code map (validated file:line):
- Render: `ChatTemplate::render_with[_breakpoints]` `src/chat_template.rs:226/:338`;
  image drop at `append_block_text` `:920`; breakpoints `collect_breakpoints`
  `:532-556` (granularity AfterTools/AfterSystem/AfterMessage — a
  breakpoint can never land inside a media chunk; validated).
- Tokenize: full `src/session/mod.rs:1371`, per-partial `:1375`; cache-off
  and trace paths `:1277-1285`, `:1397-1401`, `:2668`.
- KV walk: `kv_setup_and_chunk_prefill` `:1476`; `PrefixCache` `:218-277`;
  LCP `:304/:383`; hash fast path `:340`; tip extension `:2015`;
  error wipe `record_cache_miss_on_error` `:1813`.
- Predictor: non-resuming `memory_clear` `src/predictor.rs:336`; resuming
  `new_resuming` `:367` (prefills trailing text itself at start_pos —
  correct splice primitive, validated).
- Batch FIXME `src/batch.rs:30`; decoder prefill `src/llama_cpp/decoder.rs:491`;
  `restore_to` dense-position sanity check `:648-649`.

### Work items

1. **Marker emission + media collection** (`chat_template.rs`):
   `append_block_text` emits `mtmd.marker()` for `Block::Image`; new
   `collect_media` sibling to `collect_breakpoints` gathers decoded
   bitmaps in order. Images + no Mtmd → typed error (no silent drop).
   ⚠ **Marker injection**: user text containing the literal marker changes
   the marker count seen by `mtmd_tokenize` (hard error or misalignment) —
   escape or reject marker occurrences in non-media text; typed error fine.
2. **Tokenization via mtmd in ALL prepare paths** ⚠ — cache-on, cache-off
   (`:1397-1401`), `prepare_call` (`:1277`), and `top_k_trace` (`:2668`)
   must route media prompts through `Mtmd::tokenize`; plain
   `Model::tokenize` would BPE the marker as prose. Per-partial counting
   uses **placeholder bitmaps with per-partial slices** (n_bitmaps must
   equal marker count in that partial; count = images in `messages[..=i]`).
   Efficiency: tokenize with placeholders for counting/identity; only
   tokenize with pixels when a media chunk actually needs (re-)encoding —
   avoids re-running image preprocessing on every cache-hit turn.
   (Validated: partial-vs-full text-chunk tokenization is stable except
   the partial's final text chunk; generalize the existing fail-open
   prefix check `:1380-1387` to per-chunk comparison.)
3. **PrefixCache generalization** — `prev_tokens: Vec<Token>` →
   `Vec<CacheEntry>`, `CacheEntry = Token(Token) | Media { hash: [u8;32],
   n_tokens: u32, n_pos: u32 }`. LCP compares media by hash. Backoff-by-1
   stays entry-wise (media boundaries are special-token framed; no BPE
   risk — optionally skip backoff when the boundary entry is media).
   ⚠ **Number-space discipline** (the big one): introduce an entry-index
   newtype and ONE translation helper `pos_of_entry(idx) -> llama_pos`
   (position = Σ 1|n_pos of prior entries); funnel **every** engine
   position argument through it — `restore_to` (`:1587`),
   `checkpoint_pos`/`prefill_chunk` (`:1663-1670`), and the three missed
   `forget_pos` sites (`:1634`, `:1778`, `:1797`) — so a raw usize can't
   reach the engine. `internal_tip` stored **entry-space**;
   `compute_tip_extension` (`:2015-2057`) redone pos-space vs pos-space
   (M-RoPE image: n_tokens≈1024 cells all at pos_0, n_pos≈32 — current
   arithmetic underflows, truncates entries by a position count, and
   stores a pos-space tip compared entry-space at `:400`/`:356`).
   ⚠ **Hash mixing at BOTH sites**: one `hash_partial_with_media(text,
   &bitmap_hashes[..k])` helper used for per-partial hashes (`:1385`) AND
   the tip hash (`:2461`); mixing only one side either loses the tip
   silently or false-hits image A's KV for image B.
4. **`EmbdBatch` + Rust eval loop** ⚠ — `llama_batch_init` hard-allocates
   `pos` at n_tokens (llama-batch.cpp:894); M-RoPE needs n_tokens×4
   planes. Extending `Batch` is structurally insufficient (upstream's
   `decode_embd_batch` hand-assembles for the same reason). New
   `EmbdBatch`: Rust-owned buffers, hand-assembled `llama_batch` view, no
   `llama_batch_free`, logits all-false (nothing reads logits after an
   embd decode). Eval loop inherent on `LlamaCppDecoder` beside
   `prefill_inherent`: text runs → existing prefill; media →
   `mtmd_encode_chunk` + `mtmd_get_output_embd` → **NaN/Inf scan (pre-KV)**
   → EmbdBatch decode with normal or M-RoPE positions
   (`mtmd_helper_image_get_decoder_pos`); non-causal attn toggle around
   Gemma-style image decode **with reset on error paths** + up-front
   image-fits-in-ubatch check. All Results propagated — no `.expect` (the
   predictor's `:341/:382/:432` expects are exactly what KV-cell
   exhaustion would hit). Differential-test against bound
   `mtmd_helper_eval_chunks` (same context, same logits).
5. **Media-aware KV walk in Session** ⚠ — replace the chunked-prefill
   loop + suffix handoff (`:1658-1675`) with ONE walk over
   `[cache_read, end)`: text runs → `prefill_chunk`, media → eval-loop
   encode+decode, checkpoints at translated breakpoint positions; only the
   **trailing all-text run** (typed text-only so the non-resuming branch
   can't compile with media) goes to `predict_pieces_resuming`. Media
   present ⇒ `prefill_start > 0` structurally ⇒ the non-resuming
   `memory_clear` constructor is unreachable. Covers both validated
   gap shapes: media inside a prefill chunk (image mid-conversation,
   breakpoint later) and media in the suffix (empty-suffix backoff
   `:1573-1582` on identical-prompt retry).
6. **n_ctx fit check in cell space** ⚠ — position-based check
   (`predictor.rs:412-414`) undercounts M-RoPE images (1024 cells, 32
   positions). Up-front Session check: Σ n_tokens over entries +
   max_tokens ≤ n_ctx, typed error.
7. **Bookkeeping**: `make_usage`/`cache_read` (`:1685`, `:2510`) report
   media n_tokens, not entry counts. Document + `debug_assert` the
   dense-position assumption near `restore_to`'s sanity check
   (`decoder.rs:648`) — holds today only because breakpoints are
   message-granular and message-close text follows every marker.
8. NaN error surfacing: typed error names the offending bitmap id; caller
   decides (drop image / abort); Session routes through
   `record_cache_miss_on_error` so poisoned KV never survives.

## Verification

- **A**: 3-OS CI green with and without `mtmd`; packaged crate builds.
- **B**: unit tests, no model needed: bitmap dims/hash, Block::Image →
  RGB8 round-trip, placeholder-vs-real typestate; local `#[ignore]`:
  `Mtmd::from_path` + tokenize chunk structure against a real mmproj.
- **C+D**:
  - Differential: our eval loop vs `mtmd_helper_eval_chunks`, same
    context — identical logits.
  - CI (SmolVLM-256M Q8_0 + f16 mmproj, ~275MB, CPU): "breed of dog, one
    word" → logit-ordering assertion (samoyed-ish tokens > nonsense
    baseline like "potato") or weak contains("dog"). Plumbing test, not a
    vision-quality test.
  - Local `#[ignore]`, `--features cuda`: Qwen 3.6 + Gemma 4, same
    question, **grammar-constrained** to a fixed breed list (existing
    structured-generation support) → expect "samoyed".
  - Cache: same prompt+image twice → prefix reuse (restore, no
    re-encode); image bytes swapped, text identical → miss at the media
    entry (hash mixing test); image mid-conversation + later breakpoint →
    correct checkpoint positions (translation test).
  - Adversarial: prompt text containing literal `<__media__>` → typed
    error; 0-dim/mismatched-len bitmap → construction error; NaN
    injection (if we can force one, e.g. hand-built mmproj or a mock) →
    typed error + clean KV afterward.

## Open items (carried into implementation sessions)

- Naming: `Mtmd` (lean) vs `LlamaCppMultimodal` — trivial rename either way.
- Phase B refines the interface sketch before writing code.
- SmolVLM CI download caching (GitHub actions cache vs re-download).
