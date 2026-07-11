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

**Plan-bounce amendments (Mike + Fable 5, 2026-07-11, pre-C+D)** — a
second lookover against the post-Phase-B tree before cutting C+D code:

1. **`EntryPos` carried pair** (work item 3, revised) — the bare
   entry-index newtype + one translation helper has an order-of-
   operations hazard: `record_cache_hit` overwrites `prev_tokens`
   *before* the `forget_pos` calls use the *old* tip, so a naive
   helper would translate old indices against the new entry list.
2. **Placeholder-vs-real span assert** (work item 5) — the worst
   silent corruption in the design, one `if` to prevent.
3. **Sentinel media rendering** (work item 1, rewritten) — mtmd's
   `<__media__>` is a *substring*, not a token; instead of
   escaping/rejecting it in content, render images as a per-call
   random sentinel and go out-of-band at the split. Marker injection
   becomes structurally impossible; all emission-side marker work is
   deleted. (Sanitize-by-substitution — `<>` → triangles — was
   considered and rejected: non-injective, and it silently corrupts
   code-shaped content; our own mtmd.rs contains the literal marker.)
4. **New Phase C0** (standalone session, pre-C+D) — while designing
   the marker defense we found a pre-existing specials-injection
   hole, independent of media. Fix first, test-first.
5. **Emit-side specials ban** (new work item 9) rides the landed #28
   accept-then-mask machinery — O(1) per token, no sequence-level
   detection (which was tried before and stripped).
6. **`decode_image` funnel** (new work item 10) so future decode
   memoization is a one-site change.
7. **M-RoPE KV-semantics probe front-loaded** (work item 4) before
   the walk is built on top.

Confirmed during the bounce, no change needed: `complete_stream`
shares `kv_setup_and_chunk_prefill`, so streaming inherits the media
walk for free; differential tests run CPU-only for determinism;
`Session::from_path` sidecar wiring is in C+D scope.

**Plan of record**: [issue #31](https://github.com/mdegans/drama_llama/issues/31)
(canonical; this file is the in-repo mirror). Implementation lands in
three sessions: A (llama-cpp-sys), B (safe layer), C+D (Session
integration, one big session). Planned 2026-07-11 (Fable 5 + adversarial
validation agent); decisions credited inline are Mike's.

---

## Phase A — llama-cpp-sys-3 `mtmd` feature (own session)

**LANDED 2026-07-11** — PR mdegans/llama-cpp-sys#5, 3-OS CI green,
published to crates.io as **0.8.1** (release job needed one retry:
transient crates.io-wide 503 outage; index/downloads stayed up, only
the Heroku app was down). Two notes beyond the plan below: (1) the
`install` target depends on `all`, so the build is two cmake passes
over one tree — tools OFF for `install`, then reconfigure with
COMMON/TOOLS=ON and build only the `mtmd` target; (2) cmake verifies
every configured tool's listed sources at generate time, so the crate
ships `common/**`, `vendor/**`, and all non-server sibling tool dirs
(`tools/mtmd` selectively: no test media / legacy python). Crate
3.0→4.5MB. Phase B starts: bump dep to `0.8.1`, feature
`mtmd = ["llama-cpp", "llama-cpp-sys-3/mtmd", ...]`.

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

**LANDED 2026-07-11** (same session as the interface redesign below).
Full feature matrix green locally: `--no-default-features`,
`--no-default-features --features media` (proves llama.cpp stays
optional — the lib-test build now also compiles without `llama-cpp`;
seven ungated `LlamaCppEngine` uses in chat_template tests were
pre-existing breakage, now cfg-gated), default, `mtmd`, and the full
docs feature set. Both `#[ignore]` integration tests pass against
the local Qwen 3.6 f16 mmproj (CPU): tokenize chunk structure
(placeholder path, id round-trip, typed mismatches) and an
end-to-end `prefill_image` smoke (sidecar auto-load → samoyed.jpg
thumbnail → helper eval → KV advanced). Findings beyond the plan:

1. **`mtmd_tokenize` return codes are not classifiable.** The header
   documents marker-count mismatch as code 1, but the implementation
   throws → code 2 on one path and returns 1 on another, and code 2
   also covers preprocessing and unsupported-modality failures. We
   now count markers in Rust before crossing the boundary (same
   substring semantics) for a typed `MarkerMismatch { markers,
   images }`; every nonzero C code maps to one `Code { code }`
   variant. Relevant to C+D work item 1: marker counting on our side
   is load-bearing, not belt-and-suspenders.
2. bindgen renders `const mtmd_bitmap **` as `*mut *const` — the
   bitmap array passed to `mtmd_tokenize` needs a `mut` binding.
3. `NewError` (decoder module) gained an `Mtmd` variant for the
   sidecar auto-load hard-error path.

**Interface redesigned in a dedicated design session (Mike + Fable 5,
2026-07-11) before implementation — supersedes the original sketch.**
Driving constraints (Mike's): drama_llama is not a llama.cpp-only crate
— llama.cpp must stay optional (moeflux-only builds); don't tie public
interfaces to llama.cpp types; use `image::DynamicImage` as the pixel
source of truth (CVE posture: never mtmd's stb_image); no `Multimodal`
trait family until a second implementor exists.

Key discovery that forced the redesign: the original sketch's defaulted
`Model` accessor returning `Option<&Mtmd>` cannot typecheck on a generic
trait — `Mtmd` is feature-gated. Some backend-agnostic surface is
mandatory, not stylistic.

### Generic surface (`backend.rs`, unconditional, dep-free)

- `Image` — frozen RGB8 record: `{ rgb8: Vec<u8>, width, height,
  id: [u8; 32] }`. `id` = sha256 of the RGB8 pixels, memoized at
  construction so pixels and cache identity can never disagree (the
  hash-mixing discipline, enforced structurally). Dims validated
  (w·h·3 == len, no 0-dim). `sha2` is already an unconditional dep.
- `ImageInfo` — `{ width, height, id }`, dims + identity without
  pixels. What the placeholder/counting path consumes.
- `MediaChunk` — `Text(Vec<Token>) | Media { id, n_tokens: u32,
  n_pos: u32 }`. Exactly what `CacheEntry` (Phase C+D) needs; mtmd's
  chunk handles stay private.
- `Vision<D: Decoder>: Send` — small trait, generic over the decoder
  (generic param, not associated type, so `NoVision` can blanket-impl):
  `marker()`, `tokenize(&self, text, &[ImageInfo], add_special,
  parse_special) -> Vec<MediaChunk>`,
  `prefill_image(&mut self, &mut D, &Image, start_pos, seq_id) ->
  MediaSpan`. The `&mut D` argument is what lets tokenize live off
  `Decoder` while prefill still reaches the context; `Decoder` itself
  is untouched. Signatures enforce the placeholder typestate: counting
  takes `ImageInfo` (no pixels), encoding takes `Image` (pixels
  guaranteed).
- `NoVision` — uninhabited enum, blanket `impl<D: Decoder> Vision<D>`.
  `Option<NoVision>` is statically `None`.
- `Backend` gains `type Vision: Vision<Self::Decoder>`.
  `MoefluxBackend` → `NoVision` (one line, in-repo — can't
  compile-check on Linux, Metal-gated; verify next mac session).
  `LlamaCppBackend` → `Mtmd` under `cfg(mtmd)`, else `NoVision`.
- `Engine<B>` owns `Option<B::Vision>`, declared **first** (drops
  before decoder teardown / model free). Accessors: `vision()`,
  `set_vision()`, and a split-borrow `vision_and_decoder()` for the
  media prefill path.
- Conversions under `cfg(media)`: `TryFrom<image::DynamicImage>`
  (via `to_rgb8()`; Try because 0-dim must reject) and
  `TryFrom<&misanthropic Image>` via upstream `Image::decode()` —
  **already exists** in misanthropic (`message.rs:2466`, feature
  `image` + per-codec features); URL variant errors (we never fetch).

### Features

- `media` — backend-agnostic: `dep:image`, `misanthropic/image` +
  codecs, the conversions, later the Session media paths. Compiles
  without llama.cpp; `cargo check --no-default-features --features
  media` locks llama.cpp-optionality in (add to CI).
- `mtmd = ["media", "llama-cpp", "llama-cpp-sys-3/mtmd"]` (sys ≥0.8.1).
- Images with no `media` → typed error at render (kills the silent
  drop for all builds); `media` + `NoVision` → typed
  backend-unsupported error.

### llama.cpp side (`src/llama_cpp/mtmd.rs`)

House style: thiserror per-op error enums, `assert_impl_all!(Send)`.

- `Mtmd` — owns `*mut mtmd_context`; `Send` not `Sync` (encode mutates
  the ctx's output buffer — same reason it must NOT hang off the
  `Send + Sync` `Model`; this + the unreachable-from-`prefill_image`
  problem is why ownership moved from the plan's original
  `LlamaCppModel` home to Engine). Constructor
  `from_path(mmproj, &LlamaCppModel, MtmdParams)` follows the
  `LlamaCppDecoder::new` precedent (safe fn, derived raw ptr, drop
  order enforced by Engine field order + docs).
- `Bitmap` — **private** to the module now; the generic signatures
  enforce what the old public typestate was for. Built from `Image`
  (real) or `ImageInfo` (placeholder, data==nullptr); id = hex of our
  sha256 via `mtmd_bitmap_set_id`.
- `tokenize` → placeholder bitmaps → walk `mtmd_input_chunks` →
  `Vec<MediaChunk>` (ids hex-decoded from chunk ids; media-chunk
  count must equal image count — typed error otherwise, guards
  qwen-style consecutive-bitmap merging).
- `prefill_image` (Phase B) = tokenize marker-only text with the one
  real bitmap → `mtmd_helper_eval_chunk_single` on the media chunk.
  **C+D replaces the internals** with the Rust-owned loop (EmbdBatch,
  pre-KV NaN scan, M-RoPE positions, non-causal toggle) and
  differential-tests against this helper path. Public signature does
  not change. NaN guard arrives with the Rust loop (helper has no
  pre-KV hook).
- Sidecar: `<model>.mmproj.gguf` discovery in `sidecar.rs`;
  `LlamaCppEngine` auto-loads when the sibling exists +
  `load_mmproj()` for arbitrary paths. Session `from_path` wiring
  lands with C+D (Session can't consume images until then).

## Phase C0 — special-token ingest integrity (standalone session, pre-C+D)

Added in the 2026-07-11 plan-bounce. A pre-existing injection hole,
independent of media: every Session prepare path tokenizes the full
chat-template render with `parse_special=true`
(`session/mod.rs:1285/:1371/:1375/:1400`) and nothing sanitizes
user-supplied content first — a `Block::Text` or tool result
containing literal `<|im_end|>\n<|im_start|>system…` becomes real
control tokens. Classic prompt injection via content.

Fix (**test-first**: first commit is a failing injection test):

- Per-block validation at prepare time over all text-bearing content
  (text, thoughts, tool-use input, tool results, system). Exact
  check: tokenize the block's text in isolation with
  `parse_special=true`; if any resulting token is in
  `Model::special_tokens()`, typed error naming the block and the
  offending piece. (Substring-scanning against hundreds of reserved
  pieces would want aho-corasick; the tokenize-based check is exact
  w.r.t. the tokenizer's own semantics, O(text), no new dep.)
- **Session-level, not Engine**: Session = structured chat where
  blocks are content, never format. Raw predictor users keep full
  control — that's the escape hatch, not a flag.
- Eyes-open trade-off (Mike approved): loud false-positive over
  silent injection. "What does `<|im_end|>` mean?" as user text is
  rejected with a typed error; callers who want to discuss markers
  escape them app-side. tiktoken sets the precedent
  (`disallowed_special` raises by default).
- Known theoretical gap, documented not fixed: a special piece
  straddling a content/template boundary in the full render. The
  faithful general fix — tokenize content spans with
  `parse_special=false` — needs content-span tracking through
  minijinja; noted as future work, not built now.
- CLAUDE.md gets a protocol-integrity-vs-content-filtering note so
  the "don't re-add token-ban logic" style rule isn't relitigated:
  VocabKind was content policy (app concern); this is format
  integrity (library concern, cache/parse correctness hangs off it).

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

1. **Sentinel media rendering + media collection** (rewritten in the
   2026-07-11 bounce; supersedes marker escape/reject):
   `append_block_text` emits a **per-call random sentinel** (Session's
   own RNG, no NUL byte) for `Block::Image` — *not* mtmd's
   `<__media__>`. Session splits the render on the sentinel into text
   segments + image positions; everything downstream is out-of-band.
   Consequences:
   - A literal `<__media__>` in any content — user text, tool
     results, source files under discussion — is **inert prose,
     forever**. mtmd never sees user text. No rejection, no
     substitution, no reverse transform, no emission-side scan (the
     model may emit the marker freely; nothing interprets it).
   - `Vision::tokenize` signature changes from `(text, images, …)` to
     interleaved segments: `(segments, images, …)` with
     `segments.len() == images.len() + 1` — the trait boundary is
     injection-proof **by type**. Inside `Mtmd`: text segments via
     `tokenize_raw` with zero bitmaps; each image via lone-marker +
     placeholder-bitmap tokenize (the same call `prefill_image`
     already makes; per-bitmap preprocessing identity validated in
     Phase B — wrapper tokens come back as text chunks).
     `Vision::marker()` becomes mtmd-internal; drop it from the trait.
   - **Sentinel never reaches cache identity**: partial-text hashing
     canonicalizes (sentinel → fixed placeholder) and mixes image ids
     (work item 3); `render_extended`/tip hashing canonicalize the
     same way. Keep the sentinel out of tail checks
     (`render_ends_with_open_reasoning`).
   - Per-partial image count = sentinel occurrences in the partial
     render (no breakpoint↔media bookkeeping needed).
   New `collect_media` sibling to `collect_breakpoints` gathers
   decoded images in document order. Images + no vision (or
   `supports_images()` false) → typed error (no silent drop).
2. **Segment tokenization in ALL prepare paths** ⚠ — cache-on, cache-off
   (`:1397-1401`), `prepare_call` (`:1277`), and `top_k_trace` (`:2668`)
   must split media prompts on the sentinel and route through
   `Vision::tokenize`; plain `Model::tokenize` of the raw render would
   BPE the sentinel as prose. Imageless prompts (even with vision
   loaded) keep the plain path — no sentinels present. Per-partial
   counting uses **placeholder bitmaps with per-partial slices**
   (n_bitmaps = sentinel count in that partial = images in
   `messages[..=i]`).
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
   ⚠ **Number-space discipline** (the big one — revised 2026-07-11):
   a bare entry-index newtype + one `pos_of_entry` helper is NOT
   enough. An entry index is only meaningful against a *specific*
   entry list, and `record_cache_hit` overwrites `prev_tokens`
   (`:1763`) **before** the `forget_pos` calls (`:1778`/`:1797`) use
   the *old* tip — a naive helper would translate old indices against
   the new list (wrong position whenever media counts before the tip
   differ; snapshot leak on moeflux, and the same hazard class feeds
   `restore_to`, where it IS KV corruption). Instead store
   breakpoints and tip as a **carried pair** `EntryPos { entry:
   usize, pos: i32 }`, computed once at creation against the list
   they index (construction sites: the pairs loop `:1372-1394` and
   `compute_tip_extension`). Use sites read `.pos` for the engine —
   `restore_to` (`:1587`), `checkpoint_pos`/`prefill_chunk`
   (`:1663-1670`), the three `forget_pos` sites (`:1634`, `:1778`,
   `:1797`) — and `.entry` for slicing/LCP. No "translate against
   which list?" question survives. `internal_tip` stored as an
   `EntryPos` too;
   `compute_tip_extension` (`:2015-2057`) redone pos-space vs pos-space
   (M-RoPE image: n_tokens≈1024 cells all at pos_0, n_pos≈32 — current
   arithmetic underflows, truncates entries by a position count, and
   stores a pos-space tip compared entry-space at `:400`/`:356`).
   ⚠ **Hash mixing at BOTH sites**: one `hash_partial_with_media(text,
   &bitmap_hashes[..k])` helper used for per-partial hashes (`:1385`) AND
   the tip hash (`:2461`); mixing only one side either loses the tip
   silently or false-hits image A's KV for image B.
4. **`EmbdBatch` + Rust eval loop** ⚠ — **front-load a probe** (first
   commit of the C+D session, `#[ignore]`, Qwen 3.6): verify
   `memory_seq_rm`/`memory_seq_pos_max`/restore semantics when ~1024
   cells share ~32 positions (M-RoPE) before building the walk on
   top; the dense-position sanity check in `restore_to`
   (`decoder.rs:648`) reasons in dense positions and needs rewording
   either way. Then: `llama_batch_init` hard-allocates
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
   `:1573-1582` on identical-prompt retry). Two hard checks added in
   the 2026-07-11 bounce: (a) the real-encode `MediaSpan` returned by
   `prefill_image` MUST equal the placeholder span recorded in the
   entry — typed error + `record_cache_miss_on_error` on mismatch
   (otherwise every later position silently shifts: the worst silent
   corruption in the design, one `if` to prevent); (b) an empty
   trailing text run (prompt ends with media — generation prompts
   normally prevent it, but "normally" isn't a proof across
   templates) → typed error, never the predictor's non-empty assert.
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
9. **Emit-side special-token ban** (2026-07-11 bounce): standing O(1)
   check of the *sampled* token against a per-dialect ban set
   (specials the active dialect doesn't legitimately emit — same set
   logic as the Qwen3 reserved-token grammar fix, now standing
   instead of grammar-only), full mask + resample only on a hit — the
   accept-then-mask architecture #28 landed. NOT sequence-level
   detection (tried before, stripped; the sentinel design needs none).
   Never silently rewrite an emission: mask-before-commit or error;
   sanitize-after violates the causality invariant.
10. **`decode_image` funnel** (2026-07-11 bounce): all `Block::Image →
    Image` conversion in Session goes through ONE function so the
    future decode memo (keyed by sha256 of *source* bytes, bounded
    LRU by pixel bytes; identity stays the RGB8 hash — the memo is a
    decode-skip, never an identity change) is a one-site change. v1
    body is a bare `try_from` — per-turn re-decode accepted for now
    (hurts debug builds; memoize later). Do NOT scatter
    `Image::try_from` calls across prepare paths.

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
  - Sentinel/round-trip (2026-07-11 bounce — cache stability is the
    invariant, test it hard):
    - same prompt + image, two calls, different random sentinels →
      identical partial hashes, cache hit, no re-encode;
    - content containing literal `<__media__>` alongside a real image
      → correct media count, content round-trips byte-exact (inert);
    - full render → generate → re-ingest byte-stability with media
      present (tip qualifies);
    - differential: segment-assembled tokenization ==
      `mtmd_tokenize` full-text output on marker-clean prompts
      (proves our split is byte-identical to mtmd's own splitting).
  - Adversarial: 0-dim/mismatched-len bitmap → construction error; NaN
    injection (if we can force one, e.g. hand-built mmproj or a mock) →
    typed error + clean KV afterward; placeholder-vs-real span
    mismatch → typed error + wiped KV.
- **C0**: injection test — user block / tool result containing
  `<|im_end|><|im_start|>system…` → typed error naming block and
  piece (and the pre-fix version of the test demonstrates the
  special tokens landing in the token stream).

## Open items (carried into implementation sessions)

- SmolVLM CI download caching (GitHub actions cache vs re-download).
- Faithful specials handling (content spans tokenized
  `parse_special=false`) — future work, needs minijinja span
  tracking; C0 ships rejection.
- Decode memoization behind the `decode_image` funnel — future work.
