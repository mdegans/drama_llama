# `/v1/models` — `Catalog<B>`, `FromPath::peek`, `Session::model_info`

**Landed 2026-09-18.** Read before touching `bin/blallama`'s listing
routes, `src/catalog.rs`, or proposing a cross-backend metadata API.

## What it is

- `GET /v1/models` and `GET /v1/models/{id}` on blallama, in
  misanthropic's own `Models` / `ModelInfo` types (`Client::models()`
  parses it — that's the integration test). `/api/tags` is now *derived*
  from the same entries rather than hand-built with blank fields.
- Every model on disk is listed with real metadata, **loaded or not**.
  llama.cpp's server (checkout `abd41adf5`) only has metadata for the
  loaded model; its router mode lists unloaded ones by name with a
  `// TODO: add other fields, may require reading GGUF metadata`
  (`tools/server/server-models.cpp:2067`). So we are ahead there. It
  also has no `/api/tags` — its ollama-shaped fields ride inside
  `/v1/models`.
- One mapping, `catalog::Advertised → ModelInfo`, feeds both
  `Session::model_info()` (loaded) and `FromPath::peek()` (weightless).
  `catalog::tests::llama_cpp::peek_agrees_with_load` pins that they
  agree; `SessionTransport::new` snapshots `model_info()` now instead of
  a basename-only stub.

## Decisions (Mike, same day — "no veto")

- `id` = directory entry name (what `resolve` matches and responses
  echo); `display_name` = GGUF `general.name` via the new
  `Model::title()` accessor, falling back to the id.
- **Ceilings = `min(served n_ctx, n_ctx_train)`**, Mike's call and what
  llama.cpp reports as `meta.n_ctx` (`n_ctx_slot()`,
  `server-context.cpp:4027`). Under-advertises RoPE-scaled models
  (Qwen3: 40960 native / 131072 under YaRN) — same as llama.cpp, the
  honest side. `peek_info` warns when the configured `n_ctx` exceeds the
  trained window. After a real load, `Catalog::refresh` overwrites the
  entry with the decoder's padded `n_ctx`.
- Capabilities: `structured_outputs` always; `image_input` iff the mmproj
  sidecar resolves (`sidecar::mmproj_path`) / a `Vision` loaded;
  `thinking` iff `dialect.reasoning.mode != ReasoningMode::None`, types
  `{enabled}`. Everything else false. Observed on the dev box: the Qwen
  *Base* GGUF and cogito-32b correctly list `thinking: false`.
- `created_at` = file mtime (the ollama `modified_at` analog).
- A failed peek is **not cached**: degraded name-only entry + `warn!`, so
  a corrupt file stays listable and the load reports the real error.

## Why the peek is a `vocab_only` load, not a gguf reader

"Use what they expose" (Mike): `LlamaCppModel::from_file` already takes
raw `llama_model_params`; `vocab_only = true` gives a real model (vocab,
meta, template source, `n_ctx_train`) with no tensors mapped and no GPU
touched, so the existing `Model` trait answers everything and there is
no second FFI surface. moeflux's `MoefluxModel::from_mlx_dir` was already
tokenizer + JSON.

**Cost, measured (debug build, warm page cache):** ~1.3 s per model on
the libllama side (tensor-metadata walk + tokenizer build; 0.7 s for a
32B dense, 1.4 s for gpt-oss), ~7 ms for our dialect analysis. Ten models
= 12–23 s cold. Hence the cache and the startup warm-up — blallama calls
`catalog.models()` on the blocking pool right after bind, stage-logged
(`read_model_metadata` / `model_metadata_read` with `elapsed_ms`, #95
style: a wedge names its file). If that cost ever matters on the
request path, `gguf_init_from_file(no_alloc)` is the lighter upstream
API (`ggml/include/gguf.h:79`; the pattern `tools/mtmd/clip.cpp:1174`
uses for `mtmd_get_cap_from_file`) — but it means re-implementing
bos/eos piece lookup and every accessor. Not worth it until measured.

## Concurrency shape

`Catalog` is sync (std mutexes); blallama wraps calls in
`spawn_blocking_or_bust`. Two locks: the cache's (held only to read or
insert) and a **peek lane** that serializes misses, with a re-check
after waiting so a second miss for the same model reuses the first's
result. Cached reads never wait on a peek in flight. This was Mike's
"separate cache" note — irrelevant for Agora (one client) but a
multi-client server shouldn't queue listings behind a vocab load.
`/v1/messages` only needs `Catalog::list` + `resolve` (names, no lock),
and `load_session`'s `refresh` takes the cache lock briefly, so requests
overlap the warm-up freely (the blallama suite exercises that overlap).

## Not verified on device

The moeflux `peek` compiles under `just permutations` but no moeflux
model was mounted this session. Its `n_ctx` is
`moeflux::riir::variants::MAX_SEQ_LEN` (1 048 576), same constant the
decoder reports; `min` with `config.json`'s `max_position_embeddings`
makes the advertised ceiling sane. Check it the first time `just test
moeflux` runs after this.

## Known accepted divergence

A `*.template.jinja` sidecar that fails to *compile* is skipped by the
load (falls back to embedded) but analyzed to the default dialect by the
peek — `thinking` could differ for that one pathological case. Fixing it
means threading the compile result through `analyze_dialect_source`;
not done because the load-side warning already names the broken sidecar.
