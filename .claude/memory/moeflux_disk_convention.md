# moeflux on-disk convention (drama_llama side)

Captured 2026-04-25 alongside the v0.8.0 Session-generic landing.
Forward-looking — current MLX exports use a flat sibling layout;
upstream moeflux conversion scripts will eventually consume / produce
this layout directly.

## Layout

`MoefluxEngine::from_path(parent: &Path)` (in `src/moeflux/engine.rs`)
expects:

```
<parent>/
├── mlx/             ← MLX export: tokenizer.json, config.json,
│                       tokenizer_config.json, chat_template.jinja
├── artifacts/       ← model_weights.bin, model_weights.json, vocab.bin
└── root/            ← packed_experts/ subdir
```

Plus runtime parameter defaults:
- `experts_per_tok = 8`  (Qwen3 MoE 4-bit setup)
- `use_2bit = false`

Power-user override: `MoefluxEngine::from_paths(...)` takes all five
args explicitly. Both constructors stay supported.

## Why these names

Mike's original sketch (April 2026) used a hyphen-suffix flat sibling
layout:

```
qwen3-6-35b-a3b-mlx-4bit/
qwen3-6-35b-a3b-artifacts/
qwen3-6-35b-a3b-root/
```

The blallama discovery flow is "ls models dir, pick by single path,
load." For symmetry with `LlamaCppEngine::from_path(file: PathBuf)`,
both backends needed a single-path entry point. Restructuring under a
parent directory (`qwen3-6-35b-a3b/{mlx,artifacts,root}/`) is the
minimal change that keeps blallama's UX consistent across backends.

Names dropped redundant suffixes (`mlx-4bit` → `mlx`, etc.) since
the parent already carries the model identity. Quantization variants
can move under `mlx/` (e.g. `mlx-2bit/` if we ever expose 2-bit
externally) without breaking the convention.

## Migration path

1. **Done in v0.8.0:** drama_llama exposes `MoefluxEngine::from_path`
   that consumes this layout. blallama's `--backend moeflux` requires
   it.
2. **Pending upstream:** the moeflux conversion script (and
   `extract_weights.py` / `repack_experts.py`) currently produces
   the flat sibling layout. When it gets next-touched, it should
   produce the parent-with-subdirs layout directly so callers don't
   need to mv files around.
3. **Pending physical move on Mike's box:** the existing
   `/Volumes/Temp Backup/models/moeflux/qwen3-6-35b-a3b-*` dirs need
   to be reorganized under `/Volumes/Temp Backup/models/moeflux/qwen3-6-35b-a3b/{mlx,artifacts,root}/`
   before blallama's moeflux path can smoke-test against them.
   `tests/moeflux_smoke.rs` and `tests/cross_backend.rs` still hit
   the flat layout via `MoefluxEngine::from_paths` — those are
   override callers and don't need migration.

## Decision: not enforced upstream-first

Mike's call (April 2026): "Land upstream folder change first unless
it's more than just a change to the entrypoint. If it's a lot of
work, #1 wins." The drama_llama-side wrapper is ~30 lines and
doesn't depend on a moeflux release. We landed the wrapper now and
captured the convention here so the upstream change, when it
happens, lines up with what consumers expect.
