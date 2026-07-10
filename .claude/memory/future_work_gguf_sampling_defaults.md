# Future work: seed sidecar sampling defaults from GGUF metadata

**Date:** 2026-07-10. Qwen3.6-35B-A3B (Unsloth) ships
`general.sampling.temp = 1.0`, `general.sampling.top_k = 20`,
`general.sampling.top_p = 0.95` in GGUF metadata — the vendor's
recommended sampling, machine-readable. drama_llama's sidecar
(`sampling.toml`) currently writes *crate* defaults on first load, so
models run off-recommendation until someone edits the sidecar.

Sketch: when writing the initial sidecar (and/or when no sidecar
exists), read `general.sampling.*` via `Model::get_meta` and use those
values where present, falling back to crate defaults. Keys are
optional and newer-convention — most GGUFs won't have them.

Also confirmed while checking: there are NO thinking-related GGUF
keys; thinking defaults live entirely in the chat template's Jinja
(`enable_thinking` on-when-unset, `preserve_thinking` gate). The
template is the metadata for thinking behavior.
