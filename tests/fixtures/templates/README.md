# Vendored chat-template fixtures

The `.jinja` files here are vendored from
[llama.cpp](https://github.com/ggml-org/llama.cpp)
`models/templates/` at commit 52b3df00 (b9754), MIT license, for
testing the dialect analyzer against the same corpus upstream pins
its auto-parser expectations on (`tests/test-chat-auto-parser.cpp`).

The Gemma 4, gpt-oss and Qwen3.6/3.8 templates (stock dumps and their
cache-stable patches) were promoted to shipped artifacts in
crate-root `templates/` — they back the `baked` registry
(`src/baked.rs`, issue #88). Provenance and patch notes moved to
`templates/README.md`; the test helpers resolve fixture names from
both locations.
