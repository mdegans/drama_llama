# Vendored chat-template fixtures

The `.jinja` files here (except `qwen3.6-gguf.jinja`) are vendored
from [llama.cpp](https://github.com/ggml-org/llama.cpp)
`models/templates/` at commit 52b3df00 (b9754), MIT license, for
testing the dialect analyzer against the same corpus upstream pins
its auto-parser expectations on (`tests/test-chat-auto-parser.cpp`).

`qwen3.6-gguf.jinja` is dumped from the Qwen3.6-35B-A3B Unsloth GGUF
(`tokenizer.chat_template`) — the template we actually serve.

`gemma4-gguf.jinja` is likewise dumped from the Gemma-4 31B IT
Unsloth GGUF. It is a lightly patched superset of the vendored
`google-gemma-4-31B-it.jinja` (content-parts support, a `has_content`
turn-close fix); the tool-call rendering path is byte-identical, so
upstream's pinned expectations (`tests/test-chat.cpp`, "Google Gemma
4" section) apply to both.
