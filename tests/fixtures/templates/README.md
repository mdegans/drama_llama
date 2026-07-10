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

`gemma4-cache-stable.jinja` is drama_llama's cache-stability patch of
`gemma4-gguf.jinja`: model turns re-render the thinking channel the
model actually generated against (real reasoning, gated by
`preserve_thinking` for aged turns; the empty
`<|channel>thought\n<channel|>` scaffold otherwise), so the KV cache
stays a byte prefix of the next render across tool turns. Deploy it
as a `<model>.template.jinja` sidecar next to the GGUF. Everything
outside the thinking-channel block is byte-identical to
`gemma4-gguf.jinja`.
