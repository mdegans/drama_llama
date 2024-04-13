# `drama_llama`

`drama_llama` is yet another Rust wrapper for [`llama.cpp`]. It is a work in progress and not intended for production use.

## Supported Features

- Iterators yielding tokens and pieces.
- Stop criteria at regex, token sequence, and/or string sequence.
- Metal support. CUDA may be enabled with the `cuda` and `cuda_f16` features.
- Rust-native sampling code. All sampling methods from llama.cpp have been translated.
- N-gram based repetition penalties with custom exclusions for n-grams that should not be penalized.
- Support for N-gram blocking with a default, hardcoded blocklist.

<!-- The code has been rewritten not because I think I can do better, but because I wanted to understand it, and translation forces that. Usually. There are likely bugs. -->

## Contributing

- Code is poetry. Make it pretty.
- Respect is universal.
- Use `rustfmt`.

## Roadmap

- [ ] LLama3 support. Coming soon after [`llama.cpp`] itself supports it.
- [ ] Tiktoken as the tokenizer instead of llama.cpp.
- [ ] Reworked, functional, public, candidate API
- [ ] Candidate iterator with fine-grained control over sampling
- [ ] Grammar constraints (maybe or maybe not [`llama.cpp`] style)
- [ ] Async streams, better parallelism with automatic batch scheduling
- [ ] Backends other than [`llama.cpp`] (eg. [MLC](https://github.com/twiceyuan/mlc-llm-llama2), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [Ollama](https://github.com/pepperoni21/ollama-rs))

[`llama.cpp`]: https://github.com/ggerganov/llama.cpp
