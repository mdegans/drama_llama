# `drama_llama`

![llama with drama mask logo](logo.svg)

`drama_llama` is yet another Rust wrapper for [`llama.cpp`]. It is a work in progress and not intended for production use. The API _will_ change.

For examples, see the `bin` folder. There are two example binaries.

- **[Dittomancer](bin/dittomancer/README.md)** - Chat with well represented personalities in the training.
- **[Regurgitater](bin/regurgitater/README.md)** - Test local language models for memorized content.

## Supported Features

- LLaMA 3 Support.
- Iterators yielding tokens and pieces.
- Stop criteria at regex, token sequence, and/or string sequence.
- Metal support. CUDA may be enabled with the `cuda` and `cuda_f16` features.
- Rust-native sampling code. All sampling methods from llama.cpp have been translated.
- N-gram based repetition penalties with custom exclusions for n-grams that should not be penalized.
- Support for N-gram blocking with a default, hardcoded blocklist.

<!-- The code has been rewritten not because I think I can do better, but because I wanted to understand it, and translation forces that. Usually. There are possible bugs. Much of the sampling code is untested in generation, but also covered by unit tests. -->

## Contributing

- Code is poetry. Make it pretty.
- Respect is universal.
- Use `rustfmt`.

## Roadmap

- [x] Candidate iterator with fine-grained control over sampling
- [ ] Examples for new Candidate API.
- [ ] Support for chaining sampling methods using `SampleOptions`. `mode` will
      become `modes` and applied one after another until only a single
      Candidate token remains.
- [ ] Common command line options for sampling. Currently this is not exposed.
- [ ] API closer to Ollama. Potentially support for something like `Modelfile`.
- [ ] Logging (non-blocking) and benchmark support.
- [ ] Better chat and instruct model support.
- [ ] Web server. Tokenization in the browser.
- [ ] Tiktoken as the tokenizer for some models instead of llama.cpp's internal one.
- [ ] Reworked, functional, public, candidate API
- [ ] Grammar constraints (maybe or maybe not [`llama.cpp`] style)
- [ ] Async streams, better parallelism with automatic batch scheduling
- [ ] Backends other than [`llama.cpp`] (eg. [MLC](https://github.com/twiceyuan/mlc-llm-llama2), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [Ollama](https://github.com/pepperoni21/ollama-rs))

## Known issues

- With LLaMA 3, safe vocabulary is not working yet so `--vocab unsafe` must be
  passed as a command line argument or `VocabKind::Unsafe` used for an `Engine`
  constructor.
- The model doesn't load until genration starts, so there can be a long pause
  on first generation. However because `mmap` is used, on subsequent process
  launches, the model should already be cached by the OS.

[`llama.cpp`]: https://github.com/ggerganov/llama.cpp

## Generative AI Disclosure

- Generative, AI, specifically Microsoft's Bing Copilot, GitHub Copilot, and
  Dall-E 3 were used for portions of this project. See inline comments for
  sections where generative AI was used. Completion was also used for getters,
  setters, and some tests. Logos were generated with Dall-E and post processed
  in Inkscape.
