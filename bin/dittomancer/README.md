# Dittomancer

Dittomancer is a tool to summon simulacra of living, dead, real or fictional
entities well represented in language models. It's similar to other local
language model tools that prompt models for chat, but with a very different
intent.

## Requirements

- Read `fred_rogers.toml` for an example of how to use the tool and create your
  own `.toml` file to your needs.
- You will need a `.gguf` format model [such as
  LLaMA 2](https://huggingface.co/TheBloke/Llama-2-70B-GGUF). Foundation models
  (not tuned) will likely work better for this purpose unless the were
  specifically tuned on the character in question.
- Read the root [`TERMS_OF_USE.md`](../../TERMS_OF_USE.md). You must agree with
  the terms to use this tool.

## Running

From the crate root, run:

```bash
$ cargo run --features="webchat cli" --bin dittomancer -- --model models/model.gguf --prompt bin/dittomancer/fred_rogers.toml
```

Finally, go to the link shown on a line like

```text
🚀 Rocket has launched from http://127.0.0.1:8000
```

The binary can also be installed with

```bash
$ cargo install --features="webchat cli" --path . --bin dittomancer
```

## Faq

- **Did you come up with the name?** No. The name is taken from [this
  generation](https://generative.ink/artifacts/hpmor-325/variant_extrusion/#variant_extrusion_start).
  It's not intended to endorse Eliezer Yudkowsky, Less Wrong, or the author of
  the series which shall not be named. It's simply a better, yet still
  imperfect, descriptor than "necromancer".

  > _A Dittomancy book is able to hook into your own spreads of probability, and
  > guide the future that you, yourself, are most likely to create. Do you
  > understand? A Dittomancy copy of a book exists in an unusual state at all
  > times; it is a superposed state until the moment one reads it, at which time
  > it becomes correlated with the reader’s mind, the superposition collapsing
  > onto a particular branch of possible worlds, which thence comes to pass. -
  > GPT_

- **Don't you think this a bad idea?** Probably. Oh yes very much so. The whole
  idea of generative AI is of questionable benefit to humanity. That being said
  others are alredy doing this, thank you Meta, and for every Charles Manson,
  there are decent contributions to humanity whose ideas do deserve to spread.
- **Don't you think Fred Rogers would hate this?** Absolutely. He also hated TV.
- **Doesn't this violate the LLaMA "Responsible Use" document?** _Possibly_, but
  Meta doesn't enforce it, I never accepted it, and this utility does not bundle
  LLaMA. Technically it is model agnostic. I will care when Meta starts to care
  about flagrant
  [bigotry](https://huggingface.co/datasets/cognitivecomputations/open-instruct-uncensored/blob/main/remove_refusals.py#L17)
  rampant in the crypto-bro dumpster fire that is the "open source" language
  model community.

## Known Issues

- The responses are not streamed to the client, so they can take a while
  depending on model and system. PRs welcome to fix this. The `regurgitater` bin
  has an example of how to do it. For the moment, the output is streamed to the
  command line only.
- When using LLaMA 3, `--vocab unsafe` should be passed as a command line option
  however, keep in mind that there is out output sanitization or vocabulary
  restrictions.

## Roadmap

- [ ] Updated Fred Rogers toml where Charlie Rose take a call from the audience
      and we "patch the chat through" at that point. This way the human does not
      have to play Charlie Rose. The setting can be reframed as a recently
      discovered outtake.
- [ ] Sampling Options. Currently "Locally Typical" sampling is used and the
      Generation options are not available to be set. These options likely
      belong in the `.toml` file itself and/or as command line options.
