# `regurgitater`

Is a tool to get language models to regurgitate memorized content. Generally this is a mistake, as in a "oops we trained on your data without paying you and it's legal nya nya nya" kind of mistake that happens all too frequently in the "AI" industry.

The tool works by, for a given text, submitting the beginning of the text as context and comparing the generated completion to ground truth. Greedy sampling is used so this generation is deterministic. In other words, you will not have to repeat the process 10,000 times to get the results you're after.

## Usage

````bash
$ cargo run --features="webchat cli stats" --bin regurgitater -- --model ~/models/llama/70b/llama-2-70b.Q6_K.gguf
```cd ~

## Faq

- **What is greedy sampling?** When you submit some tokens to a language model, you get back a probability distribution of all possible tokens for the one next token. Greedy sampling always picks the most likely token from this list (as opposed to, for example, throwing some digital dice and choosing from the top k most probable tokens).
- **Are you aware the name is spelled wrong?** Yes. It's funny because tater ha ha.
- **Did you paint the vomiting llama?** No. That was Bing Copilot and Dall-E 3.
````

## Known Issues

- When using LLaMA 3, `--vocab unsafe` should be passed as a command line option
  however, keep in mind that there is out output sanitization or vocabulary
  restrictions.
