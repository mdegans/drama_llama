# Grammar + reserved-token loop (Qwen3 finding)

Captured 2026-04-25 during v0.8.0 blallama-on-moeflux smoke testing.

## Symptom

Grammar-constrained generation (output_config with JSON schema) on
Qwen3.5-A17B via blallama produced valid JSON in the response body
but `output_tokens=512`, `stop_reason=max_tokens`, `tok_per_sec=1.4`.
The JSON itself was ~30 tokens (~73 bytes); the remaining ~480
tokens were emissions in the 248000-248320 range that decoded to
empty strings. Histogram showed scattered IDs each at count ~3, not
a single token in a loop.

## Root cause (90% confidence)

Three-layer interaction:

1. **Model logit distribution post-JSON.** After the JSON closes,
   Qwen3 puts non-trivial probability on reserved/empty-piece
   vocab slots (the high-vocab tail Qwen carves out for future-use
   tokens) and *less* on `<|im_end|>` (EOS). Probably a training
   artifact — the chat template structure doesn't strongly
   condition "JSON closes → EOS next" the way a hand-trained
   tool-calling pipeline would.

2. **Tokenizer empty-piece behavior.** Reserved vocab IDs decode
   to empty strings via the HF tokenizer's
   `decode(skip_special_tokens=false)`. They're allocated vocab
   slots without registered text content. `Model::special_tokens()`
   only enumerates *named* entries (im_start, im_end, endoftext,
   tool_call, etc.) — it does NOT include the reserved tail.

3. **Grammar byte-stream filter.** GBNF / JSON grammars accept or
   reject candidates by their byte contribution. Empty-piece tokens
   contribute zero bytes — *every* grammar trivially accepts them
   regardless of state. EOS (`<|im_end|>`) decodes to non-empty
   text that doesn't extend valid JSON, so grammar rejects it.

Result: of the model's preferred post-completion tokens, the
grammar accepts only the reserved ones (zero bytes pass). The
"all-candidates-rejected → force EOS" fallback in
`sample/grammar.rs:1814` never fires because empty-piece tokens
keep the kept set non-empty.

## Cross-backend evidence

Tested A3B (Qwen3.6-35B-A3B) on **both** moeflux and llama.cpp
GGUF Q4_K_M paths with identical prompts. **With our fix in
place**, both backends produce equivalent token counts, wall
clock (modulo moeflux's disk-streaming vs llama.cpp's RAM-loaded
weights — llama.cpp slightly faster on the small model), and
correct termination. This rules out moeflux's known upstream
routing bug as the cause: the routing bug is moeflux-specific;
identical fixed behavior across backends means the issue lives
at the shared model/grammar interaction layer.

**Unverified**: we did NOT compare A17B cross-backend — A17B
doesn't fit on llama.cpp's path on the test hardware (96GB
MacBook). The hypothesis transfers because both Qwen3 variants
share the tokenizer + chat template structure, but treat that as
inference, not direct evidence. We also did NOT compare A3B
*without* the fix on llama.cpp; we don't know whether llama.cpp
naturally avoided the loop or hit it identically pre-fix. Both
explanations are consistent with the data.

## Fix landed (v0.8.0)

Three layers of defense, all in commits between `e6ea365` and
`59391ab`:

- **`SamplingMode::Deny { range }`** — sample-time mask over a
  token-id range. Computed once per Session via a bounded scan
  from the highest vocab id downward (terminates after 64
  consecutive content tokens), yielding `Some(min..n_vocab)` or
  `None`. Prepended to the modes chain in `prepare_call`. Removes
  reserved tokens from the candidate set entirely so grammar's
  "all-rejected → force EOS" path can fire.
- **`Model::extra_eos_tokens()`** — additional EOS-like tokens
  beyond `eos`/`eot`. MoefluxModel exposes the tail of the
  `eos_token_id` array (Qwen3's `<|endoftext|>`). Wired into
  `add_model_stops` and Session's `eos_pieces` filter set.
- **`any_grammar_complete()` break in `run_call`** — captures
  Arc<Mutex<State>> handles for every grammar/json mode (incl.
  deferred), polls `is_complete()` after each piece, breaks the
  loop the moment the matcher accepts. Avoids waiting for EOS at
  all when grammar is satisfied.

## Architectural follow-up (deferred)

The principled fix is in the grammar matcher itself: when a
matcher reaches its accept state, it should reject empty-piece
tokens (or, equivalently, recognize EOS as a valid terminal in
the grammar source). Either:

- **Matcher level**: `grammar_filter` / `json_filter` checks "if
  matcher is_complete AND candidate piece is empty, reject."
  ~1 hour change in `sample/grammar.rs` and `sample/json.rs`.
- **Source level**: extend GBNF to support EOS-as-terminal,
  `root := <body> EOS`. Multi-day work — parser, matcher, and
  output_config compiler all need to know.

Neither is urgent. Current three-layer defense is sufficient for
all observed bug classes; the architectural cleanup buys
elegance, not behavior.

## What NOT to revisit without new evidence

- Moeflux upstream routing bug as cause — ruled out by A3B
  cross-backend test.
- Per-token wall-clock concerns — prefill dominates for cold
  cache + large input (534 tokens × ~0.4s = ~210s). Generation
  rate matches Mike's expected ~2.7 tok/s warm, ~1.4 tok/s cold.
- Whether reserved tokens are "truly" empty-piece — confirmed by
  histogram diagnostic. Not a tokenizer quirk; they're vocab
  slots without registered text content.
