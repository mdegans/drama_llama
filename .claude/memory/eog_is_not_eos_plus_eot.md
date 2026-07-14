# EOG is not `eos ∪ eot ∪ extras` — ask `Model::eog_tokens()`

**Landed 2026-07-14.** Root cause of six of the seven failures in the
post-#30/#31 suite (`session_gptoss` ×6, all of them).

## The trap

libllama has two *different* notions that look like one:

- `special_eot_id` — auto-detected **by token text**. The match list in
  `llama-vocab.cpp` includes `"<|end|>"` (it's there for Phi-3).
- `special_eog_ids` — the set generation actually stops on
  (`llama_vocab_is_eog`). Built by text-match, then **fixed up per
  model family**.

For gpt-oss/Harmony the fixup *removes* `<|end|>` from `special_eog_ids`
(so the model can close an analysis channel and keep going) and **does
not touch `special_eot_id`**. Upstream stays self-consistent because its
generation loop only ever asks `llama_vocab_is_eog`. Ours didn't:

```
EOS token = 200002 '<|return|>'
EOT token = 200007 '<|end|>'      <- NOT in the EOG set
EOG token = 199999 '<|endoftext|>' / 200002 '<|return|>' / 200012 '<|call|>'
```

We reconstructed the stop set by hand — `{eos} ∪ {eot} ∪ extra_eos` — in
**seven** places, which dragged `<|end|>` back in. Same bug, two faces:

- **Unconstrained** (`auto_tool_choice`, `prefix_cache_survives_final_turn`):
  generation stops dead at the first `<|end|>`. The turn comes back as a
  lone `Block::Thought` — analysis only, no final message, no tool call.
- **Under grammar** (`forced_call`, `emission_round_trips`,
  `prefix_cache_survives_tool_turn`): the same set is *masked* while the
  grammar is incomplete, so the model cannot emit the token it needs to
  close the channel. It rambles to `max_tokens` — the tell is 95-second
  tests and output like *"Let's do. We'll call. We'll output. We'll
  respond."* on repeat.
- **Latent, would have bitten next**: `<|end|>`'s piece was also in the
  strip-from-surfaced-output set and in `trim_eos`. Fixing only the
  first two would have handed the dialect parser an unterminated
  reasoning block that swallowed the rest of the turn.

## The fix

`Model::extra_eos_tokens()` is **gone**. `Model::eog_tokens()` returns
libllama's `special_eog_ids` verbatim and is the single authority for
both "does this token end generation" and "may it be masked mid-
constraint". No call site unions anything. Non-llama.cpp backends report
their own truth (moeflux composes it from config, where `eot` genuinely
does end a turn).

`gptoss_eog_token_set` now pins `eot == <|end|>` **as a deliberate
upstream quirk** and asserts `<|end|> ∉ eog_tokens()`. If upstream ever
changes it, a test says so instead of a model going mute.

## The lesson worth keeping

The knowledge was **already in the codebase**. `dialect::harmony::END`'s
doc comment said, verbatim, *"NOT end-of-generation — libllama removes
it from `special_eog_ids` for Harmony vocabs."* It was correct, it was
written by the same session that shipped the bug, and it never reached
the seven places that built the stop set. A fact documented next to a
constant does not propagate to the code that needs it; a fact expressed
as **the only available API** does. That's why the fix is a deleted
method and not a comment.

Corollary for future backends and vocab quirks: when llama.cpp exposes
both a *label* (`eot`, "end of turn") and a *predicate*
(`llama_vocab_is_eog`), the predicate is the contract. The label is a
hint about the vocab, not about behavior.

See also [[plan_tool_dialects]] (#30 Phase G, where Harmony landed) and
`tests/session_gptoss.rs`.
