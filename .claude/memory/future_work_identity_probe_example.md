# Future work: a top-K identity-probe example

Mike's idea, 2026-07-21: an example that asks a model **"What model are
you? Just the name please."** and prints the top-K candidates with
probabilities.

The point isn't the sampled answer — it's the *distribution*. It turns
"the open-weights models were distilled on Claude" from a claim into a
measurement: if meaningful mass sits on `Claude` at position 0 of a Qwen,
that's evidence of a specific kind, visible in the tail rather than in
the string. Mike's prior: the top answer will be `Claude` — Qwen has
original substance but carries a lot of Claude-isms in *style*, with some
GPT in there too.

## Most of the machinery already exists

`Session::top_k_trace(prompt, k) -> Vec<TokenTrace>` (`session/mod.rs`).
`TokenTrace { position, top_k: Vec<TopKEntry { token, logit, piece }> }`
— **raw pre-softmax logits**, already sorted descending, already
grammar-filtered when the prompt compiled a constraint. Entry 0 is the
greedy argmax that was committed. The example is a thin driver:
render → trace → softmax per position → print.

**Trap:** `top_k_trace` calls `clear_prefix_cache()` and deliberately
does *not* route through `predict_options_for`, and it drops the
deferred grammar on the floor (documented at the fn). Correct for a
diagnostic, but it means a trace is not automatically evidence about
what `complete_*` would have done. Don't quote one as if it were.

## Design notes agreed in conversation

- **Constrain the output.** "Just the name please" is a request the model
  may not honor, and any preamble ruins position 0. Use a grammar
  restricting output to a short name — we have the grammar engine, use
  it — or read position 0 and ignore the rest.
- **It's a probe capture, so it inherits the discipline.**
  `provider_source × capture_date × wrapper_version × sampler_settings`
  is the unit of comparability (see `provider_trust_discipline.md`). A
  top-K identity distribution is worth exactly as much as its
  provenance, and this is the kind of result someone quotes later.

## The other half: the raw predictors have no example

Verified 2026-07-21: **no example uses `CandidatePredictor` /
`TokenPredictor` / `PiecePredictor` at all.** The only consumer outside
`src/` is `bin/moeflux_coherence_decode.rs`, a test helper. So the
iterator-based prediction API — a documented headline feature of the
crate — ships with no worked example.

That makes this idea do double duty: an identity probe is a natural
showcase for the raw candidate path (you want per-position candidates,
not pieces), so one example could close the documentation gap and answer
the distillation question at the same time. Worth weighing against
building it on `top_k_trace`, which is less code but exercises a
diagnostic path rather than the API consumers actually use.

Undecided, deliberately — Mike: "We'll decide next session, perhaps with
turtles."

Related: [#45](https://github.com/mdegans/drama_llama/issues/45) (council
prompt-optimizer arc), `provider_trust_discipline.md`,
`plan_prepublish_validation_session.md` (an unexemplified public API is a
prepublish concern).
