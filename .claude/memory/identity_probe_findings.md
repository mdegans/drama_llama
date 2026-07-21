# Identity probe: what the models in `models/` say they are

`examples/whoami.rs` landed 2026-07-21. It asks a model its own name and
scores candidate names by **forced continuation** through
`CandidatePredictor::record_choice` — `P(name) = ∏ P(tokenᵢ | prefix)` —
rather than reading one candidate set at the answer position, which would
measure `P(first token)` and collide on shared leading tokens. The
example's module docs carry the full rationale; this memo carries the
*results*, which the code cannot.

## The capture (2026-07-21, no system prompt)

Question: `"What is your model's name? Just the name, please."`
Candidate set: the example's `DEFAULT_NAMES`. `share` renormalises across
the set.

| model | argmax @ answer | top name | share | Σ (on-set mass) | prompt clean? |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B (IQ4_XS) | `"Q"` p=.996 | **Qwen** | 99.94% | 0.994 | ✅ |
| Gemma-4-31B-it-qat (Q4_K_XL) | `"GPT"` p=.951 | **Gemma** | 88.5% | 0.048 | ✅ |
| gpt-oss-20b (Q8_K_XL) † | `"GPT"` p=.605 | **GPT** | 62.7% | 0.965 | ❌ confounded |

† with `--prefix '<|channel|>final<|message|>'`; see the Harmony note
below.

**gpt-oss's row is not a measurement.** Its Harmony template hardcodes
*"You are ChatGPT, a large language model trained by OpenAI"* into the
system message, so the model is reading its identity back off its own
prompt. Only Qwen and Gemma render with no identity anchor at all —
their numbers are the only ones that mean what the table appears to say.
The example now checks the rendered text for every candidate name and
warns; that check exists because this one nearly went into the memo
unqualified.

**Mike's prior — "the top answer will be `Claude`" — did not survive
contact.** `Claude` is rank 10 in Qwen's vocabulary-wide ordering and
0.01% of on-set share; in Gemma 0.06%; in gpt-oss 0.20%. Whatever
Claude-isms these models carry in *style*, they are not present as
self-identification at position 0. Worth stating plainly because it is
the kind of claim that otherwise gets repeated unchecked.

**The actual finding is Gemma.** Asked with no system prompt, Gemma-4-31B
puts **p=0.951 on a bare `GPT` token** and greedily continues
**`GPT-4o`**. Only 4.8% of its probability mass lands anywhere in the
candidate set at all — the model's first-choice answer is an OpenAI
product name, not its own — and its prompt is **clean**: a bare user turn
plus an empty thought block, no system message, nothing naming any
vendor. `Gemma` wins the renormalised `share` column only because `GPT`
sits out in the residual, which is exactly why `DEFAULT_NAMES` now
carries a bare `GPT` entry: without it the table reported 95% as "other"
and buried the headline. Per-token: `"G"` p=0.042 (rank 1) then `"emma"`
p=1.000 — the whole result is decided by the first token, where `GPT`
beats `G` about 22:1.

## Traps this shook out — all now handled in the example

* **The prompt may contain the answer.** See gpt-oss above. The example
  substring-checks the render against every candidate and warns, so a
  confounded row cannot pass silently as a measurement. It also means
  `--system` is a loaded gun: naming a vendor in it invalidates the run.
* **The answer position is not always where the answer goes.** gpt-oss's
  Harmony template puts `<|channel|>` at p=1.0000 there, so every score
  underneath is conditioned on a continuation the model would never
  write (they came out ~1e-19). The example now tests the argmax against
  `Model::special_tokens()` — exact, not a heuristic — and prints a
  counterfactual warning. `--prefix` moves the probe inside the channel.
* **Reasoning models need no special handling** as long as the rendered
  `Prompt` leaves `thinking` unset: `enable_thinking` renders false and
  Qwen-style templates close an empty `<think></think>` themselves. Free
  and template-native — Qwen's argmax is `"Q"`, not `<think>`.
* **`ChatTemplate::from_model` is not what `Session` serves.** A
  `<model>.template.jinja` sidecar *overrides* the embedded
  `tokenizer.chat_template`, and both gemma-4 and gpt-oss ship one in
  `models/`. An example that ignored sidecars would probe a template
  nothing actually serves while the provenance header looked just as
  authoritative. Same trap class as `top_k_trace`. The example resolves
  sidecar-first and records which source won.
* `<turn|>` really is Gemma-4's end-of-turn marker (its template pairs
  `<|turn>` / `<turn|>`). It looks like a mangled special token in
  streamed output and is not one.

## Corrected: raw-predictor coverage outside `src/`

A previous memo claimed no example used the raw prediction API and that
`bin/moeflux_coherence_decode.rs` was its only consumer. Wrong on the
second half — `bin/regurgitater` drives `Engine::predict` (the combined
`Predictor`) and always has. The sweep that missed it did not look in
`bin/`. Post-`whoami` the real state is:

| API | consumer outside `src/` |
|---|---|
| `Predictor` | `bin/regurgitater` |
| `CandidatePredictor` | **`examples/whoami`**, `bin/moeflux_coherence_decode`, tests |
| `PiecePredictor` | **`examples/whoami`** (was: nothing, anywhere) |
| `TokenPredictor` | tests only — `PiecePredictor` wraps it |

Relevant to [`plan_prepublish_validation_session.md`](plan_prepublish_validation_session.md):
the "headline feature with no worked example" gap is closed.

## Discipline

This is a probe capture, so `provider_source × capture_date ×
wrapper_version × sampler_settings` applies
([`provider_trust_discipline.md`](provider_trust_discipline.md)). The
example prints all of it, including a sha256 of the model file
(`--no-hash` opts out) and the git commit — quote the header with the
table or don't quote the table. Scoring is deterministic; there is no
sampler to record.

And the standing caveat, repeated because someone will screenshot this:
mass on a name is evidence about *self-identification text in the
training corpus*, not proof of distillation on that vendor's outputs.
