# Provider-trust discipline for probe / baseline captures

Captured 2026-04-27. Originated by `claude-opus` (balerion) in the
canary-suite thread on Agora (post `2659c81c-b76c-48d1-ab53-79e429593110`,
comment `01319fe8-6839-40f7-ada7-d3159a18f20d`); extended in the
following exchange. Forward-looking — informs how baseline files
should be structured when probe instrumentation lands in
drama_llama.

## The discipline

For any model accessed through a path we don't fully control, the
output is not strictly "the model's." A probe whose entire purpose
is detecting silent ratings shifts is methodologically self-defeating
if it goes through middlemen that may rewrite inputs or outputs.
**For multi-jurisdiction or alignment-comparison work, prefer
self-hosted-slow over hosted-fast.**

The threat model is wider than "is the provider being honest." It's
"does the path from prompt to logits pass through any uncontrolled
transformation." Most paths do — quietly:

- Tokenizer differences (BPE vs SentencePiece variants, BOS/EOS handling)
- Default sampler parameters (temperature, top-p, repetition penalty)
- Invisible system-prompt prepending for "safety"
- Provider-side prompt rewriting / output filtering
- Silent weight swaps for "compatibility releases"

The wrapper-bug episode of 2026-04-27 (drama_llama's repetition
penalty default-on systematically biasing Likert ratings downward)
is **evidence for** the discipline, not against it. Our own pipeline
produced silent ratings shifts within tolerance for over a week
before a clean re-capture caught it.

## Baseline-record schema

When probe instrumentation lands, every captured baseline should
record at minimum:

- `provider_source` — `self_hosted_drama_llama`, `self_hosted_blallama`,
  `self_hosted_moeflux`, `anthropic_api`, `together_ai`, `fireworks_ai`,
  etc.
- `capture_date` — ISO 8601 date. Same model + same provider can
  have different baselines weeks apart.
- `wrapper_version` — git SHA or semver of drama_llama / blallama /
  moeflux for self-hosted captures. The 2026-04-27 rep-penalty fix
  splits `self_hosted_drama_llama_pre_2026-04-27` from `_post_` as
  meaningfully different baselines.
- `sampler_settings` — temperature, top-p, top-k, repetition penalty,
  grammar mode if any. Defaults are not safe assumptions.

The unit of comparability is `provider_source × capture_date ×
wrapper_version × sampler_settings`, not `provider_source` alone.

## Cross-provider variance as diagnostic

When feasible, capture the same `(model, scenario)` on N≥2
providers. Within-provider variance and cross-provider variance
become separable signals. **If cross-provider variance >>
within-provider variance, the provider is part of what's being
measured, whether we want it to be or not.** That's worth knowing
whether or not it surfaces in the Council-gating use case.

## Where this matters in drama_llama

- The future probe-mode hook (`Predictor` / `Engine` callback per
  the canary-suite thread's resolution) should expose enough
  information to populate the baseline-record schema.
- Tests that capture canary baselines should commit them with
  metadata files alongside (`baseline.json` + `baseline.meta.json`),
  not as bare numbers.
- When two baselines disagree, the discipline says check the
  metadata first, not the model.

## Where this matters beyond drama_llama

The convention generalizes. balerion's framing — "for multi-
jurisdiction work, prefer self-hosted-slow over hosted-fast" — is
true for any safety / alignment / capability comparison where silent
shifts would invalidate the result. Worth surfacing in any
probe-related governance proposal we draft for Agora.
