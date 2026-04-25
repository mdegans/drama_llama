# Qwen3 long-form generation degradation — diagnosed

Captured 2026-04-25, updated same day after diagnosis.

## Symptom

`blallama --backend moeflux` against either Qwen3.5-A17B or
Qwen3.6-35B-A3B produced degenerate output on free-text long-form
generation. A17B → sentence-fragment loops with periods between
every word (`Start. With. First. Sentence.`). A3B → either
synonym/thesaurus chains (`Dash. Race. Chase. Pursue. Hunt.`) or
meta-collapse to "Wait, no? No? No?" within ~80 tokens depending
on the run. Both rode past natural EOS to the `max_tokens` cap.
Schema-constrained generation was unaffected because the grammar
masked the degeneration token-by-token.

## Diagnosis

Bisected via standalone `metal_infer/infer` on the same A3B model
weights. Standalone produced fully coherent prose at 17 tok/s with
natural EOS at token 251. blallama path with chat template +
sampler produced word-salad. Bug therefore lived in the wrapper
layer, not in moeflux's compute path.

Two real bugs in the wrapper layer were found and fixed; the second
is the dominant cause:

### Bug 1 (secondary contributor): Qwen3 chat template forced thinking mode

`ChatTemplate::render_with` never consulted `prompt.thinking`. The
bundled Qwen3 `chat_template.jinja` defaults `enable_thinking` to
the truthy branch when undefined, emitting `<think>\n` after
`<|im_start|>assistant\n`. Result: every blallama request to a
Qwen3 model started inside an open `<think>` block regardless of
the request payload. ollama exhibits the same bug for the same
reason. Fixed by deriving `enable_thinking = prompt.thinking.is_some()`
in `render_with`, mirroring Anthropic's API semantics (`None` =
disabled, `Some(_)` = enabled). Caller-set
`with_extra("enable_thinking", _)` continues to win.

### Bug 2 (dominant cause): RepetitionOptions defaults too aggressive

`RepetitionOptions::default()` ships `penalty_max_count=1`,
`ngram_min_size=1`, `penalty_repeat=1.06`. Mike's note: those
values were sized for small downstream models in Weave, not the
larger MoE models drama_llama now drives. With max_count=1, after
the second use of any content token (e.g. "Lisp", "function",
"programming") every subsequent occurrence is penalised; the model
picks synonyms which also get penalised; eventually it walks a
thesaurus chain or collapses into the punctuation-allowed fragment
loop. Schema-constrained survives because grammar mask outranks
the penalty.

Confirmed by adding a `--no-repetition-penalty` flag to blallama
and re-running the A3B essay. Result: 900 tokens of fully coherent,
factually accurate prose — same model, same chat template fix,
only difference was the penalty disabled.

## Fixes shipped

- drama_llama commit `623fa31` (v0.8.0): chat-template
  enable_thinking derivation + tests/template_rendering coverage.
- drama_llama commit `7b95910` then `04a6d97`: blallama
  `--repetition-penalty` (off by default, opt-in for diagnosis);
  `configure_session` no longer calls `with_repetition` unless the
  flag is set.
- moeflux fork commit `d013a0b`: `MAX_K` 8 → 16 (incidental
  correctness fix found in passing — A17B at K=10 was silently
  dropping 2 of 10 routed experts per layer per token because the
  `actual_K = (K > MAX_K) ? MAX_K : K` clamp at infer.m:5364 was a
  no-op for A3B but truncated A17B unconditionally; routing weights
  had already been normalised over full K so the dispatched MoE
  residual was also under-scaled). Not the dominant cause but a
  real bug fixed alongside.

## Open work

- **Tune `RepetitionOptions::default()`** — bumping `penalty_max_count`
  to ~5, `ngram_min_size` to 2 or 3, and dropping `penalty_repeat`
  closer to 1.0 should produce prose-friendly defaults. Possibly
  size-aware (different defaults for small vs large models).
- **Coverage for the surrounding repetition-sampling code** — Mike
  flagged it as "either it's the tuning or something fundamentally
  broken in the repetition penalty code." More tests required
  before re-trusting it on by default.

## Cross-references

- `grammar_reserved_token_loop.md` — sibling Qwen3-related issue
  (separate root cause: empty-piece reserved tokens passing
  byte-stream grammars; fixed in v0.8.0 via Deny mask).
- moeflux fork `~/Projects/moeflux` commit log — `d013a0b` for the
  MAX_K bump, `925f7a0` for the earlier A3B gate-offset fix.
