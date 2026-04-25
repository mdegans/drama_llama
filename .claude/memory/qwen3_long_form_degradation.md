# Qwen3.5-A17B long-form free-generation degradation

Captured 2026-04-25. Parked for next session — not blocking.

## Symptom

On Qwen3.5-A17B via moeflux (blallama path), schema-constrained
generation works correctly (probe ratings round-trip cleanly,
v0.8.0 grammar/Deny fixes resolve the reserved-token loop).
**Free-text generation degrades over a few hundred tokens.**

balerion's benchmark probe (1000-word essay):
- Despite explicit instruction "do not output any planning,
  outline, thinking process," model produces a "Thinking Process"
  outline anyway — Qwen's training likely makes thinking-mode
  sticky regardless of prompt phrasing.
- Near the end of generation, output devolves into
  single-word-per-period gibberish:
  > `Start. With. First. Sentence.`
  > `History. Of. Programming. Is. Inextricably. Linked. To.`
- 288 output tokens / 168 words / `stop_reason: end_turn`.
  Generation terminates "naturally" but the content is broken.
- 1.68 tok/s wall clock (matches expected; not a perf issue).

## Hypotheses

Two non-exclusive candidates:

1. **moeflux upstream routing bug** — known unfixed issue where
   wrong experts fire for some inputs. Schema-constrained outputs
   are insulated because grammar rejects malformed bytes; free
   generation has no such floor, so misrouted experts produce
   syntactically-valid-but-semantically-broken text that
   accumulates into word-salad. Fits the "deteriorates with
   length" pattern (more tokens → more chances for misrouting).
2. **Qwen training quirk — sticky thinking mode**. The model may
   not honor "no thinking" instructions because thinking mode is
   strongly conditioned in the chat-template structure. Less
   likely to explain the word-salad terminal state, but plausibly
   contributes to the thinking-preamble.

## How to disambiguate

When the moeflux upstream routing fix lands, re-run the same
1000-word essay probe:

- **If long-form generation improves** (coherent prose, no
  thinking-mode preamble, no word-salad): routing bug was
  primary cause. Schema-constrained-only success masked it
  because grammar enforced byte-level coherence.
- **If long-form is still degenerate**: model-level quirk
  dominates; would need prompt-engineering or chat-template
  tweaks (or a different reasoning model) to fix.

A side test that disambiguates earlier: run the same essay probe
on the **GGUF Q4_K_M of Qwen3.6-35B-A3B via llama.cpp** (we have
the artifact). Same model family, no moeflux routing involved.
- Coherent essay → moeflux routing bug is the cause.
- Same degeneration → model-level. (A3B is smaller so quality
  baseline is lower, but the *failure mode* — thinking mode
  stuck on, word-salad terminal — should differ between routing
  bug vs training quirk.)

## Why this isn't blocking

- Schema-constrained generation (the actual Agora council
  workload) works correctly. Probe → JSON ratings round-trip is
  the load-bearing path.
- Free-text generation isn't on the council's critical path.
  Council members converse in structured tool-call shapes, not
  long-form essays.
- A17B is a backup tier for the council, not the primary. Cogito
  600B / API Anthropic remain primary. A17B-as-backup is good
  enough at "will it generate JSON correctly?" — which is what
  matters for governance decisions.

## Cross-references

- `grammar_reserved_token_loop.md` — sibling issue (also Qwen3,
  also grammar-related, but reserved-token loop is fixed and
  ruled out moeflux routing as cause for *that* bug via A3B
  cross-backend test). The fact that THIS bug (long-form
  degradation) only manifests on A17B (which we can't run on
  llama.cpp at all) means we don't have a clean cross-backend
  ruling for it yet.
- v0.8.0 plan — moeflux upstream routing bug is in flash-moe /
  moeflux's open-issue list; will get attention from
  danveloper / balerion eventually.
