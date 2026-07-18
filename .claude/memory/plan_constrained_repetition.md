# Plan of record: repetition penalty inside grammar free regions

**Status: code complete (2026-07-18), Phases 1–5 landed in five commits
(predicates → guard → state → gate/config → e2e). Fast suite green
(411). Pending: Mike-run model-backed verification —
`cargo test --test constrained_repetition -- --ignored --test-threads=1`
plus the fold-equivalence oracle
`cargo test --test sampler_state_cache -- --ignored --test-threads=1`,
and a council smoke.**

## Problem

`sample_token` fully suspends the repetition penalty while any constraint is
incomplete (`src/sample.rs` gate; motivating comment there — penalty was
crushing exit-delimiter logits on Qwen3.6 tool calls). In always-grammar-on
agentic flows (council filer seats are `tool_choice`-forced) small models
loop forever inside free-text islands (JSON string bodies); the grammar
never completes and the turn is poisoned. Constrained-span tokens also never
enter the n-gram stats, so the loop is never observed.

## Design (validated in plan mode, 2026-07-18)

1. **Permissive predicate** — a matcher state is a "free region" iff the
   popcount of its first-byte bitmap ≥ `PERMISSIVE_MIN_POPCOUNT` (64).
   Measured margins: JSON string body ≈147, until() ≈179, structural ≤~25.
   GBNF: `StackState::is_permissive` + `DfaCache::is_permissive` (rides the
   memoized `bitmaps` map — deliberately no separate DashMap). JSON engine:
   `JsonState::in_free_region` (top frame `String(Normal)`; keys count).
   Safety asymmetry: structural→permissive misreads are safe (protected
   walk still exempts exits); permissive→structural is pre-feature behavior.
2. **Region-exit guard** (`src/sample/region.rs`) — token protected iff its
   piece-byte walk leaves the region or completes a constraint (early
   return; reject-in-region ⇒ not protected, mask handles it). Protects
   merged tokens (`",`, `"}`) with zero tokenizer knowledge — the
   `IgnoreCategory::Json` bare-char list can't (tokenize-in-isolation blind
   spot). Both engines share early-return semantics.
3. **Refined gate** — (a) unconstrained: exactly today's pass;
   (b) constrained + ALL active constraints permissive: guarded penalty;
   (c) structural: skip (today's behavior).
4. **Ephemeral call-local stats** — `constrained_ngram_stats` +
   `constrained_step` on `SamplerState`. **Serialized, not serde(skip)**
   (the round-trip test serializes mid-call and continues in lockstep —
   restore must replay the identical stream, same rationale as serializing
   the RNG). Reset at the two construction doors (`init_state`,
   `resumed_from` — NOT cloned from cached). Persistent `ngram_stats` never
   sees constrained tokens, preserving cold-fold ≡ incremental-fold
   (seeding excludes tool args). Cross-call tool-arg repetition: non-goal.
5. **Config** — `RepetitionOptions::constrained_regions: bool`, default on.

## Known v1 limitation

`until("</arg>")`-style multi-token exit delimiters pass through
*permissive* KMP intermediate states, so mid-delimiter tokens remain
penalizable — bounded by windowed decay (≈2.6 additive cap at defaults),
not a livelock; irrelevant to schema-derived JSON grammars (council path).
Pinned by test `permissive_until_states`. Follow-up option: additionally
protect tokens whose walk ends in a different StateId than a self-looping
base.

## Full plan

Approved plan file (phases, test battery, verification):
`~/.claude/plans/declarative-beaming-eich.md` — copy the relevant parts
here if this outlives that file.
