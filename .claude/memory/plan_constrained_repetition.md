# Plan of record: repetition penalty inside grammar free regions

**Status: COMPLETE (2026-07-18). Phases 1–5 landed in five commits
(predicates → guard → state → gate/config → e2e). Fast suite green
(411); model-backed verification all green same day
(`constrained_repetition`, `sampler_state_cache` fold oracle,
`tip_invariant`, `dialect_auto_toolcall`); council smoke clean — all
filings closed, honest cache counters, no budget burns. Remaining
follow-ups tracked in the section below.**

## Open follow-ups (queue for a future session)

1. **until() delimiter protection (optional, low priority)** — the v1
   limitation: mid-delimiter tokens of `until("</arg>")`-style exits
   remain penalizable (intermediate KMP states are permissive).
   Sketch: additionally protect tokens whose walk ends in a different
   `StateId` than a self-looping base. Becomes relevant when the
   dialect work (plan_tool_dialects phases D–G, Harmony) starts
   emitting until() grammars into live sessions — do it then, not
   before.
2. **Changelog note** — DONE (2026-07-19): logged under `[0.8.0]
   ### Changed` alongside the top-k-1024-before-locally-typical
   default-chain change. `set_constrained_regions(false)` documented
   as the escape hatch.
3. **Artist trailing `\"` curiosity** (council, not this feature) —
   both artist outputs ended with an escaped quote. Mike's approach:
   dump per-seat prompts at the end of a run and read the full
   context before guessing.

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
