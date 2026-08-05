# Plan of record: repetition penalty inside grammar free regions

**Status: COMPLETE (2026-07-18). Phases 1–5 landed in five commits
(predicates → guard → state → gate/config → e2e). Fast suite green
(411); model-backed verification all green same day
(`constrained_repetition`, `sampler_state_cache` fold oracle,
`tip_invariant`, `dialect_auto_toolcall`); council smoke clean — all
filings closed, honest cache counters, no budget burns. Remaining
follow-ups tracked in the section below.**

## #106 extension: history seeding (COMPLETE, 2026-08-05)

The v1 design's "call-local accumulator starts empty; cross-call
tool-arg repetition a non-goal" was correct for determinism but wrong
for Agora's all-structured workload — the penalty was effectively
disabled end-to-end (issue #106). Landed in four commits (flags →
fold arms → seeding → retune) + validation:

- **Three `RepetitionOptions` flags, all default ON (Default and
  serde-default)**: `seed_tool_results` (ToolResult nested Text),
  `seed_tool_args` (ToolUse/ServerToolUse arg *string values* via a
  fold variant of `value_free_text` — keys/numbers/bools excluded),
  `seed_constrained_regions` (clone the folded corpus into
  `constrained_ngram_stats` in `fold_and_snapshot`, strictly after
  the last breakpoint snapshot, `constrained_step` rebased to
  `step`). The doors (`init_state`/`resumed_from`) still zero the
  constrained fields — **history pressure flows through prompt
  content, never through carried state**, which is how "never reset"
  (Mike's original framing) was reconciled with cold≡resume.
- **Retune**: window 256→2048, decay 0.95→0.99, penalty_freq
  0.125→0.025 (cap ≈2.6 unchanged). The two window-scaled model tests
  pin the old tuning explicitly. Existing sidecars pin old numbers —
  delete/edit to adopt.
- **Load-bearing subtleties** (the vet caught #1–2 pre-landing):
  (1) rebase, not zero — `saturating_sub` means un-rebased seeds
  count at full weight *forever*; pinned by
  `test_constrained_seed_step_rebase_decays`' counterfactual.
  (2) ProbeHook captures PRE-penalty candidates — pressure asserts
  must be unit-level (`sample_token`) or stats-presence, never
  probe-logit. (3) capability gate (grammar/Json/deferred) so
  pure-prose calls don't clone corpora into cached tips. (4) new
  fold arms tokenize via `tokenize_special(_, false, false)` — no
  per-block auto-BOS. (5) short-block hole: `windows(max)` seeds
  nothing from blocks < `ngram_max_size` — this is what keeps
  digit-echo pins immune, and it drops ids/enum leaves naturally.
- **Validation (2026-08-05, all green)**: fast suite 535;
  `incremental_fold_matches_cold_fold_with_tools` (cold≡warm
  bit-exact incl. constrained fields, tool blocks in suffix); digit
  pins on all four suites (Qwen, Gemma 4, gpt-oss, Mistral 4 via
  `DRAMA_LLAMA_MISTRAL_MODEL` — the on-disk quant is Q4_K_XL, not
  the suite's default IQ3_S); pressure e2e
  `seeded_history_pressures_constrained_region`: with TWO thread
  posts sharing a phrase (surgical needs effective>1 — a once-seen
  phrase exerts zero), seeded-on paraphrased ("majestic bison
  gathering…") where seeded-off echoed verbatim. Exactly the Agora
  fix, observed end-to-end.
- **Known limitations (documented in code)**: thought-preamble gap
  (deferred-grammar JSON body never feels its own thought — seed
  runs pre-generation); tip resumes seed live-BPE n-grams a cold
  fold can't derive (pre-existing tip approximation, now also shapes
  regime (b)); fold-rule flags join `ignored_categories` in the
  "changed mid-session ⇒ cold≢warm until slot miss" class.
- **Tuning knob for Agora** (next): sidecar `surgical = false`
  (broad) so *single* prior occurrences press too — matches Mike's
  standing broad-for-text-gen preference; possibly larger window.

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
