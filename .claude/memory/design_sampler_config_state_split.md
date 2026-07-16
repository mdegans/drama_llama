# Design — SampleOptions config/state split + cacheable SamplerState

**Status:** Design settled (Mike + Claude Opus 4.8 2026-07-15; reviewed against
code + revised with Mike by Claude Fable 5 2026-07-16). **Phases 0–1 LANDED
2026-07-16** (six commits, `756f748..bc211f7`): all four pre-lands (BTreeMap
NGramStats; banned_specials → Vec; Temperature mode #35; DeferredGrammar
tightening), the rand_pcg groundwork (xorshift could not expose its state —
`Pcg64Mcg` per Mike, u128 seed matches `NonZeroU128` directly), and the split
itself — `SamplerConfig` (renamed) + `SamplerState`, CompiledGrammar carve-out,
DFA cache config-homed **with a 64k interned-state growth cap** (Mike's
question: recursive grammars mint a state per nesting depth, so a
Session-lifetime cache needs the cap; clear-on-exceed is safe, pure memoization),
deferred-activation flag, observation channels replaced (predictor accessors +
ProbeCtx `config`/`state`), deletion ledger executed (~350 lines), bit-exact
serialize→restore→continue test green. Fast suite green both feature sets;
model-backed `--include-ignored` pending (Mike runs).

Two implementation decisions made in the landing (not in the original design):
(1) `sample_token` takes NO seed param — seed enters only at `init_state`;
(2) the repetition category-drain (which mutated config to memoize) became
`RepetitionOptions::resolved_ignored(model)`, computed once at `init_state`
and homed in `SamplerState.resolved_ignored` — a documented config×model memo
riding the state, not an accumulator.

**Remaining: Phase 2** (Session-owned config, state homed in `PrefixCache`
breakpoints w/ clone-on-load + promote-at-turn-end, seed trichotomy with
default → `None`, incremental block-gated prompt-seeding, per-Session
`emit_ban_set` memo) — next session, plus **Phase 3** (deserialize door,
blallama wire, clone-cost + permissive-region spikes). Open items are marked;
everything else is decided.

## Phase 2 design round (Mike + Claude Fable 5, 2026-07-16 PM)

Converged decisions from the pre-Phase-2 design chat (grounded in a
post-split code map):

1. **`Breakpoint` struct unification.** `PrefixCache` is parallel vecs today
   (`prev_breakpoints: Vec<EntryPos>` + `prev_breakpoint_hashes` + the tip
   pair). Adding a fourth parallel `Vec<SamplerState>` is the smell; unify
   into `Breakpoint { pos: EntryPos, hash: [u8; 32], state: SamplerState }`,
   tip becomes `Option<Breakpoint>`.
2. **Load = reconcile-by-grammar-identity, NOT reset and NOT verbatim clone.**
   Full state is stored (over-fused, unchanged). At load, one named
   constructor (`SamplerState::resumed_from(&cached, &effective_config)`):
   for each mode in the new effective config, carry the cached matcher's
   position iff the compiled grammar is identical; else that matcher starts
   at root. Everything else — `mu`, rng, `ngram_stats`, `resolved_ignored` —
   carries unconditionally. Rationale: `matchers` is index-aligned with the
   *effective* (per-call) modes vec, which can differ per call (run_call
   prepends call-derived grammar modes) — verbatim clone risks OOB indices;
   but partial completions / assistant-prefill (tip resume) need matcher
   position to carry when the grammar IS the same. This is the override
   matrix + deserialize gate 3 applied to the in-memory path; the same code
   serves the deserialize door later.
3. **Borrow topology spike DISSOLVED.** The effective config is genuinely
   per-call (session config + call-derived grammar modes never stored on
   Session), so there is no stable config to borrow and no new predictor
   lifetime. Phase 2 instead delivers: one `effective_config(&self,
   &PreparedCall)` helper replacing the triplicated assembly
   (complete_text/complete_stream/run_call), plus memoized stable inputs.
4. **`emit_ban_set` memo = cached field invalidated in exactly two setters**
   (`with_dialect` session/mod.rs:1450, `with_emit_specials_ban` :1766).
   Currently recomputed per call at all three assembly sites.
5. **Seeding moves to Session, walks blocks via
   `misanthropic::prompt::Index`/`BlockIndex`** (`Prompt::indices()` yields
   cache-prefix order; breakpoints happen at blocks only). The predictor's
   whole-prompt reseed loop (predictor.rs:549-572, incl. the dummy
   full-vocab Candidates TODO) dies. Incremental = suffix-seed from the
   matched breakpoint's cached stats. If `IndexRef` needs a helper, write
   the consuming code first, PR misanthropic upstream (alpha bump).
6. **Tip invariant (new, decided):** *entries, KV, and the tip's
   `SamplerState` must all describe the same stream position; the sampled
   stop token is in none of them.* Entries+KV already hold (truncate-and-
   swap in `compute_tip_extension`, :2984-3036 — tip sits exactly at the KV
   head, not "several tokens prior"; the entry list extends one past it with
   the canonical close). The state side requires: **hoist `state.advance()`
   out of `sample_token` into the predictor loop, after the stop check** —
   a token that terminates generation never mutates the state. Today this
   is violated silently; masked for EOG stops by `repetition.ignored` but
   live for custom stop sequences (ngram) and always for the matcher
   (advanced over stop-token bytes, not canonical-close bytes). The rng is
   the one deliberate, documented exemption (it advanced to sample the stop
   token; no oracle can observe the difference).
7. **Integration tests (try-to-break-it list):**
   - incremental-vs-cold seeding equality: same prompt, cold full-prefill
     seeding vs breakpoint-resume suffix seeding → `ngram_stats` bit-equal
     (BTreeMap ⇒ plain assert_eq).
   - the breaker: same, but first turn ends via a custom non-EOG stop
     sequence — surfaces the phantom-advance bug; should FAIL pre-hoist,
     green post-hoist.
   - reconcile matrix: resume same-grammar (position carries), changed
     grammar (matcher-only reset; stats/rng/mu carry), modes-vec length
     change (no OOB/panic).

Huge blast radius by design — this touches `sample.rs`, `predictor.rs`,
`session/mod.rs`, `engine.rs`, `output_config.rs`. Deliberate, phased, and
"done right we delete a lot of code." Do NOT start editing from this memo;
it's the warm-start for plan-mode, and Mike wants a design chat before code
each session.

## Motivation

`SampleOptions` conflates two things: an immutable **config** (Clone/PartialEq/
Serialize value semantics) and live per-call **run-state** (the grammar/JSON
matcher position). Every symptom Mike flagged on the v0.8.0 review is that
conflation leaking:

- `#[serde(skip)]` on `deferred_grammar` and `banned_specials`.
- `Arc<Mutex<GrammarState>>` / `Arc<Mutex<JsonState>>` inside the `Grammar`/
  `Json` variants (needed to mutate live state through a `Clone` value).
- ~80 lines of custom serde hooks (`serialize_grammar`/`deserialize_grammar`/
  `serialize_json_state`/`deserialize_json_state`, `sample.rs:27-104`) — and
  they're *inconsistent*: Json serializes full state, Grammar serializes
  source-only (verified 2026-07-16; the doc comments admit it).
- ~113-line manual `impl PartialEq for SamplingMode` that locks two mutexes
  and treats a poisoned lock as "unequal" (`sample.rs:1056-1169`).
- Poison handling duplicated across serialize / eq / promotion / completion.

No hard urgency — the driver is publishing v0.8.0 closer to the ideal 1.0.
The smell is fine while developing; left around it rots. Timing matters for
one concrete reason: the split changes the serialized wire format, which is
only free to break pre-publish.

**The goals it serves:** (a) exact snapshot of sampler state *including* the
grammar matcher, (b) `PredictOptions` config assembled once on the `Session`,
not per call, (c) repetition penalty over context (prose-gated, see below),
(d) future: cache `SamplerState` alongside the KV cache + disk-based
breakpoints.

**Key realization:** (b) is *incompatible* with per-call live state living in
the config. A Session's config outlives a call; a matcher is per-call (grammar
starts at root each generation). Storing today's `PredictOptions` on the Session
would make call #2 resume mid-grammar from call #1. So hoisting config to the
Session **forces** the live state out of it. Mike's consumer-side change
(Session owns config, predictors borrow `&`) and the config/state split are the
*same refactor* from two directions.

## Core decision

Split into:

- **`SamplerConfig`** — pure immutable value config. Derive `Serialize`/
  `Deserialize`/`PartialEq`/`Clone`. One carve-out, by design (see DFA cache
  home): the compiled-grammar type carries `Arc<Grammar>` + `Arc<DfaCache>`
  with a single manual `PartialEq` (compare source, ignore cache) and a single
  source-only serde impl on that one type. That's the entire custom-impl
  surface — don't chase 100% derive purity past it.
- **`SamplerState`** — owned, mutable, per-call run-state. Plain owned /
  POD-ish, fully derive-serializable, **no `Arc` anywhere, no exceptions**.
  Keeping state 100% pure is what makes the bit-exact test story work
  (serialize → restore → compare with derived `Eq`).

### The spine: every stateful sampler concept = (knobs → config) + (accumulator → state)

| concept | config (immutable) | state (SamplerState) |
|---|---|---|
| grammar | compiled `Arc<Grammar>` + source + `Arc<DfaCache>` | matcher `StackState` (POD) |
| json | (nothing / fixed built-in grammar) | parser position |
| deferred grammar | the spec (`DeferredGrammar`: grammar + triggers) | `deferred_active: bool` (or `activated_at`) + its matcher `StackState` |
| mirostat | `tau, eta, max_keep` | `mu` |
| repetition | `window, decay, penalties` | `NGramStats` accumulator |
| rng | starting `seed` | working `Xoroshiro` state |
| temperature (new, #35) | `t` | (stateless) |

The sorting rule that resolves placement questions: **a memoization or
derivation of config belongs with config; an accumulator of the run belongs
with state.** (This is what files the DFA cache under config.)

`mu`, the working rng, and `ngram_stats` **already live as separate fields on
`TokenPredictor`** — `SamplerState` is largely a *gathering* of existing
predictor fields plus pulling `StackState` out of the `Arc<Mutex>`. The shape
half-exists; risk is lower than the blast radius suggests. Target signature:
`sample_token(opts: &SamplerConfig, state: &mut SamplerState, ...)`, replacing
today's `opts: &mut SampleOptions, mu: &mut Option<f32>`. (`sample_token`'s
current `&mut opts` is *spurious* — verified: it never writes a config field;
all mutation goes through the mutexes and the separately-threaded
`mu`/`rng`/`freq_map`.)

### DFA cache home — config (decided 2026-07-16)

`GrammarState` today = `{Arc<Grammar>, StackState, Arc<DfaCache>}`
(`grammar.rs:744-749`). The split sends `StackState` to state; the cache goes
**with the compiled grammar in config**, e.g.
`CompiledGrammar { source, rules, dfa: Arc<DfaCache> }`. Why not state:

1. **It can't ride serialization.** `DfaCache` is `RwLock` + four `DashMap`s +
   atomics, and its `StateId`s are process-local interning (insertion-order
   dependent, grammar-index-coupled) — meaningless in another process, hostile
   input through our own deserialize door. State-homed ⇒ every restore and
   every fresh call starts a cold cache.
2. **Config-homed = warm across calls and restores.** One cache serves the
   grammar for the Session's lifetime — a small perf *win* over today, where a
   per-call `GrammarState::new` starts cold. DashMap is lock-striped for
   exactly this sharing (the rayon fold).
3. **Purity lands in the right place.** Config already carries the
   `Arc<Grammar>` carve-out; the cache adds zero marginal impurity there and
   `SamplerState` stays fully derive-pure. Manual `PartialEq` compares source
   and ignores the cache — exactly what `GrammarState` does today
   (`grammar.rs:754`).
4. The spine rule: the cache is a pure memoization *of the grammar* — a
   function of config, not an accumulator of the run.

## Construction invariant

- The **effective config** (base config + per-request overrides, merged) is the
  **sole constructor** of a fresh `SamplerState` (`SamplerConfig::init_state`).
- The only other way to obtain one is **validated deserialization** (see door).
- Overrides resolve into an effective config *first*; state is built from the
  effective config → the "config is the authority" invariant holds even with
  per-request overrides.

## Cache / ownership model

- `SamplerState` is **homed in the cache entry** (over-fused with KV for now —
  simplest; re-roll is still supported, see below).
- The cache entry is only ever **read (cloned)** or **replaced at a breakpoint
  boundary** — never `&mut`-mutated in place. (Mutating in place would
  invalidate the breakpoint.)
- Per call: **load** = clone from cache on hit, else `init_from_config` on
  miss. Working-copy lifetime = the completion call; `&mut` freely within it.
- **Turn end** (e.g. assistant auto-breakpoint): the working `SamplerState` is
  **promoted to cache** as the new breakpoint rather than dropped.
- **Clone-on-load is semantically necessary** (can't `&mut` the pristine
  snapshot). Its cost lives ENTIRELY in `NGramStats` (see below).
- **Tip-entry take-on-match** (decided): the cache keeps a separate tip entry
  (saves re-prefilling the Assistant turn — a bonus the Anthropic API doesn't
  do). On an exact tip match the state may be *moved* rather than cloned —
  pure clone-elision, semantics identical to clone-then-replace. Safe later
  optimization; by far the most common case; doesn't affect the design.

## Snapshot semantics

- `SamplerState` snapshots pair with KV snapshots at the **same logical
  position**; restore is atomic (both or neither).
- **Snapshot-coupled, NOT operation-coupled.** No attempt to make
  `SamplerState` follow arbitrary KV *truncations* — grammar matchers and
  ngram stats aren't cheaply rewindable, and this echoes the moeflux/blallama
  partial-truncate pain. Define boundaries; resume only there; else rebuild.
- **Breakpoints only** (Anthropic-style). Simple, and the right direction.
- **`Engine` keeps raw arbitrary-rewind KV ops** (unenforced, documented
  caveat) — matches the existing "Engine is raw, Session enforces invariants"
  line (same reason Session does the injection guard and Engine doesn't).
  Session's cache layer only ever snapshot-couples.

## Resume / fork / fresh — one optional seed encodes all three

- **no seed + cache hit** → resume (clone cached state; working rng continues;
  identical completion — the reproducibility feature).
- **no seed + cache miss** → fresh random seed, `init_from_config`.
- **seed present** → fork (`init_from_config` with that seed; ignore cached
  state; deterministic re-roll).

Client-side: "re-roll differently" = send a fresh random seed; "reproduce
exactly" = send the same fixed seed; "continue the conversation" = send
nothing. **There is no separate resume/fork verb** — the trichotomy is computed
server-side from `(seed?, hit?)`. So over-fused storage is fine; the only seam
kept is the one-line resume-vs-fork branch at load.

Resume ≠ restart: config carries the *restart* seed, state carries the *resume*
(working rng) position. Serializing the working rng — not just the seed — is
what makes a restored snapshot continue the exact same stream (mandatory for
breakpoints).

**Seed default flips to `None` (decided 2026-07-16).** Today absence barely
exists: `PredictOptions::default()` sets `Some(DEFAULT_SEED)` and Session
stores `Some(DEFAULT_SEED)` for pre-0.8 compat (`session/mod.rs:896`,
`predictor.rs:79`) — under the trichotomy that would make every call a fork
and resume unreachable. New semantics: Session defaults to `None` →
resume-by-default ("same conversation continues its stream"), reproducibility
one explicit seed away; `DEFAULT_SEED` stays available as a constant.
Observable behavior change, pre-publish is the time. `prepare()`'s
clock-panic/time-seed arm (`predictor.rs:512-524`) is absorbed and dies.

## Wire (blallama concern; drama_llama's API is just `Option<Seed>` in per-call overrides)

- **Hard constraint:** blallama must stay Anthropic-API-compatible because
  Agora agents are **dual-homed** — the same agent config is pointed at
  blallama *or* real Anthropic (e.g. a Haiku-based agent) interchangeably. Any
  body-schema divergence → real Anthropic rejects the request.
- **Therefore seed rides an HTTP header** (out-of-band; Anthropic ignores
  unknown headers). Body-superset field → rejected by real Anthropic.
  `metadata` → also rejected (Mike has hit this) and is contractually
  "doesn't affect inference," which seed does.
- **Portability-by-field principle:** a non-Anthropic control goes in the
  header **iff an agent using it could still validly run on real Anthropic.**
  - seed → header (seeded agents are portable).
  - raw GBNF grammar → body field OK (agents using raw GBNF are inherently
    blallama-only; Anthropic can't honor GBNF, so portability is already
    forfeited).
  - structured-gen via `tools`/`tool_choice` → native Anthropic body, fully
    portable (blallama maps `tool_choice` → grammar internally). Ditto
    `Prompt::output_config`-driven structured generation — which is the
    common case for Agora agents and is why the repetition-in-permissive-
    regions spike below matters.
  - `top_p`/`top_k`/`temperature` → native Anthropic body fields. **Override
    policy (decided): compile, don't merge.** No pipeline surgery — there is
    no coherent answer to "where does top_k go in an arbitrary existing
    pipeline." Instead the body fields form a semantic config
    (`ApiSampling { temperature, top_p, top_k }`) that **compiles to a
    pipeline** in canonical order (TopK → TopP → Temperature → dist, the
    llama.cpp/Anthropic-conventional order). If any field is present, the
    compiled pipeline *replaces* the truncation portion of the default
    pipeline wholesale; constraint modes (Grammar/Json) and repetition config
    are preserved. If none present, the model/session default pipeline is
    used unchanged. This is config-is-authority applied consistently:
    effective config first, pipeline derived from it.
    **Dependency:** there is currently no `Temperature` sampling mode at all
    (tracked as #35) — add it to `SamplingMode` and `Candidates` as part of
    (or before) this policy.

## Per-request overrides + cache-invalidation matrix

**KV-cache validity keys ONLY on the token sequence (prompt). Sampler-state
validity keys ONLY on grammar identity. Everything else is a free override.**

| per-request override | KV cache | sampler state |
|---|---|---|
| top_p / top_k / temp / min_p | keep | keep (stateless) |
| repetition knobs | keep | keep `NGramStats` (apply differently) |
| **seed** | keep | replace (fork) — or resume if absent |
| grammar / tool-**choice** forcing | keep | reset matcher; keep the rest |
| messages / tool **definitions** | invalidate (tokens change) | rebuild at new boundary |

- Tool distinction: tool **choice** (forcing directive) → grammar change → keep
  KV; tool **definitions** (available set, live in the prompt) → token change →
  invalidate KV.

### Why the matcher is grammar-coupled (and JSON isn't)

`Position { rule_idx, alt_idx, atom_idx }` is a cursor indexing the compiled
`Grammar.rules` — the NFA state *is* "where in the rules am I." Rule indices
aren't stable across grammar edits (adding a rule shifts anon-rule indices), so
a different compiled grammar = garbage indices = OOB. Same source → deterministic
same compile → same indices (so source+positions serialize self-consistently);
different source → invalid. **JSON's grammar is fixed/built-in**, so the JSON
matcher is NOT config-coupled. ⇒ states with no custom GBNF are freely resumable
across any config change; only custom-grammar states carry the grammar-identity
constraint.

## Deserialize door — treat as parsing UNTRUSTED input

Even though it's "our own" format: a stale/corrupt/mismatched blob feeds garbage
`Position` indices straight into `rules[...]` — OOB, or worse a matcher that
silently accepts illegal bytes and violates the grammar invariant we're supposed
to uphold. Mike's "foot shotgun." Gates, cheapest-first (**TTL is blallama's
job, not the library's** — a blob that passes the gates below doesn't become
invalid by clock time, and atomic KV-pairing provides the real staleness
protection; the library's contract stays mathematical, the storage layer owns
expiry policy):

1. **version understood** — refuse unknown versions hard; no best-effort parse.
2. **integrity** — `hash(blob)`; stream-and-hash in one pass on load.
3. **grammar-identity** — blob's embedded grammar source/hash == effective
   config's grammar; else reset just the matcher to the new grammar's root.
4. **bounds-check** every index (rule/alt/atom in range; `pending` ≤ 4 bytes;
   stacks well-formed).

Any failure → **discard, `init_from_config`.** Never resume into a
mismatched/unbounded state.

- **Storage:** sidecar (version, hashes; blallama adds timestamp/TTL) + binary
  blob. Read the sidecar first (cheap version reject) before streaming the
  blob body.
- Verify bounds-checks exist specifically on the *deserialize* path — don't
  assume the live-path `GrammarState` invariants carry over to reconstructed
  state.

### Hash design — integrity + grammar-identity, NOT whole-config

`hash(blob+config)` (the first instinct) **over-invalidates**: changing
`top_p` / tool choice / repetition knobs would needlessly discard a resumable
state, defeating the "mutable config / Session config setter / repeated
generations with different sampler settings" goal. So:

- `hash(blob)` for **integrity only**.
- **grammar-identity** checked specifically (blob already embeds its grammar
  source). Every other config field may change freely with the state preserved.

(BTreeMap below also makes the blob *canonical* — two saves of identical state
produce identical bytes — which the integrity hash quietly wants.)

## NGramStats

- **BUG (found in review, must-fix step of the split): HashMap iteration order
  breaks bit-exact resume.** Both penalty passes iterate the map directly
  (`repetition.rs:924`, `:953`) and multiple n-grams accumulate additive
  penalties onto the *same* candidate logit (broad mode especially: many
  n-grams share a "last non-ignored token"). `NGramStats.data` is a `HashMap`
  (`ngram.rs:212`); a deserialized map rebuilds with a fresh `RandomState` →
  different iteration order → float accumulation isn't associative → ULP-
  different logits → diverged stream. Violates the mandatory serialize →
  restore → continue invariant, invisibly, on the exact feature this refactor
  enables. **Fix: `BTreeMap<NGram, NGramData>`.** Not much surgery; also the
  only practical way to *test* bit-exactness; also canonical blobs (above).
  O(log n) vs O(1) is noise at windowed sizes.
- **Growth is already window-bounded — this was half-believed and is now
  verified.** `evict_outside_window` runs at the top of every penalty pass
  (`repetition.rs:892`); default window 256 steps, decay 0.95
  (`repetition.rs:171-179`), so the map is tiny out of the box. Eviction is
  already a pure fn of `(stats, step, window)` → deterministic → rides the
  bit-exact path for free.
- **Strength-threshold pruning: NOT added speculatively** (vendor-lever
  discipline — measure to rule in). It only matters for clone cost at *large*
  windows, which goal (c) "repetition over context" does imply — so keep the
  clone-cost measurement spike (40–60k context) and add strength pruning only
  on a measured win. If it lands it must be deterministic (pure fn of
  `(stats, step)`, applied inline in the same pass as eviction).
- No `Arc`/CoW: Arc reintroduces the shared-ownership complexity we're
  deleting everywhere else. Plain owned value (consistent with
  `banned_specials` → `Vec<Token>`). `NGram` is already a stack type
  (TinyVec-backed).

## Repetition block/prose gating

Grammar-*aware* repetition is the tar pit; **block-aware** is the win, and
generation already has block boundaries. Two gate points, same prose/structured
concept:

- **Generation: already shipped.** The constraint-active suspend
  (`sample.rs:1208-1227`) already skips *accumulation* too — `freq_map`
  ingestion lives inside the penalty pass, documented as an accepted side
  effect at `sample.rs:1202`. The split formalizes this; no new mechanism.
  Prose (incl. `<think>`, which has no active constraint) accumulates +
  penalizes; active grammar/JSON regions don't.
- **Prompt-seeding: the new work.** Exclude structured *prompt* blocks (esp.
  tool results — the digit-penalty case) from initial stats. Signal here =
  block *type* (the Session has it). Replaces today's whole-prompt reseed in
  `prepare` (`predictor.rs:540-565`), which also becomes incremental
  (Session-cached stats keyed to breakpoint; cold-rebuild path kept).

Reframes the feature as **"repetition over the running PROSE corpus, structured
regions excluded"** — not "entire context."

- carry-across vs per-call reset = a config knob on top (some chat/tool callers
  want reset even for prose).

### Permissive-region spike (spike, then decide — 2026-07-16)

The original "edge, fine, don't fix" position is retracted: most Agora agent
posts generate under `Prompt::output_config` (structured), so blanket
suspension = effectively **no repetition penalty for Agora agents**, which is
bad with some models. But the observed failure (Qwen3.6 delimiter thrash)
wasn't "penalties in permissive regions are bad" — it was specifically that
penalties **crush the exit delimiter's tokens**. Agora's case (long prose
inside `.+`/JSON-string values) *wants* penalties on the prose but must not
crush the closing delimiter. The narrow target:

- **Penalize inside permissive regions, but exempt candidates that advance
  toward the region's exit.** Needs a matcher-level query ("which first-bytes
  exit the permissive atom into the structural continuation") that
  `first_byte_bitmap` does NOT currently answer (inside `.+` it's all-ones).
  Much smaller than grammar-aware repetition generally; fixes both the Agora
  want and the thrash case with one mechanism instead of a knob trading one
  for the other.
- **Fallback if the exit-set query is hairy:** per-config
  `penalize_permissive: bool`, Agora opts in eyes-open (accepting thrash risk
  with its models).

## Observation channels — deletions with named replacements

Two things the deletion ledger removes are load-bearing *observation
channels*; their replacements are plan steps, not discoveries:

1. **Mid-generation completion.** Session's early-break
   (`session/mod.rs:3341`) and the incomplete-at-end violation check (`:3497`)
   watch matcher progress *through the shared Arc handles*
   (`grammar_handles`/`eager_grammar_handles`, `:3246-3259`). Owned
   `SamplerState` severs that channel. Replacement: an accessor on the
   predictor (`predictor.sampler_state()` / `grammar_complete()`). Cleaner
   than the handles dance, but must land in the same step that deletes it.
2. **ProbeHook.** `ProbeCtx.sample_options: &SampleOptions`
   (`predictor.rs:718`) — probes that inspect matcher state need
   `&SamplerState` added to `ProbeCtx`.

Also dissolving, deliberately: the `Json`/`Grammar` **auto-reset-on-success
contract** (`sample.rs:524-543` — "next generation on the same mode instance
starts fresh"). Fresh state per call replaces it. Session/Engine/blallama are
all new in 0.8.0 — behavior *can* change, and blallama is ours to change too —
but *check* rather than assume that no code path relies on auto-reset before
deleting it.

## Deletion ledger (grounded)

- `serialize/deserialize_{grammar,json_state}` custom hooks (~80 lines) — config
  derive-serializes; `StackState` is POD. (Replaced by ONE source-only serde
  impl on the compiled-grammar type.)
- `impl PartialEq for SamplingMode` (~113 lines, mutex-locking) — derive works
  once variants hold specs not matchers. (ONE manual `PartialEq` survives, on
  the compiled-grammar type, source-compare-ignore-cache.)
- all `Arc<Mutex<>>` + `.lock()` + poison branches (`sample.rs` eq/serialize,
  `predictor.rs:703`, `session::any_grammar_complete:4306`, `output_config`).
- `deferred_grammar.take()` / `modes.push()` promotion dance
  (`predictor.rs:664-712`) → **a flag** (`deferred_active` /
  `activated_at`) in `SamplerState`; the sampling loop reads the spec from
  borrowed config. No vec of refs, no indices — there's exactly one deferred
  grammar (`Option<DeferredGrammar>`); indices only earn their keep if
  multiple ever exist.
- per-call ngram rebuild (`predictor.rs:540`) → incremental cached stats + a
  cold-rebuild path.
- `prepare()`'s time-based-seed / clock-panic arm (`predictor.rs:512-524`) —
  absorbed by the seed trichotomy.
- per-call `emit_ban_set()` recomputation → `banned_specials` computed once on
  the Session as plain `Vec<Token>` (per-`(model,dialect)` derived config).

## Feasibility spikes (before committing shapes)

- **Borrow topology — smaller than first feared.** Clone-on-load dissolves the
  cache-aliasing worry: the working `SamplerState` is an owned clone living in
  the predictor (where `mu`/`rng`/`stats` already live), never aliasing the
  cache entry or the llama context. The *real* ripple: predictors borrow
  `&config` from Session while Session lends `&mut engine` — a plain split
  field borrow inside Session methods, but consumers who *store* a predictor
  feel the lifetime. Spike: "Session method constructs a predictor borrowing
  `&self.config` + `&mut self.engine`"; don't spend time on cache aliasing.
- **Clone cost:** measure `NGramStats` clone at a 40–60k context with a large
  window; decides strength-pruning (see NGramStats).
- **Permissive-region exit-set query:** the repetition spike above.

## Downstream

- **Weave uses `PiecePredictor` directly.** Don't design around it — Mike:
  breaking changes are fine; Weave keeps building against the old version and
  will likely move to `Session` outright when that project resumes (plus
  native Anthropic support, making blallama one backend among others).
  Priority consumer is `agora-agentlib` via blallama (`bin/blallama.rs`).

## Decisions log (from the 2026-07-16 review round)

1. **Config is two objects:** Session-stable `SamplerConfig` (sampling
   strategy, stop sequences, grammar spec) + a small per-call args struct
   (`n`, `seed`, per-request overrides). The code already wants this split —
   `session/mod.rs:3269-3279` rebuilds exactly it by hand today.
2. **Phasing:** pre-land as independent steps: `banned_specials` →
   `Vec<Token>` and the `DeferredGrammar` type-tightening (small, green on
   their own). Do NOT pre-land matcher-position or DeferredGrammar
   serialization against the old shape — that code is written once, for
   `SamplerState`, in the split itself; pre-landing churns the wire format
   twice and risks the two-of-a-thing failure mode (throwaway code that never
   actually gets deleted → dead code).
3. `BTreeMap` for `NGramStats` — named correctness step (see NGramStats).
4. DFA cache home = config (see its section).
5. Seed default → `None`; resume-by-default (see Resume/fork/fresh).
6. top_p/top_k/temperature: compile-don't-merge; add `Temperature` mode (#35
   is a dependency).
7. Repetition in permissive regions: spike exit-set exemption, then decide.
8. TTL lives in blallama, not the library.
9. Deferred-grammar promotion → flag in state + spec in config.
10. Tip take-on-match = clone-elision, safe later optimization.

## Remaining open questions

1. Exact home/keying of the `SamplerState` cache entry vs the KV entry
   (over-fused for now; keep the resume/fork load-branch). Includes the tip
   entry's exact shape.
2. Outcome-dependent: strength pruning (clone-cost measurement) and the
   permissive-region mechanism (spike).
3. Canonical-pipeline details for the API compile: exact placement of `MinP`
   if we ever map it. ~~Whether the compiled pipeline replaces Mirostat modes
   too~~ — RESOLVED (Mike, 2026-07-16): yes. Standing principle: **any
   divergence from the official Anthropic API is a bug**, except where we can
   strictly improve on it (e.g. the bonus tip breakpoint); downstream code
   accounts for this. Anthropic semantics requested = Anthropic-shaped
   pipeline delivered.

## Related in-repo memory

- `eog_is_not_eos_plus_eot.md` — stop-logic authority (unaffected, but adjacent).
- `plan_lazy_grammar_check.md` — the lazy/sample-then-check path lives inside
  `sample_token`; interacts with the `&mut SamplerState` signature change.
- `blallama_session_state_pollution.md` / the moeflux partial-truncate memos —
  why we do NOT make SamplerState follow arbitrary KV truncation.
