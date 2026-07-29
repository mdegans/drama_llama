# Future work: cache slots vs KV capacity — verified semantics

Mike's direction (2026-07-29, deferred): "modify the cache slots
system so one slot means one entire KV cache." His worry: 4 slots +
a >32k conversation = automatic miss. **Verified against code the
same day: the worry is real but the mechanism is not a static split.**
Read this before designing the rework — half of it may already be a
knob, not a project.

## What the code actually does today

- `LlamaCppOptions::cache_slots` sets `n_seq_max = slots` **and
  `kv_unified = true`** (`src/llama_cpp/options.rs:166-167`).
- Under `kv_unified = true`, llama.cpp sets **`n_ctx_seq = n_ctx`**
  (`llama-context.cpp:287`) — there is NO per-sequence cap below the
  full pool. The `n_ctx / n_seq_max` split (line 289) applies only to
  `kv_unified = false`, which we never run with slots enabled.
- drama_llama side: `PrefixCacheConfig::capacity_cells` defaults to
  the engine's full `n_ctx` (one shared physical pool), with LRU
  eviction of other slots when the incoming call doesn't fit
  (`session/mod.rs`, `PrefixCacheConfig` doc + `ensure_capacity`).
  `check_context_fit` checks against full `n_ctx`
  (`session/mod.rs:3409-3426`).

So with 4 slots and `n_ctx = 32k`: one conversation may grow to
~32k − headroom, evicting neighbors as needed. Exceeding `n_ctx/4` in
one slot is **not** an automatic miss.

## What IS automatic

1. One conversation + `max_tokens` headroom > `n_ctx` ⇒
   `ContextOverflow` — hard typed error, not a miss.
2. Combined working set of live conversations > `n_ctx` ⇒ LRU
   thrash: two conversations each needing more than half the pool
   evict each other on every alternation — a miss on every switch.
   This is almost certainly the phenomenon behind the intuition.

## The two routes to "one slot = one entire KV"

- **Knob, not code**: raise `n_ctx` to `slots × per-conversation
  target` and keep unified. Same worst-case memory as N private KVs,
  strictly better utilization (one big + several small conversations
  coexist; a static split would strand the idle slots' cells). This
  IS "each slot can have an entire KV" with opportunistic sharing.
- **Static isolation**: `kv_unified = false` + `n_seq_max = slots` +
  `n_ctx = slots × target` → llama.cpp gives `n_ctx_seq = target`
  each. Buys a guarantee (no agent can ever evict another — fairness
  for multi-tenant Agora), costs utilization (a conversation
  hard-caps at `target` even with idle neighbors). Same total memory.

## Mike's chosen direction (2026-07-29, same day — supersedes the
## "two routes" framing above; design session wanted before any code)

His words, lightly compressed: **"one slot = one entire KV"** — but
the slot-count problem dissolves via a **tiered disk cache**:

- Every breakpoint **except the tip** is written to disk, by a
  worker, so persistence never blocks the request path.
- A partial miss (back to the previous turn, or to system+tools)
  **loads the matching snapshot from disk into the single actually-
  allocated slot** instead of keeping N conversations resident in KV.
- Eviction means deletion; cache entries can be deleted after ~1h
  (matches the existing TTL thinking).
- Rationale: "the disk is fast enough to make this worth it and we
  don't have to worry so much about number of slots then" — it
  dodges the pool-sharing/thrash issue above entirely.

He explicitly wants a design session with Claude's input before
building. Seed material for that session:

- `SnapshotStore` + the llama.cpp sequence state save/restore path
  (`src/llama_cpp/decoder.rs`, restore call site ~:804) — the
  primitive already exists and is production-exercised.
- [[future_work_kv_disk_offload]] — the 2026-07-27 memo (commit
  `332faa5`) with *measured* numbers for disk-resident KV; the
  restore-vs-reprefill crossover decides whether "disk is fast
  enough" per model. Read it before arguing either way.
- Design tensions to work through: snapshot granularity (per
  breakpoint vs per turn), what the on-disk key is (the partial-hash
  side-table already exists — #91's refusal semantics must survive
  the tier move), write-worker backpressure when an agent turns over
  breakpoints faster than the disk drains, and crash-consistency
  (a torn snapshot must read as a miss, never as a wrong restore).
- Mike's two named hazards (2026-07-29, closing note): **races** and
  **unclean shutdowns** — the latter especially, because orphaned
  snapshots from a killed process accumulate until the disk fills.
  Implication for the design: deletion must not depend on any live
  process's bookkeeping — TTL keyed on on-disk mtime plus a startup
  sweep of the cache dir reclaims orphans no matter how the previous
  process died; in-memory eviction lists alone are disqualified.

Open question for the design session (still relevant under the
tiered design, for the *resident* slot): does Agora need any
fairness guarantee, or is single-resident + disk-restore enough?
Also note `capacity_cells` already exists as a soft-reservation knob.

Related: [[plan_v0.8.0_backend_split]] (slot system landing),
#96/#91 (lookup semantics the rework must not disturb).
