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

Open question for the design session: does Agora need the fairness
guarantee, or is bigger-unified-pool + LRU good enough? Also note
`capacity_cells` already exists as a soft-reservation knob.

Related: [[plan_v0.8.0_backend_split]] (slot system landing),
#96/#91 (lookup semantics the rework must not disturb).
