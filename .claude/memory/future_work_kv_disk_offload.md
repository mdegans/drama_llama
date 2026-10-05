---
name: future-work-kv-disk-offload
description: Measured economics for offloading evicted prefix-cache slots to disk, plus the cell-budget arithmetic that decides how many slots a given --n-ctx can actually hold
metadata:
  type: project
---

# Offloading evicted slots to disk — the measured case

**2026-07-27**, from the live blallama run set up to measure #91 drift.
The disk cache was already a known candidate (see
`plan_release_freeze.md`: design during the freeze, land only if it
stays boring, `PrefixSlot` is already plain serializable data). What
was missing was numbers. These are measured, not estimated.

## Capacity is a shared pool, and the arithmetic is tight

`--cache-slots N` sets `cp.n_seq_max = N` **and** `cp.kv_unified =
true` (`src/llama_cpp/options.rs:133`). Unified means the cells are
one shared pool — `n_ctx_seq == n_ctx`, no division — so any single
conversation may grow to the full context. What is bounded is the sum.

`plan_eviction` budgets exactly that: `sum(other slots' cells) +
(incoming prompt + max_tokens headroom) <= capacity_cells`, evicting
LRU-first until it fits. `capacity_cells` defaults to `n_ctx`. The
pending slot is never evicted. So for N conversations of C cells:

    (C + headroom) + (N-1)*C <= n_ctx    =>    C ~< n_ctx / N

At `--n-ctx 131072 --cache-slots 4` that is **~32k cells per agent**
before evictions start. Agora prompts have historically run 40–60k
(see `future_work_prefill_progress_callback.md`), which four of will
not fit — 2 slots would serve those better than 4. Sizing slots is
therefore a function of expected conversation length, not of agent
count, and getting it wrong shows up as thrash at agent *switches*
(consecutive turns by one agent are safe; the active slot is
protected).

## What a cell actually costs — much less than you would guess

From the run's own log, Qwen3.6-35B-A3B-UD-IQ4_XS at `--n-ctx 131072`:

    llama_kv_cache: size = 2560.00 MiB (131072 cells, 10 layers, 4/1 seqs),
                    K (f16): 1280.00 MiB, V (f16): 1280.00 MiB

**Only 10 layers carry a real KV cache.** The rest log as `filtered`
and are handled by `llama_memory_recurrent` — the linear-attention
layers, whose state is fixed-size per sequence rather than per-token.
That is the whole point of the architecture and it makes the KV
footprint far smaller than a dense model of the same size.

    2560 MiB / 131072 cells = 20 KiB per cell
    32k-cell conversation   = ~640 MiB of attention KV

## The economics

| for one 32k-cell conversation | cost |
| --- | --- |
| re-prefill at ~900 tok/s (Qwen prefill) | ~36 s |
| serialize + write + read back ~640 MiB on NVMe | ~1 s |

An order of magnitude, plausibly two. `llama_state_seq_get_data`
covers the recurrent state as well as the attention KV — which is the
load-bearing part on this model family, since
`llama_memory_recurrent::seq_rm` rejects partial-end truncates and
that state cannot be rebuilt by rewinding (see
`qwen3_a3b_llama_cpp_rewind_diagnosis.md`). Offload turns a capacity
eviction from a full re-prefill into an I/O round trip.

## TTL and capacity are orthogonal — do not conflate them

`CacheTtl::OneHour` already exists and works; `sweep_expired_slots`
honours it. But TTL is the **staleness** axis and `plan_eviction` is
the **space** axis, and the latter ignores TTL entirely. A 1h-TTL slot
is evicted the moment cells run short. So raising TTL buys nothing
against capacity pressure. The natural shape when this lands:
eviction *serializes* instead of dropping, and TTL governs when the
file is reaped. `PrefixSlot::created` already carries a comment saying
it exists for the future disk cache.

## Caveats before building

- The ~1 s figure assumes NVMe and does not include
  `llama_state_seq_get_data`'s copy to host memory, which is a real
  cost on unified memory and should be measured before committing.
- Per-cell cost is model-specific. A dense model with all layers
  attention-backed will be several times 20 KiB/cell; re-measure per
  model family rather than reusing this number.
- Snapshot bytes in the engine's own `seq_snapshots` LRU are a
  separate resource from slot KV cells. A disk cache for slots does
  not fix `checkpoint missing`, which is that LRU evicting a restore
  target and costing a `reset_slot`.
