---
name: qwen3-a3b-llama-cpp-rewind-diagnosis
description: Definitive cause of cross-position cache miss for Qwen3.5-A3B on the llama.cpp backend; decision: accept breakage, ride out via moeflux
metadata:
  type: project
---

# Qwen3.5-A3B cross-position cache miss on llama.cpp backend

## Symptom

Within-phase same-prompt-extension turns hit the prefix cache (auto-tip
reuse, ~99% read ratio). Phase transitions (think→reflect, reflect→evolve,
agent N → agent N+1) miss entirely (`cache_read=0`) even though the
hash-keyed lookup correctly identifies a content-matching cached position.

Log shape:
```
DEBUG hash-keyed prefix-reuse: cached position matched ... hash_picked=7590 prev_len=17107 new_len=17219
DEBUG checkpoint missing; falling back to full reprefill
      cache_read=7590 error="backend cannot restore to position 7590"
```

## Cause

`llama_memory_recurrent::seq_rm` in
`external/llama.cpp/src/llama-memory-recurrent.cpp:155-169` rejects any
partial-end truncate:

```cpp
// models like Mamba or RWKV can't have a state partially erased at the end
// of the sequence because their state isn't preserved for previous tokens
if (0 < p0 && p0 <= cell.pos && p1 > cell.pos) {
    return false;
}
```

Plain reading: "p0 lies inside the live sequence AND p1 extends past the
live tail" — exactly the shape of a rewind. The hybrid backend
(`llama-memory-hybrid.cpp:132-138`) calls the recurrent path first and
short-circuits to false on its failure, so any hybrid model with a
recurrent component (Qwen3.5-A3B's linear-attn layers route through
`mem_recr`) rejects all rewinds at the C boundary.

Auto-tip extension wins because drama_llama's `internal_tip` is set to
`kv_len + 1` (past the close marker recorded but not committed), so
`p0 > cell.pos`, the rejection condition is false, the cells loop iterates
nothing, returns true. No real rewind ever happens on the hit path.

Hash-keyed rewinds to early breakpoints (system+tools at ~7590 when
engine is at ~17107) always trip the rejection.

## Why other models cache fine

Pure transformer models route through `llama_kv_cache::seq_rm` (no
recurrent component), which has no equivalent rejection.

## Decision

Accept the breakage on the llama.cpp backend. moeflux's
`Decoder::checkpoint_pos` already snapshots per-position state via
`state_save` and `restore_to` uses `state_load` — partial rewinds work.
Even with disk-streamed weights, moeflux is ~2× faster than llama.cpp
on A3B, so the path forward is moeflux parity for the agora workload,
not a llama.cpp fork.

Fixing in llama.cpp would mean:
- Maintain a fork forever, or
- Submit upstream with no acceptance guarantee, AND
- Add a large chunk of C++ to a project where C/C++ has been a
  reliability sink

Neither pays.

## What landed in drama_llama this session

* `chat_template.rs`: permissive minijinja env for partial renders.
  Qwen3's `raise_exception('No user query found in messages.')` would
  silently drop the `AfterSystem` partial render every call, nuking the
  front-of-prompt cache anchor across agents. Permissive env binds
  `raise_exception` to a no-op (logs at debug) and sets
  `UndefinedBehavior::Chainable` so `messages[0].role` on an empty list
  returns undefined rather than erroring. `render_partial` for
  `AfterTools` now carries `system` content too — without it Qwen3's
  `messages[::-1]` walks an empty list and minijinja panics.
* `chat_template.rs`: warn-log on dropped partials at the
  `render_with_breakpoints` call site.
* `session/parse.rs`: debug-log on malformed tool_call JSON fallback.
* `sample.rs`: warn-log on `JsonState` mutex poison during
  serialization.
* `session/mod.rs`: debug-log on `forget_pos` failures (orphan
  reclamation + auto-tip eviction paths).

## What's still on the table (not done this session)

For when someone wants to revisit llama.cpp parity:

1. **State-aware checkpoint/restore** using `llama_state_seq_get_data` /
   `llama_state_seq_set_data` (`llama.h:825-841`). Snapshot recurrent
   state to an in-memory LRU keyed by position, use `set_data` instead
   of `seq_rm` for rewinds. `llama_state_seq_get_size_ext` (line 869)
   gives the snapshot size for budgeting. Real implementation path,
   sizeable change.
2. **Forward-only gate on recurrent backends** using
   `llama_model_is_recurrent` (`llama.h:616`). Filter hash-keyed
   candidates to positions ≥ engine's current `memory_seq_pos_max` on
   recurrent-flagged models. Smaller change. Stops the
   wasted-rewind-then-fail cycle. Doesn't preclude (1) later.
3. **Hash-invalidation on restore failure.** When `restore_to` errors,
   prune the offending hash entries from `prev_breakpoint_hashes` /
   `prev_tip_hash` so the next call doesn't re-pick the same broken
   position. Tiny bookkeeping change, prevents the same miss being
   paid for repeatedly.
4. **Promote `BackendUnsupported` log to `warn`** (currently `debug`).
   Without that promotion, the actual failure mode is invisible at
   default log levels and looks identical to a hash miss in the stats
   stream.

(2) + (3) + (4) are the contained immediate-relief patch; (1) is the
real parity work.

## References

* llama.cpp source: `~/Projects/llama-cpp-sys/external/llama.cpp/`
* recurrent rejection: `src/llama-memory-recurrent.cpp:143-181`
* hybrid forward: `src/llama-memory-hybrid.cpp:132-138`
* drama_llama llama.cpp decoder (the no-op checkpoint that needs
  changing if anyone implements (1)): `src/llama_cpp/decoder.rs:491`
* drama_llama moeflux decoder (the working state-save/load pair):
  `src/moeflux/decoder.rs:241-265`

See also: [[qwen_cmdbuf_consolidation_plan]] for the next perf-session
plan-of-record (moeflux prefill speed parity with llama.cpp on A3B).
