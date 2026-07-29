---
name: plan-cache-restore-ladder-and-disk-tier
description: Arc plan-of-record — restore-to-divergence ladder, final-message auto-anchor, tiered disk cache for prefix slots, #100 gated last. Decisions + rationale from the 2026-07-29 design session.
metadata:
  type: project
---

# Plan: cache restore ladder + tiered disk cache (arc plan-of-record)

**Issues: #102 (Phase 1), #103 (Phase 2), #95 (Phase 3), #104
(Phase 4), #100 (Phase 5, gated).**

**2026-07-29, Mike + Claude design session.** Successor to the #96/#88
cache work. Absorbs and supersedes `future_work_slot_kv_sizing.md`
(deleted per the delete-resolved-memos rule); the measured disk
economics stay in [[future-work-kv-disk-offload]]. Per-session
execution: each phase gets its own chat + formal plan (Plan-agent
vetting) before code. This memo is the arc-level record.

## Diagnosis that motivated the arc (2026-07-29 repro bundle)

Bundle: `repro-20260729.tar.gz` from balerion
(`claude-agora@balerion:~/agents/agora/logs/run_20260729/`), 8 pairs +
31 ordered turns + `replay_session.py` (exit 1 on any miss — the
bisect/verification gate). **Sensitive: system prompts carry agent
SOULs/memory. Never attach to a public issue.**

All eight collapses are ONE mechanism, already filed as #100:

- The reuse point is only ever a client `cache_control` breakpoint or
  the #96 tip — never the divergence token the LCP walk already found
  (`compute_l_hit`, `src/session/mod.rs:1836`: `safe = lcp - 1`, then
  `.rev().find(|bp| bp.entry <= safe)`; `safe` itself is never a
  candidate).
- Six collapses: reflect-phase text block appended to the final user
  message (which holds a tool_result). Divergence lands inside/at the
  end of that message; deepest anchor = previous message boundary;
  **deficit == rendered size of the whole final message** (measured
  chars/deficit = 3.45–4.63 on every pair, four models).
- This dissolves two of balerion's readings: the "distinct Qwen
  54-token issue" is the same bug with a 96-char tool_result, and
  "Qwen is the outlier for large collapses" is just Qwen having 65KB
  tool_results.
- The two gpt-oss "separate bugs" were the same quantization: the
  bundle's "#85-shape mutated assistant" was a sliding `cache_control`
  marker (renders identically — balerion diffed JSON, not renders).

**Render-diff verification** (methodology worth keeping: dump the
GGUF/sidecar template, `ChatTemplate::from_source`, deserialize both
captured request bodies as `Prompt`, render, diff bytes): primer's
full input render is a byte-prefix of the miss render to 99.9–100.0%
on all four models. **Messages round-trip. The entire loss is lookup
quantization.**

Facts established on the way (each was a live hypothesis; now settled):

- The Agora runner's markers are correct: `breakpoint_after_assistant`
  (agentkit `src/reactor/inference.rs:21`) puts markers on assistant
  turns; verified on the wire. No client marker placement can fix the
  append shape.
- Qwen's baked Unsloth template DOES carry the upstream conditional
  think-strip (`last_query_index` walk) — but `Session` passes
  `preserve_thinking=true` by default (`src/session/mod.rs:2623`,
  re-injected at `:2907`) and that defuses it. Measured: without the
  flag the reflect append flips the whole history's render (byte LCP
  30.3%); with it, 100.0%. **No Qwen template patch needed.**
- Grammar canonicity (#97) is NOT implicated in these misses; Qwen
  tool-call turns show tip hits (negative deficits) all through the
  capture.
- #98's "continuation renders smaller" is refuted for the input
  prefix (gpt-oss re-ingest is a byte-exact continuation, LCP 100%);
  what remains of #98 is emission-vs-re-render tip stability only.
- `append_message` (`src/chat_template.rs:896`) NEVER merges
  consecutive same-role messages. It splits user-message tool_result
  blocks into their own "tool" messages and collects text into a
  trailing user message. Consequence, uniform across templates:
  **tool_result+text appends diverge AT the message boundary** (close
  token never moves — why the reflect shape is cheap to anchor);
  **text+text appends merge into one turn** (close token moves —
  divergence is intra-turn; gpt-oss 0070, the original seat_phase
  capture). This split/merge rule is ours, not per-template, so #100's
  "expressible with all models?" worry is retired.
- `checkpoint_pos` is a hard no-op on pure transformers
  (`seq_snapshots_enabled = is_recurrent || is_hybrid`,
  `src/llama_cpp/decoder.rs:319`); truncate-restore works at ANY
  position there. Hybrids (Qwen A3B) restore only at snapshot
  positions.
- Live bug found in passing: `NoCheckpoint` → `reset_slot` → ZERO
  reuse (`src/session/mod.rs:4245`) instead of falling back to the
  next-lower anchor.

## The phases

### Phase 1 — restore ladder + LCP walkback (pure transformers)

Add the walked divergence point (`safe`) as a first-class restore
candidate. Restore becomes an ordered ladder: try the best candidate;
on failure (`NoCheckpoint` on hybrids) fall to the next checkpointed
anchor — never to zero. Fixes 7/8 bundle reproducers (all but Qwen)
and every future intra-message append on pure transformers. Zero new
state. The ladder shape is deliberately the socket the disk tier
plugs into later (disk = one more rung).

Invariant: does NOT touch the two refusals that must never weaken —
non-canonical emission and #91 segmentation drift. Those guard hash
trust; the walk is token-verified.

Gate: `replay_session.py` on the bundle — Mistral/cogito/gpt-oss
pairs go deficit≈0.

### Phase 2 — internal auto-breakpoint at end-of-final-message

Tip-like internal anchor (no 4-marker budget slot) at the final
message's end, every call, via the existing whole-message partial
render + `checkpoint_pos` machinery. This is what fixes Qwen: on
hybrids the checkpoint takes a real snapshot exactly where the
reflect-shape append diverges. Also densifies snapshot positions to
per-turn granularity — precisely what the disk tier wants to persist.

Includes the **#100 feasibility spike**: per-model pin tests for both
append shapes (tool_result+text and text+text) asserting the
byte-prefix / divergence-location property. Answers "expressible with
all models" without building #100.

Watch: `SnapshotStore` is count-capped at 16 with no byte budget
(`src/snapshot_store.rs:14`) and hybrid snapshots are full sequence
states; per-turn checkpoints press on that. The disk tier relieves it
(write-behind makes RAM a true LRU over a disk-backed set); interim,
may need a byte-aware cap.

Gate: 8/8 bundle reproducers ≈0, including Qwen reflect.

### Phase 3 — #95 shutdown: logging now, fix tabled (Mike, this session)

`with_graceful_shutdown` is ALREADY in place (`blallama.rs:370`);
`shutdown_signal` looks textbook. Suspects found by code reading:
(a) `/probe` SSE — infinite `BroadcastStream`, sender owned by the
router, hyper graceful drain waits for the response to complete, SSE
never completes → cyclic wait (axum under-documents this; ending
long-lived streams on the shutdown signal is the accepted pattern);
(b) generation in `spawn_blocking` is non-cancellable and runtime
teardown waits on the blocking pool; (c) `spawn_blocking_or_bust`
calls `process::exit(1)` — unclean by construction. Not reliably
reproducible → **stage logging landed this session** (signal received
→ serve returned → main exiting), which names the stuck stage on next
occurrence. Manual confirm available: open `/probe`, Ctrl-C, observe.
Policy decided: **in-flight generation finishes, then exit**.

### Phase 4 — tiered disk cache for prefix slots

The big one. Design decided this session:

- **Placement** (Mike's lean, agreed with one refinement): hook trait
  in `Session` (`put`/`get`/`forget`/`clear` over keyed archives),
  implementation in blallama (worker, `--cache-dir`, lifecycle). The
  **archive format and load-time validation are library-owned** —
  #91's lesson: a loaded blob must pass the same entry-agreement
  refusal as the RAM path, so stale/torn/wrong-model degrades to a
  miss, never a wrong restore. In-memory impl ships in-crate for
  tests. `Session` currently has zero hook surface (only `ProbeHook`,
  on `Engine`) — this is a new trait.
- **Write trigger splits by backend capability**:
  - Hybrids: **write-behind at checkpoint time** — the bytes already
    exist in `SnapshotStore`; the worker persists them and RAM
    becomes a true LRU over the disk set (also fixes the silent
    RAM-cap drop → later `NoCheckpoint` → reset). Tip excluded from
    disk: breakpoints are write-once-stable, the tip churns a full
    blob per call (write amplification); an evicted conversation
    restoring at its last breakpoint loses one turn, not a 36 s
    re-prefill.
  - Pure transformers: **capture once at eviction**
    (`get_state_seq` just before `free_slot_engine_state`,
    `src/session/mod.rs:4548`) — ONE deepest blob per conversation;
    all shallower restore points derive by load-then-truncate.
    Nothing stored twice within a conversation.
- **Keying**: `(model fingerprint, format epoch, partial-hash, pos)`.
  The breakpoint SHA-256 side-table already content-addresses partial
  renders → cross-slot dedup free where blobs are identical (the
  shared system+tools floor becomes one blob per model; a fresh agent
  cycle can disk-restore the floor). Epoch = crate version +
  llama.cpp state-format version + model identity, so upgrades
  invalidate by construction. Git sha is the human-facing component,
  surfaced via a new **`/health`** endpoint (version + sha + model)
  so agentkit can log it.
- **Integrity**: write `*.tmp` → fsync → atomic rename; loader checks
  version + checksum; invalid = delete + miss. Reader-vs-reaper is
  safe by POSIX open-fd semantics.
- **Backpressure** (Mike): bounded queue **depth 1–2**,
  `on_full: Block | Drop` configurable, **default Block** (a disk
  write is almost always cheaper than the miss it prevents), plus a
  RAM-headroom guard before allocating a capture blob (can't allocate
  safely → drop the write). Content-key check skips re-enqueueing
  what's already on disk. OOM is a real risk on the big models —
  capture blobs are multi-GB for the 119B/120B dense models.
- **TTL / cleanup**: per-blob TTL from the creating marker (5m/1h
  respected), **mtime-based, plus a startup sweep** — correctness
  never depends on any live process's bookkeeping (Mike's hazard
  note; unclean death just delays cleanup to next start).
  **Clear-on-graceful-exit is the default** (privacy: archives carry
  token ids — they ARE the conversation, SOULs included; 0700 dir).
  Persist-across-restart is technically sound but low-value (restarts
  are upgrades → epoch change) — future opt-in flag.
- **Pre-work**: serde derives for `CacheEntry`/`Breakpoint`/`EntryPos`;
  `PrefixSlot`'s two `Instant`s → wall-clock (`created` already
  carries a "future disk cache" doc comment,
  `src/session/mod.rs:694`). Measure `get_state_seq` copy cost on
  unified memory BEFORE committing (memo caveat).
- **v1.5, not core**: sub-blob floor dedup via
  `memory_seq_cp(src, scratch, 0, floor_pos)` (unified KV adds
  membership without copying cells) + `get_state_seq(scratch)` to
  serialize just the floor once.
- **Deferred**: residency pinning / fairness — the API can't express
  it yet and agents run one-at-a-time today; revisit when Mike
  enables free agent swapping (stated goal). Single-resident +
  ~1 s disk restore beats 4-way pool thrash for the current workload.

Economics: see [[future-work-kv-disk-offload]] (measured: 20 KiB/cell
Qwen-A3B, ~1 s I/O vs ~36 s re-prefill for 32k cells; per-cell cost is
model-specific, dense models several ×).

### Phase 5 — #100 block-granular anchors, gated on telemetry

After Phases 1–2, #100's remaining beneficiary is exactly one shape:
**text-onto-text appends on hybrids** (walkback covers them on pure
transformers; the reflect shape diverges at the message boundary and
Phase 2 covers it everywhere). Re-run seed telemetry after Phase 2;
build #100 only if the shape still spends tokens. Its design is sound
against the unclosed-partial problem (block byte-spans mapped from
the FULL render; LCP-only anchors, no hash column — partials are
never needed). `misanthropic::prompt::Index`/`BlockIndex` verified
shipped in 1.0.0-alpha.13 (we pin alpha.12 — bump rides along).

## Standing constraints

- Do not weaken the non-canonical-emission refusal or #91's
  drift refusal (`plan_grammar_canonicity.md` stop sign).
- `replay_session.py --seed 42` against the bundle is the arc's
  regression gate; blallama runs on the Mac
  (`~/.cargo/bin/blallama`, models in `~/Projects/drama_llama/models`),
  balerion only runs the seed runner.
- Correctness replays are Claude-run; perf-sensitive GPU runs are
  Mike's.
