---
name: swa-checkpoints
description: Why sliding-window and hybrid models rewind through partial checkpoints, the masked-cell recycling that made swa_full useless, the KV / checkpoint memory per fleet model, and what still needs a GPU window.
metadata:
  type: project
---

# SWA / hybrid checkpoints (branch fix/swa-checkpoints, 2026-10-01)

Commits: `05bf41b` (checkpoint rules + decoder + Session pins),
`e2c1d7c` (`swa_full = false` default + `--swa-full`). Branched from
dev `65a2230`; not pushed, not deployed.

**Read before touching `llama_cpp::checkpoint`, `LlamaCppDecoder::restore_to`,
`swa_full`, or the snapshot cap.**

## The live failure

gpt-oss (iSWA, window 128), cohort night 2026-10-01, twice (02:34:32Z,
07:49:24Z): `restore_failed`, hash hit at entry 10088, "no checkpoint
stored at position 10088", fallback 0, 10k tokens re-prefilled. The same
anchor had restored fine at 02:30 and 02:32 — between those and 02:34
another slot decoded ~30k tokens.

Cause, from llama.cpp source (b11123, `llama-kv-cache.cpp` `find_slot`):
a cell is usable when empty **or SWA-masked for its owner's head**, so a
batch for sequence B recycles the window cells of an idle sequence A
below A's window. This holds with `swa_full = true` too — the full-size
cache only makes it rarer. `apply_ubatch` then purges the owner's lower
positions to keep `[pos_min, pos_max]` contiguous. So after A's
truncate to its anchor, either the head is gone (`NoCheckpoint`, what we
saw) or **the head is there but part of the window is not** — which the
old `restore_to` (head check only) called a lossless rewind. That second
case is silent KV corruption; it needs `pos_min <= pos - n_swa`.

Nothing was checkpointed because the decoder only snapshotted
`is_recurrent || is_hybrid`. gpt-oss/Gemma 4 counted as dense.

## The design (matches llama-server's context checkpoints)

`Checkpointing::for_model(recurrent, hybrid, n_swa)`:
dense → `Off`; SWA or recurrent or hybrid-with-dense-attention →
`Partial`; hybrid whose attention also slides → `Whole` (llama.cpp's
plain hybrid memory omits attention from a PARTIAL state entirely).

- `Partial` = `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` (value 1, a `#define`
  bindgen drops): iSWA writes only the SWA cells unmasked for the head
  (n_swa cells); hybrid writes only the recurrent state. Restore = load
  partial, **then** `seq_rm(pos, -1)` (a recurrent layer refuses the
  truncate until its state is back at `pos`), then check head.
- A partial checkpoint at P is valid only while the dense KV `[0, P)`
  is the one it was taken over → every trait `memory_seq_rm/cp/keep`,
  `set_state`, `set_state_seq` drops partial checkpoints above the
  change; a failed restore's truncate drops those above `pos`. Since
  the review follow-up the inherent `memory_*` mutators (incl.
  `seq_add/div`) are `&mut self` and invalidate too: they shadowed the
  trait methods and were reachable via `Engine::vision_and_decoder` —
  on iSWA a raw clear + stale window checkpoint "restored" over empty
  dense layers (iSWA pos_max reports kv_swa only). Only
  `context_ptr_mut` bypasses now.
- Checkpoints are only taken when `pos == pos_max + 1`.
- Gotcha: `llama_model_n_swa` is a hyperparameter and a `vocab_only`
  load skips hyperparameters (`load_hparams` returns early) — it reads
  `0` there. `is_hybrid` / `is_recurrent` are arch-based and do answer.
  The vocab-only fleet test reads `{arch}.attention.sliding_window`.
- Truncate-first is kept: when the window survived it is free. The
  intact check is `pos_min <= pos - n_swa + 1` (`is_masked_swa`: the
  token at pos reads `[pos-n_swa+1, pos)`; find_slot recycles
  `pos - n_swa`), NOT the server's one-cell margin — that margin cell
  is the first one a neighbour takes. Contiguity from pos_min is
  llama.cpp's `apply_ubatch` invariant.
- A truncate-success restore in `Partial` mode takes the missing
  checkpoint there (head == pos); `checkpoint()` on a stored Partial
  one only refreshes LRU (present ⇒ valid, by invalidation). Session's
  `checkpoint_at` stays strictly above the restored entry on purpose:
  re-checkpointing there would cost moeflux / forced-Whole a full-state
  copy every call; the `restore_to` trait doc promises the anchor stays
  restorable instead. The Session mock asserts checkpoint pos == head.
- Byte budget (`CheckpointBudget`, default 8 GiB total / 4 GiB per
  seq; `--checkpoint-mib` / `--checkpoint-slot-mib`): LRU by bytes, a
  per-seq overrun evicts only that seq, an oversized single checkpoint
  is refused (`snapshot_over_budget`). Gemma 4 ⇒ ≤ 10 checkpoints.
  Each seq's LOWEST snapshot (the system/tools anchor) is evicted last,
  under every bound (`SnapshotStore::evict_oldest`): restores refresh
  only the anchor they land on, so pure LRU evicted it first. Peak =
  budget + one checkpoint (serialized before the store evicts), and
  the budget is per decoder, not per process.
- `memory_seq_add` / `memory_seq_div` invalidate from where cells land
  (`p0 + delta` for a shift back, `p0 / d`), not `p0`.
- DEBUG logs under `drama_llama::snapshot_store`: `checkpoint_restore`
  (`restore_via` truncate|checkpoint, bytes, ms), `checkpoint_taken`
  (bytes, ms), `checkpoint_invalidated` (count; `memory_clear` and
  `force(false)` too; Whole mode drops nothing, so logs nothing).
- `default_context_params()` sets `swa_full = false` too, so
  `LlamaCppEngine::new(path, None, None, …)` agrees with the options.

## Memory, from GGUF metadata (f16 KV, f32 recurrent; n_ctx 131072, 4 slots, n_ubatch 512)

| model | per-token KV | swa_full=true | swa_full=false | checkpoint | ×24 (cap) |
|---|---|---|---|---|---|
| gpt-oss-120b (18/36 SWA, 8×64) | 72 KiB | 9.0 GiB | 4.5 GiB + 36 MiB | 4.5 MiB (128 cells) | 108 MiB |
| Gemma 4 31B (50/60 SWA 16×256, 10 full 4×512) | 880 KiB | ≈110 GiB (OOM) | 10 GiB + 3.5 GiB (4608 cells) | 800 MiB (1024 cells) | ≈19 GiB, budget caps 8 GiB |
| Qwen3.6-35B-A3B (10 attn 2×256, 30 GDN) | 20 KiB | — | — | 63 MiB recurrent (was ≈650 MiB at 30k) | 1.5 GiB |
| Qwen3.8-27B (16 attn 4×256, 48 GDN) | 64 KiB | — | — | 150 MiB recurrent (was ≈2 GiB at 30k) | 3.6 GiB |
| cogito-32b, Mistral 4 | dense | — | — | none (`Off`) | 0 |

Pinned by `llama_cpp::checkpoint::tests::the_fleet_checkpoints_as_its_layers_need`
(`#[ignore]`d, vocab-only, CPU-safe; passed 6/6 on 2026-10-01 with
`DRAMA_LLAMA_MODEL_DIR=~/Projects/drama_llama/models`).

**Qwen3.8 is hybrid** (`qwen35` arch, `ssm.*` keys, `full_attention_interval
4`) — not a pure transformer as the cohort notes assumed. It was already
snapshotting (whole) before this branch.

SWA cache size formula (llama-kv-cache-iswa.cpp):
`pad256(min(n_ctx, n_swa * (unified ? n_seq_max : 1) + n_ubatch))`.

## swa_full decision

Default `false` for everything (`LlamaCppOptions::swa_full: None`), the one
documented exception to "unset = llama.cpp's library default"; it is what
llama-server/common default to. gpt-oss too: the full-size cache does not
guarantee truncation (recycling), so checkpoints are needed regardless,
and once they exist the full cache buys only the rare "checkpoint LRU-
evicted but cells survived" case — for 4.5 GiB. `--swa-full` is the escape.

## Open — GPU window (all `#[ignore]`d, `tests/swa_checkpoint.rs`)

Run one family with `just test swa_checkpoint <test filter>` (the second
argument narrows by test name inside the suite; it used to be dropped,
so every run loaded every default model): e.g.
`DRAMA_LLAMA_SWA_MODEL=… ~/.local/bin/serial -n gpu just test swa_checkpoint swa_`.

- `swa_rewind_survives_a_neighbour_recycling_the_window`,
  `swa_truncate_never_claims_a_partial_window`,
  `swa_rewind_to_the_head_is_a_plain_truncate`,
  `swa_truncate_rewind_takes_the_missing_checkpoint`,
  `swa_checkpoints_are_window_sized_and_budgeted` (gpt-oss; rerun with
  `DRAMA_LLAMA_SWA_MODEL=…/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf` — n_ctx
  is now sized from the GGUF window, 12288 for Gemma),
  `dense_models_stay_off` (cogito + Mistral 4 loads, CPU-only so the
  74 GB stays mmapped, not on Metal),
  `hybrid_partial_checkpoints_rewind_each_anchor`,
  `hybrid_checkpoint_dies_with_its_sequence` (Qwen3.6; rerun with
  `DRAMA_LLAMA_HYBRID_MODEL=models/Qwen3.8-27B-UD-Q8_K_XL.gguf`).
- Existing `tests/state_snapshot.rs`: on model.gguf (Qwen3.6, now
  `Partial` natively) `forced_snapshot_restores_after_kv_wipe` skips the
  wipe (a wipe rightly kills a partial checkpoint) and rewinds through
  the checkpoint after generation; the per-model session suites,
  `session_cache`, `tip_invariant`.
- Perf: gpt-oss / Gemma 4 tok/s with swa_full false vs true at
  `--cache-slots 4` (llama.h's "bad performance in some cases").
- Live: Gemma 4 at 131k / 4 slots should now load (≈ 13.5 GiB KV).

Risks to watch once deployed:

- Gemma 4 checkpoints are ≈ 800 MiB each (1024 window cells × 50
  layers), copied device→host at every breakpoint crossed and at the
  tip: up to ≈ 19 GiB of host (= unified) memory at 4 slots, and some
  per-call latency. Now capped by the 8 GiB / 4 GiB-per-slot budget
  (≈ 10 / 5 checkpoints); measure the copy cost via `checkpoint_taken`
  `ms`, and watch `snapshot_evicted bound=bytes|seq_bytes`.
- With a window-sized cache, a truncate almost never suffices on SWA
  models, so an anchor whose checkpoint the LRU dropped is now a real
  `restore_failed` (falls one rung) where the full-size cache often
  saved it. Watch `snapshot_evicted` vs `restore_failed` in the cohort
  log.

Gap not closed: a breakpoint at exactly the prompt's end (no generation
prompt — assistant prefill) is not checkpointed (`checkpoint_at` stops
at `suffix_start`); on Partial models that anchor fails and the ladder
falls lower. Needs a predictor hook after the suffix prefill.
