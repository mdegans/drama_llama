# Phase 7 — typed `memory_seq_rm` (deferred to next session)

**Captured 2026-04-28** at end of Phase 6 session, after surfacing
that moeflux's prompt cache is silently lossy.

## What's broken

Drama_llama's `Session::kv_setup_for_call` (`src/session/mod.rs:894`)
calls `engine.memory_seq_rm(0, l_hit, -1)` on partial cache hit. For
moeflux, this routes through `RsCtx::memory_seq_rm` →
`state::truncate` (`crates/moeflux/src/riir/state.rs:154`), which:

- Truncates full-attn KV cleanly to `[0, l_hit)` ✓
- Resets every linear-attn layer's recurrence to empty ✗
  (faithful port of the C-side `Option A` semantic — the
  `(conv_state, ssm_state)` pair folds the entire history; can't
  unwind to an arbitrary position)

Then drama_llama re-prefills `tokens[l_hit..]`. End-of-prefill
linear-attn state reflects only `[l_hit, end)`, not `[0, end)`.
**Decode after prefill produces silently-incorrect outputs on every
partial cache hit.** Full miss (`l_hit == 0` → `memory_clear`) and
full hit (no truncate needed) are correct; everything in between
degrades.

`memory_seq_rm` returns `bool` (matching C / llama.cpp) — caller
has no way to learn the truncation was lossy.

## The fix (this slice)

Make the lossy condition observable via a typed error so callers
can choose to fall back to full-miss instead.

### moeflux side

`crates/moeflux/src/riir/state.rs`:
- Introduce `CannotTruncateLinear { p0: i32, pos_max: i32 }` error.
- Change `pub fn truncate(layers, p0, p1)` to return
  `Result<(), CannotTruncateLinear>`. The existing lossy reset is
  replaced by: detect "would be lossy" (any linear-attn layer has
  some history AND `p0 > 0` AND `p0 < pos_max`); if so, return
  `Err` *without* mutating state. Caller that wants the
  current-behavior gets it via a separate
  `truncate_lossy(layers, p0, p1)`.

`crates/moeflux/src/riir/mod.rs:1303` (`RsCtx::memory_seq_rm`):
- Change signature: `pub fn memory_seq_rm(&mut self, seq_id, p0, p1)
  -> Result<(), CannotTruncateLinear>`.
- Body: forward to `truncate`, propagate error.

Test (`tests/diff_oracle.rs` and/or new
`tests/typed_memory_seq_rm.rs`):
- partial truncate on linear-attn returns Err and doesn't mutate
- full clear (`p0 = 0`) returns Ok and clears
- full hit / no-op (`p0 = pos_max`) returns Ok and is no-op
- truncate to position equal to or past current pos_max (no-op
  case) returns Ok

### drama_llama side

`src/backend.rs` (`Decoder` trait):
- Define `pub enum MemoryRmError { CannotTruncate { p0, pos_max } }`.
- Change `fn memory_seq_rm(&mut self, ...) -> bool` to return
  `Result<(), MemoryRmError>`.
- llama-cpp impl: wraps `bool` — `if ok { Ok(()) } else { Err(...) }`.
  llama.cpp returns `false` only when seq_id is invalid; map to a
  generic variant or reuse `CannotTruncate` with sentinel values.
  Cleaner: introduce `MemoryRmError::InvalidSeqId` and use it.
- moeflux impl: forward `RsCtx::memory_seq_rm`'s
  `CannotTruncateLinear` into `MemoryRmError::CannotTruncate`.

`src/engine.rs:80` (`Engine::memory_seq_rm` forward):
- Update to return the new Result.

`src/session/mod.rs:894` (`kv_setup_for_call`):
- After computing `l_hit`, attempt `engine.memory_seq_rm(0, l_hit, -1)`.
- On `Err(MemoryRmError::CannotTruncate { .. })`: log a one-line
  `tracing::debug!` ("partial cache hit lossy on this backend; full
  reprefill"), call `engine.memory_clear()`, set `l_hit = 0`,
  recompute `suffix = new_tokens.to_vec()`.
- On `Err(InvalidSeqId)`: same fallback (correctness via clear).
- On `Ok(())`: proceed as today.

`src/llama_cpp/decoder.rs:471` and `src/moeflux/decoder.rs:166`:
- Implement the new trait signature.

### Test (new)

Integration test verifying prefix-cache correctness on moeflux:
- Open moeflux Session
- Send prompt A (long), capture token sequence
- Send prompt A again (full hit) — should be identical (no decode
  required, just sample)
- Send prompt B = prompt-A + new-suffix (full hit on the prefix +
  new tokens)
- Send prompt C = prompt-A[..len-N] + different-suffix (partial
  hit) — assert output equals "fresh session, no cache" output
  (i.e., the fallback path activated and produced correct results)

That last assertion is the load-bearing one. Compare against a
fresh `Session::from_path(...)` running the same prompt without
cache. They must agree.

## Publish flow

1. Bump workspace 0.1.0-pre.1 → 0.1.0-pre.2
2. `cargo publish -p moeflux-sys --features model-qwen3-6-35b-a3b`
3. `cargo publish -p moeflux --features model-qwen3-6-35b-a3b`
4. drama_llama `Cargo.toml` bumps `moeflux` to `=0.1.0-pre.2`
5. Trait + caller updates land in same drama_llama commit

## Out of scope (Phase 8+)

- Proper checkpoint-and-restore so partial linear-attn truncation
  is non-lossy. That's a real feature: snapshot every K tokens,
  on partial-hit walk back to the nearest snapshot ≤ l_hit, restore,
  re-decode the gap. Big change; not this slice.

## Known pre-existing test guards

`tests/moeflux_session_pollution.rs` covers the bisect findings.
The "Original symptom B/C doesn't currently reproduce" memory note
is about a different surface (cross-Ctx state pollution), not the
linear-attn-truncate bug. The new test above is the missing
prefix-cache correctness guard.

## Why now (next session)

- Real correctness bug, not type beautification.
- Fix is mechanical given the design above.
- Diff oracle catches any porting mistakes on the moeflux side.
- Without this, every multi-turn moeflux conversation degrades
  silently — directly affects the Council use case.
