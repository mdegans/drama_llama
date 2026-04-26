# blallama / moeflux Session-state pollution across requests

**Status (updated 2026-04-27)**: investigated. Two real upstream
bugs identified by bisect, neither is the originally-reported
blallama symptom — that one does not currently reproduce in-process.
Fix path is the upstream RIIR, not a C-side patch. See
`riir_moeflux_strategy.md` for the rewrite plan.

**Priority**: superseded by RIIR. The bisect tests (in moeflux at
`crates/moeflux/tests/consecutive_eval_prompt.rs` and in drama_llama
at `tests/moeflux_session_pollution.rs`) become regressions that
the Rust port must pass.

## Findings from the 2026-04-27 bisect

Tests in `moeflux::crates/moeflux/tests/consecutive_eval_prompt.rs`:

- **`memory_clear` is fine within a single Ctx.** Both same-prompt
  and dirty-decode-then-different-prompt scenarios pass. The
  original `g_deferred`-not-reset hypothesis is refuted for
  intra-Ctx use.
- **Cross-Ctx state survives `mf_free_model`** — the second
  `Ctx::open` after a drop produces all-NaN logits. Process-global
  state (likely `g_deferred` holding a dangling pointer into the
  freed `ctx->hidden`) leaks across instances. Real bug, but
  blallama only opens one Ctx so it's not the user-visible symptom.
- **Resuming prefill diverges from full prefill.** `mf_memory_seq_rm(0,
  l_hit, -1)` followed by `eval_prompt(suffix, start_pos=l_hit, …)`
  produces a verbatim-loop trajectory on synthetic tokens; full
  prefill of the same final tokens does not. Per `moeflux.h` design
  notes, partial truncation of linear-attention layers resets
  recurrence state *by design* — the lossy semantic is documented
  upstream, but the re-prefilled state still diverges from a fresh
  one. With real chat-template prompts and 600-token essay outputs
  this divergence does not surface as user-visible degeneration
  (drama_llama's `tests/moeflux_session_pollution.rs` passes).

## Why the original symptom B/C don't reproduce now

Captured 2026-04-26, no longer reproducible 2026-04-27 with the
same A3B model and similar prompt shape. Possible explanations:

- The `MAX_K 8 → 16` moeflux fix (commit `d013a0b`, 2026-04-25,
  landed *just before* the symptom catalog was written) addressed
  silent expert-drop on K>10 models. A3B at K=8 wasn't supposed to
  trigger it, but the surrounding accounting may have been
  affected.
- The `repetition: None` default in v0.8.0 sidecars (commit
  `51fa347`) changed the sampling-chain composition. Memory note
  argued this was "unrelated" to the bug, but the 8-token loop in
  symptom C is exactly the regime rep-penalty bounds.
- Fabricated low-id token streams in the bisect amplify numerical
  drift; real semantic prompts may stay coherent through the same
  state divergence.

The drama_llama-side regression test (`tests/moeflux_session_pollution.rs`)
stays in-tree as a forward-looking guard: it currently passes with
the 600-token essay scenario, and any future regression of symptom
B or C will trip it.

## Three observed symptoms

The third was added 2026-04-26 evening during A3B sidecar tuning;
it strengthens the case that this is one bug presenting differently
depending on prompt content rather than three separate bugs.

### Symptom A — stall on second request after a cancelled first

1. Start blallama on A17B (moeflux backend).
2. Send essay request.
3. Mid-request, kill the curl client (`pkill -f "curl.*11435"`). The
   HTTP handler keeps running (no cancellation hook), eventually
   completes, returns the Session to the lock.
4. Send a second essay request.
5. Second request stalls indefinitely. `sample <pid>` shows the main
   thread parked in `[_MTLCommandBuffer waitUntilCompleted]` inside
   `moeflux::imp::Ctx::eval_prompt → mf_eval_prompt → mf_step_internal`.
   GPU usage is 0%, single thread is at 100% CPU.
6. Killing blallama, restarting fresh, sending same request → works
   normally (~2.3 tok/s).

### Symptom C — verbatim paragraph loop on second-different-prompt request

Discovered on A3B during default-validation tuning:

1. Start blallama on A3B (moeflux backend, just rebuilt with the
   `moeflux-model-qwen3-6-35b-a3b` feature).
2. Send Apollo essay → 600 tokens, clean output, ~65s.
3. Send "history of the early internet" essay (same Session) →
   HTTP 200 in ~58s, but the body contains the same 40-token
   paragraph repeated 7 times verbatim:

   > "The early internet, from ARPANET to the World Wide Web,
   > traces back to 1969, when ARPANET first connected four nodes:
   > UCLA, Stanford Research Institute, the University of Utah, and
   > the University of California, Santa Barbara. This initial
   > link was fragile, relying on slow transmission speeds and
   > limited data capacity."

4. Killing blallama, restarting fresh, sending the same internet
   prompt → 600 tokens, clean factual essay, ~57s.

The new windowed-decay rep penalty (with the new defaults — surgical,
penalty_repeat=1.05, etc.) doesn't break this loop. That's
independently expected: the loop is ~40 tokens long, ngram_max_size
is 4, so each individual 4-gram only sees ~1 logit of penalty per
re-emission — not enough to break a strong attractor that the model
has already locked into. **But the loop only happens on dirty
sessions.** A fresh-session run of the exact same prompt + seed +
sidecar produces a clean essay with no repetition. So the rep
penalty isn't the bug here; the Session state mismatch is what's
priming the model into a degenerate path.

### Symptom B — model gives up early after consecutive different prompts

1. Start blallama on A17B.
2. Send Apollo 11 essay request → 600 tokens, clean output, 262s.
3. Send "history of the internet" essay → 600 tokens, clean output, 275s.
4. Send "history of jazz" essay → **24 tokens**, content `"It appears
   there is a fundamental disconnect between the request to write an
   essay on the origins and cultural significance of jazz music."`,
   `stop_reason: "end_turn"`, 27s.
5. Killing blallama, restarting fresh, sending same jazz request →
   600 tokens, clean essay, 252s.

So the per-request flow itself is fine; what breaks is *consecutive
requests on the same Session*.

## Backend isolation: moeflux-only

Discovered 2026-04-26 evening: ran the 3-prompt test (Apollo →
internet → jazz) on a `Session<LlamaCppBackend>` configured against
`cogito-32b.gguf` and the **same** identical sidecar (English+Json+
Punctuation ignored, `surgical = true`, `penalty_repeat = 1.05`).
All three requests succeeded on the same Session: 600 tokens, full
factual essays, no loops, no empty bodies.

So the bug lives **below** drama_llama's `Session` / prefix-cache /
predictor layer. It manifests only when the backend is moeflux. Most
likely root is moeflux's C-side state not surviving consecutive
`mf_eval_prompt` calls with the same `seq_id` cleanly — either KV
cache positions, attention sinks, or the routed-experts cache
getting confused between calls. The drama_llama-side prefix-cache
math computes the same way for both backends, so it can't be the
prefix-length calculation alone.

This narrows the investigation: file as a moeflux upstream issue
once the Council work needs it, or work around it in
`MoefluxDecoder` by forcing a full prefix re-prefill (i.e. ignoring
prefix-cache hits) for the moeflux backend specifically until the
upstream is fixed.

## Plausible roots (ordered by likelihood)

1. **Prefix-cache prefix-length miscalculation.** All three essay
   prompts share the same chat-template prefix and diverge only at
   the user message. The prefix-cache feature in `Session` computes
   the longest common prefix between the new prompt's tokens and the
   stored previous prompt's tokens, clipped to a `cache_control`
   breakpoint (none here, so no clipping), and tells moeflux to
   resume from that position. If the math undercounts the divergence
   point — say it tells moeflux "resume from position N" when the
   tokens at positions <N actually differ from the cached KV — the
   model sees a corrupted prefix and behaves accordingly (early EOS,
   stall on the next forward pass).
2. **moeflux KV / position bookkeeping.** `mf_eval_prompt(_, tokens,
   start_pos, seq_id, _)` treats `start_pos` as the absolute KV cache
   position to begin filling from. If drama_llama passes a `start_pos`
   that is consistent with its own bookkeeping but inconsistent with
   what moeflux's `seq_id` slot actually contains (e.g. moeflux's
   internal cursor advanced past `start_pos` from a prior call),
   moeflux may silently write into stale rows.
3. **Cancelled-handler leaving moeflux mid-step.** For symptom A
   specifically: the killed curl returns from `axum::serve` early,
   but the spawned HTTP handler keeps the lock and continues calling
   `complete_response`. If `complete_response` is itself dropped (via
   tokio task cancellation chain), it could drop the predictor mid-
   token, which on the moeflux side might leave a Metal command
   buffer encoded but never committed/awaited. Subsequent calls then
   wait forever for a buffer that will never fire.

## Reproduction recipe

Symptom B is the cleanest:

```bash
./target/release/blallama "/Volumes/Temp Backup/models/blallama/" \
    --backend moeflux --port 11435 &

for prompt in apollo internet jazz; do
    curl -s -X POST http://127.0.0.1:11435/v1/messages \
        -H "Content-Type: application/json" \
        -d @/tmp/prompts/$prompt.json \
        -w "\n=== $prompt: HTTP %{http_code} %{time_total}s ===\n"
done
```

Expect Apollo and internet to succeed (~600 output tokens each), jazz
to fail (~20-30 output tokens, content saying the model can't comply).

## Next steps when picking this up

- Add a stderr trace inside `Session::prepare_call_cached` printing
  `(prev_token_count, new_token_count, longest_common_prefix_len,
  l_hit)` per call to confirm/refute the prefix-cache hypothesis.
- Compare `MoefluxDecoder::prefill` vs `prefill_resuming` (if it
  exists) — bug may be in the resuming path specifically.
- Try reproducing with `--no-prefix-cache` if such a flag exists, or
  by setting `Session::with_prefix_cache(false)` in `configure_session`
  in blallama. If pollution disappears with prefix cache off, the
  prefix-length math is the culprit. If pollution still happens, it's
  on moeflux's side.
- For symptom A specifically: try clean-killing curl (Ctrl-C in
  client) vs `pkill` and see if behavior differs. Tokio task
  cancellation may be involved.

## Relation to the v0.8.0 sidecar tuning work

This bug is **separate** from the rep-penalty growth fix and the
sidecar feature shipped in v0.8.0. The polluted-session symptom
appears in builds with `repetition: None` (per the auto-written
default sidecar) — the windowed-decay code path is not even running
when symptom B reproduces. Filing this so it isn't conflated with
the rep-penalty work or treated as a regression from it.
