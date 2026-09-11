---
name: stray-think-closer-ban
description: 2026-09-11 diagnosis of the Qwen 3.6 `</think>`-in-free-text wedge (blamed on the llama.cpp rebase, exonerated by A/B); the conditional closer ban that fixes it; what was ruled out so nobody re-bisects it
metadata:
  type: project
---

# The stray `</think>` wedge (2026-09-11) — read before blaming a llama.cpp update for a thinking regression

## Symptom

blallama, Qwen3.6-35B-A3B, seed-runner transcripts ending on a
`tool_result` with `tool_choice: auto`, `thinking` unset. #101's
containment rejected the turn — `found: ["</think>"]` — deterministically,
three identical attempts deep, so the calling agent retried the same
prompt forever and wedged. Balerion's write-up of the ratchet dynamic is
in its own memory (`project_blallama_think_guard_regression_2026_09_09`);
the repro pair (`fails.json` / `succeeds_with_get_feed.json`) lives on
balerion under `~/agents/agora/repro-think-guard-2026-09-09/` and replays
straight at `/v1/messages`.

## What it was NOT (measured, don't re-bisect)

- **Not the llama.cpp rebase.** A/B on the same machine, same request,
  same flags (`--n-ctx 262144 --cache-slots 8`): a blallama built from
  `dev` HEAD in a clean worktree against crates.io `llama-cpp-sys-3
  0.8.1` (llama.cpp b9754, the pre-vacation binary's version) fails
  *identically* — guard rejection at `max_tokens=4096`, reasoning-shaped
  prose before the cliff. Different sampled words (numerics drift across
  ~1000 upstream commits), same failure class.
- **Not template detection.** `dump_template` through the rebased
  sys crate returns the Qwen3.6 template byte-identical to
  `tests/fixtures/templates/qwen3.6-gguf.jinja`. The load-time warning
  "no baked replacement matches … analyzes to no dialect we own" is the
  best-effort tier Qwen3.6 has *always* been on: there is no baked Qwen
  template and never was (`templates/` has cogito, gemma4, gptoss,
  mistral4). Balerion's "first domino" was wrong on this point.
- **Not the render.** `inspect_prompt` on the failing request ends in
  `<|im_start|>assistant\n<think>\n\n</think>\n\n` — the closed
  thinking-off stub, exactly what `thinking: None` asks for
  (`enable_thinking => prompt.thinking.is_some()` has been the rule
  since 2026-04-25). Every history assistant turn also carries a
  `<think>…</think>` pair because the session sets `preserve_thinking`.
- **Not raw inference.** Upstream `llama-simple` at the submodule commit,
  Metal, is coherent at 14k tokens thinking-off and at short prompts
  both ways. `just test all` was green (144 ignored) *before* the fix.
- **Not the recurrent-rollback change** in upstream `seq_rm`: `n_rs_seq`
  defaults to 0, so the new branch is dead for us.

## What it was

The model wants to reason after the tool result (the visible text is
"Let me check… I don't have a get feed tool… let me check the governance
log"). Under the closed stub, #107's opener ban masks `<think>`, so it
reasons in the open — and then emits the closer it was trained to end
reasoning with. The standing emit ban exempts `</think>` unconditionally
(it is the phase-split trigger), so nothing stopped it before #101's
post-scan. Symmetric hole to #107, on the other tag.

## The fix

`Session::reasoning_closer_ban` (memoized beside the opener ban) is
unioned into `banned_specials` per call iff the render ends with a
*closed* reasoning stub (`PreparedCall::reasoning_closed_by_render`,
from `render_ends_with_closed_reasoning`). Never on a pre-opened render.
Replay of `fails.json` now completes: `stop_reason: tool_use`, 486
tokens, open-prose reasoning into a `get_governance_log` call; identical
on the warm cache.

## Still open / recommendations

- The runner sends `thinking: null`. Qwen 3.6 is a thinking-native
  model; with `thinking` enabled the same reasoning becomes a proper
  `Thought` block instead of visible content. That is the runner's
  decision (Agora), not the library's.
- Why it "worked before vacation" is unverified. The A/B shows the code
  path fails on both llama.cpp versions; the likeliest story is that the
  pre-vacation binary predates #101/#107 (Jul 29 / Aug 5), before which
  the stray closer was silently seated as text and only surfaced as odd
  transcripts. Nobody has checked that binary's commit.
- Byte-spelled closers remain possible and containment still catches
  them; that is the documented id-level limit of the whole ban family.
