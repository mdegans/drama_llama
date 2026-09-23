---
name: qwen38-reingest-probe
description: "#112 root cause and fix (2026-09-23): Qwen3.8's stock template reads reasoning only from reasoning_content; the analyzer now MEASURES InlineThink vs Field. Also the open structured-output separator gap."
metadata:
  type: project
---

# Qwen3.8 tip never anchored (#112) — the reingest probe

## Root cause

Qwen3.8 has **no baked template** (rung 3, stock). Its template differs
from Qwen3.6's in one load-bearing way: 3.6 reconstructs reasoning from
`content` (`content.split('</think>')`) *or* `reasoning_content`; 3.8
dropped the split and reads `reasoning_content` only. The analyzer never
measured the re-ingest convention — `ReasoningReingest` defaulted to
`InlineThink` for every `<think>` dialect (plus a `[THINK]` source-sniff
patch for Mistral). So on 3.8 every thinking turn re-rendered as
`<think>\n\n</think>\n\n<think>THOUGHT</think>…` — the thought as
*content* behind an empty stub — the canonicalization gate failed, and
the tip was skipped on every turn.

Replay of the 41 logged 2026-09-22 emissions (`~/blallama-debug.log`,
`raw_text_debug`) through parse → render, model-free: **39/41 unstable,
exactly the 39 logged misses**. With `Field`: 4/41 (see below).

Hypotheses in the issue that were **wrong**: prior-turn thought
stripping (3.8 defaults `preserve_thinking` on) and the literal `null`
parameter (#115 — parses as a string, re-renders unchanged; the 3.8
template's `string if string else tojson` is byte-stable for it).

## Fix

`analyzer.rs::compare_reasoning_reingest`: render `[user, assistant]`
with the thought inlined (`<think>T</think>C`, the spelling
`chat_template::append_block_text` uses) vs in `reasoning_content`.
Byte-equal → template reconstructs inline → keep `InlineThink` (3.5/3.6).
Different and the field render carries the thought → `Field` (3.8,
Mistral4). Neither → default (old Qwen3 chat ignores the field). The
Mistral `[THINK]` PATCHES entry was deleted — the probe subsumes it.

Pins: `dialect_analyzer.rs::{qwen36_gguf_xml, qwen38_gguf_field_reasoning}`,
`dialect_roundtrip.rs::qwen38_thinking_turn_round_trips` (with an
`InlineThink` control), `session_qwen38.rs` on device.

**Lesson for the per-model suites:** the shared `common::tip` scenarios
sent `thinking: None` → `enable_thinking = false`, so no suite ever
re-ingested a reasoning turn; they were green on 3.8 throughout.
`assert_tip_anchors_across_thinking_tool_rounds` exists now — new
thinking models should run it.

## Second cause: the grammar could not spell the separator

The replay with `Field` still left 4/41 unstable — all JSON
(output_config) turns: `</think>{`, `</think> {`, `</think>\n{`. On
device, *forced tool calls with thinking* failed the same way
(`</think>\n<tool_call>`); the Agora run used `auto`, where the model
writes the gap unconstrained and its habit is `\n\n`. Root: the gap
after a thought was `fws` / `ws` = `[ \t\n\r]?` — **at most one
byte** — so `\n\n` was unreachable under any grammar.

Fix: `ReasoningSyntax::separator: Option<String>`, measured by
`measure_reasoning_separator` (bytes between `reasoning.end` and
content in a rendered turn, whitespace only, else `None`). The tool
emitter spells it after `thought_close`; `OutputConfigOptions` gained
`thought_separator`, which `resolve_grammar` fills from the dialect.
`None` reproduces the old grammar exactly. Measured values: Qwen3.5 /
3.6 / 3.8 / Qwen3 chat `"\n\n"`, Mistral 4 owned `""`, everything else
`None` (Gemma is `ToolsOnly`, gpt-oss hand-built).

Pins: `output_config.rs::thought_separator_is_spelled_after_every_thought`,
`dialect_roundtrip.rs::qwen38_forced_call_grammar_spells_the_thought_separator`,
analyzer separator asserts, and `session_qwen38.rs`
(`thinking_tool_turn_…`, `thinking_structured_output_…`,
`tip_anchors_across_thinking_tool_rounds_…`) on device — all green
2026-09-23.

## Side findings

- `llama-cpp-sys-3` 0.8.3 shipped without `tools/tuning` (added only
  under `GGML_METAL`, so Linux CI never configures it): every macOS
  `mtmd` build failed. Fixed in 0.8.4 (mdegans/llama-cpp-sys#9).
  Worth a macOS CI job that builds the *packaged* crate with `mtmd`.
- A seeded 3.8 structured-output run's thought read "meets onthe
  26th" → "1st": looks like the repetition penalty on spaces/digits —
  the #113 disease. Not investigated.
