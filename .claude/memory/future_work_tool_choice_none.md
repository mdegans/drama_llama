# Future work: `ToolChoice::None` has no local semantics

**Observed 2026-07-18** while building the `chat` interview example.
Filed as [issue #44](https://github.com/mdegans/drama_llama/issues/44).

The API contract for `tool_choice: none` is "the model must not use
any tool, even if tools are provided." The local session parses the
variant but does not enforce it: `src/session/mod.rs` (forced-grammar
selection, ~line 5023) lumps `ToolChoice::None` with `Auto` — no
forced grammar, and nothing suppresses free-form call emission, so the
model can still call tools.

## Why it matters

The interview flow (`chat --load`) wants a **prefix-preserving**
"don't call tools" mode: keep the tool defs rendered (schemas are
debug context, and stripping them shifts the prefix away from what the
agent saw in the original run) while forbidding new calls.
`--clear-tools` strips defs — prefix changes. `Some(ToolChoice::None)`
is the right shape (Mike suggested it 2026-07-18) but is a placebo
until the session honors it.

## Sketch

1. Library: on `ToolChoice::None`, mask the dialect's tool-call opener
   at emission — same machinery as `emit_ban_set` /
   `SampleOptions::banned_specials`, or a grammar that excludes the
   call production. Fits the #30 dialect arc.
2. Example: add a `--no-tool-use` flag to `chat` setting
   `tool_choice = Some(ToolChoice::None)` (Mike: a *separate* flag,
   not a change to `--clear-tools`).

Until then the `chat` example's catch-and-receipt path handles stray
calls gracefully, so nothing is broken — just not forbiddable.
