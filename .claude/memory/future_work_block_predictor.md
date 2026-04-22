---
name: Future work — BlockPredictor
description: Natural evolution of DeferredGrammar into a typed-Block-aware Predictor layer; replaces ad-hoc phase-split special cases with one uniform hook mechanism.
type: project
---

Status: unscheduled, captured for long-term. Deferred past v0.7 tag.

## Idea

Compose a `BlockPredictor` on top of `PiecePredictor` that emits
typed `Block` values (already defined in the crate as
`Block::Text`, `Block::Thought`, `Block::ToolUse`, etc.) as they
close, instead of raw pieces. The hooks on block entry/exit
replace the ad-hoc `DeferredGrammar` mechanism shipped in commit
`99456b5`.

```rust
struct BlockPredictor<'a> {
    inner: PiecePredictor<'a>,
    parser: StreamingBlockParser,
    hooks: BlockHooks,
}
// emits: Option<Block> per closed block, or a streaming shape that
// yields partial Block updates.

struct BlockHooks {
    on_enter: Box<dyn FnMut(&Block, &mut SampleOptions)>,
    on_exit:  Box<dyn FnMut(&Block, &mut SampleOptions)>,
}
```

Use cases that fall out cleanly:

- `on_enter(Block::Text)` → push JSON grammar into modes;
  `on_exit` → pop.
- `on_enter(Block::ToolUse)` → push tool-call grammar;
  `on_exit` → pop. Multi-turn tool use in one completion.
- Repetition penalty scope per-block (e.g. apply only within
  `Block::Text` not during `Block::Thought`).
- Per-block stop conditions (stop on `</think>` exit only, not
  mid-text).

## Relationship to shipped DeferredGrammar

`DeferredGrammar` is a special-case of this: one-shot
`on_enter(Block::Text)` that fires on `</think>` trigger bytes. A
proper BlockPredictor would subsume it, plus support tool-call
phase-split, per-block repetition, and so on. Ship-soon path:
BlockPredictor could live alongside DeferredGrammar, with the
latter implemented in terms of the former once both exist.

## What's missing

The current `BlockParser` parses a completed string into blocks.
For `BlockPredictor` we need a **streaming** variant:

- `feed(piece: &str) -> Vec<BlockEvent>` where `BlockEvent` is
  `BlockStart(BlockKind)` / `BlockEnd(Block)` / `TextAppend(&str)`.
- Internal state machine that mirrors the existing
  `BlockParser` semantics but can incrementally commit pieces.
- Careful UTF-8 boundary handling — pieces may split codepoints.

Non-trivial but localized: all the streaming logic lives in
`src/block.rs` (or wherever `BlockParser` is defined).

## Why defer

- The DeferredGrammar shipped in v0.7 solves the immediate
  reactor perf problem; 17.6 tok/s is past the 15 tok/s target.
- Streaming BlockParser is real surgery — parser state machine
  + tests for boundary cases.
- No downstream (Weave, Agora) is asking for multi-block
  phase-split yet.

Revisit when: a consumer wants per-tool-call grammar activation,
multi-block repetition penalty scoping, or streaming Block
emission from the core iterator API.
