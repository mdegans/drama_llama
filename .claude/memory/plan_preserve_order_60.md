# Plan: #60 — declaration-order tool arguments (`preserve_order`)

Status: **specced 2026-07-23, deferred to its own session** (Mike's
call at the pre-publish review). Read alongside
[issue #60](https://github.com/mdegans/drama_llama/issues/60) and
`byte_exact_round_trip_invariant.md`.

## The unblock that makes this newly feasible

The remembered blocker was minijinja: `serde_json/preserve_order`
alone is useless if the template engine re-sorts maps when it renders
`tool.parameters`. **Mike checked 2026-07-23: minijinja also exposes a
`preserve_order` feature** (`minijinja/preserve_order`), so the whole
pipeline — schemars derive → `serde_json::Map` → minijinja render →
GBNF emitter → parser → re-render — can run insertion-ordered end to
end. Verify both features' semantics against the pinned versions at
implementation time; don't trust this memo's recollection of a
version's flag.

## Why this must be one deliberate session, not a flag flip

1. **Byte-stability is keyed on sorted order today.**
   `src/dialect/emit.rs` (header comment) standardizes alphabetical
   key order; `render_reference` and the grammar emitter both assume
   it. Flipping to insertion order changes the *canonical bytes* of
   every rendered tool call — the prefix cache's round-trip invariant
   moves with it. Every dialect round-trip fixture re-baselines.
2. **The grammar emitter must emit fields in the same order the
   schema declares** — that is the *point* (reasoning-ish fields
   before answer-ish fields, so a small model conditions later args
   on earlier ones) — and the parser's healed/lenient paths must stop
   normalizing through `serde_json::Map` in a way that re-sorts
   (today `Map` = BTreeMap = implicit sort; with `preserve_order` it
   becomes IndexMap = insertion order, which is what we want but also
   what makes accidental insertion-order dependence a new bug class).
3. **The publish hazard inverts.** Today a *downstream* crate
   enabling `serde_json/preserve_order` silently breaks our sorted
   assumption via feature unification (found in the 0.8.0 pre-publish
   review). Once *we* enable it deliberately and stop assuming sorted
   order anywhere, that hazard is gone — the fix for #60 and the fix
   for the hazard are the same work. Until then the exposure is
   documented here and in the emit.rs header.
4. **schemars field order**: verify the derive actually emits
   `properties` in declaration order (it uses its own Map type;
   check the pinned schemars 1.x behavior and its `preserve_order`
   story). If it sorts, the whole chain is moot until that's solved.

## Sketch (for the implementing session)

- Enable `serde_json/preserve_order` + `minijinja/preserve_order` in
  Cargo.toml (non-optional — half-on is worse than either state).
- Sweep `src/dialect/emit.rs` + `parse.rs` + `grammar_compile.rs` for
  sorted-order assumptions (the emit.rs header names itself; grep for
  `BTreeMap`-shaped reasoning about JSON maps and "alphabetical").
- Re-baseline `tests/dialect_roundtrip.rs` and any fixture asserting
  rendered argument order; the round-trip property (parse → re-render
  byte-identical) is the gate, not any specific order.
- Model-backed validation: the #58 round-trip separator test, plus a
  small-model A/B (does ordering reasoning-first actually help — the
  motivating claim) is optional but would justify the change in the
  changelog.
- Changelog: breaking-adjacent behavior change (canonical bytes of
  rendered calls change → warm prefix caches from 0.8.0 won't match).
