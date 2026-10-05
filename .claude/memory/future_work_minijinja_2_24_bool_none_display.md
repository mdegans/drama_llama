# `minijinja` 2.24.0 renders bare `{{ bool }}` / `{{ none }}` as `True`/`False`/`None`, not `true`/`false`/`none` — 3 unignored tests red on a fresh resolve

**Found 2026-09-25, incidentally, while running `just test`/`just check`
for the `/v1/models` catalog-capabilities work — unrelated to that
task, not fixed here.** Reproduces on `dev` at `381d476` in a fresh
worktree. **Correction (main thread, same day):** `Cargo.lock` is
*gitignored* (`.gitignore:15`), not committed. The main checkout's
local lock still pins 2.19.0 and is green; any *fresh* resolve (a new
worktree, CI, every crates.io consumer of drama_llama) gets 2.24.0
and is red. So CI and downstream users are exposed, and the local dev
box hides it:

```
just test
...
FAIL chat_template::tests::test_thinking_disabled_renders_off
FAIL tests/dialect_roundtrip.rs::reconstruct_gemma4_thought_and_values
FAIL tests/template_rendering.rs::enable_thinking_derives_from_prompt_thinking
```

## Root cause, confirmed by reading both versions' source

`Cargo.toml` pins `minijinja = "2.5"` (loose), so a fresh resolve
picks `2.24.0`. Diffed against `2.19.0` (also cached
locally) in `~/.cargo/registry/src/.../minijinja-{2.19.0,2.24.0}/src/value/mod.rs`:

- **2.19.0**, `impl fmt::Display for Value`: `ValueRepr::Bool(val) =>
  val.fmt(f)` — Rust's native `bool::fmt`, i.e. lowercase `true`/`false`.
  `None` goes through a different arm entirely (not the capitalized one).
- **2.24.0**: `ValueRepr::Bool(val) => f.write_str(if val { "True" }
  else { "False" })`, and `ValueRepr::None => f.write_str("None")` —
  Python/real-Jinja2-style capitalization, in **two** separate `Display`
  impls in that file (both changed).

So somewhere between 2.19.0 and 2.24.0 upstream deliberately made bare
`{{ value }}` interpolation of a bool/none match actual Jinja2/Python
semantics instead of Rust's. Semver-legal (both 2.x), but a real output
break for anything that interpolates a raw bool/None and expects the
old lowercase spelling — which several owned templates and round-trip
tests do, directly or via a rendered tool-call argument.

## Why this matters more than 3 failing tests

`reconstruct_gemma4_thought_and_values` shows the sharp edge: a tool
call argument re-render produced `flag:False`/`maybe:None` where the
canonical (model-emitted) form was `flag:false`/`maybe:none` —
**invalid JSON**. If any *served* path renders tool-call arguments
through raw Jinja interpolation (rather than through `json_canon`/
`json_dumps`, which serialize properly and are presumably unaffected —
not verified either way this session), this is a live correctness bug
for boolean/null tool arguments, not just a test-fixture mismatch.
Worth checking which path Gemma 4 (and any other dialect whose template
interpolates raw values) actually uses before assuming it's cosmetic.

## Not investigated this session (out of scope for the task in progress)

- Whether this is scoped to `{{ bare_value }}` interpolation only, or
  also affects `|string`, string concatenation (`~`), or `tojson` (the
  JSON-safe path most tool-call serialization should already prefer).
- Whether `minijinja-contrib`'s `pycompat` feature (already enabled,
  `unknown_method_callback` only) has an *opt-in* knob for the *old*
  lowercase behavior, or whether the fix has to be at every call site
  that interpolates a raw bool/None (e.g. an explicit `|lower` filter,
  or routing through `tojson`/`json_canon` instead of `{{ }}`).
- Whether pinning `minijinja = "=2.19"` (or similar) is an acceptable
  stopgap vs. fixing the interpolation sites — a version pin is a one-
  line Cargo.toml change but freezes out real upstream fixes/features.
- Whether any *other* owned template (`templates/*.jinja`) has the same
  raw-interpolation shape and is silently emitting the new casing today
  (only the 3 tests above happened to pin the old behavior explicitly).

## Suggested first step for whoever picks this up

`grep -rn '{{ *[a-zA-Z_.]* *}}'` across `templates/*.jinja` plus the
three failing tests' exact assertions is probably enough to scope
whether this is "3 test fixtures need updating to match intentional new
behavior" or "3 tests correctly caught a live JSON-correctness
regression." The gemma4 dialect round-trip test strongly suggests the
latter for at least tool-call arguments.
