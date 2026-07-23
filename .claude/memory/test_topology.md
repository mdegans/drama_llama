# Test topology: configurations × tiers (#68)

**Read before adding a test recipe, a feature, or a `#[cfg(feature =
"llama-cpp")]`.** Landed 2026-07-22, closing [#68].

`scripts/test.py` owns the topology. The justfile delegates to it and the
git hooks call the justfile, so the tests that gate a commit are the same
ones that gate a push. Python rather than bash because this repo builds on
Windows and because `argparse` gives it a real `--help`.

## The two axes that used to be one

    configuration   which features are on — i.e. which backend(s) exist
    tier            which tests run — unignored / ignored / all

Conflating them is what produced the original mess. `just test full` meant
`--run-ignored only` — *only* the model tests, not everything — so neither
`just test` nor `just test full` ever ran the whole suite and nothing said
so. `full` is now a **hard error** that names both successors rather than
an alias for either: silently doing the old thing would be wrong for anyone
who read it as "full", and silently doing the new thing would turn a
2-minute habit into an hour.

The nextest *profile* is still called `full` (`.config/nextest.toml`,
`test-threads = 1`). That name is fine — it means "the serialized profile"
and is not user-facing.

## Configurations

| name | backend | notes |
|---|---|---|
| `trait-layer` | none | no C dep; the canary for llama.cpp leaking into generic code |
| `llama-cpp` | llama.cpp | + `cuda` on Linux. The dev loop; what the hook runs |
| `llama-cpp-cpu` | llama.cpp | no CUDA. The CI-eligible llama.cpp build |
| `moeflux` | moeflux | macOS. Supported as of #68 — it did not build before |
| `both` | both | macOS. Where the documented `E0034` hazard lives |

`llama-cpp` is named **explicitly**, never inherited from `default`, and no
configuration is tested under the name "default" — Mike's call: `llama-cpp`
may not always be a default feature, so a matrix that says "default" is
asserting something that can quietly change meaning.

`trait-layer` deliberately has no `cli`/`axum`: a `--backend` selector over
an empty set of backends is a `compile_error!` by design (`src/cli.rs`).

Off Linux, `llama-cpp-cpu` resolves identically to `llama-cpp` (there is no
CUDA to tell them apart), so it shares the target dir instead of building
llama.cpp a second time, and `check` reports it as a duplicate rather than
counting it as a permutation it did not really test.

## What the moeflux-only fix bought

The library needed **one** `#[cfg(feature = "llama-cpp")]`, on the
`impl Session<LlamaCppBackend>` block in `src/session/mod.rs` — all three
lib errors were inside it. The test targets needed more (see the git log
for the sweep).

Two of #68's complaints then *dissolved* rather than needing a workaround:

- **`cfg(moeflux)` unit tests were unreachable from any recipe.** The old
  moeflux run scoped itself to `binary(~moeflux) + binary(~cross_backend)`
  and the lib unit-test binary is `drama_llama`, which matches neither.
  Widening the filterset would have dragged 119 ignored model tests in.
  With moeflux-only building, that configuration has no llama.cpp model
  tests to avoid, so it runs its **entire universe with no filterset** and
  the lib tests come along for free.
- **`cli`/`axum` implied `llama-cpp`**, so a moeflux-only *front-end* was
  unbuildable even though the library supported it. Both now imply nothing;
  `src/cli.rs` was already fully cfg-gated per backend and its doc comment
  had anticipated the change ("makes splitting `cli` into `clap`-only a
  one-line change rather than a hunt").

## Two mechanisms gate a whole test file; prefer `required-features`

The tree currently uses both, and the split is historical rather than
principled:

- **`required-features` in `Cargo.toml`** — cargo does not build the target
  at all. `tests/session.rs`, `session_gemma4`, `session_gptoss`,
  `output_config`, `blallama`.
- **`#![cfg(feature = "llama-cpp")]` at the top of the file** — the target
  still builds, as an empty binary. The older convention; ~10 files.

`required-features` is better on two counts. It skips the build instead of
compiling an empty binary in every configuration that lacks the feature,
which the permutation matrix now pays for 4-5 times over. And an empty test
binary is indistinguishable in `nextest list` from one whose tests were all
silently gated away — `required-features` makes the target vanish honestly,
which is the whole theme of #68.

**Follow-up, mechanical:** convert the `#![cfg(...)]` files. Not a
find-and-replace — the per-file feature set has to be established
empirically, because the doc comments lie about it (`session_gemma4` and
`session_gptoss` both say to run with `--features serde`; neither needs it,
and `inspect_prompt` documents `--features cli` while using nothing from
`cli`). One `cargo check` per file.

## Triaging the dead-code warnings the new configurations surface

Building configurations nobody built before means seeing warnings nobody
saw before. They are **not one thing**, and the difference decides the fix:

**A. Asymmetric pair — one backend uses each.** Real signal, worth a
`#[cfg_attr(not(feature = "..."), allow(dead_code))]`. `SnapshotStore` is
the type case: moeflux `get`s (its `Ctx` is a disjoint field) while
llama.cpp must `take` + re-insert around its `&mut self` FFI call. `get`
already had the attribute; `take`/`len` never got the mirror image,
because until #68 nothing compiled the build that would warn.

**B. Backend-free build, backend-dependent code.** Expected, and *not*
worth an attribute. In `trait-layer` there is no `Session` to construct,
so anything reachable only from a session — e.g. `seed_prose_block`'s
calls to `resolved_ignored_contains` / `seed_prompt_ngram` — is
unreachable by construction and every generic over `M: Model` goes
uninstantiated. Sprinkling `allow(dead_code)` to silence this would be
noise proportional to the crate, and would suppress the class-A warnings
that *do* mean something. The gate's job for `trait-layer` is that it
**compiles**; warnings there are a property of the configuration.

**The trap in both cases** is concluding "unused" from a search. Two of
these looked dead and were not — see the counting/pipe discipline above.
Check the call site before believing the warning, and check it with
complete output.

## `just test moeflux` spans two configurations

Not an accident, and don't "simplify" it back:

- `-c moeflux -t all` — the moeflux suites **and** every `cfg(moeflux)`
  unit test.
- `-c both -t all --filter cross_backend` — the cross-backend oracle diffs
  llama.cpp against moeflux, so it exists *only* in a build with both
  linked. It is not reachable from the moeflux-only configuration.

## Verifying changes to any of this

`--dry-run` prints the cargo invocations and runs nothing:

    just test moeflux "" --dry-run
    just permutations --dry-run

This is the only practical way to check the wiring — actually running a
mode costs tens of minutes. **Do not verify modes by running them in a
loop piped through `head`**: the SIGPIPE kills the loop, and before it does
it launches the real suite. (Done here, 2026-07-22. Same family as the
standing "don't pipe through `tail`" rule, one step worse — the pipe didn't
just hide the output, it started work nobody asked for.)

The other guard: when gating tests behind a feature, get a
`cargo nextest list` count for the llama.cpp configuration **before** the
change and diff it after. Gating a backend-agnostic test behind
`llama-cpp` is silent coverage loss, which is the exact failure #68 exists
to prevent, and the count catches it where review does not.

Count it with a pattern that matches test names, not with "indented
lines" — cargo's own `    Finished \`test\` profile ...` is indented four
spaces and inflates the count by one:

    cargo nextest list ... --run-ignored all \
      | grep -cE '^    [a-zA-Z_][a-zA-Z0-9_:]*$'

At the time of writing that is **587** for the `llama-cpp` configuration
(468 unignored + 119 ignored), which a plain `nextest run` will confirm in
its "Starting N tests ... (M skipped)" line. Cross-check the two; an
off-by-one in the guard number sends the next person hunting a test that
was never there.

## `-x/--exclude`: for a machine with only *some* of the weights

Added 2026-07-23 for the self-hosted runner. `-x NAME` drops everything
whose test **or** binary name contains `NAME`, repeatable, composing with
`--filter`:

    scripts/test.py run -t ignored -x session_gptoss -x media_e2e_gemma

It exists because the CI box has `models/model.gguf` and its projector but
not the Gemma 4 or gpt-oss weights, and those suites **`panic!` rather than
skip** when the file is absent. A filterset rather than a runtime skip on
purpose: a test that reports itself green because its weights were missing
is the exact silent coverage loss this whole memo is about, whereas an
excluded one shows in nextest's "N tests run, M skipped" line and in the
printed command. Same reason `--filter` matches both name kinds — the
caller shouldn't have to know that `session_gemma4` is a binary while
`media_e2e_gemma` is a test inside the lib binary.

## CI (#51, #70) — landed, and now runs the model tier

The old blocker (a `[patch.crates-io]` pointing misanthropic at a sibling
working copy, which does not exist on a runner) is gone; the patch was
dropped once `misanthropic 1.0.0-alpha.12` shipped. CI landed in `5218aea`
and moved to a self-hosted box on 2026-07-23, which is what made the
weights-dependent tiers automatable at all. See
[[plan_ci_self_hosted_runner]] for the box, the weights layout, the
`clean: false` / no-rust-cache reasoning, and what the `model` job skips.

[#68]: https://github.com/mdegans/drama_llama/issues/68
[#51]: https://github.com/mdegans/drama_llama/issues/51
