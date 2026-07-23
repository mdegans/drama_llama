# drama_llama task runner — run `just` (or `just --list`) to see recipes.
#
# The test topology lives in `scripts/test.py`, not here. CI calls that script
# and the git hooks call these recipes, so the tests that gate a commit are
# byte-for-byte the ones that gate a push, and neither can drift from the other
# by editing only one of them (#68). The recipes below are thin wrappers that
# exist for muscle memory and for the hooks; `python3 scripts/test.py --help`
# is the real interface, and it is what to call on Windows (these recipe bodies
# are bash).

# Feature set for the rustdoc gate (`just doc` / `just check`) and for
# `just example`. Matches the `llama-cpp` configuration in scripts/test.py, so
# the doc build REUSES the `just test` compilation on macOS — a rustdoc pass,
# not a recompile. A pre-commit doc gate is only worth having if it's ~free.
# That set is the full doc-visible surface, so this is also the release sweep;
# there is no longer a wider set to run by hand.
#
# `llama-cpp` is named explicitly rather than inherited from `default`: it may
# not always be a default feature, and the permutation gate deliberately does
# not test any configuration by the name "default".
base_features := "llama-cpp,cli,toml,axum,serde,stats,json-schema,mtmd,webchat,egui"
gpu_features  := base_features + if os() == "linux" { ",cuda" } else { "" }
doc_features  := base_features

# moeflux is macOS-only (Metal kernels) and selects its model at COMPILE time —
# each model is its own feature and exactly one must be enabled, which is what
# implies `moeflux` itself. Hence a variable rather than a constant:
#   just --set moeflux_model cogito-v2-671b test moeflux
# a3b is the fast one; a17b runs at ~2 tok/s and exists for backup Agora
# council work and tests. (Phase 7 replaces the compile-time selection with a
# runtime variant config; don't lean further into it in the meantime.)
moeflux_model := "qwen3-6-35b-a3b"

# Target dir for the recipes that still drive cargo directly. Kept separate
# from the CPU build's dir so alternating the two does not evict each other's
# llama.cpp build (a CUDA rebuild is ~20-40 min); scripts/test.py knows the
# same rule.
gpu_target := justfile_directory() / "target"

# Default recipe: the fast GPU test loop.
default: test

# Run tests (via scripts/test.py; `just setup` installs cargo-nextest). Modes:
#   just test            unignored tests, llama.cpp, GPU-accelerated
#   just test ignored    ONLY the #[ignore]'d model tests, serialized
#   just test all        genuinely everything — unignored AND ignored
#   just test cpu        unignored, no CUDA (separate target dir)
#   just test moeflux    the moeflux configurations, INCLUDING model tests
#   just test moeflux-quick   moeflux unit tests only — no weights, seconds
#   just test both       unignored tests with BOTH backends linked
#   just test NAME       tests/suites matching NAME, any tier, uncaptured
#
# `just test moeflux` loads real weights (`--run-ignored all`) and takes
# ~12 minutes on the default a3b variant. On `qwen3-5-a17b` (~2 tok/s) it
# takes hours; the script warns before starting. `just permutations` runs
# NO tests at all — it is `cargo check --all-targets`, a compile gate.
#
# `moeflux` and `NAME` take an optional filter — the moeflux suites are ~12
# minutes and re-running one shouldn't cost all of them:
#   just test moeflux cross_backend
#
# NOTE: `full` used to mean `--run-ignored only`, i.e. ONLY the model tests and
# not everything. That ambiguity is the reason it is now an error rather than
# an alias for either — pick `ignored` (the old behaviour) or `all`.
#
# Output of every mode lands in target/test-logs/.
test mode="" filter="" *args:
    #!/usr/bin/env bash
    set -euo pipefail
    # Trailing args go straight to the script — mainly `--dry-run`, which
    # prints the cargo invocations without running them and is the only way to
    # check this wiring without paying tens of minutes per mode:
    #   just test moeflux "" --dry-run
    run() { python3 scripts/test.py run --moeflux-model "{{moeflux_model}}" "$@" {{args}}; }
    # `just test NAME` consumes the mode as the filter; every other mode names
    # a configuration and takes the filter as its second argument.
    filter="{{filter}}"
    case "{{mode}}" in
      "")        run -c llama-cpp     -t unignored ;;
      ignored)   run -c llama-cpp     -t ignored   ;;
      all)       run -c llama-cpp     -t all       ;;
      cpu)       run -c llama-cpp-cpu -t unignored ;;
      both)      run -c both          -t unignored ${filter:+--filter "$filter"} ;;
      moeflux-quick)
        # The moeflux-only configuration's fast tier: every `cfg(moeflux)`
        # unit test, no weights, seconds not hours. This is the "did I break
        # the moeflux build" check; `just test moeflux` is the real thing.
        run -c moeflux -t unignored ${filter:+--filter "$filter"}
        ;;
      moeflux)
        # Two configurations, because "the moeflux work" spans two. The
        # moeflux-only universe holds the moeflux suites AND every
        # `cfg(moeflux)` unit test under src/ — those were unreachable from
        # any recipe before #68, because the old filterset named only the
        # moeflux/cross_backend *binaries* and the lib binary is neither.
        run -c moeflux -t all ${filter:+--filter "$filter"}
        # The cross-backend oracle diffs llama.cpp against moeflux, so it
        # only exists in a build with both linked — it is NOT reachable from
        # the moeflux-only configuration above.
        run -c both -t all --filter "${filter:-cross_backend}"
        ;;
      full)
        echo "just test full: removed — it meant ONLY the ignored tests," >&2
        echo "  which is now 'just test ignored'. For everything (unignored" >&2
        echo "  AND ignored) use 'just test all'." >&2
        exit 2
        ;;
      *)         run -c llama-cpp --filter "{{mode}}" ;;
    esac

# Coverage, via cargo-llvm-cov (`just setup` installs it).
#   just coverage              everything — unignored AND model tests
#   just coverage unignored    the fast tier only, no weights
#   just coverage "" --html --open      browsable per-file report
#   just coverage "" --doctests         + doctest coverage (needs nightly)
#
# `--doctests` is what CI runs. It switches the whole run to `cargo +nightly`,
# because llvm-cov's doctest support is unstable and profraw from two
# toolchains will not merge. Leave it off locally unless you're reproducing
# CI's number — it forces a separate nightly build of everything.
#
# Defaults to the `all` tier because the fast tier alone reports every
# generation path as dead code — most of this crate is only reachable with a
# model loaded, so `unignored` coverage measures the wrong thing. Expect the
# same ~12 minutes `just test all` costs, plus a full instrumented rebuild
# into target/llvm-cov-target (llama.cpp included, the first time).
#
# `tests/` and `examples/` are excluded from the report; `#[cfg(test)] mod
# tests` inside src/ cannot be, so the headline percentage is an upper bound.
# The per-file table is the part to actually read.
coverage tier="all" *args:
    python3 scripts/test.py coverage --moeflux-model "{{moeflux_model}}" \
      -c llama-cpp -t "{{tier}}" {{args}}

# The permutation gate: every feature configuration compiles, test targets
# included. Slower than `just check` (it builds moeflux and the no-backend
# trait layer), so it is NOT in the pre-commit hook — run it before a release
# or after touching cfgs, features, or anything in src/llama_cpp/ or
# src/moeflux/. `--ci` narrows to the configurations a runner can build.
permutations *args:
    python3 scripts/test.py check --moeflux-model "{{moeflux_model}}" {{args}}

# Run an example against the same feature set and target dir as `just test`, so
# the two share one llama.cpp build instead of evicting each other's.
# `tokio`/`repl` are appended: the chat-loop examples (chat, council, swarm)
# require them, and the extra deps don't touch the llama.cpp build.
#   just example whodunit
#   just example whodunit models/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf
example name *args:
    CARGO_TARGET_DIR="{{gpu_target}}" \
      cargo run --release --features "{{gpu_features}},tokio,repl" --example {{name}} -- {{args}}

# Format the tree. The pre-commit hook checks this, so the tree must be a fixed
# point of rustfmt: one that isn't makes every later diff carry someone else's
# reformatting. Note rustfmt's output varies across toolchain versions — if two
# machines fight over the same lines, that's why, and the fix is to agree on a
# toolchain, not to keep re-running this.
fmt:
    cargo fmt

# Build the docs with broken/private intra-doc links promoted to hard errors —
# the rustdoc gate the pre-commit hook enforces (issue #47). Shares `just test`'s
# target dir + feature set, so on macOS it reuses that build and only runs the
# rustdoc pass. A regression fails the commit; fix the link, point it at the
# public item, or drop it to a plain `code span`.
doc:
    #!/usr/bin/env bash
    set -euo pipefail
    export CARGO_TARGET_DIR="{{gpu_target}}"
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --features "{{doc_features}}"

# Run the doctests. Separate from `just test` because nextest has no doctest
# support at all, so none of the test recipes run a single one — and the
# top-level module doc is `include_str!("../README.md")`, so the README's
# examples are doctests. Shares `just doc`'s build.
#
#   just doctest              the llama-cpp configuration (fast; in `just check`)
#   just doctest all          every configuration — what CI's `gate` job runs
#
# Use `all` before a release or after touching a doc example that names a
# feature-gated type. A doctest is a per-configuration claim and `just
# permutations` cannot see it: `cargo check --all-targets` does not compile
# doctests. Three examples were broken in the moeflux-only and trait-layer
# builds for want of this.
doctest config="llama-cpp":
    python3 scripts/test.py doctest -c "{{config}}"

# Fast static gate: rustfmt-clean + rustdoc-clean + doctests, no model tests
# (those are `just test`, which the pre-commit hook runs separately). Run by
# hand for a quick "is the tree lint-clean" without paying the test-suite cost.
#
# The doctests are in here rather than alongside `just test` because nextest
# cannot run them, so `just test` runs zero — and because they cost ~2s against
# a warm build, which is well inside this recipe's budget. They are load-bearing
# now that the crate root is `#![doc = include_str!("../README.md")]`: the
# README's examples are the only thing keeping the front page honest.
check:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "+ cargo fmt --check"
    cargo fmt --check
    echo "+ just doc (rustdoc -D warnings)"
    just doc
    echo "+ just doctest"
    just doctest
    echo "check: ok"

# Point git at the versioned hooks in .githooks/ — one config line, no copying,
# and the hook stays under review like any other file. See .githooks/pre-commit
# for what it gates (rustfmt + the unignored tests).
install-hooks:
    git config core.hooksPath .githooks
    @echo "hooks installed: core.hooksPath -> .githooks"

# Install the dev tools these recipes need.
setup:
    cargo install cargo-nextest --locked
    cargo install cargo-llvm-cov --locked
    rustup component add llvm-tools-preview
