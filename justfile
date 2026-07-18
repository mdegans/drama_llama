# drama_llama task runner — run `just` (or `just --list`) to see recipes.
#
# Feature selection is OS-aware: CUDA is enabled automatically on Linux; on
# macOS llama.cpp uses Metal with no feature flag, so `test` and `test cpu`
# build the same thing there. CUDA is deliberately kept OUT of the crate's
# *default* features and chosen here instead — that keeps a bare `cargo build`
# portable (macOS and CI don't drag in the nvcc/C build), while `just` still
# gives the batteries-included, GPU-accelerated dev loop on Linux.

# Shared library/test feature set. `test` and `test full` use the GPU set;
# `test cpu` is the same set minus `cuda`. Keeping the rest identical means the
# llama.cpp build is NOT evicted when switching between `test` and `test full`
# (the nvcc-rebuild trap — one feature set per GPU session).
base_features := "cli,toml,axum,serde,stats,json-schema,mtmd"
gpu_features  := base_features + if os() == "linux" { ",cuda" } else { "" }

# moeflux is macOS-only (Metal kernels) and selects its model at COMPILE time —
# each model is its own feature, exactly one enabled at a time. Hence a variable
# rather than a constant: override with
#   just --set moeflux_model cogito-v2-671b test moeflux
# (Phase 7 replaces the compile-time selection with a runtime variant config;
# don't lean further into it in the meantime.) Enabling moeflux rebuilds
# drama_llama but NOT llama-cpp-sys — that crate's own features are untouched, so
# the llama.cpp C build survives the switch.
moeflux_model := "qwen3-6-35b-a3b"

# Separate target dirs for the GPU (CUDA/Metal) and CPU builds, so alternating
# `just test` and `just test cpu` does not evict each other's llama.cpp build
# (a CUDA rebuild is ~20-40 min). `test` and `test full` share the GPU dir.
gpu_target := justfile_directory() / "target"
cpu_target := justfile_directory() / "target" / "cpu"

# Every run is tee'd here, so a failure can be read back after the fact instead
# of re-run to be seen. Under `target/`, hence already gitignored.
log_dir := justfile_directory() / "target" / "test-logs"

# Default recipe: the fast GPU test loop.
default: test

# Run the test suite via cargo-nextest (`just setup` installs it). Modes:
#   just test          unignored tests, GPU-accelerated (CUDA on Linux / Metal on macOS)
#   just test full     also the long-running #[ignore]'d GPU/model tests, serialized
#   just test cpu      unignored tests, CPU-only (no CUDA; separate target dir)
#   just test moeflux  the moeflux + cross-backend suites (macOS; needs the weights
#                      mounted). Takes an optional filter — these are 12 minutes
#                      of tests and re-running one shouldn't cost all of them:
#                        just test moeflux cross_backend
#                      The model variant is a compile-time feature, so it's a
#                      justfile variable rather than an argument:
#                        just --set moeflux_model cogito-v2-671b test moeflux
#   just test NAME     just the tests (or suites) matching NAME — see below
# Anything that isn't a known mode is a substring filter over test *and* binary
# names, run with the ignored tests included, serialized, and uncaptured (so the
# suites' block/emission dumps are visible on a pass, not only on a failure):
#   just test forced_call_parses_to_tool_use   one test
#   just test session_gptoss                   one suite
# Output of every mode lands in target/test-logs/<mode>.log.
# Modes: (none)=unignored+GPU · full=+ignored · cpu=no CUDA · moeflux=moeflux suites · NAME=filter
test mode="" filter="":
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p "{{log_dir}}"
    case "{{mode}}" in
      "")
        export CARGO_TARGET_DIR="{{gpu_target}}"
        name=unignored
        cmd=(cargo nextest run --features "{{gpu_features}}" --no-fail-fast)
        ;;
      full)
        # Only the #[ignore]'d GPU/model tests, one at a time (profile 'full'
        # caps test-threads=1 — a 30B model barely fits once on the card).
        export CARGO_TARGET_DIR="{{gpu_target}}"
        name=full
        cmd=(cargo nextest run --features "{{gpu_features}}" --profile full --run-ignored only --no-fail-fast)
        ;;
      cpu)
        export CARGO_TARGET_DIR="{{cpu_target}}"
        name=cpu
        cmd=(cargo nextest run --features "{{base_features}}" --no-fail-fast)
        ;;
      moeflux)
        # The moeflux + cross-backend suites are cfg-gated on the moeflux feature
        # AND macOS, so a bare `just test` can't reach them — they need their own
        # feature set. Ignored by default (they want ~40 GB of expert shards on a
        # mounted volume), hence --run-ignored all.
        if [ "$(uname -s)" != "Darwin" ]; then
          echo "just test moeflux: macOS only (moeflux is Metal-backed)" >&2
          exit 2
        fi
        export CARGO_TARGET_DIR="{{gpu_target}}"
        # Universe is the moeflux + cross-backend binaries; an optional filter
        # narrows to one test within them.
        set="binary(~moeflux) + binary(~cross_backend)"
        if [ -n "{{filter}}" ]; then
          set="($set) & (test(~{{filter}}) + binary(~{{filter}}))"
          name="moeflux-$(printf '%s' '{{filter}}' | tr -cs '[:alnum:]_.-' '_')"
        else
          name="moeflux"
        fi
        cmd=(cargo nextest run \
             --features "{{gpu_features}},moeflux-model-{{moeflux_model}}" \
             --profile full --run-ignored all --no-capture --no-fail-fast \
             -E "$set")
        ;;
      *)
        # Filter mode. `--run-ignored all` so a named long-running test runs
        # without also having to remember which of the two lists it's on.
        export CARGO_TARGET_DIR="{{gpu_target}}"
        name="$(printf '%s' '{{mode}}' | tr -cs '[:alnum:]_.-' '_')"
        cmd=(cargo nextest run --features "{{gpu_features}}" --profile full \
             --run-ignored all --no-capture --no-fail-fast \
             -E 'test(~{{mode}}) + binary(~{{mode}})')
        ;;
    esac
    log="{{log_dir}}/${name}.log"
    echo "+ ${cmd[*]}"
    echo "+ log: ${log}"
    "${cmd[@]}" 2>&1 | tee "${log}"

# Run an example against the same feature set and target dir as `just test`, so
# the two share one llama.cpp build instead of evicting each other's.
# `tokio`/`repl` are appended: the chat-loop examples (chat, council,
# swarm) require them, and the extra deps don't touch the llama.cpp build.
#   just example whodunit
#   just example whodunit models/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf
# Run an example with the shared feature set / target dir.
example name *args:
    CARGO_TARGET_DIR="{{gpu_target}}" \
      cargo run --release --features "{{gpu_features}},tokio,repl" --example {{name}} -- {{args}}

# Format the tree. The pre-commit hook checks this, so the tree must be a fixed
# point of rustfmt: one that isn't makes every later diff carry someone else's
# reformatting. Note rustfmt's output varies across toolchain versions — if two
# machines fight over the same lines, that's why, and the fix is to agree on a
# toolchain, not to keep re-running this.
# Format the tree with rustfmt (the pre-commit hook enforces this).
fmt:
    cargo fmt

# Point git at the versioned hooks in .githooks/ — one config line, no copying,
# and the hook stays under review like any other file. See .githooks/pre-commit
# for what it gates (rustfmt + the unignored tests).
# Point git at the versioned pre-commit hook in .githooks/.
install-hooks:
    git config core.hooksPath .githooks
    @echo "hooks installed: core.hooksPath -> .githooks"

# Install the dev tools these recipes need (cargo-nextest).
setup:
    cargo install cargo-nextest --locked
