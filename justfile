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
# (the nvcc-rebuild trap — one feature set per GPU session; see
# .claude/memory/gpu-session-ops.md).
base_features := "cli,toml,axum,serde,stats,json-schema,mtmd"
gpu_features  := base_features + if os() == "linux" { ",cuda" } else { "" }

# Separate target dirs for the GPU (CUDA/Metal) and CPU builds, so alternating
# `just test` and `just test cpu` does not evict each other's llama.cpp build
# (a CUDA rebuild is ~20-40 min). `test` and `test full` share the GPU dir.
gpu_target := justfile_directory() / "target"
cpu_target := justfile_directory() / "target" / "cpu"

# Default recipe: the fast GPU test loop.
default: test

# Run the test suite via cargo-nextest (`just setup` installs it). Modes:
#   just test        unignored tests, GPU-accelerated (CUDA on Linux / Metal on macOS)
#   just test full   also the long-running #[ignore]'d GPU/model tests, serialized
#   just test cpu    unignored tests, CPU-only (no CUDA; separate target dir)
# Modes: (none)=unignored+GPU · full=+ignored GPU tests (serialized) · cpu=unignored, no CUDA
test mode="":
    #!/usr/bin/env bash
    set -euo pipefail
    case "{{mode}}" in
      "")
        CARGO_TARGET_DIR="{{gpu_target}}" \
          cargo nextest run --features "{{gpu_features}}"
        ;;
      full)
        # Only the #[ignore]'d GPU/model tests, one at a time (profile 'full'
        # caps test-threads=1 — a 30B model barely fits once on the card).
        CARGO_TARGET_DIR="{{gpu_target}}" \
          cargo nextest run --features "{{gpu_features}}" --profile full --run-ignored only
        ;;
      cpu)
        CARGO_TARGET_DIR="{{cpu_target}}" \
          cargo nextest run --features "{{base_features}}"
        ;;
      *)
        echo "just test: unknown mode '{{mode}}' — use (nothing) | full | cpu" >&2
        exit 2
        ;;
    esac

# Install the dev tools these recipes need (cargo-nextest).
setup:
    cargo install cargo-nextest --locked
