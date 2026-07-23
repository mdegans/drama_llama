# CI + self-hosted runner — state and next-session plan

**Written 2026-07-22 night, for a session that will likely run on the
remote box** where `~/.claude/` does not exist. Per-host personal memory
has bitten us before; this lives in the repo on purpose. Pairs with
[[test_topology]] and issue #70.

## Where CI stands (first run, `5218aea`)

The workflow (`.github/workflows/ci.yml`) went green on its first push,
**both OSes**, with one exception:

- `fmt` — green.
- `gate` (permutation matrix, `check --ci`) — **green on macOS and
  Linux.** First time this repo's moeflux / both-backends configurations
  have ever compiled on a machine that isn't Mike's.
- `test` (unignored tier) — green **except four tests that need a real
  model.** They are *metadata* tests, not generation. Each took ~11s to
  fail, which is the puzzle (see below).

### The four model-needing "unignored" tests — RESOLVED 2026-07-23, and the
### first reading of them was wrong

They are `llama_cpp::model::tests::{test_model, test_metadata,
test_model_desc, test_recommended_sampling}`, and they are **not**
mis-classified. All four go through `load_test_model_cpu`
(`src/llama_cpp/model.rs`), which loads with **`n_gpu_layers = 0` on
purpose** — the doc comment records the failure that motivated it, two of
three dying with "unable to allocate CUDA0 buffer" on a 24 GB card when
the parallel runner stacked them.

So the tier invariant is **"the unignored tier puts nothing on the GPU"**,
not "the unignored tier loads no model". Under that reading these four are
correct as written: metadata-only, VRAM-free, safe at full parallelism,
and cheap on RAM because llama.cpp mmaps and all four map the same file.
The earlier note here — `#[ignore]` them or give CI a fixture — would have
been a regression. Do not act on it; the fix was to give CI the weights.

**The 11s-to-fail is still unexplained** and is now moot for CI (with
`models/model.gguf` linked they pass). Curious, not load-bearing: three
took ~11.0s to fail on an *absent* file while the fourth took 0.12s.
Best guess is backend/device enumeration on first use, unverified.

## The runner is up (2026-07-23)

The box — **32 cores, 128 GiB RAM, a 3090** — is registered, CUDA toolkit
installed, rootless nvidia-docker set up (our jobs run *outside* a
container, so that part is for other repos). Its first run reproduced the
hosted result exactly: same four failures.

Weights live **outside the workspace**, at
`~/src/drama_llama/models` for the `ghrunner-drama-llama` account:

    Qwen3.6-35B-A3B-UD-IQ3_S.gguf          13G   (hardlinked to model.gguf)
    Qwen3.6-35B-A3B-UD-IQ3_S.mmproj.gguf  861M   (→ model.mmproj.gguf)

**Note the quant differs from the laptop's** — the box is **IQ3_S**, Mike's
Mac is **IQ4_XS**. Two consequences. Generation assertions that depend on
output quality may behave differently on the box than they do locally, so a
CI-only failure in a content assertion is a quant suspect before it is a
regression. And `test_recommended_sampling_across_models` names
`Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` literally, so on the box it skips *all
three* of its cases and passes vacuously.

No `.sampling.toml` sidecar is shipped: the run generates it from GGUF
metadata, which exercises that path for free.

### What the `model` job does, and what it deliberately skips

`ci.yml`'s `model` job is Linux-only: `-c llama-cpp` (CUDA on Linux)
`-t ignored`, serialized by the `full` nextest profile, behind a
**`concurrency` group not keyed on the ref** so a second PR queues instead
of landing on a busy card.

Three suites are excluded with `test.py -x` because their weights are not
on the box — Gemma 4 (17 GB) and gpt-oss (13 GB) — and, importantly, those
tests **`panic!` rather than skip** when the file is missing, so they
cannot be left to no-op: `session_gemma4`, `session_gptoss`, and
`media_e2e_gemma` (same Gemma weights, but inside the lib binary, hence a
test name rather than a binary name). Expected count: **99 of 119** ignored
tests. Delete an `-x` the day its weights land on the box.

`-x` is a filterset, not a runtime skip, on purpose — an absent-weights
test that reports itself green is exactly the silent coverage loss #68
exists to prevent.

### Self-hosted changes two Actions habits

- **`clean: false` on every checkout.** The default runs `git clean -ffdx`,
  which deletes `target/` — and a self-hosted workspace persists, so that
  default is the difference between reusing the llama.cpp build and paying
  20-40 min for it every run. Must be uniform across jobs: with one runner
  they share one workspace, so a single `clean: true` job wipes the others'
  build. Cost: a file deleted on a branch lingers, notably a removed
  `tests/*.rs` that cargo keeps compiling. `git clean -ffdx` on the box
  clears it.
- **No `Swatinem/rust-cache`.** Nothing to restore once the workspace
  persists, and a CUDA llama.cpp build does not fit in a 10 GB Actions
  cache.

Build parallelism is capped at 16 of 32 cores (`CARGO_BUILD_JOBS`, which
cargo forwards to build scripts as `NUM_JOBS`, which is what the `cmake`
crate reads for llama.cpp). **Never cap tests the same way**:
`NEXTEST_TEST_THREADS` overrides the `full` profile's `test-threads = 1`,
which is the only thing keeping two models off one card.

### macOS leg: still the uncertain one

A Parallels VM, mid-setup, Xcode and friends still to install. **Metal in
the VM is unverified** and deliberately out of scope: moeflux would want
the expert shards mounted in, tens of GB for a second copy of what the host
already has. So generation testing is **NVIDIA-only** and macOS stays a
compile-and-fast-tests gate; Metal generation is covered by the pre-commit
hook on the machine we develop on.

Open on macOS: the `test` job's four metadata tests need
`models/model.gguf` there too. The cheap answer is a **read-only Parallels
share of the host's models directory**, pointed at with
`DRAMA_LLAMA_MODELS` in the runner's environment (the `link-models`
composite action reads it) — no second 13 GB copy.

## VRAM: probably fine, watch the first run

Default context is **32k** (`cli::DEFAULT_N_CTX`). IQ3_S is ~13 GiB and a
32k KV cache for this architecture is order 3 GiB, so ~16 of the 3090's 24
GB — it should fit, mmproj included. If it does not, the standing decision
(Mike's, and unchanged) is **change the model, don't fight the VRAM**: pick
a smaller Qwen and re-baseline the affected expectations, noting in each
what model it assumes. A smaller model is also faster, and
grammar-constrained tool calling — the capability the interesting tests
exercise — should still work.

## Open issues

- **#70** — CI cost/runner. Also holds "drop the `push` trigger, keep
  `pull_request` + `workflow_dispatch`". **Deliberately not done yet**:
  `workflow_dispatch` does not appear until the workflow exists on the
  default branch, so with `push` gone and `v0.8.0` unmerged there would be
  no way to trigger a run at all. Do it when v0.8.0 merges to main. Note
  the motivation has *shifted* — self-hosted minutes are free, so the
  reason to drop `push` is now GPU contention, not credit.
- **GPU contention across repos** — the box serves several. An Actions
  `concurrency` group only orders jobs within this repo; nothing stops
  another repo's runner from taking the card at the same time. Mike has
  said he wants to elaborate on this off-tracker. A machine-level lock
  (flock around the model job) is the shape of the answer if one is needed.
- **#69** — moeflux prefetch telemetry; unrelated to CI, revisit on a17b
  perf work.
