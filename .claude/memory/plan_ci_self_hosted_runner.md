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

Weights live **outside the workspace**, in a **root-owned, immutable
`/models`** — shared by every account on the box rather than copied per
runner account. As of 2026-07-23 it holds the full set the suite wants:
`model.gguf`/`model.mmproj.gguf` (Qwen3.6-35B-A3B **IQ3_S**), the IQ4_XS
Qwen pair, the Gemma 4 pair, and gpt-oss.

Immutability works because the `link-models` composite action links only
`*.gguf`. Every path a test *writes* — the `.sampling.toml` sidecar, the
Gemma/gpt-oss `.template.jinja` sidecars — resolves to a plain file in the
workspace `models/`, beside the symlink rather than through it. `/models`
does carry its own sidecars; they are deliberately not linked, so each run
regenerates the sampling sidecar from GGUF metadata and exercises that path.

**The `model.gguf` quant differs from the laptop's** — the box is
**IQ3_S**, Mike's Mac is **IQ4_XS** (he sized down after the default n_ctx
went to 32k). So a CI-only failure in a content assertion is a quant
suspect before it is a regression. Mike wants exactly this exposed: the
tests should be flexible enough to work across Qwen variants, and the box
can run an 8-bit occasionally to check the other direction. Known
hard-coded variant: `test_recommended_sampling_across_models` names
`Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` literally — it passes on the box only
because that file happens to be in `/models` too.

### What the `model` job does

`ci.yml`'s `model` job is Linux-only: `-c llama-cpp` (CUDA on Linux)
`-t ignored`, serialized by the `full` nextest profile, behind a
**`concurrency` group not keyed on the ref** so a second PR queues instead
of landing on a busy card. All 119 ignored tests, since `/models` is
complete.

If a runner ever has a *partial* model directory, exclude the affected
suites by name rather than expecting them to no-op — they **`panic!`
rather than skip** when their file is absent:

    -x session_gemma4 -x session_gptoss -x media_e2e_gemma

`-x` is a filterset, not a runtime skip, on purpose: an absent-weights test
that reports itself green is exactly the silent coverage loss #68 exists to
prevent.

**VRAM watch item.** `session_gemma4` loads a *dense* 31B (Q4_K_XL, 17 GB)
at the default 32k context. On a 24 GB 3090 that is close to the edge —
a `unable to allocate CUDA0 buffer` there is arithmetic, not a regression.
The Qwen MoE at 13 GB has plenty of room.

### Self-hosted changes one Actions habit

**No `Swatinem/rust-cache`**: a CUDA llama.cpp build does not fit in a 10 GB
Actions cache. Checkout is left on its **default `clean: true`**, so every
run rebuilds from scratch — Mike's call, and measured: both llama.cpp
builds are quick on this hardware (M2 Max 96 GB on the Mac side, and the
box is comparable depending on the work). The alternative, `clean: false`,
buys a warm `target/` but has a nastier failure mode: a file deleted on a
branch lingers, notably a removed `tests/*.rs` that cargo keeps compiling.
If the measurement ever argues for persistence, the fix is a target dir
**outside** the workspace, not `clean: false` — with one runner every job
shares one workspace, so a single cleaning job wipes the others' build.

Build parallelism is **per runner**, in each job's `env`: 16 of the Linux
box's 32 cores (it does other work), 4 on the macOS VM, which is all it was
given. `CARGO_BUILD_JOBS` is what cargo forwards to build scripts as
`NUM_JOBS`, which is what the `cmake` crate reads for llama.cpp, so one knob
covers both halves. **Never cap tests the same way**: `NEXTEST_TEST_THREADS`
overrides the `full` profile's `test-threads = 1`, which is the only thing
keeping two models off one card.

### macOS leg: still the uncertain one

A Parallels VM on the M2 Max: **4 cores, 32 GiB** of the host's 96. Metal
in the VM is unverified and deliberately out of scope — moeflux would want
the expert shards mounted in, tens of GB for a second copy of what the host
already has. So generation testing is **NVIDIA-only** and macOS stays a
compile-and-fast-tests gate; Metal generation is covered by the pre-commit
hook on the machine we develop on.

The `test` job's four metadata tests need `models/model.gguf` there too, so
the host's models directory is shared into the VM and `DRAMA_LLAMA_MODELS`
points at wherever Parallels mounts it (the `link-models` action reads it;
the mount point is not hard-coded anywhere).

**Watch for mmap over the share.** llama.cpp mmaps weights by default, and
mmap over a Parallels shared folder is the kind of thing that either works
fine or fails into a full read of the file. Four of these tests run *in
parallel* in a 32 GiB VM against a 13-17 GB file; if the mmap does not
hold, that is an OOM rather than a slow test. Fallbacks in order of
preference: a small real GGUF inside the VM (must be >1B params —
`test_model` asserts `n_params() > 1_000_000_000`), or
`-x llama_cpp::model::tests` on the macOS leg only.

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
- **GPU contention with the rest of the box** — it is not a dedicated CI
  machine. An Actions `concurrency` group only orders jobs *within this
  repo*; it cannot see other repos' runners, and it certainly cannot see
  non-CI workloads that pull a model onto the card periodically. Quiet
  today, so not a problem yet, but a future intermittent CUDA OOM or a
  slow model job is this before it is a regression. Options when it bites,
  in order of how much they cost: pause the other workload around a push
  (works, unpleasant), a machine-level `flock` around the model job, or a
  VRAM precondition check that fails fast with a legible message rather
  than mid-suite. Ask Mike before designing — he has context here that is
  deliberately not written down in this repo.
- **#69** — moeflux prefetch telemetry; unrelated to CI, revisit on a17b
  perf work.
