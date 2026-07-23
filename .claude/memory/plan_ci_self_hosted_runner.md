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
the host's models directory is shared into the VM **read-only**, landing at
`/Volumes/My Shared Files/models` (note the spaces — every path in the
`link-models` action is quoted for exactly this). `DRAMA_LLAMA_MODELS`
points at it.

**Set it in `~/actions-runner/.env`, not in a shell rc file.** The runner
runs under launchd (macOS) / systemd (Linux); neither sources `.zshrc`,
`.zprofile`, or `.zshenv`. `.env` in the runner's own directory is the
documented mechanism and is read by `runsvc.sh` on both platforms. No
quotes around the value — the file is parsed as `KEY=rest-of-line`, so
quotes would end up *in* the path.

Note the share carries Mike's laptop layout, so on macOS `model.gguf` is
the **IQ4_XS** Qwen while on Linux it is IQ3_S. Free variant coverage,
and the reason a metadata assertion could in principle disagree across
the two legs.

**macOS is CPU-only in CI, and that is now a decision rather than a
default** (Mike, 2026-07-23) — Metal in the VM is unproven and weights
come over a share. Worth knowing what "CPU-only" can and cannot mean here:
the crate has no switch for it. On macOS the `llama-cpp-cpu` configuration
is *identical* to `llama-cpp` because Metal is unconditional in llama.cpp's
build; what actually keeps the GPU out of it is that the unignored tier
puts nothing there (`n_gpu_layers = 0`) and the `model` job is Linux-only.

### What a fresh runner actually needs

Two of the three were learned the hard way on 2026-07-23, and they were
**all the same failure wearing different hats** — see the PATH note below.

- **`cmake` AND `ninja`.** `llama-cpp-sys-3`'s build script drives
  llama.cpp through `cmake::Config::new(..).generator("Ninja")` — the
  generator is *hard-coded* there, so ninja is not a preference CI could
  route around by choosing Makefiles. Neither ships with Xcode CLT. Hosted
  `macos-latest` preinstalls both, which is exactly why the first hosted CI
  run never surfaced either. `.github/actions/runner-path` now checks both
  up front and names whichever is missing.
- **Xcode Command Line Tools** — clang, the macOS SDK, libclang for
  bindgen, git, python3.
- **rustup**, for `dtolnay/rust-toolchain@stable` to have something to
  drive.
- **NOT the Metal toolchain.** Predicted as a blocker, wrong, and now
  disproven empirically as well as from source: the `moeflux` and
  `trait-layer` configurations both checked clean on the CLT-only VM.
  Neither crate compiles a `.metal` at build time — `llama-cpp-sys-3` sets
  `GGML_METAL_EMBED_LIBRARY=ON` (embeds the shader *source*, compiles it at
  runtime through the Metal framework) and the published `moeflux` crate
  has **no build.rs at all**, just `include_str!("shaders.metal")` +
  `new_library_with_source` (`src/riir/backend/gpu/metal.rs`). Whether
  Metal *runs* in the VM remains open — CI does not ask, macOS being
  CPU-only.
- Both `llama-cpp-sys-3` and `moeflux` are crates.io dependencies with
  vendored sources — no submodules, no sibling working copies. The
  `[patch.crates-io]` class of blocker that held #51 up cannot recur here.

### The failure mode to suspect first: the runner service's PATH

Every single failure of the first two real runs was this, and it will be
the first guess for the next unexplained one too. The runner runs under
**launchd (macOS) / systemd (Linux)**, neither of which sources a login
shell. So `.zshrc`, `.zprofile`, `.zshenv`, `brew shellenv` — none of it is
visible to the service, which gets a minimal PATH snapshotted when the
runner was configured. **Everything installed afterwards is invisible.**

It cost three separate-looking failures: no `.zshrc` to put
`DRAMA_LLAMA_MODELS` in (answer: `~/actions-runner/.env`), `cmake` not
found on macOS, `nvcc` not found on Linux — the latter reported as
`No CMAKE_CUDA_COMPILER could be found` *while CMake had located the CUDA
toolkit headers at /usr/local/cuda perfectly well*. Read that message
carefully if it recurs: "could be found" is PATH, whereas "The CUDA
compiler identification is unknown" would be a host-compiler mismatch and
an entirely different fix.

Fixed in-repo (`.github/actions/runner-path`) rather than per-runner,
because per-runner `.path`/`.env` state is invisible to review, multiplies
by runner instance — the Linux paths say `runner-1`, and Linux jobs from
one workflow run were observed executing **concurrently**, so there is more
than one — and evaporates the next time a runner is re-registered. The
action no-ops on any runner that already has a sane PATH.

**Corollary worth remembering: more than one Linux runner is registered.**
The GPU concurrency group on the `model` job is therefore load-bearing, not
belt-and-braces.

### Measured build times (2026-07-23, run 30003151215)

Mike's "builds are quick on balerion" was right by a wide margin, and this
is what retires the `clean: false` question — there is nothing here worth
the stale-file hazard.

- `permutations (linux)`, **clean tree, all three Linux configurations**
  (trait-layer, llama.cpp+CUDA, llama.cpp-CPU — two full cmake+ninja
  llama.cpp builds among them): **3m16s**.
- `llama-cpp-cpu` alone, from-scratch llama.cpp build plus
  `check --all-targets`: ~36s.

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

## What the ignored tier caught on its first CUDA run (2026-07-23)

**119 tests, 464s.** First run: 113 passed / 6 failed. After two fixes:
**116 passed / 3 failed.** The bet in #70 — that a model-capable runner
turns CI from a compile gate into a real regression gate — paid out
immediately, and paid out on a *library* bug, not on CI plumbing.

### Two real defects, fixed

- **`bin/blallama` dropped every symlinked model.** `list_entries` used
  `DirEntry::metadata()`, which **does not traverse the link** — it is
  `symlink_metadata` in all but name, returning `is_symlink() == true` and
  `is_file() == false`. The comment directly above it asserted the
  opposite, which is how it survived review: the code read as though it
  had already handled the case. `/api/tags` came back empty.
  **Not CI-only** — the same doc comment names the moeflux
  `mlx`/`artifacts`/`root` symlink layout as the case it exists to
  support, and that was broken too. Invisible on a dev box because a dev
  `models/` holds regular files and hardlinks; CI is simply the first
  environment to symlink them in. Verified with a standalone program
  before believing either the docs or the comment.
- **`test_usage_counters_across_append_only_calls` needed two models
  resident.** It held the cache-on session alive while constructing the
  cache-off one: 13 + 13 GB, free on a 96 GB Mac, impossible on a 24 GB
  card. `drop(session)` first; nothing below it needed the session.

The generalizable lesson: **a 24 GB card and a symlinked, read-only model
directory are both configurations no developer here runs**, so they are
exactly where the untested assumptions were. Expect the next batch of
CI-only failures to have the same flavour rather than to be regressions.

### The other three, all resolved — and one is an upstream bug

- **`all_shapes_match_python_jinja2`** — `uv` was not on the *service's*
  PATH (`/usr/bin/env: 'uv': No such file or directory`). Third instance
  of the PATH failure above; `$HOME/.local/bin` added to `runner-path`.
  Deliberately NOT in that action's required-tools list: one test needs
  uv, and failing the whole job early over it is worse than one red test.

- **`gptoss_eog_token_set` — an upstream llama.cpp bug, and my first
  diagnosis was wrong.** `eot()` is `<|end|>` on macOS and
  `<|endoftext|>` on Linux. I concluded the two `.gguf` files must
  differ, reasoning that identical bytes and a pinned `llama-cpp-sys-3`
  cannot produce different answers. Mike ran `sha256sum`: **identical**.
  The reasoning was only ever valid if the *compiled code* were identical
  too, and it is not — different platform, different C++ standard
  library.

  `llama_vocab::impl::load` auto-detects EOT by iterating `token_to_id`
  and taking the **first** entry whose text is on a candidate list
  (llama-vocab.cpp, loop at ~2564). `token_to_id` is a
  `std::unordered_map<std::string, llama_token>` (~1811). Iteration order
  of an unordered container is unspecified, so for any vocab holding two
  or more candidates the winner is whichever the standard library hashed
  first. gpt-oss holds both `<|end|>` and `<|endoftext|>`. libc++ yields
  one, libstdc++ the other. **Worth reporting upstream**; Mike notes it
  may already have changed, and a `llama-cpp-sys` bump will tell us.

  Harmless for us *precisely because* of
  [[eog_is_not_eos_plus_eot]] — we stop on `eog_tokens()` and never on a
  set built from `eot()`. The label wobbles across platforms; the
  predicate does not. That memo's thesis is now load-bearing rather than
  cautious.

  The test was also **hiding its own point**: the eot pin ran *before*
  the eog assertions, so on Linux it aborted first and the
  `<|end|>`-stays-generatable carve-out — the entire reason the test
  exists — was never verified there. Reordered, and the pin now accepts
  either value.

- **`regression_llama_cpp_baseline` — the golden is now cross-quant.**
  Mike's ask: cross-quant invariance is desirable, at least across the
  quants we actually use.

  Measured before deciding. Running IQ3_S against the IQ4_XS golden,
  `prompt_tokens`, `tokens` and `pieces` reproduce **exactly**; only
  logit magnitudes move (0.59-0.99 nats, 15/20 shared ids at prefill).
  So the harness's teeth — `tokens` alone is 32 argmax assertions over a
  248k vocab — are quant-invariant. They are now asserted
  unconditionally, while magnitudes are asserted only against the quant
  that produced them and otherwise fall back to the treatment
  `logits_step_n` already gets (argmax asserted, tail printed). Same rule
  the file already encoded, applied along one more axis.

  **Not** a widened tolerance: 15/20 shared ids is a membership change,
  and per [[logit_comparability_across_backends]] a tolerance loose
  enough to swallow one has stopped testing what it exists to test.

  The old identity guard could not have caught this: it compared
  `model_meta.n_vocab`, which is equal across every quant of one model —
  an architecture check wearing an identity label. `model_meta.desc` now
  carries llama.cpp's own description, which names the quant
  (`qwen35moe 35B.A3B IQ4_XS - 4.25 bpw`).

  Side finding from regenerating the golden: **top-K membership is not
  stable run-to-run on one machine either** — token-level signals came
  back byte-identical, the logit tail did not. Consistent with the file's
  own note that ranks 10/11 sit 0.0001 apart, and independent support for
  not asserting those numbers across quants.

### Where it ended (2026-07-23)

**All six jobs green, both OSes, `119 tests run: 119 passed` in 461s.**
The arc took four CI rounds: PATH → ninja → the two real defects → the
three data/provenance fixes.

Timings, for judging whether something later has regressed:

| job | time | note |
|---|---|---|
| `model` (linux, cuda) | ~7m45s | 119 model tests, serialized |
| `gate` (linux) | ~3m15s | three configs, two llama.cpp builds |
| `unignored` (macos) | ~8m04s | Metal build + 468 tests on 4 cores |
| `unignored` (linux) | ~2m50s | |

Balerion beats the Mac on the CPU-bound tests — `json_integration_lazy_grammar`
~21s vs ~60s — because the grammar engine is rayon-parallel and rayon's
pool defaults to `available_parallelism()`. That is now capped
(`RAYON_NUM_THREADS`, per-runner) so it stops taking all 32 on a shared
box. Verified in rayon-core 1.13.0 `get_num_threads`: RAYON_NUM_THREADS
first, then `available_parallelism()` — **not** the `num_cpus` crate,
though the deprecated `RAYON_RS_NUM_CPUS` is still honoured.

### Provenance the golden now carries, and why each field exists

Every one of these was added because something *looked* like identity and
was not:

- `backend` is `llama-cpp/<accel>`, not `llama-cpp` — the old value could
  not distinguish CUDA from Metal. Only the family before the `/` is
  asserted.
- `model_meta.desc` carries the quant; `n_vocab` cannot.
- `system_info` is llama.cpp's own machine view. Recorded, never
  asserted.
- The GPU **driver** is invisible in-process, so CI tees `nvidia-smi`
  into the uploaded artifact — kept on green runs too, because the useful
  one is the earlier passing run you diff a later failure against.

**Determinism is measured, not assumed**: three consecutive captures on
one machine are *bitwise* identical, logit tail included, max delta 0.0.
So any future golden diff is cross-backend, cross-quant, or real — there
is no run-to-run noise to budget for. Details in `tests/regression.rs`'s
module docs.

### Two wrong conclusions I reached, both the same shape

Worth keeping, because the failure mode is subtle and I hit it twice in
one session: **treating "same name" as "same inputs."**

1. gpt-oss `eot` differed across machines, so I concluded the two
   identically-named `.gguf` files must differ — "identical bytes cannot
   give different answers." Mike's `sha256sum` said identical. The
   reasoning was only valid if the *compiled code* were identical too,
   and it was not: different platform, different C++ standard library.
2. Regenerating the golden changed the logit tail, so I reported
   run-to-run nondeterminism. The old golden had been captured on **CUDA**
   (commit `7762b9b`, and the file's own doc comment said so); I had
   regenerated on Metal. Cross-backend, exactly as documented.

Both times the fix was the same: **establish what actually differs before
reasoning from what appears not to.** Check the provenance of a fixture
before treating it as a control.

### A discipline the tooling cannot enforce

**Do not `git push` while the model job is in flight.** The workflow's
`concurrency.cancel-in-progress: true` is Mike's stated preference and is
right in general, but it cancels the *whole run*, and a job-level
concurrency group cannot opt out of that. Cost a live GPU job once here.
Check `gh run view --json jobs` first; a *queued* model job is free to
supersede, a running one is not.

## Open issues

- **#70 — closed, and the `push` trigger STAYS.** Mike's call,
  2026-07-23: "we can afford to wait 10m each push. It doesn't tie up my
  mac or balerion. It gives us useful feedback if it fails." The whole
  premise of that issue was hosted-minutes cost, and self-hosting
  dissolved it. **Do not re-propose dropping `push`** — it was considered
  and rejected on its merits, not forgotten. (`paths-ignore` already
  keeps docs-only pushes free, verified in practice: a `.claude/**`-only
  push creates no run at all and so cannot even cancel one in flight.)
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

## What comes next for CI (Mike, 2026-07-23)

**Coverage, paired with the README rewrite**, so the badge lands in the
same session it has somewhere to live. His reasoning on badges is worth
keeping because it shapes what we point one at: a green badge on a crate
is a good signal, a red one reads as "they bothered once and stopped —
or it points at main and they got unlucky." So the badge should point at
something that is *reliably* green, and be worth trusting when it isn't.

Two things that make coverage here unusual, both worth remembering
before designing that session:

- **Meaningful coverage needs the self-hosted box.** 119 of the tests
  cannot run without weights and a GPU, and they are the ones exercising
  the interesting paths — Session, the cache, the dialects, mtmd. A
  coverage number from the unignored tier alone would systematically
  understate the tested surface *and* mis-attribute which code is
  actually exercised. This is a second payoff from the runner that was
  not part of #70's argument.
- **It has to go through `scripts/test.py`**, like everything else
  (#68), or the coverage build silently becomes a fifth configuration
  nobody else runs. `cargo-llvm-cov` has first-class nextest support
  (`cargo llvm-cov nextest`), which is the natural fit — but the
  tier/configuration axes are the script's to own, not the workflow's.

**After 1.0**: stricter branch protection on `main`, with only new
releases pushed there. That changes the required-checks calculus — see
the `paths-ignore` caveat in `ci.yml`, since a skipped required check
blocks a merge rather than passing it.
