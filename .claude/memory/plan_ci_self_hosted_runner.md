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

### The four model-needing "unignored" tests — two things to fix/understand

1. **They violate the tier invariant.** The whole premise (see
   [[test_topology]], `.githooks/pre-commit`, `.config/nextest.toml`) is
   that the *unignored* tier loads no model. Four unignored tests that
   need `models/model.gguf` means either they are mis-classified and want
   `#[ignore]`, or the invariant needs a stated exception. They pass
   locally only because `models/model.gguf` exists there. **Find them**
   (they'll be the ones failing in the CI `test` job logs — uploaded as
   `test-logs-*` artifacts on failure) and decide: `#[ignore]` them, or
   give CI a tiny real GGUF fixture so metadata loading has something to
   read.
2. **The 11s-to-fail is unexplained.** A missing model should fail fast.
   11s suggests something is actually *loading* — a broken symlink chased,
   a GGUF metadata read attempted against a placeholder, or a retry/timeout
   somewhere in `Model::from_file`. Hypothesis only; do not build on it.
   Worth understanding before deciding the fix, because "give CI a fixture"
   vs "`#[ignore]` them" depends on what those 11s are.

## Next session: stand up the self-hosted runner

The box: **32 cores, 128 GiB RAM, a 3090.** Already used for other
projects. Two payoffs (from #70): free minutes (Mike has exhausted GH
Actions credit before) and — the bigger one — it can run the *model*
tests, which no hosted runner can. That turns CI from a compile gate into
a real regression gate, and is the only way the `ignored` tier and CUDA
ever get automated.

### The workflow is ALREADY switched to self-hosted (`b426f3c`+, night of
### 2026-07-22)

Done ahead of standing up the runner, to keep Mike's next push off hosted
minutes. `ci.yml` now targets `[self-hosted, linux]` and
`[self-hosted, macos]` for all three jobs. **Consequence to expect:** with
no runner registered yet, pushed runs sit **queued/pending** — they do not
fail and do not spend hosted minutes. That is intended. When the runner
registers, pending runs may pick up; cancel the stale ones if they're
noise.

**So the first thing on the box is register a runner whose labels match**
(`self-hosted` + `linux` / `macos`). If you pick different labels, edit the
matrix `runner:` arrays to match — GitHub matches labels literally.

### Linux leg (the straightforward one)

Mike's stated plan: new dedicated account, rootless Docker, a systemd unit
for the Docker side and one for the GH runner itself, then register the
runner. The workflow side already calls `scripts/test.py` — the #68
invariant holds, the runner runs what the hook runs. Model-test jobs (when
added — not on yet) **need a `concurrency` group**: a 30B-class model
barely fits once, the `full` nextest profile caps `test-threads = 1`, and
two PRs must not land on the GPU at the same time.

### macOS leg (the uncertain one)

Trickier: a new macOS VM, and **it is unknown whether Metal works inside
the VM.** If it does not, the moeflux / Metal generation tests cannot run
there and generation testing is **NVIDIA-only**; the macOS leg stays
compile-gate (`check --ci`) like the hosted one. Decide this by testing
Metal in the VM before wiring any generation job to it — do not assume
either way.

## The context-budget snag (may force a smaller model)

Default context is **32k** (`cli::DEFAULT_N_CTX`) and the test model is
**~17 GiB**. Unknown whether that context fits on a **3090 (24 GB)**,
*especially with the image/mtmd model sidecar also resident*. If it does
not, the runner needs a **smaller Qwen**, which **would change some test
expectations** (token counts, specific-output assertions, metadata
values).

Mike is **not opposed** to switching: a smaller model is faster, and
grammar-constrained tool calling should still work fine — which is the
capability the interesting tests exercise. So if 32k + 17 GiB + sidecar
does not fit, the move is: pick a smaller Qwen, re-baseline the affected
test expectations against it, and note in each what model it assumes.
Don't fight the VRAM; change the model.

## Open issues

- **#70** — CI cost/runner. Mike added a note there this evening. Also
  holds the "drop the `push` trigger, keep `pull_request` +
  `workflow_dispatch`" change — do that first thing (one line), since the
  `push` leg was only there to validate the workflow and it has.
- **#69** — moeflux prefetch telemetry; unrelated to CI, revisit on a17b
  perf work.
