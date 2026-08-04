#!/usr/bin/env python3
"""Single source of truth for drama_llama's test topology.

The justfile and CI both call this script; the git hooks call the
justfile. That chain is the point: the tests that run before a commit
are byte-for-byte the ones that run in CI, and neither can drift from
the other by editing only one of them.

Two orthogonal axes, which the old justfile-only setup conflated:

    configuration   which features are on, i.e. which backend(s) exist
    tier            which tests run — unignored, ignored, or all

`just test full` used to mean `--run-ignored only`, so neither it nor a
bare `just test` ever ran the whole suite; you needed both, and nothing
said so. Here `ignored` is the honest name for that set and `all` is a
genuinely-everything mode (issue #68).

Python rather than shell because this repo builds on Windows too, and
because argparse gives the thing a real `--help`.

Usage
-----
    scripts/test.py run                       # llama-cpp, unignored
    scripts/test.py run --tier all            # llama-cpp, everything
    scripts/test.py run -c moeflux --tier all # moeflux-only, everything
    scripts/test.py run --filter strawberry   # one test, any tier
    scripts/test.py run -t ignored -x session_gptoss   # minus a suite
    scripts/test.py check --config all        # the permutation gate
    scripts/test.py coverage -t all           # instrumented, with a report
    scripts/test.py configs                   # what exists, and why
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Set by --dry-run. Every cargo invocation goes through `run_command`, so
# printing-and-returning there is enough to make the whole topology
# inspectable without paying for a build — which is how the wiring gets
# tested at all, given that actually running it costs tens of minutes.
DRY_RUN = False

# The moeflux backend selects its model at COMPILE time — each model is
# its own feature and exactly one must be enabled, which is what implies
# `moeflux` itself. a3b is the fast one (~12 tok/s); a17b runs at ~2 and
# exists for backup Agora council work and tests. Phase 7 replaces the
# compile-time selection with a runtime variant config; don't lean
# further into it in the meantime.
DEFAULT_MOEFLUX_MODEL = "qwen3-6-35b-a3b"

# Variant -> why you might not want to run the model tests on it. a3b is
# the fast one and the default for exactly that reason; the others are
# real configurations that are simply not a dev-loop cost. Empty string
# means "no warning".
MOEFLUX_MODELS: dict[str, str] = {
    "qwen3-6-35b-a3b": "",
    "qwen3-5-a17b": "~2 tok/s — the model-test tiers take hours, "
    "possibly all night. It exists for backup Agora council work and "
    "tests, not for the dev loop.",
    "cogito-v2-671b": "does not fit in RAM on this class of machine; "
    "experts re-page on every routing change and warm tokens cost "
    "~12 s each. The model-test tiers are not practical here.",
}

# Features that name no backend: they compile against the trait layer and
# are legal in every configuration. `cli` and `axum` belong here as of
# #68 — both used to imply `llama-cpp`, which made a moeflux-only
# front-end impossible to build even though the library supported it.
# `agora-agentkit` is the implicit optional-dependency feature gating
# the `soul_forge` example (its `Soul` source of truth). Backend-blind,
# so it rides with the agnostic group — otherwise no configuration
# would ever compile the example and it could rot silently.
AGNOSTIC = ["toml", "serde", "stats", "json-schema", "egui", "agora-agentkit"]

# `webchat` and `mtmd` name concrete llama.cpp types, so they are not
# agnostic and never appear in a moeflux-only set. `mtmd` implies
# `media`; moeflux configurations ask for `media` directly and get typed
# "media unsupported" errors from `NoVision` rather than silent drops.
LLAMA_ONLY = ["llama-cpp", "webchat", "mtmd", "cli", "axum"]
MOEFLUX_EXTRA = ["media", "cli", "axum"]


@dataclass
class Config:
    """One point in the feature-permutation space."""

    name: str
    features: list[str]
    help: str
    # Adds the moeflux model feature, which is what actually enables the
    # backend. Parameterized because the variant is a compile-time
    # choice, not a runtime one.
    moeflux: bool = False
    # Linux-only accelerator features. CUDA is deliberately kept OUT of
    # the crate's default features and chosen here instead: a bare
    # `cargo build` stays portable while the dev loop stays accelerated.
    linux_extra: list[str] = field(default_factory=list)
    macos_only: bool = False
    # Own CARGO_TARGET_DIR. Only the CPU config needs one — it differs
    # from the GPU config by `cuda`, and alternating between them in a
    # shared dir evicts llama-cpp-sys's C build (a ~20-40 min rebuild).
    # Every other config leaves llama-cpp-sys's own features untouched,
    # so its build survives and they can share `target/`.
    target_subdir: str | None = None
    # Buildable on a hosted runner: no GPU, no model weights, no CUDA
    # toolchain. Note this is about *building*; the weights-dependent tiers
    # (`ignored`, `all`) stay local in every configuration, so what CI runs is
    # the permutation gate plus the `unignored` tier (#51). Building llama.cpp
    # on a runner is slow but proven — llama-cpp-sys already does it on three
    # OSes; it is CUDA, not the C build, that CI cannot have.
    ci_eligible: bool = True

    def feature_list(self, model: str) -> list[str]:
        features = list(self.features)
        if self.moeflux:
            features.append(f"moeflux-model-{model}")
        if self.linux_extra and platform.system() == "Linux":
            features.extend(self.linux_extra)
        # Deduped: the per-backend groups overlap on the front-end
        # features, so `both` would otherwise name `cli,axum` twice.
        # Cargo tolerates that; `--help` output shouldn't have to.
        return list(dict.fromkeys(features))

    def target_dir(self) -> Path:
        base = REPO / "target"
        # The split dir exists only because CUDA makes two *different*
        # llama.cpp builds that would evict each other. Off Linux there is no
        # CUDA, this config's features are identical to `llama-cpp`, and a
        # separate dir would just build llama.cpp a second time for nothing.
        if self.target_subdir and platform.system() == "Linux":
            return base / self.target_subdir
        return base


CONFIGS: dict[str, Config] = {
    c.name: c
    for c in [
        Config(
            name="trait-layer",
            features=AGNOSTIC + ["media"],
            help="No backend at all: the backend-agnostic trait layer "
            "with no C dependency. Cheapest thing in the matrix and the "
            "only one with nothing to install, so it is the CI canary "
            "for accidental llama.cpp leakage into generic code. Note "
            "`cli`/`axum` are absent: a `--backend` selector over an "
            "empty set of backends is a hard error by design.",
        ),
        Config(
            name="llama-cpp",
            features=AGNOSTIC + LLAMA_ONLY,
            linux_extra=["cuda"],
            help="llama.cpp only, GPU-accelerated (CUDA on Linux, Metal "
            "on macOS — Metal needs no feature flag). The everyday dev "
            "loop and what the pre-commit hook runs. Named explicitly "
            "rather than relying on `default`, since `llama-cpp` may not "
            "always be a default feature.",
            # On Linux this config carries `cuda`, so it wants nvcc. The
            # CI-eligible llama.cpp build is `llama-cpp-cpu`.
            ci_eligible=False,
        ),
        Config(
            name="llama-cpp-cpu",
            features=AGNOSTIC + LLAMA_ONLY,
            target_subdir="cpu",
            help="llama.cpp with no CUDA. On macOS this is identical to "
            "`llama-cpp` (Metal is unconditional) and shares its target "
            "dir; on Linux it is the no-accelerator build, in its own "
            "dir so the two do not evict each other's C build. This is "
            "the llama.cpp configuration a hosted runner can have.",
        ),
        Config(
            name="moeflux",
            features=AGNOSTIC + MOEFLUX_EXTRA,
            moeflux=True,
            macos_only=True,
            help="moeflux only — no llama.cpp, no C dependency. A "
            "supported configuration as of #68; before that it did not "
            "build at all. Because there is no llama.cpp in this build "
            "there are no llama.cpp model tests to avoid, so `--tier "
            "all` runs this configuration's entire universe with no "
            "filterset — which is how every `cfg(moeflux)` unit test "
            "under `src/` becomes reachable (it was not, from any "
            "recipe, before #68).",
        ),
        Config(
            name="both",
            features=AGNOSTIC + LLAMA_ONLY + MOEFLUX_EXTRA,
            moeflux=True,
            macos_only=True,
            help="Both backends linked at once. This is the "
            "configuration the crate's documented hazard lives in — see "
            "the `LlamaCppSession` doc block, which exists to warn that "
            "a bare `Session::from_path(..)` compiles with one backend "
            "and breaks with two. Worth running the unignored tier here "
            "regularly; it used to be compiled and then almost entirely "
            "unrun.",
        ),
    ]
}

@dataclass(frozen=True)
class Tier:
    """Which tests run, and under which nextest profile.

    A tier is deliberately NOT a fixed argv. `cargo llvm-cov` defines
    its own `--profile` (the *cargo* build profile), so passing nextest's
    `--profile full` through it selects a cargo profile that does not
    exist. The two callers therefore render the same tier differently —
    `run` as a flag, `coverage` as `NEXTEST_PROFILE` in the environment,
    which nextest reads and llvm-cov never sees. Rendering per caller
    keeps that collision explicit instead of rewriting an argv string
    somewhere downstream.
    """

    # nextest profile. `full` is `test-threads = 1` (.config/nextest.toml),
    # the only thing keeping two 13-19 GB models off one card.
    profile: str | None
    # nextest --run-ignored value.
    run_ignored: str | None

    def flags(self) -> list[str]:
        """As command-line arguments, for plain `cargo nextest run`."""
        args: list[str] = []
        if self.profile:
            args += ["--profile", self.profile]
        if self.run_ignored:
            args += ["--run-ignored", self.run_ignored]
        return args

    def env(self) -> dict[str, str]:
        """The half that must travel as environment, plus its flags.

        See the class doc: only the profile is ambiguous, so only the
        profile moves. `--run-ignored` is not an llvm-cov option and is
        forwarded to nextest untouched.
        """
        return {"NEXTEST_PROFILE": self.profile} if self.profile else {}

    def env_flags(self) -> list[str]:
        """The flags that are safe to pass *through* `cargo llvm-cov`."""
        return ["--run-ignored", self.run_ignored] if self.run_ignored else []


# unignored  the fast loop: no model is loaded, so it stays parallel.
# ignored    ONLY the #[ignore]'d tests. Each loads a 13-19 GB model onto
#            the GPU and only one fits, hence profile `full`. This is what
#            `just test full` meant, under a name that says so.
# all        genuinely everything, serialized for the same reason.
TIERS: dict[str, Tier] = {
    "unignored": Tier(profile=None, run_ignored=None),
    "ignored": Tier(profile="full", run_ignored="only"),
    "all": Tier(profile="full", run_ignored="all"),
}


def log_path(name: str) -> Path:
    d = REPO / "target" / "test-logs"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.log"


def run_command(
    cmd: list[str],
    target_dir: Path,
    log: Path | None,
    extra_env: dict[str, str] | None = None,
) -> int:
    """Run `cmd`, streaming to stdout and (optionally) tee'ing to `log`.

    Tee'd rather than redirected so a failure can be read back after the
    fact instead of re-run to be seen — and streamed rather than
    captured so a long model test shows progress instead of going quiet
    for ten minutes.

    `extra_env` is printed alongside the command for the same reason the
    target dir is: a run that is only reproducible if you also knew about
    an invisible variable is not reproducible.
    """
    env = dict(os.environ, CARGO_TARGET_DIR=str(target_dir), **(extra_env or {}))
    print(f"+ CARGO_TARGET_DIR={target_dir}", flush=True)
    for key, value in (extra_env or {}).items():
        print(f"+ {key}={value}", flush=True)
    print(f"+ {' '.join(cmd)}", flush=True)
    if log is not None:
        print(f"+ log: {log}", flush=True)
    if DRY_RUN:
        return 0

    with subprocess.Popen(
        cmd,
        cwd=REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        # A test's output is not guaranteed to be valid UTF-8, and the
        # runner must never be the thing that fails. llama.cpp's model
        # loader dumps `tokenizer.ggml.tokens`, and a byte-level BPE
        # vocab (pixtral/gpt2 — Mistral Small 4 is the case that found
        # this) contains lone continuation bytes. Strict decoding
        # raised UnicodeDecodeError out of the read loop and killed the
        # whole run before a single test executed.
        errors="replace",
        bufsize=1,
    ) as proc, (
        open(log, "w", encoding="utf-8") if log else _NullWriter()
    ) as sink:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            sink.write(line)
        return proc.wait()


class _NullWriter:
    """`open()`-shaped sink for the no-log case, so the caller has one path."""

    def __enter__(self) -> "_NullWriter":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def write(self, _: str) -> None:
        return None


def resolve_config(name: str) -> Config:
    try:
        config = CONFIGS[name]
    except KeyError:
        sys.exit(
            f"unknown config {name!r}; choose from "
            f"{', '.join(CONFIGS)} (see `test.py configs`)"
        )
    if config.macos_only and platform.system() != "Darwin":
        sys.exit(f"config {name!r} is macOS-only (moeflux is Metal-backed)")
    return config


def require_nextest() -> None:
    if shutil.which("cargo-nextest") is None:
        sys.exit(
            "cargo-nextest not found — install it with `just setup` "
            "(or `cargo install cargo-nextest --locked`)"
        )


def warn_slow_variant(config: Config, tier: str, model: str) -> None:
    """Say so before spending a night on it.

    Only the tiers that actually load weights are worth warning about —
    `unignored` never touches a model, so a slow variant costs nothing
    there. A warning rather than a refusal: running a17b's model tests is
    a legitimate thing to want, just not by accident.
    """
    if not config.moeflux or tier == "unignored":
        return
    note = MOEFLUX_MODELS.get(model, "")
    if note:
        print(f"\n!! moeflux variant {model}: {note}", file=sys.stderr)
        print(
            f"!! This runs the model tests. For the fast variant, drop "
            f"--moeflux-model (defaults to {DEFAULT_MOEFLUX_MODEL}); for "
            f"no model tests, use `-t unignored`.\n",
            file=sys.stderr,
        )


def filterset(include: str | None, exclude: list[str]) -> str:
    """The nextest filterset for one include substring and N excludes.

    A substring is matched against test AND binary names, because the
    thing a caller names may be either — `session_gemma4` is a binary,
    `media_e2e_gemma` is a test inside the lib binary, and the caller
    should not have to know which.

    Exclusion exists for machines with a *partial* model directory: the
    CI box has `models/model.gguf` and its projector, not the Gemma 4 or
    gpt-oss weights, and the suites that need those `panic!` rather than
    skip. Expressed as a filterset rather than as a runtime skip on
    purpose — a skipped-because-absent test that reports itself green is
    exactly the silent coverage loss #68 exists to prevent, whereas an
    excluded one shows up in nextest's own "N tests run, M skipped" line
    and in the printed command.
    """
    expr = (
        f"test(~{include}) + binary(~{include})" if include else "all()"
    )
    if exclude:
        dropped = " + ".join(
            f"test(~{name}) + binary(~{name})" for name in exclude
        )
        expr = f"({expr}) - ({dropped})"
    return expr


def selection(
    config: Config, args: argparse.Namespace
) -> tuple[Tier, list[str], str]:
    """Resolve (tier, trailing nextest args, log name) for a test run.

    Shared by `run` and `coverage` so the two cannot select different
    tests — the whole point of this script, one level down. The tier is
    returned rather than rendered because its profile has to travel
    differently for each; see `Tier`.
    """
    if args.filter:
        # A named test is asked for by name, so run it whichever list it
        # is on rather than making the caller remember — and uncaptured,
        # so the suites' block/emission dumps are visible on a pass and
        # not only on a failure.
        tier, extra = TIERS["all"], ["--no-capture"]
        name = f"{config.name}-{sanitize(args.filter)}"
    else:
        tier, extra = TIERS[args.tier], []
        name = f"{config.name}-{args.tier}"

    if args.filter or args.exclude:
        extra += ["-E", filterset(args.filter, args.exclude)]

    return tier, extra, name


def preflight_gpu(features: list[str]) -> int:
    """Fail a `cuda` run before the build if the driver is unusable.

    Strictly a fast path to the same verdict
    `llama_cpp::tests::cuda_build_has_a_gpu_device` reaches — that test
    asks ggml what this build actually found and stays the authority,
    because a `nvidia-smi` that works proves nothing about a llama.cpp
    that cmake quietly configured without CUDA. What this buys is the
    ~40 minutes between "start a cold cuda build" and "watch the model
    tier pass on the CPU": an unusable driver reports here in two
    seconds.

    `DRAMA_LLAMA_ALLOW_CPU_FALLBACK=1` skips it, same as the test.
    """
    if "cuda" not in features or platform.system() != "Linux":
        return 0
    if os.environ.get("DRAMA_LLAMA_ALLOW_CPU_FALLBACK"):
        print("= preflight: CPU fallback allowed — skipping", flush=True)
        return 0
    if shutil.which("nvidia-smi") is None:
        print(
            "preflight: built with `cuda` but no `nvidia-smi` on PATH.\n"
            "Set DRAMA_LLAMA_ALLOW_CPU_FALLBACK=1 to run on the CPU anyway.",
            file=sys.stderr,
        )
        return 1
    probe = subprocess.run(
        ["nvidia-smi"], capture_output=True, text=True, check=False
    )
    if probe.returncode != 0:
        print(
            "preflight: `nvidia-smi` failed — the driver is unusable, so a "
            "`cuda` build will silently run every model test on the CPU.\n"
            "\n"
            f"{(probe.stderr or probe.stdout).strip()}\n"
            "\n"
            "An unattended driver upgrade with the old kernel module still "
            "loaded needs a reboot (or an nvidia module reload).\n"
            "Set DRAMA_LLAMA_ALLOW_CPU_FALLBACK=1 to run on the CPU anyway.",
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    if not DRY_RUN:
        require_nextest()
    config = resolve_config(args.config)
    # `--filter` implies the `all` tier, so it warns too.
    warn_slow_variant(
        config, "all" if args.filter else args.tier, args.moeflux_model
    )
    features = config.feature_list(args.moeflux_model)
    if not DRY_RUN:
        rc = preflight_gpu(features)
        if rc != 0:
            return rc
    tier, extra, name = selection(config, args)

    cmd = [
        "cargo",
        "nextest",
        "run",
        "--no-default-features",
        "--features",
        ",".join(features),
        "--no-fail-fast",
        *tier.flags(),
        *extra,
    ]

    return run_command(cmd, config.target_dir(), log_path(name))


# Directories that execute during the run but do not count as covered
# surface. Two different justifications, deliberately kept apart:
#
# BY CATEGORY — `tests/`, `examples/`, `benches/` are test *inputs*, not
# the thing under test. Integration-test bodies are ~100% covered by
# definition and would inflate the number by several points while saying
# nothing about `src/`.
#
# BY DECISION — the two toy binaries. Mike's call, 2026-07-23, with the
# reasoning recorded on the issue: they are demos (regurgitater exists to
# show a model reciting The Hobbit chapter one), they fail obviously when
# they fail, and testing them is real work for little signal. Between
# them they are ~284 lines, about 1.2 points of the total. Excluded so
# the headline number describes the library and the serving surface
# rather than being dragged by two things nobody depends on.
#
# `bin/blallama` is NOT excluded: it is a real serving surface with real
# integration tests, and it went 17.6% -> 49.4% once its coverage was
# actually being recorded.
COVERAGE_IGNORE_DIRS = [
    "tests",
    "examples",
    "benches",
    "bin/regurgitater",
    "bin/settings_tool",
]
COVERAGE_IGNORE = (
    r"(^|/)(" + "|".join(COVERAGE_IGNORE_DIRS) + r")/"
)

# What none of the above CAN exclude is `#[cfg(test)] mod tests` inside
# `src/`: llvm-cov filters by file, and those live in the same file as
# the code they test. There is a lot of that in this crate, so read the
# reported percentage as an upper bound. The per-file table, which is the
# actually useful output, is unaffected.

# Where the artifacts land. Under `target/` (not the config's own target
# dir) for the same reason the test logs are, so a run in either the CUDA
# or the CPU configuration leaves its report somewhere one path can find.
COVERAGE_DIR = REPO / "target" / "coverage"


def require_llvm_cov() -> None:
    if shutil.which("cargo-llvm-cov") is None:
        sys.exit(
            "cargo-llvm-cov not found — install it with `just setup` "
            "(or `cargo install cargo-llvm-cov --locked`)"
        )


# The toolchain used wherever this script needs nightly (`coverage
# --doctests`, `docsrs`). Overridable so CI can pin a dated nightly when
# the current one is broken — nightly-2026-07-24 ICEs codegen'ing tokio
# under this crate's test profile (opt-level=3 + debug-assertions). A
# `+name` on the cargo command line outranks every other rustup override
# (RUSTUP_TOOLCHAIN, the action's default, rust-toolchain files), so a
# pin only works if it lands where the `+` is written: here.
NIGHTLY = os.environ.get("DRAMA_LLAMA_NIGHTLY", "nightly")


def require_nightly() -> None:
    """`--doctests` is nightly-only; fail before paying for a build."""
    try:
        installed = subprocess.run(
            ["rustup", "toolchain", "list"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        # No rustup (a distro toolchain, say). `cargo +nightly` will say
        # something useful on its own; don't block on a guess.
        return
    if NIGHTLY not in installed:
        sys.exit(
            f"--doctests needs the {NIGHTLY} toolchain (llvm-cov's doctest "
            f"support is unstable) — `rustup toolchain install {NIGHTLY}`"
        )


def cmd_coverage(args: argparse.Namespace) -> int:
    """Instrumented test run, then one or more reports over the same data.

    Split into `--no-report` + N × `report` on purpose: the profraw data
    is expensive (a full instrumented rebuild plus, at `-t all`, every
    model test) and every output format is a cheap re-read of it. So a
    CI job can emit the human summary, the lcov for an uploader, and the
    JSON the badge is cut from without running the suite three times.

    `--doctests` adds a second instrumented pass over the doctests and
    merges it into the same report. It is opt-in because llvm-cov's
    doctest support is nightly-only — and because BOTH passes then have
    to be nightly, since profraw written by a stable-built binary will
    not merge with profraw from a nightly-built one. So the flag
    switches the whole run to `cargo +nightly` rather than just the
    doctest pass. CI takes it: `test` and `model` stay on stable, so
    correctness confidence is unchanged, and measuring on nightly buys
    nightly-toolchain testing for free.
    """
    if not DRY_RUN:
        require_nextest()
        require_llvm_cov()
        if args.doctests:
            require_nightly()
    config = resolve_config(args.config)
    warn_slow_variant(
        config, "all" if args.filter else args.tier, args.moeflux_model
    )
    features = config.feature_list(args.moeflux_model)
    tier, extra, name = selection(config, args)
    name = f"coverage-{name}"
    target_dir = config.target_dir()
    cargo_features = [
        "--no-default-features",
        "--features",
        ",".join(features),
    ]

    # One toolchain for every pass. See the docstring: mixing them
    # produces profraw the merge step cannot read.
    cargo = ["cargo", f"+{NIGHTLY}"] if args.doctests else ["cargo"]

    code = run_command(
        [
            *cargo,
            "llvm-cov",
            "nextest",
            "--no-report",
            *cargo_features,
            "--no-fail-fast",
            # `tier.flags()` would pass `--profile full` to llvm-cov,
            # which reads it as a *cargo* profile. See `Tier`.
            *tier.env_flags(),
            *extra,
        ],
        target_dir,
        log_path(name),
        tier.env(),
    )
    # Deliberately not an early return. A red run still produced coverage
    # data, and on the run you most want to look at — the one that broke
    # something — bailing here would throw it away.
    if code != 0:
        print(
            f"\n!! tests exited {code}; reporting on partial data anyway",
            file=sys.stderr,
        )

    if args.doctests:
        # Second instrumented pass, same `--no-report` accumulation. Not
        # gated on the first one passing, for the reason above.
        rc = run_command(
            [*cargo, "llvm-cov", "--doc", "--no-report", *cargo_features],
            target_dir,
            log_path(f"{name}-doc"),
        )
        if rc != 0:
            print(
                f"\n!! doctests exited {rc}; reporting anyway",
                file=sys.stderr,
            )
            code = code or rc

    if not DRY_RUN:
        COVERAGE_DIR.mkdir(parents=True, exist_ok=True)
    reports: list[list[str]] = []
    if args.lcov:
        reports.append(["--lcov", "--output-path", str(args.lcov)])
    if args.json:
        reports.append(
            ["--json", "--summary-only", "--output-path", str(args.json)]
        )
    if args.html:
        reports.append(
            ["--html", "--output-dir", str(COVERAGE_DIR / "html")]
            + (["--open"] if args.open else [])
        )
    # The human summary goes last so it is the final thing on the
    # terminal, and it carries `--fail-under-lines` because it is the one
    # report guaranteed to run.
    #
    # `--summary-only` alone, NOT `--summary-only --text`: in llvm-cov's
    # vocabulary "text" is the annotated-source format, so adding it turns
    # a 40-line per-file table into 46k lines of listing. The per-file
    # table is the default and is what this is for.
    summary = ["--summary-only"]
    if args.fail_under is not None:
        summary += ["--fail-under-lines", str(args.fail_under)]
    reports.append(summary)

    failed = code
    for report in reports:
        # No feature flags here. `cargo llvm-cov report --help` lists
        # `--no-default-features` and `--features`, but the subcommand
        # rejects them at parse time ("invalid option ... for subcommand
        # 'report'"); the help text is shared across subcommands and
        # overstates this one. They are not needed either — `report`
        # reads the object list and profraw files the run above left
        # behind, so it already knows what was built.
        rc = run_command(
            [
                *cargo,
                "llvm-cov",
                "report",
                *(["--doctests"] if args.doctests else []),
                "--ignore-filename-regex",
                COVERAGE_IGNORE,
                *report,
            ],
            target_dir,
            None,
        )
        failed = failed or rc
    return failed


def selected_configs(
    name: str, model: str, ci: bool = False
) -> list[tuple[Config, list[str]]]:
    """Resolve `-c NAME` (or `-c all`) to the configurations to actually run.

    Shared by `check` and `doctest`, the two subcommands that sweep the
    permutation space rather than naming one point in it. Prints its own
    skip lines: a configuration silently not running is the failure mode
    this whole script exists to prevent, so every omission says why.
    """
    names = list(CONFIGS) if name == "all" else [name]
    out: list[tuple[Config, list[str]]] = []
    # Two configs can resolve to the same (features, target dir) — notably
    # llama-cpp and llama-cpp-cpu off Linux, where there is no CUDA to tell
    # them apart. Running the second is a cache hit rather than a cost, but
    # reporting it as a distinct permutation would overstate the coverage.
    seen: dict[tuple[str, str], str] = {}
    for n in names:
        config = CONFIGS[n]
        if config.macos_only and platform.system() != "Darwin":
            print(f"= skip {n}: macOS-only", flush=True)
            continue
        if ci and not config.ci_eligible:
            print(f"= skip {n}: not CI-eligible", flush=True)
            continue
        features = config.feature_list(model)
        key = (",".join(features), str(config.target_dir()))
        if key in seen:
            print(
                f"= skip {n}: identical to {seen[key]} on this platform",
                flush=True,
            )
            continue
        seen[key] = n
        out.append((config, features))
    return out


def sweep(
    label: str,
    configs: list[tuple[Config, list[str]]],
    build: "callable",
) -> int:
    """Run `build(features)` in each configuration; report every failure.

    Never short-circuits. Knowing that three configurations are broken
    beats rediscovering them one commit at a time.
    """
    failures: list[str] = []
    for config, features in configs:
        print(f"\n=== {label}: {config.name} ===", flush=True)
        code = run_command(
            build(features),
            config.target_dir(),
            log_path(f"{label}-{config.name}"),
        )
        if code != 0:
            failures.append(config.name)

    if failures:
        print(
            f"\n{label}: FAILED for {', '.join(failures)}", file=sys.stderr
        )
        return 1
    print(f"\n{label}: ok", flush=True)
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """The permutation gate: every configuration compiles, tests included.

    `--all-targets` is the point — the library building is not the same
    claim as the test targets building, and #68 was filed because a
    configuration that compiled as a library had ~38 broken test
    targets nobody had ever built.

    Note it does NOT cover doctests: `cargo check` does not compile them.
    That is `doctest`'s job, and it is why that subcommand sweeps
    configurations too rather than running just one.

    The `badge` verification is deliberately NOT here. It was added
    here on 2026-07-24 (6d4a9e2) on the theory that this gated the
    pre-commit hook — it does not. The hook runs `just check`; this is
    `just permutations`, which the hook skips as too slow, so the gate
    never ran and a stale badge reached CI in #82. It lives in the
    `just check` recipe now, and CI runs `badge` as its own step after
    the test job.
    """
    return sweep(
        "check",
        selected_configs(args.config, args.moeflux_model, args.ci),
        lambda features: [
            "cargo",
            "check",
            "--all-targets",
            "--no-default-features",
            "--features",
            ",".join(features),
        ],
    )


def cmd_doctest(args: argparse.Namespace) -> int:
    """Run the doctests, which nextest cannot.

    This is not a stylistic preference: cargo-nextest's process-per-test
    model has no doctest support at all (nextest-rs/nextest#16), so every
    other recipe in this file runs zero of them. Without this subcommand
    the ` ```rust ` blocks in `lib.rs` — the README among them, mounted
    with `#![doc = include_str!]` — would compile in nobody's CI and rot
    exactly as fast as an untested example.

    **It takes `-c all`, and CI uses it**, because a doctest is a
    per-configuration claim exactly like a test target is. `cargo check
    --all-targets` does NOT compile doctests, so the permutation gate
    cannot see them: `Backend::set_log_callback`'s example named
    `LlamaCppBackend` — a `llama-cpp`-gated type, in a doc comment on a
    backend-*agnostic* trait method — and failed to compile in every
    configuration without that feature. `trait-layer` exists as the
    canary for precisely that leak and could not catch this one.

    Same features and target dir as `check`/`just doc`, so it reuses
    those builds rather than making more.
    """
    return sweep(
        "doctest",
        selected_configs(args.config, args.moeflux_model, args.ci),
        lambda features: [
            "cargo",
            "test",
            "--doc",
            "--no-default-features",
            "--features",
            ",".join(features),
        ],
    )


def cmd_clippy(args: argparse.Namespace) -> int:
    """Clippy with warnings promoted to errors.

    The tree went warning-clean on 2026-07-24 (release eve, 86 warnings
    swept: auto-fix for the mechanical half, hand-fixes or site-level
    `#[allow]`-with-a-reason for the rest — never a blanket crate-level
    allow, which hides new fires along with the old). This gate is what
    keeps it clean: one new warning is one red job while the diff that
    introduced it is still open.

    Takes `-c` like `run` does and defaults to the same configuration,
    so the lint claim covers the feature set the tests actually build —
    not just default features — and reuses that build's target dir. CI
    passes `-c llama-cpp-cpu` for the same reason the test job does:
    identical Rust surface, no CUDA C build on the clock.
    """
    config = resolve_config(args.config)
    features = config.feature_list(args.moeflux_model)
    return run_command(
        [
            "cargo",
            "clippy",
            "--all-targets",
            "--no-default-features",
            "--features",
            ",".join(features),
            "--",
            "-D",
            "warnings",
        ],
        config.target_dir(),
        log_path(f"clippy-{config.name}"),
    )


def cmd_docsrs(_: argparse.Namespace) -> int:
    """Rehearse the docs.rs build: nightly, scrape-examples, their features.

    docs.rs builds this crate with the feature list and cargo-args from
    `[package.metadata.docs.rs]` — including `-Zrustdoc-scrape-examples`,
    a nightly-only pass that compiles every example's `//!` docs. Nothing
    else here runs that pass: `just doc` is stable and never scrapes, so
    a broken intra-doc link in an example header compiles in nobody's
    gate and first fails in public, on docs.rs, after publish. Three did
    exactly that in the 0.8.0 pre-release sweep.

    The feature list is read from the manifest rather than restated, so
    this rehearsal cannot drift from what docs.rs will actually do. One
    deliberate difference: `-D warnings`. docs.rs would "succeed" and
    render the broken link; here that is the failure being tested for.

    NO `--no-default-features`: docs.rs keeps the default features and
    adds the metadata list on top, so the rehearsal must too.
    """
    import tomllib

    manifest = tomllib.loads((REPO / "Cargo.toml").read_text("utf-8"))
    meta = manifest["package"]["metadata"]["docs"]["rs"]
    if not DRY_RUN:
        nightly = subprocess.run(
            ["cargo", f"+{NIGHTLY}", "--version"], capture_output=True
        )
        if nightly.returncode != 0:
            sys.exit(
                f"{NIGHTLY} toolchain not found — scrape-examples is "
                "nightly-only (docs.rs builds on nightly). "
                f"`rustup toolchain install {NIGHTLY}`"
            )
    # The cpu config's target dir, not the default one: the docs.rs
    # feature set has no `cuda`, and on Linux building llama-cpp-sys
    # without it in the shared dir would evict the CUDA C build
    # (a ~20-40 min rebuild). Matches the eviction logic in `Config`.
    return run_command(
        [
            "cargo",
            f"+{NIGHTLY}",
            "doc",
            "--no-deps",
            *meta["cargo-args"],
            "--features",
            ",".join(meta["features"]),
        ],
        CONFIGS["llama-cpp-cpu"].target_dir(),
        log_path("docsrs"),
        {"RUSTDOCFLAGS": "-D warnings"},
    )


def cmd_badge(args: argparse.Namespace) -> int:
    """Verify README.md's hand-counted test numbers against reality.

    The README carries a shields.io `tests-N` badge and a prose claim
    ("N tests across M binaries … U that run in seconds and I that load
    real weights"). Hand-maintained counts drift silently — this
    compares them against `cargo nextest list` for the same
    configuration the numbers describe, and `--fix` rewrites them.

    CI runs this after the test job (the test binaries are already
    built, so the list is nearly free), and the `just check` recipe
    runs it, which is what puts it in front of every commit — a
    deliberate reversal of the original "keep the hook fast" exclusion
    after a stale badge failed the 0.8.1 PR in CI.

    The counts are the same in every configuration, so the hook can
    verify them against whichever target dir is already warm rather
    than building a second one. That invariant is a house rule, not an
    accident: a backend-specific test puts its `cfg` on the **body**,
    never on `#[test]` (see `cuda_build_has_a_gpu_device`). Gate the
    attribute and the badge means one number under `llama-cpp` and
    another under `llama-cpp-cpu` — #82 went red exactly that way.
    """
    import json as _json
    import re as _re

    require_nextest()
    config = resolve_config(args.config)
    features = config.feature_list(args.moeflux_model)
    cmd = [
        "cargo",
        "nextest",
        "list",
        "--message-format",
        "json",
        "--no-default-features",
        "--features",
        ",".join(features),
    ]
    print(f"+ CARGO_TARGET_DIR={config.target_dir()}", flush=True)
    print(f"+ {' '.join(cmd)}", flush=True)
    if DRY_RUN:
        return 0
    proc = subprocess.run(
        cmd,
        cwd=REPO,
        env=dict(os.environ, CARGO_TARGET_DIR=str(config.target_dir())),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return proc.returncode

    suites = _json.loads(proc.stdout).get("rust-suites", {})
    # Count every suite, empty ones included — matching nextest's own
    # "Starting N tests across M binaries" line, which is where the
    # README's phrasing comes from.
    binaries = len(suites)
    total = ignored = 0
    for suite in suites.values():
        cases = suite.get("testcases", {})
        total += len(cases)
        ignored += sum(1 for c in cases.values() if c.get("ignored"))
    unignored = total - ignored

    readme = REPO / "README.md"
    text = readme.read_text(encoding="utf-8")
    patterns = {
        "badge": (r"tests-(\d+)-blue", (total,)),
        "total/binaries": (
            r"(\d+) tests across (\d+) binaries",
            (total, binaries),
        ),
        "unignored/ignored": (
            r"(\d+) that run in\s+seconds and (\d+) that load",
            (unignored, ignored),
        ),
    }
    stale = []
    for label, (pattern, want) in patterns.items():
        m = _re.search(pattern, text)
        if m is None:
            print(f"badge: README pattern for {label} not found", flush=True)
            stale.append(label)
        elif tuple(int(g) for g in m.groups()) != want:
            print(
                f"badge: {label} says {'/'.join(m.groups())}, "
                f"nextest says {'/'.join(str(w) for w in want)}",
                flush=True,
            )
            stale.append((label, pattern, want))

    if not stale:
        print(
            f"badge: ok ({total} tests, {binaries} binaries, "
            f"{unignored} + {ignored})",
            flush=True,
        )
        return 0
    if not args.fix:
        print(
            "badge: STALE — update README.md or run "
            "`scripts/test.py badge --fix`",
            file=sys.stderr,
        )
        return 1
    for entry in stale:
        if isinstance(entry, str):
            print(f"badge: cannot fix missing pattern for {entry}")
            return 1
        _, pattern, want = entry
        it = iter(want)

        def sub(m: "_re.Match[str]") -> str:
            s, out = m.group(0), []
            last = 0
            for g in range(1, (m.lastindex or 0) + 1):
                out.append(s[last : m.start(g) - m.start(0)])
                out.append(str(next(it)))
                last = m.end(g) - m.start(0)
            out.append(s[last:])
            return "".join(out)

        text = _re.sub(pattern, sub, text, count=1)
    readme.write_text(text, encoding="utf-8")
    print("badge: README.md updated", flush=True)
    return 0


def cmd_configs(_: argparse.Namespace) -> int:
    for name, config in CONFIGS.items():
        flags = []
        if config.macos_only:
            flags.append("macOS-only")
        if config.ci_eligible:
            flags.append("CI-eligible")
        suffix = f"  [{', '.join(flags)}]" if flags else ""
        print(f"{name}{suffix}")
        print(f"  features: {','.join(config.feature_list(DEFAULT_MOEFLUX_MODEL))}")
        for line in wrap(config.help, 68):
            print(f"  {line}")
        print()
    return 0


def wrap(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(" ".join(text.split()), width)


def sanitize(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "_.-") else "_" for c in s)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="test.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--moeflux-model",
            default=DEFAULT_MOEFLUX_MODEL,
            choices=MOEFLUX_MODELS,
            help="moeflux compile-time model variant (default: "
            "%(default)s, the fast one)",
        )
        p.add_argument(
            "-n",
            "--dry-run",
            action="store_true",
            help="print the cargo invocations without running them",
        )

    p_run = sub.add_parser("run", help="run a test tier in one configuration")
    p_run.add_argument(
        "-c",
        "--config",
        default="llama-cpp",
        help="feature configuration (default: %(default)s)",
    )
    p_run.add_argument(
        "-t",
        "--tier",
        default="unignored",
        choices=list(TIERS),
        help="which tests (default: %(default)s). `ignored` is the "
        "model-loading set; `all` is genuinely everything",
    )
    p_run.add_argument(
        "-f",
        "--filter",
        help="substring matched against test AND binary names; implies "
        "the `all` tier and uncaptured output",
    )
    p_run.add_argument(
        "-x",
        "--exclude",
        action="append",
        default=[],
        metavar="NAME",
        help="substring of a test OR binary name to leave out; repeatable "
        "and composes with --filter. For a machine with only some of the "
        "weights: the CI box has models/model.gguf but not the Gemma 4 or "
        "gpt-oss files, and those suites panic rather than skip",
    )
    add_common(p_run)
    p_run.set_defaults(func=cmd_run)

    p_cov = sub.add_parser(
        "coverage",
        help="instrumented run + coverage report (cargo-llvm-cov)",
        description=cmd_coverage.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Same selection flags as `run`, and shared code behind them: a
    # coverage number for a different set of tests than the ones that
    # gate a commit would be measuring something nobody runs.
    p_cov.add_argument(
        "-c",
        "--config",
        default="llama-cpp",
        help="feature configuration (default: %(default)s)",
    )
    p_cov.add_argument(
        "-t",
        "--tier",
        default="all",
        choices=list(TIERS),
        help="which tests (default: %(default)s). Unlike `run` this "
        "defaults to everything — the unignored tier alone reports the "
        "generation paths as dead code, which is the opposite of true",
    )
    p_cov.add_argument("-f", "--filter", help="see `run --filter`")
    p_cov.add_argument(
        "-x",
        "--exclude",
        action="append",
        default=[],
        metavar="NAME",
        help="see `run --exclude`",
    )
    p_cov.add_argument(
        "--lcov",
        nargs="?",
        const=COVERAGE_DIR / "lcov.info",
        type=Path,
        help="also write lcov (default path: %(const)s)",
    )
    p_cov.add_argument(
        "--json",
        nargs="?",
        const=COVERAGE_DIR / "summary.json",
        type=Path,
        help="also write the JSON summary, which is what a badge is cut "
        "from (default path: %(const)s)",
    )
    p_cov.add_argument(
        "--html",
        action="store_true",
        help=f"also write a browsable report to {COVERAGE_DIR / 'html'}",
    )
    p_cov.add_argument(
        "--open",
        action="store_true",
        help="open the HTML report in a browser (implies --html)",
    )
    p_cov.add_argument(
        "--doctests",
        action="store_true",
        help="also measure the doctests. Switches the WHOLE run to "
        "`cargo +nightly` — llvm-cov's doctest support is nightly-only, "
        "and profraw from two toolchains will not merge",
    )
    p_cov.add_argument(
        "--fail-under",
        type=float,
        metavar="PCT",
        help="exit nonzero if line coverage is below PCT",
    )
    add_common(p_cov)
    p_cov.set_defaults(func=cmd_coverage)

    p_check = sub.add_parser(
        "check", help="permutation gate: cargo check --all-targets"
    )
    p_check.add_argument(
        "-c",
        "--config",
        default="all",
        help="one configuration, or `all` (default: %(default)s)",
    )
    p_check.add_argument(
        "--ci",
        action="store_true",
        help="skip configurations that need a GPU, weights or a C toolchain",
    )
    add_common(p_check)
    p_check.set_defaults(func=cmd_check)

    p_doc = sub.add_parser(
        "doctest",
        help="run the doctests (nextest cannot — see the subcommand doc)",
        description=cmd_doctest.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_doc.add_argument(
        "-c",
        "--config",
        default="llama-cpp",
        help="one configuration, or `all` (default: %(default)s). The "
        "default is the fast local loop; CI sweeps `all`, because a "
        "doctest naming a feature-gated type is broken per-configuration "
        "and `check` cannot see it",
    )
    p_doc.add_argument(
        "--ci",
        action="store_true",
        help="skip configurations that need a GPU, weights or a C toolchain",
    )
    add_common(p_doc)
    p_doc.set_defaults(func=cmd_doctest)

    p_clippy = sub.add_parser(
        "clippy",
        help="clippy with -D warnings (the tree is warning-clean; keep it)",
        description=cmd_clippy.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_clippy.add_argument(
        "-c",
        "--config",
        default="llama-cpp",
        help="configuration to lint (default: %(default)s, same as "
        "`run`; CI uses llama-cpp-cpu)",
    )
    add_common(p_clippy)
    p_clippy.set_defaults(func=cmd_clippy)

    p_docsrs = sub.add_parser(
        "docsrs",
        help="rehearse the docs.rs build (nightly + scrape-examples)",
        description=cmd_docsrs.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common(p_docsrs)
    p_docsrs.set_defaults(func=cmd_docsrs)

    p_badge = sub.add_parser(
        "badge",
        help="verify README's test-count badge against `nextest list`",
        description=cmd_badge.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_badge.add_argument(
        "-c",
        "--config",
        default="llama-cpp",
        help="configuration the README numbers describe "
        "(default: %(default)s)",
    )
    p_badge.add_argument(
        "--fix",
        action="store_true",
        help="rewrite README.md's numbers instead of failing",
    )
    add_common(p_badge)
    p_badge.set_defaults(func=cmd_badge)

    p_configs = sub.add_parser("configs", help="list configurations")
    p_configs.set_defaults(func=cmd_configs)

    args = parser.parse_args()
    global DRY_RUN
    DRY_RUN = getattr(args, "dry_run", False)
    # `--open` with nothing to open is a silent no-op otherwise.
    if getattr(args, "open", False):
        args.html = True
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
