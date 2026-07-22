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
    scripts/test.py check --config all        # the permutation gate
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
MOEFLUX_MODELS = [
    "qwen3-6-35b-a3b",
    "qwen3-5-a17b",
    "cogito-v2-671b",
]

# Features that name no backend: they compile against the trait layer and
# are legal in every configuration. `cli` and `axum` belong here as of
# #68 — both used to imply `llama-cpp`, which made a moeflux-only
# front-end impossible to build even though the library supported it.
AGNOSTIC = ["toml", "serde", "stats", "json-schema", "egui"]

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

# tier -> extra nextest arguments.
#
#   unignored  the fast loop: no model is loaded, so it stays parallel.
#   ignored    ONLY the #[ignore]'d tests. Each loads a 13-19 GB model
#              onto the GPU and only one fits, hence profile `full`
#              (test-threads = 1). This is what `just test full` meant,
#              under a name that says so.
#   all        genuinely everything, serialized for the same reason.
TIERS: dict[str, list[str]] = {
    "unignored": [],
    "ignored": ["--profile", "full", "--run-ignored", "only"],
    "all": ["--profile", "full", "--run-ignored", "all"],
}


def log_path(name: str) -> Path:
    d = REPO / "target" / "test-logs"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.log"


def run_command(cmd: list[str], target_dir: Path, log: Path | None) -> int:
    """Run `cmd`, streaming to stdout and (optionally) tee'ing to `log`.

    Tee'd rather than redirected so a failure can be read back after the
    fact instead of re-run to be seen — and streamed rather than
    captured so a long model test shows progress instead of going quiet
    for ten minutes.
    """
    env = dict(os.environ, CARGO_TARGET_DIR=str(target_dir))
    print(f"+ CARGO_TARGET_DIR={target_dir}", flush=True)
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


def cmd_run(args: argparse.Namespace) -> int:
    if not DRY_RUN:
        require_nextest()
    config = resolve_config(args.config)
    features = config.feature_list(args.moeflux_model)

    cmd = [
        "cargo",
        "nextest",
        "run",
        "--no-default-features",
        "--features",
        ",".join(features),
        "--no-fail-fast",
    ]

    if args.filter:
        # A named test is asked for by name, so run it whichever list it
        # is on rather than making the caller remember — and uncaptured,
        # so the suites' block/emission dumps are visible on a pass and
        # not only on a failure.
        cmd += TIERS["all"] + ["--no-capture"]
        cmd += ["-E", f"test(~{args.filter}) + binary(~{args.filter})"]
        name = f"{config.name}-{sanitize(args.filter)}"
    else:
        cmd += TIERS[args.tier]
        name = f"{config.name}-{args.tier}"

    return run_command(cmd, config.target_dir(), log_path(name))


def cmd_check(args: argparse.Namespace) -> int:
    """The permutation gate: every configuration compiles, tests included.

    `--all-targets` is the point — the library building is not the same
    claim as the test targets building, and #68 was filed because a
    configuration that compiled as a library had ~38 broken test
    targets nobody had ever built.
    """
    if args.config == "all":
        names = list(CONFIGS)
    else:
        names = [args.config]

    failures: list[str] = []
    # Two configs can resolve to the same (features, target dir) — notably
    # llama-cpp and llama-cpp-cpu off Linux, where there is no CUDA to tell
    # them apart. Checking the second is a cache hit rather than a cost, but
    # reporting it as a distinct permutation would overstate the coverage.
    seen: dict[tuple[str, str], str] = {}
    for name in names:
        config = CONFIGS[name]
        if config.macos_only and platform.system() != "Darwin":
            print(f"= skip {name}: macOS-only", flush=True)
            continue
        if args.ci and not config.ci_eligible:
            print(f"= skip {name}: not CI-eligible", flush=True)
            continue
        features = config.feature_list(args.moeflux_model)
        key = (",".join(features), str(config.target_dir()))
        if key in seen:
            print(
                f"= skip {name}: identical to {seen[key]} on this platform",
                flush=True,
            )
            continue
        seen[key] = name
        cmd = [
            "cargo",
            "check",
            "--all-targets",
            "--no-default-features",
            "--features",
            ",".join(features),
        ]
        print(f"\n=== check: {name} ===", flush=True)
        code = run_command(
            cmd, config.target_dir(), log_path(f"check-{name}")
        )
        if code != 0:
            failures.append(name)

    if failures:
        print(f"\ncheck: FAILED for {', '.join(failures)}", file=sys.stderr)
        return 1
    print("\ncheck: ok", flush=True)
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
    add_common(p_run)
    p_run.set_defaults(func=cmd_run)

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

    p_configs = sub.add_parser("configs", help="list configurations")
    p_configs.set_defaults(func=cmd_configs)

    args = parser.parse_args()
    global DRY_RUN
    DRY_RUN = getattr(args, "dry_run", False)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
