#!/usr/bin/env python3
"""Derive CI cargo test package sets from the Cargo workspace."""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
WORKFLOWS = ROOT / ".github" / "workflows"

# The status checks the `main` branch ruleset marks required. A job outside this
# set can go red without blocking a merge, so a package tested only there is,
# for merge purposes, untested: #1982 left `shape_dispatch_gate` failing on
# `main` with both required checks green, because the only lane that ran it was
# advisory. See #2015.
#
# This mirrors a repository setting that a workflow file cannot read, so it is
# maintained by hand. `verify_required_tier` asserts every name here still
# matches a job in the workflow, which catches the rename direction. A ruleset
# edit that *drops* a required check is not observable from inside the repo and
# is the one drift this gate cannot see.
REQUIRED_JOB_NAMES = frozenset({"Fast (Linux x86_64)", "Rust quality"})

# Every job this repository's workflows define, by file. `_stray_job_attribute`
# refuses a job key indented one level too far, but it is reached through
# `workflow_jobs()`, which reads `ci.yml` alone; the other 23 files were
# unguarded until #2206. Scope and host are separate choices, and only the host
# has to be required: this inventory is checked from `rust-quality`, so a job
# that vanishes anywhere blocks a merge everywhere.
#
# The names are display names -- `name:` when a job sets one, otherwise the job
# key -- because that is what GitHub reports as the status check and what a
# branch ruleset matches on.
#
# A stray-key rule fires on the *mechanism* of accidental nesting, so it cannot
# see a job deleted outright: the file simply has fewer jobs, and nothing about
# it is malformed. That is why this is an inventory and not another rule. The
# cost is that adding or renaming a job means editing this table, and the
# failure message prints the exact replacement block to paste.
WORKFLOW_JOB_INVENTORY: dict[str, tuple[str, ...]] = {
    "audit.yml": (
        "audit",
    ),
    "benchmark.yml": (
        "Kernel micro-benchmarks",
    ),
    "ci.yml": (
        "CLI ORT (${{ matrix.name }})",
        "CUDA compile (Linux x86_64)",
        "CUDA compile (Windows x86_64)",
        "Detect change scope",
        "EP conformance (Linux x86_64)",
        "Fast (Linux x86_64)",
        "Rust (Windows ARM64)",
        "Rust coverage (${{ matrix.name }})",
        "Rust quality",
    ),
    "diff-guard.yml": (
        "Deletion ratio",
        "Root file allowlist",
    ),
    "hostlock.yml": (
        "Host lock conformance",
    ),
    "miri.yml": (
        "Miri unsafe-crate soundness",
    ),
    "mobius-producer-conformance.yml": (
        "Mobius metadata packages (signal)",
    ),
    "publish-ep-plugins.yml": (
        "Publish nxrt-ep-cpu",
        "Publish nxrt-ep-cuda",
        "nxrt-ep-cpu wheel (${{ matrix.name }})",
        "nxrt-ep-cuda wheel (${{ matrix.name }})",
    ),
    "publish.yml": (
        "Build ${{ matrix.package }} (${{ matrix.platform.name }})",
        "Publish nxrt sdist to PyPI",
        "Publish onnx-genai sdists to PyPI",
        "Publish onnx-genai wheels to PyPI",
        "publish",
    ),
    "squad-ci.yml": (
        "test",
    ),
    "squad-docs.yml": (
        "build",
    ),
    "squad-heartbeat.yml": (
        "heartbeat",
    ),
    "squad-insider-release.yml": (
        "release",
    ),
    "squad-issue-assign.yml": (
        "assign-work",
    ),
    "squad-label-enforce.yml": (
        "enforce",
    ),
    "squad-preview.yml": (
        "validate",
    ),
    "squad-promote.yml": (
        "Promote dev → preview",
        "Promote preview → main (release)",
    ),
    "squad-release.yml": (
        "release",
    ),
    "squad-triage.yml": (
        "triage",
    ),
    "sync-squad-labels.yml": (
        "sync-labels",
    ),
    "visualizer-test.yml": (
        "Node security and leaf-accounting tests",
    ),
    "weight-cache-guard.yml": (
        "A per-thread buffer needs a per-process bound",
        "New weight-derived caches must be governed",
    ),
    "wheels.yml": (
        "CPU wheel (${{ matrix.name }})",
        "CUDA wheel scaffold",
        "Publish nxrt CPU wheels",
    ),
    "wiki-lint.yml": (
        "Notes stand on their own",
    ),
}

# Crates that are intentionally not selected by any CI cargo-test lane.
# Every entry is a written exception to the default rule: workspace members are
# tested unless they are listed here with a reason.
DENYLIST: dict[str, str] = {
    # Build script downloads a native ONNX Runtime distribution; this is why CI
    # must never use bare `cargo test --workspace`.
    "onnx-genai-ort-sys": "build script downloads a native ONNX Runtime distribution",
    # Benchmark crate includes CUDA/native performance entry points; benchmark
    # execution belongs in explicit perf/GPU jobs, not every PR test lane.
    "onnx-genai-bench": "benchmark/perf crate; not a unit-test CI target",
    # CUDA EP tests require a CUDA device. Compile coverage is handled by the
    # CUDA compile job; runtime GPU honesty is checked separately.
    "onnx-runtime-ep-cuda": "runtime tests require a CUDA GPU",
    # PyO3 extension crates need wheel/extension-module packaging so the test
    # binary can find the generated module and runtime DLLs. Plain cargo test on
    # CI is not the right harness.
    "onnx-genai-python": "PyO3 extension crate requires wheel packaging/runtime DLL staging",
    "onnx-runtime-python": "PyO3 extension crate requires wheel packaging/runtime DLL staging",
}

# Packages that need the ORT-backed lane because they directly or transitively
# load ONNX Runtime. They are still tested by default; they are only kept out of
# the offline lanes to preserve the ort-sys download constraint above.
ORT_BACKED = frozenset(
    {
        "onnx-genai",
        "onnx-genai-capi",
        "onnx-genai-cli",
        "onnx-genai-engine",
        "onnx-genai-ort",
        "onnx-genai-server",
    }
)

# MLAS is Linux-tested for coverage and separately built on Windows ARM64; keep
# it out of the cross-platform coverage/ARM test lanes that intentionally avoid
# native MLAS execution there.
LINUX_ONLY = frozenset({"mlas-sys"})


def workspace_packages() -> set[str]:
    metadata = json.loads(
        subprocess.check_output(
            ["cargo", "metadata", "--no-deps", "--format-version", "1"],
            cwd=ROOT,
            text=True,
        )
    )
    return {package["name"] for package in metadata["packages"]}


def package_args(packages: list[str]) -> str:
    # One argument per line makes command substitution work in both bash and
    # PowerShell. A single space-separated line becomes one native argument in
    # PowerShell, so `cargo` sees an invalid package name containing spaces.
    # Callers must receive LF-only separators; see the newline handling in
    # `main`, which bash depends on and PowerShell hides.
    return "\n".join(arg for package in packages for arg in ("-p", package))


def lane_packages(lane: str) -> list[str]:
    packages = workspace_packages()
    denied = set(DENYLIST)
    if unknown := sorted(denied - packages):
        raise SystemExit(f"workspace test deny-list contains non-workspace package(s): {unknown}")

    offline = packages - denied - ORT_BACKED
    match lane:
        case "offline-linux":
            selected = offline
        case "offline-cross-platform":
            selected = offline - LINUX_ONLY
        case "ort-backed":
            selected = ORT_BACKED & packages
        case "linux-only":
            selected = LINUX_ONLY & packages
        case "lint":
            # Every package any test lane compiles. `cargo clippy` only ever
            # *checks*, so the ort-sys/CUDA constraint that splits the test
            # lanes does not apply to it: nothing here is linked or run.
            selected = packages - denied
        case _:
            raise SystemExit(f"unknown lane '{lane}'")
    return sorted(selected)


LANES = ("offline-linux", "offline-cross-platform", "ort-backed", "linux-only", "lint")

# `cargo clippy ...` spelled across a folded YAML scalar. A step ends at a blank
# line or at the next `- name:`/key at or below the invocation's indentation.
_CLIPPY = re.compile(r"^(?P<indent>\s*)(?:run:\s*)?.*\bcargo clippy\b")
# A second cargo command inside the same block. `clippy` itself is excluded so a
# continuation line of the invocation being read does not end it.
_CARGO_AGAIN = re.compile(r"(?:^|[\s;&|])cargo\s+(?:\+\S+\s+)?(?!clippy\b)\S+")
_GENERATOR = re.compile(
    r"\$\(\s*python3?\s+\.github/scripts/workspace_test_packages\.py\s+cargo-args\s+([a-z-]+)\s*\)"
)
# The same call, bound to a shell variable. The guarded form these workflows use
# moves the lane name off the `cargo clippy` line, so the block below has to
# read backwards to find it.
_GENERATOR_ASSIGN = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*)=\"?\$\(\s*python3?\s+"
    r"\.github/scripts/workspace_test_packages\.py\s+cargo-args\s+[a-z-]+\s*\)"
)
# Any assignment to a shell name. Needed so a later non-generator assignment
# shadows an earlier generator one, which is the property `packages_tested_by`
# already documents; recording only generator assignments would let
# `packages="$(... lint)"` followed by `packages="-p one-crate"` keep crediting
# the whole lint lane.
_ANY_ASSIGN = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=")
# A YAML key or list item, at the indentation that ends a step. The forward
# scan in `clippy_blocks` already refuses to cross one of these; the backwards
# walk has to refuse the same boundary or it reads into the step above.
_STEP_BOUNDARY = re.compile(r"\s*(-\s+\w+:|\w[\w-]*:)")


def _generator_vars_before(lines: list[str], index: int, base: int) -> dict[str, str]:
    """Generator assignments earlier in the same `run:` block, nearest first.

    A step that wants to refuse an empty expansion has to bind the generator to
    a variable first, because `cargo clippy $(...)` runs happily on nothing:
    the substitution fails, the outer command still exits 0, and the step lints
    the default members instead of the lane. Binding it moves the lane name off
    the invocation line and out of the text `clippy_blocks` returns.

    The read is deliberately narrow in both directions. It stops at a blank
    line, at a line indented less than the invocation, and at a YAML key or
    list item at or below that indentation -- the same boundary the forward
    scan refuses to cross, because a `- run:` step carrying no `name:` sits at
    the invocation's own indentation and `indent < base` alone would walk
    straight through it into the step above. `clippy_blocks` then appends what
    it finds only when the invocation actually references the variable.
    Crediting every assignment in scope would be a join rather than a dataflow
    read, and this scanner's whole documented risk is over-reading.
    """
    assigned: dict[str, str] = {}
    shadowed: set[str] = set()
    # If the invocation line is itself the list item or carries the `run:` key,
    # its block starts there and has no earlier lines. Walking anyway would
    # cross into the step above, whose body is indented *deeper* than a `- run:`
    # line -- so the indent and boundary tests below never fire and the whole
    # previous step is read as if it were this one's.
    opening = lines[index].lstrip()
    if opening.startswith("- ") or re.match(r"(-\s+)?[\w-]+:", opening):
        return assigned
    for position in range(index - 1, -1, -1):
        line = lines[position]
        if not line.strip():
            break
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if indent < base:
            break
        if indent <= base and _STEP_BOUNDARY.match(stripped):
            break
        assignment = _ANY_ASSIGN.match(stripped)
        if not assignment:
            continue
        # Walking backwards, the first assignment seen is the last one that
        # ran, so a plain assignment here shadows a generator assignment above
        # it and the lane must not be credited.
        name = assignment.group(1)
        if name in assigned or name in shadowed:
            continue
        if _GENERATOR_ASSIGN.match(stripped):
            assigned[name] = line
        else:
            shadowed.add(name)
    return assigned


def clippy_blocks(lines: list[str]) -> list[str]:
    """The `cargo clippy` invocations in one workflow's lines, as raw text.

    Three things end a block, and each one is a false-green vector if it is
    left out. Every `-p` token this returns is credited as lint coverage, so
    the whole risk lives in over-reading:

    * a blank line or a key at or below the invocation's indentation -- the
      step is over, and reading on would credit the *next* step's packages;
    * a comment, wherever it sits. A de-indented one ends the step; a
      deeper-indented one is skipped rather than appended. This function's
      first version only did the former, so `# -p ghost` indented inside a
      folded scalar was harvested as coverage -- prose granting a lint claim,
      which is exactly what the comment handling exists to prevent;
    * a second `cargo` command in the same `run:` block. `cargo clippy ... &&
      cargo test -p x` or a `run: |` running both would otherwise credit `x`
      to clippy on the strength of a step that only *tests* it.

    All three fail quietly if missed: the package looks linted and nothing
    says otherwise. Hence `clippy_block_arms` in the self-test.
    """
    found: list[str] = []
    for index, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            continue
        match = _CLIPPY.match(line)
        if not match:
            continue
        base = len(match.group("indent"))
        block = [line]
        for follow in lines[index + 1 :]:
            if not follow.strip():
                break
            stripped = follow.lstrip()
            indent = len(follow) - len(stripped)
            if indent <= base and (
                stripped.startswith("#") or re.match(r"\s*(-\s+\w+:|\w[\w-]*:)", follow)
            ):
                break
            if stripped.startswith("#"):
                continue
            block.append(follow)
        text = "\n".join(block)
        # Truncate at the first non-clippy `cargo`, wherever it falls -- the
        # same line (`cargo clippy ... && cargo test -p x`) or a later one in a
        # `run: |` block. Doing this line-wise instead misses the `&&` form,
        # which is how the first version of this cut credited `x` to clippy.
        after = _CARGO_AGAIN.search(text, match.end())
        if after:
            text = text[: after.start()]
        # Recover a package list the invocation reads from a variable, but only
        # the ones it actually names. The trailing boundary matters: without it
        # `$packages` matches inside `$packages_extra` and credits a lane the
        # invocation never used. See `_generator_vars_before`.
        for variable, source in _generator_vars_before(lines, index, base).items():
            if re.search(r"\$\{?" + re.escape(variable) + r"\}?(?![A-Za-z0-9_])", text):
                text = f"{text}\n{source}"
        found.append(text)
    return found


def clippy_commands() -> list[tuple[Path, str]]:
    """Every `cargo clippy` invocation in `.github/workflows`, as raw text."""
    found: list[tuple[Path, str]] = []
    for workflow in sorted(WORKFLOWS.glob("*.yml")):
        for block in clippy_blocks(workflow.read_text().splitlines()):
            found.append((workflow, block))
    return found


def linted_packages() -> set[str]:
    """Packages reached by some clippy invocation, generators expanded.

    Generator calls are *expanded*, not skipped: a `-p` list that is computed
    counts exactly as much as one that is spelled out, and a list that is
    spelled out cannot hide behind the fact that some other step computes one.
    """
    linted: set[str] = set()
    for _, command in clippy_commands():
        for lane in _GENERATOR.findall(command):
            if lane in LANES:
                linted.update(lane_packages(lane))
        linted.update(re.findall(r"-p\s+([A-Za-z0-9_-]+)", command))
    return linted


def verify(simulate_missing: str | None = None, simulate_unlinted: str | None = None) -> int:
    packages = workspace_packages()
    tested = (
        set(lane_packages("offline-linux"))
        | set(lane_packages("offline-cross-platform"))
        | set(lane_packages("ort-backed"))
        | set(lane_packages("linux-only"))
    )
    if simulate_missing:
        tested.discard(simulate_missing)
    denied = set(DENYLIST)
    uncovered = sorted(packages - tested - denied)
    stale_tested = sorted(tested - packages)
    stale_denied = sorted(denied - packages)

    commands = clippy_commands()
    linted = linted_packages() & packages
    if simulate_unlinted:
        linted.discard(simulate_unlinted)
    unlinted = sorted(tested - linted)

    failed = False
    if uncovered or stale_tested or stale_denied:
        failed = True
        print("Workspace test package coverage check failed.", file=sys.stderr)
        if uncovered:
            print(
                "Workspace member(s) are in neither a CI test lane nor the deny-list:",
                file=sys.stderr,
            )
            for package in uncovered:
                print(f"  - {package}", file=sys.stderr)
            print(
                "Add each package to a derived test lane, or add a DENYLIST entry with a reason.",
                file=sys.stderr,
            )
        if stale_tested:
            print(f"Non-workspace package(s) in tested lanes: {stale_tested}", file=sys.stderr)
        if stale_denied:
            print(f"Non-workspace package(s) in deny-list: {stale_denied}", file=sys.stderr)

    # Positive control: the lint half is computed by scanning workflow text, so
    # a scanner that silently matches nothing would report full coverage of an
    # empty set. Finding no clippy at all means the scanner broke, not that the
    # repo stopped linting.
    if not commands:
        failed = True
        print(
            f"No `cargo clippy` invocation found under {WORKFLOWS}; the lint-coverage "
            "scanner is broken or the workflows moved.",
            file=sys.stderr,
        )
    if unlinted:
        failed = True
        print("Workspace lint coverage check failed.", file=sys.stderr)
        print(
            "Package(s) are compiled and tested by CI and linted by nothing:",
            file=sys.stderr,
        )
        for package in unlinted:
            print(f"  - {package}", file=sys.stderr)
        print(
            "Every tested package must be reached by some `cargo clippy` step. The Linux\n"
            "offline clippy steps select `cargo-args lint`, so a package added to a test\n"
            "lane is linted automatically -- this failing means a clippy step went back to\n"
            "a hand-written -p list, or a new lane is tested but not linted.",
            file=sys.stderr,
        )
    if failed:
        return 1
    print(
        f"workspace test package coverage ok: {len(tested)} tested, {len(denied)} denied"
    )
    print(
        f"workspace lint coverage ok: {len(tested)} tested, all linted by "
        f"{len(commands)} clippy invocation(s)"
    )
    return 0


# A `cargo test`/`cargo llvm-cov` invocation is the only thing that counts as
# executing a package's tests. `cargo build` and `cargo clippy --all-targets`
# compile the same test targets but never run an assertion, so they must not
# satisfy this gate.
_RUNS_TESTS = re.compile(r"cargo\s+(?:\+\S+\s+)?(?:test|llvm-cov)\b")
_LANE_CALL = re.compile(r"workspace_test_packages\.py\s+cargo-args\s+([a-z-]+)")
_PACKAGE_FLAG = re.compile(r"(?:^|\s)(?:-p|--package)[\s=]+([A-Za-z0-9_-]+)")
_VAR_ASSIGN = re.compile(
    r"(?:^|\s)(?P<var>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>\"[^\"]*\"|'[^']*'|\S*)"
)
_VAR_REF = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?")


# `ubuntu-latest` does not ship PyYAML in the interpreter `python` resolves to,
# and PEP 668 blocks installing it globally, so this gate parses the workflow
# itself. Only the two block styles the file uses are handled, and anything
# unrecognised is reported rather than assumed: an under-read here makes the
# gate stricter (packages look unenforced and it fails loudly), so the direction
# that must be guarded is over-reading, which is what job-slice attribution and
# comment stripping are for.
# A job key may be quoted and may carry a trailing comment; both are valid YAML.
# Missing one silently folds that job's steps into the *previous* job, and since
# `rust-coverage` follows required `rust-quality`, that direction manufactures a
# false green. Anything at job-key depth that does not parse here is therefore a
# hard error rather than body text -- see `_at_job_depth`.
_JOB_KEY = re.compile(
    r"""^\ \ (?:"(?P<dq>[^"]+)"|'(?P<sq>[^']+)'|(?P<plain>[A-Za-z0-9_.-]+))\s*:\s*(?:\#.*)?$"""
)
_JOB_NAME = re.compile(r"^    name:\s*(.+?)\s*$")
_TRAILING_COMMENT = re.compile(r"\s+\#.*$")


def _at_job_depth(line: str) -> bool:
    """True for a line indented exactly two spaces, i.e. a direct child of `jobs:`."""
    return len(line) > 2 and line.startswith("  ") and not line[2].isspace()


# Every key GitHub accepts directly under a job. A four-space key outside this
# set is the signature of a job whose own key was indented one level too far:
# still valid YAML, so no parser complains, but the job ceases to exist and its
# checks stop *appearing* on a PR instead of going red. @holden hit exactly this
# and `gh pr checks` reported fourteen green with zero failures while two checks
# had silently ceased to be produced. Enumerating what is legal and rejecting
# the rest is the same choice as `_at_job_depth`: a new job attribute breaks
# this loudly, which is the direction that can be noticed.
_JOB_ATTRIBUTES = frozenset(
    {
        "concurrency",
        "container",
        "continue-on-error",
        "defaults",
        "env",
        "environment",
        "if",
        "name",
        "needs",
        "outputs",
        "permissions",
        "runs-on",
        "secrets",
        "services",
        "steps",
        "strategy",
        "timeout-minutes",
        "uses",
        "with",
    }
)
_JOB_ATTR_KEY = re.compile(
    r"""^\ \ \ \ (?:"(?P<dq>[^"]+)"|'(?P<sq>[^']+)'|(?P<plain>[A-Za-z0-9_.-]+))\s*:"""
)


def _stray_job_attribute(line: str) -> str | None:
    """A four-space key that GitHub would not accept as a job attribute."""
    if line.lstrip().startswith("#"):
        return None
    match = _JOB_ATTR_KEY.match(line)
    if match is None:
        return None
    key = match.group("dq") or match.group("sq") or match.group("plain")
    return None if key in _JOB_ATTRIBUTES else key


def _scalar(value: str) -> str:
    """A YAML scalar as written in this workflow: optional quotes, optional comment."""
    value = _TRAILING_COMMENT.sub("", value).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return value
_RUN_KEY = re.compile(r"^(\s*)(- )?run:\s*(.*)$")
_FOLDED = {">", ">-", ">+"}
_LITERAL = {"|", "|-", "|+"}


def workflow_jobs() -> dict[str, str]:
    """Map each job's display name to its YAML body with comments removed."""
    return parse_jobs(WORKFLOW.read_text(encoding="utf-8"), source=str(WORKFLOW))


def parse_jobs(text: str, source: str = "<workflow>") -> dict[str, str]:
    """`workflow_jobs` over arbitrary text, so the parser itself is testable."""
    lines = text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.rstrip() == "jobs:")
    except StopIteration:
        raise SystemExit(f"{source}: no top-level `jobs:` block to check")

    jobs: dict[str, str] = {}
    key: str | None = None
    body: list[str] = []

    def flush() -> None:
        nonlocal key, body
        if key is None:
            return
        name = key
        for line in body:
            if match := _JOB_NAME.match(line):
                name = _scalar(match.group(1))
                break
        if name in jobs:
            raise SystemExit(f"{source}: two jobs are both named {name!r}")
        jobs[name] = "\n".join(
            line for line in body if not line.lstrip().startswith("#")
        )
        key, body = None, []

    for line in lines[start + 1 :]:
        if line.strip() and not line.startswith(" ") and not line.startswith("#"):
            break
        if _at_job_depth(line) and not line.lstrip().startswith("#"):
            match = _JOB_KEY.match(line)
            if match is None:
                raise SystemExit(
                    f"{source}: line at job-key depth is not a job key this gate "
                    f"can parse: {line!r}. Refusing to guess -- absorbing it into the "
                    f"previous job would credit its steps to that job."
                )
            flush()
            key = match.group("dq") or match.group("sq") or match.group("plain")
            continue
        if key is not None:
            if stray := _stray_job_attribute(line):
                raise SystemExit(
                    f"{source}: {key!r} contains {stray!r}, which is not a job "
                    f"attribute GitHub accepts. A job key indented one level too "
                    f"far is valid YAML and silently becomes a key of the job "
                    f"above it; GitHub then reports that job's checks as absent "
                    f"rather than failing, which no pass/fail gate can see."
                )
            body.append(line)
    flush()
    if not jobs:
        raise SystemExit(f"{source}: parsed no jobs; the gate would check nothing")
    return jobs


# A step's `if:` decides whether it runs at all, and this gate cannot evaluate
# GitHub's expression language. Crediting a guarded step would let `if: false`
# -- or a copy-pasted `if: runner.os == 'Windows'` on a Linux-only job -- remove
# a package's only required executor while the gate still reported it covered.
# So only steps that are guaranteed to have run in a *successful* job are
# counted: no `if:` (which defaults to `success()`), or an expression on this
# list. Anything else is not credited, which can only make the gate stricter and
# it names the step when it refuses.
_STEP_ITEM = re.compile(r"^\s*- ")
_STEP_IF = re.compile(r"^\s*if:\s*(.+?)\s*$")
# `continue-on-error: true` lets the step (or job) fail without failing the
# check, so its tests run but cannot block a merge -- which is the only property
# this gate cares about. Crediting it would satisfy the gate with coverage that
# is decorative.
_STEP_COE = re.compile(r"^\s*continue-on-error:\s*(.+?)\s*$")
_JOB_COE = re.compile(r"^    continue-on-error:\s*(.+?)\s*$", re.MULTILINE)


# A step this gate credits only runs if its *job* runs. Both required jobs carry
# a job-level `if:`, and a skipped required check SATISFIES a ruleset -- so the
# whole tier can be waved through by never executing. @holden demonstrated the
# live route with real builds (#2077): two `.md` files under `docs/` are pulled
# into Rust by `include_str!`, so a pure-markdown edit is classified docs-only,
# both required jobs skip, and the tree does not compile.
#
# The classifier's correctness is #2081's problem, not this gate's. What IS this
# gate's problem is that it models step-level `if:` (see `unconditional_run_blocks`)
# and would credit a step in a job whose own condition had been quietly narrowed.
# So: the required jobs may carry the repo's known docs-only guard or no guard at
# all, and anything else is refused rather than interpreted.
_JOB_IF = re.compile(r"^    if:\s*(.+?)\s*$", re.MULTILINE)
_ALLOWED_JOB_IF = frozenset({"needs.changes.outputs.docs_only != 'true'"})


def job_condition(job_body: str) -> str | None:
    """The job-level `if:`, or None. Step-level `if:` is indented deeper."""
    match = _JOB_IF.search(job_body)
    return match.group(1) if match else None


def verify_required_job_conditions(jobs: dict[str, str]) -> int:
    """Refuse a required job whose own `if:` is not one this gate has reasoned about."""
    problems = []
    # Iterating the intersection alone would return 0 for a required job that is
    # simply absent -- the same "unread value degrades to the most permissive
    # verdict" shape this gate exists to refuse. `verify_required_tier` already
    # rejects a missing required job earlier, so this is defence in depth: it
    # keeps the guarantee inside the function rather than in the caller's order.
    for name in sorted(REQUIRED_JOB_NAMES - set(jobs)):
        problems.append((name, "<required job absent from the workflow>"))
    for name in sorted(REQUIRED_JOB_NAMES & set(jobs)):
        condition = job_condition(jobs[name])
        if condition is not None and condition not in _ALLOWED_JOB_IF:
            problems.append((name, condition))
    if not problems:
        return 0
    print("Required-lane coverage check failed.", file=sys.stderr)
    print(
        "A required job carries a job-level `if:` this gate has not reasoned about.\n"
        "A skipped required check satisfies the ruleset, so narrowing this condition\n"
        "silently voids every guarantee below it -- the steps still exist and still\n"
        "never run.",
        file=sys.stderr,
    )
    for name, condition in problems:
        print(f"  - {name}: if: {condition}", file=sys.stderr)
    print(
        f"Known-good conditions: {sorted(_ALLOWED_JOB_IF)} (or no `if:` at all).",
        file=sys.stderr,
    )
    return 1
_FALSEY = {"false", "${{ false }}", "'false'", '"false"'}
_STEP_SHELL = re.compile(r"^\s*shell:\s*(.+?)\s*$")
# A command whose failure is swallowed runs its tests and reports success
# anyway. `||` does this unconditionally. A pipe does it only without pipefail,
# which is the difference between GitHub's default Linux shell (`bash -e`) and
# an explicit `shell: bash` (`bash --noprofile --norc -eo pipefail`).
_TEST_CALL = r"cargo\s+(?:\+\S+\s+)?(?:test|llvm-cov)\b[^\n;&|]*"
_OR_AFTER_TEST = re.compile(_TEST_CALL + r"\|\|")
_PIPE_AFTER_TEST = re.compile(_TEST_CALL + r"\|(?!\|)")


def _swallow_reason(block: str, pipefail: bool) -> str | None:
    if _OR_AFTER_TEST.search(block):
        return "`||` after a cargo test swallows its failure"
    if not pipefail and _PIPE_AFTER_TEST.search(block):
        return "cargo test piped without pipefail; its failure is masked"
    return None
_RUNS_IN_A_PASSING_JOB = {"success()", "always()", "true", "${{ true }}", "${{ success() }}"}


def job_steps(job_body: str) -> list[tuple[str | None, str, str | None]]:
    """Each step as (reason it must not be credited, its YAML text, its shell)."""
    steps: list[tuple[str | None, str, str | None]] = []
    current: list[str] = []
    for line in job_body.splitlines():
        if _STEP_ITEM.match(line):
            if current:
                steps.append(_step_entry(current))
            current = [line]
        elif current:
            current.append(line)
    if current:
        steps.append(_step_entry(current))
    return steps


def _step_entry(lines: list[str]) -> tuple[str | None, str, str | None]:
    """(reason this step must not be credited, step text, its `shell:`)."""
    first = lines[0]
    dash_indent = len(first) - len(first.lstrip())
    # The step's own keys sit at the SHALLOWEST column under the dash -- not at
    # `dash + 2`. YAML allows any run of spaces after the dash (`-   name:`),
    # which puts the real keys deeper than dash+2; an equality test then reads
    # `if:` as shell text and credits a guarded step. Taking the minimum also
    # survives `- run: |` leading the step, where the first following line is
    # shell text deeper than the sibling keys.
    deeper = [
        len(line) - len(line.lstrip())
        for line in lines[1:]
        if line.strip() and len(line) - len(line.lstrip()) > dash_indent
    ]
    body_indent = min(deeper) if deeper else dash_indent + 2
    keys: dict[str, str] = {}
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        if index == 0:
            stripped = first.lstrip()
            text = stripped[1:].lstrip() if stripped.startswith("- ") else ""
        elif len(line) - len(line.lstrip()) == body_indent:
            # A key at the step's own depth. Anything deeper is shell text
            # inside a run block and must not be read as a step key.
            text = line.lstrip()
        else:
            continue
        if match := _STEP_IF.match(text):
            keys.setdefault("if", match.group(1))
        elif match := _STEP_COE.match(text):
            keys.setdefault("continue-on-error", match.group(1))
        elif match := _STEP_SHELL.match(text):
            keys.setdefault("shell", match.group(1))
    reason: str | None = None
    condition = keys.get("if")
    if condition is not None and condition not in _RUNS_IN_A_PASSING_JOB:
        reason = f"if: {condition}"
    tolerated = keys.get("continue-on-error")
    if reason is None and tolerated is not None and tolerated not in _FALSEY:
        reason = f"continue-on-error: {tolerated}"
    return reason, "\n".join(lines), keys.get("shell")


def unconditional_run_blocks(
    job_body: str, also_credit: frozenset[str] = frozenset()
) -> tuple[list[str], list[str]]:
    """Run blocks that a successful job is guaranteed to have executed.

    Returns those blocks and the `if:` expressions that were refused, so a
    caller can say *why* a package looks unexecuted rather than only that it is.

    `also_credit` names refusal reasons a *particular* caller can discharge, and
    defaults to discharging none. Only the Windows-coverage check passes it, to
    credit `if: runner.os == 'Windows'` on the platform where that is true. It is
    a parameter rather than an entry in `_RUNS_IN_A_PASSING_JOB` precisely so it
    cannot leak into the required-tier gate, where a Windows-only step must stay
    refused -- both required jobs run on Linux, so crediting it there would let
    a step that never executes satisfy the gate.
    """
    kept: list[str] = []
    refused: list[str] = []
    if match := _JOB_COE.search(job_body):
        if match.group(1) not in _FALSEY:
            # The job itself is allowed to fail, so nothing it runs can block a
            # merge. Refuse every block rather than a step at a time.
            return [], [f"continue-on-error: {match.group(1)} (job level)"]
    for reason, text, shell in job_steps(job_body):
        blocks = run_blocks(text)
        if not blocks:
            continue
        if reason is not None and reason not in also_credit:
            refused.append(reason)
            continue
        pipefail = shell is not None and shell.split()[0] == "bash"
        for block in blocks:
            swallowed = _swallow_reason(block, pipefail)
            if swallowed:
                refused.append(swallowed)
            else:
                kept.append(block)
    return kept, refused


def run_blocks(job_body: str) -> list[str]:
    """Every `run:` scalar in a job, folded or literal, as shell text."""
    lines = job_body.splitlines()
    blocks: list[str] = []
    index = 0
    while index < len(lines):
        match = _RUN_KEY.match(lines[index])
        if not match:
            index += 1
            continue
        indent = len(match.group(1)) + (2 if match.group(2) else 0)
        header = match.group(3).strip()
        index += 1
        if header and header not in _FOLDED and header not in _LITERAL:
            blocks.append(header)
            continue
        # A folded scalar is one shell command wrapped across lines; a literal
        # one is a script whose newlines separate commands. Collapsing the two
        # would either lose a command boundary or invent one.
        joiner = " " if header in _FOLDED else "\n"
        collected: list[str] = []
        while index < len(lines):
            line = lines[index]
            if line.strip() and (len(line) - len(line.lstrip())) <= indent:
                break
            collected.append(line.strip())
            index += 1
        blocks.append(joiner.join(part for part in collected if part))
    return blocks


def packages_tested_by(commands: Iterable[str]) -> set[str]:
    """Packages whose tests the given shell commands actually execute.

    Known, deliberate under-read: a `cargo test` continued across physical lines
    with a trailing backslash puts its `-p` flags in fragments that carry no
    cargo invocation, so they are not attributed. `Rust quality`'s three MLAS
    steps are written that way and are credited to no lane by this function --
    harmless today because `Fast` covers that package, and harmless in principle
    because it can only ever make the gate stricter. It is NOT joined here on
    purpose: joining continuations moves attribution in the permissive
    direction, and this gate's whole thesis is that an over-read is fatal while
    an under-read merely fails loudly. If a package's only required-lane
    execution is ever written this way, this gate will refuse it rather than
    credit it, and the fix is to put the invocation on one line.
    Bounded exception, added when the guarded two-line form landed: a lane
    resolved into a shell variable is credited to a later `cargo test` in the
    same block *only if that command references the variable*. The reference is
    what does the crediting, so this stays a dataflow read rather than a join --
    an assignment the cargo invocation never uses is still worth nothing, and
    the self-test arms below fail if it is ever worth something. Reassignment
    replaces a variable's lanes rather than adding to them, so shadowing the
    name with a non-lane value drops the credit instead of keeping it.
    """
    tested: set[str] = set()
    for command in commands:
        lanes_by_var: dict[str, set[str]] = {}
        for fragment in re.split(r"&&|\|\||;|\n", command):
            for assignment in _VAR_ASSIGN.finditer(fragment):
                lanes_by_var[assignment.group("var")] = set(
                    _LANE_CALL.findall(assignment.group("value"))
                )
            if not _RUNS_TESTS.search(fragment):
                continue
            for lane in _LANE_CALL.findall(fragment):
                tested.update(lane_packages(lane))
            for name in _VAR_REF.findall(fragment):
                for lane in lanes_by_var.get(name, ()):
                    tested.update(lane_packages(lane))
            tested.update(_PACKAGE_FLAG.findall(fragment))
    return tested


def required_lane_commands(skip_lane: str | None = None) -> tuple[list[str], set[str]]:
    jobs = workflow_jobs()
    commands: list[str] = []
    for name in sorted(REQUIRED_JOB_NAMES & set(jobs)):
        blocks, refused = unconditional_run_blocks(jobs[name])
        for condition in refused:
            print(
                f"note: {name!r} has a conditional step this gate will not credit: "
                f"if: {condition}",
                file=sys.stderr,
            )
        for block in blocks:
            if skip_lane and re.search(rf"cargo-args\s+{re.escape(skip_lane)}\b", block):
                continue
            commands.append(block)
    return commands, set(jobs)


# `ci.yml` states, in a comment beside the step this PR made Windows-only, that
# `CLI ORT` "remains the only place these tests run on Windows". That was prose,
# and prose does not fail. @holden measured the gap it left: indenting a job key
# by two extra spaces is *valid YAML*, silently reparents the job into the one
# above it, and GitHub then reports that job's checks as **absent** rather than
# failing -- so `gh pr checks` shows no red, because it shows nothing at all. A
# gate keyed on "pending=0 and no failures" is true of a vanished check.
#
# Deleting the `cli-ort` block outright is the quieter form: still valid YAML,
# still a valid workflow, and every gate here passed it before this check
# existed. Measured, on the real workflow, before writing it: all three of
# `verify`, `verify-required-tier` and `self-test` exited 0 with the sole
# Windows executor of the ORT-backed tests removed.
_WINDOWS_RUNNER = re.compile(r"windows-[A-Za-z0-9_.-]+")

# Refusal reasons that are discharged *on a Windows runner only*. `job_steps`
# formats a refused condition as `if: <expr>`, so these match its output.
_WINDOWS_STEP_IF = frozenset(
    {
        "if: runner.os == 'Windows'",
        'if: runner.os == "Windows"',
        "if: ${{ runner.os == 'Windows' }}",
    }
)


def windows_ort_executors(jobs: dict[str, str] | None = None) -> dict[str, set[str]]:
    """Jobs that can run on Windows and execute ORT-backed packages there."""
    jobs = workflow_jobs() if jobs is None else jobs
    found: dict[str, set[str]] = {}
    for name, body in jobs.items():
        if not _WINDOWS_RUNNER.search(body):
            continue
        blocks, _ = unconditional_run_blocks(body, also_credit=_WINDOWS_STEP_IF)
        if tested := packages_tested_by(blocks) & set(ORT_BACKED):
            found[name] = tested
    return found


def verify_windows_ort_coverage(jobs: dict[str, str] | None = None) -> int:
    """Fail if no job still runs the ORT-backed tests on a Windows runner."""
    executors = windows_ort_executors(jobs)
    covered: set[str] = set()
    for tested in executors.values():
        covered |= tested
    if missing := sorted(set(ORT_BACKED) - covered):
        print("Windows ORT coverage check failed.", file=sys.stderr)
        print(
            "No job runs these packages' tests on a Windows runner:", file=sys.stderr
        )
        for package in missing:
            print(f"  - {package}", file=sys.stderr)
        print(
            "No required check builds the ORT graph on Windows, so this coverage "
            "exists only here. A job that stops running them does not go red -- "
            "its checks stop appearing, which no pass/fail gate can see.",
            file=sys.stderr,
        )
        return 1
    print(
        f"windows ORT coverage ok: {len(covered)} package(s) via "
        f"{sorted(executors)}"
    )
    return 0


def verify_required_tier(simulate_dropped_lane: str | None = None) -> int:
    commands, job_names = required_lane_commands(simulate_dropped_lane)

    if missing_jobs := sorted(REQUIRED_JOB_NAMES - job_names):
        print(
            "Required-lane coverage check failed.\n"
            f"REQUIRED_JOB_NAMES names job(s) that no longer exist in {WORKFLOW.name}: "
            f"{missing_jobs}\n"
            "Renaming a required job also breaks the branch ruleset; update both.",
            file=sys.stderr,
        )
        return 1

    if rc := verify_required_job_conditions(workflow_jobs()):
        return rc

    packages = workspace_packages()
    required_tested = packages_tested_by(commands) & packages
    unenforced = sorted(packages - required_tested - set(DENYLIST))
    if unenforced:
        print("Required-lane coverage check failed.", file=sys.stderr)
        print(
            "Workspace member(s) whose tests no required status check executes:",
            file=sys.stderr,
        )
        for package in unenforced:
            print(f"  - {package}", file=sys.stderr)
        print(
            "Their tests can go red on `main` while every required check is green.\n"
            f"Run them from one of {sorted(REQUIRED_JOB_NAMES)}, or add a DENYLIST "
            "entry with a reason.",
            file=sys.stderr,
        )
        return 1
    print(
        f"required-lane coverage ok: {len(required_tested)} package(s) tested by "
        f"{sorted(REQUIRED_JOB_NAMES)}, {len(DENYLIST)} denied"
    )
    return verify_windows_ort_coverage()


def workflow_files() -> list[Path]:
    """Every file GitHub will read as a workflow, both extensions it accepts."""
    return sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))


def verify_workflow_integrity(simulate_deleted_job: str | None = None) -> int:
    """Every workflow file parses, and defines exactly the jobs it is recorded as defining.

    `verify_required_tier` covers `ci.yml` because `REQUIRED_JOB_NAMES` is an
    inventory of it, and branch protection covers `ci.yml` again because a
    required check that stops reporting blocks the merge. Neither reaches a file
    that contains no required job. Measured on `diff-guard.yml`: nesting
    `root-files` two spaces deeper leaves valid YAML, drops the root allowlist
    from what GitHub runs, and `verify`, `verify-required-tier` and `self-test`
    all still returned 0.
    """
    files = workflow_files()
    failures: list[str] = []
    parsed: dict[str, list[str]] = {}
    for path in files:
        if path.name not in WORKFLOW_JOB_INVENTORY:
            continue
        try:
            found = sorted(parse_jobs(path.read_text(encoding="utf-8"), source=str(path)))
        except SystemExit as refusal:
            failures.append(str(refusal))
            continue
        if simulate_deleted_job and simulate_deleted_job in found:
            found.remove(simulate_deleted_job)
        parsed[path.name] = found

    failures += _integrity_failures(
        {path.name for path in files}, parsed, WORKFLOW_JOB_INVENTORY
    )
    seen = sum(len(jobs) for jobs in parsed.values())

    if failures:
        print("Workflow job integrity check failed.", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    if not seen:
        print(
            "Workflow job integrity check failed: no jobs were read at all, so this "
            "gate proved nothing.",
            file=sys.stderr,
        )
        return 1
    print(f"workflow job integrity ok: {len(files)} file(s), {seen} job(s) as recorded")
    return 0


def _integrity_failures(
    on_disk: set[str],
    parsed: dict[str, list[str]],
    inventory: dict[str, tuple[str, ...]],
) -> list[str]:
    """Every difference between what the workflows define and what is recorded."""
    failures: list[str] = []
    for name in sorted(set(inventory) - on_disk):
        failures.append(
            f"{name}: recorded in WORKFLOW_JOB_INVENTORY but not on disk. Deleting a "
            "workflow file deletes its checks silently; remove the entry in the same "
            "commit if that is intended."
        )
    for name in sorted(on_disk - set(inventory)):
        failures.append(
            f"{name}: on disk but absent from WORKFLOW_JOB_INVENTORY, so its jobs are "
            "unguarded. Add it with the block this gate prints."
        )
    for name, found in sorted(parsed.items()):
        expected = list(inventory[name])
        if not found:
            failures.append(
                f"{name}: defines no jobs at all. A workflow that runs nothing reports "
                "no checks, which is indistinguishable from one that passed."
            )
            continue
        if found == expected:
            continue
        if gone := sorted(set(expected) - set(found)):
            failures.append(
                f"{name}: job(s) GitHub will no longer run: {gone}. A job key indented "
                "one level too far, or a block deleted outright, removes the check "
                "rather than failing it."
            )
        if added := sorted(set(found) - set(expected)):
            failures.append(f"{name}: job(s) not recorded in the inventory: {added}.")
        failures.append(
            f"{name}: if this change is intended, replace its entry with:\n"
            + _inventory_entry(name, found)
        )
    return failures


def _inventory_entry(name: str, jobs: Iterable[str]) -> str:
    body = "".join(f'            "{job}",\n' for job in jobs)
    return f'        "{name}": (\n{body}        ),'


# Job bodies whose keys are written in the ways YAML allows. A key form this
# parser fails to recognise does not degrade to "unknown job": the body is
# appended to whatever job came last, so an advisory job's steps are credited to
# the required job above it. `rust-coverage` sits directly below `rust-quality`,
# so that is a concrete route to a false green, and it is the reason
# `_at_job_depth` raises instead of guessing. Measured before the fix: trailing
# comment and quoted forms both produced a passing gate on a workflow whose
# required job ran none of the tests.
_PARSER_FIXTURE = """jobs:
  alpha:
    name: Required Job
    steps:
      - run: cargo test -p pkg-alpha
{key}
    name: Advisory Job
    steps:
      - run: cargo test -p pkg-beta
"""


_CONDITIONAL_ARMS: tuple[tuple[str, str | None, list[str], list[str]], ...] = (
    # label, step YAML (None = job-level arm), packages credited, reasons refused
    (
        "unconditional step is credited",
        "      - name: x\n        run: cargo test -p pkg-alpha\n",
        ["pkg-alpha"],
        [],
    ),
    # Reviewer finding: the step-key column is not always dash+2. YAML allows any
    # run of spaces after the dash, which pushes the real keys deeper; reading the
    # column as dash+2 dropped `if:` as shell text and CREDITED a guarded step.
    (
        "extra spaces after the dash still expose the step's if:",
        "      -   name: x\n          if: false\n          run: cargo test -p pkg-alpha\n",
        [],
        ["if: false"],
    ),
    (
        "extra spaces after the dash still expose continue-on-error",
        "      -   name: x\n          continue-on-error: true\n          run: cargo test -p pkg-alpha\n",
        [],
        ["continue-on-error: true"],
    ),
    (
        "a run: block leading the step does not hide a later if:",
        "      - run: |\n          cargo test -p pkg-alpha\n        if: false\n",
        [],
        ["if: false"],
    ),
    (
        "if: always() is credited",
        "      - name: x\n        if: always()\n        run: cargo test -p pkg-alpha\n",
        ["pkg-alpha"],
        [],
    ),
    (
        "if: false is not credited",
        "      - name: x\n        if: false\n        run: cargo test -p pkg-alpha\n",
        [],
        ["if: false"],
    ),
    (
        "a platform guard is not credited",
        "      - name: x\n        if: runner.os == 'Windows'\n        run: cargo test -p pkg-alpha\n",
        [],
        ["if: runner.os == 'Windows'"],
    ),
    (
        "if: on the list-item line is still a step condition",
        "      - if: false\n        run: cargo test -p pkg-alpha\n",
        [],
        ["if: false"],
    ),
    (
        "continue-on-error: true is not credited",
        "      - name: x\n        continue-on-error: true\n        run: cargo test -p pkg-alpha\n",
        [],
        ["continue-on-error: true"],
    ),
    (
        "job-level continue-on-error refuses the whole job",
        None,
        [],
        ["continue-on-error: true (job level)"],
    ),
    (
        "continue-on-error: false is credited",
        "      - name: x\n        continue-on-error: false\n        run: cargo test -p pkg-alpha\n",
        ["pkg-alpha"],
        [],
    ),
    (
        "`|| true` after a cargo test is not credited",
        "      - name: x\n        run: cargo test -p pkg-alpha || true\n",
        [],
        ["`||` after a cargo test swallows its failure"],
    ),
    (
        "piped cargo test without pipefail is not credited",
        "      - name: x\n        run: cargo test -p pkg-alpha | tee out.log\n",
        [],
        ["cargo test piped without pipefail; its failure is masked"],
    ),
    (
        "piped under `shell: bash` (pipefail) is credited",
        "      - name: x\n        shell: bash\n        run: cargo test -p pkg-alpha | tee out.log\n",
        ["pkg-alpha"],
        [],
    ),
    (
        "an && chain is credited",
        "      - name: x\n        run: cargo test -p pkg-alpha && echo done\n",
        ["pkg-alpha"],
        [],
    ),
    (
        "`if:` inside shell text is not a step condition",
        "      - name: x\n        run: |\n          if: not-a-condition\n          cargo test -p pkg-alpha\n",
        ["pkg-alpha"],
        [],
    ),
)


def _conditional_arms() -> int:
    """A step the job may skip must not count as having run its tests."""
    failures = 0
    for label, step, want_credited, want_refused in _CONDITIONAL_ARMS:
        body = (
            "    continue-on-error: true\n    steps:\n"
            "      - name: x\n        run: cargo test -p pkg-alpha\n"
            if step is None
            else "    steps:\n" + step
        )
        kept, refused = unconditional_run_blocks(body)
        credited = sorted(packages_tested_by(kept))
        if credited != want_credited or refused != want_refused:
            failures += 1
            print(f"  FAIL  {label}", file=sys.stderr)
            print(f"        credited want={want_credited} got={credited}", file=sys.stderr)
            print(f"        refused  want={want_refused} got={refused}", file=sys.stderr)
        else:
            print(f"  ok    {label} -> credited {credited}, refused {refused}")
    return failures


_PARSER_ARM_COUNT = 8


# Each arm is (label, workflow text, the packages the scanner must credit).
# Every one of these fails *quietly* if the terminator it exercises is removed:
# the package appears linted and the gate stays green. The two prose arms are
# the reviewer's, not mine -- I had left `# -p ghost` inside a folded scalar
# harvestable while the docstring claimed comments could not grant coverage.
_CLIPPY_BLOCK_ARMS: tuple[tuple[str, str, list[str]], ...] = (
    (
        "a plain clippy step credits its own packages (control)",
        "      - name: lint\n"
        "        run: >-\n"
        "          cargo clippy --all-targets\n"
        "          -p real\n"
        "          -- -D warnings\n",
        ["real"],
    ),
    (
        "the next step's packages are not credited to clippy",
        "      - name: lint\n"
        "        run: cargo clippy -p real --all-targets -- -D warnings\n"
        "      - name: test\n"
        "        run: cargo test -p tested-not-linted\n",
        ["real"],
    ),
    (
        # Contrived on purpose: with the indent rule in place this is the only
        # shape whose verdict the blank line actually decides. Without it the
        # branch is unobservable, and an unobservable branch is one nobody can
        # tell is broken.
        "a blank line ends the block even when what follows is indented deeper",
        "      - name: lint\n"
        "        run: >-\n"
        "          cargo clippy -p real --all-targets -- -D warnings\n"
        "\n"
        "          -p ghost-after-blank\n",
        ["real"],
    ),
    (
        "a following non-cargo step is not credited either",
        "      - name: lint\n"
        "        run: cargo clippy -p real --all-targets -- -D warnings\n"
        "      - name: shell lint\n"
        "        run: ./scripts/lint.sh -p ghost-next\n",
        ["real"],
    ),
    (
        "a second cargo command in the same block is not clippy",
        "      - name: lint and test\n"
        "        run: |\n"
        "          cargo clippy -p real --all-targets -- -D warnings\n"
        "          cargo test -p tested-not-linted\n",
        ["real"],
    ),
    (
        "an && chain to a cargo test is not clippy either",
        "      - name: lint then test\n"
        "        run: cargo clippy -p real -- -D warnings && cargo test -p tested-not-linted\n",
        ["real"],
    ),
    (
        "a deeper-indented comment inside the block grants nothing",
        "      - name: lint\n"
        "        run: >-\n"
        "          cargo clippy --all-targets\n"
        "            # -p ghost-deeper is prose, not an invocation\n"
        "          -p real\n"
        "          -- -D warnings\n",
        ["real"],
    ),
    (
        "a comment that merely discusses clippy starts no block",
        "      # cargo clippy -p ghost-prose would be nice to add\n"
        "      - name: test\n"
        "        run: cargo test -p tested-not-linted\n",
        [],
    ),
)


def _clippy_block_arms() -> int:
    """Prove the lint scanner stops where it says it stops."""
    failures = 0
    for label, text, want in _CLIPPY_BLOCK_ARMS:
        blocks = clippy_blocks(text.splitlines())
        got = sorted(set(re.findall(r"-p\s+([A-Za-z0-9_-]+)", "\n".join(blocks))))
        if got != sorted(want):
            failures += 1
            print(f"  FAIL  {label}", file=sys.stderr)
            print(f"        credited want={sorted(want)} got={got}", file=sys.stderr)
            print(f"        blocks: {blocks}", file=sys.stderr)
        else:
            print(f"  ok    {label} -> credited {got}")
    return failures


_WORKFLOW_INTEGRITY_ARMS = (
    "an inventoried file that vanished from disk",
    "a workflow file nobody recorded",
    "a file whose jobs all disappeared",
    "an exact match is accepted (control)",
    "a renamed job is reported in both directions",
    "every workflow file on disk is actually read",
)


def _workflow_integrity_arms() -> int:
    """Prove the inventory refuses each way a job can stop running, and only those."""
    failures = 0
    inventory = {"a.yml": ("Alpha", "Beta"), "b.yml": ("Gamma",)}

    def arm(label: str, on_disk: set[str], parsed: dict[str, list[str]], expect: list[str]) -> int:
        found = _integrity_failures(on_disk, parsed, inventory)
        text = "\n".join(found)
        missing = [needle for needle in expect if needle not in text]
        if expect and not found:
            print(f"  FAIL  {label}: accepted; expected it to name {expect}", file=sys.stderr)
            return 1
        if not expect and found:
            print(f"  FAIL  {label}: refused a correct inventory: {found}", file=sys.stderr)
            return 1
        if missing:
            print(f"  FAIL  {label}: refused without naming {missing}", file=sys.stderr)
            return 1
        print(f"  ok    {label} -> {'refused' if found else 'accepted'}")
        return 0

    both = {"a.yml", "b.yml"}
    failures += arm(
        "an inventoried file that vanished from disk",
        {"a.yml"},
        {"a.yml": ["Alpha", "Beta"]},
        ["b.yml", "not on disk"],
    )
    failures += arm(
        "a workflow file nobody recorded",
        both | {"c.yml"},
        {"a.yml": ["Alpha", "Beta"], "b.yml": ["Gamma"]},
        ["c.yml", "unguarded"],
    )
    failures += arm(
        "a file whose jobs all disappeared",
        both,
        {"a.yml": ["Alpha", "Beta"], "b.yml": []},
        ["b.yml", "defines no jobs at all"],
    )
    # Without this the suite would be passed perfectly by a gate that refuses
    # every input, which is the same defect one sign flipped.
    failures += arm(
        "an exact match is accepted (control)",
        both,
        {"a.yml": ["Alpha", "Beta"], "b.yml": ["Gamma"]},
        [],
    )
    failures += arm(
        "a renamed job is reported in both directions",
        both,
        {"a.yml": ["Alpha", "Beta renamed"], "b.yml": ["Gamma"]},
        ["no longer run", "not recorded in the inventory", "Beta renamed"],
    )

    # `verify_workflow_integrity` derives both sides of its file comparison from
    # `workflow_files()`, so a file that helper silently skipped would be absent
    # from `on_disk` *and* from `parsed`, and would read as agreement. This is
    # the only arm that resolves the file list against something else.
    label = "every workflow file on disk is actually read"
    independently = {
        name
        for name in os.listdir(WORKFLOWS)
        if name.endswith((".yml", ".yaml")) and (WORKFLOWS / name).is_file()
    }
    read = {path.name for path in workflow_files()}
    if not independently:
        failures += 1
        print(
            f"  FAIL  {label}: found no workflow files, so this arm proved nothing",
            file=sys.stderr,
        )
    elif skipped := sorted(independently - read):
        failures += 1
        print(f"  FAIL  {label}: workflow_files() never opened {skipped}", file=sys.stderr)
    else:
        print(f"  ok    {label} -> {len(read)} file(s), none skipped")
    return failures


def _parser_arms() -> int:
    """Prove a job boundary is never absorbed into the job above it."""
    failures = 0
    for label, key in (
        ("plain job key (control)", "  beta:"),
        ("job key with a trailing comment", "  beta:  # advisory lane"),
        ('job key in double quotes', '  "beta":'),
        ("job key in single quotes", "  'beta':"),
    ):
        text = _PARSER_FIXTURE.format(key=key)
        try:
            jobs = parse_jobs(text)
        except SystemExit as exit_error:
            print(f"  FAIL  {label}: parser refused a valid workflow: {exit_error}", file=sys.stderr)
            failures += 1
            continue
        required_body = jobs.get("Required Job", "")
        leaked = "pkg-beta" in required_body
        if sorted(jobs) != ["Advisory Job", "Required Job"] or leaked:
            failures += 1
            print(f"  FAIL  {label}", file=sys.stderr)
            print(f"        jobs parsed: {sorted(jobs)}", file=sys.stderr)
            if leaked:
                print("        the advisory job's step was credited to the required job", file=sys.stderr)
        else:
            print(f"  ok    {label} -> two jobs, no step credited across the boundary")

    # An unrecognised line at job-key depth must stop the gate, not extend a job.
    label = "unparseable line at job-key depth is fatal"
    try:
        parse_jobs(_PARSER_FIXTURE.format(key="  beta: not-a-job-body"))
    except SystemExit:
        print(f"  ok    {label} -> refused")
    else:
        failures += 1
        print(f"  FAIL  {label}: parsed without complaint", file=sys.stderr)

    # A job key indented one level too far is *valid YAML*: the job silently
    # becomes a key of the job above it, and GitHub then stops producing its
    # checks rather than failing them. `gh pr checks` cannot express an absence,
    # so a `pending=0 && failed=[]` merge gate is true of a deleted check.
    label = "job key indented one level too far is fatal"
    try:
        jobs = parse_jobs(_PARSER_FIXTURE.format(key="    beta:"))
    except SystemExit as exit_error:
        if "beta" in str(exit_error):
            print(f"  ok    {label} -> refused, naming 'beta'")
        else:
            failures += 1
            print(f"  FAIL  {label}: refused without naming the stray key", file=sys.stderr)
    else:
        failures += 1
        print(f"  FAIL  {label}: parsed as {sorted(jobs)}", file=sys.stderr)

    # Control for the arm above: the guard must not reject the ordinary job
    # attributes that legitimately sit at that same four-space depth, or it
    # would refuse every real workflow and its refusals would mean nothing.
    label = "real job attributes at four-space depth are accepted"
    attributes = "    needs: alpha\n    if: success()\n    runs-on: ubuntu-latest\n"
    try:
        jobs = parse_jobs(_PARSER_FIXTURE.format(key="  beta:\n" + attributes))
    except SystemExit as exit_error:
        failures += 1
        print(f"  FAIL  {label}: refused a valid workflow: {exit_error}", file=sys.stderr)
    else:
        if sorted(jobs) == ["Advisory Job", "Required Job"]:
            print(f"  ok    {label} -> two jobs")
        else:
            failures += 1
            print(f"  FAIL  {label}: jobs parsed: {sorted(jobs)}", file=sys.stderr)

    # The ORT-backed tests run on Windows in exactly one job, and no required
    # check builds that graph on Windows at all. Deleting that job is valid
    # YAML and a valid workflow, so nothing else here would notice.
    label = "no Windows executor for the ORT-backed tests is fatal"
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = verify_windows_ort_coverage(jobs={"Some Linux Job": "    runs-on: ubuntu-latest\n"})
    text = out.getvalue() + err.getvalue()
    if code == 1 and all(package in text for package in ORT_BACKED):
        print(f"  ok    {label} -> refused, naming all {len(ORT_BACKED)} packages")
    else:
        failures += 1
        print(f"  FAIL  {label}: rc={code}, named={text!r}", file=sys.stderr)
    return failures



_OK_IF = "needs.changes.outputs.docs_only != 'true'"
_REQ = sorted(REQUIRED_JOB_NAMES)


def _job_body(condition: str | None = None, step_condition: str | None = None) -> str:
    body = "    name: x\n    runs-on: ubuntu-latest\n"
    if condition is not None:
        body += f"    if: {condition}\n"
    body += "    steps:\n      - name: t\n"
    if step_condition is not None:
        body += f"        if: {step_condition}\n"
    return body + "        run: cargo test -p pkg-alpha\n"


# label, jobs, expected exit, names the report must contain
_JOB_CONDITION_ARMS: tuple[tuple[str, dict, int, list[str]], ...] = (
    # Positive control FIRST: a zero from the arms below has to be attributable
    # to the condition being acceptable, not to a harness that cannot refuse.
    ("job-if: the known docs-only guard is accepted",
     {n: _job_body(_OK_IF) for n in _REQ}, 0, []),
    ("job-if: absent is accepted (the job always runs)",
     {n: _job_body(None) for n in _REQ}, 0, []),
    ("job-if: narrowed on one required job is refused",
     {_REQ[0]: _job_body(_OK_IF),
      _REQ[1]: _job_body(_OK_IF + " && github.event_name == 'push'")}, 1, [_REQ[1]]),
    ("job-if: replaced wholesale is refused",
     {_REQ[0]: _job_body("false"), _REQ[1]: _job_body(_OK_IF)}, 1, [_REQ[0]]),
    # Discriminator: a step-level `if:` is indented deeper and must not be read
    # as the job's. Without this arm a regex that matched any `if:` would pass
    # every arm above -- the fixtures would not distinguish the two.
    # Discriminator. The fixture carries ONLY a step-level `if:` -- no job-level
    # one -- because a job-level guard would be matched first by `.search()` and
    # the arm would pass under a regex that wrongly matched both. Monkeypatching
    # `_JOB_IF` to `^\s*if:` makes exactly this arm fail, which is what makes it
    # a control rather than a restatement.
    ("job-if: a step-level if: is not read as the job's",
     {n: _job_body(None, step_condition="false") for n in _REQ}, 0, []),
    ("job-if: both present -- the job's is the one read",
     {n: _job_body(_OK_IF, step_condition="false") for n in _REQ}, 0, []),
    ("job-if: an absent required job is refused, not passed over",
     {_REQ[0]: _job_body(_OK_IF)}, 1, [_REQ[1]]),
)


def _job_condition_arms() -> int:
    failures = 0
    for label, jobs, want_code, want_named in _JOB_CONDITION_ARMS:
        out, err = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(out), redirect_stderr(err):
                code = verify_required_job_conditions(jobs)
        except SystemExit as exit_error:
            print(f"  FAIL  {label}\n        raised SystemExit: {exit_error}", file=sys.stderr)
            failures += 1
            continue
        text = out.getvalue() + err.getvalue()
        missing = [name for name in want_named if name not in text]
        if code != want_code or missing:
            failures += 1
            print(f"  FAIL  {label}\n        exit want={want_code} got={code}", file=sys.stderr)
            if missing:
                print(f"        expected the report to name: {missing}", file=sys.stderr)
        else:
            named = f", naming {want_named}" if want_named else ""
            print(f"  ok    {label} -> exit {code}{named}")
    return failures

def _probe_guarded_assignment(used: bool = True) -> int:
    """Attribution probe for the guarded two-line form.

    `used=True` is the form the workflow ships. `used=False` is the mutation
    that matters: the same assignment, with a `cargo test` that never mentions
    the variable. If that arm ever credits the lane, the dataflow read has
    become a join and the gate has moved in the permissive direction.
    """
    resolver = "python .github/scripts/workspace_test_packages.py cargo-args ort-backed"
    invocation = (
        "cargo test --locked $packages -- --test-threads=1"
        if used
        else "cargo test --locked -p onnx-genai-cli"
    )
    block = "\n".join(
        [
            f'packages="$({resolver})"',
            'test -n "$packages" || { echo "::error::empty"; exit 1; }',
            invocation,
        ]
    )
    tested = packages_tested_by([block])
    want = set(ORT_BACKED) if used else {"onnx-genai-cli"}
    if tested != want:
        print(
            "attribution mismatch: "
            f"unexpected={sorted(tested - want)} absent={sorted(want - tested)}",
            file=sys.stderr,
        )
        return 1
    print("attribution as stated: " + ", ".join(sorted(tested)))
    return 0


def _probe_clippy_generator_var(shape: str = "referenced") -> int:
    """Attribution probe for a clippy step whose package list is in a variable.

    `referenced` is the form the workflow ships. Every other shape is a way the
    backwards read could become a join, and each one fails quietly if it
    regresses: an over-credited package looks linted and nothing says
    otherwise, so `verify` stays green while the coverage claim is false.

    * `unreferenced` -- the assignment is in the step and the invocation never
      mentions it.
    * `prefix` -- the invocation names `$packages_extra`, which contains the
      assigned name as a prefix.
    * `shadowed` -- a plain assignment replaces the generator before the
      invocation runs, which is the property `packages_tested_by` documents.
    * `neighbour` -- the assignment belongs to the *previous* step, and the
      clippy step carries no `name:` so it sits at the invocation's own
      indentation.
    """
    lane = "python .github/scripts/workspace_test_packages.py cargo-args linux-only"
    step = [
        "      - name: Clippy offline crates",
        "        shell: bash",
        "        run: |",
        f'          packages="$({lane})"',
        '          test -n "$packages" || { echo "::error::empty"; exit 1; }',
    ]
    if shape == "referenced":
        lines = step + ["          cargo clippy --locked --all-targets $packages -- -D warnings"]
        want = set(lane_packages("linux-only"))
    elif shape == "unreferenced":
        lines = step + [
            "          cargo clippy --locked --all-targets -p onnx-genai-cli -- -D warnings"
        ]
        want = {"onnx-genai-cli"}
    elif shape == "prefix":
        lines = step + [
            '          packages_extra="-p onnx-genai-cli"',
            "          cargo clippy --locked --all-targets $packages_extra"
            " -p onnx-genai-cli -- -D warnings",
        ]
        want = {"onnx-genai-cli"}
    elif shape == "shadowed":
        lines = step + [
            '          packages="-p onnx-genai-cli"',
            "          cargo clippy --locked --all-targets $packages"
            " -p onnx-genai-cli -- -D warnings",
        ]
        want = {"onnx-genai-cli"}
    elif shape == "neighbour":
        lines = step + [
            "          cargo test --locked $packages",
            "      - run: cargo clippy --locked --all-targets $packages -- -D warnings",
        ]
        want = set()
    else:  # pragma: no cover - a typo in an arm must not read as a pass
        print(f"unknown probe shape {shape!r}", file=sys.stderr)
        return 1

    credited: set[str] = set()
    for block in clippy_blocks(lines):
        for found in _GENERATOR.findall(block):
            if found in LANES:
                credited.update(lane_packages(found))
        credited.update(re.findall(r"-p\s+([A-Za-z0-9_-]+)", block))
    if credited != want:
        print(
            "clippy attribution mismatch: "
            f"unexpected={sorted(credited - want)} absent={sorted(want - credited)}",
            file=sys.stderr,
        )
        return 1
    print("clippy attribution as stated: " + (", ".join(sorted(credited)) or "credited nothing"))
    return 0


def self_test() -> int:
    """Prove both gates still refuse, on content and not merely on exit code.

    An exit status is not enough to tell a refusal from a crash: `python` not
    being on PATH exits 127, and every `SystemExit` in this file exits 1, so a
    workflow-parse failure is indistinguishable from a real verdict. Each arm
    below therefore states the packages it expects to be named and fails if the
    guard reports anything else -- including reporting nothing.
    """
    arms: list[tuple[str, object, dict, int, list[str]]] = [
        ("verify (control)", verify, {}, 0, []),
        (
            "verify --simulate-missing onnx-genai-engine",
            verify,
            {"simulate_missing": "onnx-genai-engine"},
            1,
            ["onnx-genai-engine"],
        ),
        (
            "verify --simulate-unlinted onnx-runtime-ep-cpu",
            verify,
            {"simulate_unlinted": "onnx-runtime-ep-cpu"},
            1,
            ["onnx-runtime-ep-cpu"],
        ),
        ("verify-required-tier (control)", verify_required_tier, {}, 0, []),
        # Driven through the real entry point, not the helper: the defect this
        # gate exists for lives in *which files get read*, and a helper handed
        # the right dict by hand cannot see a file that was never opened.
        ("verify-workflow-integrity (control)", verify_workflow_integrity, {}, 0, []),
        (
            'verify-workflow-integrity --simulate-deleted-job "Root file allowlist"',
            verify_workflow_integrity,
            {"simulate_deleted_job": "Root file allowlist"},
            1,
            ["Root file allowlist", "diff-guard.yml"],
        ),
        (
            "verify-required-tier --simulate-dropped-lane ort-backed",
            verify_required_tier,
            {"simulate_dropped_lane": "ort-backed"},
            1,
            sorted(ORT_BACKED),
        ),
        (
            "packages_tested_by credits a lane the cargo call uses",
            _probe_guarded_assignment,
            {"used": True},
            0,
            sorted(ORT_BACKED),
        ),
        (
            "packages_tested_by refuses a lane the cargo call never uses",
            _probe_guarded_assignment,
            {"used": False},
            0,
            ["onnx-genai-cli"],
        ),
        (
            "linted_packages credits a lane the clippy call reads from a variable",
            _probe_clippy_generator_var,
            {"shape": "referenced"},
            0,
            sorted(lane_packages("linux-only")),
        ),
        (
            "linted_packages refuses a lane the clippy call never references",
            _probe_clippy_generator_var,
            {"shape": "unreferenced"},
            0,
            ["onnx-genai-cli"],
        ),
        (
            "linted_packages refuses a variable whose name is only a prefix",
            _probe_clippy_generator_var,
            {"shape": "prefix"},
            0,
            ["onnx-genai-cli"],
        ),
        (
            "linted_packages refuses a generator a plain assignment shadowed",
            _probe_clippy_generator_var,
            {"shape": "shadowed"},
            0,
            ["onnx-genai-cli"],
        ),
        (
            "linted_packages refuses a variable assigned by the previous step",
            _probe_clippy_generator_var,
            {"shape": "neighbour"},
            0,
            ["credited nothing"],
        ),
    ]
    failures = 0
    for label, function, kwargs, want_code, want_named in arms:
        out, err = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(out), redirect_stderr(err):
                code = function(**kwargs)
        except SystemExit as exit_error:  # a crash must not read as a verdict
            print(f"  FAIL  {label}\n        raised SystemExit: {exit_error}", file=sys.stderr)
            failures += 1
            continue
        text = out.getvalue() + err.getvalue()
        missing = [package for package in want_named if package not in text]
        if code != want_code or missing:
            failures += 1
            print(f"  FAIL  {label}", file=sys.stderr)
            print(f"        exit want={want_code} got={code}", file=sys.stderr)
            if missing:
                print(f"        expected the report to name: {missing}", file=sys.stderr)
            print("        ---\n" + "\n".join(f"        {line}" for line in text.splitlines()), file=sys.stderr)
        else:
            named = f", naming {want_named}" if want_named else ""
            print(f"  ok    {label} -> exit {code}{named}")
    failures += _parser_arms()
    failures += _conditional_arms()
    failures += _job_condition_arms()
    failures += _clippy_block_arms()
    failures += _workflow_integrity_arms()
    total = (
        len(arms)
        + _PARSER_ARM_COUNT
        + len(_CONDITIONAL_ARMS)
        + len(_JOB_CONDITION_ARMS)
        + len(_CLIPPY_BLOCK_ARMS)
        + len(_WORKFLOW_INTEGRITY_ARMS)
    )
    if failures:
        print(f"workspace test package self-test: {failures} arm(s) failed", file=sys.stderr)
        return 1
    print(f"workspace test package self-test: {total}/{total} arms behaved as stated")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    cargo_args = subparsers.add_parser("cargo-args")
    cargo_args.add_argument(
        "lane",
        choices=list(LANES),
    )
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument(
        "--simulate-missing",
        help="drop one package from the derived tested set to prove the guard fails",
    )
    verify_parser.add_argument(
        "--simulate-unlinted",
        help="drop one package from the scanned linted set to prove the lint guard fails",
    )
    subparsers.add_parser("self-test")
    required_parser = subparsers.add_parser("verify-required-tier")
    required_parser.add_argument(
        "--simulate-dropped-lane",
        help="ignore required-job steps that run this lane, to prove the guard fails",
    )
    integrity_parser = subparsers.add_parser("verify-workflow-integrity")
    integrity_parser.add_argument(
        "--simulate-deleted-job",
        help="drop one job from what the workflows appear to define, to prove the guard fails",
    )
    args = parser.parse_args()

    if args.command == "verify":
        return verify(args.simulate_missing, args.simulate_unlinted)
    if args.command == "self-test":
        return self_test()
    if args.command == "verify-required-tier":
        return verify_required_tier(args.simulate_dropped_lane)
    if args.command == "verify-workflow-integrity":
        return verify_workflow_integrity(args.simulate_deleted_job)
    # Windows Python writes stdout in text mode and translates "\n" into
    # "\r\n". bash splits command substitution on IFS, which contains newline
    # but not carriage return, so each token would keep a trailing "\r" and
    # cargo would reject the package name. PowerShell strips CRLF when it
    # splits native output into an array, which is why this stayed invisible
    # while the Windows lanes ran pwsh.
    sys.stdout.reconfigure(newline="\n")
    print(package_args(lane_packages(args.lane)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
