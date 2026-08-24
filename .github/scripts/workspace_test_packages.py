#!/usr/bin/env python3
"""Derive CI cargo test package sets from the Cargo workspace."""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
from collections.abc import Iterable
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

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
        case _:
            raise SystemExit(f"unknown lane '{lane}'")
    return sorted(selected)


def verify(simulate_missing: str | None = None) -> int:
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
    if uncovered or stale_tested or stale_denied:
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
        return 1
    print(
        f"workspace test package coverage ok: {len(tested)} tested, {len(denied)} denied"
    )
    return 0


# A `cargo test`/`cargo llvm-cov` invocation is the only thing that counts as
# executing a package's tests. `cargo build` and `cargo clippy --all-targets`
# compile the same test targets but never run an assertion, so they must not
# satisfy this gate.
_RUNS_TESTS = re.compile(r"cargo\s+(?:\+\S+\s+)?(?:test|llvm-cov)\b")
_LANE_CALL = re.compile(r"workspace_test_packages\.py\s+cargo-args\s+([a-z-]+)")
_PACKAGE_FLAG = re.compile(r"(?:^|\s)(?:-p|--package)[\s=]+([A-Za-z0-9_-]+)")


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
            body.append(line)
    flush()
    if not jobs:
        raise SystemExit(f"{source}: parsed no jobs; the gate would check nothing")
    return jobs


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
    """Packages whose tests the given shell commands actually execute."""
    tested: set[str] = set()
    for command in commands:
        for fragment in re.split(r"&&|\|\||;|\n", command):
            if not _RUNS_TESTS.search(fragment):
                continue
            for lane in _LANE_CALL.findall(fragment):
                tested.update(lane_packages(lane))
            tested.update(_PACKAGE_FLAG.findall(fragment))
    return tested


def required_lane_commands(skip_lane: str | None = None) -> tuple[list[str], set[str]]:
    jobs = workflow_jobs()
    commands: list[str] = []
    for name in sorted(REQUIRED_JOB_NAMES & set(jobs)):
        for block in run_blocks(jobs[name]):
            if skip_lane and re.search(rf"cargo-args\s+{re.escape(skip_lane)}\b", block):
                continue
            commands.append(block)
    return commands, set(jobs)


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
    return 0


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


_PARSER_ARM_COUNT = 5


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
    return failures


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
        ("verify-required-tier (control)", verify_required_tier, {}, 0, []),
        (
            "verify-required-tier --simulate-dropped-lane ort-backed",
            verify_required_tier,
            {"simulate_dropped_lane": "ort-backed"},
            1,
            sorted(ORT_BACKED),
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
    parser_failures = _parser_arms()
    failures += parser_failures
    total = len(arms) + _PARSER_ARM_COUNT
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
        choices=["offline-linux", "offline-cross-platform", "ort-backed", "linux-only"],
    )
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument(
        "--simulate-missing",
        help="drop one package from the derived tested set to prove the guard fails",
    )
    subparsers.add_parser("self-test")
    required_parser = subparsers.add_parser("verify-required-tier")
    required_parser.add_argument(
        "--simulate-dropped-lane",
        help="ignore required-job steps that run this lane, to prove the guard fails",
    )
    args = parser.parse_args()

    if args.command == "verify":
        return verify(args.simulate_missing)
    if args.command == "self-test":
        return self_test()
    if args.command == "verify-required-tier":
        return verify_required_tier(args.simulate_dropped_lane)
    print(package_args(lane_packages(args.lane)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
