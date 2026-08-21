#!/usr/bin/env python3
"""Check that `cargo` commands written down in the repo still resolve.

Documented commands rot silently. A renamed feature, a deleted binary or a
helper that never existed at all will sit in a script or a runbook indefinitely,
because nothing executes it and nothing checks it. When someone eventually
copies the line, the failure looks like a problem with their environment rather
than with the line.

This validates the parts of a `cargo` invocation that can be checked without
running it, against `cargo metadata`: the package exists, and any `--bin`,
`--example`, `--test`, `--bench` and `--features` it names exist in that
package. It deliberately does not try to check anything semantic.

Historical records are skipped: dated files under `docs/` and everything in
`.squad/` state what was run at the time and must not be rewritten to match the
present tree.

Usage:
    python3 scripts/check_documented_commands.py [--verbose]

Exits non-zero if any documented command names something that does not exist.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

# Markdown and prose wrap these tokens in punctuation: `cargo test -p foo`,
# "...-p foo, which...". Strip anything that cannot appear in a cargo name.
TRAILING = "`\"',.:;)]}>*_"
COMMAND_RE = re.compile(r"cargo\s+(?:\+\S+\s+)?(?:build|run|test|check|clippy|bench)\b(.*)")
TEMPLATED = ("${", "$(", "{{", "<pkg>", "<package>", "<name>")
TARGET_FLAGS = {"--bin": "bins", "--example": "examples", "--test": "tests", "--bench": "benches"}


def workspace_packages() -> dict[str, dict[str, set[str]]]:
    raw = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    packages = {}
    for package in json.loads(raw)["packages"]:
        targets = package["targets"]
        packages[package["name"]] = {
            "features": set(package["features"]),
            "bins": {t["name"] for t in targets if "bin" in t["kind"]},
            "examples": {t["name"] for t in targets if "example" in t["kind"]},
            "tests": {t["name"] for t in targets if "test" in t["kind"]},
            "benches": {t["name"] for t in targets if "bench" in t["kind"]},
        }
    return packages


def is_historical(path: str) -> bool:
    return ".squad/" in path or bool(re.search(r"/(?:19|20)\d{2}-\d{2}(?:-\d{2})?", path))


def flag_values(tokens: list[str], flag: str) -> list[str]:
    values = []
    for index, token in enumerate(tokens):
        if token == flag and index + 1 < len(tokens):
            candidate = tokens[index + 1]
            if not candidate.startswith("-"):
                values.append(candidate)
        elif token.startswith(flag + "="):
            values.append(token.split("=", 1)[1])
    return values


def check_line(path: str, lineno: int, line: str, packages) -> list[str]:
    match = COMMAND_RE.search(line)
    if not match:
        return []
    tail = match.group(1)
    if any(marker in tail for marker in TEMPLATED):
        return []
    try:
        tokens = shlex.split(tail.replace("\\", " "), posix=True)
    except ValueError:
        return []
    if "--" in tokens:
        tokens = tokens[: tokens.index("--")]
    tokens = [token.strip(TRAILING) for token in tokens]

    names = flag_values(tokens, "-p") + flag_values(tokens, "--package")
    if not names:
        return []
    name = names[0]
    if name not in packages:
        # Prose often mentions crates that are not workspace members (upstream
        # ones, or planned ones). Only a name that looks like ours is a finding.
        return [f"unknown package `{name}`"] if name.startswith(("onnx-", "nxrt-")) else []

    package = packages[name]
    problems = []
    for flag, kind in TARGET_FLAGS.items():
        for target in flag_values(tokens, flag):
            if target not in package[kind]:
                have = ", ".join(sorted(package[kind])) or "none"
                problems.append(f"`{name}` has no {flag[2:]} `{target}` (has: {have})")
    for spec in flag_values(tokens, "--features") + flag_values(tokens, "-F"):
        for feature in re.split(r"[,\s]+", spec.strip()):
            if not feature:
                continue
            if "/" in feature:
                dependency, sub = feature.split("/", 1)
                if dependency in packages and sub not in packages[dependency]["features"]:
                    problems.append(f"`{dependency}` has no feature `{sub}`")
            elif feature not in package["features"]:
                have = ", ".join(sorted(package["features"]))
                problems.append(f"`{name}` has no feature `{feature}` (has: {have})")
    return problems


def self_test(packages) -> int:
    """Prove the check can still detect, rather than passing because it is inert.

    A checker that has quietly stopped matching anything is indistinguishable
    from a clean tree, and this one matches on a regex over prose.
    """
    cases = [
        ("cargo build -p onnx-genai-cli --features no-such-feature", "no feature"),
        ("cargo build -p onnx-genai-bench --bin no_such_bin", "no bin"),
        ("cargo test -p onnx-genai-nonexistent", "unknown package"),
        ("cargo bench -p onnx-runtime-ep-cpu --bench no_such_bench", "no bench"),
        ("cargo test -p onnx-genai-engine --features onnx-genai-ort/no-such", "no feature"),
    ]
    failures = 0
    for line, expected in cases:
        problems = check_line("<self-test>", 1, line, packages)
        if not any(expected in problem for problem in problems):
            print(f"self-test FAILED to detect: {line}\n  got: {problems}")
            failures += 1
    # A command that resolves must stay silent, or the check would be noise.
    clean = "cargo build --release -p onnx-genai-cli --features native-cuda --bin onnx-genai"
    if check_line("<self-test>", 1, clean, packages):
        print(f"self-test flagged a valid command: {clean}")
        failures += 1
    if failures:
        return 1
    print(f"documented-command check self-test: {len(cases) + 1} case(s) passed")
    return 0


def main() -> int:
    verbose = "--verbose" in sys.argv
    packages = workspace_packages()
    if "--self-test" in sys.argv:
        return self_test(packages)
    tracked = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout.split()

    findings, checked = [], 0
    for path in tracked:
        if not re.search(r"\.(sh|py|ps1|yml|yaml|md)$", path) or is_historical(path):
            continue
        try:
            text = Path(path).read_text(errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            problems = check_line(path, lineno, line, packages)
            if COMMAND_RE.search(line):
                checked += 1
            for problem in problems:
                findings.append((path, lineno, problem, line.strip()))

    if verbose or findings:
        print(f"checked {checked} documented cargo commands")
    for path, lineno, problem, line in findings:
        print(f"\n{path}:{lineno}\n  {problem}\n  {line[:140]}")
    if findings:
        print(f"\n{len(findings)} documented command(s) name something that does not exist.")
        return 1
    if verbose:
        print("all resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
