#!/usr/bin/env python3
"""Verify CUDA integration tests cannot silently pass on CPU-only runners."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CUDA_CRATE = ROOT / "crates" / "onnx-runtime-ep-cuda"
TESTS = CUDA_CRATE / "tests"
SUMMARY = re.compile(
    r"test result: ok\. (?P<passed>\d+) passed; (?P<failed>\d+) failed; "
    r"(?P<ignored>\d+) ignored; (?P<measured>\d+) measured; (?P<filtered>\d+) filtered out"
)


@dataclass(frozen=True)
class TestBinary:
    target: str
    executable: Path


@dataclass(frozen=True)
class TargetResult:
    target: str
    inventory: int
    passed: int
    failed: int
    ignored: int


def run(command: list[str | Path]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(part) for part in command],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def build_test_binaries() -> list[TestBinary]:
    result = run(
        [
            "cargo",
            "test",
            "--locked",
            "-p",
            "onnx-runtime-ep-cuda",
            "--features",
            "cuda",
            "--tests",
            "--no-run",
            "--message-format=json",
        ]
    )
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr, end="")
        print(result.stderr, file=sys.stderr, end="")
        raise RuntimeError("cargo test --no-run failed")

    binaries: dict[str, Path] = {}
    for line in result.stdout.splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if message.get("reason") != "compiler-artifact":
            continue
        target = message.get("target", {})
        if "test" not in target.get("kind", []):
            continue
        src_path = Path(target.get("src_path", ""))
        if not src_path.is_absolute():
            src_path = ROOT / src_path
        try:
            src_path.relative_to(TESTS)
        except ValueError:
            continue
        executable = message.get("executable")
        if executable:
            binaries[target["name"]] = Path(executable)

    return [TestBinary(target, binaries[target]) for target in sorted(binaries)]


def list_inventory(binary: TestBinary) -> int:
    result = run([binary.executable, "--list"])
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr, end="")
        print(result.stderr, file=sys.stderr, end="")
        raise RuntimeError(f"{binary.target} --list failed")
    return sum(1 for line in result.stdout.splitlines() if line.endswith(": test"))


def run_ignored_check(binary: TestBinary) -> tuple[int, int, int]:
    result = run([binary.executable, "--color", "never"])
    output = result.stdout + result.stderr
    if result.returncode != 0:
        print(output, file=sys.stderr, end="")
        raise RuntimeError(f"{binary.target} failed while checking ignored status")
    matches = list(SUMMARY.finditer(output))
    if not matches:
        raise RuntimeError(f"{binary.target} did not print a libtest summary")
    summary = matches[-1]
    return (
        int(summary.group("passed")),
        int(summary.group("failed")),
        int(summary.group("ignored")),
    )


def validate_result(result: TargetResult) -> list[str]:
    errors: list[str] = []
    if result.inventory == 0:
        errors.append(f"{result.target}: Cargo reported no integration tests")
    if result.failed:
        errors.append(f"{result.target}: {result.failed} tests failed while checking ignored status")
    if result.passed:
        errors.append(
            f"{result.target}: {result.passed} tests executed without gpu-tests; CUDA tests must be ignored, not pass"
        )
    if result.ignored != result.inventory:
        errors.append(
            f"{result.target}: Cargo inventory has {result.inventory} tests but libtest reported {result.ignored} ignored"
        )
    return errors


def self_test() -> None:
    good = TargetResult("fixture_good", inventory=2, passed=0, failed=0, ignored=2)
    silent_skip = TargetResult("fixture_silent_skip", inventory=1, passed=1, failed=0, ignored=0)
    drift = TargetResult("fixture_drift", inventory=3, passed=0, failed=0, ignored=2)
    if validate_result(good):
        raise AssertionError("good fixture should pass")
    if not any("executed without gpu-tests" in error for error in validate_result(silent_skip)):
        raise AssertionError("silent-skip fixture should fail when a test passes instead of being ignored")
    if not any("inventory has 3 tests" in error for error in validate_result(drift)):
        raise AssertionError("inventory drift fixture should fail")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", help="run parser fixtures only")
    args = parser.parse_args()

    self_test()
    if args.self_test:
        print("CUDA honesty checker self-test passed")
        return 0

    cargo_toml = (CUDA_CRATE / "Cargo.toml").read_text(encoding="utf-8")
    errors: list[str] = []
    if "gpu-tests = []" not in cargo_toml:
        errors.append("crates/onnx-runtime-ep-cuda/Cargo.toml must define a gpu-tests feature")

    binaries = build_test_binaries()
    results: list[TargetResult] = []
    for binary in binaries:
        inventory = list_inventory(binary)
        passed, failed, ignored = run_ignored_check(binary)
        target_result = TargetResult(binary.target, inventory, passed, failed, ignored)
        results.append(target_result)
        errors.extend(validate_result(target_result))

    total_inventory = sum(result.inventory for result in results)
    total_ignored = sum(result.ignored for result in results)
    if total_inventory == 0:
        errors.append("Cargo reported no CUDA integration tests")

    if errors:
        print("CUDA test honesty check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(
        f"CUDA test honesty check passed for {total_inventory} Cargo-discovered integration tests "
        f"across {len(results)} targets ({total_ignored} ignored without gpu-tests)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
