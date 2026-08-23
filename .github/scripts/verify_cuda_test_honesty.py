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
# Device memory moved to its own crate so that using an allocator does not drag
# in every kernel. Its GPU tests moved with it, and a check naming only the
# execution provider would stop seeing them -- which is precisely the hole this
# script exists to close.
MEMORY_CRATE = ROOT / "crates" / "onnx-runtime-cuda-memory"
CUDA_CRATES = (CUDA_CRATE, MEMORY_CRATE)
TESTS = CUDA_CRATE / "tests"
TEST_DIRS = tuple(crate / "tests" for crate in CUDA_CRATES)
# CUDA integration-test targets conventionally end in `_gpu`. Keep historical
# CUDA targets that predate that convention explicit so a genuinely CPU-only
# target is not policed as a device test merely because it lives in a CUDA
# crate.
CUDA_TARGETS_WITHOUT_SUFFIX = frozenset({"matmul_nbits_marlin_numerics"})

# CUDA-named targets that must run in every configuration, and so cannot be
# held to the "ignored, not passed" rule the rest of the suite is checked
# against.
#
# `suite_canary_gpu` is the test that exists because the rest of the suite can
# skip silently. It is a no-op unless `NXRT_REQUIRE_CUDA` says a GPU is meant to
# be present, and where that is set it fails loudly. Giving it the `gpu-tests`
# ignore would remove it from exactly the runs it was written to police -- a
# CPU-only machine that believes it tested a GPU.
#
ALWAYS_RUN = frozenset({"suite_canary_gpu"})
SUMMARY = re.compile(
    r"test result: (?:ok|FAILED)\. (?P<passed>\d+) passed; (?P<failed>\d+) failed; "
    r"(?P<ignored>\d+) ignored; (?P<measured>\d+) measured; (?P<filtered>\d+) filtered out"
)


@dataclass(frozen=True)
class FeatureConfig:
    name: str
    features: str


@dataclass(frozen=True)
class TestBinary:
    target: str
    executable: Path


@dataclass(frozen=True)
class IgnoredResult:
    target: str
    inventory: int
    passed: int
    failed: int
    ignored: int


@dataclass(frozen=True)
class ActiveResult:
    target: str
    inventory: int
    passed: int
    failed: int
    ignored: int


BASE_CONFIG = FeatureConfig("without-gpu-tests", "cuda")
GPU_CONFIG = FeatureConfig("with-gpu-tests", "cuda,gpu-tests")


def run(command: list[str | Path]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(part) for part in command],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def is_cuda_test_target(target: str) -> bool:
    return (
        target not in ALWAYS_RUN
        and (target.endswith("_gpu") or target in CUDA_TARGETS_WITHOUT_SUFFIX)
    )


def parse_test_binaries_from_json(stdout: str) -> list[TestBinary]:
    binaries: dict[str, Path] = {}
    for line in stdout.splitlines():
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
        if not any(src_path.is_relative_to(tests) for tests in TEST_DIRS):
            continue
        executable = message.get("executable")
        if executable and is_cuda_test_target(target["name"]):
            binaries[target["name"]] = Path(executable)

    return [
        TestBinary(target, binaries[target])
        for target in sorted(binaries)
        if target not in ALWAYS_RUN
    ]


def build_test_binaries(config: FeatureConfig) -> list[TestBinary]:
    result = run(
        [
            "cargo",
            "test",
            "--locked",
            "-p",
            "onnx-runtime-ep-cuda",
            "-p",
            "onnx-runtime-cuda-memory",
            "--features",
            config.features,
            "--tests",
            "--no-run",
            "--message-format=json",
        ]
    )
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr, end="")
        print(result.stderr, file=sys.stderr, end="")
        raise RuntimeError(f"cargo test --no-run failed for {config.name}")
    return parse_test_binaries_from_json(result.stdout)


def list_inventory(binary: TestBinary) -> frozenset[str]:
    result = run([binary.executable, "--list"])
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr, end="")
        print(result.stderr, file=sys.stderr, end="")
        raise RuntimeError(f"{binary.target} --list failed")
    return frozenset(line.removesuffix(": test") for line in result.stdout.splitlines() if line.endswith(": test"))


def run_libtest(binary: TestBinary) -> tuple[int, int, int, int]:
    result = run([binary.executable, "--color", "never"])
    output = result.stdout + result.stderr
    matches = list(SUMMARY.finditer(output))
    if not matches:
        print(output, file=sys.stderr, end="")
        raise RuntimeError(f"{binary.target} did not print a libtest summary")
    summary = matches[-1]
    return (
        result.returncode,
        int(summary.group("passed")),
        int(summary.group("failed")),
        int(summary.group("ignored")),
    )


def collect_inventories(binaries: list[TestBinary]) -> dict[str, frozenset[str]]:
    return {binary.target: list_inventory(binary) for binary in binaries}


def compare_inventories(
    base: dict[str, frozenset[str]], gpu: dict[str, frozenset[str]]
) -> list[str]:
    errors: list[str] = []
    base_targets = set(base)
    gpu_targets = set(gpu)
    for target in sorted(gpu_targets - base_targets):
        errors.append(f"{target}: exists only with gpu-tests enabled; CUDA tests must not hide from CPU inventory")
    for target in sorted(base_targets - gpu_targets):
        errors.append(f"{target}: exists only without gpu-tests enabled; inventories must stay reconciled")
    for target in sorted(base_targets & gpu_targets):
        base_only = base[target] - gpu[target]
        gpu_only = gpu[target] - base[target]
        for test in sorted(gpu_only):
            errors.append(f"{target}::{test}: test exists only with gpu-tests enabled")
        for test in sorted(base_only):
            errors.append(f"{target}::{test}: test exists only without gpu-tests enabled")
    return errors


def validate_ignored_result(result: IgnoredResult) -> list[str]:
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


def validate_active_no_cuda_result(result: ActiveResult) -> list[str]:
    errors: list[str] = []
    if result.inventory == 0:
        errors.append(f"{result.target}: Cargo reported no gpu-tests integration tests")
    if result.passed:
        errors.append(
            f"{result.target}: {result.passed} tests passed with gpu-tests on a no-CUDA host; CUDA tests must fail loud or remain ignored"
        )
    if result.failed + result.ignored != result.inventory:
        errors.append(
            f"{result.target}: Cargo inventory has {result.inventory} tests but active no-CUDA run reported "
            f"{result.failed} failed + {result.ignored} ignored"
        )
    return errors


def declares_gpu_tests_feature(manifest: str) -> bool:
    """True if the manifest defines a `gpu-tests` feature, whatever its value.

    The property this guard cares about is that the feature *exists*, so that
    `--features gpu-tests` is a meaningful flag for the crate. An earlier
    version matched the literal string `gpu-tests = []`, which silently
    conflated "declares the feature" with "declares it with an empty value" and
    so rejected the legitimate forwarding form
    `gpu-tests = ["onnx-runtime-cuda-memory/gpu-tests"]`.

    The key is looked for only inside the `[features]` table. A file-wide match
    would accept a *dependency* named `gpu-tests`, or one under a
    `[target.'cfg(...)'.dependencies]` table -- which is the same mistake in a
    new costume: matching a string that usually co-occurs with the property
    instead of the property itself.
    """
    in_features = False
    for raw_line in manifest.splitlines():
        line = raw_line.strip()
        if line.startswith("["):
            in_features = line.split("#", 1)[0].strip() == "[features]"
            continue
        if in_features and re.match(r"gpu-tests\s*=", line):
            return True
    return False


def self_test() -> None:
    for manifest, expected in (
        ("[features]\ngpu-tests = []\n", True),
        ('[features]\ngpu-tests = ["onnx-runtime-cuda-memory/gpu-tests"]\n', True),
        ("[features]\ngpu-tests = [\n    \"dep/gpu-tests\",\n]\n", True),
        ('[dependencies]\nfoo = "1"\n\n[features]\ngpu-tests = []\n', True),
        ('[features]\ngpu-tests = []\n\n[dependencies]\nfoo = "1"\n', True),
        ("[features] # the table\ngpu-tests = []\n", True),
        ("[features]\ndefault = []\n", False),
        ("[features]\n# gpu-tests = []\n", False),
        ("[features]\ngpu-tests-extra = []\n", False),
        ("[dependencies]\nfoo = { features = [\"gpu-tests\"] }\n", False),
        ('[dependencies]\ngpu-tests = "1.0"\n', False),
        ('[target.\'cfg(unix)\'.dependencies]\ngpu-tests = "1.0"\n', False),
    ):
        if declares_gpu_tests_feature(manifest) is not expected:
            raise AssertionError(f"gpu-tests feature detection wrong for: {manifest!r}")

    for cpu_target in (
        "deferred_release_queue",
        "no_built_in_eager_allocator",
        "vmm_release_quarantine",
    ):
        if is_cuda_test_target(cpu_target):
            raise AssertionError(f"CPU-only target {cpu_target} was classified as CUDA")
    if not is_cuda_test_target("matmul_nbits_marlin_numerics"):
        raise AssertionError("historical CUDA target without _gpu suffix was not classified as CUDA")

    fixture_stdout = "\n".join(
        [
            "not json",
            json.dumps(
                {
                    "reason": "compiler-artifact",
                    "target": {
                        "name": "fixture_gpu",
                        "kind": ["test"],
                        "src_path": str(TESTS / "fixture_gpu.rs"),
                    },
                    "executable": str(ROOT / "target" / "debug" / "deps" / "fixture_gpu.exe"),
                }
            ),
            json.dumps(
                {
                    "reason": "compiler-artifact",
                    "target": {
                        "name": "deferred_release_queue",
                        "kind": ["test"],
                        "src_path": str(TESTS / "deferred_release_queue.rs"),
                    },
                    "executable": str(
                        ROOT / "target" / "debug" / "deps" / "deferred_release_queue.exe"
                    ),
                }
            ),
            json.dumps(
                {
                    "reason": "compiler-artifact",
                    "target": {
                        "name": "matmul_nbits_marlin_numerics",
                        "kind": ["test"],
                        "src_path": str(TESTS / "matmul_nbits_marlin_numerics.rs"),
                    },
                    "executable": str(
                        ROOT / "target" / "debug" / "deps" / "matmul_nbits_marlin_numerics.exe"
                    ),
                }
            ),
            json.dumps(
                {
                    "reason": "compiler-artifact",
                    "target": {
                        "name": "onnx_runtime_ep_cuda",
                        "kind": ["lib"],
                        "src_path": str(CUDA_CRATE / "src" / "lib.rs"),
                    },
                    "executable": str(ROOT / "target" / "debug" / "deps" / "lib.exe"),
                }
            ),
        ]
    )
    parsed = parse_test_binaries_from_json(fixture_stdout)
    expected = [
        TestBinary("fixture_gpu", ROOT / "target" / "debug" / "deps" / "fixture_gpu.exe"),
        TestBinary(
            "matmul_nbits_marlin_numerics",
            ROOT / "target" / "debug" / "deps" / "matmul_nbits_marlin_numerics.exe",
        ),
    ]
    if parsed != expected:
        raise AssertionError(f"JSON parser fixture returned {parsed!r}")

    if compare_inventories({"target": frozenset({"a"})}, {"target": frozenset({"a"})}):
        raise AssertionError("matching inventories should pass")
    hidden = compare_inventories({"target": frozenset({"a"})}, {"target": frozenset({"a", "gpu_only"})})
    if not any("gpu_only" in error for error in hidden):
        raise AssertionError("gpu-tests-only inventory drift should be caught")

    good_ignored = IgnoredResult("fixture_good", inventory=2, passed=0, failed=0, ignored=2)
    silent_without_feature = IgnoredResult("fixture_silent_skip", inventory=1, passed=1, failed=0, ignored=0)
    if validate_ignored_result(good_ignored):
        raise AssertionError("good ignored fixture should pass")
    if not any("executed without gpu-tests" in error for error in validate_ignored_result(silent_without_feature)):
        raise AssertionError("without-gpu-tests silent pass should fail")

    good_active = ActiveResult("fixture_active", inventory=2, passed=0, failed=2, ignored=0)
    silent_with_feature = ActiveResult("fixture_active_silent", inventory=1, passed=1, failed=0, ignored=0)
    if validate_active_no_cuda_result(good_active):
        raise AssertionError("active fail-loud fixture should pass")
    if not any("passed with gpu-tests" in error for error in validate_active_no_cuda_result(silent_with_feature)):
        raise AssertionError("gpu-tests-enabled silent pass should fail")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", help="run parser fixtures only")
    args = parser.parse_args()

    self_test()
    if args.self_test:
        print("CUDA honesty checker self-test passed")
        return 0

    errors: list[str] = []
    for crate in CUDA_CRATES:
        manifest = (crate / "Cargo.toml").read_text(encoding="utf-8")
        if not declares_gpu_tests_feature(manifest):
            errors.append(f"crates/{crate.name}/Cargo.toml must define a gpu-tests feature")

    base_binaries = build_test_binaries(BASE_CONFIG)
    gpu_binaries = build_test_binaries(GPU_CONFIG)
    base_inventory = collect_inventories(base_binaries)
    gpu_inventory = collect_inventories(gpu_binaries)
    errors.extend(compare_inventories(base_inventory, gpu_inventory))

    base_by_target = {binary.target: binary for binary in base_binaries}
    gpu_by_target = {binary.target: binary for binary in gpu_binaries}

    ignored_results: list[IgnoredResult] = []
    for target, inventory in sorted(base_inventory.items()):
        _, passed, failed, ignored = run_libtest(base_by_target[target])
        result = IgnoredResult(target, len(inventory), passed, failed, ignored)
        ignored_results.append(result)
        errors.extend(validate_ignored_result(result))

    active_results: list[ActiveResult] = []
    for target, inventory in sorted(gpu_inventory.items()):
        _, passed, failed, ignored = run_libtest(gpu_by_target[target])
        result = ActiveResult(target, len(inventory), passed, failed, ignored)
        active_results.append(result)
        errors.extend(validate_active_no_cuda_result(result))

    total_base_inventory = sum(len(inventory) for inventory in base_inventory.values())
    total_gpu_inventory = sum(len(inventory) for inventory in gpu_inventory.values())
    total_ignored = sum(result.ignored for result in ignored_results)
    total_active_failed = sum(result.failed for result in active_results)
    total_active_ignored = sum(result.ignored for result in active_results)
    if total_base_inventory == 0 or total_gpu_inventory == 0:
        errors.append("Cargo reported no CUDA integration tests")

    if errors:
        print("CUDA test honesty check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(
        "CUDA test honesty check passed: "
        f"{total_base_inventory} tests/{len(base_inventory)} targets without gpu-tests "
        f"({total_ignored} ignored), {total_gpu_inventory} tests/{len(gpu_inventory)} targets with gpu-tests "
        f"({total_active_failed} fail-loud, {total_active_ignored} ignored, 0 passed on this no-CUDA host)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
