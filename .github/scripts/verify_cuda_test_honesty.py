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
# Targets that must run in every configuration, and so cannot be held to the
# "ignored, not passed" rule the rest of the suite is checked against.
#
# `suite_canary_gpu` is the test that exists because the rest of the suite can
# skip silently. It is a no-op unless `NXRT_REQUIRE_CUDA` says a GPU is meant to
# be present, and where that is set it fails loudly. Giving it the `gpu-tests`
# ignore would remove it from exactly the runs it was written to police -- a
# CPU-only machine that believes it tested a GPU.
#
# `capture_sync_contract` is a static source audit: it reads the CUDA kernel
# sources and fails if any capture-eligible path reaches an unreviewed
# unconditional `synchronize()`. It touches no device, so it MUST run on the
# CPU-only lane and legitimately *passes* there -- holding it to the
# "ignored, not passed" GPU rule would silence exactly the check that keeps a
# capture-unsafe sync from landing. It is another "checking the checker" case,
# not a GPU test that happens to be awkward. It still runs on the CPU lane via
# the dedicated `cargo test ... --test capture_sync_contract` CI step.
#
# `dummy_fill_and_crossover` is a pure-CPU design probe for the #759 dummy-page
# VMM KV scheme: it proves the correctness-safe dummy fill value (zeros, never
# NaN) from additive-masking algebra and derives the fixed-stride+dummy vs
# bucket-growth memory crossover from real model KV geometry. It issues no CUDA
# calls and touches no device (unlike every `*_gpu` sibling in the same crate),
# so it legitimately *passes* on the CPU-only lane and cannot honor the
# "ignored, not passed" GPU rule -- it is a CPU probe, not a GPU test. Its name
# deliberately omits the `_gpu` suffix its device-bound siblings carry.
#
# `matmul_nbits_marlin_oracle` is the pure-CPU half of the int4 `MatMulNBits`
# parity gate: it self-checks the `f64` dequant->GEMM oracle against an
# independent `f32` reference and validates the justified tolerance envelope
# (`Envelope`/`ParityReport`) the GPU gate depends on. It shares the device-free
# machinery in `tests/marlin_numerics/mod.rs` with the GPU target
# (`matmul_nbits_marlin_numerics`) but issues no CUDA calls and touches no
# device, so it legitimately *passes* on the CPU-only lane and cannot honor the
# "ignored, not passed" GPU rule -- it is a CPU probe validating the checker's
# own numeric ground truth, not a GPU test. It was split out of the GPU numerics
# target precisely so that target stays purely-CUDA (all tests ignored without
# `gpu-tests`); its name deliberately omits the `_gpu`/`_numerics` GPU suffixes.
# It still runs on the CPU lane via a dedicated
# `cargo test ... --test matmul_nbits_marlin_oracle` CI step (#1177).
#
# `deferred_release_queue`, `vmm_release_quarantine` and
# `no_built_in_eager_allocator` are the #1186 memory refactor's CPU-side
# probes, and they are the same case as `dummy_fill_and_crossover` above rather
# than three more awkward GPU tests.
#
# The first two exist because the rules they check are state machines, not
# driver calls: the deferred release queue's ordering, bounding, exactly-once
# execution and device-loss behaviour are expressed over the `ReleaseFence` and
# `DeferredReleaseAction` contracts, and the rule deciding whether a partially
# released VMM address may be reused lives in `onnx_runtime_cuda_memory::release`
# with no CUDA symbol in it. Both are driven by fakes -- a scripted fence, a
# scripted driver that fails the Nth `cuMemUnmap` on demand -- so they issue no
# CUDA calls and legitimately pass on a CPU host. #636 is the reason they were
# written this way: it measured 44 tests silently skipped for months because the
# rules had been left inside `*_gpu.rs`. Ignoring them here would put them back
# in exactly that position.
#
# `no_built_in_eager_allocator` is a static source audit like
# `capture_sync_contract`: it reads the two crates' sources and pins the exact
# set of eager `malloc_sync`/`free_sync` sites outside the allocator seam. Its
# whole value is proving a negative that the GPU tests structurally cannot --
# a new eager site on a path they do not exercise would not turn them red -- so
# it must run on the CPU-only lane or it checks nothing at all.
#
# Anything added here needs the same argument: not "this one is awkward" but
# "this one is checking the checker" (or otherwise a genuine CPU-only probe that
# issues no CUDA calls).
ALWAYS_RUN = frozenset(
    {
        "suite_canary_gpu",
        "capture_sync_contract",
        "dummy_fill_and_crossover",
        "matmul_nbits_marlin_oracle",
        "deferred_release_queue",
        "vmm_release_quarantine",
        "no_built_in_eager_allocator",
    }
)
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
        if executable:
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


def self_test() -> None:
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
    if parsed != [TestBinary("fixture_gpu", ROOT / "target" / "debug" / "deps" / "fixture_gpu.exe")]:
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
        if "gpu-tests = []" not in manifest:
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
