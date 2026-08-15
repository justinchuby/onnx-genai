#!/usr/bin/env python3
"""Derive CI cargo test package sets from the Cargo workspace."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

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
    args = parser.parse_args()

    if args.command == "verify":
        return verify(args.simulate_missing)
    print(package_args(lane_packages(args.lane)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
