#!/usr/bin/env python3
"""Verify CUDA integration tests cannot silently pass on CPU-only runners."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CUDA_CRATE = ROOT / "crates" / "onnx-runtime-ep-cuda"
TESTS = CUDA_CRATE / "tests"
REQUIRED_ATTR_PARTS = (
    "cfg_attr",
    "not(feature = \"gpu-tests\")",
    "ignore = \"requires CUDA device; enable the gpu-tests feature on a CUDA runner\"",
)


def main() -> int:
    errors: list[str] = []
    cargo_toml = (CUDA_CRATE / "Cargo.toml").read_text(encoding="utf-8")
    if "gpu-tests = []" not in cargo_toml:
        errors.append("crates/onnx-runtime-ep-cuda/Cargo.toml must define a gpu-tests feature")

    test_count = 0
    for path in sorted(TESTS.rglob("*.rs")):
        rel = path.relative_to(ROOT)
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()

        for index, line in enumerate(lines):
            if line.strip() != "#[test]":
                continue
            test_count += 1
            attrs = " ".join(line.strip() for line in lines[max(0, index - 8) : index + 3])
            has_gpu_gate = all(part in attrs for part in REQUIRED_ATTR_PARTS)
            has_unconditional_ignore = "#[ignore" in attrs
            if not (has_gpu_gate or has_unconditional_ignore):
                errors.append(f"{rel}:{index + 1}: CUDA test missing visible ignore gate")

        if re.search(r"else\s*\{\s*return\s*(?:;|\})", text):
            errors.append(f"{rel}: contains a let-else early return that would pass without running CUDA")
        if re.search(r"(?m)^\s*return\s*;", text):
            errors.append(f"{rel}: contains a bare early return that would pass without running CUDA")
        if re.search(r"return\s*\(0\.0,\s*0\)", text):
            errors.append(f"{rel}: contains a sentinel return that would pass without running CUDA")

    if test_count == 0:
        errors.append("no CUDA integration tests were found")

    if errors:
        print("CUDA test honesty check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(f"CUDA test honesty check passed for {test_count} integration tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
