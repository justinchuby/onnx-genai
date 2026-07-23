#!/usr/bin/env python3
"""Run real-weight greedy CUDA parity against committed token goldens.

Build first:
  source .cudaenv.sh
  cargo build --release -p onnx-genai-bench \
    --features bench-native,cuda --bin profile_native

Then run:
  python3 scripts/check_native_ort_parity.py

Use --model repeatedly to select cases and --gpu to choose an idle GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


TOKEN_RE = re.compile(r"^generated_token_ids: (\[.*\])$", re.MULTILINE)


def first_divergence(left: list[int], right: list[int]) -> int | None:
    return next((i for i, pair in enumerate(zip(left, right)) if pair[0] != pair[1]), None)


def run_backend(
    binary: Path,
    model: Path,
    backend: str,
    prompt: str,
    tokens: int,
    gpu: int,
) -> list[int]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["ONNX_GENAI_CUDA_GRAPH"] = "0"
    command = [
        str(binary),
        "--model",
        str(model),
        "--ep",
        "cuda",
        "--backend",
        backend,
        "--tokens",
        str(tokens),
        "--steady",
        "--warmups",
        "0",
        "--runs",
        "1",
        "--prompt",
        prompt,
    ]
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )
    match = TOKEN_RE.search(completed.stdout)
    if match is None:
        raise RuntimeError(f"{backend} output omitted generated_token_ids:\n{completed.stdout}")
    return json.loads(match.group(1))


def expected_tokens(case: dict[str, object], backend: str) -> list[int]:
    expected = case[f"expected_{backend}"]
    if expected == "same_as_native":
        expected = case["expected_native"]
    assert isinstance(expected, list)
    return expected


def check_case(
    case: dict[str, object],
    binary: Path,
    model_root: Path,
    gpu: int,
    prompt: str,
    tokens: int,
) -> None:
    key = str(case["key"])
    model = model_root / str(case["model_path"])
    actual = {
        backend: run_backend(binary, model, backend, prompt, tokens, gpu)
        for backend in ("native", "ort")
    }
    for backend in ("native", "ort"):
        expected = expected_tokens(case, backend)
        if actual[backend] != expected:
            divergence = first_divergence(actual[backend], expected)
            raise AssertionError(
                f"{key} {backend} differs from its golden at index {divergence}"
            )

    divergence = first_divergence(actual["native"], actual["ort"])
    if case["classification"] == "exact":
        if divergence is not None:
            raise AssertionError(f"{key} must be exact; first divergence is {divergence}")
    else:
        minimum = int(case["minimum_common_prefix"])
        common_prefix = tokens if divergence is None else divergence
        if common_prefix < minimum:
            raise AssertionError(
                f"{key} diverged at {common_prefix}, earlier than locked minimum {minimum}"
            )
        oracle_index = int(case["oracle_index"])
        oracle_token = int(case["exact_q4_f32_oracle_token"])
        if actual["native"][oracle_index] != oracle_token:
            raise AssertionError(
                f"{key} native token {actual['native'][oracle_index]} at {oracle_index} "
                f"does not match exact-Q4 f32 oracle {oracle_token}"
            )

    aligned = sum(a == b for a, b in zip(actual["native"], actual["ort"]))
    prefix = tokens if divergence is None else divergence
    print(f"PASS {key}: common_prefix={prefix}/{tokens}, aligned={aligned}/{tokens}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path("tests/parity/native_ort_cuda_golden.json"),
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("target/release/profile_native"),
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("~/.foundry/cache/models").expanduser(),
    )
    parser.add_argument("--model", action="append", help="case key; repeat to select cases")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    golden = json.loads(args.golden.read_text())
    selected = set(args.model or [])
    cases = [
        case
        for case in golden["models"]
        if not selected or str(case["key"]) in selected
    ]
    unknown = selected - {str(case["key"]) for case in cases}
    if unknown:
        parser.error(f"unknown model key(s): {', '.join(sorted(unknown))}")
    if not args.binary.exists():
        parser.error(f"benchmark binary not found: {args.binary}")

    for case in cases:
        check_case(
            case,
            args.binary,
            args.model_root,
            args.gpu,
            golden["prompt"],
            int(golden["tokens"]),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
