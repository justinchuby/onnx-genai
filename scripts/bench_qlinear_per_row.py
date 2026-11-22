#!/usr/bin/env python3
"""Validate/analyze protocol-1 serial per-row requantization diagnostic records.

Producer: kernels::qlinear_matmul::tests::bench_per_row_requantization_neon_ab.
A/A failures remain in the result, with separate CPU/wall qualification flags.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import NoReturn, TextIO

WIDTHS = (7, 8, 9, 17, 65, 257, 2048)
DTYPES = ("Int8", "Uint8")
ROUNDS = 21
U64_MAX = (1 << 64) - 1


def _bad(line: int, reason: str) -> NoReturn:
    raise ValueError(
        f"requantization record {line}: {reason}; use complete protocol-1 output "
        "from the exact named diagnostic, without merging runs"
    )


def _integer(value: str, line: int, field: str, maximum: int) -> int:
    try:
        parsed = int(value)
    except ValueError:
        _bad(line, f"{field}={value!r} is not an integer")
    if not 0 <= parsed <= maximum:
        _bad(line, f"{field}={parsed} is outside 0..{maximum}")
    return parsed


def analyze(records: TextIO) -> dict:
    samples: dict[tuple, tuple] = {}
    witnesses: dict[tuple, int] = {}
    sequence = []
    for line, row in enumerate(csv.reader(records), 1):
        if not row:
            _bad(line, "empty record")
        if len(row) < 4 or row[1] != "1":
            _bad(line, "missing fields or unsupported protocol version")
        n = _integer(row[2], line, "width", max(WIDTHS))
        dtype = row[3]
        if n not in WIDTHS or dtype not in DTYPES:
            _bad(line, f"unknown cell {n}/{dtype}")
        if row[0] == "QRE_SAMPLE" and len(row) == 10:
            repetition = _integer(row[4], line, "round", ROUNDS - 1)
            arm = _integer(row[5], line, "arm", 2)
            iterations = _integer(row[6], line, "iterations", 2_000_000)
            if iterations != max(1, 2_000_000 // n):
                _bad(line, "iteration count disagrees with protocol-1 producer")
            cpu = _integer(row[7], line, "cpu_ns", U64_MAX)
            wall = _integer(row[8], line, "wall_ns", U64_MAX)
            if not cpu or not wall:
                _bad(line, "elapsed CPU and wall time must be positive")
            checksum = _integer(row[9], line, "checksum", U64_MAX)
            key = (n, dtype, repetition, arm)
            if key in samples:
                _bad(line, f"duplicate sample {key}")
            samples[key] = (iterations, cpu, wall, checksum)
            sequence.append(("sample", *key))
        elif row[0] == "QRE_WITNESS" and len(row) == 5:
            key = (n, dtype)
            hits = _integer(row[4], line, "native calls", U64_MAX)
            expected = 0 if n < 8 else 2 * ROUNDS * (2_000_000 // n + 1)
            if key in witnesses or hits != expected:
                _bad(line, f"duplicate or incorrect native-call witness for {key}")
            witnesses[key] = hits
            sequence.append(("witness", *key))
        else:
            _bad(line, "unknown record kind or wrong field count")
    expected_cells = {(n, dtype) for n in WIDTHS for dtype in DTYPES}
    expected_samples = {
        (n, dtype, repetition, arm)
        for n, dtype in expected_cells
        for repetition in range(ROUNDS)
        for arm in range(3)
    }
    if set(samples) != expected_samples or set(witnesses) != expected_cells:
        raise ValueError(
            f"incomplete requantization corpus: {len(samples)}/{len(expected_samples)} "
            f"samples, {len(witnesses)}/{len(expected_cells)} witnesses; rerun the "
            "exact named diagnostic and preserve every record"
        )
    expected_sequence = []
    for n in WIDTHS:
        for dtype in DTYPES:
            expected_sequence.extend(
                ("sample", n, dtype, repetition, (repetition + offset) % 3)
                for repetition in range(ROUNDS)
                for offset in range(3)
            )
            expected_sequence.append(("witness", n, dtype))
    if sequence != expected_sequence:
        raise ValueError(
            "requantization record order does not match protocol-1 rotating arms; "
            "preserve the producer's original record order"
        )
    cells = []
    for n in WIDTHS:
        for dtype in DTYPES:
            rows = [
                samples[n, dtype, r, arm] for r in range(ROUNDS) for arm in range(3)
            ]
            if len({row[3] for row in rows}) != 1:
                raise ValueError(
                    f"output checksum mismatch in {n}/{dtype}; investigate correctness"
                )
            cell = {"n": n, "dtype": dtype, "native_calls": witnesses[n, dtype]}
            for metric, index in [("cpu", 1), ("wall", 2)]:
                medians = [
                    statistics.median(
                        samples[n, dtype, r, arm][index]
                        / (samples[n, dtype, r, arm][0] * n)
                        for r in range(ROUNDS)
                    )
                    for arm in range(3)
                ]
                aa = max(medians[1] / medians[2], medians[2] / medians[1])
                cell[metric] = {
                    "median_ns_per_element": medians,
                    "native_aa_ratio": aa,
                    "qualified": aa <= 1.05,
                    "portable_over_native": medians[0] / statistics.mean(medians[1:]),
                }
            cells.append(cell)
    return {
        "protocol": 1,
        "sample_count": len(samples),
        "cell_count": len(cells),
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    args = parser.parse_args()
    try:
        with args.records.open(newline="") as records:
            result = analyze(records)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
