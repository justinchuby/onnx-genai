#!/usr/bin/env python3
"""Validate protocol-1 real Apple NBits KN/NT execute diagnostics.

Keep failed A/A cells visible, qualified independently for CPU and wall clocks.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import NoReturn, TextIO

SHAPES = (
    (2, 33, 3),
    (8, 257, 65),
    (32, 1024, 1024),
    (128, 1024, 1024),
    (16, 2048, 2048),
)
ORDERS = ((0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1), (2, 1, 0), (1, 0, 2))
ROUNDS = 24
U64_MAX = (1 << 64) - 1


def iterations(shape: tuple[int, int, int]) -> int:
    m, k, n = shape
    return min(2048, max(4, 8_000_000 // (m * k * n)))


def analyze(records: TextIO) -> dict:
    samples, witnesses, sequence = {}, {}, []

    def bad(line: int, reason: str) -> NoReturn:
        raise ValueError(
            f"Apple execute record {line}: {reason}; preserve complete protocol-1 "
            "output from one exact named diagnostic, without merging runs"
        )

    def integer(value: str, line: int, field: str, maximum: int = U64_MAX) -> int:
        try:
            parsed = int(value)
        except ValueError:
            bad(line, f"{field}={value!r} is not an integer")
        if not 0 <= parsed <= maximum:
            bad(line, f"{field}={parsed} is outside 0..{maximum}")
        return parsed

    for line, row in enumerate(csv.reader(records), 1):
        if len(row) < 6 or row[1] != "1":
            bad(line, "missing fields or unsupported protocol version")
        shape = tuple(integer(row[i], line, "shape", 2048) for i in (2, 3, 4))
        retained = integer(row[5], line, "retention", 1)
        if shape not in SHAPES:
            bad(line, f"unknown shape {shape}")
        cell = (*shape, retained)
        count = iterations(shape)
        if row[0] == "ANT_SAMPLE" and len(row) == 12:
            round_ = integer(row[6], line, "round", ROUNDS - 1)
            arm = integer(row[7], line, "arm", 2)
            if integer(row[8], line, "iterations", 2048) != count:
                bad(line, "iteration count disagrees with the producer")
            cpu = integer(row[9], line, "cpu_ns")
            wall = integer(row[10], line, "wall_ns")
            checksum = integer(row[11], line, "checksum")
            if not cpu or not wall:
                bad(line, "elapsed CPU and wall time must be positive")
            key = (*cell, round_, arm)
            if key in samples:
                bad(line, f"duplicate sample {key}")
            samples[key] = (cpu, wall, checksum)
            sequence.append(("sample", *key))
        elif row[0] == "ANT_WITNESS" and len(row) == 10:
            observed = tuple(integer(v, line, "witness") for v in row[6:])
            _, k, n = shape
            expected = (
                ROUNDS * (count + 1),
                1 if retained else 2 * ROUNDS * (count + 1),
                0,
                k * n * 4 if retained else 0,
            )
            if cell in witnesses or observed != expected:
                bad(line, f"duplicate or incorrect builds/Host witness for {cell}")
            witnesses[cell] = observed
            sequence.append(("witness", *cell))
        else:
            bad(line, "unknown record kind or wrong field count")
    expected_sequence = []
    for shape in SHAPES:
        for retained in (0, 1):
            for round_ in range(ROUNDS):
                for arm in ORDERS[round_ % len(ORDERS)]:
                    expected_sequence.append(("sample", *shape, retained, round_, arm))
            expected_sequence.append(("witness", *shape, retained))
    if sequence != expected_sequence:
        raise ValueError(
            "incomplete or reordered Apple execute corpus; preserve all 720 samples "
            "and 10 witnesses in the producer's six-order sequence"
        )
    cells = []
    for shape in SHAPES:
        for retained in (0, 1):
            key = (*shape, retained)
            rows = [
                samples[(*key, round_, arm)]
                for round_ in range(ROUNDS)
                for arm in range(3)
            ]
            if len({row[2] for row in rows}) != 1:
                raise ValueError(
                    f"output checksum mismatch for {key}; investigate numerical correctness"
                )
            cell = {
                "m": shape[0],
                "k": shape[1],
                "n": shape[2],
                "retained": retained,
                "builds_and_host_bytes": witnesses[key],
            }
            for metric, index in (("cpu", 0), ("wall", 1)):
                medians = [
                    statistics.median(
                        samples[(*key, round_, arm)][index] / iterations(shape)
                        for round_ in range(ROUNDS)
                    )
                    for arm in range(3)
                ]
                aa = max(medians[1] / medians[2], medians[2] / medians[1])
                cell[metric] = {
                    "median_ns_per_call": medians,
                    "native_aa_ratio": aa,
                    "qualified": aa <= 1.05,
                    "legacy_over_native": medians[0] / statistics.mean(medians[1:]),
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
