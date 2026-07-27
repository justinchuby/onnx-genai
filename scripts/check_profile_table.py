#!/usr/bin/env python3
"""Check the profile README's summary table against the committed samples.

The table in `examples/profiles/README.md` is prose about files that sit next
to it, so it can go stale silently: regenerating the samples does not touch the
table, and a reader has no way to tell which one is current. That is not
hypothetical -- the table once reported a decode throughput from before a
fusion change while the sample beside it reported the number after, and the
narrative drew a conclusion from the stale figure.

This reads both and fails if they disagree, so the samples stay the source of
truth for anything said about them.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROFILES = REPO / "examples" / "profiles"
README = PROFILES / "README.md"

# Table column -> sample file.
COLUMNS = {
    "ORT+CPU": "qwen2.5-0.5b-cpu.txt",
    "ORT+CPU f16": "qwen2.5-0.5b-f16-cpu.txt",
    "native": "qwen2.5-0.5b-native.txt",
    "native f16": "qwen2.5-0.5b-f16-native.txt",
    "ORT+Metal": "qwen2.5-0.5b-metal.txt",
    "native+MLX": "qwen2.5-0.5b-native-mlx.txt",
}

# Table row label -> (label in the sample, unit, decimal places in the table).
ROWS = {
    "model load": ("model load", "ms", 0),
    "time to first token": ("time to first token", "ms", 0),
    "decode throughput": ("decode throughput", "tok/s", 1),
    "end-to-end": ("end-to-end throughput", "tok/s", 1),
}


def sample_values(path: Path) -> dict[str, float]:
    """Every `label   <number> <unit>` line in a --profile report."""
    values: dict[str, float] = {}
    for line in path.read_text().splitlines():
        match = re.match(r"^(\S.*?)\s{2,}([0-9]+(?:\.[0-9]+)?)\s*(ms|tok/s)\s*$", line)
        if match:
            values.setdefault(match.group(1).strip(), float(match.group(2)))
    return values


def table_rows() -> dict[str, list[str]]:
    """The fenced summary table under '## What the samples show'."""
    text = README.read_text()
    section = text.split("## What the samples show", 1)
    if len(section) != 2:
        sys.exit(f"{README}: no '## What the samples show' section")
    block = section[1].split("```")
    if len(block) < 2:
        sys.exit(f"{README}: no fenced table after 'What the samples show'")

    rows: dict[str, list[str]] = {}
    for line in block[1].splitlines():
        for label in ROWS:
            if line.startswith(label):
                rows[label] = re.findall(
                    r"([0-9]+(?:\.[0-9]+)?)\s*(?:ms|tok/s)", line[len(label) :]
                )
    return rows


def main() -> int:
    samples = {}
    for column, filename in COLUMNS.items():
        path = PROFILES / filename
        if not path.is_file():
            sys.exit(f"missing profile sample: {path}")
        samples[column] = sample_values(path)

    rows = table_rows()
    missing = [label for label in ROWS if label not in rows]
    if missing:
        sys.exit(f"{README}: table is missing row(s): {', '.join(missing)}")

    problems: list[str] = []
    for label, (sample_label, unit, places) in ROWS.items():
        cells = rows[label]
        if len(cells) != len(COLUMNS):
            problems.append(
                f"row '{label}' has {len(cells)} value(s), expected {len(COLUMNS)}"
            )
            continue
        for cell, (column, values) in zip(cells, samples.items()):
            if sample_label not in values:
                problems.append(
                    f"{COLUMNS[column]} reports no '{sample_label}'"
                )
                continue
            expected = values[sample_label]
            try:
                actual = float(cell)
            except ValueError:
                problems.append(f"{label} / {column}: '{cell}' is not a number")
                continue
            # Half a unit in the table's last decimal place: any correct
            # rounding of the sample passes, a stale number does not.
            tolerance = 0.5 * 10**-places
            if abs(actual - expected) > tolerance:
                problems.append(
                    f"{label} / {column}: table says {cell} {unit}, "
                    f"{COLUMNS[column]} says {expected:g} {unit}"
                )

    if problems:
        print("examples/profiles/README.md disagrees with the committed samples:\n")
        for problem in problems:
            print(f"  - {problem}")
        print(
            "\nThe samples are the source of truth. Update the table (or "
            "regenerate the samples) so they agree."
        )
        return 1

    print(
        f"profile table agrees with {len(COLUMNS)} sample(s) "
        f"across {len(ROWS)} row(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
