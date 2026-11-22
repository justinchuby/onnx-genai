"""Validate SiLU raw samples and reproduce the published ratio ranges."""

import argparse
import collections
import csv
import gzip
import hashlib
import io
import itertools
import json
import statistics
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("metadata", type=Path)
args = parser.parse_args()


def require(condition, message):
    if not condition:
        raise ValueError(message)


try:
    metadata = json.loads(args.metadata.read_text())
    require(
        metadata["schema"] == 2,
        "unsupported metadata schema; regenerate explicit schema 2 metadata",
    )
    arms = metadata["arms"]
    reps = metadata["repetitions"]
    require(
        len(arms) >= 3 and len(set(arms)) == len(arms) and reps > 0,
        "declare distinct arms and a positive repetition count",
    )
    matrix = metadata["matrix"]
    axes = ["sweep", "width", "mode", "pattern", "n"]
    require(
        all(
            matrix[axis] and len(set(matrix[axis])) == len(matrix[axis])
            for axis in axes
        ),
        "matrix axes must be nonempty and contain no duplicate entries",
    )
    expected_cells = set(itertools.product(*(matrix[axis] for axis in axes)))
    expected_samples = len(expected_cells) * len(arms) * reps
    batch = metadata["batch"]
    require(
        batch["target_elements"] > 0 and 0 < batch["min_iters"] <= batch["max_iters"],
        "batch target/min/max must describe a positive iteration range",
    )
    compressed = (args.metadata.parent / metadata["samples_file"]).read_bytes()
    require(
        hashlib.sha256(compressed).hexdigest() == metadata["samples_sha256"],
        "raw sample checksum differs; restore the samples archive named by the metadata",
    )
    rows = list(csv.DictReader(io.StringIO(gzip.decompress(compressed).decode())))
    require(
        len(rows) == metadata["samples_count"] == expected_samples,
        f"expected {expected_samples} raw samples, got {len(rows)}; restore the complete archive",
    )
    groups = collections.defaultdict(list)
    for row in rows:
        rep, order = int(row["rep"]), int(row["order"])
        require(
            0 <= rep < reps and 0 <= order < len(arms),
            f"invalid sample repetition/order: {rep}/{order}",
        )
        require(
            row["arm"] == arms[(rep + order) % len(arms)],
            f"incorrect rotating arm at repetition {rep}, order {order}",
        )
        n = int(row["n"])
        require(n > 0, f"invalid vector length {n}; timing requires nonempty input")
        require(
            int(row["iters"])
            == max(
                batch["min_iters"],
                min(batch["max_iters"], (batch["target_elements"] + n - 1) // n),
            ),
            f"incorrect batch length for n={n}",
        )
        require(
            float(row["cpu_ns"]) > 0 and float(row["wall_ns"]) > 0,
            f"nonpositive timing for n={n}",
        )
        key = (int(row["sweep"]), int(row["width"]), row["mode"], row["pattern"], n)
        groups[key].append(row)
    require(
        len(groups) == metadata["cell_count"] == len(expected_cells),
        f"expected {len(expected_cells)} cells, got {len(groups)}; restore complete data",
    )
    require(
        set(groups) == expected_cells,
        "raw samples do not cover the declared sweep/width/mode/pattern/length matrix",
    )
    summary_keys = [
        tuple(s[k] for k in ["sweep", "width", "mode", "pattern", "n"])
        for s in metadata["summary"]
    ]
    require(
        len(summary_keys) == len(expected_cells)
        and set(summary_keys) == expected_cells,
        "published summaries contain duplicate or missing matrix cells",
    )
    for summary in metadata["summary"]:
        key = tuple(summary[k] for k in ["sweep", "width", "mode", "pattern", "n"])
        group = groups[key]
        for arm in arms:
            arm_rows = [r for r in group if r["arm"] == arm]
            require(
                len(arm_rows) == reps
                and {int(r["rep"]) for r in arm_rows} == set(range(reps)),
                f"cell {key}, arm {arm}: expected all {reps} repetitions exactly once",
            )
            for metric in ["cpu_ns", "wall_ns"]:
                require(
                    statistics.median(float(r[metric]) for r in arm_rows)
                    == summary["median"][arm][metric],
                    f"cell {key}, arm {arm}: published {metric} median differs from raw samples",
                )
        m = summary["median"]
        for field, ratio in metadata["ratios"].items():
            numerator = ratio["numerator"]
            denominator = ratio["denominator"]
            metric = ratio["metric"]
            require(
                m[numerator][metric] / m[denominator][metric] == summary[field],
                f"cell {key}: published {field} differs from medians",
            )
    print(
        f"Validated {len(rows)} samples and {len(groups)} cells; ratios are MLAS/native (>1 favors native)."
    )
    for pattern in matrix["pattern"]:
        for n in matrix["n"]:
            cells = [
                s
                for s in metadata["summary"]
                if s["pattern"] == pattern and s["n"] == n
            ]
            ratios = [s["wall_ratio"] for s in cells]
            print(f"{pattern:20} {n:8} {min(ratios):.3f}–{max(ratios):.3f}")
except (OSError, ValueError, KeyError, TypeError, UnicodeError) as error:
    parser.error(f"{args.metadata}: cannot validate SiLU evidence: {error}")
