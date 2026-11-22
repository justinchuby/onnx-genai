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
    compressed = (args.metadata.parent / metadata["samples_file"]).read_bytes()
    require(
        hashlib.sha256(compressed).hexdigest() == metadata["samples_sha256"],
        "raw sample checksum differs; restore the samples archive named by the metadata",
    )
    rows = list(csv.DictReader(io.StringIO(gzip.decompress(compressed).decode())))
    require(
        len(rows) == metadata["samples_count"] == 17010,
        f"expected 17010 raw samples, got {len(rows)}; restore the complete archive",
    )
    groups = collections.defaultdict(list)
    for row in rows:
        rep, order = int(row["rep"]), int(row["order"])
        require(
            0 <= rep < 21 and 0 <= order < 3,
            f"invalid sample repetition/order: {rep}/{order}",
        )
        require(
            row["arm"] == ["mlas", "native", "native_aa"][(rep + order) % 3],
            f"incorrect rotating arm at repetition {rep}, order {order}",
        )
        n = int(row["n"])
        require(n > 0, f"invalid vector length {n}; timing requires nonempty input")
        require(
            int(row["iters"]) == max(1, min(16384, ((1 << 20) + n - 1) // n)),
            f"incorrect batch length for n={n}",
        )
        require(
            float(row["cpu_ns"]) > 0 and float(row["wall_ns"]) > 0,
            f"nonpositive timing for n={n}",
        )
        key = (int(row["sweep"]), int(row["width"]), row["mode"], row["pattern"], n)
        groups[key].append(row)
    require(
        len(groups) == metadata["arm_summary_count"] == 270,
        f"expected 270 cells, got {len(groups)}; restore complete data",
    )
    expected_cells = set(
        itertools.product(
            [1, 2, 3],
            [1, 4, 8],
            ["serial", "runtime"],
            ["in_band", "sparse_repairs", "dense_negative_tail"],
            [3, 17, 4096, 65537, 1048581],
        )
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
        for arm in ["mlas", "native", "native_aa"]:
            arm_rows = [r for r in group if r["arm"] == arm]
            require(
                len(arm_rows) == 21
                and {int(r["rep"]) for r in arm_rows} == set(range(21)),
                f"cell {key}, arm {arm}: expected all 21 repetitions exactly once",
            )
            for metric in ["cpu_ns", "wall_ns"]:
                require(
                    statistics.median(float(r[metric]) for r in arm_rows)
                    == summary["median"][arm][metric],
                    f"cell {key}, arm {arm}: published {metric} median differs from raw samples",
                )
        m = summary["median"]
        for field, numerator, denominator, metric in [
            ("wall_ratio", "mlas", "native", "wall_ns"),
            ("cpu_ratio", "mlas", "native", "cpu_ns"),
            ("aa_wall_ratio", "native", "native_aa", "wall_ns"),
            ("aa_cpu_ratio", "native", "native_aa", "cpu_ns"),
        ]:
            require(
                m[numerator][metric] / m[denominator][metric] == summary[field],
                f"cell {key}: published {field} differs from medians",
            )
    print(
        f"Validated {len(rows)} samples and {len(groups)} cells; ratios are MLAS/native (>1 favors native)."
    )
    for pattern in ["in_band", "sparse_repairs", "dense_negative_tail"]:
        for n in [3, 17, 4096, 65537, 1048581]:
            cells = [
                s
                for s in metadata["summary"]
                if s["pattern"] == pattern and s["n"] == n
            ]
            ratios = [s["wall_ratio"] for s in cells]
            print(f"{pattern:20} {n:8} {min(ratios):.3f}–{max(ratios):.3f}")
except (OSError, ValueError, KeyError, TypeError, UnicodeError) as error:
    parser.error(f"{args.metadata}: cannot validate SiLU evidence: {error}")
