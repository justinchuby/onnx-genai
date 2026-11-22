"""Darwin-only, standalone CPU accounting diagnostic; see cpu-clock-scope report."""

import argparse
import csv
import fcntl
import gzip
import hashlib
import io
import itertools
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

LOCK = Path("/tmp/onnx-genai-hostlock-darwin.flock")
COUNTS = (16384, 131072, 1048576)
FIELDS = [
    "kind",
    "width",
    "spin",
    "observe",
    "iterations",
    "batch",
    "rep",
    "arm",
    "wall_ns",
    "getrusage_ns",
    "process_ns",
    "thread_sum_ns",
    "main_thread_ns",
    "getrusage_drained_ns",
    "process_drained_ns",
    "wall_drained_ns",
    "checksum",
]
METRICS = [field for field in FIELDS if field.endswith("_ns")]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def analyze(raw, metadata):
    require(sha(raw) == metadata["raw_sha256"], "raw CSV checksum differs from receipt")
    reader = csv.DictReader(io.StringIO(raw.decode()))
    require(
        reader.fieldnames == ["sweep", *FIELDS], "CSV schema differs from clock probe"
    )
    rows = []
    for record in reader:
        require(
            None not in record and None not in record.values(), "CSV row is incomplete"
        )
        rows.append(
            {
                key: value if key == "kind" else int(value)
                for key, value in record.items()
            }
        )
    require(metadata["schema"] == 1, "unsupported receipt schema")
    reps, sweeps = metadata["repetitions"], metadata["sweeps"]
    require(1 <= reps <= 101 and 1 <= sweeps <= 9, "receipt matrix limits differ")
    receipts = metadata["receipts"]
    require(len(receipts) == 3 * sweeps, "missing process receipts")
    require(
        {(r["sweep"], r["width"]) for r in receipts}
        == set(itertools.product(range(1, sweeps + 1), (1, 4, 8))),
        "process receipt identities differ",
    )
    require(
        all(r["exit"] == 0 and r["wall_s"] > 0 and r["cpu_s"] > 0 for r in receipts),
        "process failed or resource receipt is empty",
    )
    expected = set(
        itertools.product(
            range(1, sweeps + 1),
            (1, 4, 8),
            (0, 1),
            (0, 1),
            COUNTS,
            (1, 16),
            range(reps),
            (0, 1),
        )
    )
    seen, idle, checksums, groups = set(), set(), {}, {}
    for row in rows:
        require(
            all(value >= 0 for key, value in row.items() if key != "kind"),
            "negative clock or case field",
        )
        sweep, width = row["sweep"], row["width"]
        require(1 <= sweep <= sweeps and width in (1, 4, 8), "unexpected sweep/width")
        if row["kind"] == "idle":
            require((sweep, width) not in idle, "duplicate idle control")
            idle.add((sweep, width))
            require(
                all(
                    row[key] == 0
                    for key in (
                        "spin",
                        "observe",
                        "iterations",
                        "batch",
                        "rep",
                        "arm",
                        "checksum",
                        "thread_sum_ns",
                        "main_thread_ns",
                    )
                ),
                "idle control performed work",
            )
            require(row["wall_ns"] >= 20_000_000, "idle control slept less than 20 ms")
            continue
        require(row["kind"] == "sample", "unknown row kind")
        key = tuple(
            row[field]
            for field in (
                "sweep",
                "width",
                "spin",
                "observe",
                "iterations",
                "batch",
                "rep",
                "arm",
            )
        )
        require(
            key in expected and key not in seen, f"duplicate/unexpected sample {key}"
        )
        seen.add(key)
        checksum_key = (width, row["iterations"], row["batch"])
        require(row["checksum"] > 0, "empty work checksum")
        require(
            checksums.setdefault(checksum_key, row["checksum"]) == row["checksum"],
            "A/A work checksum differs between cases",
        )
        require(
            all(
                row[field] > 0
                for field in METRICS
                if field not in ("thread_sum_ns", "main_thread_ns")
            ),
            "empty timing",
        )
        for immediate, drained in (
            ("wall_ns", "wall_drained_ns"),
            ("getrusage_ns", "getrusage_drained_ns"),
            ("process_ns", "process_drained_ns"),
        ):
            require(row[drained] >= row[immediate], "drained window shrank")
        for field in ("thread_sum_ns", "main_thread_ns"):
            require(
                (row[field] > 0) == bool(row["observe"]),
                "thread-clock ablation differs",
            )
        groups.setdefault(key[:6], {0: [], 1: []})[row["arm"]].append(row)
    require(
        seen == expected,
        f"incomplete matrix: got {len(seen)}, expected {len(expected)}",
    )
    require(
        idle == set(itertools.product(range(1, sweeps + 1), (1, 4, 8))),
        "missing idle controls",
    )
    cells = []
    for key, arms in sorted(groups.items()):
        medians = {
            str(arm): {
                metric: statistics.median(row[metric] for row in samples)
                for metric in METRICS
            }
            for arm, samples in arms.items()
        }
        ratios = {
            metric: medians["0"][metric] / medians["1"][metric]
            for metric in METRICS
            if medians["1"][metric] > 0
        }
        cells.append(
            dict(
                zip(
                    ("sweep", "width", "spin", "observe", "iterations", "batch"),
                    key,
                    strict=True,
                )
            )
            | {"medians": medians, "aa_0_over_1": ratios}
        )
    return {
        "schema": 1,
        "sample_count": len(seen),
        "idle_count": len(idle),
        "cells": cells,
        "idle": [row for row in rows if row["kind"] == "idle"],
    }


def run(directory, reps, sweeps):
    require(
        sys.platform == "darwin", "runner requires Darwin; use hostlock.sh for Linux"
    )
    require(
        1 <= reps <= 101 and 1 <= sweeps <= 9, "use repetitions 1..101 and sweeps 1..9"
    )
    directory.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_suffix(".c").resolve()
    binary = directory / "probe"
    command = [
        "clang",
        "-std=c11",
        "-O3",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-pthread",
        str(source),
        "-o",
        str(binary),
    ]
    subprocess.run(command, check=True)
    metadata = {
        "schema": 1,
        "repetitions": reps,
        "sweeps": sweeps,
        "source_sha256": sha(source.read_bytes()),
        "runner_sha256": sha(Path(__file__).read_bytes()),
        "binary_sha256": sha(binary.read_bytes()),
        "compile_command": command,
        "compiler": subprocess.check_output(["clang", "--version"], text=True),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "lock": str(LOCK),
        "receipts": [],
    }
    raw = io.StringIO()
    writer = csv.writer(raw, lineterminator="\n")
    writer.writerow(["sweep", *FIELDS])
    # Stable shared inode, never unlink. Nonblocking admission; retain the lock
    # over the whole sweep. Load receipts cannot prove isolation or CPU affinity.
    with LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for sweep in range(1, sweeps + 1):
            widths = (1, 4, 8)
            shift = (sweep - 1) % 3
            widths = widths[-shift:] + widths[:-shift] if shift else widths
            for width in widths:
                name = f"sweep{sweep}-workers{width}"
                stdout_path, stderr_path = (
                    directory / f"{name}.csv",
                    directory / f"{name}.stderr",
                )
                load, start = os.getloadavg(), time.monotonic()
                with (
                    stdout_path.open("wb") as stdout,
                    stderr_path.open("wb") as stderr,
                    subprocess.Popen(
                        [str(binary), str(width), str(reps)],
                        stdout=stdout,
                        stderr=stderr,
                    ) as child,
                ):
                    pid = child.pid
                    _, status, usage = os.wait4(pid, 0)
                    child.returncode = os.waitstatus_to_exitcode(status)
                receipt = {
                    "sweep": sweep,
                    "width": width,
                    "pid": pid,
                    "exit": os.waitstatus_to_exitcode(status),
                    "wall_s": time.monotonic() - start,
                    "cpu_s": usage.ru_utime + usage.ru_stime,
                    "maxrss_bytes": usage.ru_maxrss,
                    "load_before": load,
                    "load_after": os.getloadavg(),
                    "stderr": stderr_path.read_text(),
                }
                metadata["receipts"].append(receipt)
                (directory / "metadata.json").write_text(
                    json.dumps(metadata, indent=2) + "\n"
                )
                print(json.dumps(receipt), flush=True)
                require(receipt["exit"] == 0, f"probe failed: inspect {stderr_path}")
                reader = csv.reader(io.StringIO(stdout_path.read_text()))
                require(next(reader) == FIELDS, "child CSV schema differs")
                for row in reader:
                    writer.writerow([sweep, *row])
    data = raw.getvalue().encode()
    metadata["raw_sha256"] = sha(data)
    with (directory / "samples.csv.gz").open("wb") as output:
        output.write(gzip.compress(data, mtime=0))
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    result = analyze(data, metadata)
    (directory / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Validated {result['sample_count']} samples and {result['idle_count']} idle controls"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "analyze"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--repetitions", type=int, default=21)
    parser.add_argument("--sweeps", type=int, default=3)
    args = parser.parse_args()
    if args.action == "run":
        run(args.directory.resolve(), args.repetitions, args.sweeps)
    else:
        metadata = json.loads((args.directory / "metadata.json").read_text())
        result = analyze(
            gzip.decompress((args.directory / "samples.csv.gz").read_bytes()), metadata
        )
        saved = json.loads((args.directory / "results.json").read_text())
        require(
            result == saved,
            "saved summaries differ from raw samples; regenerate report",
        )
        print(
            f"Validated {result['sample_count']} samples and {len(result['cells'])} A/A cells"
        )


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        sys.exit(
            f"CPU-clock diagnostic failed: {error}. Inspect the output directory and retry in a fresh directory."
        )
