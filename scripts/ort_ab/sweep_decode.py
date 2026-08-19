#!/usr/bin/env python3
"""Sweep decode-shape matmul cells across thread counts, ours vs ORT.

`bench_generic` already alternates our EP and ORT inside one process, so each
line is a paired measurement. What this adds is the *thread-count sweep* and an
aggregation that survives a contended host: it reports the median of the
per-run p50 **and** the min of the per-run min, because on a shared machine the
min is the only statistic that is not a measure of the other tenants.

Absolute milliseconds are printed, not just the ratio -- the whole question
these rows raise is whether we get slower with more threads or whether ORT
simply scales while we stay flat, and a ratio cannot tell those apart.
"""

from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
from pathlib import Path

LINE = re.compile(
    r"native=(?P<native>[\d.]+) ms .*?ort=(?P<ort>[\d.]+) ms .*?"
    r"native/ort=(?P<ratio>[\d.]+) .*?"
    r"native_min=(?P<native_min>[\d.]+) ort_min=(?P<ort_min>[\d.]+)"
)


def run_one(binary: Path, model: Path, threads: int, runs: int, warmups: int):
    cmd = [
        str(binary),
        "--model",
        str(model),
        "--native-threads",
        str(threads),
        "--ort-intra-threads",
        str(threads),
        "--runs",
        str(runs),
        "--warmups",
        str(warmups),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        m = LINE.search(line)
        if m:
            return {k: float(v) for k, v in m.groupdict().items()}
    raise RuntimeError(f"no result line for {model} t={threads}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", type=Path, default=Path("target/release/bench_generic"))
    ap.add_argument("--models", nargs="+", type=Path, required=True)
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--runs", type=int, default=7)
    ap.add_argument("--warmups", type=int, default=3)
    args = ap.parse_args()

    print(
        f"{'model':26s} {'t':>3s} {'native_p50':>10s} {'ort_p50':>8s} "
        f"{'ratio_p50':>9s} {'native_min':>10s} {'ort_min':>8s} {'ratio_min':>9s}"
    )
    for model in args.models:
        for threads in args.threads:
            trials = []
            for _ in range(args.trials):
                trials.append(
                    run_one(args.binary, model, threads, args.runs, args.warmups)
                )
            native_p50 = statistics.median(t["native"] for t in trials)
            ort_p50 = statistics.median(t["ort"] for t in trials)
            native_min = min(t["native_min"] for t in trials)
            ort_min = min(t["ort_min"] for t in trials)
            print(
                f"{model.stem:26s} {threads:3d} {native_p50:10.3f} {ort_p50:8.3f} "
                f"{native_p50 / ort_p50:9.3f} {native_min:10.3f} {ort_min:8.3f} "
                f"{native_min / ort_min:9.3f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
