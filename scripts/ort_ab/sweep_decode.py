#!/usr/bin/env python3
"""Sweep decode-shape matmul cells across thread counts, ours vs ORT.

`bench_generic` already alternates our EP and ORT inside one process, so each
line is a paired measurement. What this adds is the *thread-count sweep* and an
aggregation that survives a contended host: it reports the median of the
per-run p50, the median of the per-run p90 (the statistic
`docs/performance/CPU_MATMUL_ASSIGNMENT.md` tabulates) **and** the min of the
per-run min, because on a shared machine the min is the only statistic that is
not partly a measure of the other tenants.

Absolute milliseconds are printed, not just the ratio -- the whole question
these rows raise is whether we get slower with more threads or whether ORT
simply scales while we stay flat, and a ratio cannot tell those apart.

The min-of-min above was written before the host lock existed, and it is a way
to *survive* contention rather than to exclude it: it is blind to an SMT
sibling and to steady external load, which is exactly what `73c76458c` says
about per-run efficiency and A/A nulls. So this sweep now refuses to run
without a declaration held by an ancestor -- a thread sweep is the most
saturating thing we run, and it is the least defensible thing to run beside
somebody else's measurement. `--unlocked` exists for smoke tests and stamps
every row.
"""

from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
from pathlib import Path

import hostlock_gate

LINE = re.compile(
    r"native=(?P<native>[\d.]+) ms .*?ort=(?P<ort>[\d.]+) ms .*?"
    r"native/ort=(?P<ratio>[\d.]+) "
    r"native_p90=(?P<native_p90>[\d.]+) ort_p90=(?P<ort_p90>[\d.]+) "
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


def end_of_window_verdict(start: str, end: str) -> tuple[int, str | None]:
    """What to say once the rows are already on the screen.

    `ab.py` buffers, so it can stamp the end-of-window label onto every row.
    This driver streams, so it cannot relabel what has been printed -- which
    makes the exit code and one stderr line the only honest places left to
    put the finding.

    The three outcomes are kept distinct because they mean different things
    to the person reading them. A handoff (`changed`) means the thread counts
    above and below the change were compared across it: not half-good data,
    discard it. A failed end-read (`unverified-end`) means we do not know --
    the rows may be perfectly good, and telling someone to discard them would
    assert a specific false fact about data that is probably fine. Collapsing
    the second into the first is the exact conflation `window_label` and its
    tests exist to prevent.
    """
    if end == "changed":
        return 4, (
            f"host_lock=changed: the declaration covering this sweep did not "
            f"hold for the whole of it (started {start}). Every row above "
            "spans the change -- discard them."
        )
    if end == "unverified-end":
        return 5, (
            f"host_lock=unverified-end: the lock could not be re-read when "
            f"the sweep finished, so the {start} label on every row above is "
            "unverified at the far end. The rows may be sound; nothing here "
            "establishes that they are."
        )
    return 0, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", type=Path, default=Path("target/release/bench_generic"))
    ap.add_argument("--models", nargs="+", type=Path, required=True)
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--runs", type=int, default=7)
    ap.add_argument("--warmups", type=int, default=3)
    ap.add_argument(
        "--unlocked",
        action="store_true",
        help="run without a host-lock declaration and stamp every row (smoke tests only)",
    )
    args = ap.parse_args()

    # Before the first child, not after: refusing later has already put load on
    # a host somebody else declared.
    lock_label, prov = hostlock_gate.require(
        "python3 scripts/ort_ab/sweep_decode.py <your args>", unlocked=args.unlocked
    )
    columns = hostlock_gate.lock_columns(lock_label, prov)
    print(
        " ".join(f"{k}={v}" for k, v in columns.items()),
        flush=True,
    )

    print(
        f"{'model':26s} {'t':>3s} {'native_p50':>10s} {'ort_p50':>8s} "
        f"{'ratio_p50':>9s} {'ratio_p90':>9s} {'native_min':>10s} {'ort_min':>8s} "
        f"{'ratio_min':>9s} {'host_lock':>14s}"
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
            native_p90 = statistics.median(t["native_p90"] for t in trials)
            ort_p90 = statistics.median(t["ort_p90"] for t in trials)
            native_min = min(t["native_min"] for t in trials)
            ort_min = min(t["ort_min"] for t in trials)
            print(
                f"{model.stem:26s} {threads:3d} {native_p50:10.3f} {ort_p50:8.3f} "
                f"{native_p50 / ort_p50:9.3f} {native_p90 / ort_p90:9.3f} "
                f"{native_min:10.3f} {ort_min:8.3f} "
                f"{native_min / ort_min:9.3f} {lock_label:>14s}",
                flush=True,
            )

    end_label = hostlock_gate.window_label(
        lock_label, prov, hostlock_gate.read_provenance()
    )
    code, complaint = end_of_window_verdict(lock_label, end_label)
    if complaint:
        print(complaint, file=sys.stderr)
    return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
