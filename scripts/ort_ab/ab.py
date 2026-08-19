#!/usr/bin/env python3
"""Interleaved A/B driver for bench_generic.

Runs two binaries (or one) alternately over a model/thread grid so host drift
affects both arms equally, and records every trial's native p50, ORT p50 and the
within-run native/ort ratio. The ratio is the publishable metric on this
contended host; absolutes drift by >4x.

`--null-control` adds a third arm that is the *first arm's own binary under a
second name*. It measures nothing about the change, which is the point: whatever
delta it reports is this host's noise floor for that cell, measured in the same
invocation as the real comparison. Any real delta smaller than the null delta is
not a result. See section 37.4 of the CPU-EP benchmark ledger for the run that
made this non-optional: two binaries traced to be on the identical code path
still measured ~40% apart on the median at two threads.

`--native-only` drops the ORT arm from every child process and compares native
times directly. Use it for any native-vs-native A/B. ORT's intra-op pool
spin-waits, so a paired run steals cores from the native arm -- measured at up
to 6x depression on long cells -- and the resulting noise routinely exceeds the
effect under test. See `sebastian-paired-harness-coresidency`.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path
from statistics import median

RESULT = re.compile(
    r"native=(?P<native>[\d.]+) ms .*?ort=(?P<ort>[\d.]+) ms .*?"
    r"native/ort=(?P<ratio>[\d.]+) native_p90=(?P<np90>[\d.]+) ort_p90=(?P<op90>[\d.]+) "
    r"native_min=(?P<nmin>[\d.]+) ort_min=(?P<omin>[\d.]+) "
    r"native_spread=(?P<nspread>[\d.]+) ort_spread=(?P<ospread>[\d.]+).*?parity=(?P<parity>\w+)"
)

# `--native-only` prints no ORT figures at all, so it needs its own pattern:
# the ratio and every `ort_*` field are absent rather than empty, and the
# surviving native fields carry a ` ms` unit that the paired line omits.
RESULT_NATIVE_ONLY = re.compile(
    r"native=(?P<native>[\d.]+) ms .*?native_p90=(?P<np90>[\d.]+)(?: ms)? "
    r"native_min=(?P<nmin>[\d.]+)(?: ms)? "
    r"native_spread=(?P<nspread>[\d.]+).*?parity=(?P<parity>\w+)"
)


def run_one(
    binary: Path,
    model: Path,
    threads: int,
    runs: int,
    warmups: int,
    env=None,
    native_only: bool = False,
):
    cmd = [
        str(binary.resolve()),
        "--model",
        str(model),
        "--runs",
        str(runs),
        "--warmups",
        str(warmups),
        "--native-threads",
        str(threads),
        "--ort-intra-threads",
        str(threads),
    ]
    if native_only:
        cmd.append("--native-only")
    child_env = None
    if env:
        child_env = dict(os.environ)
        child_env.update(env)
    out = subprocess.run(cmd, capture_output=True, text=True, env=child_env)
    m = (RESULT_NATIVE_ONLY if native_only else RESULT).search(out.stdout)
    if not m:
        sys.stderr.write(out.stdout[-2000:] + out.stderr[-2000:])
        raise RuntimeError(f"no result line for {model} threads={threads} bin={binary.name}")
    d = m.groupdict()
    r = {k: (v if k == "parity" else float(v)) for k, v in d.items()}
    if native_only:
        # There is no ORT arm to divide by, so the comparable metric is the
        # native time itself. Keeping the key name lets every downstream
        # summary stay one code path, and `native_only` marks the CSV so a
        # machine consumer reading `ratio` cannot mistake milliseconds for a
        # dimensionless ratio.
        r["ort"] = float("nan")
        r["ratio"] = r["native"]
    r["native_only"] = int(native_only)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, help="name=path pairs")
    ap.add_argument(
        "--arm-env",
        nargs="*",
        default=[],
        help="arm=KEY=VALUE overrides applied to that arm's child process",
    )
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--threads", nargs="+", type=int, default=[8])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--runs", type=int, default=15)
    ap.add_argument("--warmups", type=int, default=5)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument(
        "--null-control",
        action="store_true",
        help="add a duplicate of the first arm, under the name 'null', so the "
        "run measures its own noise floor alongside the real comparison",
    )
    ap.add_argument(
        "--native-only",
        action="store_true",
        help="run each arm with --native-only and compare native times "
        "directly, with no ORT arm in the process. Required for "
        "native-vs-native A/B: ORT's intra-op pool spin-waits, so a paired "
        "run depresses the native arm (up to 6x on long cells here) and its "
        "noise swamps the comparison",
    )
    args = ap.parse_args()

    arms = {}
    for spec in args.arms:
        name, _, path = spec.partition("=")
        arms[name] = Path(path)
    baseline = next(iter(arms))
    if args.null_control:
        if "null" in arms:
            ap.error("--null-control needs the arm name 'null' to be free")
        # Same file, same environment overrides, second name. The delta this
        # arm reports against `baseline` is pure host noise by construction.
        arms["null"] = arms[baseline]
    arm_env: dict[str, dict[str, str]] = {name: {} for name in arms}
    for spec in args.arm_env:
        name, _, kv = spec.partition("=")
        key, _, value = kv.partition("=")
        arm_env.setdefault(name, {})[key] = value
    if args.null_control:
        # The control has to be the baseline in every respect the driver can
        # vary, environment included, or it stops measuring only noise.
        arm_env["null"] = dict(arm_env.get(baseline, {}))

    rows = []
    for model in args.models:
        model_path = Path(model)
        for threads in args.threads:
            for trial in range(args.trials):
                order = list(arms.items())
                if trial % 2 == 1:
                    order = order[::-1]
                for name, binary in order:
                    r = run_one(
                        binary,
                        model_path,
                        threads,
                        args.runs,
                        args.warmups,
                        env=arm_env.get(name),
                        native_only=args.native_only,
                    )
                    r.update(
                        model=model_path.stem, threads=threads, trial=trial, arm=name
                    )
                    rows.append(r)
                    tail = (
                        f"parity={r['parity']}"
                        if args.native_only
                        else f"ort={r['ort']:8.3f} ratio={r['ratio']:6.3f} "
                        f"parity={r['parity']}"
                    )
                    print(
                        f"{model_path.stem:28s} t={threads:<3d} trial={trial} {name:6s} "
                        f"native={r['native']:8.3f} " + tail,
                        flush=True,
                    )

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    metric = "native ms" if args.native_only else "native/ort ratio"
    print(f"\n=== medians ({metric}, lower is better) ===")
    keys = sorted({(r["model"], r["threads"]) for r in rows})
    for model, threads in keys:
        line = [f"{model:28s} t={threads:<3d}"]
        for name in arms:
            cell = [
                r
                for r in rows
                if r["model"] == model and r["threads"] == threads and r["arm"] == name
            ]
            if cell:
                sel = [r["ratio"] for r in cell]
                nat = [r["native"] for r in cell]
                # A timing from a cell that does not reproduce ORT's answer is
                # not a timing. Mark it in the summary rather than letting it
                # blend into a median that reads as a clean win.
                bad = sum(1 for r in cell if r["parity"] != "PASS")
                flag = f" PARITY_FAIL={bad}/{len(cell)}" if bad else ""
                label = "native_p50" if args.native_only else "ratio_p50"
                line.append(
                    f"{name}: {label}={median(sel):6.3f} "
                    f"[{min(sel):.3f}-{max(sel):.3f}] native_p50={median(nat):8.3f}ms{flag}"
                )
        print("  ".join(line))

    if len(arms) > 1:
        print(
            "\n=== deltas vs "
            f"'{baseline}' (median {metric}; negative = arm is faster) ==="
        )
        if args.null_control:
            print(
                "The 'null' column is the same binary as "
                f"'{baseline}'. Its delta is this host's noise floor for the "
                "cell, so a real delta no larger than it is not a result."
            )
        for model, threads in keys:
            def cell_p50(arm_name: str) -> float | None:
                sel = [
                    r["ratio"]
                    for r in rows
                    if r["model"] == model
                    and r["threads"] == threads
                    and r["arm"] == arm_name
                ]
                return median(sel) if sel else None

            base_p50 = cell_p50(baseline)
            if base_p50 is None or base_p50 == 0.0:
                continue
            null_delta = None
            if args.null_control:
                null_p50 = cell_p50("null")
                if null_p50 is not None:
                    null_delta = abs(null_p50 / base_p50 - 1.0) * 100.0
            parts = [f"{model:28s} t={threads:<3d}"]
            for name in arms:
                if name == baseline:
                    continue
                p50 = cell_p50(name)
                if p50 is None:
                    continue
                delta = (p50 / base_p50 - 1.0) * 100.0
                verdict = ""
                if name != "null" and null_delta is not None:
                    verdict = (
                        "  WITHIN NOISE"
                        if abs(delta) <= null_delta
                        else f"  > noise ({null_delta:.2f}%)"
                    )
                parts.append(f"{name}: {delta:+7.2f}%{verdict}")
            print("  ".join(parts))

    failed = sum(1 for r in rows if r["parity"] != "PASS")
    if failed:
        print(
            f"\nWARNING: {failed}/{len(rows)} trials did not match ORT numerically. "
            "Those cells' timings are not comparable and must not be published."
        )


if __name__ == "__main__":
    main()
