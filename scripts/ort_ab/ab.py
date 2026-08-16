#!/usr/bin/env python3
"""Interleaved A/B driver for bench_generic.

Runs two binaries (or one) alternately over a model/thread grid so host drift
affects both arms equally, and records every trial's native p50, ORT p50 and the
within-run native/ort ratio. The ratio is the publishable metric on this
contended host; absolutes drift by >4x.
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


def run_one(binary: Path, model: Path, threads: int, runs: int, warmups: int, env=None):
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
    child_env = None
    if env:
        child_env = dict(os.environ)
        child_env.update(env)
    out = subprocess.run(cmd, capture_output=True, text=True, env=child_env)
    m = RESULT.search(out.stdout)
    if not m:
        sys.stderr.write(out.stdout[-2000:] + out.stderr[-2000:])
        raise RuntimeError(f"no result line for {model} threads={threads} bin={binary.name}")
    d = m.groupdict()
    return {k: (v if k == "parity" else float(v)) for k, v in d.items()}


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
    args = ap.parse_args()

    arms = {}
    for spec in args.arms:
        name, _, path = spec.partition("=")
        arms[name] = Path(path)
    arm_env: dict[str, dict[str, str]] = {name: {} for name in arms}
    for spec in args.arm_env:
        name, _, kv = spec.partition("=")
        key, _, value = kv.partition("=")
        arm_env.setdefault(name, {})[key] = value

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
                    )
                    r.update(
                        model=model_path.stem, threads=threads, trial=trial, arm=name
                    )
                    rows.append(r)
                    print(
                        f"{model_path.stem:28s} t={threads:<3d} trial={trial} {name:6s} "
                        f"native={r['native']:8.3f} ort={r['ort']:8.3f} "
                        f"ratio={r['ratio']:6.3f} parity={r['parity']}",
                        flush=True,
                    )

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== medians (native/ort ratio, lower is better) ===")
    keys = sorted({(r["model"], r["threads"]) for r in rows})
    for model, threads in keys:
        line = [f"{model:28s} t={threads:<3d}"]
        for name in arms:
            sel = [
                r["ratio"]
                for r in rows
                if r["model"] == model and r["threads"] == threads and r["arm"] == name
            ]
            nat = [
                r["native"]
                for r in rows
                if r["model"] == model and r["threads"] == threads and r["arm"] == name
            ]
            if sel:
                line.append(
                    f"{name}: ratio_p50={median(sel):6.3f} "
                    f"[{min(sel):.3f}-{max(sel):.3f}] native_p50={median(nat):8.3f}ms"
                )
        print("  ".join(line))


if __name__ == "__main__":
    main()
