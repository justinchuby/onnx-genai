#!/usr/bin/env python3
"""Interleaved multi-arm A/B for the native-vs-MLAS graduation bench.

`docs/performance/CPU_MLAS_MIGRATION.md` requires three things of any number
used to graduate a native route away from MLAS: it must be taken at a stated
thread width, the run-to-run spread of the ratio must be smaller than the win
being claimed, and reps that did not actually get the CPU must be discarded
rather than averaged in. This script produces all three.

An *arm* is a CPU mask plus an optional environment override. Arms are run once
each per rep, rotating which arm goes first, so drift in host load lands on
every arm equally instead of on whichever one ran last. Each rep's os.wait4
rusage is recorded, and reps materially below the median CPU-per-wall of their
siblings are discarded (the guard from #1809, which is what makes this
measurable on a shared box instead of requiring a quiet one).

Build the bench binary first:

    cargo bench -p onnx-runtime-ep-cpu --bench native_vs_mlas \
        --features mlas --no-run

Compare two thread widths:

    python3 scripts/bench_native_vs_mlas_width.py \
        target/release/deps/native_vs_mlas-<hash> \
        --arm wide:0-31 --arm narrow:16,20,22,26,28,30 \
        --reps 6 --out width_ab.json

Compare a kernel toggle at one width:

    python3 scripts/bench_native_vs_mlas_width.py \
        target/release/deps/native_vs_mlas-<hash> \
        --arm 'off:16,20,22,26,28,30:ONNX_GENAI_CPU_MM_SIMD_M1_GEMV=0' \
        --arm 'on:16,20,22,26,28,30:ONNX_GENAI_CPU_MM_SIMD_M1_GEMV=1' \
        --reps 6 --out m1_gemv_ab.json

Pick a narrow mask as one sibling per physical core inside a single L3 domain
(`/sys/devices/system/cpu/cpuN/topology/thread_siblings_list` and
`.../cache/index3/shared_cpu_list`), choosing cores that are currently idle. A
case whose verdict differs between reps of the same binary on the same arm is
reported as UNSTABLE rather than given a verdict: that disagreement is the
finding, not an inconvenience to be averaged away.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

GRADUATION_RATIO = 1.05
TRUST_FLOOR = 0.95


def cpu_count(spec):
    total = 0
    for part in spec.split(","):
        if "-" in part:
            low, high = part.split("-")
            total += int(high) - int(low) + 1
        else:
            total += 1
    return total


def parse_arm(spec):
    """NAME:CPUS[:KEY=VAL,KEY=VAL]"""
    fields = spec.split(":")
    if len(fields) not in (2, 3):
        raise argparse.ArgumentTypeError(f"bad arm {spec!r}, want NAME:CPUS[:ENV]")
    env = {}
    if len(fields) == 3 and fields[2]:
        for pair in fields[2].split(","):
            key, _, value = pair.partition("=")
            env[key] = value
    return {"name": fields[0], "cpus": fields[1], "env": env}


def run_once(binary, arm, rep):
    ncpu = cpu_count(arm["cpus"])
    env = dict(os.environ)
    env.update(arm["env"])
    started = time.time()
    proc = subprocess.Popen(
        ["taskset", "-c", arm["cpus"], binary, "--bench"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    out = proc.stdout.read()
    _, status, usage = os.wait4(proc.pid, 0)
    wall = time.time() - started
    cpu = usage.ru_utime + usage.ru_stime
    record = {
        "arm": arm["name"],
        "cpus": arm["cpus"],
        "env": arm["env"],
        "ncpu": ncpu,
        "rep": rep,
        "wall_s": round(wall, 2),
        "cpu_s": round(cpu, 2),
        "cpu_per_wall": round(cpu / wall, 3),
        "occupancy": round(cpu / (wall * ncpu), 3),
        "nivcsw": usage.ru_nivcsw,
        "status": status,
        "stdout": out,
    }
    sys.stderr.write(
        f"{arm['name']:8s} rep{rep}: wall={wall:5.1f}s "
        f"cpu/wall={record['cpu_per_wall']:6.2f} "
        f"occupancy={record['occupancy']:.3f} ivcsw={usage.ru_nivcsw}\n"
    )
    return record


def parse_rows(record):
    rows = {}
    for line in record["stdout"].splitlines():
        if line.startswith("#") or line.startswith("family"):
            continue
        parts = line.split("\t")
        if len(parts) != 7:
            continue
        rows[(parts[0], parts[1])] = {
            "native": float(parts[2]),
            "mlas": float(parts[3]),
            "ratio": float(parts[4]),
            "cpu_ratio": float(parts[5]),
            "verdict": parts[6],
        }
    return rows


def trusted(records):
    """Discard reps that did not get the CPU, relative to their siblings."""
    median = statistics.median(r["cpu_per_wall"] for r in records)
    floor = TRUST_FLOOR * median
    keep = [r for r in records if r["cpu_per_wall"] >= floor]
    drop = [r for r in records if r["cpu_per_wall"] < floor]
    return keep, median, floor, drop


def summarise(name, records):
    keep, median, floor, drop = trusted(records)
    env = records[0]["env"]
    label = f" env={env}" if env else ""
    print(
        f"\n=== arm {name}: cpus={records[0]['cpus']} "
        f"({records[0]['ncpu']} cpus){label} ==="
    )
    print(
        f"reps={len(records)} trusted={len(keep)} cpu_per_wall median={median:.3f} "
        f"floor={floor:.3f} discarded={[r['cpu_per_wall'] for r in drop] or 'none'}"
    )
    samples = {}
    for record in keep:
        for key, row in parse_rows(record).items():
            samples.setdefault(key, []).append(row)
    print(
        f"{'family':12s} {'case':24s} {'native':>8s} {'mlas':>8s} {'ratio':>7s} "
        f"{'min':>7s} {'max':>7s} {'spread':>7s} {'cpu_r':>6s}  verdict"
    )
    summary = {}
    for key, rows in samples.items():
        ratios = [r["ratio"] for r in rows]
        median_ratio = statistics.median(ratios)
        spread = (max(ratios) - min(ratios)) / median_ratio * 100
        verdicts = sorted({r["verdict"] for r in rows})
        verdict = verdicts[0] if len(verdicts) == 1 else "UNSTABLE:" + "|".join(verdicts)
        if verdict.startswith("native-graduates") and spread > 100 * (median_ratio - 1.0):
            verdict += "  (SPREAD EXCEEDS WIN -- not a graduation)"
        summary[key] = {
            "ratio": median_ratio,
            "spread": spread,
            "native": statistics.median(r["native"] for r in rows),
            "mlas": statistics.median(r["mlas"] for r in rows),
        }
        print(
            f"{key[0]:12s} {key[1]:24s} "
            f"{summary[key]['native']:8.4f} {summary[key]['mlas']:8.4f} "
            f"{median_ratio:7.3f} {min(ratios):7.3f} {max(ratios):7.3f} "
            f"{spread:6.1f}% "
            f"{statistics.median(r['cpu_ratio'] for r in rows):6.3f}  {verdict}"
        )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary")
    parser.add_argument("--arm", action="append", type=parse_arm, required=True)
    parser.add_argument("--reps", type=int, default=6)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if os.path.exists(args.out):
        sys.exit(f"refusing to clobber existing {args.out}")

    records = []
    for rep in range(1, args.reps + 1):
        shift = rep % len(args.arm)
        for arm in args.arm[shift:] + args.arm[:shift]:
            records.append(run_once(args.binary, arm, rep))

    with open(args.out, "w") as handle:
        json.dump({"arms": args.arm, "runs": records}, handle, indent=1)
    print(f"\nwrote {args.out}")

    summaries = {}
    for arm in args.arm:
        arm_records = [
            r for r in records if r["arm"] == arm["name"] and r["status"] == 0
        ]
        if arm_records:
            summaries[arm["name"]] = summarise(arm["name"], arm_records)

    names = [a["name"] for a in args.arm if a["name"] in summaries]
    if len(names) == 2:
        first, second = names
        print(f"\n=== does the verdict survive {first} -> {second}? ===")
        print(f"{'family':12s} {'case':24s} {first:>9s} {second:>9s}  flips?")
        for key, left in summaries[first].items():
            right = summaries[second].get(key)
            if right is None:
                continue
            flips = (left["ratio"] >= GRADUATION_RATIO) != (
                right["ratio"] >= GRADUATION_RATIO
            )
            print(
                f"{key[0]:12s} {key[1]:24s} {left['ratio']:9.3f} "
                f"{right['ratio']:9.3f}  {'FLIPS' if flips else ''}"
            )


if __name__ == "__main__":
    main()
