#!/usr/bin/env python3
"""Falsifier: is the width-16 A/A null foreign load on the pinned CPUs?

`2026-08-24-acc0-w16-null-page-backing.md` narrowed the null to a single
question. The pool builds 15 workers on 15 verified distinct physical cores, and
yet `cpu_s_per_token / ms_token` says a slow launch runs on **9.8-12.2 of its
sixteen lanes** while a fast one runs on 15.3-16.1 -- no overlap, decided before
the first token, constant for the life of the process. The leading hypothesis is
that some of the pinned CPUs are shared with another process, which turns those
workers into permanent stragglers and drops effective width in discrete steps.

That hypothesis is directly falsifiable and this is the falsifier. For each
launch, per-CPU busy jiffies are read from `/proc/stat` for exactly the pinned
set, before and after, and the child's own total CPU time is subtracted:

    foreign_s = sum over pinned cpus of busy_delta  -  child_cpu_s

`child_cpu_s` comes from `getrusage(RUSAGE_CHILDREN)` deltas, so it covers the
whole process rather than the steady phase alone, and the child is pinned to the
same set, so all of its CPU time lands inside the sum. What is left is CPU time
on our cores that is not ours.

Pre-registered before the first launch
--------------------------------------
    ACCEPT "the width-16 null is foreign load on the pinned CPUs" iff
      (1) n_trusted >= 10, and
      (2) both modes were sampled (at least one launch on each side of the
          lane cut) -- otherwise this run says nothing either way, and
      (3) Spearman rho between foreign_s and effective_lanes <= -0.70
          (more foreign time, fewer lanes realized), and
      (4) the slow-mode launches (effective_lanes < 13) show a median foreign_s
          at least 2x the fast-mode median.

    REJECT otherwise. Two distinct rejections are possible and they are checked
    in this order, because the order matters:

      - Both modes present but foreign_s spanning less than 1.0 CPU-second
        across the whole run is the *strongest* rejection available: the effect
        varies by 3.1 lanes while the putative cause is constant. Note this is
        the opposite reading of a narrow range from the page-backing probe,
        where a narrow range invalidated an ACCEPT. A narrow range can never
        support an ACCEPT and can only support a REJECT once the effect is
        known to have varied -- hence condition (2) is tested first.
      - Both modes present, foreign_s varying, but uncorrelated with lanes.

    A null result here is informative either way: it would mean the lost lanes
    are lost to something inside the process, which is a different
    investigation.

The cut at 13 lanes is not fitted here -- it is the midpoint of the 3.1-lane gap
already published (12.16 to 15.30), chosen before this run.
"""
import argparse
import json
import os
import resource
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import acc0_gap_matrix as H  # noqa: E402
from acc0_w16_page_backing import spearman  # noqa: E402

MODEL, BLOCK, ACC, SESSIONS = "llama", 32, 0, 1
WIDTH = 16
LANE_CUT = 13.0
MIN_TRUSTED = 10
RHO_BAR = -0.70
FOREIGN_MULTIPLE = 2.0
MIN_RANGE_S = 1.0
CLK = os.sysconf("SC_CLK_TCK")


def pinned_cpus():
    return [int(c) for c in H.PIN.split(",")]


def cpu_busy(cpus):
    """Busy CPU-seconds per pinned CPU, summed. Idle and iowait excluded."""
    wanted = {f"cpu{c}" for c in cpus}
    total = 0.0
    with open("/proc/stat") as fh:
        for line in fh:
            name, _, rest = line.partition(" ")
            if name not in wanted:
                continue
            fields = [int(v) for v in rest.split()]
            # user nice system idle iowait irq softirq steal guest guest_nice
            busy = sum(fields[:3]) + sum(fields[5:8])
            total += busy / CLK
    return total


def launch(binary, tokens, reps, cpus):
    env = dict(os.environ)
    env.update({
        "CARGO_INCREMENTAL": "0",
        "PROBE_MODEL": MODEL, "PROBE_BLOCK": str(BLOCK),
        "PROBE_ACCURACY": str(ACC), "PROBE_SESSIONS": str(SESSIONS),
        "PROBE_TOKENS": str(tokens), "PROBE_REPS": str(reps),
        "ONNX_GENAI_CPU_DECODE_THREADS": str(WIDTH),
    })
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    busy0 = cpu_busy(cpus)
    out = subprocess.run(["taskset", "-c", H.PIN, os.path.abspath(binary)],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, env=env, cwd=H.HERE, timeout=900).stdout
    busy1 = cpu_busy(cpus)
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)

    child_cpu = ((r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime))
    row, width_ok = {}, False
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("steady"):
            row["ms_token"] = float(s.split()[1])
        elif s.startswith("cpu phase=steady ") and "unavailable" not in s:
            for field in s.split()[2:]:
                k, _, v = field.partition("=")
                try:
                    row[k] = float(v)
                except ValueError:
                    pass
        elif s.startswith("decode_width"):
            width_ok = s.endswith("as_requested")
    if "ms_token" not in row or "cpu_s_per_token" not in row or not width_ok:
        return None

    row["child_cpu_s"] = child_cpu
    row["pinned_busy_s"] = busy1 - busy0
    row["foreign_s"] = max(0.0, (busy1 - busy0) - child_cpu)
    row["lanes"] = row["cpu_s_per_token"] / (row["ms_token"] / 1000.0)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--launches", type=int, default=12)
    ap.add_argument("--tokens", type=int, default=192)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    cpus = pinned_cpus()
    rows = []
    for i in range(args.launches):
        r = launch(args.binary, args.tokens, args.reps, cpus)
        if r is None:
            print(f"launch {i + 1:2d}  DISCARDED")
            continue
        rows.append(r)
        print(f"launch {i + 1:2d}  ms_token={r['ms_token']:7.3f}  "
              f"lanes={r['lanes']:5.2f}  "
              f"pinned_busy={r['pinned_busy_s']:7.2f}s  "
              f"child_cpu={r['child_cpu_s']:7.2f}s  "
              f"foreign={r['foreign_s']:6.2f}s")
        sys.stdout.flush()

    if not rows:
        print("no trusted launches")
        return 1

    foreign = [r["foreign_s"] for r in rows]
    lanes = [r["lanes"] for r in rows]
    rho = spearman(foreign, lanes)
    slow = [r for r in rows if r["lanes"] < LANE_CUT]
    fast = [r for r in rows if r["lanes"] >= LANE_CUT]
    fs = statistics.median([r["foreign_s"] for r in slow]) if slow else 0.0
    ff = statistics.median([r["foreign_s"] for r in fast]) if fast else 0.0
    multiple = (fs / ff) if ff > 0 else (float("inf") if fs > 0 else 0.0)
    rng = max(foreign) - min(foreign)

    print()
    print(f"n_trusted       : {len(rows)} (bar {MIN_TRUSTED})")
    print(f"lanes           : {min(lanes):.2f} - {max(lanes):.2f}   "
          f"slow(<{LANE_CUT}) {len(slow)} / fast {len(fast)}")
    print(f"foreign_s       : {min(foreign):.2f} - {max(foreign):.2f}   "
          f"range {rng:.2f} (bar {MIN_RANGE_S})")
    print(f"foreign median  : slow {fs:.2f}s vs fast {ff:.2f}s   "
          f"multiple {multiple:.2f} (bar {FOREIGN_MULTIPLE})")
    print(f"spearman rho    : {'n/a' if rho is None else f'{rho:+.4f}'} "
          f"(bar {RHO_BAR})")

    if len(rows) < MIN_TRUSTED:
        verdict = f"REPORT NOTHING -- n_trusted {len(rows)} < {MIN_TRUSTED}"
    elif not slow or not fast:
        verdict = ("REPORT NOTHING -- only one mode was sampled, so a foreign "
                   "time reading has nothing to explain")
    elif rng < MIN_RANGE_S:
        verdict = ("REJECT -- both modes appeared while foreign time on the "
                   "pinned CPUs stayed constant; the cause does not vary")
    elif rho is not None and rho <= RHO_BAR and multiple >= FOREIGN_MULTIPLE:
        verdict = "ACCEPT -- the null is foreign load on the pinned CPUs"
    else:
        verdict = "REJECT -- the lanes are not lost to foreign load"
    print(f"VERDICT         : {verdict}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"rows": rows, "rho": rho, "range": rng,
                       "multiple": multiple, "verdict": verdict}, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
