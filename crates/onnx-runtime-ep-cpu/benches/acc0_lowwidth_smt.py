#!/usr/bin/env python3
"""Is the t=2 sub-one-core CPU reading two workers sharing one physical core?

The claim under test
--------------------
An earlier table of mine reported `Percent of CPU` of 98 / 71 / 186 at decode
widths 1 / 2 / 4. The t=2 cell is the odd one: 71% is *less than a single core*
while two workers exist, and adding the second worker apparently *reduced* total
CPU consumed. I attributed it provisionally to wake/park latency.

A cross-agent hypothesis (2026-08-24) is that this is a placement artifact --
that the two workers were pinned to cpus 0 and 1, which are SMT siblings of one
physical core, so "t=2" was really one core, and the reading needs re-taking on
a placement that gives each worker its own core.

Why the arithmetic already argues against that, and why I am measuring anyway
-----------------------------------------------------------------------------
The same message supplies the fact that refutes it. A pinned scalar probe on
this host measured cpu 1 delivering 55% of the work of an uncontended CPU while
reporting a CPU *share* of 1.000, because its sibling cpu 0 was busy. That is
the defining property of SMT: it steals throughput without stealing CPU-time. A
thread on a contended logical CPU still accrues a full second of CPU per second
of wall.

So two spinning workers co-located on one physical core should read ~200% of
CPU with poor throughput -- SMT contention cannot produce a reading *below* one
core. A sub-one-core reading means the workers were not on-CPU at all, which is
a parking observation, not a placement one.

That is a deduction from someone else's numbers, which is exactly the move this
ledger keeps catching out. So it gets measured, with the co-location forced
directly rather than inferred.

Design
------
Three placement arms per width, all with `ONNX_GENAI_CPU_DECODE_THREADS` set
explicitly (never the default, which derives from cpuset size):

    cores     -- taskset to `w` even CPUs: one per physical core.
    siblings  -- taskset to the first `w` CPUs: adjacent, so at w=2 that is
                 cpus 0,1 = one physical core, and at w=4 two cores. This is
                 the hypothesised bad configuration, forced.
    default   -- no taskset; whatever the EP chooses on this tree.

Placement is read from `/proc/<tid>/Cpus_allowed_list` for every live
`onnx-genai-spmd` thread in the same process that produces the timing, polled
from t=0 (a fixed sample offset selects for slow launches -- see
acc0_w16_mode_placement.py).

`lanes` = cpu_s per wall second = `Percent of CPU` / 100, reported alongside a
work-completed rate, because a CPU-time metric structurally cannot see SMT
contention and this run contains a deliberately SMT-contended arm.

Pre-registered, before the first launch
---------------------------------------
    ACCEPT "the t=2 anomaly is SMT co-location" iff the forced `siblings` arm
    at w=2 reproduces lanes < 1.0 while the `cores` arm at w=2 does not.

    REJECT iff the `siblings` arm at w=2 shows lanes >= 1.5 -- i.e. co-locating
    two workers on one core does not suppress CPU-time, so co-location cannot
    be what a sub-one-core reading is made of.

    REPORT NOTHING if the `cores` arm does not achieve `w` distinct physical
    cores, or the `siblings` arm at w=2 does not achieve exactly 1, since then
    the intended contrast was never applied.

The rule is stated over the *forced* arms, not the default one, so it cannot be
voided by whatever placement this tree happens to choose.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import acc0_gap_matrix as H  # noqa: E402
from acc0_w16_mode_placement import worker_cpus, l3_of  # noqa: E402

POLL_S = 0.02
POLL_LIMIT_S = 30.0
SIBLING_LANE_FLOOR = 1.5
ANOMALY_LANE_CUT = 1.0


def pin_for(arm, width):
    if arm == "cores":
        return ",".join(str(c) for c in range(0, 2 * width, 2))
    if arm == "siblings":
        return ",".join(str(c) for c in range(width))
    return None


def proc_cpuset(pid):
    """The whole process's allowed CPUs, from the main thread.

    Counting distinct cores from the `onnx-genai-spmd` threads alone undercounts
    by one: width `w` builds `w - 1` named worker threads and runs the w-th lane
    on the dispatcher thread itself (which is why width 16 shows 15 spmd
    threads while measuring ~16 lanes of CPU). The forced contrast in this probe
    is applied with `taskset` to the *process*, so the process cpuset is both
    the thing that was actually manipulated and immune to that off-by-one.
    """
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("Cpus_allowed_list:"):
                    return line.split()[1].strip()
    except OSError:
        pass
    return None


def parse_cpu_list(spec):
    out = []
    if not spec:
        return out
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part.isdigit():
            out.append(int(part))
    return out


def one_launch(binary, arm, width, tokens, reps):
    env = dict(os.environ)
    env.update({
        "PROBE_MODEL": "llama", "PROBE_BLOCK": "32", "PROBE_ACCURACY": "0",
        "PROBE_SESSIONS": "1", "PROBE_TOKENS": str(tokens),
        "PROBE_REPS": str(reps),
        "ONNX_GENAI_CPU_DECODE_THREADS": str(width),
    })
    pin = pin_for(arm, width)
    cmd = ([] if pin is None else ["taskset", "-c", pin]) + [os.path.abspath(binary)]
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, env=env,
                            cwd=H.HERE)
    # Wait for `width` masks that are *all* single CPUs. Accepting the first
    # read with `width` entries raced the EP's own pinning: a thread sampled
    # between spawn and sched_setaffinity still carries the inherited mask, so
    # a two-worker arm reported one pinned CPU and one wide mask, and the
    # distinct-core count came out 1 instead of 2. That would have voided the
    # contrast gate on a probe whose whole job is counting cores.
    masks, cpuset, deadline = [], None, time.time() + POLL_LIMIT_S
    while time.time() < deadline:
        # Re-read every iteration and keep the latest. Reading once at the
        # first iteration caught `taskset` before it had applied the mask and
        # reported 0-31 for every arm -- the same sample-instant defect as the
        # worker-mask race above, one level up. The settled value is the one
        # present when the pool is complete.
        latest = proc_cpuset(proc.pid)
        if latest:
            cpuset = latest
        if proc.poll() is not None:
            break
        seen = worker_cpus(proc.pid)
        if len(seen) >= width and all(m.isdigit() for m in seen):
            masks = seen
            break
        if len(seen) > len(masks):
            masks = seen
        time.sleep(POLL_S)
    out, _ = proc.communicate(timeout=900)
    wall = time.time() - t0

    row = {"arm": arm, "width": width, "wall_s": wall,
           "n_workers": len(masks), "masks": sorted(masks)}
    singles = [int(m) for m in masks if m.isdigit()]
    row["pinned_cpus"] = sorted(singles)
    row["worker_cores"] = len({c // 2 for c in singles})
    row["cpuset"] = cpuset
    set_cpus = parse_cpu_list(cpuset)
    row["cpuset_cpus"] = set_cpus
    row["distinct_cores"] = len({c // 2 for c in set_cpus}) if set_cpus else None
    row["l3"] = sorted({l3_of(c) for c in set_cpus})

    width_ok = False
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
        elif "confined the process to" in s:
            # The EP's own account of the cpuset, independent of the /proc read.
            row["ep_confined"] = s[s.index("confined the process to"):]
    row["width_ok"] = width_ok
    # Width 1 runs the decode inline and never builds an SPMD pool, so an empty
    # mask list is the correct observation there, not a failed sample.
    if width == 1 and not masks:
        row["no_pool"] = True
    if "ms_token" not in row or "cpu_s_per_token" not in row or (
            not masks and width > 1):
        row["discarded"] = (
            "no onnx-genai-spmd threads alive at the sample point" if not masks
            else "no `steady` row" if "ms_token" not in row
            else "no usable `cpu phase=steady` row")
        row["tail"] = "\n".join(out.strip().splitlines()[-5:])
        return row
    row["lanes"] = row["cpu_s_per_token"] / (row["ms_token"] / 1000.0)
    row["pct_cpu"] = 100.0 * row["lanes"]
    row["tok_per_s"] = 1000.0 / row["ms_token"]
    return row


def med(rows, key):
    vals = [r[key] for r in rows if key in r and "discarded" not in r]
    return statistics.median(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--reps-per-cell", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=192)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    cells = []
    for width in (1, 2, 4):
        for arm in ("cores", "siblings", "default"):
            if arm == "siblings" and width == 1:
                continue  # degenerate: one CPU is one CPU
            cells.append((arm, width))

    rows = []
    # Interleave repetitions so a drift in host state cannot land on one arm.
    for rep in range(args.reps_per_cell):
        for arm, width in cells:
            r = one_launch(args.binary, arm, width, args.tokens, args.reps)
            r["rep"] = rep
            rows.append(r)
            if "discarded" in r:
                print(f"  {arm:<9} w={width}  DISCARD  {r['discarded']}")
            else:
                print(f"  {arm:<9} w={width}  ms={r['ms_token']:7.3f}  "
                      f"lanes={r['lanes']:5.2f}  pct_cpu={r['pct_cpu']:6.1f}  "
                      f"cores={r['distinct_cores']}  cpuset={r['cpuset']}")
            sys.stdout.flush()

    # Persist before summarising: the previous run lost 5 minutes of clean data
    # to a formatting error in the verdict block, because the dump came last.
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)

    print()
    print(f"{'arm':<9} {'w':>2}  {'n':>2}  {'ms/tok':>8}  {'lanes':>6}  "
          f"{'pct_cpu':>8}  {'cores':>5}")
    table = {}
    for arm, width in cells:
        sel = [r for r in rows
               if r["arm"] == arm and r["width"] == width and "discarded" not in r]
        table[(arm, width)] = sel
        if not sel:
            print(f"{arm:<9} {width:>2}  {'0':>2}  (all discarded)")
            continue
        print(f"{arm:<9} {width:>2}  {len(sel):>2}  {med(sel,'ms_token'):8.3f}  "
              f"{med(sel,'lanes'):6.2f}  {med(sel,'pct_cpu'):8.1f}  "
              f"{str(sel[0]['distinct_cores']):>5}")

    sib2 = table.get(("siblings", 2), [])
    cor2 = table.get(("cores", 2), [])
    print()
    if not sib2 or not cor2:
        print("VERDICT: REPORT NOTHING -- a w=2 arm produced no trusted rows")
    elif sib2[0]["distinct_cores"] != 1 or cor2[0]["distinct_cores"] != 2:
        print(f"VERDICT: REPORT NOTHING -- intended contrast not applied "
              f"(siblings cores={sib2[0]['distinct_cores']} want 1, "
              f"cores cores={cor2[0]['distinct_cores']} want 2)")
    else:
        sl, cl = med(sib2, "lanes"), med(cor2, "lanes")
        print(f"w=2 siblings lanes={sl:.2f}   w=2 cores lanes={cl:.2f}")
        if sl < ANOMALY_LANE_CUT <= cl:
            print("VERDICT: ACCEPT -- forcing two workers onto one physical "
                  "core reproduces the sub-one-core reading.")
        elif sl >= SIBLING_LANE_FLOOR:
            print(f"VERDICT: REJECT -- co-locating two workers on one core "
                  f"leaves CPU-time at {sl:.2f} lanes (>= "
                  f"{SIBLING_LANE_FLOOR}). SMT contention does not suppress "
                  f"CPU-time, so it cannot be what a sub-one-core reading is "
                  f"made of.")
        else:
            print(f"VERDICT: REPORT NOTHING -- siblings lanes {sl:.2f} falls "
                  f"between the two pre-registered thresholds "
                  f"({ANOMALY_LANE_CUT}, {SIBLING_LANE_FLOOR}).")

    # Throughput is reported because a CPU-time metric cannot see SMT at all;
    # if co-location costs anything, it must show up here.
    if sib2 and cor2:
        print(f"work-completed check: siblings {med(sib2,'tok_per_s'):.2f} tok/s "
              f"vs cores {med(cor2,'tok_per_s'):.2f} tok/s "
              f"(ratio {med(sib2,'tok_per_s')/med(cor2,'tok_per_s'):.3f})")


if __name__ == "__main__":
    main()
