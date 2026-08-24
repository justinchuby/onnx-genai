#!/usr/bin/env python3
"""Does the width-16 fast/slow mode correspond to a different worker placement?

The claim under test
--------------------
A cross-agent hypothesis (2026-08-24) is that the fast mode of the width-16 A/A
null simply *is* the correctly-placed pool -- that a slow launch confines its
workers to 8 physical cores inside one L3 while a fast launch spreads over 16,
and that `main` "can never enter" the fast mode.

`decode_placement_census.sh` already refutes the standing form of that claim
categorically: on this tree, three identical runs put 15 workers on
`0,2,...,28`, one per physical core, across both L3 instances. But the census
kills each launch before it produces a timing, so it cannot say whether the
*mode* a launch lands in is associated with a different placement. Strictly, the
census shows placement is constant over three launches; it does not show it is
constant over launches *of both modes*, because it never learned which mode it
had.

This closes that gap. Each launch is sampled twice -- placement read from
`/proc/<tid>/Cpus_allowed_list` while the pool is live, then the launch is
allowed to run to completion and its `ms_token` and CPU row parsed -- so every
row carries a placement and a mode taken from the *same process*.

Pre-registered before the first launch
--------------------------------------
    ACCEPT "the mode is placement" iff, over n >= 10 launches sampling both
    modes, the set of pinned CPUs differs between at least one fast launch and
    at least one slow launch.

    REJECT iff both modes appear and every launch reports a byte-identical
    pinned-CPU set.

    REPORT NOTHING if only one mode is sampled -- with a placement that is
    constant by construction, a single-mode run cannot distinguish "placement
    does not vary" from "the mode did not vary either".

This is a categorical rule, not a statistical one, and deliberately so: worker
placement is a set of integers read out of the kernel, not a measurement with an
error bar. One launch of each mode with identical placement settles it. The n>=10
bar exists only to make sampling both modes likely, not to average anything.

Note the asymmetry with a timing A/B: this probe cannot be confounded by host
load, because a competitor changes how fast a pinned worker runs but not which
CPU it is pinned to.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import acc0_gap_matrix as H  # noqa: E402

WIDTH = 16
MIN_TRUSTED = 10
LANE_CUT = 13.0
# Poll for the pool from t=0 rather than sampling at a fixed offset. The first
# version of this file slept 4 s and then read /proc, and every launch it
# managed to sample was slow -- not because slow launches are placed
# differently, but because a fast launch has already finished and torn the pool
# down by then (`wall_s=1.13` at 192 tokens x 2 reps). A probe whose sampling
# instant is later than a fast launch's whole lifetime *selects for the mode it
# is supposed to classify blindly*. Poll fast, take the first complete read.
POLL_S = 0.02
POLL_LIMIT_S = 30.0


def l3_of(cpu):
    return "L3#0" if cpu < 16 else "L3#1"


def worker_cpus(pid):
    """Pinned CPU of every live `onnx-genai-spmd` thread, or None if unpinned."""
    out = []
    taskdir = f"/proc/{pid}/task"
    try:
        tids = os.listdir(taskdir)
    except OSError:
        return out
    for tid in tids:
        try:
            with open(f"{taskdir}/{tid}/comm") as fh:
                if fh.read().strip() != "onnx-genai-spmd":
                    continue
            with open(f"{taskdir}/{tid}/status") as fh:
                for line in fh:
                    if line.startswith("Cpus_allowed_list:"):
                        out.append(line.split()[1].strip())
                        break
        except OSError:
            continue
    return out


def launch(binary, tokens, reps):
    env = dict(os.environ)
    env.update({
        "PROBE_MODEL": "llama", "PROBE_BLOCK": "32", "PROBE_ACCURACY": "0",
        "PROBE_SESSIONS": "1", "PROBE_TOKENS": str(tokens),
        "PROBE_REPS": str(reps),
        "ONNX_GENAI_CPU_DECODE_THREADS": str(WIDTH),
    })
    proc = subprocess.Popen(["taskset", "-c", H.PIN, os.path.abspath(binary)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=env, cwd=H.HERE)
    masks, deadline = [], time.time() + POLL_LIMIT_S
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        seen = worker_cpus(proc.pid)
        # Take the first read that has the full pool; a partial read during
        # spawn would understate the worker count and invent a placement
        # difference that is really a race with thread creation.
        if len(seen) >= WIDTH - 1:
            masks = seen
            break
        if seen and len(seen) > len(masks):
            masks = seen
        time.sleep(POLL_S)
    out, _ = proc.communicate(timeout=900)

    row = {"n_workers": len(masks), "masks": sorted(masks)}
    singles = [int(m) for m in masks if re.fullmatch(r"\d+", m)]
    row["pinned_cpus"] = sorted(singles)
    row["unpinned"] = len(masks) - len(singles)
    row["l3"] = {d: sum(1 for c in singles if l3_of(c) == d)
                 for d in ("L3#0", "L3#1")}
    row["distinct_cores"] = len({c // 2 for c in singles})

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
    have_ms = "ms_token" in row
    have_cpu = "cpu_s_per_token" in row
    if not (have_ms and have_cpu and width_ok and masks):
        return {"discarded": discard_reason(out, masks, width_ok,
                                            have_ms, have_cpu),
                "tail": "\n".join(out.strip().splitlines()[-6:])}
    row["lanes"] = row["cpu_s_per_token"] / (row["ms_token"] / 1000.0)
    row["mode"] = "fast" if row["lanes"] >= LANE_CUT else "SLOW"
    return row


def discard_reason(row_out, masks, width_ok, have_ms, have_cpu):
    """Say which failure happened. A harness that reports a workload failure and
    a parse failure identically cannot tell you which one it hit -- this cost a
    whole reconnaissance run of the page-backing probe, and cost this file its
    first run too."""
    if not masks:
        return "no onnx-genai-spmd threads alive at the sample point"
    if not have_ms:
        return "no `steady` row -- the workload did not reach steady state"
    if not have_cpu:
        return "no usable `cpu phase=steady` row"
    if not width_ok:
        return "decode_width did not report as_requested"
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--launches", type=int, default=14)
    ap.add_argument("--tokens", type=int, default=192)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rows = []
    for i in range(args.launches):
        r = launch(args.binary, args.tokens, args.reps)
        if r is None or "discarded" in r:
            why = r["discarded"] if r else "launch returned nothing"
            print(f"launch {i + 1:2d}  DISCARDED -- {why}")
            if r and r.get("tail"):
                for line in r["tail"].splitlines():
                    print(f"             | {line}")
            sys.stdout.flush()
            continue
        rows.append(r)
        cpus = ",".join(str(c) for c in r["pinned_cpus"])
        print(f"launch {i + 1:2d}  {r['mode']:>4}  ms={r['ms_token']:7.3f}  "
              f"lanes={r['lanes']:5.2f}  workers={r['n_workers']:2d}  "
              f"cores={r['distinct_cores']:2d}  "
              f"{r['l3']['L3#0']}/{r['l3']['L3#1']} L3  cpus={cpus}")
        sys.stdout.flush()

    if not rows:
        print("no trusted launches")
        return 1

    fast = [r for r in rows if r["mode"] == "fast"]
    slow = [r for r in rows if r["mode"] == "SLOW"]
    placements = {tuple(r["pinned_cpus"]) for r in rows}

    print()
    print(f"n_trusted        : {len(rows)} (bar {MIN_TRUSTED})")
    print(f"modes sampled    : fast {len(fast)}, slow {len(slow)}")
    print(f"distinct placements observed : {len(placements)}")
    for p in sorted(placements):
        who = [r["mode"] for r in rows if tuple(r["pinned_cpus"]) == p]
        print(f"  {len(who)}x  {','.join(str(c) for c in p)}   "
              f"modes: {sorted(set(who))}")

    print()
    if len(rows) < MIN_TRUSTED:
        print(f"VERDICT: REPORT NOTHING -- n_trusted {len(rows)} < {MIN_TRUSTED}")
    elif not fast or not slow:
        print("VERDICT: REPORT NOTHING -- only one mode was sampled, so a "
              "constant placement is uninformative")
    elif len(placements) == 1:
        print("VERDICT: REJECT -- both modes appeared on a byte-identical "
              "pinned-CPU set. The mode is not placement.")
    else:
        fp = {tuple(r["pinned_cpus"]) for r in fast}
        sp = {tuple(r["pinned_cpus"]) for r in slow}
        if fp & sp:
            print("VERDICT: REJECT -- placement varies, but at least one "
                  "placement carries both modes, so it does not determine the "
                  "mode.")
        else:
            print("VERDICT: ACCEPT -- fast and slow launches used disjoint "
                  "placements.")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
