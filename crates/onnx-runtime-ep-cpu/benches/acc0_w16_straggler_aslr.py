#!/usr/bin/env python3
"""Does the process's address layout pick the width-16 straggler?

WHY THIS EXISTS
---------------
`acc0_w16_straggler_identity.py` measured 24 trusted width-16 launches and
returned three results that together leave exactly one shape of explanation
standing:

    ops_spread      = 0.0000 in every single launch   (equal assignment)
    median work_skew= 0.5702                          (one lane 57% over mean)
    lane->cpu map   = 1 distinct map over 24 launches (fixed placement)
    straggler lane  = 10 distinct lanes, top 5/24     (the victim moves)

So every lane is handed exactly the same number of ops, every lane is on the
same physical core it was on last launch, and one lane still takes ~57% longer
-- a different lane each time, and the same lane for the whole of any one
process (`straggler_share` up to 0.972 within a launch).

That combination excludes a static work partition and excludes placement. What
is left has to be a property that is (a) fixed for the lifetime of a process,
(b) different between processes, and (c) able to make equal work take unequal
time. The obvious candidate is the **address layout**: ASLR re-bases the weight
arena and every other mapping on each exec, and where a lane's slice lands
relative to cache sets, page colours and 4 KiB/2 MiB boundaries is then fixed
for that process.

This probe does not argue that. It turns ASLR off and looks.

THE TEST
--------
Two arms, interleaved, same binary, same env, differing only in whether the
kernel randomises the address space:

    aslr    the default
    fixed   `setarch -R`, which disables randomisation for the child

If the address layout selects the victim, then with the layout held constant
the victim must stop moving: every `fixed` launch has a byte-identical map and
should therefore elect the same lane. If the victim keeps moving under a fixed
layout, the layout is not the selector.

PRE-REGISTERED RULE (written before the first launch)
-----------------------------------------------------
Trust: `acc0_w16_worker_split.trusted()`, unmodified. `MIN_PER_ARM = 10`
trusted launches per arm or the run reports nothing.

Let `conc(arm)` be the share of that arm's trusted launches won by its most
frequent `straggler_idx`. Chance is 1/15 = 0.067.

    ACCEPT (layout is the selector) iff
        conc(fixed) >= 0.80  AND  conc(aslr) <= 0.50
    REJECT (layout is not the selector) iff
        conc(fixed) <  0.50
    otherwise REPORT NOTHING.

The `conc(aslr) <= 0.50` half of ACCEPT is not decoration: if the victim were
concentrated in *both* arms, the concentration would not be attributable to
the arm at all, and the honest reading would be that this probe reproduced a
fixed lane rather than a layout effect.

CONTROLS
--------
1.  **The knob is verified, not trusted.** `#1792` on this project is a
    user-facing placement control that is completely inert -- `off`,
    `numa-split` and an explicit CPU list all produce byte-identical
    placement -- and it shipped because nobody checked that the knob moved
    anything. So before any launch, this probe execs a child under each arm
    twice and compares the first line of `/proc/self/maps`. `fixed` must
    produce the same base twice and `aslr` must produce different bases;
    if either fails, the run aborts without measuring. A `setarch -R` that
    silently did nothing would otherwise produce `conc(fixed) == conc(aslr)`
    and be reported as a clean REJECT.
2.  **The phenomenon must survive the manipulation.** If median `work_skew` in
    the `fixed` arm falls below `SKEW_FLOOR = 0.30`, there is no straggler left
    to attribute and the comparison is vacuous. That is reported as its own
    outcome -- and it would itself be a strong result, since it would mean
    disabling ASLR removes the imbalance.
3.  **Assignment must stay equal in both arms** (`ops_spread <= 0.01`), or the
    arms differ in what they were asked to do rather than in where it landed.
4.  Arms are interleaved launch-by-launch, not run in blocks, so a drift in
    host conditions cannot be absorbed by one arm.

Every branch of this rule, including ACCEPT, is a publishable outcome. REJECT
is the one the current evidence does not predict but is fully expected to be
possible, in which case the candidate list is empty again and this record says
so.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import acc0_w16_straggler_identity as I  # noqa: E402
import acc0_w16_worker_split as W  # noqa: E402

MIN_PER_ARM = 10
LAYOUT_ACCEPT = 0.80
LAYOUT_REJECT = 0.50
ASLR_MAX_CONC = 0.50
SKEW_FLOOR = 0.30
OPS_SPREAD_MAX = 0.01
WIDTH = 16
ARMS = ("aslr", "fixed")


def map_base(prefix):
    """First line of a child's `/proc/self/maps`, as that child sees it."""
    r = subprocess.run(
        f"{prefix}cat /proc/self/maps",
        shell=True, capture_output=True, text=True, timeout=60,
    )
    return r.stdout.splitlines()[0].split()[0] if r.stdout else None


def verify_knob():
    """Abort unless `setarch -R` demonstrably changes what it claims to."""
    fixed = [map_base("setarch -R ") for _ in range(3)]
    rand = [map_base("") for _ in range(6)]
    ok_fixed = len(set(fixed)) == 1 and all(fixed)
    ok_rand = len(set(rand)) > 1
    print(f"control: setarch -R bases {fixed} -> {'CONSTANT' if ok_fixed else 'NOT CONSTANT'}")
    print(f"control: default    bases {len(set(rand))} distinct of {len(rand)} "
          f"-> {'RANDOMISED' if ok_rand else 'NOT RANDOMISED'}")
    if not (ok_fixed and ok_rand):
        print("VERDICT: ABORT -- the arm's mechanism is not in effect; nothing measured.")
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--launches", type=int, default=16, help="per arm")
    ap.add_argument("--tokens", type=int, default=192)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--blocktime", type=int, default=0)
    ap.add_argument("--quiet-limit", type=int, default=40)
    ap.add_argument("--out", default="bb/w16_straggler_aslr.json")
    ap.add_argument("--replay", default=None)
    args = ap.parse_args()

    if args.replay:
        return report(json.loads(pathlib.Path(args.replay).read_text()), args.launches)
    if not verify_knob():
        return 1
    print()

    records = []
    for i in range(args.launches):
        # Interleaved, and the order alternates so neither arm always follows
        # the other into a warmed page cache.
        order = ARMS if i % 2 == 0 else tuple(reversed(ARMS))
        for arm in order:
            sub = argparse.Namespace(**vars(args))
            sub.binary = ("setarch -R " if arm == "fixed" else "") + args.binary
            with W.H.LoadWatch() as watch:
                workers = W.parse_workers(W.run_width(sub, WIDTH))
            rec = {"widths": {str(WIDTH): W.derive(workers)},
                   "peak": watch.peak, "peak_limit": args.quiet_limit}
            keep = W.trusted(rec)
            row = {"launch": i, "arm": arm, "trusted": keep,
                   "table": I.lane_table(workers) if keep else None}
            records.append(row)
            pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(args.out).write_text(json.dumps(records, indent=1))
            t = row["table"]
            if t:
                print(f"L{i:<2} {arm:<5} wall={t['wall_s']:.3f} "
                      f"ops_spread={t['ops_spread']:.4f} skew={t['work_skew']:.3f} "
                      f"straggler=idx{t['straggler_idx']}/cpu{t['straggler_cpu']}", flush=True)
            else:
                print(f"L{i:<2} {arm:<5} UNTRUSTED (peak={watch.peak})", flush=True)
            time.sleep(1)

    return report(records, args.launches)


def report(records, attempted) -> int:
    by = {a: [r["table"] for r in records if r["arm"] == a and r["table"]] for a in ARMS}
    print()
    for a in ARMS:
        print(f"{a:<5} trusted {len(by[a])}/{attempted}")
    if any(len(by[a]) < MIN_PER_ARM for a in ARMS):
        print(f"VERDICT: REPORT NOTHING -- need {MIN_PER_ARM} trusted launches per arm")
        return 0

    conc, top = {}, {}
    for a in ARMS:
        c = collections.Counter(t["straggler_idx"] for t in by[a])
        top[a], n = c.most_common(1)[0]
        conc[a] = n / len(by[a])
        skew = statistics.median(t["work_skew"] for t in by[a])
        spread = statistics.median(t["ops_spread"] for t in by[a])
        maps = len({json.dumps(sorted((int(k), v) for k, v in t["lane_cpu"].items())) for t in by[a]})
        print()
        print(f"{a}: top lane idx{top[a]} {n}/{len(by[a])} = {conc[a]:.3f}  (chance 0.067)")
        print(f"   lanes {dict(sorted(c.items()))}")
        print(f"   median work_skew {skew:.3f}   median ops_spread {spread:.4f}   "
              f"lane->cpu maps {maps}")

    print()
    fixed_skew = statistics.median(t["work_skew"] for t in by["fixed"])
    if fixed_skew < SKEW_FLOOR:
        print(f"CONTROL 2 FIRED: fixed-arm work_skew {fixed_skew:.3f} < {SKEW_FLOOR}.")
        print("VERDICT: the imbalance does not survive disabling ASLR -- report that,")
        print("         not a concentration comparison, which would be vacuous.")
        return 0
    for a in ARMS:
        sp = statistics.median(t["ops_spread"] for t in by[a])
        if sp > OPS_SPREAD_MAX:
            print(f"CONTROL 3 FIRED: {a} ops_spread {sp:.4f} > {OPS_SPREAD_MAX}; arms differ in assignment.")
            print("VERDICT: REPORT NOTHING")
            return 0

    if conc["fixed"] >= LAYOUT_ACCEPT and conc["aslr"] <= ASLR_MAX_CONC:
        print(f"VERDICT: ACCEPT -- with the address layout held fixed the victim stops moving "
              f"(conc {conc['fixed']:.3f} vs {conc['aslr']:.3f} under ASLR).")
        print(f"         The width-16 straggler is selected by the process address layout; "
              f"lane idx{top['fixed']} is the victim of this particular layout.")
    elif conc["fixed"] < LAYOUT_REJECT:
        print(f"VERDICT: REJECT -- a byte-identical address layout still moves the victim "
              f"(conc {conc['fixed']:.3f} < {LAYOUT_REJECT}).")
        print("         Address layout is not the selector. Candidate list is empty again.")
    else:
        print(f"VERDICT: REPORT NOTHING -- conc(fixed)={conc['fixed']:.3f} landed between "
              f"{LAYOUT_REJECT} and {LAYOUT_ACCEPT}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
