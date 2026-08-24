#!/usr/bin/env python3
"""Is the width-16 straggler a work-assignment defect, a lane, or a CPU?

WHY THIS EXISTS
---------------
`acc0_w16_mode_worker_split.py` closed the mode question (REPORT NOTHING: under
worker profiling the width-16 `wall_s` values are one distribution, not two) but
surfaced an aggregate lead it did not chase: straggler wait was 0.313 of the
window, one worker held 0.565 of `last_arrivals` against a chance share of
0.067, and `work_skew` was 0.562.

That lead sits directly on an open contradiction in the ledger.
`output_chunk_len_for` returns `n.div_ceil(tasks)` with `tasks <= threads`, and
every llama projection width divides evenly by 16 -- so a static reading of the
source predicts *no* skew at all. I deliberately did not resolve that from the
source, because "the code looks even" is not a measurement and the barrier does
not care what the code looks like.

Re-reading the 16 stored launches of the mode run gives the shape the lead
actually has:

    straggler_idx across 16 launches: idx0 x5, idx14 x3, idx9 x2, idx1 x2,
                                      idx3/4/5/8 x1 each
    straggler_share within a launch:  up to 0.972

So the straggler is *persistent inside a process* and its *identity changes
between processes*. A static `div_ceil` partition cannot produce that: it would
name the same lane every launch. Something per-process and durable is picking a
victim -- and `migrations=0` says whatever it picks, it keeps.

THE DISCRIMINATOR, AND WHY IT NEEDS NO NEW EP CODE
--------------------------------------------------
The existing instrument already reports `timed_ops` per worker, and `derive()`
throws it away -- it takes `workers[0]["timed_ops"]` as *the* op count, which
silently assumes every lane got the same number. That assumption is the exact
question, so this probe stops assuming it and reads all fifteen.

    timed_ops UNEQUAL  ->  lanes are handed different amounts of work.
                           `output_chunk_len_for` is implicated and the
                           contradiction resolves in favour of the measurement.

    timed_ops EQUAL    ->  every lane was handed the same work and one lane
       and work_ns          took longer to do it. The partition is exonerated
       SKEWED               and the excess is execution time, not assignment.
                            The question becomes "why is this lane slow",
                            which is a placement/memory question.

These are mutually exclusive and jointly exhaustive over the observed skew, so
the run cannot come back ambiguous on the headline.

PRE-REGISTERED RULE (written before the first launch)
-----------------------------------------------------
Trust: a launch counts only if `acc0_w16_worker_split.trusted()` accepts it --
this probe adds no trust criterion of its own and reimplements nothing.
`MIN_LAUNCHES = 8` trusted launches are required for any verdict at all.

H1 -- assignment.  For each launch compute
        ops_spread = (max(timed_ops) - min(timed_ops)) / mean(timed_ops)
    ASSIGNMENT is the mechanism iff the median `ops_spread` over trusted
    launches is >= OPS_SPREAD_ACCEPT (0.10), i.e. lanes differ by >=10% in the
    number of ops they were given.
    It is REJECTED iff the median is <= OPS_SPREAD_REJECT (0.01). Between the
    two, REPORT NOTHING and say so.

H2 -- lane.  A lane index is structural iff it is the straggler in at least
    DOMINANT_LANE_SHARE (0.5) of trusted launches. (Chance share is 1/15 =
    0.067; the stored run's best was 0.31, which does not clear this.)

H3 -- cpu.  Two parts, and the first is categorical and needs no timing:
    (a) is the lane->cpu map identical across every trusted launch?
    (b) is the straggler's *cpu* concentrated at >= DOMINANT_CPU_SHARE (0.5)?
    If (a) holds and neither H2 nor H3(b) clears, then the victim is neither a
    fixed lane nor a fixed CPU under a fixed placement, and placement is
    REJECTED as the selector -- which is the outcome the stored data predicts.

Every one of these can come back negative. If they all do, the honest output is
"the straggler is real, persistent within a process, and unexplained", plus the
H1 verdict, which is worth publishing on its own because it settles whether the
partition is at fault.

CONTROLS
--------
* `timed_ops == 0` for any lane means profiling was not actually on. That is
  already `derive()`'s vacuity guard and it is kept: a lane with no timed ops
  would otherwise read as a perfectly balanced zero.
* The lane->cpu map is read from the same profile records as the timing, in the
  same launch, so it cannot describe a different process than the one measured.
  This is the sampling-instant rule that cost me four defects: never pair a
  placement read with a timing read taken at a different instant.
* An A/A-style null is not meaningful here (there is one arm), so instead the
  chance share 1/n_workers is printed next to every concentration figure, so a
  reader can see what "no effect" would look like.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics
import time
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import acc0_w16_worker_split as W  # noqa: E402

MIN_LAUNCHES = 8
OPS_SPREAD_ACCEPT = 0.10
OPS_SPREAD_REJECT = 0.01
DOMINANT_LANE_SHARE = 0.5
DOMINANT_CPU_SHARE = 0.5
WIDTH = 16


def lane_table(workers):
    """Everything this probe needs from one launch's profile records."""
    ops = [w["timed_ops"] for w in workers]
    work = [w["work_ns"] / 1e9 for w in workers]
    arr = [w["last_arrivals"] for w in workers]
    mean_ops = statistics.fmean(ops)
    mean_work = statistics.fmean(work)
    straggler = arr.index(max(arr))
    return {
        "n": len(workers),
        "lane_cpu": {w["idx"]: w["cpu"] for w in workers},
        "ops": ops,
        "ops_spread": ((max(ops) - min(ops)) / mean_ops) if mean_ops > 0 else 0.0,
        "work_skew": (max(work) / mean_work - 1.0) if mean_work > 0 else 0.0,
        "straggler_idx": workers[straggler]["idx"],
        "straggler_cpu": workers[straggler]["cpu"],
        "straggler_share": (max(arr) / sum(arr)) if sum(arr) else 0.0,
        "slowest_idx": workers[work.index(max(work))]["idx"],
        "wall_s": workers[0]["wall_s"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--launches", type=int, default=14)
    ap.add_argument("--tokens", type=int, default=192)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--blocktime", type=int, default=0)
    ap.add_argument("--quiet-limit", type=int, default=40)
    ap.add_argument("--out", default="bb/w16_straggler_identity.json")
    ap.add_argument("--replay", default=None)
    args = ap.parse_args()

    if args.replay:
        records = json.loads(pathlib.Path(args.replay).read_text())
        return report([r["table"] for r in records if r["table"]], len(records))

    records = []
    for i in range(args.launches):
        # `run_width` + `derive` + `trusted` are the validated instrument; this
        # probe reuses all three and only keeps the raw records that `derive`
        # discards. `one_launch` is not used because it drops them.
        with W.H.LoadWatch() as watch:
            workers = W.parse_workers(W.run_width(args, WIDTH))
        rec = {
            "widths": {str(WIDTH): W.derive(workers)},
            "peak": watch.peak,
            "peak_limit": args.quiet_limit,
        }
        keep = W.trusted(rec)
        row = {"launch": i, "trusted": keep, "table": lane_table(workers) if keep else None}
        records.append(row)
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps(records, indent=1))
        t = row["table"]
        if t:
            print(
                f"L{i:<2} wall={t['wall_s']:.3f} ops_spread={t['ops_spread']:.4f} "
                f"skew={t['work_skew']:.3f} straggler=idx{t['straggler_idx']}/cpu{t['straggler_cpu']} "
                f"slowest=idx{t['slowest_idx']}",
                flush=True,
            )
        else:
            print(f"L{i:<2} UNTRUSTED (peak={watch.peak})", flush=True)
        time.sleep(1)

    return report([r["table"] for r in records if r["table"]], args.launches)


def report(good, attempted) -> int:
    print()
    print(f"trusted {len(good)}/{attempted}")
    if len(good) < MIN_LAUNCHES:
        print(f"VERDICT: REPORT NOTHING -- need {MIN_LAUNCHES} trusted launches")
        return 0

    n = good[0]["n"]
    chance = 1.0 / n

    med_spread = statistics.median(t["ops_spread"] for t in good)
    med_skew = statistics.median(t["work_skew"] for t in good)
    print(f"median ops_spread = {med_spread:.4f}  (accept >={OPS_SPREAD_ACCEPT}, reject <={OPS_SPREAD_REJECT})")
    print(f"median work_skew  = {med_skew:.4f}")
    if med_spread >= OPS_SPREAD_ACCEPT:
        h1 = "ACCEPT: lanes are handed unequal work; the partition is implicated"
    elif med_spread <= OPS_SPREAD_REJECT:
        h1 = "REJECT: every lane got the same ops; the excess is execution time, not assignment"
    else:
        h1 = "REPORT NOTHING: ops_spread landed between the pre-registered bounds"
    print(f"H1 assignment -> {h1}")

    lanes = collections.Counter(t["straggler_idx"] for t in good)
    cpus = collections.Counter(t["straggler_cpu"] for t in good)
    top_lane, top_lane_n = lanes.most_common(1)[0]
    top_cpu, top_cpu_n = cpus.most_common(1)[0]
    print()
    print(f"straggler lane concentration: idx{top_lane} {top_lane_n}/{len(good)} "
          f"= {top_lane_n / len(good):.3f}  (chance {chance:.3f}, need >={DOMINANT_LANE_SHARE})")
    print(f"straggler cpu  concentration: cpu{top_cpu} {top_cpu_n}/{len(good)} "
          f"= {top_cpu_n / len(good):.3f}  (chance {chance:.3f}, need >={DOMINANT_CPU_SHARE})")
    print(f"  lanes seen: {dict(sorted(lanes.items()))}")
    print(f"  cpus  seen: {dict(sorted(cpus.items()))}")

    maps = {json.dumps(sorted((int(k), v) for k, v in t["lane_cpu"].items())) for t in good}
    stable = len(maps) == 1
    print(f"lane->cpu maps distinct across trusted launches: {len(maps)} "
          f"({'STABLE' if stable else 'VARIES'})")
    if stable:
        print(f"  map: {json.dumps(sorted((int(k), v) for k, v in good[0]['lane_cpu'].items()))}")

    h2 = top_lane_n / len(good) >= DOMINANT_LANE_SHARE
    h3 = top_cpu_n / len(good) >= DOMINANT_CPU_SHARE
    print(f"H2 lane -> {'ACCEPT idx' + str(top_lane) if h2 else 'REJECT'}")
    print(f"H3 cpu  -> {'ACCEPT cpu' + str(top_cpu) if h3 else 'REJECT'}")
    if stable and not h2 and not h3:
        print("H3(a) placement is fixed and identical every launch, yet the victim moves:")
        print("      the selector is neither a lane nor a CPU. Placement REJECTED as selector.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
