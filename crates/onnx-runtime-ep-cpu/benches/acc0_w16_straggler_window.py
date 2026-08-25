#!/usr/bin/env python3
"""Is the width-16 "straggler" a fixed lane, or the maximum of noise?

WHY THIS EXISTS
---------------
Three probes have now excluded every mechanism proposed for the width-16
straggler, and each exclusion was clean:

    assignment      `ops_spread` = 0.0000 in 24/24 launches
    placement       one lane->cpu map across 24 launches, victim still moves
    address layout  `setarch -R` gives conc 0.267, identical to ASLR's 0.267

When three sharp hypotheses about *which lane is slow* all fail in the same
direction, the assumption they share is the thing to test. Every one of them
assumed there is a lane that is slow. That assumption comes from two numbers,
and neither has a null model:

    work_skew       = max(work_ns) / mean(work_ns) - 1
    straggler_share = max(last_arrivals) / sum(last_arrivals)

`work_skew` is a maximum over fifteen lanes. Take fifteen samples of *any*
noisy quantity and the maximum sits above the mean; at 15 lanes a completely
symmetric jitter distribution yields a positive `work_skew` forever. The
metric cannot return zero, so a large value is not by itself evidence of an
imbalance -- and I have been reading it as if it were, across three records.

THE TEST, AND WHY WINDOW LENGTH SEPARATES THEM
----------------------------------------------
`straggler_share` is a *cumulative* count over every op in the profile window,
so its behaviour as the window grows distinguishes the two explanations
without needing any new counter in the EP:

    a genuinely slow lane   is last on nearly every op, so its share is
                            roughly constant as the window grows, and tends
                            toward 1.0

    max-of-noise            is whichever lane won the most coin flips, so its
                            share regresses toward the chance share 1/n as the
                            window grows -- the excess above 1/n shrinks like
                            1/sqrt(ops)

Same binary, same placement, same everything; only `--tokens` changes, which
scales the number of profiled ops per launch.

PRE-REGISTERED RULE (written before the first launch)
-----------------------------------------------------
Trust: `acc0_w16_worker_split.trusted()`, unmodified. `MIN_PER_ARM = 8`.

Let `excess(arm) = median(straggler_share) - 1/n`, the concentration above
chance, and let `R = excess(long) / excess(short)`.

    FIXED LANE   iff R >= 0.70   -- concentration substantially survives a
                                    4x longer window
    NOISE        iff R <= 0.40   -- concentration decays toward chance, and
                                    the "straggler" is a max-of-noise artefact
    otherwise REPORT NOTHING.

0.40 is chosen because pure coin-flipping predicts the excess falling by
1/sqrt(4) = 0.50 for a 4x window, so a decay at or below 0.40 is at least as
fast as chance, while 0.70 is comfortably above it. The band between is
deliberately wide: this is a shape test on medians, not a precise estimator,
and a value inside it does not license either story.

CONTROLS
--------
1.  **The window really did grow.** `ops` is read from the profile records and
    the long arm must show >= 3x the short arm's ops. A `--tokens` that did
    not reach the profiled region would otherwise produce two identical arms
    and a free FIXED LANE verdict -- the same shape as the inert knob in
    `#1792` and as my own four sampling-instant defects, where the instrument
    reported confidently on something it had not varied.
2.  **Assignment stays equal** in both arms (`ops_spread <= 0.01`), so the
    arms differ only in window length.
3.  Arms are interleaved and the order alternates, so host drift cannot be
    absorbed by one arm.
4.  The chance share `1/n` is printed next to every figure, so a reader can
    see what "no effect" looks like without recomputing it.

NOISE is the outcome that costs me the most: it would retract the "straggler
lead" recorded in `docs/benchmarks/2026-08-24-acc0-straggler-lead.md` and the
0.565-share figure in the ledger, and it would mean three probes were built to
find the mechanism of an artefact. It is written into the rule for exactly
that reason.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import acc0_w16_straggler_identity as I  # noqa: E402
import acc0_w16_worker_split as W  # noqa: E402

MIN_PER_ARM = 8
FIXED_LANE_R = 0.70
NOISE_R = 0.40
OPS_SPREAD_MAX = 0.01
OPS_GROWTH_MIN = 3.0
WIDTH = 16


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--launches", type=int, default=10, help="per arm")
    ap.add_argument("--short", type=int, default=192)
    ap.add_argument("--long", type=int, default=768)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--blocktime", type=int, default=0)
    ap.add_argument("--quiet-limit", type=int, default=40)
    ap.add_argument("--out", default="bb/w16_straggler_window.json")
    ap.add_argument("--replay", default=None)
    args = ap.parse_args()

    if args.replay:
        return report(json.loads(pathlib.Path(args.replay).read_text()), args.launches)

    arms = (("short", args.short), ("long", args.long))
    records = []
    for i in range(args.launches):
        order = arms if i % 2 == 0 else tuple(reversed(arms))
        for name, tokens in order:
            sub = argparse.Namespace(**vars(args))
            sub.tokens = tokens
            with W.H.LoadWatch() as watch:
                workers = W.parse_workers(W.run_width(sub, WIDTH))
            rec = {"widths": {str(WIDTH): W.derive(workers)},
                   "peak": watch.peak, "peak_limit": args.quiet_limit}
            keep = W.trusted(rec)
            table = I.lane_table(workers) if keep else None
            if table:
                table["ops"] = workers[0]["timed_ops"]
                table["straggler_share"] = max(w["last_arrivals"] for w in workers) / max(
                    1, sum(w["last_arrivals"] for w in workers))
            row = {"launch": i, "arm": name, "tokens": tokens, "trusted": keep, "table": table}
            records.append(row)
            pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(args.out).write_text(json.dumps(records, indent=1))
            if table:
                print(f"L{i:<2} {name:<5} tok={tokens:<4} ops={table['ops']:<6} "
                      f"share={table['straggler_share']:.4f} skew={table['work_skew']:.3f} "
                      f"idx{table['straggler_idx']}", flush=True)
            else:
                print(f"L{i:<2} {name:<5} tok={tokens:<4} UNTRUSTED (peak={watch.peak})", flush=True)
            time.sleep(1)

    return report(records, args.launches)


def report(records, attempted) -> int:
    by = {a: [r["table"] for r in records if r["arm"] == a and r["table"]]
          for a in ("short", "long")}
    print()
    for a in ("short", "long"):
        print(f"{a:<5} trusted {len(by[a])}/{attempted}")
    if any(len(by[a]) < MIN_PER_ARM for a in ("short", "long")):
        print(f"VERDICT: REPORT NOTHING -- need {MIN_PER_ARM} trusted launches per arm")
        return 0

    n = by["short"][0]["n"]
    chance = 1.0 / n
    stats = {}
    for a in ("short", "long"):
        ops = statistics.median(t["ops"] for t in by[a])
        share = statistics.median(t["straggler_share"] for t in by[a])
        skew = statistics.median(t["work_skew"] for t in by[a])
        spread = statistics.median(t["ops_spread"] for t in by[a])
        stats[a] = {"ops": ops, "share": share, "skew": skew, "spread": spread,
                    "excess": share - chance}
        print(f"{a:<5} median ops={ops:<8.0f} share={share:.4f} "
              f"excess={share - chance:+.4f} skew={skew:.3f} ops_spread={spread:.4f}")
    print(f"chance share = 1/{n} = {chance:.4f}")

    growth = stats["long"]["ops"] / max(1e-9, stats["short"]["ops"])
    print(f"\nCONTROL 1 window growth = {growth:.2f}x (need >= {OPS_GROWTH_MIN}x)")
    if growth < OPS_GROWTH_MIN:
        print("CONTROL 1 FIRED: the window did not actually grow; nothing was varied.")
        print("VERDICT: REPORT NOTHING")
        return 0
    for a in ("short", "long"):
        if stats[a]["spread"] > OPS_SPREAD_MAX:
            print(f"CONTROL 2 FIRED: {a} ops_spread {stats[a]['spread']:.4f} > {OPS_SPREAD_MAX}")
            print("VERDICT: REPORT NOTHING")
            return 0

    if stats["short"]["excess"] <= 0:
        print("VERDICT: REPORT NOTHING -- the short arm shows no concentration above chance,")
        print("         so there is no ratio to take.")
        return 0

    r = stats["long"]["excess"] / stats["short"]["excess"]
    print(f"\nR = excess(long)/excess(short) = {r:.3f}   "
          f"(chance decay for a {growth:.0f}x window is ~{1 / growth ** 0.5:.2f})")
    if r >= FIXED_LANE_R:
        print(f"VERDICT: FIXED LANE -- concentration survives a {growth:.0f}x longer window "
              f"(R={r:.3f} >= {FIXED_LANE_R}).")
        print("         One lane really is last on most ops; the lead stands and the")
        print("         remaining question is what makes that lane slow.")
    elif r <= NOISE_R:
        print(f"VERDICT: NOISE -- concentration decays toward chance at least as fast as "
              f"coin-flipping (R={r:.3f} <= {NOISE_R}).")
        print("         The width-16 'straggler' is a max-of-noise artefact. `work_skew` and")
        print("         `straggler_share` have no null model and must not be read as an")
        print("         imbalance. The straggler lead is RETRACTED.")
    else:
        print(f"VERDICT: REPORT NOTHING -- R={r:.3f} landed between {NOISE_R} and {FIXED_LANE_R}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
