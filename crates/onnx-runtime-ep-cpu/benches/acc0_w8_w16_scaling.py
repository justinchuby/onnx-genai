#!/usr/bin/env python3
"""Same-session t=8 vs t=16 acc0 scaling, both arms.

Why
---
`2026-08-23-acc0-gap-at-width-16.md` established the gap at width 16 (~1.78x)
and at width 8 (1.120x, from a separate run), and inferred a scaling wall:
native converts the 8->16 doubling into ~1.16x while ORT converts it into
~1.74x.  That inference crosses two runs with different token budgets, so it is
indicative only.  This measures both widths **in the same launch**, so the
scaling factors are paired and the token budget is identical.

Design
------
One launch = four arms in sequence, order rotated by launch:

    native@8, ORT@8, native@16, ORT@16

Every arm is a fresh process; each width is pinned to its own physical cores
(`native_pin`), which is what the native process confines itself to anyway, so
both arms get the same machine at each width.

Pre-registered before the first run
-----------------------------------
The claim under test is the *scaling ratio* `tps(16)/tps(8)` per arm, and the
quantity of interest is whether native's is materially below ORT's.

    ACCEPT "native scales worse" iff
      (1) n_trusted >= 6, and
      (2) native's scaling ratio is below ORT's in >= 80% of paired launches,
          i.e. the sign is consistent rather than an average of a wash.

Condition (2) is a sign test on paired cells, deliberately not a comparison of
medians: at width 16 both arms carry a +-35% A/A null, and a median-of-ratios
can be moved by that null while a per-launch sign cannot be moved by symmetric
noise as easily.

Nothing here re-uses the width-16 gap number.  A scaling ratio is internal to
one arm, so it is unaffected by whatever the cross-arm gap turns out to be.
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import acc0_gap_matrix as H  # noqa: E402

MODEL, BLOCK, ACC, SESSIONS = "llama", 32, 0, 1
WIDTHS = (8, 16)
MIN_TRUSTED = 6
SIGN_FRACTION = 0.80


def wait_quiet_runnable(ceiling, limit, period=10.0):
    start = time.time()
    while True:
        runnable = H.LoadWatch.runnable()
        busy = H.competing_load()
        if runnable <= ceiling and not busy:
            return runnable, []
        if time.time() - start >= limit:
            return runnable, busy
        time.sleep(period)


def one_launch(args, launch):
    pre, busy = wait_quiet_runnable(args.quiet_runnable, args.quiet_limit)
    rec = {"launch": launch, "runnable_pre": pre,
           "competitors": [c[2] for c in busy], "arms": {}}
    # Rotate which width goes first so a warm page cache or a monotone drift
    # cannot systematically favour one width over the other.
    order = WIDTHS if launch % 2 == 0 else tuple(reversed(WIDTHS))
    rec["order"] = order
    with H.LoadWatch() as watch:
        for w in order:
            n = H.native(args.binary, MODEL, BLOCK, ACC, w, SESSIONS,
                         args.tokens, args.reps)
            o = H.ort(MODEL, BLOCK, ACC, w, SESSIONS, args.tokens, args.reps)
            rec["arms"][str(w)] = {"native": n, "ort": o}
    rec["runnable_peak"] = watch.peak
    rec["trusted"] = (watch.peak <= max(WIDTHS) + args.slack) and not busy
    a8, a16 = rec["arms"]["8"], rec["arms"]["16"]
    rec["native_scale"] = a16["native"]["tps"] / a8["native"]["tps"]
    rec["ort_scale"] = a16["ort"]["tps"] / a8["ort"]["tps"]
    rec["gap8"] = a8["ort"]["tps"] / a8["native"]["tps"]
    rec["gap16"] = a16["ort"]["tps"] / a16["native"]["tps"]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--launches", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--slack", type=int, default=10)
    ap.add_argument("--quiet-runnable", type=int, default=4)
    ap.add_argument("--quiet-limit", type=int, default=600)
    ap.add_argument("--out", default="w8_w16_scaling.json")
    args = ap.parse_args()
    args.binary = os.path.abspath(args.binary)

    print(f"# pre-registered: n>={MIN_TRUSTED}, native_scale<ort_scale in "
          f">={SIGN_FRACTION:.0%} of paired launches")
    hdr = (f"{'L':>3} {'order':>9} {'pk':>3} {'T':>2} {'nat8':>7} {'ort8':>7} "
           f"{'nat16':>7} {'ort16':>7} {'natX':>6} {'ortX':>6} "
           f"{'gap8':>6} {'gap16':>6}")
    print(hdr)
    print("-" * len(hdr))
    cells = []
    for launch in range(args.launches):
        try:
            c = one_launch(args, launch)
        except Exception as e:
            sys.stderr.write(f"launch {launch} failed: {e}\n")
            continue
        cells.append(c)
        a8, a16 = c["arms"]["8"], c["arms"]["16"]
        print(f"{c['launch']:>3} {str(c['order']):>9} {c['runnable_peak']:>3} "
              f"{'y' if c['trusted'] else 'N':>2} "
              f"{a8['native']['tps']:>7.1f} {a8['ort']['tps']:>7.1f} "
              f"{a16['native']['tps']:>7.1f} {a16['ort']['tps']:>7.1f} "
              f"{c['native_scale']:>6.3f} {c['ort_scale']:>6.3f} "
              f"{c['gap8']:>6.3f} {c['gap16']:>6.3f}", flush=True)
        with open(args.out, "w") as f:
            json.dump(cells, f, indent=1)

    tr = [c for c in cells if c["trusted"]]
    print()
    if len(tr) < MIN_TRUSTED:
        print(f"VERDICT: REPORT NOTHING (n_trusted={len(tr)} < {MIN_TRUSTED})")
        return
    wins = sum(1 for c in tr if c["native_scale"] < c["ort_scale"])
    frac = wins / len(tr)
    ns = sorted(c["native_scale"] for c in tr)
    os_ = sorted(c["ort_scale"] for c in tr)
    g8 = sorted(c["gap8"] for c in tr)
    g16 = sorted(c["gap16"] for c in tr)
    v = "ACCEPT" if frac >= SIGN_FRACTION else "NOT ESTABLISHED"
    print(f"VERDICT: {v} — native scaled worse in {wins}/{len(tr)} "
          f"paired launches ({frac:.0%})")
    print(f"  native t=8->16 scaling: median {statistics.median(ns):.3f} "
          f"[{ns[0]:.3f}, {ns[-1]:.3f}]")
    print(f"  ORT    t=8->16 scaling: median {statistics.median(os_):.3f} "
          f"[{os_[0]:.3f}, {os_[-1]:.3f}]")
    print(f"  gap at t=8 : median {statistics.median(g8):.3f} "
          f"[{g8[0]:.3f}, {g8[-1]:.3f}]")
    print(f"  gap at t=16: median {statistics.median(g16):.3f} "
          f"[{g16[0]:.3f}, {g16[-1]:.3f}]")
    print(f"\nraw: {args.out}")


if __name__ == "__main__":
    main()
