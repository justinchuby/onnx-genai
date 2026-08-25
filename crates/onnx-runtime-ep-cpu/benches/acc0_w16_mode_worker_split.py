#!/usr/bin/env python3
"""Which component of the width-16 window does the slow mode spend its extra time in?

Why this and not another new instrument
---------------------------------------
Everything about the width-16 bimodality that could be excluded from the outside
now has been: worker placement (both modes on a byte-identical pinned set, 21
launches), THP/page backing, foreign load on the pinned set (<=1.7% of pinned
busy against ~23% required), clock/boost state (user CPU per token flat at 1.025
against a required 1.52), and a static spare-tile steal.

The last candidate I was carrying -- weight-arena placement across the two
L3/CCX domains -- is **malformed on this host and is hereby dropped**:
`numactl --hardware` reports **one** NUMA node covering all 32 CPUs with a
single distance of 10. There is no second memory domain to place an arena in,
and an L3 is a cache, not an allocation target: a read-only weight stream is
simply replicated into whichever L3 reads it. I carried that candidate for two
records without checking that the host could express it.

What the outside-in evidence now says is narrow and specific: in a slow launch
realized lanes fall ~15.5 -> ~12.2 while **both** user and sys CPU per token
stay flat. The missing lanes are not running and not in the kernel. That is a
statement about *participation*, which is exactly what the in-EP
`SpmdWorkerProfile` counters already measure -- and which nobody has yet read
stratified by mode.

So this file adds no instrument. It runs the validated
`acc0_w16_worker_split.py`, whose per-worker `work_ns` / `wake_ns` /
`last_arrivals` deltas are bracketed by the same two points that bracket `wall`,
and re-scores its output split by mode. `derive()` and `trusted()` are imported
and called, never reimplemented, so no threshold in the underlying instrument
can drift away from the one that produced the published aggregate split.

Mode discriminator
------------------
The workload is a fixed token/rep count, so the width-16 `wall_s` that the
instrument already records *is* the mode: a slow launch takes ~1.5x longer. The
cut is placed at the largest gap in the sorted `wall_s` values rather than at a
fixed constant, because worker profiling costs two clock reads per worker per op
and shifts the absolute numbers -- a constant tuned on unprofiled runs would be
in the wrong place here.

Pre-registered before the first launch
--------------------------------------
    REPORT NOTHING unless both modes are present with >= 3 trusted launches
    each, and the between-mode gap in `wall_s` is at least 3x the widest
    within-mode spread. Profiling perturbs timing and may suppress the
    bimodality entirely; if it does, that is the finding and no mechanism is
    named.

    Otherwise the slow mode's extra window time is attributed to whichever of
    the four components -- useful work, straggler wait, wake latency,
    dispatcher/serial -- absorbs the largest share of it, and the attribution is
    reported *with* the other three so a near-tie is visible rather than hidden.

    A component only counts as the mechanism if it absorbs at least
    `DOMINANT_SHARE` (0.5) of the extra window. Below that, report the split and
    name nothing.

Note the built-in control: `useful work` per token must be near-identical
between modes. The outside-in measurement already established that user CPU per
token is flat, so a large `useful work` difference here would mean the two
instruments disagree and the run should be thrown away, not interpreted.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import acc0_w16_worker_split as W  # noqa: E402

WIDTH = 16
MIN_PER_MODE = 3
GAP_RATIO = 3.0
DOMINANT_SHARE = 0.5


def components(st):
    """The four-way split of one launch's window, in fractions of wall.

    Same decomposition the underlying instrument prints; computed from its own
    derived fields so the two cannot disagree.
    """
    work = st["work_frac"]
    wake = st["wake_frac"]
    strag = max(0.0, st["straggler_share"] * st["n_workers"] - 1.0) / \
        max(st["n_workers"] - 1, 1)
    strag = min(strag, max(0.0, 1.0 - work - wake))
    return {
        "useful work": work,
        "straggler wait": strag,
        "wake latency": wake,
        "dispatcher/serial": max(0.0, 1.0 - work - strag - wake),
    }


def split_modes(walls):
    """Cut at the largest gap. Returns (cut, gap, widest_within_mode_spread)."""
    s = sorted(walls)
    if len(s) < 2:
        return None, 0.0, 0.0
    gaps = [(s[i + 1] - s[i], i) for i in range(len(s) - 1)]
    gap, i = max(gaps)
    cut = (s[i] + s[i + 1]) / 2.0
    lo, hi = s[:i + 1], s[i + 1:]
    spread = max(max(lo) - min(lo), max(hi) - min(hi))
    return cut, gap, spread


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--launches", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=192)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default="bb/wsplit_modes.json")
    ap.add_argument("--replay", default=None)
    args = ap.parse_args()

    if args.replay:
        recs = json.load(open(args.replay))
    else:
        cmd = [sys.executable,
               os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "acc0_w16_worker_split.py"),
               "--binary", os.path.abspath(args.binary),
               "--launches", str(args.launches),
               "--tokens", str(args.tokens), "--reps", str(args.reps),
               "--widths", str(WIDTH), "--out", os.path.abspath(args.out)]
        print("running the validated instrument:", " ".join(cmd[-8:]))
        sys.stdout.flush()
        subprocess.run(cmd, check=False)
        recs = json.load(open(args.out))

    ok = [r for r in recs if W.trusted(r)]
    print(f"\ntrusted launches: {len(ok)} of {len(recs)} "
          f"(instrument's own residual + load gates)")
    if len(ok) < 2 * MIN_PER_MODE:
        print(f"VERDICT: REPORT NOTHING -- {len(ok)} trusted launches, need "
              f"{2 * MIN_PER_MODE}")
        return

    walls = [r["widths"][str(WIDTH)]["wall_s"] for r in ok]
    cut, gap, spread = split_modes(walls)
    fast = [r for r in ok if r["widths"][str(WIDTH)]["wall_s"] < cut]
    slow = [r for r in ok if r["widths"][str(WIDTH)]["wall_s"] >= cut]
    print(f"wall_s range {min(walls):.3f} - {max(walls):.3f}   cut {cut:.3f}   "
          f"gap {gap:.3f}   widest within-mode spread {spread:.3f}")
    print(f"modes: fast {len(fast)}, slow {len(slow)}")

    if len(fast) < MIN_PER_MODE or len(slow) < MIN_PER_MODE:
        print(f"VERDICT: REPORT NOTHING -- need >= {MIN_PER_MODE} launches per "
              f"mode; worker profiling may have suppressed the bimodality, "
              f"which would itself be the finding.")
        return
    if gap < GAP_RATIO * spread:
        print(f"VERDICT: REPORT NOTHING -- the between-mode gap {gap:.3f} is "
              f"not {GAP_RATIO}x the within-mode spread {spread:.3f}; this is "
              f"one distribution, not two.")
        return

    fw = statistics.median([r["widths"][str(WIDTH)]["wall_s"] for r in fast])
    sw = statistics.median([r["widths"][str(WIDTH)]["wall_s"] for r in slow])
    extra = sw - fw
    print(f"\nfast wall {fw:.3f} s   slow wall {sw:.3f} s   "
          f"extra {extra:.3f} s ({sw/fw:.3f}x)")

    fc = {k: statistics.median([components(r["widths"][str(WIDTH)])[k]
                                for r in fast])
          for k in components(fast[0]["widths"][str(WIDTH)])}
    sc = {k: statistics.median([components(r["widths"][str(WIDTH)])[k]
                                for r in slow])
          for k in components(slow[0]["widths"][str(WIDTH)])}

    print(f"\n{'component':>18} {'fast frac':>10} {'slow frac':>10} "
          f"{'fast s':>9} {'slow s':>9} {'delta s':>9} {'share':>7}")
    deltas = {}
    for k in fc:
        fs, ss = fc[k] * fw, sc[k] * sw
        deltas[k] = ss - fs
        share = deltas[k] / extra if extra else float("nan")
        print(f"{k:>18} {fc[k]:>10.3f} {sc[k]:>10.3f} {fs:>9.3f} {ss:>9.3f} "
              f"{deltas[k]:>+9.3f} {share:>7.2f}")

    # Control: useful work per launch must not move, or the two instruments
    # disagree and nothing here is interpretable.
    work_ratio = (sc["useful work"] * sw) / (fc["useful work"] * fw)
    print(f"\ncontrol: useful work slow/fast = {work_ratio:.4f} "
          f"(outside-in user CPU/token ratio was 1.0250)")
    if not 0.9 <= work_ratio <= 1.15:
        print("VERDICT: REPORT NOTHING -- useful work moved between modes, "
              "which contradicts the flat user-CPU measurement. The two "
              "instruments disagree; throw the run away rather than interpret "
              "it.")
        return

    top = max(deltas, key=lambda k: deltas[k])
    share = deltas[top] / extra if extra else 0.0
    if share >= DOMINANT_SHARE:
        print(f"VERDICT: the slow mode's extra {extra:.3f} s is absorbed by "
              f"**{top}** ({100*share:.0f}% of it).")
    else:
        print(f"VERDICT: REPORT NOTHING -- no single component absorbs "
              f"{DOMINANT_SHARE:.0%} of the extra window (largest is {top} at "
              f"{100*share:.0f}%). The split above is the result.")


if __name__ == "__main__":
    main()
