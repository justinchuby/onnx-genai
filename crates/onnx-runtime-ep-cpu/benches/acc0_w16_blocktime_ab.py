#!/usr/bin/env python3
"""Does the decode pool's blocktime spin buy anything at width 16?

Why
---
`acc0_w8_w16_cpu_split.py` returned **BURN-DOMINATED** for the native arm across
the t=8 -> t=16 doubling: `R_busy = 1.057` (the workers are *more* occupied at
16, not parked) while `R_cpu = 1.449` (each token costs 45% more CPU), 13
trusted launches, 100% sign consistency.  The same run measured ORT in the same
window at `R_cpu = 1.074`, which rules out the host -- a DRAM or L3 ceiling
would inflate both arms.

It also named where the extra CPU goes: the **system** fraction rises from
0.062 to 0.212 across the doubling (sign 100%), against ORT's 0.000 at both
widths.  `sched_yield` is charged to system time, and `decode_spmd`'s worker
wait is a `KMP_BLOCKTIME` analogue that spins `SPIN_LOOP_BUDGET` iterations and
then *yields* between clock checks for the remainder of a 500 us window before
parking on a futex.

That is an attribution from a correlation, and this file is the intervention
that tests it.  `ONNX_GENAI_CPU_DECODE_BLOCKTIME_US=0` parks as soon as the
sense line is not already advanced, removing the yield ramp entirely.  If the
sys time is the ramp, `0` must collapse it; if throughput also improves, the
ramp is not merely visible but expensive.

Scope, stated up front rather than in a footnote
------------------------------------------------
This workload is a **zero-gap** decode loop: five projections dispatched
back-to-back with no model-level work between them.  That is the right shape
for the acc0 gap this is chasing -- it is the same loop the published
native-vs-ORT cells run -- but it is *not* sufficient evidence to change the
shipped default, because a real generation loop has sampling, KV bookkeeping
and detokenisation between decodes, and a longer gap is exactly where an early
park pays a wake cost that this harness cannot see.  A default change needs the
gap-aware harness (#1395).  What this run can establish is narrower and still
worth having: whether the blocktime spin is *costing* us throughput on the
workload the gap is measured on.

Pre-registered before the first run
-----------------------------------
Primary claim, at width 16 only:

    ACCEPT "blocktime=0 is a throughput win at width 16" iff
      (1) n_trusted >= 6, and
      (2) median tps(bt0)/tps(bt500) >= 1.10, and
      (3) that ratio exceeds 1.0 in >= 80% of paired launches, and
      (4) effect-over-null: (median ratio - 1) >= 3 * aa_halfwidth, where
          aa_halfwidth is the median |ratio - 1| of a bt500-vs-bt500 A/A pair
          taken in the same launches.

Mechanism claim, reported separately and never used to rescue (1)-(4):

    ACCEPT "the win is the yield ramp" iff the throughput claim is accepted
    AND median sys_frac(bt0) < median sys_frac(bt500) with >= 80% sign
    consistency.  A throughput win without the sys collapse means the knob
    changed something else and the attribution is unproven.

Regression guard at width 8, which can veto nothing but must be reported:

    FLAG if median tps(bt0)/tps(bt500) at width 8 <= 0.95 with >= 80% sign
    consistency.  A fix that helps 16 by hurting 8 is a trade, not a win, and
    has to be presented as one.
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
PRIMARY_WIDTH = 16
CONTROL_WIDTH = 8
CONTROL_BLOCKTIME = "500"
TEST_BLOCKTIME = "0"

MIN_TRUSTED = 6
MIN_RATIO = 1.10
SIGN_FRACTION = 0.80
EFFECT_OVER_NULL = 3.0
REGRESSION_RATIO = 0.95


def arm(args, width, blocktime):
    return H.native(args.binary, MODEL, BLOCK, ACC, width, SESSIONS,
                    args.tokens, args.reps,
                    extra={"ONNX_GENAI_CPU_DECODE_BLOCKTIME_US": blocktime})


def cpu_of(a):
    """`(tps, sys_frac, cpu_per_token)` from the self-consistent CPU row."""
    cpu = a.get("cpu")
    if not cpu or "tps_rep" not in cpu:
        return None
    total = cpu["user_s"] + cpu["sys_s"]
    if total <= 0:
        return None
    return (cpu["tps_rep"], cpu["sys_s"] / total, cpu["cpu_s_per_token"])


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
           "competitors": [c[2] for c in busy], "tokens": args.tokens,
           "widths": {}}
    # Three arms at the primary width -- control, test, control again -- with
    # the order rotated so neither arm systematically owns the warm cache. The
    # second control is the A/A null, taken in the same launch as the effect it
    # has to clear, which is the only way the null describes the same host state.
    flip = launch % 2 == 1
    rec["flipped"] = flip
    trusted = True
    with H.LoadWatch() as watch:
        for width in (PRIMARY_WIDTH, CONTROL_WIDTH):
            if flip:
                b = arm(args, width, TEST_BLOCKTIME)
                a = arm(args, width, CONTROL_BLOCKTIME)
                a2 = arm(args, width, TEST_BLOCKTIME)
                aa_arm = TEST_BLOCKTIME
            else:
                a = arm(args, width, CONTROL_BLOCKTIME)
                b = arm(args, width, TEST_BLOCKTIME)
                a2 = arm(args, width, CONTROL_BLOCKTIME)
                aa_arm = CONTROL_BLOCKTIME
            ca, cb, ca2 = cpu_of(a), cpu_of(b), cpu_of(a2)
            if not (ca and cb and ca2):
                trusted = False
                rec.setdefault("untrusted_reason", []).append(
                    f"w{width} missing cpu row")
                continue
            rec["widths"][str(width)] = {
                "control": a, "test": b, "aa": a2, "aa_arm": aa_arm,
                "ratio": cb[0] / ca[0],
                "aa_ratio": (ca2[0] / ca[0]) if aa_arm == CONTROL_BLOCKTIME
                            else (ca2[0] / cb[0]),
                "sys_frac_control": ca[1], "sys_frac_test": cb[1],
                "cpt_control": ca[2], "cpt_test": cb[2],
            }
    rec["runnable_peak"] = watch.peak
    rec["trusted"] = (trusted and not busy
                      and watch.peak <= PRIMARY_WIDTH + args.slack
                      and str(PRIMARY_WIDTH) in rec["widths"]
                      and str(CONTROL_WIDTH) in rec["widths"])
    return rec


def med(cells, width, key):
    v = [c["widths"][str(width)][key] for c in cells
         if str(width) in c["widths"]]
    return statistics.median(v) if v else float("nan")


def frac(cells, width, key, pred):
    v = [c["widths"][str(width)][key] for c in cells
         if str(width) in c["widths"]]
    return (sum(1 for x in v if pred(x)) / len(v)) if v else 0.0


def verdict(cells):
    n = len(cells)
    lines = []
    if n < MIN_TRUSTED:
        return [f"REPORT NOTHING (n_trusted={n} < {MIN_TRUSTED})"]

    ratio = med(cells, PRIMARY_WIDTH, "ratio")
    sign = frac(cells, PRIMARY_WIDTH, "ratio", lambda r: r > 1.0)
    aa = [abs(c["widths"][str(PRIMARY_WIDTH)]["aa_ratio"] - 1.0)
          for c in cells if str(PRIMARY_WIDTH) in c["widths"]]
    aa_half = statistics.median(aa) if aa else float("nan")
    effect = ratio - 1.0

    ok = (ratio >= MIN_RATIO and sign >= SIGN_FRACTION
          and effect >= EFFECT_OVER_NULL * aa_half)
    lines.append(
        f"THROUGHPUT at w={PRIMARY_WIDTH}: "
        f"{'ACCEPT' if ok else 'REJECT'} -- ratio={ratio:.4f} "
        f"(need >={MIN_RATIO}), sign={sign:.0%} (need >={SIGN_FRACTION:.0%}), "
        f"effect={effect:+.4f} vs {EFFECT_OVER_NULL}x A/A half-width "
        f"{aa_half:.4f} = {EFFECT_OVER_NULL * aa_half:.4f}, n={n}")

    d_sys = [c["widths"][str(PRIMARY_WIDTH)]["sys_frac_test"]
             - c["widths"][str(PRIMARY_WIDTH)]["sys_frac_control"]
             for c in cells if str(PRIMARY_WIDTH) in c["widths"]]
    sys_sign = sum(1 for d in d_sys if d < 0) / len(d_sys) if d_sys else 0.0
    sys_med = statistics.median(d_sys) if d_sys else float("nan")
    mech = ok and sys_med < 0 and sys_sign >= SIGN_FRACTION
    lines.append(
        f"MECHANISM (yield ramp): {'ACCEPT' if mech else 'UNPROVEN'} -- "
        f"sys_frac shift {sys_med:+.4f}, sign {sys_sign:.0%}; "
        f"sys_frac {med(cells, PRIMARY_WIDTH, 'sys_frac_control'):.3f} -> "
        f"{med(cells, PRIMARY_WIDTH, 'sys_frac_test'):.3f}")

    r8 = med(cells, CONTROL_WIDTH, "ratio")
    s8 = frac(cells, CONTROL_WIDTH, "ratio", lambda r: r < 1.0)
    flagged = r8 <= REGRESSION_RATIO and s8 >= SIGN_FRACTION
    lines.append(
        f"REGRESSION at w={CONTROL_WIDTH}: "
        f"{'FLAGGED -- this is a trade, not a win' if flagged else 'none'} -- "
        f"ratio={r8:.4f}, down-sign={s8:.0%}")
    return lines


def report(cells):
    tr = [c for c in cells if c["trusted"]]
    print()
    for line in verdict(tr):
        print(line)
    if len(tr) < MIN_TRUSTED:
        return
    print()
    print(f"{'w':>3} {'ratio':>7} {'aa':>7} {'sysC':>6} {'sysT':>6} "
          f"{'cptC':>9} {'cptT':>9}")
    for w in (PRIMARY_WIDTH, CONTROL_WIDTH):
        print(f"{w:>3} {med(tr, w, 'ratio'):>7.4f} "
              f"{med(tr, w, 'aa_ratio'):>7.4f} "
              f"{med(tr, w, 'sys_frac_control'):>6.3f} "
              f"{med(tr, w, 'sys_frac_test'):>6.3f} "
              f"{med(tr, w, 'cpt_control'):>9.5f} "
              f"{med(tr, w, 'cpt_test'):>9.5f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--launches", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--slack", type=int, default=10)
    ap.add_argument("--quiet-runnable", type=int, default=4)
    ap.add_argument("--quiet-limit", type=int, default=300)
    ap.add_argument("--deadline-min", type=float, default=0.0,
                    help="stop at the first launch boundary after this many "
                         "minutes; a clock stops the run, the numbers never do")
    ap.add_argument("--out", default="w16_blocktime_ab.json")
    ap.add_argument("--replay", help="re-score an existing JSON, run nothing")
    args = ap.parse_args()

    if args.replay:
        with open(args.replay) as f:
            report(json.load(f))
        return
    args.binary = os.path.abspath(args.binary)

    print(f"# pre-registered: n>={MIN_TRUSTED}, ratio>={MIN_RATIO}, "
          f"sign>={SIGN_FRACTION:.0%}, effect>={EFFECT_OVER_NULL}x A/A "
          f"half-width; mechanism needs a sys_frac fall at >={SIGN_FRACTION:.0%}")
    hdr = (f"{'L':>3} {'pk':>3} {'T':>2} {'r16':>7} {'aa16':>7} "
           f"{'sysC':>6} {'sysT':>6} {'r8':>7}")
    print(hdr)
    print("-" * len(hdr))
    cells = []
    started = time.time()
    for launch in range(args.launches):
        if args.deadline_min and (time.time() - started) / 60.0 >= args.deadline_min:
            print(f"# deadline {args.deadline_min:g} min reached at launch "
                  f"{launch}; stopping on the clock", flush=True)
            break
        try:
            c = one_launch(args, launch)
        except Exception as e:
            sys.stderr.write(f"launch {launch} failed: {e}\n")
            continue
        cells.append(c)
        p = c["widths"].get(str(PRIMARY_WIDTH), {})
        q = c["widths"].get(str(CONTROL_WIDTH), {})
        nan = float("nan")
        print(f"{c['launch']:>3} {c['runnable_peak']:>3} "
              f"{'y' if c['trusted'] else 'N':>2} "
              f"{p.get('ratio', nan):>7.4f} {p.get('aa_ratio', nan):>7.4f} "
              f"{p.get('sys_frac_control', nan):>6.3f} "
              f"{p.get('sys_frac_test', nan):>6.3f} "
              f"{q.get('ratio', nan):>7.4f}", flush=True)
        with open(args.out, "w") as f:
            json.dump(cells, f, indent=1)

    report(cells)


if __name__ == "__main__":
    main()
