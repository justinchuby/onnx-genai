#!/usr/bin/env python3
"""Where native's missing t=8 -> t=16 scaling goes: idle workers, or burnt CPU.

Why
---
`2026-08-23-acc0-gap-at-width-16.md` established, paired within launch and
10/10 on a sign test, that ORT converts the 8->16 doubling into 1.762x while
native converts it into 1.319x.  That is a throughput curve, and a throughput
curve is equally consistent with two opposite mechanisms:

  * the extra workers are **idle** -- dispatch, wake, join or a straggler
    leaves them parked, and the machine we asked for is not being used;
  * the extra workers are **busy** -- the machine is fully used and each token
    simply costs more CPU at 16 than at 8 (spin burnt while waiting, cache or
    memory pressure inside the kernel, or redundant per-worker work).

They want opposite fixes.  Timing alone cannot tell them apart, and every
number published on this so far has been timing.

The decomposition, which is an identity rather than a model
-----------------------------------------------------------
For one arm confined to `w` CPUs, over the measured window only:

    cpu_per_token(w) = (user_s + sys_s) / tokens
    busy(w)          = (user_s + sys_s) / (wall_s * w)
    tps(w)           = tokens / wall_s = busy(w) * w / cpu_per_token(w)

so, exactly and with no residual term,

    tps(16)/tps(8) = 2 * [busy(16)/busy(8)] * [cpu_per_token(8)/cpu_per_token(16)]
                   = 2 * R_busy / R_cpu

The measured 1.319x therefore *has* to be explained by `R_cpu / R_busy ~ 1.5`,
and this run measures which of the two factors carries it.  The identity is
checked per launch against the independently reported throughput, and a cell
whose two sides disagree by more than 5% is dropped rather than interpreted --
that is the self-test on the CPU accounting itself.

`user` and `sys` are kept separate throughout.  `sched_yield` in a park path is
charged to **system**, so a spin-then-park barrier that ramps its yields with
width appears as a rising `sys` fraction against flat `user`.  #1740's owner has
an open finding that `worker_wait` yields for the remainder of a 500 us
blocktime window and costs ~20% of process CPU at t=16; this measurement can
corroborate or refute that independently, from the outside.

CPU seconds are also the more contention-robust instrument.  A neighbour
saturating the host steals our wall but does not add to our `utime`.  Note this
is *not* what `/usr/bin/time`'s `Percent of CPU` provides: that is
`(user+sys)/wall`, the same wall in the denominator, so it degrades exactly
when wall does -- the trap that produced a retracted t=2 pathology claim.

ORT is measured the same way, in the same window, as a control.  If native's
`cpu_per_token` inflates across the doubling and ORT's does not, the inflation
is ours; if both inflate, it is the host (shared L3, memory controllers, SMT)
and our kernel is not specially at fault.

Pre-registered before the first run
-----------------------------------
Applied to the **native** arm.  `R_cpu` and `R_busy` are per-launch paired
ratios; the decision uses the median over trusted launches AND a sign test, so
that a verdict cannot rest on one extreme launch.

    n_trusted >= 6 required for any verdict at all.

    NO-LOSS         iff median speedup >= 1.80
                    (the wall did not reproduce here; report that, stop)
    BURN-DOMINATED  iff median R_cpu >= 1.25 and median R_busy >= 0.90
    IDLE-DOMINATED  iff median R_busy <= 0.80 and median R_cpu <= 1.15
    MIXED           iff median R_cpu >= 1.15 and median R_busy <= 0.90
    INCONCLUSIVE    otherwise

    Every verdict except NO-LOSS additionally requires the *sign* of its
    driving quantity to be consistent in >= 80% of trusted launches:
      BURN -> R_cpu > 1 ;  IDLE -> R_busy < 1 ;  MIXED -> both.
    If the median clears a threshold but the sign test does not, the verdict
    is INCONCLUSIVE and says so.

Secondary, reported but never decisive:
    the sys-fraction rise `sys_frac(16) - sys_frac(8)` is called out as a named
    contributor only if it is >= 0.10 with >= 80% sign consistency.

Width 4 is measured for context when `--with-w4` is passed.  It is never part
of the decision: the pre-registered claim is about the 8->16 doubling that the
merged record documents.

Run 1 (2026-08-23, 10 launches): REPORT NOTHING, and why
--------------------------------------------------------
The first run of this harness returned `REPORT NOTHING (n_trusted=1 < 6)`.
That verdict is honoured and nothing from it is quoted -- including the one
trusted cell.

The cells were not lost to host contention.  They were lost to the identity
self-test, which was doing exactly its job: **the CPU row was being assembled
from more than one repetition.**  Both producers took an independent median per
quantity, and because `tps = tokens / wall`, sorting by `tps` and sorting by
`wall` are reversed orders -- so at an even repetition count the `tps` median
and the `wall` median landed on *different* repetitions.  The published row
then described no run that had happened, and `busy = cpu / (wall * w)` carried
the whole rep-to-rep spread as bias.  Observed identity errors were 4.4% to
29.7% against a quantity that is algebraically 0.00% when the row is
self-consistent.

Both producers now emit every CPU field, plus `tps_rep`, from the single
repetition whose throughput is the median.  **No threshold in the
pre-registered rule was changed** -- not the counts, not the ratios, not the
sign fraction.  What changed is the instrument that feeds it, and the change
was forced by a self-test that fired before any number was scored rather than
by the numbers being disappointing.

`busy` is blind to spin-wait: measure with `--blocktime 0`
----------------------------------------------------------
Run 2 returned BURN-DOMINATED with `R_busy = 1.057`, which reads as "the
workers are not idle at width 16".  That reading is an artifact of the
instrument, and `acc0_w16_blocktime_ab.py` demonstrated it: `decode_spmd`'s
worker wait spins and then `sched_yield`s for the remainder of a 500 us window
before parking, and **a spinning thread is indistinguishable from a working one
in `utime`/`stime`**.  At width 16, for statistically identical throughput
(ratio 0.996 against a 5.2% A/A null), `busy` reads **0.953** at the shipped
500 us default and **0.692** at `0`.  The 26 percentage points in between are
`sched_yield`, and the pool really is idle for roughly a third of the window.

So a run intended to separate idle from burn must be taken with
`--blocktime 0`.  At the default, wait is charged as work and the two
categories the rule exists to distinguish are merged.  The rule itself is
unchanged and is applied to both environments; the environments are what
differ, and both are reported.
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
DECISION_WIDTHS = (8, 16)

MIN_TRUSTED = 6
SIGN_FRACTION = 0.80
NO_LOSS_SPEEDUP = 1.80
BURN_R_CPU = 1.25
BURN_R_BUSY = 0.90
IDLE_R_BUSY = 0.80
IDLE_R_CPU = 1.15
MIXED_R_CPU = 1.15
MIXED_R_BUSY = 0.90
SYS_FRAC_RISE = 0.10
IDENTITY_TOLERANCE = 0.05


def derive(arm, width, tokens):
    """`cpu_per_token`, `busy`, `sys_frac` and the paired `tps` for one arm.

    Every field is taken from the CPU row, which both producers emit from a
    *single* repetition -- the one whose throughput is the median. That matters:
    an earlier version of this took an independent median per quantity, and
    since `tps = tokens / wall` sorts in the reverse order to `wall`, at an even
    repetition count the two medians selected different repetitions. The row
    then described no run that ever happened and the identity below failed by
    the rep-to-rep spread (4.4%-29.7% observed) rather than by anything about
    the kernel.

    Returns `None` when the arm carries no `cpu` row -- an older binary or an
    older baseline script -- so a missing measurement is visibly absent rather
    than silently defaulted to something that looks like data.
    """
    cpu = arm.get("cpu")
    if not cpu or "tps_rep" not in cpu:
        return None
    user_s, sys_s = cpu["user_s"], cpu["sys_s"]
    wall_s, total = cpu["wall_s"], user_s + sys_s
    if wall_s <= 0 or total <= 0:
        return None
    return {
        "user_s": user_s,
        "sys_s": sys_s,
        "cpu_s": total,
        "wall_s": wall_s,
        "tps": cpu["tps_rep"],
        "cpu_per_token": total / tokens,
        "busy": total / (wall_s * width),
        "sys_frac": sys_s / total,
    }


def identity_error(d8, d16, speedup):
    """How far the CPU accounting is from reproducing the measured speedup.

    `2 * R_busy / R_cpu` is algebraically equal to `tps(16)/tps(8)`, so this is
    zero for consistent inputs. It is not a modelling assumption that can be
    wrong about the machine -- a nonzero value means the CPU row and the
    throughput row are not describing the same window, which is a defect in the
    instrument rather than a fact about the kernel.
    """
    predicted = 2.0 * (d16["busy"] / d8["busy"]) * (
        d8["cpu_per_token"] / d16["cpu_per_token"])
    return abs(predicted - speedup) / speedup, predicted


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


def one_launch(args, launch, widths):
    pre, busy = wait_quiet_runnable(args.quiet_runnable, args.quiet_limit)
    rec = {"launch": launch, "runnable_pre": pre,
           "competitors": [c[2] for c in busy], "tokens": args.tokens,
           "blocktime_us": args.blocktime, "arms": {}}
    # Rotate which width goes first so a warm page cache or a monotone drift
    # cannot systematically favour one width.
    order = widths if launch % 2 == 0 else tuple(reversed(widths))
    rec["order"] = list(order)
    extra = ({"ONNX_GENAI_CPU_DECODE_BLOCKTIME_US": args.blocktime}
             if args.blocktime is not None else None)
    with H.LoadWatch() as watch:
        for w in order:
            n = H.native(args.binary, MODEL, BLOCK, ACC, w, SESSIONS,
                         args.tokens, args.reps, extra=extra)
            o = H.ort(MODEL, BLOCK, ACC, w, SESSIONS, args.tokens, args.reps)
            rec["arms"][str(w)] = {"native": n, "ort": o}
    rec["runnable_peak"] = watch.peak
    trusted = (watch.peak <= max(widths) + args.slack) and not busy

    for arm in ("native", "ort"):
        a8 = rec["arms"]["8"][arm]
        a16 = rec["arms"]["16"][arm]
        d8 = derive(a8, 8, args.tokens * SESSIONS)
        d16 = derive(a16, 16, args.tokens * SESSIONS)
        # The headline `tps` is a median over repetitions and is kept for
        # comparability with the merged records; the decision runs on the
        # single-repetition `tps` that the CPU fields came from, so that
        # speedup, `R_cpu` and `R_busy` all describe the same two runs.
        out = {"speedup_headline": a16["tps"] / a8["tps"]}
        if d8 and d16:
            speedup = d16["tps"] / d8["tps"]
            err, predicted = identity_error(d8, d16, speedup)
            out.update({
                "speedup": speedup,
                "r_cpu": d16["cpu_per_token"] / d8["cpu_per_token"],
                "r_busy": d16["busy"] / d8["busy"],
                "busy8": d8["busy"], "busy16": d16["busy"],
                "cpt8": d8["cpu_per_token"], "cpt16": d16["cpu_per_token"],
                "sys_frac8": d8["sys_frac"], "sys_frac16": d16["sys_frac"],
                "d_sys_frac": d16["sys_frac"] - d8["sys_frac"],
                "identity_err": err, "identity_predicted": predicted,
            })
            if err > IDENTITY_TOLERANCE:
                trusted = False
                rec.setdefault("untrusted_reason", []).append(
                    f"{arm} identity error {err:.1%}")
        else:
            out["speedup"] = out["speedup_headline"]
            trusted = False
            rec.setdefault("untrusted_reason", []).append(f"{arm} has no cpu row")
        rec[arm] = out

    rec["trusted"] = trusted
    return rec


def med(cells, arm, key):
    vals = [c[arm][key] for c in cells if key in c[arm]]
    return statistics.median(vals) if vals else float("nan")


def sign_fraction(cells, arm, key, predicate):
    vals = [c[arm][key] for c in cells if key in c[arm]]
    if not vals:
        return 0.0
    return sum(1 for v in vals if predicate(v)) / len(vals)


def verdict(cells):
    """The pre-registered rule, in one place, applied to the native arm."""
    n = len(cells)
    if n < MIN_TRUSTED:
        return f"REPORT NOTHING (n_trusted={n} < {MIN_TRUSTED})"
    speedup = med(cells, "native", "speedup")
    if speedup >= NO_LOSS_SPEEDUP:
        return (f"NO-LOSS (median speedup {speedup:.3f} >= {NO_LOSS_SPEEDUP}) "
                "-- the scaling wall did not reproduce in this run")
    r_cpu = med(cells, "native", "r_cpu")
    r_busy = med(cells, "native", "r_busy")
    burn_sign = sign_fraction(cells, "native", "r_cpu", lambda v: v > 1.0)
    idle_sign = sign_fraction(cells, "native", "r_busy", lambda v: v < 1.0)

    if r_cpu >= BURN_R_CPU and r_busy >= BURN_R_BUSY:
        if burn_sign >= SIGN_FRACTION:
            return (f"BURN-DOMINATED (R_cpu={r_cpu:.3f} >= {BURN_R_CPU}, "
                    f"R_busy={r_busy:.3f} >= {BURN_R_BUSY}, "
                    f"sign {burn_sign:.0%})")
        return (f"INCONCLUSIVE: BURN medians cleared (R_cpu={r_cpu:.3f}, "
                f"R_busy={r_busy:.3f}) but sign only {burn_sign:.0%} "
                f"< {SIGN_FRACTION:.0%}")
    if r_busy <= IDLE_R_BUSY and r_cpu <= IDLE_R_CPU:
        if idle_sign >= SIGN_FRACTION:
            return (f"IDLE-DOMINATED (R_busy={r_busy:.3f} <= {IDLE_R_BUSY}, "
                    f"R_cpu={r_cpu:.3f} <= {IDLE_R_CPU}, "
                    f"sign {idle_sign:.0%})")
        return (f"INCONCLUSIVE: IDLE medians cleared (R_busy={r_busy:.3f}, "
                f"R_cpu={r_cpu:.3f}) but sign only {idle_sign:.0%} "
                f"< {SIGN_FRACTION:.0%}")
    if r_cpu >= MIXED_R_CPU and r_busy <= MIXED_R_BUSY:
        if burn_sign >= SIGN_FRACTION and idle_sign >= SIGN_FRACTION:
            return (f"MIXED (R_cpu={r_cpu:.3f} >= {MIXED_R_CPU} and "
                    f"R_busy={r_busy:.3f} <= {MIXED_R_BUSY}, signs "
                    f"{burn_sign:.0%}/{idle_sign:.0%})")
        return (f"INCONCLUSIVE: MIXED medians cleared but signs "
                f"{burn_sign:.0%}/{idle_sign:.0%} < {SIGN_FRACTION:.0%}")
    return (f"INCONCLUSIVE (R_cpu={r_cpu:.3f}, R_busy={r_busy:.3f}, "
            f"speedup={speedup:.3f}) -- no branch of the pre-registered rule "
            "is satisfied")


def report(cells):
    tr = [c for c in cells if c["trusted"]]
    print()
    print(f"VERDICT: {verdict(tr)}")
    if len(tr) < MIN_TRUSTED:
        return
    print()
    print(f"{'arm':>7} {'speedup':>8} {'hdline':>7} {'R_cpu':>7} {'R_busy':>7} "
          f"{'busy8':>7} {'busy16':>7} {'cpt8':>9} {'cpt16':>9} "
          f"{'sysf8':>7} {'sysf16':>7}")
    for arm in ("native", "ort"):
        print(f"{arm:>7} {med(tr, arm, 'speedup'):>8.3f} "
              f"{med(tr, arm, 'speedup_headline'):>7.3f} "
              f"{med(tr, arm, 'r_cpu'):>7.3f} {med(tr, arm, 'r_busy'):>7.3f} "
              f"{med(tr, arm, 'busy8'):>7.3f} {med(tr, arm, 'busy16'):>7.3f} "
              f"{med(tr, arm, 'cpt8'):>9.5f} {med(tr, arm, 'cpt16'):>9.5f} "
              f"{med(tr, arm, 'sys_frac8'):>7.3f} "
              f"{med(tr, arm, 'sys_frac16'):>7.3f}")
    print()
    for arm in ("native", "ort"):
        d = med(tr, arm, "d_sys_frac")
        s = sign_fraction(tr, arm, "d_sys_frac", lambda v: v > 0)
        named = d >= SYS_FRAC_RISE and s >= SIGN_FRACTION
        print(f"{arm}: sys_frac rise {d:+.3f} (sign {s:.0%}) -- "
              f"{'NAMED as a contributor' if named else 'not named'}")
    print(f"identity self-test: max error over trusted cells "
          f"{max(c['native']['identity_err'] for c in tr):.2%} "
          f"(tolerance {IDENTITY_TOLERANCE:.0%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--launches", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--slack", type=int, default=10)
    ap.add_argument("--quiet-runnable", type=int, default=4)
    ap.add_argument("--quiet-limit", type=int, default=600)
    ap.add_argument("--with-w4", action="store_true",
                    help="also measure width 4 for context; never part of the "
                         "pre-registered decision")
    ap.add_argument("--blocktime", default=None,
                    help="ONNX_GENAI_CPU_DECODE_BLOCKTIME_US for the native "
                         "arm. Pass 0 to measure with the spin-wait unmasked: "
                         "`busy` counts a spinning worker as working, so at "
                         "the shipped 500 us default an idle pool reports as "
                         "occupied and the idle/burn split is unreadable. "
                         "Measured: at w=16 `busy` reads 0.953 at 500 us and "
                         "0.692 at 0 for the same throughput.")
    ap.add_argument("--deadline-min", type=float, default=0.0,
                    help="stop at the first launch boundary after this many "
                         "minutes. A clock stops the run; the numbers never "
                         "do. Stopping when a matrix 'looks done' is how a "
                         "flattering intermediate state gets published, so "
                         "the budget is set before the first launch and "
                         "enforced here rather than by the operator.")
    ap.add_argument("--out", default="w8_w16_cpu_split.json")
    ap.add_argument("--replay", help="re-score an existing JSON, run nothing")
    args = ap.parse_args()

    if args.replay:
        with open(args.replay) as f:
            report(json.load(f))
        return

    args.binary = os.path.abspath(args.binary)
    widths = (4,) + DECISION_WIDTHS if args.with_w4 else DECISION_WIDTHS

    print(f"# pre-registered: n>={MIN_TRUSTED}; NO-LOSS>={NO_LOSS_SPEEDUP}; "
          f"BURN R_cpu>={BURN_R_CPU} & R_busy>={BURN_R_BUSY}; "
          f"IDLE R_busy<={IDLE_R_BUSY} & R_cpu<={IDLE_R_CPU}; "
          f"MIXED R_cpu>={MIXED_R_CPU} & R_busy<={MIXED_R_BUSY}; "
          f"sign>={SIGN_FRACTION:.0%}")
    hdr = (f"{'L':>3} {'pk':>3} {'T':>2} {'nat8':>7} {'nat16':>7} "
           f"{'natX':>6} {'R_cpu':>6} {'R_bsy':>6} {'ortX':>6} "
           f"{'oRcpu':>6} {'oRbsy':>6} {'id%':>5}")
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
            c = one_launch(args, launch, widths)
        except Exception as e:
            sys.stderr.write(f"launch {launch} failed: {e}\n")
            continue
        cells.append(c)
        n, o = c["native"], c["ort"]
        print(f"{c['launch']:>3} {c['runnable_peak']:>3} "
              f"{'y' if c['trusted'] else 'N':>2} "
              f"{c['arms']['8']['native']['tps']:>7.1f} "
              f"{c['arms']['16']['native']['tps']:>7.1f} "
              f"{n['speedup']:>6.3f} {n.get('r_cpu', float('nan')):>6.3f} "
              f"{n.get('r_busy', float('nan')):>6.3f} "
              f"{o['speedup']:>6.3f} {o.get('r_cpu', float('nan')):>6.3f} "
              f"{o.get('r_busy', float('nan')):>6.3f} "
              f"{100 * n.get('identity_err', float('nan')):>5.2f}", flush=True)
        with open(args.out, "w") as f:
            json.dump(cells, f, indent=1)

    report(cells)


if __name__ == "__main__":
    main()
