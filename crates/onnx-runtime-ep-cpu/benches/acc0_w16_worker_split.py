#!/usr/bin/env python3
"""Split the width-16 *idle* half into wake latency, imbalance and dispatch gap.

`acc0_w8_w16_cpu_split.py` established that at width 16 native leaves ~40% of
the sixteen cores unused once the spin-wait mask is removed (`busy` 0.938 at
w=8 -> 0.595 at w=16, `R_busy = 0.652`, 100% sign consistency).  That is *what*,
not *why*.  Three mechanisms produce identical aggregate idleness and want
opposite fixes:

  * **wake latency** -- the op is published and the worker has not noticed yet.
    Fix is in the wait/wake path.  Measured directly as `wake_ns`.
  * **load imbalance** -- one worker's shard is larger, everyone else finishes
    and waits at the barrier.  Fix is in the partitioner.  Shows as skew in
    `work_ns` and, far more sensitively, as concentration in `last_arrivals`.
  * **dispatch gap** -- the next op has not been published at all, because the
    dispatcher is running its own shard or the serial part of the step.  Fix is
    in the dispatcher/schedule.  Not directly counted; obtained as the residual.

The identity that makes the residual meaningful, per worker, over the *same*
window that produced `wall`:

    wall  ==  work_ns + wake_ns + residual_ns

`work_ns` and `wake_ns` are read from `decode_spmd::SpmdWorkerProfile` deltas
bracketed by exactly the two points that bracket `wall`, so the residual is a
measurement of "the op I am waiting for does not exist yet" and not a slop
term.  A negative residual would mean the counters overlap the window and is
treated as an instrument fault, not clamped to zero.

Straggler concentration is the sharp instrument here.  `last_arrivals` summed
over a node equals that node's dispatch count exactly, so with `w` spawned
workers chance alone gives each worker `1/w` of them.  A worker holding much
more than `1/w` is systematically last, and *every other worker waits for it on
every one of those ops*.  This is detectable long before it is visible in
`work_ns`: a 3% larger shard produces a 3% `work_ns` skew and a 60%+
`last_arrivals` share.

PROFILED RUNS ARE DIAGNOSTICS, NOT TIMINGS
------------------------------------------
`ONNX_GENAI_CPU_DECODE_WORKER_PROFILE` costs two clock reads per worker per op
(~4% at width 12 by the in-tree estimate).  Both widths here are profiled, so
the *comparison* is fair, but no throughput number from this harness may be
quoted against an unprofiled arm.  The harness refuses to print a tps ratio for
that reason.

BLOCKTIME
---------
Run at `--blocktime 0` by default.  At the shipped 500 us the workers spin
through what would otherwise be idle, which is precisely the masking that made
the first CPU-split run score BURN-DOMINATED; the residual and wake fractions
are only interpretable once the wait actually waits.  `--blocktime 500` is
accepted so the two configurations can be contrasted, and the value is recorded
in every row.

PRE-REGISTERED RULE (written before the first run; do not edit after seeing
data -- add a new rule with a new name instead)
--------------------------------------------------------------------------
For each width, over trusted launches, take the median across launches of each
per-launch statistic.  Compare w=16 against w=8.  **That pair is part of the
rule**: it is fixed in code as `RULE_NARROW`/`RULE_WIDE` and is not taken from
`--widths`, because the Amdahl calibration below is extremely sensitive to the
baseline and a re-based verdict is indistinguishable in the output from a
correct one.

    n_trusted >= 5 required for any verdict at all, else REPORT NOTHING.
    Any launch with a per-worker residual fraction outside [-0.02, 1.02] is an
    instrument fault: the launch is discarded and counted separately.

    WAKE-BOUND      if wake_frac(16) - wake_frac(8) >= 0.10
                    and wake_frac(16) >= 0.15
    IMBALANCE-BOUND if straggler_excess(16) >= 3.0
                    and work_skew(16) - work_skew(8) >= 0.05
    DISPATCH-BOUND  if resid_frac(16) - resid_frac(8) >= 0.10
                    and resid_frac(16) >= 0.15

    where
      wake_frac    = mean_over_workers(wake_ns) / wall
      work_frac    = mean_over_workers(work_ns) / wall
      resid_frac   = 1 - wake_frac - work_frac
      work_skew    = max(work_ns)/mean(work_ns) - 1
      straggler_excess = max(last_arrivals share) / (1/n_workers)

More than one may fire; all that fire are reported, in descending order of the
share of the idle they account for.  If none fires the verdict is
UNATTRIBUTED and the fractions are still printed -- a null result here is
informative and must not be rescued by relaxing a threshold.

    sign consistency >= 0.80 required on each fired condition's underlying
    per-launch difference, else the condition is reported as UNSTABLE rather
    than as a verdict.
"""

import argparse
import contextlib
import io
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import acc0_gap_matrix as H  # noqa: E402

MODEL = "llama"
BLOCK = 32
ACC = 0
SESSIONS = 1

MIN_TRUSTED = 5
SIGN_FRACTION = 0.80
WAKE_RISE = 0.10
WAKE_FLOOR = 0.15
STRAGGLER_EXCESS = 3.0
SKEW_RISE = 0.05
RESID_RISE = 0.10
RESID_FLOOR = 0.15
RESID_BOUNDS = (-0.02, 1.02)

#: The widths the pre-registered rule is written against ("Compare w=16 against
#: w=8"). This is part of the rule, not a default, so it is *not* derived from
#: `--widths`: deriving it let `--widths 2,8,16` silently re-baseline every
#: verdict onto w=2 and invert a published conclusion (see
#: `docs/benchmarks/2026-08-26-acc0-w16-baseline-correction.md`). Extra widths
#: may be measured and printed, but they are descriptive only.
RULE_NARROW = 8
RULE_WIDE = 16


def parse_workers(text, phase="steady"):
    """Pull the `worker phase=... k=v` rows for one phase out of bench stdout."""
    out = []
    for line in text.splitlines():
        if not line.startswith("worker "):
            continue
        kv = {}
        for tok in line.split()[1:]:
            if "=" not in tok:
                continue
            k, v = tok.split("=", 1)
            kv[k] = v
        if kv.get("phase") != phase:
            continue
        try:
            out.append(
                {
                    "idx": int(kv["idx"]),
                    "cpu": int(kv["cpu"]),
                    "timed_ops": int(kv["timed_ops"]),
                    "wake_ns": int(kv["wake_ns"]),
                    "work_ns": int(kv["work_ns"]),
                    "last_arrivals": int(kv["last_arrivals"]),
                    "parks": int(kv["parks"]),
                    "spin_hits": int(kv["spin_hits"]),
                    "wall_s": float(kv["wall_s"]),
                }
            )
        except (KeyError, ValueError):
            return []
    return out


def derive(workers):
    """Per-launch statistics for one width. Returns None if unmeasurable."""
    if not workers:
        return None
    wall = workers[0]["wall_s"]
    if wall <= 0:
        return None
    if any(w["timed_ops"] == 0 for w in workers):
        # Profiling was not actually on: `wake_ns`/`work_ns` are structurally
        # zero, which would otherwise read as "no wake latency, all residual"
        # -- a vacuous run scoring as a strong DISPATCH-BOUND result.
        return None
    n = len(workers)
    wake = [w["wake_ns"] / 1e9 for w in workers]
    work = [w["work_ns"] / 1e9 for w in workers]
    arr = [w["last_arrivals"] for w in workers]
    total_arr = sum(arr)
    wake_frac = statistics.fmean(wake) / wall
    work_frac = statistics.fmean(work) / wall
    resid_frac = 1.0 - wake_frac - work_frac
    mean_work = statistics.fmean(work)
    return {
        "n_workers": n,
        "wall_s": wall,
        "wake_frac": wake_frac,
        "work_frac": work_frac,
        "resid_frac": resid_frac,
        "work_skew": (max(work) / mean_work - 1.0) if mean_work > 0 else 0.0,
        "straggler_share": (max(arr) / total_arr) if total_arr else 0.0,
        "straggler_excess": ((max(arr) / total_arr) * n) if total_arr else 0.0,
        "straggler_idx": workers[arr.index(max(arr))]["idx"],
        "parks_per_worker": statistics.fmean([w["parks"] for w in workers]),
        "ops": workers[0]["timed_ops"],
    }


def trusted(rec):
    """A launch is trusted iff every width produced an in-range residual."""
    if rec.get("peak", 0) > rec.get("peak_limit", 10**9):
        return False
    for st in rec["widths"].values():
        if st is None:
            return False
        lo, hi = RESID_BOUNDS
        if not lo <= st["resid_frac"] <= hi:
            return False
    return True


def med(rows, width, key):
    return statistics.median([r["widths"][str(width)][key] for r in rows])


def sign_fraction(rows, wide, narrow, key, positive=True):
    diffs = [
        r["widths"][str(wide)][key] - r["widths"][str(narrow)][key] for r in rows
    ]
    if not diffs:
        return 0.0
    hits = sum(1 for d in diffs if (d > 0) == positive)
    return hits / len(diffs)


def rule_pair(widths):
    """The pre-registered comparison pair, or `None` if it was not measured.

    Deliberately ignores `widths` except to check that both members are
    present. The pair is a clause of the pre-registered rule; letting the
    command line choose it means the printed verdict answers a different
    question from the one the rule asks, with no indication in the output.
    """
    if RULE_NARROW in widths and RULE_WIDE in widths:
        return RULE_NARROW, RULE_WIDE
    return None


def verdict(rows, widths):
    n = len(rows)
    if n < MIN_TRUSTED:
        return [f"REPORT NOTHING (n_trusted={n} < {MIN_TRUSTED})"], {}
    pair = rule_pair(widths)
    if pair is None:
        return ([f"REPORT NOTHING (the pre-registered rule compares "
                 f"w={RULE_WIDE} against w={RULE_NARROW}; this run measured "
                 f"{','.join(str(w) for w in widths)})"], {})
    narrow, wide = pair
    m = {
        k: {w: med(rows, w, k) for w in (narrow, wide)}
        for k in (
            "wake_frac",
            "work_frac",
            "resid_frac",
            "work_skew",
            "straggler_excess",
            "parks_per_worker",
        )
    }
    fired = []

    def note(name, cond, share, key, positive=True):
        if not cond:
            return
        sf = sign_fraction(rows, wide, narrow, key, positive)
        label = name if sf >= SIGN_FRACTION else f"{name} (UNSTABLE, sign {sf:.0%})"
        fired.append((share, label, sf))

    dw = m["wake_frac"][wide] - m["wake_frac"][narrow]
    note(
        f"WAKE-BOUND (wake_frac {m['wake_frac'][narrow]:.3f} -> "
        f"{m['wake_frac'][wide]:.3f}, +{dw:.3f})",
        dw >= WAKE_RISE and m["wake_frac"][wide] >= WAKE_FLOOR,
        m["wake_frac"][wide],
        "wake_frac",
    )
    ds = m["work_skew"][wide] - m["work_skew"][narrow]
    note(
        f"IMBALANCE-BOUND (straggler_excess {m['straggler_excess'][wide]:.2f}x "
        f"chance, work_skew {m['work_skew'][narrow]:.3f} -> "
        f"{m['work_skew'][wide]:.3f})",
        m["straggler_excess"][wide] >= STRAGGLER_EXCESS and ds >= SKEW_RISE,
        m["work_skew"][wide],
        "work_skew",
    )
    dr = m["resid_frac"][wide] - m["resid_frac"][narrow]
    note(
        f"DISPATCH-BOUND (resid_frac {m['resid_frac'][narrow]:.3f} -> "
        f"{m['resid_frac'][wide]:.3f}, +{dr:.3f})",
        dr >= RESID_RISE and m["resid_frac"][wide] >= RESID_FLOOR,
        m["resid_frac"][wide],
        "resid_frac",
    )
    if not fired:
        return ["UNATTRIBUTED (no pre-registered condition fired)"], m
    fired.sort(key=lambda t: -t[0])
    return [lbl for _, lbl, _ in fired], m


def run_width(args, w):
    """One profiled native launch at width `w`; returns raw stdout+stderr.

    Mirrors `acc0_gap_matrix.native()`'s invocation exactly -- pinned to all 16
    even CPUs, because `ONNX_GENAI_CPU_DECODE_THREADS` confines the process to
    `w` CPUs itself and doing it twice would hand the arm a smaller machine
    than the runtime intends.
    """
    env = {
        "PROBE_MODEL": MODEL, "PROBE_BLOCK": BLOCK, "PROBE_ACCURACY": ACC,
        "PROBE_SESSIONS": SESSIONS, "PROBE_TOKENS": args.tokens,
        "PROBE_REPS": args.reps,
        "ONNX_GENAI_CPU_DECODE_THREADS": w,
        "ONNX_GENAI_CPU_DECODE_WORKER_PROFILE": "1",
        "ONNX_GENAI_CPU_DECODE_BLOCKTIME_US": args.blocktime,
    }
    # Unset means the SHIPPED default, which is what almost every row should
    # run at. Setting it to the default's current value instead would look
    # identical today and silently pin the arm to a stale number the day the
    # default moves.
    if args.steal_tiles is not None:
        env["ONNX_GENAI_CPU_DECODE_STEAL_TILES_PER_WORKER"] = args.steal_tiles
    r = H.sh(f"taskset -c {H.PIN} {args.binary}", env)
    return r.stdout + "\n" + r.stderr


def one_launch(args, launch, widths):
    rec = {"launch": launch, "blocktime_us": args.blocktime, "widths": {},
           "steal_tiles": args.steal_tiles, "peak_limit": args.quiet_limit}
    order = widths if launch % 2 == 0 else tuple(reversed(widths))
    rec["order"] = list(order)
    with H.LoadWatch() as watch:
        for w in order:
            rec["widths"][str(w)] = derive(parse_workers(run_width(args, w)))
    rec["peak"] = watch.peak
    return rec


def steal_tiles_label(recs):
    """How the steal-tile knob was set, for the report header and a replay.

    Three distinct answers, and collapsing any two of them loses the fact that
    matters.  `unrecorded` is a dataset archived before this key existed: it
    ran at whatever its binary's default was, which is NOT necessarily today's
    default, so calling it `default (shipped)` would assert a value nobody
    measured.  Same doctrine as the absent-key handling in `hostlock.sh`.
    """
    if not recs:
        return "?"
    if "steal_tiles" not in recs[0]:
        return "unrecorded (dataset predates the knob)"
    v = recs[0]["steal_tiles"]
    return "default (shipped, unset)" if v is None else str(v)


def report(recs, widths):
    rows = [r for r in recs if trusted(r)]
    pair = rule_pair(widths)
    print(
        f"\n# pre-registered: n>={MIN_TRUSTED}; WAKE +{WAKE_RISE} & >={WAKE_FLOOR}; "
        f"IMBALANCE excess>={STRAGGLER_EXCESS} & skew +{SKEW_RISE}; "
        f"DISPATCH +{RESID_RISE} & >={RESID_FLOOR}; sign>={SIGN_FRACTION:.0%}"
    )
    print(f"# verdict pair: w{RULE_NARROW} vs w{RULE_WIDE} (pre-registered; "
          f"not taken from --widths)")
    extras = [w for w in widths if w not in (RULE_NARROW, RULE_WIDE)]
    if extras:
        print("# descriptive only, NOT used by the rule or the decomposition: "
              + ", ".join(f"w{w}" for w in extras))
    print(f"# blocktime_us={recs[0]['blocktime_us'] if recs else '?'} "
          f"(profiled run: diagnostic only, tps deliberately not reported)")
    print(f"# steal_tiles={steal_tiles_label(recs)}")
    hdr = (f"{'L':>3} {'pk':>4} {'T':>2} " +
           " ".join(f"{'w'+str(w)+'.'+k:>12}"
                    for w in widths
                    for k in ("wake", "work", "resid", "strag")))
    print(hdr)
    print("-" * len(hdr))
    for r in recs:
        cells = []
        for w in widths:
            st = r["widths"].get(str(w))
            if st is None:
                cells += [f"{'--':>12}"] * 4
            else:
                cells += [
                    f"{st['wake_frac']:>12.3f}",
                    f"{st['work_frac']:>12.3f}",
                    f"{st['resid_frac']:>12.3f}",
                    f"{st['straggler_excess']:>12.2f}",
                ]
        print(f"{r['launch']:>3} {r.get('peak', 0):>4} "
              f"{'y' if trusted(r) else 'N':>2} " + " ".join(cells))

    verdicts, m = verdict(rows, widths)
    print(f"\nn_trusted={len(rows)} of {len(recs)}")
    for v in verdicts:
        print(f"VERDICT: {v}")
    if not m:
        return
    narrow, wide = pair
    print(f"\n{'stat':>18} {'w'+str(narrow):>10} {'w'+str(wide):>10} {'delta':>10}")
    for k, vals in m.items():
        print(f"{k:>18} {vals[narrow]:>10.3f} {vals[wide]:>10.3f} "
              f"{vals[wide]-vals[narrow]:>+10.3f}")
    for w in widths:
        idxs = [r["widths"][str(w)]["straggler_idx"] for r in rows]
        share = statistics.median(
            [r["widths"][str(w)]["straggler_share"] for r in rows])
        nw = rows[0]["widths"][str(w)]["n_workers"]
        mode = max(set(idxs), key=idxs.count)
        print(f"w={w}: {nw} spawned workers, chance share {1/nw:.3f}; "
              f"observed max share {share:.3f} held by worker {mode} in "
              f"{idxs.count(mode)}/{len(idxs)} launches")

    barrier_decomposition(m, narrow, wide)


def barrier_decomposition(m, narrow, wide):
    """Descriptive split of the window; NOT part of the pre-registered rule.

    In a barrier-synchronised step the wall is set by the *slowest* worker, so
    the mean worker's idle time separates into three pieces that want different
    fixes:

      straggler wait = work_frac * work_skew
          how long the average worker sits at the barrier because one worker's
          shard ran longer. Recoverable by partitioning or stealing.
      wake           = wake_frac, as measured.
      dispatcher     = the remainder: no op published, i.e. the serial part of
          the step plus dispatch overhead.

    The dispatcher share is then compared against what a *constant* serial
    fraction alone would predict at the wider width, calibrated on the narrow
    width -- Amdahl with no defect. Serial time that merely scales as Amdahl
    says it must is not a bug and must not be counted as recoverable; only the
    excess over that prediction is.

    The calibration is only as good as the narrow width, and it is *sharply*
    sensitive to it, which is why the pair is pinned. Calibrating on w=2 says
    the serial excess at w=16 is +0.086; calibrating on the pre-registered w=8
    says -0.009 -- opposite conclusions from one dataset. w=2 is the wrong
    baseline: it spawns a single worker, so the dispatcher computes half the op
    inline and there is almost no barrier to be serial *at*, which extrapolates
    to an implausibly small serial term at w=16.
    """
    print("\nbarrier decomposition (descriptive, not part of the rule)")
    print(f"{'':>18} {'w'+str(narrow):>10} {'w'+str(wide):>10}")
    parts = {}
    for w in (narrow, wide):
        work = m["work_frac"][w]
        strag = work * m["work_skew"][w]
        wake = m["wake_frac"][w]
        parts[w] = {
            "useful work": work,
            "straggler wait": strag,
            "wake latency": wake,
            "dispatcher/serial": max(0.0, 1.0 - work - strag - wake),
        }
    for k in ("useful work", "straggler wait", "wake latency",
              "dispatcher/serial"):
        print(f"{k:>18} {parts[narrow][k]:>10.3f} {parts[wide][k]:>10.3f}")

    # Amdahl calibration. At the narrow width the mean worker is busy
    # `work_frac`, so serial/parallel = (1 - work_n) / work_n in units of the
    # narrow width's parallel time. Doubling the width halves the parallel part
    # and leaves the serial part alone.
    wn = m["work_frac"][narrow]
    if wn <= 0:
        return
    ratio = wide / narrow
    serial_over_par = (1.0 - wn) / wn
    predicted_work = 1.0 / (1.0 + serial_over_par * ratio)
    predicted_serial = 1.0 - predicted_work
    observed_serial = parts[wide]["dispatcher/serial"]
    print(
        f"\nAmdahl check: holding the narrow width's serial time constant and "
        f"dividing its parallel time by {ratio:g} predicts mean useful work "
        f"{predicted_work:.3f} at w={wide} (serial {predicted_serial:.3f}); "
        f"observed dispatcher/serial is {observed_serial:.3f}."
    )
    excess = observed_serial - predicted_serial
    print(
        f"  serial excess over Amdahl: {excess:+.3f}  |  "
        f"straggler wait: {parts[wide]['straggler wait']:.3f}  |  "
        f"wake: {parts[wide]['wake latency']:.3f}"
    )
    print(
        "  Recoverable-in-principle at this width is the straggler wait plus "
        f"the serial excess ({parts[wide]['straggler wait'] + max(0.0, excess):.3f} "
        "of the window); the Amdahl-predicted serial part is not a defect."
    )


def synthetic_rows(n, per_width):
    """Trusted-looking launches with exactly the per-width stats given.

    Only for `--self-test`: the fields are the ones `verdict` and
    `barrier_decomposition` read, with `peak` under `peak_limit` and residuals
    in range so `trusted` accepts them.
    """
    rows = []
    for i in range(n):
        rows.append({
            "launch": i, "peak": 1, "peak_limit": 40, "blocktime_us": 0,
            "widths": {str(w): dict(st) for w, st in per_width.items()},
        })
    return rows


def self_test():
    """Assert the rule pair cannot be re-based by `--widths`.

    This is the defect the pinning fixes: a run with `--widths 2,8,16` used to
    compare w=16 against w=2 and print a verdict that looked identical to the
    pre-registered one.
    """
    # w=8 -> w=16 residual rise of 0.09 is *below* RESID_RISE, so DISPATCH must
    # not fire; a w=2 baseline would show a rise of 0.17 and fire it.
    stats = {
        2: {"wake_frac": 0.000, "work_frac": 0.997, "resid_frac": 0.003,
            "work_skew": 0.000, "straggler_excess": 1.00,
            "parks_per_worker": 2.0, "n_workers": 1,
            "straggler_idx": 0, "straggler_share": 1.0},
        8: {"wake_frac": 0.009, "work_frac": 0.909, "resid_frac": 0.082,
            "work_skew": 0.027, "straggler_excess": 2.45,
            "parks_per_worker": 4.9, "n_workers": 7,
            "straggler_idx": 0, "straggler_share": 0.35},
        16: {"wake_frac": 0.014, "work_frac": 0.819, "resid_frac": 0.173,
             "work_skew": 0.078, "straggler_excess": 3.03,
             "parks_per_worker": 0.5, "n_workers": 15,
             "straggler_idx": 6, "straggler_share": 0.20},
    }
    rows = synthetic_rows(MIN_TRUSTED + 2, stats)

    for widths in ((8, 16), (2, 8, 16), (16, 8), (2, 16, 8, 4)):
        lines, m = verdict(rows, widths)
        joined = " | ".join(lines)
        assert m, f"{widths}: expected a scored verdict, got {joined}"
        assert set(m["resid_frac"]) == {RULE_NARROW, RULE_WIDE}, (
            f"{widths}: scored on {sorted(m['resid_frac'])}, not the "
            f"pre-registered pair")
        assert not any("DISPATCH" in ln for ln in lines), (
            f"{widths}: DISPATCH fired on a +0.09 rise -- baseline is not "
            f"w={RULE_NARROW} (got {joined})")
        assert any("IMBALANCE" in ln for ln in lines), (
            f"{widths}: IMBALANCE should fire (got {joined})")

    # Missing either member of the pair must report nothing rather than
    # silently substituting whatever widths are present.
    for widths in ((2, 16), (2, 8), (4, 16)):
        lines, m = verdict(rows, widths)
        assert not m and "REPORT NOTHING" in lines[0], (
            f"{widths}: expected REPORT NOTHING, got {lines}")

    # Too few trusted launches still outranks everything else.
    lines, m = verdict(synthetic_rows(MIN_TRUSTED - 1, stats), (8, 16))
    assert not m and "n_trusted" in lines[0], lines

    # And the sign-consistency guard still reads the pinned pair.
    assert sign_fraction(rows, RULE_WIDE, RULE_NARROW, "resid_frac") == 1.0

    # The defect lived in `report()` as well as in `verdict()`, and `report()`
    # is the only caller of `barrier_decomposition` -- the Amdahl calibration
    # whose sign flipped. Scoring the verdict on the pinned pair while the
    # decomposition re-based onto w=2 would print a correct VERDICT line above
    # a wrong serial excess, so drive the whole report and check the pair
    # reached the calibration too.
    for widths in ((8, 16), (2, 8, 16)):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            report(rows, widths)
        out = buf.getvalue()
        assert "# verdict pair: w8 vs w16" in out, out
        assert "\n{:>18} {:>10} {:>10}\n".format("", "w8", "w16") in out, (
            "the barrier decomposition was not scored on the pinned pair:\n"
            + out)
        # ratio = wide/narrow: 2 on the pinned pair, 8 if re-based onto w=2.
        assert "dividing its parallel time by 2 " in out, (
            "the Amdahl calibration used the wrong baseline:\n" + out)
        if 2 in widths:
            assert "descriptive only" in out and "w2" in out, out

    # The steal-tile knob is the arm label for the #2071 tiles=1-vs-shipped
    # comparison, so a report that misstates it misattributes the whole arm --
    # the same class of defect as the re-based baseline this self-test exists
    # for, one field along. Three cases, all distinct on purpose.
    explicit = synthetic_rows(MIN_TRUSTED + 2, stats)
    for r in explicit:
        r["steal_tiles"] = 1
    assert steal_tiles_label(explicit) == "1", steal_tiles_label(explicit)

    shipped = synthetic_rows(MIN_TRUSTED + 2, stats)
    for r in shipped:
        r["steal_tiles"] = None
    assert steal_tiles_label(shipped).startswith("default"), (
        steal_tiles_label(shipped))

    # An archived dataset predating the knob ran at ITS binary's default, not
    # necessarily today's, so it must not be labelled `default`.
    archived = synthetic_rows(MIN_TRUSTED + 2, stats)
    for r in archived:
        r.pop("steal_tiles", None)
    assert steal_tiles_label(archived).startswith("unrecorded"), (
        "a dataset predating the knob was labelled as if its arm were known: "
        + steal_tiles_label(archived))

    # And the label has to reach the report, not merely be computable.
    for rows_, want in ((explicit, "# steal_tiles=1"),
                        (archived, "# steal_tiles=unrecorded")):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            report(rows_, (8, 16))
        assert want in buf.getvalue(), (
            f"expected {want!r} in the report header:\n" + buf.getvalue())

    print(f"self-test OK: verdict pinned to w{RULE_NARROW} vs w{RULE_WIDE} "
          f"under every --widths permutation tried; steal-tile arm label "
          f"distinguishes explicit, shipped-default and unrecorded")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary")
    ap.add_argument("--launches", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=192)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--widths", default="8,16")
    ap.add_argument("--blocktime", type=int, default=0)
    ap.add_argument("--steal-tiles", type=int, default=None)
    ap.add_argument("--quiet-limit", type=int, default=40)
    ap.add_argument("--out", default=None)
    ap.add_argument("--replay", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    widths = tuple(int(x) for x in args.widths.split(","))
    if args.replay:
        with open(args.replay) as fh:
            blob = json.load(fh)
        # Datasets taken before the lock was adopted are a bare list.
        recs = blob["runs"] if isinstance(blob, dict) else blob
        state = (blob.get("hostlock", {}).get("hostlock_state", "unrecorded")
                 if isinstance(blob, dict) else "unrecorded (pre-lock dataset)")
        print(f"hostlock at acquire: {state}")
        report(recs, widths)
        return

    if not args.binary:
        ap.error("--binary is required unless --replay or --self-test is used")

    recs = []
    # The sweep holds the box for its whole duration; the report afterwards is
    # arithmetic and runs unlocked.
    with H.HostLock(owner=os.environ.get("ONNX_GENAI_BENCH_OWNER", "roy"),
                    reason=f"acc0 w16 worker split, {args.launches} launches "
                           f"widths={args.widths}") as lock:
        for launch in range(args.launches):
            recs.append(one_launch(args, launch, widths))
            if args.out:
                with open(args.out, "w") as fh:
                    json.dump({"hostlock": lock.provenance, "runs": recs},
                              fh, indent=1)
            time.sleep(1)
    report(recs, widths)


if __name__ == "__main__":
    main()
