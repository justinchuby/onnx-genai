#!/usr/bin/env python3
"""Does work-stealing absorb the width-16 straggler?

WHY THIS EXISTS
---------------
The straggler is now characterised and every mechanism proposed for it has been
excluded by measurement:

    real            excess concentration RISES with a 4x window, R = 1.690;
                    one lane last on a median 72% of 3840 ops (chance 0.067)
    costly          straggler wait is ~0.31 of the width-16 window
    not assignment  `ops_spread` = 0.0000 in 24/24 launches
    not placement   one lane->cpu map over 24 launches, victim moves anyway
    not layout      `setarch -R` concentration 0.267 == ASLR's 0.267
    not page size   THP off leaves `work_skew` at 1.023x

Five exclusions and no candidate. But a cause is not required to act, and the
engineering statement does not depend on one: **the static even split is
optimal only if every lane runs at the same speed, and measurement says one
lane out of fifteen runs ~55% slower for the whole life of a process.** A
barrier then makes all fifteen wait for it.

`DecodeSchedule::Steal` already exists in `decode_spmd.rs`, reachable through
`ONNX_GENAI_CPU_DECODE_SCHEDULE=steal`, and it is **not the default** --
`decode_schedule_from_raw` falls through to `Fixed`. A dynamic claim is exactly
the structure that absorbs a persistently slow lane without knowing why it is
slow. So the question is whether turning it on recovers the straggler wait, and
that is a default-selection question, which is mine.

WHY THIS IS NOT THE PROFILED INSTRUMENT
---------------------------------------
Every result quoted above came from the worker-split instrument with
`ONNX_GENAI_CPU_DECODE_WORKER_PROFILE=1`, which is diagnostic-only: two clock
reads per worker per op, and it is already known to dissolve the width-16
bimodality. It also cannot score this arm on its own terms -- under stealing
`ops_spread` is non-zero *by design*, so `work_skew` stops being comparable
between arms.

This probe therefore measures the production path with profiling **off** and
gates on **`ms_token`**, the quantity a user experiences. `H.native()` is the
same steady-state reader used by the rest of the acc0 work, including its
non-vacuity check that the realized decode width matches the requested one.

PRE-REGISTERED RULE (written before the first launch)
-----------------------------------------------------
Three arms, interleaved, rotating order, `--launches` each:

    fixed   default (`DecodeSchedule::Fixed`)
    steal1  `ONNX_GENAI_CPU_DECODE_SCHEDULE=steal`, default tile granularity
            (`STEAL_TILES_PER_WORKER=1`)
    steal4  the same, with `ONNX_GENAI_CPU_DECODE_STEAL_TILES_PER_WORKER=4`
    null    identical to `fixed` in every respect, carried under a different
            label

RULE AMENDMENT (2026-08-24, before any data was taken under this rule)
----------------------------------------------------------------------
The original rule had a single `steal` arm at default granularity. A smoke run
found `ONNX_GENAI_CPU_DECODE_SCHEDULE=steal` to be **entirely inert** in a
default build -- the parse arms selecting it were `#[cfg(feature = "mlas")]`
while the claim path they select is pure-native -- so that run compared three
identical arms and produced no usable data. The selector is now un-gated.

While fixing it, reading `work_stealing_segments_aligned` showed the default
`STEAL_TILES_PER_WORKER = 1` yields exactly `total_workers` tiles, and the
function then returns `worker_row_segments_aligned` -- *the same segments the
fixed split uses*. With one tile per worker a dynamic claim can absorb a lane
that **wakes late** (an awake worker takes the absent one's tile) but cannot
absorb a lane that **executes slowly**, because that lane still holds one whole
tile to the end. My own measurements say this straggler usually computes longer
(`straggler_idx == slowest_idx` at 0.667/0.667/0.684 across three experiments,
chance 0.067), so the default granularity is predicted *a priori* not to help.
`steal4` is therefore the arm carrying the hypothesis, and `steal1` is retained
as the granularity control that discriminates the two mechanisms.

RULE AMENDMENT 2 (2026-08-24, after the first 8x4 run, before any claim)
------------------------------------------------------------------------
That run came back with every arm split into two well-separated clusters
(~1.5-2.0 ms and ~2.8-3.7 ms, a 2.3x ratio with an empty band between them).
A median over a bimodal mixture is an estimator of **how many launches landed
in each mode**, not of the effect, and with 8 launches the mode count moves by
one sample and drags the median across the gap. The A/A null passed at 0.8%
purely because `fixed` and `null` happened to place their medians on the same
side; that is not evidence the harness resolved anything.

So the pooled verdict is void whenever the pooled sample is bimodal, and the
report stratifies: split at the widest internal gap, then run the same null
test and the same accept/reject bar *within each mode*. This can only ever
weaken a claim -- it never manufactures one -- and it is applied to the data
that motivated it, which is stated here rather than hidden.

Let `M(arm)` be that arm's median `ms_token`.

    ACCEPT (steal should become the default at this width) iff
        M(steal) <= (1 - MIN_GAIN) * M(fixed)     with MIN_GAIN = 0.05
REL_GAP = 0.25      # cluster split: a gap wider than this fraction of the median
MIN_MODE_N = 3      # a cluster smaller than this is not a mode
        AND the observed null asymmetry |M(null)/M(fixed) - 1| < MIN_GAIN / 2
    REJECT iff
        M(steal) >= M(fixed)
    otherwise REPORT NOTHING.

The null arm is not decoration. An earlier width-16 study on this host measured
an **A/A null of +/-21.5%**, which is larger than any effect worth shipping;
without carrying the null in the same window there is no way to know whether
that is still true, and a 5% gain read against a 21% null is noise. If the null
asymmetry exceeds MIN_GAIN/2 the run reports the null and nothing else --
including when the arithmetic would otherwise have said ACCEPT.

CONTROLS
--------
1.  **The knob is verified, not trusted.** `ONNX_GENAI_CPU_DECODE_AFFINITY`
    (#1792) on this project is a user-facing control that is entirely inert, so
    before measuring, this probe checks that the `steal` arm actually reports a
    stealing pool. `decode_spmd.rs` prints `path=work-stealing-pool` when
    `PersistenceMode::On` meets `DecodeSchedule::Steal`, so the arm's own
    `decode_width` line carries the evidence. A `steal` arm whose launches all
    report the fixed path is a non-manipulation and aborts the run.
2.  **Width is non-vacuous.** `H.native()` already rejects any row whose
    realized width does not match the requested one; that check is kept, not
    bypassed.
3.  Arms are interleaved launch-by-launch with a rotating order, so host drift
    cannot be absorbed by one arm.
4.  Distributions are printed in full, not just medians, so a reader can see
    overlap directly rather than inferring it from a point estimate.

REJECT is a perfectly good outcome and is the one to expect if the straggler's
cost is dominated by something a dynamic claim cannot recover -- for instance
if the slow lane is slow in a way that stealing merely relocates.
"""

from __future__ import annotations

import argparse
import math
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import acc0_gap_matrix as H  # noqa: E402

MIN_GAIN = 0.05
REL_GAP = 0.25      # cluster split: a gap wider than this fraction of the median
MIN_MODE_N = 3      # a cluster smaller than this is not a mode
ARMS = ("fixed", "steal1", "steal4", "null")
STEAL_ARMS = ("steal1", "steal4")
ENV = {
    "fixed": {},
    "steal1": {"ONNX_GENAI_CPU_DECODE_SCHEDULE": "steal"},
    "steal4": {
        "ONNX_GENAI_CPU_DECODE_SCHEDULE": "steal",
        "ONNX_GENAI_CPU_DECODE_STEAL_TILES_PER_WORKER": "4",
    },
    "null": {},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--launches", type=int, default=8, help="per arm")
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--model", default="llama")
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--acc", default="4")
    ap.add_argument("--sessions", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=192)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default="bb/w16_steal.json")
    ap.add_argument("--replay", default=None)
    args = ap.parse_args()

    if args.replay:
        return report(json.loads(pathlib.Path(args.replay).read_text()), args.launches)

    records = []
    for i in range(args.launches):
        order = ARMS[i % len(ARMS):] + ARMS[: i % len(ARMS)]
        for arm in order:
            extra = dict(ENV[arm])
            try:
                steady = H.native(args.binary, args.model, args.block, args.acc,
                                  args.width, args.sessions, args.tokens, args.reps,
                                  extra or None)
            except RuntimeError as e:
                print(f"L{i:<2} {arm:<5} FAILED: {e}", flush=True)
                records.append({"launch": i, "arm": arm, "steady": None})
                continue
            records.append({"launch": i, "arm": arm, "steady": steady})
            pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(args.out).write_text(json.dumps(records, indent=1))
            print(f"L{i:<2} {arm:<5} ms_token={steady['ms_token']:.4f} "
                  f"tps={steady['tps']:.2f} spread={steady['spread']:.3f} "
                  f"| {steady['decode_width']}", flush=True)
            time.sleep(1)

    return report(records, args.launches)




def clusters(vals, rel_gap):
    """Split a sorted sample wherever consecutive values differ by more than
    `rel_gap` of the sample median. Returns a list of lists, low to high."""
    vals = sorted(vals)
    if not vals:
        return []
    thresh = rel_gap * statistics.median(vals)
    out, cur = [], [vals[0]]
    for prev, cur_v in zip(vals, vals[1:]):
        if cur_v - prev > thresh:
            out.append(cur)
            cur = []
        cur.append(cur_v)
    out.append(cur)
    return out


def modes_of(vals):
    """Classify one arm's launches into (fast, slow, excluded).

    The mode is a property of a *process*, so each arm is clustered on its own
    sample -- the whole point of the experiment is that an arm may move where
    its slow mode sits. Requires a wide, well-populated split before it will
    call an arm bimodal, so an arm that is genuinely unimodal is not carved up
    into invented modes.
    """
    cs = clusters(vals, REL_GAP)
    big = [c for c in cs if len(c) >= MIN_MODE_N]
    excluded = [v for c in cs if len(c) < MIN_MODE_N for v in c]
    if len(big) < 2:
        return None, None, excluded
    # Two largest by population, then ordered by value.
    big = sorted(sorted(big, key=len, reverse=True)[:2], key=lambda c: c[0])
    excluded += [v for c in cs if len(c) >= MIN_MODE_N and c not in big for v in c]
    return big[0], big[1], excluded


def report(records, attempted) -> int:
    by = {a: [r["steady"] for r in records if r["arm"] == a and r["steady"]] for a in ARMS}
    print()
    for a in ARMS:
        vals = sorted(s["ms_token"] for s in by[a])
        print(f"{a:<7} n={len(vals)} ms_token={[f'{v:.4f}' for v in vals]}")
    if any(len(by[a]) < 3 for a in ARMS):
        print("VERDICT: REPORT NOTHING -- an arm produced fewer than 3 usable launches")
        return 0

    # CONTROL 1: every steal arm must actually be a stealing pool, and the
    # fixed/null arms must not be. An inert knob makes all four arms the same
    # experiment wearing four labels -- which is exactly what the first smoke
    # run of this probe turned out to be.
    for arm in STEAL_ARMS:
        paths = {s["decode_width"] for s in by[arm]}
        print(f"CONTROL 1 {arm} width lines: {paths}")
        if not [p for p in paths if "work-stealing" in p]:
            print(f"CONTROL 1 FIRED: no launch in {arm} reported a work-stealing pool.")
            print("VERDICT: ABORT -- the knob did not take effect; nothing is compared.")
            return 0
    for arm in ("fixed", "null"):
        paths = {s["decode_width"] for s in by[arm]}
        if [p for p in paths if "work-stealing" in p]:
            print(f"CONTROL 1 FIRED: {arm} reported a work-stealing pool: {paths}")
            print("VERDICT: ABORT -- the baseline is not the fixed split.")
            return 0

    raw = {a: [s["ms_token"] for s in by[a]] for a in ARMS}
    m = {a: statistics.median(v) for a, v in raw.items()}
    print()
    for a in ARMS:
        print(f"{a:<7} pooled median ms_token = {m[a]:.4f}")

    split = {a: modes_of(raw[a]) for a in ARMS}
    bimodal = {a: split[a][0] is not None for a in ARMS}
    print(f"\nper-arm bimodality: "
          + ", ".join(f"{a}={bimodal[a]}" for a in ARMS))

    if not all(bimodal.values()):
        null_asym = abs(m["null"] / m["fixed"] - 1.0)
        print(f"A/A null asymmetry (pooled) = {null_asym:.4f} "
              f"(must be < {MIN_GAIN / 2:.4f})")
        if null_asym >= MIN_GAIN / 2:
            print("VERDICT: REPORT THE NULL ONLY -- the harness cannot resolve an effect")
            print("         this size in this window. No claim about `steal` is licensed.")
            return 0
        best = min(STEAL_ARMS, key=lambda a: m[a])
        gain = 1.0 - m[best] / m["fixed"]
        print(f"best steal arm = {best}  gain = {gain * 100:+.2f}%")
        if m[best] <= (1 - MIN_GAIN) * m["fixed"]:
            print(f"VERDICT: ACCEPT ({best})")
        elif all(m[a] >= m["fixed"] for a in STEAL_ARMS):
            print("VERDICT: REJECT -- no steal granularity beats the fixed split.")
        else:
            print("VERDICT: REPORT NOTHING -- below the bar.")
        return 0

    # ---- every arm is bimodal: the pooled median is void (RULE AMENDMENT 2) ----
    print("\nEvery arm is bimodal, so a pooled median estimates the MODE FRACTION and")
    print("not the effect. The pooled verdict is void; reporting per mode.\n")
    print(f"{'arm':<7}{'n_fast':>7}{'med_fast':>10}{'n_slow':>7}{'med_slow':>10}"
          f"{'frac_fast':>11}{'excl':>6}")
    stat = {}
    for a in ARMS:
        fast, slow, excl = split[a]
        n = len(fast) + len(slow)
        stat[a] = {"fast": fast, "slow": slow, "frac": len(fast) / n, "excl": excl}
        print(f"{a:<7}{len(fast):>7}{statistics.median(fast):>10.4f}"
              f"{len(slow):>7}{statistics.median(slow):>10.4f}"
              f"{len(fast) / n:>11.3f}{len(excl):>6}")
    allexcl = sorted(v for a in ARMS for v in stat[a]["excl"])
    if allexcl:
        print(f"\nexcluded (clusters below n={MIN_MODE_N}, reported not hidden): "
              f"{[f'{v:.3f}' for v in allexcl]}")

    verdicts = {}
    for mode in ("fast", "slow"):
        base, nul = stat["fixed"][mode], stat["null"][mode]
        asym = abs(statistics.median(nul) / statistics.median(base) - 1.0)
        print(f"\n{mode} mode: fixed={statistics.median(base):.4f} "
              f"null={statistics.median(nul):.4f}  A/A asymmetry={asym:.4f}")
        if asym >= MIN_GAIN / 2:
            print(f"  REPORT THE NULL ONLY in the {mode} mode -- unresolvable here.")
            continue
        for a in STEAL_ARMS:
            g = 1.0 - statistics.median(stat[a][mode]) / statistics.median(base)
            v = "ACCEPT" if g >= MIN_GAIN else "REJECT" if g <= 0 else "NOTHING"
            verdicts[(a, mode)] = (g, v)
            print(f"  {a}: {g * 100:+.2f}%  -> {v}")

    # The whole verdict turns on the mode fraction, so test it against its own
    # A/A null rather than eyeballing it: `fixed` and `null` are the same
    # configuration, so their fraction difference is this harness's noise floor
    # for a fraction, and a steal arm has to clear that to have moved anything.
    print("\nmode-fraction test (the quantity the verdict turns on):")
    f_fixed, f_null = stat["fixed"]["frac"], stat["null"]["frac"]
    aa = abs(f_fixed - f_null)
    print(f"  A/A floor |fixed - null| = {aa:.3f}")
    for a in STEAL_ARMS:
        d = abs(stat[a]["frac"] - f_fixed)
        n1 = len(stat["fixed"]["fast"]) + len(stat["fixed"]["slow"])
        n2 = len(stat[a]["fast"]) + len(stat[a]["slow"])
        pool = (len(stat["fixed"]["fast"]) + len(stat[a]["fast"])) / (n1 + n2)
        se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2)) if 0 < pool < 1 else 0.0
        z = d / se if se else 0.0
        print(f"  {a}: frac={stat[a]['frac']:.3f} vs fixed={f_fixed:.3f} "
              f"delta={d:.3f} z={z:.2f} {'RESOLVED' if z >= 2.0 and d > aa else 'UNRESOLVED'}")

    # A per-mode win is not a user-visible win if the arm also changes how often
    # each mode is reached. Combine them into the quantity a user experiences.
    print("\nmode-weighted expectation (per-arm fractions, per-arm mode medians):")
    ev = {}
    for a in ARMS:
        s = stat[a]
        ev[a] = s["frac"] * statistics.median(s["fast"]) + \
            (1 - s["frac"]) * statistics.median(s["slow"])
        print(f"  {a:<7} E[ms_token] = {ev[a]:.4f}")
    null_ev_asym = abs(ev["null"] / ev["fixed"] - 1.0)
    print(f"  A/A null on the expectation = {null_ev_asym:.4f}")

    print()
    if null_ev_asym >= MIN_GAIN / 2:
        print("VERDICT: STRATIFIED, NO NET CLAIM -- per-mode lines above stand, but the")
        print("         A/A null on the mode-weighted expectation is not resolvable, so")
        print("         no statement about end-to-end latency is licensed.")
        return 0
    best = min(STEAL_ARMS, key=lambda a: ev[a])
    net = 1.0 - ev[best] / ev["fixed"]
    print(f"net (best={best}): {net * 100:+.2f}%")
    if net >= MIN_GAIN:
        print(f"VERDICT: ACCEPT ({best}) -- wins per mode AND in expectation.")
    elif any(v == "ACCEPT" for (a, mo), (g, v) in verdicts.items()):
        print("VERDICT: SPLIT RESULT -- work-stealing wins decisively inside a mode but")
        print("         does not move end-to-end latency, because it also changes how")
        print("         often each mode is reached. Not a default change on this evidence.")
    else:
        print("VERDICT: REJECT -- no mode and no expectation favours work-stealing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
