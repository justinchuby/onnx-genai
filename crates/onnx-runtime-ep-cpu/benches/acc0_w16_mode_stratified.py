#!/usr/bin/env python3
"""Re-score a width-16 A/B after removing the bimodal null, without touching the rule.

Why this exists
---------------
`docs/benchmarks/2026-08-23-acc0-width-16-worker-attribution.md` recorded a
REJECT for `ONNX_GENAI_CPU_DECODE_STEAL_TILES_PER_WORKER=2`: ratio 1.2327 at 88%
sign consistency, refused because the rule requires the effect to be at least 3x
the A/A half-width and the A/A half-width in that run was 0.2154. The verdict was
honoured and remains honoured.

`docs/benchmarks/2026-08-24-acc0-w16-null-page-backing.md` then established what
that 0.2154 *is*. The width-16 null is not a spread, it is **two modes**: every
process launch draws either ~3.5 ms/token at 15-16 effective lanes or ~5.9
ms/token at 10-12, the two clusters do not overlap, and the mode ratio is
**1.687x**. The A/B harness launches each arm as its own process, so each arm
draws independently and a paired ratio is a mixture of 1.0, 1.69 and 1/1.69.
The null is not measurement noise about the arms -- it is a *third* variable
being resampled between them.

The rejected mechanism list matters here, because it is what makes the mode a
nuisance variable rather than the thing under test: worker placement (categorical
`/proc` census), transparent-hugepage backing (pre-registered, rho -0.19) and
foreign load on the pinned CPUs (bounded at 1.7% of pinned CPU time, 13x too
small) are all excluded. What is left is a persistent straggler inside the
process: same user work, +170% `sys`, decided before the first token.

What this file does and does not do
-----------------------------------
It does **not** edit the rule. It imports `verdict()` from the A/B harness and
calls it, so the thresholds (n>=6, ratio>=1.10, sign>=80%, effect>=3x A/A
half-width), the mechanism claim and the width-8 regression guard are the same
code that produced the original REJECT. The only thing this file changes is
*which launches are fed to it*, and it prints the unmodified verdict on the full
set next to the stratified one so the two can never be confused.

Pre-registered before the first re-score
----------------------------------------
    A width-16 sub-launch is FAST iff effective lanes
    (`cpu_s_per_token / ms_token`) >= 13.0.

    The 13.0 cut is **not fitted here**. It is the midpoint of the 3.1-lane gap
    published from A/A data alone (mode A 15.30-16.10, mode B 9.76-12.16),
    fixed before any A/B launch was re-scored.

    A launch is STRATIFIED-TRUSTED iff the harness already marked it trusted AND
    all three of its width-16 sub-launches (control, test, A/A) are FAST.
    Requiring all three keeps the comparison paired: dropping one arm of a pair
    would bias the ratio by construction.

    GATE -- the stratification must not be able to see the arm:
      Slow-sub-launch rate is computed per *configuration*, following the
      harness's arm rotation (`flip = launch % 2 == 1` puts the test value in
      the A/A slot on odd launches). If the two configurations' slow rates
      differ by more than 2x, this file REPORTS NOTHING about the A/B and
      reports the differential instead -- because a candidate that suppresses
      the slow mode is a *result*, not a filter, and using it as a filter would
      launder it into the ratio.

    PAIRED MODE IMBALANCE -- reported always, and the reason this file exists:
      The gate above is the right check for whether *filtering* is biased, and
      it is the wrong check for whether the *unfiltered* ratio is. The ratio is
      formed from the control and test sub-launches specifically, so what
      manufactures a spurious effect is an imbalance between those two, not
      between the pooled configurations. On the original steal-tiles run the
      pooled rates were 33% vs 25% (gate PASS, 1.33x) while control drew 4 slow
      launches of 8 against test's 1 -- and since a slow control deflates the
      denominator by the 1.687x mode ratio, that alone produces a median above
      1.0 with high sign consistency out of nothing. This diagnostic is printed
      unconditionally because a run whose control and test arms drew the slow
      mode at different rates cannot support *any* claim about the ratio,
      accept or reject.

    Only if the gate passes is the stratified verdict printed, and it is printed
    as "the same rule on the same data with the mode removed", never as a new
    rule.

What a stratified ACCEPT would and would not license
----------------------------------------------------
It would license "at width 16, in the fast mode, steal=2 beats steal=1 by the
measured ratio on the zero-gap decode loop". It would **not** license a default
change: the shipped default has to hold in *both* modes and on a gapped
generation loop, and this file deliberately discards one of the two modes. A
default change needs the mode understood, not filtered.
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import acc0_w16_blocktime_ab as AB  # noqa: E402

LANE_CUT = 13.0
MAX_RATE_RATIO = 2.0
# Published mode ratio from the A/A record; used only to state how much
# ratio a given paired imbalance can manufacture, never to correct anything.
MODE_RATIO = 1.687
MAX_IMBALANCE = 0.15
W = str(AB.PRIMARY_WIDTH)


def lanes(sub):
    """Effective lanes: CPU-seconds burned per wall-second of decode."""
    cpu = sub.get("cpu")
    if not cpu or "cpu_s_per_token" not in cpu or not sub.get("ms_token"):
        return None
    return cpu["cpu_s_per_token"] / (sub["ms_token"] / 1000.0)


def user_per_token(sub):
    cpu = sub["cpu"]
    return cpu["cpu_s_per_token"] * (1.0 - cpu["sys_frac"])


def config_of(rec, name, control, test):
    """Which knob value a sub-launch ran, following the harness's rotation."""
    if name == "control":
        return control
    if name == "test":
        return test
    return test if rec.get("flipped") else control


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="output of the A/B harness")
    ap.add_argument("--control", default="1", help="control knob value, label only")
    ap.add_argument("--test", default="2", help="test knob value, label only")
    args = ap.parse_args()

    with open(args.json) as fh:
        cells = json.load(fh)
    trusted = [c for c in cells if c.get("trusted") and W in c.get("widths", {})]

    print(f"# lane cut {LANE_CUT} (published from A/A data, not fitted here); "
          f"rule imported from {AB.__name__}, thresholds untouched")
    print()
    print("UNMODIFIED RULE, ALL TRUSTED LAUNCHES")
    for line in AB.verdict(trusted):
        print(f"  {line}")

    print()
    print(f"{'L':>3} {'cfg':>4} {'arm':>8} {'ms':>7} {'lanes':>6} "
          f"{'sysf':>6} {'user/tok':>9} {'mode':>5}")
    per_cfg = {}
    fast_recs = []
    for rec in trusted:
        w = rec["widths"][W]
        all_fast = True
        for name in ("control", "test", "aa"):
            sub = w[name]
            ln = lanes(sub)
            cfg = config_of(rec, name, args.control, args.test)
            fast = ln is not None and ln >= LANE_CUT
            all_fast = all_fast and fast
            tot, slow = per_cfg.get(cfg, (0, 0))
            per_cfg[cfg] = (tot + 1, slow + (0 if fast else 1))
            print(f"{rec['launch']:>3} {cfg:>4} {name:>8} "
                  f"{sub['ms_token']:>7.3f} "
                  f"{('n/a' if ln is None else f'{ln:6.2f}')} "
                  f"{sub['cpu']['sys_frac']:>6.3f} "
                  f"{user_per_token(sub):>9.5f} "
                  f"{'fast' if fast else 'SLOW':>5}")
        if all_fast:
            fast_recs.append(rec)

    print()
    print("PAIRED MODE IMBALANCE (control vs test, the arms the ratio is "
          "formed from)")
    n = len(trusted)
    per_arm = {}
    for name in ("control", "test", "aa"):
        slow = sum(1 for rec in trusted
                   if (lanes(rec["widths"][W][name]) or 0.0) < LANE_CUT)
        per_arm[name] = slow
        print(f"  {name:>8}: slow {slow}/{n} = {(slow / n if n else 0):.1%}")
    imbalance = abs(per_arm["control"] - per_arm["test"]) / n if n else 0.0
    # A launch whose control is slow and test is fast contributes ~MODE_RATIO to
    # the ratio instead of ~1.0. The net median pull is therefore bounded by the
    # net imbalance fraction times (MODE_RATIO - 1).
    pull = imbalance * (MODE_RATIO - 1.0)
    print(f"  net imbalance {imbalance:.1%} of launches; at the published "
          f"{MODE_RATIO:.3f}x mode ratio this can manufacture up to "
          f"{pull:+.4f} of ratio on its own")
    print(f"  {'USABLE' if imbalance <= MAX_IMBALANCE else 'UNUSABLE'} "
          f"-- bar is {MAX_IMBALANCE:.0%}")

    print()
    print("SLOW-MODE RATE BY CONFIGURATION (the arm-selectivity gate)")
    rates = {}
    for cfg, (tot, slow) in sorted(per_cfg.items()):
        rates[cfg] = slow / tot if tot else 0.0
        print(f"  {cfg:>4}: {slow}/{tot} = {rates[cfg]:.1%}")
    vals = [v for v in rates.values()]
    lo, hi = (min(vals), max(vals)) if vals else (0.0, 0.0)
    rate_ratio = (hi / lo) if lo > 0 else (float("inf") if hi > 0 else 1.0)
    gate_ok = rate_ratio <= MAX_RATE_RATIO
    print(f"  rate ratio {rate_ratio:.2f} against a {MAX_RATE_RATIO:.0f}x bar: "
          f"{'PASS' if gate_ok else 'FAIL'}")

    # User CPU per token, across every sub-launch and both modes. This is the
    # control for a load-balance candidate: it must NOT move, because stealing
    # redistributes work rather than removing it. A candidate that also moved
    # this changed something else.
    ups = {}
    for rec in trusted:
        w = rec["widths"][W]
        for name in ("control", "test", "aa"):
            cfg = config_of(rec, name, args.control, args.test)
            ups.setdefault(cfg, []).append(user_per_token(w[name]))
    print()
    print("USER CPU PER TOKEN (control: a load-balance change must not move it)")
    for cfg, v in sorted(ups.items()):
        print(f"  {cfg:>4}: median {statistics.median(v):.5f}  "
              f"range {min(v):.5f} - {max(v):.5f}  n={len(v)}")
    allu = [x for v in ups.values() for x in v]
    print(f"  all : spread {max(allu) / min(allu):.3f}x across both "
          f"configurations and both modes")

    print()
    if not gate_ok:
        print("STRATIFIED VERDICT: REPORT NOTHING -- the slow-mode rate differs "
              "by more than the gate allows, so filtering on it would launder "
              "an arm effect into the ratio. The differential above IS the "
              "finding and should be measured directly.")
        return 0
    print(f"SAME RULE, FAST MODE ONLY ({len(fast_recs)} of {len(trusted)} "
          f"launches have all three width-16 arms in the fast mode)")
    for line in AB.verdict(fast_recs):
        print(f"  {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
