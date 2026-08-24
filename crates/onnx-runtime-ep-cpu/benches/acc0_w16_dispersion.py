#!/usr/bin/env python3
"""Score an A/B run for *dispersion*, not for its median.

Written and committed BEFORE the run it scores. The rule below is the whole
point of the file; reading it after seeing the numbers and adjusting it would
make it worthless.

Why a separate script
---------------------
`acc0_w16_blocktime_ab.py` is a validated instrument -- its thresholds, A/A
arm, arm rotation and width-8 regression guard have been replayed against
archived JSON and reproduce published numbers exactly. A dispersion claim needs
a different statistic, so it gets a different file rather than another edit to
that one. This script *runs nothing*: it reads the JSON the A/B already wrote,
so the two claims come from the same launches and cannot disagree about which
host state they describe.

The question
------------
Width-16 A/B work on this host is blocked by its own noise floor. Two
*identical* arms measured in the same launch differ by a median of ~11-22% and
by as much as 58%, which is large enough that no width-16 intervention of
realistic size can clear a pre-registered bar -- a +23% steal-tiles candidate
was rejected on exactly this ground, with the A/A half-width at 21.5% in the
same run.

That dispersion is not symmetric noise. Pooling 90 archived arms:

* At width 8, arms slower than 1.3x the best are 5/45, sit at execution
  positions 1 and 2 (never 3), and every one of them has an intra-arm rep
  spread of >=23% against a 6.9% median. They are self-detectable.
* At width 16 they are 15/45 -- three times as common -- spread evenly over
  all three positions, and NOT self-detectable: their median intra-arm spread
  is 22.8% against 18.9% for the normal arms, and one slow arm was internally
  consistent to 0.5% while running 1.3x slow for every one of its reps.

An arm that is uniformly slow for its whole life, with a tight internal spread,
is a per-process state, not a disturbance. The hypothesis this scores is that
the state is thread placement, and specifically the placement of the *inline
dispatcher*: the pool reserves a CPU for it (`DISPATCHER_RESERVED_CPUS`,
justified by a measured 1.57x) but nothing binds it there, so where it settles
is decided per launch by the scheduler.

The rule (pre-registered)
-------------------------
Per arm, over trusted launches, using each launch's `tps_rep` at
`PRIMARY_WIDTH`:

    D(arm) = (p90(tps) - p10(tps)) / median(tps)

`p90 - p10` rather than max-min so a single launch cannot define the result,
and normalised by the median so the two arms are comparable in scale.

    PIN-STABILISES   iff  D(test) <= DISPERSION_RATIO * D(control)
                     and  n_trusted >= MIN_TRUSTED

Self-test, evaluated FIRST and able to veto:

    The A/A arm is the same configuration as its own reference arm, so its
    dispersion estimates the same quantity. If

        |D(aa) - D(ref)| > SELFTEST_TOLERANCE * mean(D(aa), D(ref))

    the estimator is too noisy at this n to support any dispersion claim, and
    the verdict is REPORT NOTHING no matter what the arms did.

    `ref` is the arm the A/A repeats: `aa_arm` names it per launch, and the
    harness rotates it, so this is resolved per launch rather than assumed.

Note on what the A/A arm is here. In `acc0_w16_blocktime_ab.py` the A/A is a
*second run of one arm within the same launch*, so it measures within-launch
repeatability. That is the right null for the median rule. For dispersion it is
also the right self-test, because a dispersion estimator that cannot reproduce
itself across two runs of one configuration cannot distinguish two others.

This is a dispersion rule only. It says nothing about which arm is faster; the
median rule in the A/B script answers that, and the two are reported side by
side deliberately, because "faster" and "more reproducible" are different
claims and an intervention can win one and lose the other.

What it returned
----------------
On the dispatcher-pin run this was written for (`7e274a4e2`, 16 launches, 15
trusted) the **self-test vetoed the run**: |D(aa) - D(ref)| = 0.1432 against an
allowed 0.5 x 0.2591 = 0.1296. Two arms of identical configuration disagreed
about dispersion by more than the estimator's own tolerance, so no dispersion
claim was made -- D(control) = 0.3610 and D(test) = 0.0780 are unscored
observations and are not a result. An earlier 6-launch run scored
PIN-STABILISES (0.2781 -> 0.0416); it did not survive the larger n. Recorded
here because a rule that has only ever fired positively is a rule nobody has
tested. See `docs/benchmarks/2026-08-24-acc0-dispatcher-placement.md`.
"""

import argparse
import json
import statistics
import sys

PRIMARY_WIDTH = 16
MIN_TRUSTED = 6
DISPERSION_RATIO = 0.5
SELFTEST_TOLERANCE = 0.5


def pctl(values, q):
    """Linear-interpolated quantile; `statistics.quantiles` needs n >= 2."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def dispersion(values):
    if len(values) < 2:
        return float("nan")
    median = statistics.median(values)
    if median <= 0:
        return float("nan")
    return (pctl(values, 0.90) - pctl(values, 0.10)) / median


def arm_series(cells, arm):
    """Per-launch `tps_rep` for one arm at the primary width."""
    out = []
    for cell in cells:
        width = cell["widths"].get(str(PRIMARY_WIDTH))
        if not width:
            continue
        entry = width.get(arm)
        if not entry or "cpu" not in entry or "tps_rep" not in entry["cpu"]:
            continue
        out.append(entry["cpu"]["tps_rep"])
    return out


def aa_reference_series(cells, control_value):
    """The arm each launch's A/A repeats, resolved per launch via `aa_arm`.

    The harness rotates which arm is doubled, so a fixed choice here would
    compare the A/A against the *other* configuration in half the launches --
    which would not be a self-test at all.
    """
    out = []
    for cell in cells:
        width = cell["widths"].get(str(PRIMARY_WIDTH))
        if not width:
            continue
        arm = "control" if width.get("aa_arm") == control_value else "test"
        entry = width.get(arm)
        if not entry or "cpu" not in entry or "tps_rep" not in entry["cpu"]:
            continue
        out.append(entry["cpu"]["tps_rep"])
    return out


def pin_state(cells):
    """Every distinct dispatcher verdict seen, per arm.

    Non-vacuity, checked rather than assumed: an intervention arm whose pin did
    not take is not a test arm, and scoring it as one would silently report a
    null result as a negative one. `PIN-MISSED` or a missing row anywhere in
    the test arm invalidates the run.
    """
    seen = {}
    for cell in cells:
        width = cell["widths"].get(str(PRIMARY_WIDTH))
        if not width:
            continue
        for arm in ("control", "test", "aa"):
            entry = width.get(arm)
            if not entry:
                continue
            seen.setdefault(arm, set()).add(entry.get("dispatcher", "MISSING"))
    return {arm: sorted(v) for arm, v in seen.items()}


def report(cells, control_value):
    lines = []
    trusted = [c for c in cells if c.get("trusted")]
    n = len(trusted)
    lines.append(f"n_trusted={n} of {len(cells)} launches")
    if n < MIN_TRUSTED:
        lines.append(f"REPORT NOTHING (n_trusted={n} < {MIN_TRUSTED})")
        return lines

    control = arm_series(trusted, "control")
    test = arm_series(trusted, "test")
    aa = arm_series(trusted, "aa")
    ref = aa_reference_series(trusted, control_value)

    d_control, d_test, d_aa, d_ref = (
        dispersion(control), dispersion(test), dispersion(aa), dispersion(ref))
    lines.append(f"D(control)={d_control:.4f}  D(test)={d_test:.4f}")
    lines.append(f"self-test: D(aa)={d_aa:.4f}  D(aa's own arm)={d_ref:.4f}")

    mean_pair = (d_aa + d_ref) / 2.0
    gap = abs(d_aa - d_ref)
    if mean_pair <= 0 or gap > SELFTEST_TOLERANCE * mean_pair:
        lines.append(
            f"SELF-TEST FAILED: |D(aa)-D(ref)|={gap:.4f} > "
            f"{SELFTEST_TOLERANCE} x {mean_pair:.4f}")
        lines.append("REPORT NOTHING (the dispersion estimator cannot "
                     "reproduce itself at this n)")
        return lines
    lines.append(f"self-test passed: |D(aa)-D(ref)|={gap:.4f} <= "
                 f"{SELFTEST_TOLERANCE} x {mean_pair:.4f}")

    states = pin_state(trusted)
    lines.append(f"dispatcher verdicts: {states}")
    bad = [s for s in states.get("test", []) if "PIN-TOOK" not in s]
    if bad:
        lines.append(f"VACUOUS: test arm did not pin ({bad})")
        return lines

    lines.append(f"medians: control={statistics.median(control):.2f} tps  "
                 f"test={statistics.median(test):.2f} tps  "
                 f"ratio={statistics.median(test) / statistics.median(control):.4f}")
    if d_test <= DISPERSION_RATIO * d_control:
        lines.append(f"PIN-STABILISES: D(test)={d_test:.4f} <= "
                     f"{DISPERSION_RATIO} x D(control)={d_control:.4f}")
    else:
        lines.append(f"NOT STABILISED: D(test)={d_test:.4f} > "
                     f"{DISPERSION_RATIO} x D(control)={d_control:.4f}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", help="output of acc0_w16_blocktime_ab.py")
    ap.add_argument("--control-value", default="0",
                    help="the --control value the A/B ran with; used to "
                         "resolve which arm each launch's A/A repeats")
    args = ap.parse_args()
    cells = json.load(open(args.json))
    for line in report(cells, args.control_value):
        print(line)


if __name__ == "__main__":
    sys.exit(main())
