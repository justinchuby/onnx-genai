#!/usr/bin/env python3
"""acc0 gap at width 16 -- the one cell the 2026-08-23 matrix could not resolve.

Why this exists
---------------
`docs/benchmarks/2026-08-23-acc0-gap-vs-ort-by-width.md` resolved the acc0 int4
decode gap at t=1 (1.120x) and t=8 (1.120x) and explicitly could NOT resolve
t=16.  Two of three cells there passed the load guard and read 1.831 and 1.456
(median 1.643x), but the width's A/A null -- two *identical* native arms in the
same launch -- spanned 0.969 to 1.295.  A 1.64x effect measured on an instrument
with a +-30% null is not a result.

t=16 matters more than the widths that did resolve: it is the closest cell in
the matrix to an unconfined production process, and it is the only width
pointing at a large gap.  If 1.64x is real, acc0 goes back to the top of the CPU
MatMulNBits work list and the merged re-ranking is wrong.

The pre-registered acceptance rule
----------------------------------
Written here BEFORE the script was first run, because the failure this study
exists to avoid is choosing an acceptance threshold after seeing which
threshold would license the conclusion.

Let `aa_hw` be the A/A half-width over trusted cells, defined as
`max(|aa - 1|)` -- the worst deviation from unity of two identical native arms,
not a percentile, because a percentile lets an inconvenient cell be excluded by
a choice made later.

    ACCEPT  a point estimate for the gap iff
      (1) n_trusted >= 8, and
      (2) aa_hw <= 0.10, and
      (3) |gap_median - 1| >= 3 * aa_hw.

    REPORT AS A RANGE, never as a point, if (1) holds but (2) or (3) fails.

    REPORT NOTHING if (1) fails.

Condition (3) is the one that bites.  At the merged run's `aa_hw = 0.295` a gap
would have needed to exceed 1.885x to clear it -- so 1.643x correctly failed,
and would still fail even if every other cell agreed with it.

Two further rules, also pre-registered:

  * Both arms' intra-run spread is recorded for every cell and published.  The
    merged run's t=16 ORT arm spread 55.4% and 19.6%; a denominator that
    unstable cannot support a ratio however well-behaved the numerator is.
  * Arm order alternates by launch (native/ORT/native, then ORT/native/ORT-...)
    so that a monotone drift within a launch cannot masquerade as an effect.
    The merged matrix always ran native first.

What "trusted" means here
-------------------------
Identical to the in-tree harness: peak *instantaneous runnable* count sampled
throughout every arm, cell refused if it exceeds `16 + slack`.  Necessary, not
sufficient -- the documented width-16 bimodality (1.476-9.064 ms/token across
launches, burning identical CPU-seconds per wall-second) is invisible to any
load, CPU-efficiency or context-switch guard.  That is why the output is a
distribution rather than a median.

Run 1 (14 launches, `docs/benchmarks/2026-08-23-acc0-gap-at-width-16.md`): the ceiling was wrong, and it is
recorded rather than quietly widened
------------------------------------------------------------------------
Run 1 set `slack = 6` from a structural estimate of what our own cell
contributes: 16 decode workers + dispatcher + harness + sampler + shell ~= 20,
ceiling 22.  The estimate was too low.  Measured peak runnable across the 14
cells was **18, 19, 19, 22, 22, 22, 23 x6, 25, 25** -- so the cell's own
contribution reaches 25, and the guard refused 8 of 14 cells for our own
threads.

The refusals were not contention.  `competing_load()` returned **empty for all
14 cells**, and the pre-check runnable count was 2-4 every time: the host was
demonstrably quiet throughout.  So run 1 returned `REPORT NOTHING` under the
pre-registered `n_trusted >= 8` rule, and that verdict stands -- it is not
re-scored under a wider ceiling.

Run 2 raises the ceiling to `slack = 10` (26).  That number is taken from run
1's peaks, which makes it *post-hoc with respect to the run* -- disclosed here
rather than presented as a fresh structural derivation.  The reason it is
defensible is that the quantity it was fitted to (peak runnable of our own
cell) is not the outcome, and no cell it admits had a competitor: there were
none to admit.  What the ceiling still buys is the case `competing_load` cannot
see -- a sibling running many threads at under 150% CPU each.

Run 2 also lengthens each rep (768 tokens, from 384).  Run 1's intra-run
spreads were a median of 18.2% on the native arm and 16.4% on ORT, with maxima
of 35.5% and 52.7%.  That is the instrument, not the effect, and more samples
per rep is the one lever that attacks it directly.
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import acc0_gap_matrix as H  # noqa: E402

WIDTH = 16
MODEL = "llama"
BLOCK = 32
ACC = 0
SESSIONS = 1

# Pre-registered thresholds. Changing these after a run invalidates the study.
MIN_TRUSTED = 8
MAX_AA_HALFWIDTH = 0.10
EFFECT_OVER_NULL = 3.0


def verdict(cells):
    """Apply the pre-registered rule. Returns (verdict, detail dict)."""
    trusted = [c for c in cells if c["trusted"]]
    n = len(trusted)
    d = {"n_trusted": n, "n_taken": len(cells)}
    if n < MIN_TRUSTED:
        d["reason"] = (f"n_trusted={n} < {MIN_TRUSTED}; the run is too small to "
                       f"report anything, including a range")
        return "REPORT NOTHING", d

    gaps = sorted(c["gap"] for c in trusted)
    aas = sorted(c["aa"] for c in trusted)
    aa_hw = max(abs(a - 1.0) for a in aas)
    gap_med = statistics.median(gaps)
    d.update({
        "gap_median": gap_med, "gap_min": gaps[0], "gap_max": gaps[-1],
        "aa_min": aas[0], "aa_max": aas[-1], "aa_halfwidth": aa_hw,
        "effect": abs(gap_med - 1.0),
        "required_effect": EFFECT_OVER_NULL * aa_hw,
    })
    if aa_hw > MAX_AA_HALFWIDTH:
        d["reason"] = (f"A/A half-width {aa_hw:.3f} > {MAX_AA_HALFWIDTH:.2f}; the "
                       f"instrument is too loose for a point estimate")
        return "RANGE ONLY", d
    if abs(gap_med - 1.0) < EFFECT_OVER_NULL * aa_hw:
        d["reason"] = (f"effect {abs(gap_med - 1.0):.3f} < {EFFECT_OVER_NULL:.0f}x "
                       f"the A/A half-width {aa_hw:.3f}")
        return "RANGE ONLY", d
    d["reason"] = "all three pre-registered conditions met"
    return "ACCEPT", d


def wait_quiet_runnable(ceiling, limit, period=10.0):
    """Pre-check on the *instantaneous runnable count*, not the load average.

    `acc0_gap_matrix.wait_quiet` gates on `os.getloadavg()[0]`, and the
    harness's own `LoadWatch` docstring explains why that is the wrong
    instrument: a 1-minute exponential average both lags a job that has just
    started and stays elevated long after one has finished.  In practice it
    stalls: after this script's own `cargo build` the load average sat at 6.86
    on a host whose runnable count was 2, so a `threshold=3.0` pre-check slept
    through a perfectly quiet window without measuring anything.

    Gating on the same quantity `LoadWatch` polices during the arm makes the
    pre-check and the in-flight guard agree, which is the property that was
    missing.  Contract is unchanged from `wait_quiet`: it returns the
    competitors it could not wait out rather than pretending the host was
    quiet, so a caller that measures anyway must mark the cell untrusted.
    """
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
    """One paired cell. Arm order alternates by launch."""
    tokens = args.tokens
    load, busy = wait_quiet_runnable(args.quiet_runnable, args.quiet_limit)
    rec = {"launch": launch, "width": WIDTH, "tokens": tokens,
           "runnable_pre": load, "competitors": [c[2] for c in busy]}
    native_first = (launch % 2 == 0)
    with H.LoadWatch() as watch:
        if native_first:
            a1 = H.native(args.binary, MODEL, BLOCK, ACC, WIDTH, SESSIONS,
                          tokens, args.reps)
            o = H.ort(MODEL, BLOCK, ACC, WIDTH, SESSIONS, tokens, args.reps)
            a2 = H.native(args.binary, MODEL, BLOCK, ACC, WIDTH, SESSIONS,
                          tokens, args.reps)
        else:
            o1 = H.ort(MODEL, BLOCK, ACC, WIDTH, SESSIONS, tokens, args.reps)
            a1 = H.native(args.binary, MODEL, BLOCK, ACC, WIDTH, SESSIONS,
                          tokens, args.reps)
            o2 = H.ort(MODEL, BLOCK, ACC, WIDTH, SESSIONS, tokens, args.reps)
            # The A/A partner is whichever arm ran twice; when ORT ran twice the
            # null measured is ORT's, which is the arm the merged run showed to
            # be the *less* stable of the two. Both nulls are wanted.
            a2 = None
            o = o1
            rec["ort_aa"] = o2["tps"] / o1["tps"] if o1["tps"] else None
            rec["ort_b"] = o2
    rec["runnable_peak"] = watch.peak
    rec["arm_order"] = "native,ort,native" if native_first else "ort,native,ort"
    rec["native"] = a1
    rec["native_b"] = a2
    rec["ort"] = o
    rec["trusted"] = (watch.peak <= WIDTH + args.slack) and not busy
    rec["gap"] = o["tps"] / a1["tps"] if a1["tps"] else None
    if a2 is not None:
        rec["aa"] = a2["tps"] / a1["tps"] if a1["tps"] else None
    else:
        rec["aa"] = rec.get("ort_aa")
    rec["aa_arm"] = "native" if a2 is not None else "ort"
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True)
    # The harness runs every arm with cwd=<benches dir>, so a path
    # relative to this script would resolve against the wrong root.
    ap.add_argument("--launches", type=int, default=12)
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--reps", type=int, default=2)
    # Derived from what our own cell legitimately contributes, not tuned
    # against results: 16 decode workers + 1 dispatcher + the harness parent +
    # the sampler thread + the shell ~= 20, so 16 + 6 = 22 leaves headroom for
    # spin-up transients without admitting a real competitor. `competing_load`
    # (any non-ours process above 150% CPU) is the sharper of the two guards
    # and both must pass.
    ap.add_argument("--slack", type=int, default=10)
    ap.add_argument("--quiet-runnable", type=int, default=4,
                    help="instantaneous runnable ceiling for the pre-check; the load average lags and stalls this study on its own build")
    ap.add_argument("--quiet-limit", type=int, default=600)
    ap.add_argument("--out", default="w16_study.json")
    args = ap.parse_args()
    args.binary = os.path.abspath(args.binary)

    print(f"# pre-registered: n>={MIN_TRUSTED}, aa_halfwidth<={MAX_AA_HALFWIDTH}, "
          f"effect>={EFFECT_OVER_NULL}x null")
    hdr = (f"{'L':>3} {'ord':>17} {'run':>4} {'T':>2} {'nat_tps':>8} {'nat_sp':>7} "
           f"{'ort_tps':>8} {'ort_sp':>7} {'gap':>7} {'aa':>7} {'aa_arm':>7}")
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
        print(f"{c['launch']:>3} {c['arm_order']:>17} {c['runnable_peak']:>4} "
              f"{'y' if c['trusted'] else 'N':>2} "
              f"{c['native']['tps']:>8.1f} {c['native']['spread']:>7.1f} "
              f"{c['ort']['tps']:>8.1f} {c['ort']['spread']:>7.1f} "
              f"{c['gap']:>7.3f} "
              f"{(c['aa'] if c['aa'] else float('nan')):>7.3f} {c['aa_arm']:>7}",
              flush=True)
        with open(args.out, "w") as f:
            json.dump(cells, f, indent=1)

    v, d = verdict(cells)
    print()
    print(f"VERDICT: {v}")
    for k in ("n_taken", "n_trusted", "gap_median", "gap_min", "gap_max",
              "aa_min", "aa_max", "aa_halfwidth", "effect", "required_effect",
              "reason"):
        if k in d:
            val = d[k]
            print(f"  {k:>16} = {val:.4f}" if isinstance(val, float)
                  else f"  {k:>16} = {val}")

    trusted = [c for c in cells if c["trusted"]]
    if trusted:
        nat = sorted(c["native"]["tps"] for c in trusted)
        ort = sorted(c["ort"]["tps"] for c in trusted)
        print(f"\n  native tok/s distribution: "
              f"{' '.join(f'{x:.1f}' for x in nat)}")
        print(f"  ORT    tok/s distribution: "
              f"{' '.join(f'{x:.1f}' for x in ort)}")
        print(f"  native launch spread: "
              f"{100 * (nat[-1] - nat[0]) / nat[0]:.1f}%")
        print(f"  ORT    launch spread: "
              f"{100 * (ort[-1] - ort[0]) / ort[0]:.1f}%")
    print(f"\nraw: {args.out}")


if __name__ == "__main__":
    main()
