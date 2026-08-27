#!/usr/bin/env python3
"""Does the width-16 decode straggler follow the LANE or the CHUNK?

Six candidate selectors for the width-16 straggler have been measured and
rejected: work assignment (`ops_spread` 0.0000 in 24/24), lane index, CPU
placement, virtual address layout (`setarch -R`), physical page backing
(`prctl(PR_SET_THP_DISABLE)`), and mode-of-operation-explained-by-placement (240
launches on verified one-per-physical-core placement, still bimodal ~50/50).

The candidate list is empty, and #2017 explains why a seventh could not be
tested: lane `i` always runs on cpu `2i` *and* always computes output chunk `i`.
Both maps are static for the life of a process, so "lane 7 is slow" (a
thread/core/hardware property) and "chunk 7 is slow" (a data property -- cache
colouring or page interleave of that weight range) predict *identical*
observations in every dataset collected to date. They score ~0.208 against a 0.5
bar only because they are the same number. This is structural: no number of
repetitions separates them.

`ONNX_GENAI_CPU_DECODE_CHUNK_PERMUTATION` (#2030, `1c7d0be36`) permutes
lane->chunk while holding lane->cpu fixed, which breaks the tie. With
`rotate:k`, lane `i` computes canonical chunk `(i + k) mod w`, so canonical
chunk `c` is computed by lane `(c - k) mod w`.

    lane frame   = straggler_idx                  (invariant if it is the lane)
    chunk frame  = (straggler_idx + k) mod w      (invariant if it is the chunk)

At k=0 the two frames coincide, which is precisely the ambiguity; the
discrimination comes entirely from the k != 0 arms. The two frames cannot both
concentrate, so this is a genuine discriminator rather than a confirmation test.

PROFILED RUNS ARE DIAGNOSTICS, NOT TIMINGS
------------------------------------------
`ONNX_GENAI_CPU_DECODE_WORKER_PROFILE=1` is required to read `last_arrivals` at
all, and it is separately known to dissolve the width-16 *bimodality*. No
throughput number from this harness may be quoted against an unprofiled arm.
This probe deliberately reports no ms/token comparison: the question is which
index the straggler occupies, not how fast anything is.

PRE-REGISTERED RULE (written before the first launch; do not edit after seeing
data -- add a new rule with a new name instead)
-----------------------------------------------------------------------------
CONTROL 1 (non-vacuity, the #2014 lesson).  Every launch must report
  `chunk_perm=<label>` equal to the arm it was launched as.  A harness that sets
  a permutation and reads its own env back has verified nothing, so the arm is
  taken from the *binary's* report of the pool it built.  Launches whose label
  disagrees are discarded.  If any arm retains < 90% of its launches, or if any
  arm's reported labels are all identical across arms, the run REPORTS NOTHING.

CONTROL 2 (placement, so a lane-frame win cannot be a placement artefact).
  Every launch must report 16 distinct `cpu` values across its 16 workers, and
  the lane->cpu map must be identical in every launch of every arm.  If the map
  moves, placement is not being held fixed and the frames are not separable.

OBSERVABLE.  Per trusted launch, `straggler_idx` = the worker index holding the
  largest share of `last_arrivals` (the worker that is systematically last, and
  therefore the one every other worker waits for).  Launches are trusted by the
  existing `acc0_w16_worker_split.trusted` residual-bounds test.

STATISTIC.  Pooled over ALL arms, in each frame independently:
      concentration = max_i (count of launches whose frame-index == i) / N
  Chance for w=16 is 0.0625.

FLOOR.  The floor is not assumed.  For each frame, shuffle the arm labels across
  launches 2000 times (holding the observed `straggler_idx` values fixed) and
  take the 95th percentile of the resulting concentration.  This is the
  concentration that frame can reach when `k` carries no information, and it
  differs between the two frames because the chunk frame adds `k`.

VERDICT.
  LANE      lane concentration > its floor, AND lane > chunk.
  CHUNK     chunk concentration > its floor, AND chunk > lane.
  NEITHER   neither frame clears its own floor -- the victim is selected by
            something that moves when the assignment moves, i.e. an interaction
            rather than a property of either static map.  This is a real,
            pre-registered outcome, not a failure of the probe.
  AMBIGUOUS both clear their floors and differ by less than the wider floor
            margin.

The three outcomes were written down before any data existed, in
`docs/benchmarks/2026-08-24-acc0-chunk-permutation-instrument.md`, so the result
cannot be read backwards.

RULE 2 (added 2026-08-24 AFTER the n=120 pilot; RULE 1's verdict stands exactly
as recorded and is NOT edited)
-----------------------------------------------------------------------------
The pilot returned NEITHER under RULE 1, and inspecting it showed why the
pre-registered statistic is the wrong shape for this data rather than merely
unlucky. Within a process the victim is overwhelming -- median
`straggler_share` 0.815, i.e. one lane holds 81% of all last-arrivals, 12.2x
chance -- so there is definitely a victim to find. But across processes its
identity is spread over 12 of 15 lanes, with three lanes (9, 12, 14) never
selected once in 120 runs. `max cell / N` cannot see "drawn from a biased subset";
it only sees "always the same index".

So RULE 2 adds a *distributional* statistic, as a new named rule, testing the
shape of the whole histogram rather than its peak:

  chi2 = sum_i (obs_i - n/L)^2 / (n/L)   over that frame's L bins

  The two frames have different L. The lane frame has L = the number of
  profiled lanes (15): lane indices cannot exceed the worker list. The chunk
  frame has L = WIDTH (16), because CHUNK is (idx+k) % WIDTH and the dispatcher
  computes a shard without appearing in the worker list, so the chunk alphabet
  is one symbol wider than the lane alphabet. Scoring the chunk arm on 15 bins
  drops every chunk-15 sample from the sum while still dividing by the full n.
  (Corrected 2026-08-27; measured to leave every verdict in this study
  unchanged, since it perturbs the observed statistic and its shuffle null
  alike, but it is wrong and would not stay harmless at other widths.)

  * CHUNK frame null: shuffle the `k` labels across launches. This holds the
    lane frame **exactly invariant** (shuffling `k` cannot change `idx`) while
    fully randomizing the chunk frame, so it is the correct null for "does the
    chunk frame carry structure beyond what the lane frame already induces".
  * LANE frame null: uniform multinomial over L lanes at the same n. The label
    shuffle is useless here precisely because the lane frame is invariant under
    it, so the null has to come from uniformity instead.

  p is the fraction of 2000 draws with chi2 >= observed; significance at p<0.05.

RULE 2 VERDICT.
  LANE-BIASED    lane chi2 significant, chunk chi2 not. The victim is not a
                 fixed lane (RULE 1 already refuted that) but lane identity
                 still carries information -- a probabilistic lane property.
  CHUNK-BIASED   chunk chi2 exceeds its shuffle floor, lane chi2 not.
  BOTH/NEITHER   reported literally, with both p-values.

RULE 2 cannot overturn RULE 1: a NEITHER under RULE 1 means no single index
dominates in either frame, and that remains true regardless of what RULE 2 says
about the histogram's shape. RULE 2 can only refine *how* the victim is chosen.

RULE 3 (written 2026-08-24 from the n=120 pilot histogram, and pre-registered
BEFORE the confirmatory run was inspected -- the confirmatory data existed but
had not been read when this was written)
-----------------------------------------------------------------------------
The pilot's lane histogram was [4,16,9,15,1,17,12,16,13,0,1,3,0,13,0], which is
not merely non-uniform, it has two visible regularities. Both are stated here as
one-sided hypotheses with their null proportions fixed in advance, to be tested
on the independent confirmatory dataset:

  H1 (lane parity).  Odd lane indices straggle more often than chance.
      Lane `i` is pinned to cpu `2i`, so odd `i` means cpu = 2 mod 4.
      Null p0 = 7/15 = 0.4667 (odd lanes 1,3,5,7,9,11,13 of 15).
      Pilot: 80/120 = 0.667.

  H2 (L3 domain).  Lanes 0-7 (cpus 0..14, the first 32 MiB L3) straggle more
      often than lanes 8-14 (cpus 16..28, the second L3).
      Null p0 = 8/15 = 0.5333.
      Pilot: 90/120 = 0.750.

  One-sided binomial test, significance p<0.05. A pilot-suggested pattern that
  fails to replicate on independent data is reported as failing to replicate,
  and the pilot number is *not* re-quoted as if it were a result.

These are the first positive structural claims after seven rejected selectors,
so they get held to replication rather than announced from the sample that
generated them.

RULE 4 (written 2026-08-24, before any odd-rotation launch existed)
-----------------------------------------------------------------------------
A defect in the arm set found while scoring RULE 3, recorded rather than
quietly fixed. The four arms are k in {0,4,8,12}, all EVEN. Since canonical
chunk = (lane + k) mod 16, an even k preserves index parity: an odd lane always
computes an odd chunk. **H1 is therefore confounded** -- the existing data
cannot distinguish "odd lanes are slow" from "odd chunks are slow", and any
parity result from the pilot or the confirmatory run must be reported as
frame-ambiguous. H2 is NOT confounded, because k=4/8/12 move indices across the
lanes-0-7 boundary, so H2 stands on its own.

Odd k flips parity (odd lane -> even chunk), which breaks the confound. Arms
k in {1,3,5,7} pooled, with per-arm exact nulls:

  odd LANE  null = 7/15  = 0.4667  (odd lanes 1,3,..,13 of the 15 profiled)
  odd CHUNK null = 8/15  = 0.5333  for odd k (chunk odd <=> lane even, 8 of 15)

  Prediction A (parity is lane-anchored): odd-lane excess survives, odd-chunk
      sits at its null.
  Prediction B (parity is chunk-anchored): odd-chunk excess survives, odd-lane
      sits at its null.
  Prediction C: both at null -- parity was an artifact of the even-k arm set and
      is withdrawn.

  One-sided binomial, p<0.05. H2 is re-tested on the same odd-k data as an
  independent replication.
"""

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import acc0_gap_matrix as H  # noqa: E402
import acc0_w16_worker_split as W  # noqa: E402

WIDTH = 16
ROTATIONS = (0, 4, 8, 12)
DRAWS = 2000
RETAIN_FLOOR = 0.90


def arm_label(k):
    return f"rotate:{k}"


def run_arm(args, k):
    """One profiled native launch at width 16 under `rotate:k`.

    Mirrors `acc0_w16_worker_split.run_width` exactly, plus the permutation, so
    the only difference between these rows and the published straggler rows is
    the knob under test.
    """
    env = {
        "PROBE_MODEL": W.MODEL, "PROBE_BLOCK": W.BLOCK, "PROBE_ACCURACY": W.ACC,
        "PROBE_SESSIONS": W.SESSIONS, "PROBE_TOKENS": args.tokens,
        "PROBE_REPS": args.reps,
        "ONNX_GENAI_CPU_DECODE_THREADS": WIDTH,
        "ONNX_GENAI_CPU_DECODE_WORKER_PROFILE": "1",
        "ONNX_GENAI_CPU_DECODE_BLOCKTIME_US": args.blocktime,
        "ONNX_GENAI_CPU_DECODE_CHUNK_PERMUTATION": arm_label(k),
    }
    r = H.sh(f"taskset -c {H.PIN} {args.binary}", env)
    return r.stdout + "\n" + r.stderr


def reported_perm(text):
    """CONTROL 1: the permutation the *binary* says its pool was built with."""
    for line in text.splitlines():
        if line.strip().startswith("decode_width"):
            for tok in line.split():
                if tok.startswith("chunk_perm="):
                    return tok.split("=", 1)[1]
    return None


def one_launch(args, launch):
    rec = {"launch": launch, "arms": {}, "peak_limit": args.quiet_limit}
    # Rotate the arm order every launch so no arm keeps a fixed slot: a warm
    # cache or a drifting host would otherwise land on the same arm every time.
    order = ROTATIONS[launch % len(ROTATIONS):] + ROTATIONS[:launch % len(ROTATIONS)]
    rec["order"] = list(order)
    with H.LoadWatch() as watch:
        for k in order:
            text = run_arm(args, k)
            workers = W.parse_workers(text)
            rec["arms"][str(k)] = {
                "derived": W.derive(workers),
                "reported_perm": reported_perm(text),
                "cpus": [w["cpu"] for w in sorted(workers, key=lambda w: w["idx"])],
            }
    rec["peak"] = watch.peak
    return rec


def trusted_launch(rec):
    """Reuse the published residual-bounds test, shaped for this record."""
    shim = {
        "peak": rec.get("peak", 0),
        "peak_limit": rec.get("peak_limit", 10 ** 9),
        "widths": {k: v["derived"] for k, v in rec["arms"].items()},
    }
    return W.trusted(shim)


def concentration(values):
    if not values:
        return 0.0, None
    counts = Counter(values)
    idx, top = counts.most_common(1)[0]
    return top / len(values), idx


def floor_for(pairs, frame, draws, seed=20260824):
    """95th percentile concentration when the arm labels carry no information.

    Holds the observed straggler indices fixed and shuffles which arm each came
    from. The chunk frame adds `k`, so its floor is genuinely different from the
    lane frame's -- which is exactly why the floor is measured per frame rather
    than assumed to be chance.
    """
    rng = random.Random(seed)
    idxs = [idx for idx, _ in pairs]
    ks = [k for _, k in pairs]
    out = []
    for _ in range(draws):
        shuffled = ks[:]
        rng.shuffle(shuffled)
        out.append(concentration([frame(i, k) for i, k in zip(idxs, shuffled)])[0])
    out.sort()
    return out[int(0.95 * (len(out) - 1))]


LANE = lambda idx, k: idx  # noqa: E731
CHUNK = lambda idx, k: (idx + k) % WIDTH  # noqa: E731


def report(recs, args):
    lines = []
    kept = [r for r in recs if trusted_launch(r)]
    lines.append(f"launches: {len(recs)} taken, {len(kept)} trusted")
    if not kept:
        return ["REPORT NOTHING: no trusted launches"]

    # CONTROL 1 --------------------------------------------------------------
    pairs, retained, labels_seen = [], {}, set()
    for rec in kept:
        for k in ROTATIONS:
            cell = rec["arms"].get(str(k))
            if not cell or not cell["derived"]:
                continue
            retained.setdefault(k, [0, 0])
            retained[k][1] += 1
            if cell["reported_perm"] != arm_label(k):
                continue
            retained[k][0] += 1
            labels_seen.add(cell["reported_perm"])
            pairs.append((cell["derived"]["straggler_idx"], k))

    lines.append("")
    lines.append("CONTROL 1 (the binary must report the arm it was launched as)")
    bad = False
    for k in ROTATIONS:
        got, tot = retained.get(k, [0, 0])
        frac = got / tot if tot else 0.0
        flag = "" if frac >= RETAIN_FLOOR else "  <-- BELOW FLOOR"
        lines.append(f"  rotate:{k:<3} retained {got}/{tot} = {frac:.3f}{flag}")
        if frac < RETAIN_FLOOR:
            bad = True
    if len(labels_seen) < len(ROTATIONS):
        lines.append(f"  distinct labels observed: {sorted(labels_seen)}  <-- ARMS NOT DISTINCT")
        bad = True
    if bad:
        return lines + ["", "REPORT NOTHING: CONTROL 1 failed -- the arms are not what they claim"]

    # CONTROL 2 --------------------------------------------------------------
    maps = {tuple(c["cpus"]) for r in kept for c in r["arms"].values() if c["cpus"]}
    lines.append("")
    lines.append("CONTROL 2 (placement held fixed)")
    lines.append(f"  distinct lane->cpu maps across all arms and launches: {len(maps)}")
    if len(maps) != 1:
        return lines + ["", "REPORT NOTHING: CONTROL 2 failed -- placement moved between arms"]
    only = next(iter(maps))
    lines.append(f"  map: {list(only)}")
    lines.append(f"  distinct cpus: {len(set(only))}/{len(only)}")
    if len(set(only)) != len(only):
        return lines + ["", "REPORT NOTHING: CONTROL 2 failed -- lanes share cpus"]

    # The measurement -------------------------------------------------------
    lane_c, lane_i = concentration([LANE(i, k) for i, k in pairs])
    chunk_c, chunk_i = concentration([CHUNK(i, k) for i, k in pairs])
    lane_f = floor_for(pairs, LANE, args.draws)
    chunk_f = floor_for(pairs, CHUNK, args.draws)

    lines.append("")
    lines.append(f"pooled samples: {len(pairs)}   chance = {1 / WIDTH:.4f}")
    lines.append("")
    lines.append("  frame        concentration   at index   floor(95%)   clears?")
    for name, conc, at, floor in (
        ("lane ", lane_c, lane_i, lane_f),
        ("chunk", chunk_c, chunk_i, chunk_f),
    ):
        lines.append(
            f"  {name}            {conc:.4f}         {at:<6}     {floor:.4f}"
            f"       {'YES' if conc > floor else 'no'}"
        )

    lines.append("")
    lines.append("per-arm straggler index (lane frame -> chunk frame)")
    for k in ROTATIONS:
        arm = [(i, kk) for i, kk in pairs if kk == k]
        lc, li = concentration([LANE(i, kk) for i, kk in arm])
        cc, ci = concentration([CHUNK(i, kk) for i, kk in arm])
        lines.append(
            f"  rotate:{k:<3} n={len(arm):<3} lane {lc:.3f}@{li:<3} "
            f"chunk {cc:.3f}@{ci}"
        )

    # Verdict ---------------------------------------------------------------
    lane_ok, chunk_ok = lane_c > lane_f, chunk_c > chunk_f
    margin = max(lane_f, chunk_f)
    lines.append("")
    if lane_ok and chunk_ok and abs(lane_c - chunk_c) < margin - 1 / WIDTH:
        verdict = "AMBIGUOUS: both frames clear their floors and are too close to separate"
    elif lane_ok and lane_c > chunk_c:
        verdict = f"LANE: the straggler is a lane property (concentrates at lane {lane_i})"
    elif chunk_ok and chunk_c > lane_c:
        verdict = f"CHUNK: the straggler is a data property (concentrates at chunk {chunk_i})"
    else:
        verdict = (
            "NEITHER: no frame clears its own floor. The victim is not a property of "
            "either static map; it moves when the assignment moves, which makes it an "
            "interaction. Pre-registered outcome, not a probe failure."
        )
    lines.append(f"VERDICT (RULE 1): {verdict}")

    # ---- RULE 2 (added after the pilot; see module docstring) --------------
    # The lane count is taken from the pool's own reported worker list, not from
    # the largest index observed: a lane that never straggles is exactly the
    # signal here, and inferring L from the data would silently delete it.
    lanes = max(len(c["cpus"]) for r in kept for c in r["arms"].values() if c["cpus"])
    shares = [
        c["derived"]["straggler_share"]
        for r in kept
        for c in r["arms"].values()
        if c["derived"]
    ]
    lines.append("")
    lines.append("RULE 2 (distributional; RULE 1 above is unedited)")
    lines.append(
        f"  within-process victim: median straggler_share "
        f"{statistics.median(shares):.4f} over {len(shares)} runs "
        f"(chance {1 / lanes:.4f}, so {statistics.median(shares) * lanes:.1f}x)"
    )
    lane_obs, lane_p, chunk_obs, chunk_p = rule2(pairs, lanes, args.draws)
    lines.append(f"  lanes considered: L={lanes}")
    lines.append(f"  lane  frame chi2 {lane_obs:8.2f}   p={lane_p:.4f}   "
                 f"{'SIGNIFICANT' if lane_p < 0.05 else 'not significant'}")
    lines.append(f"  chunk frame chi2 {chunk_obs:8.2f}   p={chunk_p:.4f}   "
                 f"{'SIGNIFICANT' if chunk_p < 0.05 else 'not significant'}")
    counts = Counter(i for i, _ in pairs)
    lines.append(f"  lane histogram: {[counts.get(i, 0) for i in range(lanes)]}")
    if lane_p < 0.05 and chunk_p >= 0.05:
        r2 = ("LANE-BIASED: lane identity carries information but does not "
              "determine the victim")
    elif chunk_p < 0.05 and lane_p >= 0.05:
        r2 = "CHUNK-BIASED: the chunk frame carries structure the lane frame does not"
    elif lane_p < 0.05 and chunk_p < 0.05:
        r2 = "BOTH frames show structure"
    else:
        r2 = "NEITHER frame shows distributional structure"
    lines.append(f"VERDICT (RULE 2): {r2}")

    # ---- RULE 3 (pre-registered from the pilot; see module docstring) ------
    (odd, n3, p0o, po), (low, _, p0f, pf) = rule3(pairs, lanes)
    lines.append("")
    lines.append("RULE 3 (pre-registered one-sided binomial tests)")
    lines.append(
        f"  H1 lane parity   odd {odd}/{n3} = {odd / n3:.4f}  vs null {p0o:.4f}  "
        f"p={po:.5f}  {'SIGNIFICANT' if po < 0.05 else 'not significant'}"
    )
    lines.append(
        f"  H2 L3 domain     lanes0-7 {low}/{n3} = {low / n3:.4f}  vs null {p0f:.4f}  "
        f"p={pf:.5f}  {'SIGNIFICANT' if pf < 0.05 else 'not significant'}"
    )
    if all(k % 2 == 0 for _, k in pairs):
        lines.append("  NOTE: all arms have even k, so odd lane <=> odd chunk.")
        lines.append("        H1 is FRAME-AMBIGUOUS in this dataset (see RULE 4). H2 is not.")

    # ---- RULE 4 (pre-registered; needs odd-k arms) --------------------------
    r4 = rule4(pairs, lanes)
    if r4:
        (ol, n4, pl, pol), (oc, _, pc, poc), (lo4, _, plo, plo_p) = r4
        lines.append("")
        lines.append(f"RULE 4 (odd-k arms only, n={n4}; breaks the parity confound)")
        lines.append(
            f"  odd LANE   {ol}/{n4} = {ol / n4:.4f}  vs null {pl:.4f}  p={pol:.5f}  "
            f"{'SIGNIFICANT' if pol < 0.05 else 'not significant'}"
        )
        lines.append(
            f"  odd CHUNK  {oc}/{n4} = {oc / n4:.4f}  vs null {pc:.4f}  p={poc:.5f}  "
            f"{'SIGNIFICANT' if poc < 0.05 else 'not significant'}"
        )
        lines.append(
            f"  H2 replic. lanes0-7 {lo4}/{n4} = {lo4 / n4:.4f}  vs null {plo:.4f}  "
            f"p={plo_p:.5f}  {'SIGNIFICANT' if plo_p < 0.05 else 'not significant'}"
        )
        if pol < 0.05 and poc >= 0.05:
            v4 = "A: parity is LANE-anchored"
        elif poc < 0.05 and pol >= 0.05:
            v4 = "B: parity is CHUNK-anchored"
        elif pol < 0.05 and poc < 0.05:
            v4 = "both significant -- underpowered or nulls wrong; do not interpret"
        else:
            v4 = "C: neither -- parity withdrawn as an even-k artifact"
        lines.append(f"VERDICT (RULE 4): {v4}")
    return lines


def chi2_uniform(values, lanes):
    if not values:
        return 0.0
    exp = len(values) / lanes
    counts = Counter(values)
    return sum((counts.get(i, 0) - exp) ** 2 / exp for i in range(lanes))


def rule2(pairs, lanes, draws, seed=20260824):
    """Distributional test; see RULE 2 in the module docstring."""
    rng = random.Random(seed)
    idxs = [i for i, _ in pairs]
    ks = [k for _, k in pairs]
    n = len(pairs)

    lane_obs = chi2_uniform(idxs, lanes)
    # The two frames do not have the same alphabet. Lane indices run over the
    # `lanes` profiled workers, but CHUNK is (idx+k) % WIDTH and so runs over
    # WIDTH symbols -- one more than there are profiled lanes, because the
    # dispatcher computes a shard without appearing in the worker list. Scoring
    # the chunk arm on `lanes` bins silently drops every sample whose chunk
    # index is WIDTH-1 while still dividing by the full n.
    chunk_obs = chi2_uniform([CHUNK(i, k) for i, k in pairs], WIDTH)

    # Chunk null: shuffle k (lane frame is invariant under this by construction).
    chunk_null = []
    for _ in range(draws):
        sh = ks[:]
        rng.shuffle(sh)
        chunk_null.append(chi2_uniform([CHUNK(i, k) for i, k in zip(idxs, sh)], WIDTH))
    chunk_p = sum(1 for v in chunk_null if v >= chunk_obs) / draws

    # Lane null: uniform multinomial at the same n.
    lane_null = []
    for _ in range(draws):
        draw = [rng.randrange(lanes) for _ in range(n)]
        lane_null.append(chi2_uniform(draw, lanes))
    lane_p = sum(1 for v in lane_null if v >= lane_obs) / draws
    return lane_obs, lane_p, chunk_obs, chunk_p


def binom_p(successes, n, p0):
    """One-sided P(X >= successes) under Binomial(n, p0), exact."""
    from math import comb
    return sum(comb(n, i) * p0 ** i * (1 - p0) ** (n - i) for i in range(successes, n + 1))


def rule3(pairs, lanes):
    idxs = [i for i, _ in pairs]
    n = len(idxs)
    odd_lanes = [i for i in range(lanes) if i % 2 == 1]
    p0_odd = len(odd_lanes) / lanes
    odd = sum(1 for i in idxs if i % 2 == 1)

    first = [i for i in range(lanes) if i < 8]
    p0_first = len(first) / lanes
    low = sum(1 for i in idxs if i < 8)
    return (
        (odd, n, p0_odd, binom_p(odd, n, p0_odd)),
        (low, n, p0_first, binom_p(low, n, p0_first)),
    )


def rule4(pairs, lanes):
    """Pooled odd-k parity test. Nulls are exact and depend on k's parity."""
    odd_ks = [(i, k) for i, k in pairs if k % 2 == 1]
    n = len(odd_ks)
    if not n:
        return None
    odd_lane = sum(1 for i, _ in odd_ks if i % 2 == 1)
    odd_chunk = sum(1 for i, k in odd_ks if (i + k) % 2 == 1)
    p_lane = len([i for i in range(lanes) if i % 2 == 1]) / lanes
    p_chunk = len([i for i in range(lanes) if i % 2 == 0]) / lanes
    low = sum(1 for i, _ in odd_ks if i < 8)
    p_low = len([i for i in range(lanes) if i < 8]) / lanes
    return (
        (odd_lane, n, p_lane, binom_p(odd_lane, n, p_lane)),
        (odd_chunk, n, p_chunk, binom_p(odd_chunk, n, p_chunk)),
        (low, n, p_low, binom_p(low, n, p_low)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=None)
    ap.add_argument("--launches", type=int, default=25)
    ap.add_argument("--tokens", type=int, default=192)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--blocktime", type=int, default=0)
    ap.add_argument("--quiet-limit", type=int, default=40)
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--out", default=None)
    ap.add_argument("--replay", default=None)
    ap.add_argument("--rotations", default=None, help="comma-separated k values")
    args = ap.parse_args()
    if args.rotations:
        global ROTATIONS
        ROTATIONS = [int(x) for x in args.rotations.split(",")]

    if args.replay:
        with open(args.replay) as fh:
            blob = json.load(fh)
        # Older datasets are a bare list; newer ones carry the lock provenance
        # alongside the runs. Read both rather than orphan the earlier records.
        recs = blob["runs"] if isinstance(blob, dict) else blob
        if isinstance(blob, dict):
            prov = blob.get("hostlock", {})
            state = prov.get("hostlock_state", "unrecorded")
            who = prov.get("held_by", "?")
            runnable = prov.get("runnable_at_acquire", "?")
            print(f"hostlock at acquire: {state} held_by={who} "
                  f"runnable={runnable}")
        else:
            print("hostlock at acquire: unrecorded (pre-lock dataset)")
        print("\n".join(report(recs, args)))
        return

    if not args.binary:
        ap.error("--binary is required unless --replay is given")

    # The whole sweep runs under the advisory host lock, not each launch: a
    # per-launch acquire would release the box between launches and let a
    # competitor land in the middle of a matrix that is only comparable if
    # every arm saw the same machine.
    with H.HostLock(owner="roy",
                    reason=f"acc0 w16 chunk permutation, {args.launches} launches") as lock:
        recs = []
        for launch in range(args.launches):
            recs.append(one_launch(args, launch))
            if args.out:
                with open(args.out, "w") as fh:
                    json.dump({"hostlock": lock.provenance, "runs": recs}, fh, indent=1)
            print(f"launch {launch + 1}/{args.launches} done", file=sys.stderr, flush=True)
    print("\n".join(report(recs, args)))


if __name__ == "__main__":
    main()
