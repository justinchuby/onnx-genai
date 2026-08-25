#!/usr/bin/env python3
"""Finish the `dequant_panel_avx2` modulo-elimination matrix (#1809, refs #1676).

#1809 merged the elimination with a partial matrix: block-16 decode at 1.015x,
prefill nulls at m = 64/256/512, and m = 1 and m = 8 **withheld** because their
A/A null came back at 5.31% and 4.62%. This fills the withheld rows in and
takes the sweep at every power of two from 8 to 512, plus the block-16 decode
row again on current main.

Three arms, three separately built binaries from one source tree
(`build_arms.sh`):

  before   let offset_in_block = (depth + q) % block_size;
  after    let offset_in_block = offset_base + q;
  aa       a byte-identical copy of `after`

`aa` is the null. It is a *separate file* rather than a second run of the same
path, so it is a genuinely independent launch and pays every per-launch cost
the real arms pay -- ASLR, page backing, first-touch. A null taken any other
way understates the noise floor it is supposed to bound.

Method, following #1809's correction: the host gate is the **CPU efficiency of
the run itself** -- `os.wait4` rusage `(utime + stime) / wall` -- and not an
instantaneous runnable count sampled at run boundaries. A 2-second run has room
for a burst that starts after the opening sample and ends before the closing
one; that is how a 52% A/A null once passed a "host clean" check. A process
pinned to one core that is not being descheduled spends ~1.00 CPU-seconds per
wall-second, so this measures the thing directly.

Arms are interleaved **at launch granularity and rotated per round**, and every
row is reported as a distribution over independent launches. A single paired
A/B is not reported at any width: the decode loop on this host is bimodal per
process launch, and one pairing can be dominated by which mode each side
landed in.
"""

import argparse
import json
import os
import random
import resource
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
# Written by `int4_modulo_arms.sh`; override together with `MOD_ARMS_OUT`.
BIN = os.environ.get("MOD_ARMS_OUT", os.path.join(ROOT, "target/int4-modulo-arms"))
sys.path.insert(0, HERE)
import acc0_gap_matrix as H  # noqa: E402

# cpu 0 has a permanent external competitor on this host and cpu 1 is its SMT
# sibling, so a run pinned there is contended by construction. cpu 4 is an even
# cpu (one per physical core) away from both.
PIN = os.environ.get("MOD_PIN", "4")
CPU_EFF_FLOOR = 0.95


def launch(binary, env_extra, timeout=1800):
    """One launch, with its own CPU efficiency measured by rusage.

    Returns (rows, cpu_eff). `rows` maps m -> dict of the printed columns.
    """
    env = dict(os.environ)
    env.update(env_extra)
    env.pop("PROBE_MS", None)
    argv = ["taskset", "-c", PIN, binary, "--bench"]
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    proc = subprocess.run(
        argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        timeout=timeout, check=True,
    )
    wall = time.perf_counter() - start
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    rows = {}
    for line in proc.stdout.decode().splitlines():
        f = line.split()
        if len(f) == 8 and f[0].isdigit():
            rows[int(f[2])] = {
                "k": int(f[0]), "n": int(f[1]),
                "cold_ms": float(f[3]), "steady_ms": float(f[4]),
                "gflops": float(f[5]), "sum": f[6], "fnv": f[7],
            }
    return rows, (cpu / wall if wall > 0 else 0.0)


def decode_launch(binary, env_extra, timeout=1800):
    env = dict(os.environ)
    env.update(env_extra)
    argv = ["taskset", "-c", PIN, binary, "--bench"]
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    proc = subprocess.run(
        argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        timeout=timeout, check=True,
    )
    wall = time.perf_counter() - start
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    out = proc.stdout.decode()
    rec = {}
    for line in out.splitlines():
        f = line.split()
        # `  cold/steady  ms_token  ms_token_p90  tokens_s_total  spread_%`
        if len(f) == 5 and f[0] in ("cold", "steady"):
            rec[f[0]] = float(f[1])
        elif line.startswith("checksum="):
            rec["checksum"] = line.split("=", 1)[1].strip()
    rec["raw"] = out
    return rec, (cpu / wall if wall > 0 else 0.0)


#: Fixed so the published intervals are reproducible rather than merely
#: recomputable to something similar.
BOOTSTRAP_SEED = 20260825
BOOTSTRAP_RESAMPLES = 20000


def bootstrap_ratio(before_vals, after_vals, resamples=BOOTSTRAP_RESAMPLES):
    """Percentile bootstrap interval for `median(before) / median(after)`.

    The point estimate on its own is not reportable on this host. The
    launch-to-launch spread reaches 100% while the median A/A null is 0.14%, so
    a ratio of medians is stable and a ratio of any single pairing is not -- and
    only the interval says which of those the reader is looking at.

    Launches are resampled, not individual timings: the launch is the unit that
    varies (placement, page backing, which decode mode the process landed in),
    and resampling below it would understate the spread it is there to capture.

    Percentile rather than BCa. The estimator is a smooth ratio of medians on
    50+ independent samples per arm and the intervals here are wide relative to
    the bias a BCa correction would remove, so the extra machinery would buy
    precision the measurement does not have.
    """
    if not before_vals or not after_vals:
        return (float("nan"), float("nan"))
    rng = random.Random(BOOTSTRAP_SEED)
    out = []
    for _ in range(resamples):
        ra = [rng.choice(before_vals) for _ in before_vals]
        rb = [rng.choice(after_vals) for _ in after_vals]
        mb = statistics.median(rb)
        out.append(statistics.median(ra) / mb if mb else float("nan"))
    out.sort()
    return (out[int(0.025 * resamples)], out[int(0.975 * resamples)])


def verdict(lo, hi):
    """What the interval permits you to say, rather than what the point says."""
    if lo > 1.0:
        return "gain"
    if hi < 1.0:
        return "loss"
    return "null"


def ratio_stats(before_vals, after_vals):
    """Medians over independent launches, and the median ratio.

    Reported as `before / after`, so > 1.000 means `after` is faster.
    """
    mb, ma = statistics.median(before_vals), statistics.median(after_vals)
    lo, hi = bootstrap_ratio(before_vals, after_vals)
    return {
        "ci_lo": lo,
        "ci_hi": hi,
        "verdict": verdict(lo, hi),
        "before_median_ms": mb,
        "after_median_ms": ma,
        "speedup": mb / ma if ma else float("nan"),
        "before_n": len(before_vals),
        "after_n": len(after_vals),
        "before_spread_pct": (max(before_vals) - min(before_vals)) / mb * 100 if mb else 0,
        "after_spread_pct": (max(after_vals) - min(after_vals)) / ma * 100 if ma else 0,
    }


def prefill_matrix(rounds, block, shape, m_list):
    arms = ["before", "after", "aa"]
    bins = {a: os.path.join(BIN, "prefill_" + ("after" if a == "aa" else a)) for a in arms}
    bins["aa"] = os.path.join(BIN, "prefill_aa")
    env = {"PROBE_BITS": "4", "PROBE_BLOCK": str(block), "PROBE_SHAPE": shape,
           "PROBE_M_LIST": ",".join(str(m) for m in m_list)}
    samples = {a: {m: [] for m in m_list} for a in arms}
    cold = {a: {m: [] for m in m_list} for a in arms}
    fnv = {a: {} for a in arms}
    discarded = 0
    for r in range(rounds):
        # Rotate, so no arm is permanently first in a round and no arm
        # permanently inherits another's cache and frequency state.
        order = arms[r % len(arms):] + arms[: r % len(arms)]
        for arm in order:
            rows, eff = launch(bins[arm], env)
            if eff < CPU_EFF_FLOOR:
                discarded += 1
                continue
            for m, row in rows.items():
                if m not in samples[arm]:
                    continue
                samples[arm][m].append(row["steady_ms"])
                cold[arm][m].append(row["cold_ms"])
                fnv[arm].setdefault(m, set()).add(row["fnv"])
        print(f"  round {r + 1}/{rounds} done", flush=True)
    table = []
    for m in m_list:
        row = {"m": m, "block": block, "shape": shape}
        row.update(ratio_stats(samples["before"][m], samples["after"][m]))
        aa = ratio_stats(samples["aa"][m], samples["after"][m])
        row["aa_null_pct"] = abs(aa["speedup"] - 1.0) * 100
        row["aa_speedup"] = aa["speedup"]
        row["aa_ci_lo"], row["aa_ci_hi"] = aa["ci_lo"], aa["ci_hi"]
        # The null has to be shown to contain 1.000, not assumed to. An A/A arm
        # whose own interval excludes 1.000 says the instrument is biased at
        # this row, and every verdict in the row is then unreadable.
        row["aa_brackets_unity"] = aa["ci_lo"] <= 1.0 <= aa["ci_hi"]
        row["cold_speedup"] = (
            statistics.median(cold["before"][m]) / statistics.median(cold["after"][m])
        )
        row["fnv"] = {a: sorted(fnv[a].get(m, [])) for a in arms}
        # Raw per-launch steady medians, so the ratio can be given a bootstrap
        # interval rather than a bare point estimate. The launch-to-launch
        # spread on this host reaches 100% while the median A/A null is 0.1%,
        # so the point estimate is only meaningful next to its interval.
        row["samples"] = {a: samples[a][m] for a in arms}
        row["bit_identical"] = (
            len(fnv["before"].get(m, set()) | fnv["after"].get(m, set()) | fnv["aa"].get(m, set())) == 1
        )
        table.append(row)
    return table, discarded


def decode_matrix(rounds, block, tokens):
    arms = ["before", "after", "aa"]
    bins = {a: os.path.join(BIN, "decode_" + a) for a in arms}
    env = {"PROBE_BLOCK": str(block), "PROBE_TOKENS": str(tokens),
           "PROBE_SESSIONS": "1", "ONNX_GENAI_CPU_DECODE_THREADS": "1"}
    samples = {a: [] for a in arms}
    cold_samples = {a: [] for a in arms}
    checks = {a: set() for a in arms}
    raw = {}
    discarded = 0
    for r in range(rounds):
        order = arms[r % len(arms):] + arms[: r % len(arms)]
        for arm in order:
            rec, eff = decode_launch(bins[arm], env)
            if eff < CPU_EFF_FLOOR:
                discarded += 1
                continue
            raw.setdefault(arm, rec["raw"])
            if "checksum" in rec:
                checks[arm].add(rec["checksum"])
            if "steady" in rec:
                samples[arm].append(rec["steady"])
            cold_samples[arm].append(rec.get("cold", float("nan")))
        print(f"  decode round {r + 1}/{rounds} done", flush=True)
    if not samples["after"]:
        return {"error": "no parseable decode samples", "raw": raw}, discarded
    out = {"block": block, "tokens": tokens}
    out.update(ratio_stats(samples["before"], samples["after"]))
    aa = ratio_stats(samples["aa"], samples["after"])
    out["aa_null_pct"] = abs(aa["speedup"] - 1.0) * 100
    out["aa_ci_lo"], out["aa_ci_hi"] = aa["ci_lo"], aa["ci_hi"]
    out["aa_brackets_unity"] = aa["ci_lo"] <= 1.0 <= aa["ci_hi"]
    out["checksums"] = {a: sorted(checks[a]) for a in arms}
    out["bit_identical"] = len(checks["before"] | checks["after"] | checks["aa"]) == 1
    out["samples"] = samples
    return out, discarded


def route_proof(m_list, shape):
    """Prove, per row, that the modified line is on the route being timed.

    A source-level A/B has to rebuild between arms, so a null is uninterpretable
    without this: a change that never executed and a change that executed and
    cost nothing produce the same table. Timing cannot separate them; a
    deliberately wrong build can.

    Expected:
      before == after   everywhere  (the elimination is exact, so it is free)
      poison != after   on every row whose route reaches the line
      poison == after   on block 32, m = 1 -- the built-in control, because
                        that row takes the N-blocked decode kernel and never
                        calls the pack at all

    That last expectation is hardcoded from `int4_prefill_gebp_min_rows`'s gate
    rather than derived, which makes it a **tripwire as well as a control**: if
    the dispatch ever changes so that block 32 at m = 1 does reach the pack, or
    some other row stops reaching it, this reports FAIL. Read such a failure as
    "the routing moved" first and "the kernel broke" second -- `before ==
    after`, which is checked on every row independently, is the half that speaks
    to correctness.
    """
    arms = ["before", "after", "poison"]
    rows = {}
    for block in (16, 32):
        for arm in arms:
            env = {"PROBE_BITS": "4", "PROBE_BLOCK": str(block), "PROBE_SHAPE": shape,
                   "PROBE_M_LIST": ",".join(str(m) for m in m_list)}
            got, _ = launch(os.path.join(BIN, f"prefill_{arm}"), env)
            for m, row in got.items():
                rows.setdefault((block, m), {})[arm] = row["fnv"]
    print(f"{'block':>6} {'m':>5} {'before==after':>14} {'poison moves':>13}")
    ok = True
    for (block, m), got in sorted(rows.items()):
        identical = got["before"] == got["after"]
        moved = got["poison"] != got["after"]
        # Block 32 at m = 1 is the one row that is *supposed* not to move.
        expect_move = not (block == 32 and m == 1)
        ok = ok and identical and moved == expect_move
        note = "" if moved == expect_move else "  <-- UNEXPECTED"
        print(f"{block:>6} {m:>5} {str(identical):>14} {str(moved):>13}{note}")
    print("route proof:", "PASS" if ok else "FAIL")
    return {"rows": {f"{b}/{m}": g for (b, m), g in rows.items()}, "pass": ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=31)
    ap.add_argument("--decode-rounds", type=int, default=15)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--shape", default="small")
    ap.add_argument("--m-list", default="8,16,32,64,128,256,512")
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--skip-decode", action="store_true")
    ap.add_argument("--route-proof", action="store_true",
                    help="checksum-only; proves each row's route without timing it")
    ap.add_argument("--out", default=os.path.join(BIN, "modulo_matrix.json"))
    args = ap.parse_args()

    aa = os.path.join(BIN, "prefill_aa")
    if not os.path.exists(aa):
        raise SystemExit(
            f"missing {aa} -- run `int4_modulo_arms.sh`, then "
            "`cp prefill_after prefill_aa` and `cp decode_after decode_aa` in that "
            "directory. The null arm is a separate file on purpose: it is then a "
            "genuinely independent launch that pays every per-launch cost the real "
            "arms pay."
        )
    for a in ("before", "after", "aa"):
        for kind in ("prefill", "decode"):
            p = os.path.join(BIN, f"{kind}_{a}")
            if not os.path.exists(p):
                raise SystemExit(f"missing {p}")

    m_list = [int(v) for v in args.m_list.split(",")]
    if args.route_proof:
        # No lock: checksums do not depend on who else is on the machine, and
        # holding the whole host to compute one would be antisocial.
        proof = route_proof(m_list, args.shape)
        with open(args.out, "w") as fh:
            json.dump(proof, fh, indent=2)
        print(f"wrote {args.out}")
        raise SystemExit(0 if proof["pass"] else 1)

    result = {"pin": PIN, "cpu_eff_floor": CPU_EFF_FLOOR, "rounds": args.rounds}
    with H.HostLock(
        "roy",
        f"dequant_panel_avx2 modulo matrix: block{args.block} m={args.m_list}"
        f" + block16 decode ({args.rounds} rounds)",
    ):
        result["lock"] = H.lock_provenance()
        print(f"prefill matrix block={args.block} shape={args.shape}", flush=True)
        table, disc = prefill_matrix(args.rounds, args.block, args.shape, m_list)
        result["prefill"] = table
        result["prefill_discarded_launches"] = disc
        if not args.skip_decode:
            print("decode matrix block=16", flush=True)
            dec, ddisc = decode_matrix(args.decode_rounds, 16, args.tokens)
            result["decode"] = dec
            result["decode_discarded_launches"] = ddisc
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)

    print(f"\npin=cpu{PIN}  cpu_eff floor={CPU_EFF_FLOOR}  "
          f"discarded={result['prefill_discarded_launches']}")
    print(f"\nprefill block {args.block} ({args.shape}), {args.rounds} independent launches per arm")
    print(f"{'m':>5} {'before ms':>10} {'after ms':>10} {'speedup':>8} {'95% CI':>18} "
          f"{'verdict':>8} {'A/A':>7} {'A/A 95% CI':>18} {'A/A ok':>7} {'bit-id':>7}")
    for row in table:
        print(f"{row['m']:>5} {row['before_median_ms']:>10.3f} {row['after_median_ms']:>10.3f} "
              f"{row['speedup']:>8.4f} [{row['ci_lo']:.4f}, {row['ci_hi']:.4f}] "
              f"{row['verdict']:>8} {row['aa_speedup']:>7.4f} "
              f"[{row['aa_ci_lo']:.4f}, {row['aa_ci_hi']:.4f}] "
              f"{str(row['aa_brackets_unity']):>7} {str(row['bit_identical']):>7}")
    print(f"\nIntervals: percentile bootstrap over launches, "
          f"{BOOTSTRAP_RESAMPLES} resamples, seed {BOOTSTRAP_SEED}.")
    if not all(r["aa_brackets_unity"] for r in table):
        print("WARNING: an A/A interval excludes 1.000 -- the instrument is biased "
              "at that row and its verdict is not readable.")
    if not all(r["bit_identical"] for r in table):
        print("WARNING: an arm produced different output bytes -- this A/B is not "
              "measuring an exact transformation.")
    if not args.skip_decode and "error" not in result["decode"]:
        d = result["decode"]
        print(f"\ndecode block 16, {args.decode_rounds} independent launches per arm")
        print(f"  before {d['before_median_ms']:.3f}  after {d['after_median_ms']:.3f}  "
              f"speedup {d['speedup']:.4f} [{d['ci_lo']:.4f}, {d['ci_hi']:.4f}] {d['verdict']}  "
              f"A/A null {d['aa_null_pct']:.2f}% [{d['aa_ci_lo']:.4f}, {d['aa_ci_hi']:.4f}]  "
              f"bit-identical {d['bit_identical']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
