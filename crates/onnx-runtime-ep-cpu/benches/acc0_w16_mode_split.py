#!/usr/bin/env python3
"""What is different about the slow mode of the width-16 A/A null?

The A/A null at width 16 is the binding constraint on the acc0 work: it is wide
enough (+-21.5% as previously measured) to refuse a +23% candidate. Three
mechanisms have now been tested and rejected -- dispatcher CPU placement
(1.0953 against a 1.10 bar), worker placement (one per physical core in every
configuration, `decode_placement_census.sh`), and transparent-hugepage backing
of the weight arena (`thp_aa_probe.py`: thp_frac range 0.104 across 12
launches, Spearman rho -0.19).

That last run also produced the sharpest description of the null so far. Twelve
identical launches split into two tight clusters:

    slow   5.903  5.926  5.944  5.953  5.965   (5 launches, 1.05% apart)
    fast   3.458  3.497  3.508  3.519  3.690  3.868   (+ 3.558)

A slow mode whose five members agree to one percent is not noise and not a
gradient -- it is a **discrete alternative configuration selected per process
launch**. Launch order does not predict it (slow launches were 1, 2, 4, 6, 7).

So this harness stops trying to guess the mechanism and instead reads every
counter the bench already emits, in both modes, and prints the difference. The
mode split is done on the data by the largest gap in sorted `ms_token`, and the
split is reported so it can be checked rather than trusted.

This is a **diagnostic, not a test**: it has no accept/reject rule because it is
not evaluating an intervention. Its output is a table of what differs, which is
the input to a hypothesis, not a verdict on one.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import acc0_gap_matrix as H  # noqa: E402

MODEL, BLOCK, ACC, SESSIONS = "llama", 32, 0, 1
WIDTH = 16


def launch(binary, tokens, reps):
    env = dict(os.environ)
    env.update({
        "CARGO_INCREMENTAL": "0",
        "PROBE_MODEL": MODEL, "PROBE_BLOCK": str(BLOCK),
        "PROBE_ACCURACY": str(ACC), "PROBE_SESSIONS": str(SESSIONS),
        "PROBE_TOKENS": str(tokens), "PROBE_REPS": str(reps),
        "ONNX_GENAI_CPU_DECODE_THREADS": str(WIDTH),
    })
    out = subprocess.run(["taskset", "-c", H.PIN, os.path.abspath(binary)],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, env=env, cwd=H.HERE, timeout=900).stdout

    row, workers, width_ok = {}, [], False
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("steady"):
            f = s.split()
            row["ms_token"] = float(f[1])
            row["spread"] = float(f[4])
        elif s.startswith("cpu phase=steady ") and "unavailable" not in s:
            for field in s.split()[2:]:
                k, _, v = field.partition("=")
                try:
                    row[k] = float(v)
                except ValueError:
                    pass
        elif s.startswith("worker phase=steady "):
            w = {}
            for field in s.split()[2:]:
                k, _, v = field.partition("=")
                try:
                    w[k] = float(v)
                except ValueError:
                    pass
            workers.append(w)
        elif s.startswith("dispatcher reserved_cpu="):
            for field in s.split()[1:-1]:
                k, _, v = field.partition("=")
                try:
                    row[f"disp_{k}"] = float(v)
                except ValueError:
                    pass
            row["disp_verdict"] = s.split()[-1]
        elif s.startswith("decode_width"):
            width_ok = s.endswith("as_requested")

    if "ms_token" not in row or not width_ok or not workers:
        return None
    row["n_workers"] = float(len(workers))
    row["worker_cpus"] = ",".join(str(int(w["cpu"])) for w in workers
                                  if "cpu" in w)
    for key in ("parks", "spin_hits", "last_arrivals", "wake_ns", "work_ns"):
        vals = [w.get(key, 0.0) for w in workers]
        row[f"w_{key}_sum"] = sum(vals)
        row[f"w_{key}_max"] = max(vals)
    # Straggler concentration: the share of last-arrivals held by the single
    # worst worker. A barrier that always waits on the same lane looks very
    # different from one whose straggler moves.
    arrivals = [w.get("last_arrivals", 0.0) for w in workers]
    total = sum(arrivals)
    row["arrival_top_share"] = (max(arrivals) / total) if total else 0.0
    return row


def split_modes(rows):
    """Two clusters, cut at the largest gap in sorted `ms_token`.

    Returns `(fast, slow, gap_ratio)`. `gap_ratio` is the cut's size relative to
    the largest *other* gap: a clean bimodal split has a dominant gap, and a
    ratio near 1.0 means the data is a gradient and the split is arbitrary.
    """
    ordered = sorted(rows, key=lambda r: r["ms_token"])
    if len(ordered) < 4:
        return ordered, [], 0.0
    gaps = [(ordered[i + 1]["ms_token"] - ordered[i]["ms_token"], i)
            for i in range(len(ordered) - 1)]
    gaps.sort(reverse=True)
    biggest, cut = gaps[0]
    runner_up = gaps[1][0] if len(gaps) > 1 else 0.0
    ratio = (biggest / runner_up) if runner_up > 0 else float("inf")
    return ordered[:cut + 1], ordered[cut + 1:], ratio


def summarize(name, rows, keys):
    if not rows:
        return {}
    return {k: statistics.median([r[k] for r in rows if k in r])
            for k in keys if any(k in r for r in rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--launches", type=int, default=12)
    ap.add_argument("--tokens", type=int, default=192)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rows = []
    for i in range(args.launches):
        r = launch(args.binary, args.tokens, args.reps)
        if r is None:
            print(f"launch {i + 1:2d}  DISCARDED")
            continue
        rows.append(r)
        print(f"launch {i + 1:2d}  ms_token={r['ms_token']:7.3f}  "
              f"spread={r['spread']:5.1f}%  "
              f"sys_frac={r.get('sys_frac', float('nan')):.3f}  "
              f"cpu_s/tok={r.get('cpu_s_per_token', float('nan')):.6f}  "
              f"parks={r['w_parks_sum']:8.0f}  "
              f"spin={r['w_spin_hits_sum']:8.0f}  "
              f"disp_cpu={r.get('disp_observed_cpu', -1):.0f}")
        sys.stdout.flush()

    if len(rows) < 4:
        print("too few trusted launches to split")
        return 1

    fast, slow, gap_ratio = split_modes(rows)
    keys = ["ms_token", "spread", "user_s", "sys_s", "cpu_s_per_token",
            "sys_frac", "tps_rep", "w_parks_sum", "w_spin_hits_sum",
            "w_last_arrivals_sum", "w_wake_ns_sum", "w_work_ns_sum",
            "arrival_top_share", "n_workers"]
    a, b = summarize("fast", fast, keys), summarize("slow", slow, keys)

    print()
    print(f"split: {len(fast)} fast / {len(slow)} slow, "
          f"cut gap is {gap_ratio:.1f}x the next largest "
          f"({'bimodal' if gap_ratio >= 3 else 'NOT clearly bimodal'})")
    print(f"{'metric':<22}{'fast':>14}{'slow':>14}{'slow/fast':>12}")
    for k in keys:
        if k not in a or k not in b:
            continue
        ratio = (b[k] / a[k]) if a[k] else float("nan")
        print(f"{k:<22}{a[k]:>14.6g}{b[k]:>14.6g}{ratio:>12.4f}")

    cpus_fast = {r["worker_cpus"] for r in fast}
    cpus_slow = {r["worker_cpus"] for r in slow}
    print()
    print(f"worker cpu sets  fast: {len(cpus_fast)} distinct, "
          f"slow: {len(cpus_slow)} distinct, "
          f"identical across modes: {cpus_fast == cpus_slow}")
    print("dispatcher cpus  fast: "
          f"{sorted({int(r.get('disp_observed_cpu', -1)) for r in fast})}  "
          f"slow: {sorted({int(r.get('disp_observed_cpu', -1)) for r in slow})}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"rows": rows, "fast": a, "slow": b,
                       "gap_ratio": gap_ratio}, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
