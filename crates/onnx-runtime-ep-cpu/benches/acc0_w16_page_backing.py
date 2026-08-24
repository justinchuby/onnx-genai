#!/usr/bin/env python3
"""Is the width-16 A/A null a page-backing lottery?

The open blocker on the acc0 width-16 work is not a kernel deficiency, it is
that the A/A null at that width is +-21.5%, which is wide enough to refuse a
+23% candidate (`DEFAULT_STEAL_TILES_PER_WORKER = 2`). Everything measured
about the slow arm says **per-process startup state**: it is position
independent, it is internally consistent to 0.5% while running ~1.3x slow, and
it is decided before the first token. Two candidate mechanisms have now been
tested and neither survived -- dispatcher CPU placement failed its bar at
1.0953, and worker placement is one-per-physical-core in every configuration
(`decode_placement_census.sh`).

This tests a third, which fits the shape better than either: **transparent
hugepage backing of the weight arena**.

    $ cat /sys/kernel/mm/transparent_hugepage/enabled
    [always] madvise never
    $ cat /sys/kernel/mm/transparent_hugepage/defrag
    always defer defer+madvise [madvise] never

`always` + `defrag=madvise` means anonymous memory gets 2 MB pages
*opportunistically at fault time*, and when the buddy allocator cannot hand
over a free 2 MB block immediately it silently falls back to 4 KB -- no error,
no log, and no second chance, because the weights are faulted once at load and
never again. Whether a launch wins that lottery is therefore decided by the
host's free-list fragmentation at the instant it starts, is fixed for the life
of the process, and is invisible to every timing instrument.

It also predicts the width dependence. Sixteen workers streaming the packed
weights concurrently share one L2 TLB per core and the page-walk caches; at
4 KB a 22 MB projection chain needs ~5600 PTEs against ~11 at 2 MB. Eight
workers on half the weights per pass press that far less hard.

Pre-registered before the first launch
--------------------------------------
Reconnaissance first, because the hypothesis is cheap to kill: if `thp_frac`
does not *vary* across launches there is no lottery and the question is closed
without a correlation. Only if it varies is the rule below scored.

    ACCEPT "the width-16 A/A null is page backing" iff
      (1) n_trusted >= 8, and
      (2) Spearman rho between thp_frac and ms_token <= -0.70
          (more hugepage backing, faster), and
      (3) the observed thp_frac range spans >= 0.20, so the correlation is
          measured over a real spread rather than fitted to noise.

    REJECT otherwise. A rho that is small, positive, or measured across a
    degenerate range is a rejection, not "inconclusive" -- the point of a
    per-launch categorical read is that it cannot be underpowered in the way a
    timing can.

`thp_frac` is `AnonHugePages / RssAnon` from `/proc/<pid>/smaps_rollup`,
sampled while the steady-state loop runs. Both are categorical kernel
accounting, not timings, so the sample is not corrupted by a busy host -- but
the `ms_token` it is correlated against is, so this still runs under the
hostlock.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import acc0_gap_matrix as H  # noqa: E402

MODEL, BLOCK, ACC, SESSIONS = "llama", 32, 0, 1
WIDTH = 16
MIN_TRUSTED = 8
RHO_BAR = -0.70
MIN_RANGE = 0.20


def smaps_rollup(pid):
    """`(AnonHugePages_kB, Anonymous_kB)` for a live pid, or None once it exits.

    The denominator is `Anonymous`, not `RssAnon`: `smaps_rollup` on this kernel
    does not carry an `RssAnon` field at all (it is a `/proc/<pid>/status`
    field), and a probe that required both silently discarded every launch
    while looking exactly like a workload failure.
    """
    try:
        with open(f"/proc/{pid}/smaps_rollup") as fh:
            fields = {}
            for line in fh:
                key, _, rest = line.partition(":")
                if key in ("AnonHugePages", "Anonymous"):
                    fields[key] = int(rest.strip().split()[0])
            if "AnonHugePages" in fields and "Anonymous" in fields:
                return fields["AnonHugePages"], fields["Anonymous"]
    except (OSError, ValueError, IndexError):
        pass
    return None


def launch(binary, tokens, reps):
    """One launch: run the width-16 decode arm and sample its page backing.

    Sampling is repeated and the *maximum* RssAnon sample is kept, because the
    quantity of interest is the backing of the fully-resident weight set: an
    early sample catches the arena mid-fault and reports a fraction of a
    process that has not finished loading.
    """
    env = dict(os.environ)
    env.update({
        "CARGO_INCREMENTAL": "0",
        "PROBE_MODEL": MODEL, "PROBE_BLOCK": str(BLOCK),
        "PROBE_ACCURACY": str(ACC), "PROBE_SESSIONS": str(SESSIONS),
        "PROBE_TOKENS": str(tokens), "PROBE_REPS": str(reps),
        "ONNX_GENAI_CPU_DECODE_THREADS": str(WIDTH),
    })
    # `cwd=H.HERE` is not cosmetic: the bench resolves its model fixtures
    # relative to the benches directory, and run from anywhere else it exits
    # before printing a steady row -- which this harness would otherwise
    # silently score as a discard.
    proc = subprocess.Popen(
        ["taskset", "-c", H.PIN, os.path.abspath(binary)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env=env, cwd=H.HERE)
    best = None
    t0 = time.monotonic()
    while proc.poll() is None and time.monotonic() - t0 < 600:
        sample = smaps_rollup(proc.pid)
        if sample and (best is None or sample[1] > best[1]):
            best = sample
        time.sleep(0.5)
    out = proc.stdout.read() if proc.stdout else ""
    proc.wait()

    steady, width_line = None, None
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("steady"):
            f = s.split()
            steady = {"ms_token": float(f[1]), "p90": float(f[2]),
                      "tps": float(f[3]), "spread": float(f[4])}
        if s.startswith("decode_width"):
            width_line = s
    if steady is None or best is None:
        sys.stderr.write(out[-2000:])
        return None
    if width_line is None or not width_line.endswith("as_requested"):
        sys.stderr.write(f"vacuous width: {width_line}\n")
        return None
    hugepages_kb, rss_anon_kb = best
    return {
        "ms_token": steady["ms_token"], "tps": steady["tps"],
        "spread": steady["spread"],
        "anon_huge_mb": hugepages_kb / 1024.0,
        "anon_mb": rss_anon_kb / 1024.0,
        "thp_frac": hugepages_kb / rss_anon_kb if rss_anon_kb else 0.0,
    }


def spearman(xs, ys):
    """Rank correlation, ties averaged. Returns None for a degenerate input."""
    def rank(vs):
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        out = [0.0] * len(vs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            shared = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = shared
            i = j + 1
        return out

    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    if n < 3 or len(set(rx)) < 2 or len(set(ry)) < 2:
        return None
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    return num / (dx * dy) if dx and dy else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--launches", type=int, default=10)
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rows = []
    for i in range(args.launches):
        row = launch(args.binary, args.tokens, args.reps)
        if row is None:
            print(f"launch {i + 1:2d}  DISCARDED (no steady row / vacuous width)")
            continue
        rows.append(row)
        print(f"launch {i + 1:2d}  ms_token={row['ms_token']:7.3f}  "
              f"spread={row['spread']:5.1f}%  "
              f"anon_huge={row['anon_huge_mb']:8.1f} MB  "
              f"anon={row['anon_mb']:8.1f} MB  "
              f"thp_frac={row['thp_frac']:.3f}")
        sys.stdout.flush()

    if not rows:
        print("no trusted launches")
        return 1

    fracs = [r["thp_frac"] for r in rows]
    times = [r["ms_token"] for r in rows]
    lo, hi = min(fracs), max(fracs)
    rho = spearman(fracs, times)

    print()
    print(f"n_trusted     : {len(rows)} (bar {MIN_TRUSTED})")
    print(f"thp_frac      : {lo:.3f} - {hi:.3f}  range {hi - lo:.3f} "
          f"(bar {MIN_RANGE})")
    print(f"ms_token      : {min(times):.3f} - {max(times):.3f}  "
          f"ratio {max(times) / min(times):.4f}")
    print(f"spearman rho  : {'n/a' if rho is None else f'{rho:+.4f}'} "
          f"(bar {RHO_BAR})")

    if hi - lo < MIN_RANGE:
        verdict = ("REJECT -- thp_frac does not vary across launches, so there "
                   "is no page-backing lottery to explain the null")
    elif len(rows) < MIN_TRUSTED:
        verdict = f"REPORT NOTHING -- n_trusted {len(rows)} < {MIN_TRUSTED}"
    elif rho is None or rho > RHO_BAR:
        verdict = "REJECT -- correlation does not clear the pre-registered bar"
    else:
        verdict = "ACCEPT -- page backing tracks the width-16 A/A null"
    print(f"VERDICT       : {verdict}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"rows": rows, "rho": rho, "range": hi - lo,
                       "verdict": verdict}, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
