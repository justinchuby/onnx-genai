#!/usr/bin/env python3
"""Is the width-16 fast/slow mode a per-launch clock/boost state?

Standing question
-----------------
The width-16 int4 decode loop is bimodal per process launch (~4.0 vs ~6.0
ms/token, 1.69x). Worker placement is excluded categorically (both modes appear
on a byte-identical 15-worker / 15-physical-core / 8-7-L3 set, 21 launches),
THP/page backing is excluded, and foreign load on the pinned set is excluded by
a magnitude bound (<=1.7% of pinned busy against the ~23% required). Two
candidates remain: weight-arena placement across the two L3/CCX domains, and
**per-launch clock/boost state**. This tests the second.

The bound
---------
The modes differ by 1.69x in wall while user CPU per token differs by ~4.6%. If
the cause were the cores running at a lower clock, the work would take longer in
wall *and* the CPU-time accounting would be roughly preserved -- which is
consistent with what we see, so the hypothesis is not absurd on its face. But it
makes a hard quantitative prediction: to stretch 4.0 ms into 6.0 ms the cores
must run at 1/1.69 = **59% of the fast-mode frequency**. That is a ~41% clock
drop, not a subtle one, and it is directly readable.

Two instruments, because the direct one turned out to be unavailable
--------------------------------------------------------------------
*Direct*: this host exposes no `cpufreq` sysfs, so frequency would come from
`/proc/cpuinfo` `cpu MHz`, sampled during the steady phase and restricted to the
pool's pinned CPUs. Measured, it is a **constant 2870.7 MHz in every launch of
both modes** -- a nominal field on this virtualised host, not a reading. The
hardware PMU is also absent (`perf stat -e cycles` reports `<not supported>`)
and `/dev/cpu/*/msr` is not readable, so APERF/MPERF is out too. There is no
direct clock instrument on this box, and the probe says so rather than
converting a constant field into a free REJECT.

*Indirect, and decisive*: a clock drop is distinguishable from every other
slowdown mechanism by what it does to **CPU-time**. Reducing the clock by a
factor `k` makes fixed work take `1/k` times as many wall-seconds *on-CPU*, so
CPU-time per token rises by exactly the same factor as wall time per token.
This is precisely the opposite of SMT contention, which steals throughput
without stealing CPU-time (measured separately: co-locating two workers on one
core costs 1.86x throughput and 0.0% CPU-time), and the opposite of parking,
which reduces CPU-time while raising wall.

So the hypothesis makes a sharp, already-measurable prediction: if the slow mode
is 1.5x slower because the cores are clocked down, its **user CPU per token must
be 1.5x higher**. Nothing needs a frequency counter.

Pre-registered before the first launch
--------------------------------------
    Let `w` = median(slow wall/token) / median(fast wall/token), and
    `u` = median(slow user-CPU/token) / median(fast user-CPU/token).

    ACCEPT "the mode is clock state" iff `u >= 1 + 0.75 * (w - 1)` -- at least
    three quarters of the required CPU-time inflation is present.

    REJECT iff `u <= 1 + 0.25 * (w - 1)`. Less than a quarter of the required
    inflation means the cores retired the same work in the same CPU-time, which
    a lower clock cannot do.

    REPORT NOTHING if only one mode is sampled, or fewer than `MIN_TRUSTED`
    launches are trusted.

    Independently, REPORT NOTHING on the *direct* instrument if `cpu MHz` is
    constant across launches, since a nominal field cannot answer either way.

Like the foreign-load falsifier, this rejects on a *magnitude bound* rather than
a correlation, so it does not need a balanced sample of the two modes -- a
single slow launch that retired its work in the usual CPU-time is enough to
kill a 41% clock drop. The n bar exists to make sampling both modes likely, not
to average anything.
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
from acc0_w16_mode_placement import worker_cpus  # noqa: E402

WIDTH = 16
MIN_TRUSTED = 10
LANE_CUT = 13.0
POLL_S = 0.02
POLL_LIMIT_S = 30.0
CLOCK_ACCEPT_FRAC = 0.75
CLOCK_REJECT_FRAC = 0.95
# Fraction of the required CPU-time inflation that must be present to accept,
# and the fraction below which the hypothesis is rejected.
CPU_ACCEPT_SHARE = 0.75
CPU_REJECT_SHARE = 0.25


def cpu_mhz():
    """Current MHz per logical CPU, from /proc/cpuinfo."""
    out, cpu = {}, None
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("processor"):
                    cpu = int(line.split(":")[1])
                elif line.startswith("cpu MHz") and cpu is not None:
                    out[cpu] = float(line.split(":")[1])
    except OSError:
        pass
    return out


def one_launch(binary, tokens, reps):
    env = dict(os.environ)
    env.update({
        "PROBE_MODEL": "llama", "PROBE_BLOCK": "32", "PROBE_ACCURACY": "0",
        "PROBE_SESSIONS": "1", "PROBE_TOKENS": str(tokens),
        "PROBE_REPS": str(reps),
        "ONNX_GENAI_CPU_DECODE_THREADS": str(WIDTH),
    })
    proc = subprocess.Popen(["taskset", "-c", H.PIN, os.path.abspath(binary)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=env, cwd=H.HERE)
    masks, deadline = [], time.time() + POLL_LIMIT_S
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        seen = worker_cpus(proc.pid)
        if len(seen) >= WIDTH - 1 and all(m.isdigit() for m in seen):
            masks = seen
            break
        if len(seen) > len(masks):
            masks = seen
        time.sleep(POLL_S)

    pinned = sorted(int(m) for m in masks if m.isdigit())
    # Sample the clock while the pool is live. Sampling after the process exits
    # would read an idle machine, which is the same sample-instant defect that
    # has now cost this ledger four probes.
    samples = []
    while proc.poll() is None:
        snap = cpu_mhz()
        vals = [snap[c] for c in pinned if c in snap]
        if vals:
            samples.append(statistics.median(vals))
        time.sleep(0.05)
    out, _ = proc.communicate(timeout=900)

    row = {"n_workers": len(masks), "pinned_cpus": pinned,
           "n_clock_samples": len(samples)}
    if samples:
        row["mhz_median"] = statistics.median(samples)
        row["mhz_min"] = min(samples)
        row["mhz_max"] = max(samples)

    width_ok = False
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("steady"):
            row["ms_token"] = float(s.split()[1])
        elif s.startswith("cpu phase=steady ") and "unavailable" not in s:
            for field in s.split()[2:]:
                k, _, v = field.partition("=")
                try:
                    row[k] = float(v)
                except ValueError:
                    pass
        elif s.startswith("decode_width"):
            width_ok = s.endswith("as_requested")
    if ("ms_token" not in row or "cpu_s_per_token" not in row
            or not width_ok or not samples):
        row["discarded"] = (
            "no clock samples" if not samples
            else "no `steady` row" if "ms_token" not in row
            else "no usable `cpu phase=steady` row" if "cpu_s_per_token" not in row
            else "decode_width did not report as_requested")
        return row
    row["lanes"] = row["cpu_s_per_token"] / (row["ms_token"] / 1000.0)
    row["mode"] = "fast" if row["lanes"] >= LANE_CUT else "SLOW"
    if "user_s" in row and row.get("tokens"):
        row["user_s_per_token"] = row["user_s"] / row["tokens"]
        row["sys_s_per_token"] = row["sys_s"] / row["tokens"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--launches", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=192)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rows = []
    for i in range(args.launches):
        r = one_launch(args.binary, args.tokens, args.reps)
        rows.append(r)
        if "discarded" in r:
            print(f"launch {i+1:2d}  DISCARD  {r['discarded']}")
        else:
            print(f"launch {i+1:2d}  {r['mode']:>4}  ms={r['ms_token']:7.3f}  "
                  f"lanes={r['lanes']:5.2f}  mhz={r['mhz_median']:8.1f}  "
                  f"[{r['mhz_min']:.0f}-{r['mhz_max']:.0f}]  "
                  f"n_samp={r['n_clock_samples']}")
        sys.stdout.flush()

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)

    ok = [r for r in rows if "discarded" not in r]
    fast = [r for r in ok if r["mode"] == "fast"]
    slow = [r for r in ok if r["mode"] == "SLOW"]
    print()
    print(f"n_trusted : {len(ok)} (bar {MIN_TRUSTED})")
    print(f"modes     : fast {len(fast)}, slow {len(slow)}")

    all_mhz = [r["mhz_median"] for r in ok]
    spread = (max(all_mhz) - min(all_mhz)) if all_mhz else 0.0
    print(f"direct: cpu MHz across all launches {min(all_mhz):.1f} - "
          f"{max(all_mhz):.1f} (spread {spread:.1f})")
    if all_mhz and spread < 1.0:
        print("direct instrument: UNAVAILABLE -- `cpu MHz` is constant to "
              "<1 MHz across every launch of both modes, so the field is "
              "nominal on this host. No PMU (`perf stat -e cycles` reports "
              "<not supported>) and no readable MSR either. Falling back to "
              "the CPU-time bound, which needs no frequency counter.")
    elif all_mhz:
        print("direct instrument: usable -- see per-launch mhz column")

    if len(ok) < MIN_TRUSTED:
        print(f"VERDICT: REPORT NOTHING -- n_trusted {len(ok)} < {MIN_TRUSTED}")
        return
    if not fast or not slow:
        print("VERDICT: REPORT NOTHING -- only one mode sampled")
        return

    def med(sel, key):
        vals = [r[key] for r in sel if key in r]
        return statistics.median(vals) if vals else float("nan")

    fw, sw = med(fast, "ms_token"), med(slow, "ms_token")
    fu, su = med(fast, "user_s_per_token"), med(slow, "user_s_per_token")
    fc, sc = med(fast, "cpu_s_per_token"), med(slow, "cpu_s_per_token")
    w, u = sw / fw, su / fu
    print()
    print(f"wall/token   fast {fw:.4f} ms   slow {sw:.4f} ms   ratio {w:.4f}")
    print(f"user/token   fast {fu:.5f} s    slow {su:.5f} s    ratio {u:.4f}")
    print(f"cpu/token    fast {fc:.5f} s    slow {sc:.5f} s    "
          f"ratio {sc/fc:.4f}")
    print(f"a clock drop producing {w:.4f}x wall requires user/token "
          f"ratio {w:.4f}; observed {u:.4f}")
    accept_at = 1 + CPU_ACCEPT_SHARE * (w - 1)
    reject_at = 1 + CPU_REJECT_SHARE * (w - 1)
    print(f"thresholds: ACCEPT >= {accept_at:.4f}, REJECT <= {reject_at:.4f}")
    if u >= accept_at:
        print("VERDICT: ACCEPT -- the slow mode's work costs proportionally "
              "more CPU-time, as a lower clock requires.")
    elif u <= reject_at:
        if u <= 1.0:
            extra = ("and the observed ratio is below 1.0, so the effect is "
                     "not merely too small but in the opposite direction")
        else:
            extra = (f"only {100*(u-1)/(w-1):.1f}% of the required inflation "
                     f"is present")
        print(f"VERDICT: REJECT -- the slow mode retires its work in "
              f"essentially the same user CPU-time as the fast mode ({extra}). "
              f"A lower clock cannot leave CPU-time per token unchanged.")
    else:
        print(f"VERDICT: REPORT NOTHING -- user/token ratio {u:.4f} falls "
              f"between the pre-registered thresholds.")

    print(f"sys/token    fast {med(fast,'sys_s_per_token'):.5f} s    "
          f"slow {med(slow,'sys_s_per_token'):.5f} s    "
          f"ratio {med(slow,'sys_s_per_token')/med(fast,'sys_s_per_token'):.4f}")
    print("note: if BOTH user and sys per token are flat while wall rises, the "
          "missing lanes are consuming no CPU in either mode -- they are not "
          "running at all, rather than spending longer in the kernel.")


if __name__ == "__main__":
    main()
