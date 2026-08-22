#!/usr/bin/env python3
"""acc0 single-session gap: unified-definition matrix with per-cell A/A.

Every number here uses ONE definition of `tokens_s_total`
(see the table in int4_decode_loop_ab.rs):
    numerator   = sessions * tokens
    denominator = wall from barrier release to last join
    warmup      = before the barrier, never inside the clock
    over reps   = median

Adds two things the previous matrix did not have:

  * per-cell A/A -- the *same* arm run twice, interleaved, so a cell's own
    noise floor is measured rather than assumed. A cell whose A/A deviates
    from 1.00 by more than its A/B effect cannot support a conclusion.
  * achieved bandwidth -- tokens/s converted to GB/s using the exact packed
    weight footprint. Two arms at the same GB/s are both bandwidth-bound and
    the difference is elsewhere; an arm at half the bandwidth of the other is
    not bandwidth-bound at all, which is a different bug.

The outlier rule, written down before it was applied
----------------------------------------------------
A cell may be marked suspect only by a test applied *symmetrically to both
arms*. Concretely: run each arm >=5 times in independent processes and report
the distribution. A cell is quoted normally if both arms are unimodal; it is
reported as a range, never as a point, if either arm is multimodal.

Applying that rule to `qwen t=16 s=1 acc=0`, the cell previously published as
0.436x, six independent runs per arm:

    native  190.9 195.6 197.8 200.3 201.5 220.6   unimodal, +-8%
    ORT     218.3 229.8 246.2 396.9 414.9 427.7   two clusters, 1.79x apart

What that does and does not establish
-------------------------------------
It establishes that **0.436x was not a measurable quantity**: it came from
taking `max` over repetitions of ORT's fast cluster while the native arm ran
single-shot with its own warmup inside its clock. Under one definition the
same cell reads 0.70x, and the ratio cannot be quoted more precisely than the
range 0.48x-0.86x.

It does **not** yet establish that ORT is intrinsically bimodal here, and the
temptation to claim that should be resisted. The three slow ORT runs reported
intra-run spreads of 67.9%, 4.7% and 12.0% against 1.4%, 1.1% and 0.7% for the
three fast ones, and the host was later found to have two other agents running
heavy jobs on it. Elevated intra-run spread in exactly the slow cluster is the
signature of external contention, not of an internal bimodality -- a genuinely
bimodal implementation would be internally stable in *both* modes. So the
parsimonious reading is that the slow cluster is contention, which makes this
a measurement-environment finding rather than a fact about ORT.

Separating the two requires re-running both arms interleaved on a quiet host,
which is what `wait_quiet` and `competing_load` below exist to guarantee. Until
that has been done this cell has no published ratio.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = HERE
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))

PROJ = {
    "llama": [(4096, 6144), (4096, 4096), (4096, 14336), (4096, 14336), (14336, 4096)],
    "qwen": [(3584, 4608), (3584, 3584), (3584, 18944), (3584, 18944), (18944, 3584)],
}

# Even CPUs are distinct physical cores on this SMT host (siblings adjacent).
PIN = ",".join(str(c) for c in range(0, 32, 2))


def weight_bytes(model, block):
    """Packed nibbles + f32 scales for one full projection chain, in bytes.

    This is the quantity a decode token must stream: the weights do not fit in
    any cache, so every token re-reads all of them. Activations are ~5 x 4 KB
    and are noise beside it.
    """
    total = 0
    for k, n in PROJ[model]:
        blocks = (k + block - 1) // block
        total += n * blocks * (block // 2)  # packed 0.5 B/weight
        total += n * blocks * 4  # f32 scale per block
    return total


def sh(cmd, env=None, timeout=1800):
    e = dict(os.environ)
    e["CARGO_INCREMENTAL"] = "0"
    if env:
        e.update({k: str(v) for k, v in env.items()})
    return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          env=e, timeout=timeout, cwd=HERE)


def native(binary, model, block, acc, threads, sessions, tokens, reps, extra=None):
    env = {
        "PROBE_MODEL": model, "PROBE_BLOCK": block, "PROBE_ACCURACY": acc,
        "PROBE_SESSIONS": sessions, "PROBE_TOKENS": tokens, "PROBE_REPS": reps,
        "ONNX_GENAI_CPU_DECODE_THREADS": threads,
    }
    if extra:
        env.update(extra)
    r = sh(f"taskset -c {PIN} {binary}", env)
    for line in r.stdout.splitlines():
        if line.strip().startswith("steady"):
            f = line.split()
            return {"ms_token": float(f[1]), "p90": float(f[2]),
                    "tps": float(f[3]), "spread": float(f[4])}
    sys.stderr.write(r.stdout + r.stderr)
    raise RuntimeError("native arm produced no steady row")


def ort(model, block, acc, threads, sessions, tokens, reps):
    cmd = (f"taskset -c {PIN} python3 ort_matmulnbits_baseline.py "
           f"--model {model} --block {block} --accuracy {acc} --threads {threads} "
           f"--tokens {tokens} --reps {reps} --sessions {sessions}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=BENCH,
                       timeout=3600)
    m = re.search(r"tokens_s_total=([\d.]+).*spread_pct=([\d.]+)", r.stdout)
    if not m:
        sys.stderr.write(r.stdout + r.stderr)
        raise RuntimeError("ORT arm produced no throughput")
    return {"tps": float(m.group(1)), "spread": float(m.group(2))}


def competing_load():
    """Heavy processes other than us, as `(pid, pcpu, command)`.

    Load average alone is not enough. It is a 1-minute exponential average, so
    it both lags a benchmark that just started and stays elevated long after
    one has finished. This looks directly for what actually corrupts a run:
    another CPU-saturating process. That is not hypothetical -- a full matrix
    was invalidated when a second agent ran *this same decode benchmark* and a
    `cargo test` on the same 16-core host, moving one cell 8.6x (197 -> 23
    tokens/s) while each individual run still reported a reassuring <6%
    intra-run spread. A tight spread means the contention was *steady*, not
    that the host was idle.
    """
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid,pcpu,args", "--sort=-pcpu"],
            capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return []
    mine = {str(os.getpid()), str(os.getppid())}
    busy = []
    for line in out.splitlines()[1:]:
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, pcpu, cmd = parts[0], float(parts[1]), parts[2]
        if pid in mine or pcpu < 150.0:
            continue
        if "ps -eo" in cmd:
            continue
        busy.append((pid, pcpu, cmd[:70]))
    return busy


def wait_quiet(threshold=3.0, limit=900):
    """Never start a measurement on a loaded host.

    Returns `(loadavg, competitors)`. Crucially it returns the competitors it
    could not wait out rather than pretending the host was quiet: a caller that
    measures anyway must record that the cell is untrusted. Silently proceeding
    after a timeout is how a contended number gets published.
    """
    start = time.time()
    while time.time() - start < limit:
        load = os.getloadavg()[0]
        busy = competing_load()
        if load <= threshold and not busy:
            return load, []
        time.sleep(20)
    return os.getloadavg()[0], competing_load()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--models", default="llama,qwen")
    ap.add_argument("--threads", default="16")
    ap.add_argument("--sessions", default="1,2,4")
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--acc", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="acc0_gap.json")
    ap.add_argument("--aa", action="store_true", help="per-cell interleaved A/A")
    args = ap.parse_args()

    rows = []
    hdr = (f"{'model':>6} {'t':>3} {'s':>2} {'nat_tps':>9} {'nat_sp%':>8} "
           f"{'ort_tps':>9} {'ort_sp%':>8} {'ratio':>7} {'nat_GB/s':>9} "
           f"{'ort_GB/s':>9} {'A/A':>6}")
    print(hdr)
    print("-" * len(hdr))

    for model in args.models.split(","):
        wb = weight_bytes(model, args.block)
        for t in [int(x) for x in args.threads.split(",")]:
            for s in [int(x) for x in args.sessions.split(",")]:
                load, busy = wait_quiet()
                if busy:
                    sys.stderr.write(
                        f"WARNING {model} t={t} s={s}: measuring against "
                        f"{len(busy)} competing process(es); cell is UNTRUSTED\n")
                    for pid, pcpu, cmd in busy[:3]:
                        sys.stderr.write(f"    pid={pid} cpu={pcpu:.0f}% {cmd}\n")
                # Interleaved: native, ORT, native again (the A/A partner).
                a1 = native(args.binary, model, args.block, args.acc, t, s,
                            args.tokens, args.reps)
                o = ort(model, args.block, args.acc, t, s, args.tokens, args.reps)
                aa = ""
                if args.aa:
                    a2 = native(args.binary, model, args.block, args.acc, t, s,
                                args.tokens, args.reps)
                    aa = f"{a2['tps'] / a1['tps']:.3f}"
                nat_bw = a1["tps"] * wb / 1e9
                ort_bw = o["tps"] * wb / 1e9
                row = {"model": model, "threads": t, "sessions": s,
                       "native": a1, "ort": o, "ratio": a1["tps"] / o["tps"],
                       "native_gbs": nat_bw, "ort_gbs": ort_bw,
                       "weight_bytes": wb, "loadavg_at_start": load,
                       "trusted": not busy,
                       "competitors": [c[2] for c in busy],
                       "aa": float(aa) if aa else None}
                rows.append(row)
                flag = "" if not busy else "  !CONTENDED"
                print(f"{model:>6} {t:>3} {s:>2} {a1['tps']:>9.1f} "
                      f"{a1['spread']:>8.1f} {o['tps']:>9.1f} {o['spread']:>8.1f} "
                      f"{row['ratio']:>7.3f} {nat_bw:>9.1f} {ort_bw:>9.1f} {aa:>6}"
                      f"{flag}")
                sys.stdout.flush()
                with open(os.path.join(HERE, args.out), "w") as f:
                    json.dump(rows, f, indent=1)

    print()
    print("weight bytes per token per session:",
          {m: f"{weight_bytes(m, args.block) / 1e6:.1f} MB" for m in args.models.split(",")})


if __name__ == "__main__":
    main()
