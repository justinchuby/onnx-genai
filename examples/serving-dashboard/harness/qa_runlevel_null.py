#!/usr/bin/env python3
"""Run-level NULL test for the AC33 <2% acceptance protocol.

Validates perf-baseline.md 6c.5. One binary, one server, ten consecutive RUNS
of 15x512-token generations each. Runs are labelled A/B/A/B/... purely as a
sham assignment -- there is NO real difference between arms, so any apparent
"regression" is pure protocol error.

Measures the variance component that actually governs the <2% criterion:
dispersion of RUN MEANS, not dispersion of samples within a run.

Outputs raw per-sample and per-run data. Stdlib only. Touches no repo files.
"""

import json
import os
import statistics as st
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qa_baseline_harness as H

H.MAX_TOKENS = 512

RUNS = 10
ITERS_PER_RUN = 15
WARMUP = 3
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw",
                   "qa-runlevel-null.json")


def one_run(run_idx):
    samples = []
    for i in range(ITERS_PER_RUN):
        r = H.stream_once(max_tokens=512)
        samples.append({
            "i": i,
            "decode_tps": r["decode_tps"],
            "ttft_s": r["ttft_s"],
            "n_tokens": r["n_tokens"],
            "loadavg": H.loadavg(),
            "t_wall": time.time(),
        })
        print("  run %d sample %2d: %.3f tok/s (n=%d, load %.2f)"
              % (run_idx, i, r["decode_tps"], r["n_tokens"], samples[-1]["loadavg"]),
              flush=True)
    return samples


def main():
    print("warmup x%d @512 tokens..." % WARMUP, flush=True)
    for _ in range(WARMUP):
        H.stream_once(max_tokens=512)

    runs = []
    t0 = time.time()
    for k in range(RUNS):
        arm = "A" if k % 2 == 0 else "B"
        print("=== RUN %d (sham arm %s) ===" % (k, arm), flush=True)
        s = one_run(k)
        vals = [x["decode_tps"] for x in s]
        runs.append({
            "run": k, "arm": arm, "samples": s,
            "mean": st.mean(vals), "median": st.median(vals),
            "cv_pct": 100 * st.stdev(vals) / st.mean(vals),
        })
        print("  -> run %d mean %.3f  median %.3f  CV %.2f%%  (elapsed %.1f min)"
              % (k, runs[-1]["mean"], runs[-1]["median"], runs[-1]["cv_pct"],
                 (time.time() - t0) / 60), flush=True)
        with open(OUT, "w") as f:
            json.dump({"runs": runs, "config": {
                "runs": RUNS, "iters_per_run": ITERS_PER_RUN,
                "max_tokens": 512, "warmup": WARMUP}}, f, indent=1)

    means = [r["mean"] for r in runs]
    a = [r["mean"] for r in runs if r["arm"] == "A"]
    b = [r["mean"] for r in runs if r["arm"] == "B"]
    within = st.mean([r["cv_pct"] for r in runs])
    between = 100 * st.stdev(means) / st.mean(means)

    print("\n===== RESULT =====")
    print("run means: " + ", ".join("%.3f" % m for m in means))
    print("mean within-run CV : %.2f%%" % within)
    print("BETWEEN-run CV     : %.2f%%" % between)
    print("sham A mean %.3f   sham B mean %.3f   delta %+.2f%%"
          % (st.mean(a), st.mean(b), (st.mean(b) - st.mean(a)) / st.mean(a) * 100))
    print("max pairwise run-to-run delta: %+.2f%%"
          % ((max(means) - min(means)) / min(means) * 100))

    summary = {
        "run_means": means,
        "mean_within_run_cv_pct": within,
        "between_run_cv_pct": between,
        "sham_a_mean": st.mean(a), "sham_b_mean": st.mean(b),
        "sham_delta_pct": (st.mean(b) - st.mean(a)) / st.mean(a) * 100,
        "max_pairwise_delta_pct": (max(means) - min(means)) / min(means) * 100,
    }
    with open(OUT, "w") as f:
        json.dump({"runs": runs, "summary": summary, "config": {
            "runs": RUNS, "iters_per_run": ITERS_PER_RUN,
            "max_tokens": 512, "warmup": WARMUP}}, f, indent=1)
    print("wrote " + OUT)


if __name__ == "__main__":
    main()
