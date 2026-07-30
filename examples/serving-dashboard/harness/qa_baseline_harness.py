#!/usr/bin/env python3
"""AC33 clean-tree decode-throughput baseline harness.

Measures DECODE throughput only (prefill/TTFT excluded) by streaming SSE and
timestamping every token. Decode rate = (n_tokens - 1) / (t_last - t_first),
which deliberately excludes the prefill interval that ends at the first token.

Stdlib only. No repo files are touched.
"""

import http.client
import json
import os
import statistics
import subprocess
import sys
import threading
import time

HOST = "127.0.0.1"
PORT = 8123
MODEL = "qwen-scatter"

# Fixed workload knobs -- any later re-run MUST use these exact values.
PROMPT = "Write a detailed explanation of how a hash table works, including collision resolution."
MAX_TOKENS = 128
TEMPERATURE = 0.0
SINGLE_WARMUP = 3
SINGLE_ITERS = 15
CONC_WARMUP = 1
CONC_ROUNDS = 8
CONCURRENCY = 4


def loadavg():
    try:
        return os.getloadavg()[0]
    except OSError:
        return float("nan")


def stream_once(prompt=PROMPT, max_tokens=MAX_TOKENS):
    """One streaming request. Returns dict with per-token timing."""
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": TEMPERATURE,
            "stream": True,
        }
    )
    conn = http.client.HTTPConnection(HOST, PORT, timeout=600)
    t_submit = time.perf_counter()
    conn.request(
        "POST",
        "/v1/chat/completions",
        body=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:400]!r}")

    token_times = []
    finish_reason = None
    text_parts = []
    buf = b""
    while True:
        chunk = resp.read1(65536)
        if not chunk:
            break
        now = time.perf_counter()
        buf += chunk
        while b"\n\n" in buf:
            raw, buf = buf.split(b"\n\n", 1)
            for line in raw.split(b"\n"):
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    continue
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                for ch in obj.get("choices", []):
                    delta = ch.get("delta", {}) or {}
                    content = delta.get("content")
                    if content:
                        token_times.append(now)
                        text_parts.append(content)
                    if ch.get("finish_reason"):
                        finish_reason = ch["finish_reason"]
    conn.close()
    t_end = time.perf_counter()

    n = len(token_times)
    if n < 2:
        raise RuntimeError(f"only {n} token events received; cannot measure decode")

    ttft = token_times[0] - t_submit
    decode_span = token_times[-1] - token_times[0]
    decode_tps = (n - 1) / decode_span if decode_span > 0 else float("nan")
    return {
        "n_tokens": n,
        "ttft_s": ttft,
        "decode_span_s": decode_span,
        "decode_tps": decode_tps,
        "e2e_s": t_end - t_submit,
        "finish_reason": finish_reason,
        "text_head": "".join(text_parts)[:80],
    }


def summarize(values):
    vs = sorted(values)
    n = len(vs)

    def pct(p):
        if n == 1:
            return vs[0]
        idx = (n - 1) * p
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        return vs[lo] + (vs[hi] - vs[lo]) * (idx - lo)

    return {
        "n": n,
        "min": vs[0],
        "p25": pct(0.25),
        "median": statistics.median(vs),
        "p75": pct(0.75),
        "max": vs[-1],
        "mean": statistics.fmean(vs),
        "stdev": statistics.stdev(vs) if n > 1 else 0.0,
        "iqr": pct(0.75) - pct(0.25),
        "cv_pct": (statistics.stdev(vs) / statistics.fmean(vs) * 100) if n > 1 else 0.0,
        "spread_pct": (vs[-1] - vs[0]) / statistics.median(vs) * 100,
    }


def run_single():
    print(f"\n=== PHASE A: single-request decode ({SINGLE_WARMUP} warmup discarded + "
          f"{SINGLE_ITERS} measured) ===", flush=True)
    for i in range(SINGLE_WARMUP):
        r = stream_once()
        print(f"  warmup {i + 1}: {r['decode_tps']:.2f} tok/s (DISCARDED)", flush=True)

    rows = []
    for i in range(SINGLE_ITERS):
        la = loadavg()
        r = stream_once()
        r["loadavg"] = la
        rows.append(r)
        print(
            f"  iter {i + 1:2d}: decode={r['decode_tps']:6.2f} tok/s  "
            f"ttft={r['ttft_s'] * 1000:7.1f} ms  n={r['n_tokens']:3d}  "
            f"finish={r['finish_reason']}  load={la:.2f}",
            flush=True,
        )
    return rows


def run_concurrent():
    print(f"\n=== PHASE B: {CONCURRENCY}-concurrent decode ({CONC_WARMUP} warmup discarded + "
          f"{CONC_ROUNDS} measured rounds, max_batch=4) ===", flush=True)

    def one_round():
        results = [None] * CONCURRENCY
        errors = [None] * CONCURRENCY
        # Distinct prompts so nothing can be served from a shared cache.
        def worker(idx):
            try:
                results[idx] = stream_once(
                    prompt=f"{PROMPT} (variant {idx})", max_tokens=MAX_TOKENS
                )
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                errors[idx] = repr(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(CONCURRENCY)]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0
        if any(errors):
            raise RuntimeError(f"concurrent round failed: {errors}")
        return results, wall

    for i in range(CONC_WARMUP):
        res, wall = one_round()
        agg = sum(r["decode_tps"] for r in res)
        print(f"  warmup {i + 1}: aggregate={agg:.2f} tok/s wall={wall:.2f}s (DISCARDED)", flush=True)

    rounds = []
    for i in range(CONC_ROUNDS):
        la = loadavg()
        res, wall = one_round()
        agg_decode_tps = sum(r["decode_tps"] for r in res)
        total_tokens = sum(r["n_tokens"] for r in res)
        per_stream = [r["decode_tps"] for r in res]
        row = {
            "round": i + 1,
            "aggregate_decode_tps": agg_decode_tps,
            "per_stream_decode_tps": per_stream,
            "mean_per_stream_tps": statistics.fmean(per_stream),
            "wall_s": wall,
            "wall_throughput_tps": total_tokens / wall,
            "total_tokens": total_tokens,
            "ttfts_ms": [r["ttft_s"] * 1000 for r in res],
            "loadavg": la,
            "finish_reasons": [r["finish_reason"] for r in res],
        }
        rounds.append(row)
        print(
            f"  round {i + 1}: aggregate={agg_decode_tps:6.2f} tok/s  "
            f"per-stream={statistics.fmean(per_stream):5.2f} tok/s  "
            f"wall={wall:5.2f}s  wall_tput={row['wall_throughput_tps']:6.2f} tok/s  "
            f"load={la:.2f}",
            flush=True,
        )
    return rounds


def http_get(path):
    conn = http.client.HTTPConnection(HOST, PORT, timeout=30)
    conn.request("GET", path)
    data = conn.getresponse().read().decode()
    conn.close()
    return data


def main():
    print("=== machine / build state ===", flush=True)
    state = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg_start": os.getloadavg(),
        "models": http_get("/v1/models"),
    }
    print(json.dumps(state, indent=2, default=str), flush=True)

    single = run_single()
    conc = run_concurrent()

    single_tps = [r["decode_tps"] for r in single]
    single_ttft = [r["ttft_s"] * 1000 for r in single]
    agg_tps = [r["aggregate_decode_tps"] for r in conc]
    wall_tps = [r["wall_throughput_tps"] for r in conc]
    per_stream_tps = [r["mean_per_stream_tps"] for r in conc]

    out = {
        "state": state,
        "config": {
            "prompt": PROMPT,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "single_warmup": SINGLE_WARMUP,
            "single_iters": SINGLE_ITERS,
            "conc_warmup": CONC_WARMUP,
            "conc_rounds": CONC_ROUNDS,
            "concurrency": CONCURRENCY,
        },
        "single": {
            "raw": single,
            "decode_tps": summarize(single_tps),
            "ttft_ms": summarize(single_ttft),
        },
        "concurrent": {
            "raw": conc,
            "aggregate_decode_tps": summarize(agg_tps),
            "wall_throughput_tps": summarize(wall_tps),
            "per_stream_decode_tps": summarize(per_stream_tps),
        },
        "loadavg_end": os.getloadavg(),
    }

    print("\n=== SUMMARY ===", flush=True)
    print("single decode tok/s :", json.dumps(out["single"]["decode_tps"], indent=2))
    print("single TTFT ms      :", json.dumps(out["single"]["ttft_ms"], indent=2))
    print("conc aggregate tok/s:", json.dumps(out["concurrent"]["aggregate_decode_tps"], indent=2))
    print("conc wall tok/s     :", json.dumps(out["concurrent"]["wall_throughput_tps"], indent=2))

    dest = sys.argv[1] if len(sys.argv) > 1 else "/tmp/qa-baseline-raw.json"
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nwrote {dest}", flush=True)


if __name__ == "__main__":
    main()
