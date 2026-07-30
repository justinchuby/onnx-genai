#!/usr/bin/env python3
"""Prefix-cache verification on the DYNAMIC (paged-KV) model.

Two questions, treated as a measurement rather than a smoke test:

  Q1  Does `prefix_cache_hits` move off zero on the dynamic model at all?
      (On scatter it is a hardcoded literal; nobody has ever read it here.)
      Also: does it move for prompts that share NOTHING? -- that decides
      whether the counter is evidence or decoration.

  Q2  Does re-sending a long shared prefix actually reduce TTFT, by more
      than the noise floor?

Design notes:
  * TTFT, not e2e. Prefix reuse can only shorten PREFILL. e2e buries it
    under decode, which is why max_tokens is tiny here.
  * Interleaved REPEAT/UNIQUE, n=15 each, so low-frequency machine wander
    hits both arms equally.
  * The UNIQUE arm differs at the FIRST token and is matched in token
    LENGTH. Differing only at the end would still share the prefix; not
    matching length would compare two different amounts of prefill work.
  * /metrics is read only BETWEEN requests -- polling it during a
    generation blocks for the whole generation (measured 14.8s).

Stdlib only. Touches no repo files.
"""

import http.client
import json
import os
import random
import statistics as st
import sys
import time

HOST, PORT, MODEL = "127.0.0.1", 8129, "qwen-dyn"
N_PER_ARM = 15
MAX_TOKENS = 8
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw",
                   "qa-prefix-dynamic-ttft.json")

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo "
         "lima mike november oscar papa quebec romeo sierra tango uniform "
         "victor whiskey xray yankee zulu").split()

# A long, fixed body. Long enough that prefill dominates TTFT.
BODY = (" The following is a technical reference document about distributed "
        "systems, consensus protocols, replication strategies, and failure "
        "detection in partially synchronous networks. ") * 40

TAIL = " Summarize the document above in one sentence."


def unique_preamble(rng, n_words):
    """Same word COUNT every time, different words -- so the two arms do the
    same amount of prefill work but share no prefix."""
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def ttft_once(prompt):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS, "temperature": 0.0, "stream": True,
    })
    conn = http.client.HTTPConnection(HOST, PORT, timeout=600)
    t0 = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", body=body,
                 headers={"Content-Type": "application/json",
                          "Accept": "text/event-stream"})
    resp = conn.getresponse()
    ttft = None
    n = 0
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            d = json.loads(payload)
        except json.JSONDecodeError:
            continue
        delta = d.get("choices", [{}])[0].get("delta", {})
        if delta.get("content"):
            if ttft is None:
                ttft = time.perf_counter() - t0
            n += 1
    conn.close()
    return {"ttft_s": ttft, "n_tokens": n}


def metrics_counters():
    """Read prefix counters BETWEEN requests only."""
    conn = http.client.HTTPConnection(HOST, PORT, timeout=30)
    conn.request("GET", "/metrics")
    text = conn.getresponse().read().decode("utf-8", "replace")
    conn.close()
    out = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for key in ("onnx_genai_prefix_cache_hits",
                    "onnx_genai_prefix_cache_lookups"):
            if line.startswith(key + " "):
                out[key.replace("onnx_genai_", "")] = float(line.split()[-1])
    return out


def ci95(vals):
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    m = st.mean(vals)
    sem = st.stdev(vals) / (len(vals) ** 0.5)
    return (m - 1.96 * sem, m + 1.96 * sem)


def main():
    rng = random.Random(1234)
    n_words = 120
    shared_prompt = unique_preamble(random.Random(999), n_words) + BODY + TAIL

    print("priming the shared prefix (2 requests)...", flush=True)
    for _ in range(2):
        ttft_once(shared_prompt)

    log = []
    before = metrics_counters()
    print("counters before: %s" % before, flush=True)

    repeat, unique = [], []
    per_request_hits = []

    for i in range(N_PER_ARM):
        # REPEAT arm: identical long prefix every time.
        c0 = metrics_counters()
        r = ttft_once(shared_prompt)
        c1 = metrics_counters()
        repeat.append(r["ttft_s"])
        per_request_hits.append({
            "arm": "repeat", "i": i,
            "d_hits": c1.get("prefix_cache_hits", 0) - c0.get("prefix_cache_hits", 0),
            "d_lookups": c1.get("prefix_cache_lookups", 0) - c0.get("prefix_cache_lookups", 0),
            "ttft_s": r["ttft_s"]})
        print("  repeat %2d: TTFT %.4fs  d_hits %+.0f"
              % (i, r["ttft_s"], per_request_hits[-1]["d_hits"]), flush=True)

        # UNIQUE arm: differs at the FIRST token, same token length.
        up = unique_preamble(rng, n_words) + BODY + TAIL
        c0 = metrics_counters()
        r = ttft_once(up)
        c1 = metrics_counters()
        unique.append(r["ttft_s"])
        per_request_hits.append({
            "arm": "unique", "i": i,
            "d_hits": c1.get("prefix_cache_hits", 0) - c0.get("prefix_cache_hits", 0),
            "d_lookups": c1.get("prefix_cache_lookups", 0) - c0.get("prefix_cache_lookups", 0),
            "ttft_s": r["ttft_s"]})
        print("  unique %2d: TTFT %.4fs  d_hits %+.0f"
              % (i, r["ttft_s"], per_request_hits[-1]["d_hits"]), flush=True)

    after = metrics_counters()

    def summ(v):
        return {"n": len(v), "median": st.median(v), "mean": st.mean(v),
                "stdev": st.stdev(v),
                "cv_pct": 100 * st.stdev(v) / st.mean(v),
                "ci95": ci95(v)}

    sr, su = summ(repeat), summ(unique)
    delta = (sr["median"] - su["median"]) / su["median"] * 100

    print("\n===== RESULT =====")
    print("counters after: %s" % after)
    print("hits gained over %d requests: %+.0f"
          % (2 * N_PER_ARM, after.get("prefix_cache_hits", 0) - before.get("prefix_cache_hits", 0)))
    uniq_hits = sum(h["d_hits"] for h in per_request_hits if h["arm"] == "unique")
    print("hits gained by the %d UNIQUE (share-nothing) requests: %+.0f"
          % (N_PER_ARM, uniq_hits))
    print("REPEAT TTFT median %.4fs  CV %.2f%%  CI95 [%.4f, %.4f]"
          % (sr["median"], sr["cv_pct"], *sr["ci95"]))
    print("UNIQUE TTFT median %.4fs  CV %.2f%%  CI95 [%.4f, %.4f]"
          % (su["median"], su["cv_pct"], *su["ci95"]))
    print("repeat vs unique: %+.2f%%  (negative = repeat faster = reuse)" % delta)
    overlap = not (sr["ci95"][1] < su["ci95"][0] or su["ci95"][1] < sr["ci95"][0])
    print("CI overlap: %s -> %s" % (overlap,
          "INDISTINGUISHABLE" if overlap else "SEPARATED"))

    with open(OUT, "w") as f:
        json.dump({"repeat_ttft": repeat, "unique_ttft": unique,
                   "repeat_summary": sr, "unique_summary": su,
                   "delta_pct": delta, "ci_overlap": overlap,
                   "counters_before": before, "counters_after": after,
                   "per_request": per_request_hits,
                   "config": {"n_per_arm": N_PER_ARM, "max_tokens": MAX_TOKENS,
                              "preamble_words": n_words}}, f, indent=1)
    print("wrote " + OUT)


if __name__ == "__main__":
    main()
