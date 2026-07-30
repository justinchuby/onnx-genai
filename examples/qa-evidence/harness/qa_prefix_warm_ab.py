"""Warm-server interleaved A/B: does a repeated prefix beat a unique prefix?

Removes the cold-start confound (a discarded warmup request), removes drift
(strict interleaving), and uses the PM's prompt shape (~30x repeated sentence).
"""

import json
import statistics
import sys
import time
import urllib.request

PORT = int(sys.argv[1])
MODEL = "qwen2.5-0.5b"
MAX_TOKENS = 32
N = 6


def long_prompt(seed_word: str) -> str:
    sentence = (
        f"The {seed_word} subsystem coordinates paged key-value memory across "
        f"concurrent decode requests while preserving deterministic ordering. "
    )
    return sentence * 30 + f"\n\nSummarize the {seed_word} subsystem in one sentence."


def chat(prompt: str):
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        payload = json.loads(resp.read())
    return time.perf_counter() - t0, payload.get("usage", {})


def kv_counters():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/debug/kv", timeout=600) as r:
            return json.loads(r.read())
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


REPEATED = long_prompt("scheduler")

# Warm-up: establishes the cache entry for REPEATED and absorbs one-time init.
chat(REPEATED)
chat(REPEATED)

before = kv_counters()
repeat_times, unique_times = [], []
for i in range(N):
    # Strict interleave, alternating which arm goes first to cancel any ordering bias.
    if i % 2 == 0:
        repeat_times.append(chat(REPEATED)[0])
        unique_times.append(chat(long_prompt(f"alpha{i}"))[0])
    else:
        unique_times.append(chat(long_prompt(f"beta{i}"))[0])
        repeat_times.append(chat(REPEATED)[0])
after = kv_counters()

rm, um = statistics.median(repeat_times), statistics.median(unique_times)
print(
    json.dumps(
        {
            "n_per_arm": N,
            "repeated_prefix_median_s": round(rm, 4),
            "unique_prefix_median_s": round(um, 4),
            "repeated_vs_unique_pct": round((rm - um) / um * 100.0, 2),
            "repeated_raw": [round(t, 4) for t in repeat_times],
            "unique_raw": [round(t, 4) for t in unique_times],
            "kv_before": before,
            "kv_after": after,
        },
        indent=2,
    )
)
