"""Replicate the PM's Scenario-B prefix-cache experiment, with the control arm added.

Arm REPEAT : cold request with prefix P, then an identical request with prefix P.
Arm UNIQUE : cold request with prefix P, then a request with a DIFFERENT prefix Q.

If the second request is faster in BOTH arms, the speedup is first-request warmup,
not prefix reuse. Run each arm against a freshly started server.
"""

import json
import sys
import time
import urllib.request

PORT = int(sys.argv[1])
ARM = sys.argv[2]  # "repeat" | "unique"
MODEL = "qwen2.5-0.5b"
MAX_TOKENS = 48


def long_prompt(seed_word: str) -> str:
    # ~30x repeated long prefix, matching the PM's description.
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
    e2e = time.perf_counter() - t0
    usage = payload.get("usage", {})
    return e2e, usage.get("prompt_tokens"), usage.get("completion_tokens")


def prefix_counters():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=600) as r:
            text = r.read().decode()
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    out = {}
    for line in text.splitlines():
        if line.startswith("prefix_cache") and not line.startswith("#"):
            key, _, val = line.partition(" ")
            out[key] = val
    return out


P = long_prompt("scheduler")
Q = long_prompt("quantization")  # differs from token 0

before = prefix_counters()
e2e1, ptok1, ctok1 = chat(P)
mid = prefix_counters()
second_prompt = P if ARM == "repeat" else Q
e2e2, ptok2, ctok2 = chat(second_prompt)
after = prefix_counters()

delta_pct = (e2e2 - e2e1) / e2e1 * 100.0
print(
    json.dumps(
        {
            "arm": ARM,
            "req1_e2e_s": round(e2e1, 4),
            "req2_e2e_s": round(e2e2, 4),
            "delta_pct_req2_vs_req1": round(delta_pct, 2),
            "prompt_tokens": [ptok1, ptok2],
            "completion_tokens": [ctok1, ctok2],
            "counters_before": before,
            "counters_after_req1": mid,
            "counters_after_req2": after,
        },
        indent=2,
    )
)
