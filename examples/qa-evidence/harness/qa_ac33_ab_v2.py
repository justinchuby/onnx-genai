"""AC33 A/B — self-contained. Supersedes qa_ac33_ab.py.

WHY v2 EXISTS, both reasons being defects in v1 that I found by trying to re-run it:

  1. v1 did `sys.path.insert(0,'/tmp'); from qa_perf import sample`. THAT FILE WAS NEVER
     COMMITTED. v1 is unrunnable from a clean checkout -- a harness committed as evidence
     that cannot reproduce its own result. This file imports nothing but the stdlib.

  2. v1 asserted arm identity by /v1/models FIELD COUNT, on the rule "4 = pre-fix,
     7 = post-fix". That rule is WRONG. Measured across seven live arms there are THREE
     field counts, and the path-disclosure fix REMOVED a field:
         4 = predates the `path` field entirely
         7 = HAS the `path` field and DISCLOSES an absolute path   <- the leaking build
         6 = `path` field deleted by the redaction fix             <- what actually ships
     So 7 is not "newer", it is "leaking", and a probe that reads 7 as post-fix waves
     through the exact build the fix exists to eliminate.

IDENTITY IS THEREFORE A TUPLE, NOT A SCALAR, AND IT ABORTS RATHER THAN WARNS:
  (models field count, governor line count in /metrics, whether `path` is absolute)
Two independent fixes landed on this server; no single scalar covers both, and each arm
must match its EXPECTED tuple before a single token is generated.
"""
import json, os, sys, time, urllib.request

MODEL_ID = "qwen-scatter"
ARMS = {"BEFORE": 9711, "AFTER": 9712}
EXPECT = {  # (fields, governor_lines, path_is_absolute)
    "BEFORE": (4, 0, False),
    "AFTER": (6, 3, False),
}
PAIRS = int(os.environ.get("PAIRS", "12"))
TOKENS = 512


def _get(port, path, timeout=15):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as r:
        return r.read().decode()


def identity(port):
    m = json.loads(_get(port, "/v1/models"))["data"][0]
    gov = sum(1 for ln in _get(port, "/metrics").splitlines() if "governor" in ln)
    return (len(m), gov, str(m.get("path", "")).startswith("/"))


def sample(port):
    """One generation. Returns decode tok/s measured from first token to last."""
    body = json.dumps({
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "Write a detailed technical description of how a continuous batching scheduler works."}],
        "max_tokens": TOKENS, "temperature": 0, "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    n = 0
    finish = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            d = json.loads(payload)["choices"][0]
            if d.get("delta", {}).get("content"):
                if first is None:
                    first = time.perf_counter()
                n += 1
            if d.get("finish_reason"):
                finish = d["finish_reason"]
    end = time.perf_counter()
    if first is None or n < 2:
        return None
    return {
        "tokens": n, "finish": finish,
        "ttft_ms": (first - t0) * 1000,
        "decode_tps": (n - 1) / (end - first),
        "wall_tps": n / (end - t0),
        "load1": os.getloadavg()[0],
    }


def main():
    print("# ARM IDENTITY -- asserted before any generation, aborts on mismatch")
    for arm, port in ARMS.items():
        got, want = identity(port), EXPECT[arm]
        print(f"#   {arm:6} :{port}  fields/governor/path_absolute = {got}  expected {want}", flush=True)
        if got != want:
            sys.exit(f"ABORT: {arm} on :{port} is {got}, expected {want}. "
                     "The arm is not the binary this run claims to measure.")
    print("# both arms match. warmup...", flush=True)
    for p in ARMS.values():
        sample(p)

    out = []
    for i in range(PAIRS):
        for arm, port in ARMS.items():  # interleaved within each pair
            s = sample(port)
            if s is None:
                print(f"# FAILED {arm} i={i}", flush=True)
                continue
            s.update(arm=arm, i=i, port=port)
            out.append(s)
            print(f'{arm:6} i={i:2} decode={s["decode_tps"]:.3f} ttft={s["ttft_ms"]:.0f}ms '
                  f'tok={s["tokens"]} fin={s["finish"]} load={s["load1"]:.2f}', flush=True)
        json.dump(out, open(os.environ.get("OUT", "/tmp/qa_ac33_v2.json"), "w"), indent=1)
    print("# done", flush=True)


if __name__ == "__main__":
    main()
