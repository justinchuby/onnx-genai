import json, os, sys, time, urllib.request

PROMPT = "Write a detailed technical essay about distributed systems, consensus, and fault tolerance."

def sample(port, model, max_tokens=512):
    body = json.dumps({"model": model, "messages":[{"role":"user","content":PROMPT}],
                       "max_tokens":max_tokens, "temperature":0, "stream":True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type":"application/json"})
    t0 = time.perf_counter(); t_first=None; n=0; finish=None
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8","replace").strip()
            if not line.startswith("data:"): continue
            payload = line[5:].strip()
            if payload == "[DONE]": break
            try: obj = json.loads(payload)
            except Exception: continue
            ch = (obj.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            if ch.get("finish_reason"): finish = ch["finish_reason"]
            if delta.get("content"):
                n += 1
                if t_first is None: t_first = time.perf_counter()
    t_end = time.perf_counter()
    if t_first is None or n < 2: return None
    return {"tokens":n, "ttft_ms":(t_first-t0)*1000, "decode_tps":(n-1)/(t_end-t_first),
            "wall_tps":n/(t_end-t0), "finish":finish, "load1":os.getloadavg()[0]}

if __name__ == "__main__":
    port, model, n_per_arm = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    print("# warmup (discarded)", flush=True)
    sample(port, model)
    out=[]
    # A and B are THE SAME SERVER: the true effect is zero by construction.
    for i in range(n_per_arm):
        for arm in ("A","B"):
            s = sample(port, model)
            if s is None: print(f"# sample failed arm={arm} i={i}", flush=True); continue
            s["arm"]=arm; s["i"]=i; out.append(s)
            print(f'{arm} i={i} decode={s["decode_tps"]:.3f} tok/s ttft={s["ttft_ms"]:.0f}ms '
                  f'tokens={s["tokens"]} finish={s["finish"]} load={s["load1"]:.2f}', flush=True)
    json.dump(out, open("/tmp/qa_perf_raw.json","w"), indent=1)
    print("# wrote /tmp/qa_perf_raw.json", flush=True)
