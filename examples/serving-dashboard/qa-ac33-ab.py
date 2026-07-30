import json,os,statistics,time,urllib.request,math

ARMS={ "BEFORE": {"port":8151,"model":"qwen2.5-0.5b","expect_gov":0},
       "AFTER" : {"port":9242,"model":"qwen-dynamic","expect_gov":3} }
TOK=512; BLOCK=5; NBLOCKS=3   # -> 15 per arm
PROMPT="Write a detailed technical explanation of how distributed consensus works."

def gov(port):
    try:
        r=urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics",timeout=10).read().decode()
        return r.count("resource_governor_available")
    except Exception: return -1

def gen(port,model):
    body=json.dumps({"model":model,"prompt":PROMPT,"max_tokens":TOK,
                     "temperature":0,"stream":False}).encode()
    rq=urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",body,
                              {"Content-Type":"application/json"})
    t=time.perf_counter()
    try:
        d=json.load(urllib.request.urlopen(rq,timeout=900))
    except Exception as e:
        return None,0,str(e)[:60]
    el=time.perf_counter()-t
    n=d["usage"]["completion_tokens"]
    return n/el,n,None

print("=== AC33 ACCEPTANCE A/B — protocol perf-baseline.md §5.1 ===")
print("IDENTITY CHECK, SAME INVOCATION (per @1cb42f0e):")
ok=True
for a,c in ARMS.items():
    g=gov(c["port"])
    good = (g==c["expect_gov"])
    ok &= good
    print(f"  {a:6s} :{c['port']}  governor_family={g}  expected={c['expect_gov']}  {'OK' if good else 'MISMATCH'}")
if not ok:
    print("\nABORT — arm identity failed."); raise SystemExit(1)

res={"BEFORE":[],"AFTER":[]}; ok_ct={"BEFORE":0,"AFTER":0}; att={"BEFORE":0,"AFTER":0}
loads=[]
print(f"\nblocks of {BLOCK}, {NBLOCKS} blocks/arm, max_tokens={TOK}\n")
for b in range(NBLOCKS):
    for arm in ("BEFORE","AFTER"):
        c=ARMS[arm]
        for i in range(BLOCK):
            l=os.getloadavg()[0]; loads.append(l)
            tps,n,err=gen(c["port"],c["model"]); att[arm]+=1
            if err: print(f"  b{b+1} {arm:6s} #{i+1}: ERROR {err}")
            else:
                ok_ct[arm]+=1; res[arm].append(tps)
                print(f"  b{b+1} {arm:6s} #{i+1}: {tps:6.2f} tok/s  n={n}  load1={l:6.2f}")

print("\n=== RESULT ===")
for arm in ("BEFORE","AFTER"):
    v=res[arm]
    print(f"{arm}: completions {ok_ct[arm]}/{att[arm]}   <- DENOMINATOR (per @1cb42f0e)")
    if v:
        cv=100*statistics.pstdev(v)/statistics.mean(v)
        print(f"        median {statistics.median(v):6.2f}  WORST {min(v):6.2f}  best {max(v):6.2f}  CV {cv:5.2f}%  n={len(v)}")
if res["BEFORE"] and res["AFTER"]:
    mb,ma=statistics.median(res["BEFORE"]),statistics.median(res["AFTER"])
    d=100*(ma-mb)/mb
    sb,sa=statistics.stdev(res["BEFORE"]),statistics.stdev(res["AFTER"])
    se=math.sqrt(sb**2/len(res["BEFORE"])+sa**2/len(res["AFTER"]))
    rel=100*1.96*se/mb
    print(f"\ndelta (AFTER vs BEFORE, medians): {d:+.2f}%")
    print(f"95% CI of delta: {d-rel:+.2f}% .. {d+rel:+.2f}%   (half-width {rel:.2f} pts)")
    print(f"ACCEPTANCE BAND: +/-2%")
    print("VERDICT:", "INCONCLUSIVE — CI spans the band (§5 binding rule)" if (d-rel < 2 and d+rel > -2) else ("PASS" if abs(d)<2 else "FAIL"))
print(f"\nload1 min {min(loads):.2f}  max {max(loads):.2f}  (report worst case)")
