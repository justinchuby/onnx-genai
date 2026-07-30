import json,time,statistics,urllib.request,os
PORT=8151; N=6; TOK=128
def loadavg1(): return os.getloadavg()[0]
def run(prompt):
    body=json.dumps({"model":"qwen2.5-0.5b","prompt":prompt,"max_tokens":TOK,
                     "temperature":0,"stream":False}).encode()
    r=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions",body,
                             {"Content-Type":"application/json"})
    t=time.perf_counter()
    d=json.load(urllib.request.urlopen(r,timeout=600))
    el=time.perf_counter()-t
    n=d["usage"]["completion_tokens"]
    return el,n,n/el
P="Write a short paragraph about distributed systems."
A=[];B=[];loads=[]
print(f"null A/B  port={PORT}  max_tokens={TOK}  pairs={N}  (both arms are the SAME server)")
for i in range(N):
    for arm,acc in (("A",A),("B",B)):
        l=loadavg1(); el,n,tps=run(P); acc.append(tps); loads.append(l)
        print(f"  pair {i+1} arm {arm}: {tps:6.2f} tok/s  ({n} tok in {el:5.2f}s)  load1={l:6.2f}")
def cv(x): return 100*statistics.pstdev(x)/statistics.mean(x)
deltas=[100*(b-a)/a for a,b in zip(A,B)]
print(f"\narm A  median {statistics.median(A):6.2f} tok/s  CV {cv(A):5.2f}%  n={len(A)}")
print(f"arm B  median {statistics.median(B):6.2f} tok/s  CV {cv(B):5.2f}%  n={len(B)}")
print(f"\npaired delta B-A  (TRUE VALUE IS ZERO — identical binary, identical server)")
print(f"  per-pair %: {', '.join(f'{d:+.2f}' for d in deltas)}")
print(f"  mean {statistics.mean(deltas):+.2f}%   median {statistics.median(deltas):+.2f}%")
print(f"  WORST-CASE single pair: {max(deltas,key=abs):+.2f}%")
print(f"  spread min..max: {min(deltas):+.2f}% .. {max(deltas):+.2f}%")
print(f"\nload1 during run: min {min(loads):.2f}  max {max(loads):.2f}")
print(f"AC33 acceptance threshold is +/-2%.")
