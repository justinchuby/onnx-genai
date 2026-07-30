import json,sys,threading,time,urllib.request
PORT=int(sys.argv[1]); MODEL=sys.argv[2]; N=4
def stat():
    try: return json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/status",timeout=8))
    except Exception as e: return {"err":str(e)[:40]}
def gen(i,out):
    body=json.dumps({"model":MODEL,"prompt":"Explain distributed consensus in detail.","max_tokens":160,"temperature":0,"stream":False}).encode()
    rq=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions",body,{"Content-Type":"application/json"})
    t=time.perf_counter()
    try:
        d=json.load(urllib.request.urlopen(rq,timeout=600)); out[i]=(time.perf_counter()-t,d["usage"]["completion_tokens"])
    except Exception as e: out[i]=(None,str(e)[:50])
s=stat(); print(f"  idle    batch_capacity={s.get('batch_capacity')} in_flight={s.get('batch_in_flight')}")
out={}; th=[threading.Thread(target=gen,args=(i,out)) for i in range(N)]
t0=time.perf_counter()
for x in th: x.start()
peak=0; samples=[]
for _ in range(24):
    time.sleep(0.5); s=stat(); f=s.get('batch_in_flight') or 0
    samples.append(f); peak=max(peak,f)
    if not any(x.is_alive() for x in th): break
for x in th: x.join()
wall=time.perf_counter()-t0
toks=sum(v[1] for v in out.values() if v[0])
print(f"  PEAK batch_in_flight = {peak}   samples={samples[:12]}")
print(f"  wall={wall:.1f}s  completions={sum(1 for v in out.values() if v[0])}/{N}  tokens={toks}  agg={toks/wall:.2f} tok/s")
