import json,os,threading,time,urllib.request
PORT=9611
def gen():
    b=json.dumps({"model":"qwen-scatter","messages":[{"role":"user","content":"Write a long essay on consensus."}],
                  "max_tokens":512,"temperature":0,"stream":True}).encode()
    r=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",data=b,headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(r,timeout=300) as resp:
            for _ in resp: pass
    except Exception as e: print("gen err",e)
res={}
def poll(ep,stop):
    lat=[]
    while not stop.is_set():
        t=time.perf_counter()
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}{ep}",timeout=60).read()
            lat.append((time.perf_counter()-t)*1000)
        except Exception as e: lat.append(float('inf'))
        time.sleep(0.25)
    res[ep]=lat
# idle baseline
stop=threading.Event(); ts=[threading.Thread(target=poll,args=(e,stop)) for e in ("/v1/status","/v1/resources","/metrics")]
[t.start() for t in ts]; time.sleep(6); stop.set(); [t.join() for t in ts]
idle={k:v[:] for k,v in res.items()}
res.clear()
# under load: 4 concurrent generations
stop=threading.Event(); ts=[threading.Thread(target=poll,args=(e,stop)) for e in ("/v1/status","/v1/resources","/metrics")]
gs=[threading.Thread(target=gen) for _ in range(4)]
[t.start() for t in ts]; [g.start() for g in gs]
time.sleep(2); print(f"load1 during={os.getloadavg()[0]:.2f}")
[g.join() for g in gs]; stop.set(); [t.join() for t in ts]
print(f"{'endpoint':<16}{'IDLE worst':>12}{'LOAD worst':>12}{'LOAD median':>13}{'n':>5}")
for ep in ("/v1/status","/v1/resources","/metrics"):
    i=sorted(idle[ep]); l=sorted(res[ep])
    print(f"{ep:<16}{max(i):>11.1f}ms{max(l):>11.1f}ms{l[len(l)//2]:>12.1f}ms{len(l):>5}")
print(f"load1 after={os.getloadavg()[0]:.2f}")
