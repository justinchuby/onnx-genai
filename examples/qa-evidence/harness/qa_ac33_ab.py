import json,os,sys,time,urllib.request
sys.path.insert(0,'/tmp'); from qa_perf import sample
ARMS={"BEFORE":9621,"AFTER":9622}
print("# warmups"); [sample(p,"qwen-scatter") for p in ARMS.values()]
out=[]
for i in range(8):
    for arm,port in ARMS.items():
        s=sample(port,"qwen-scatter")
        if s is None: print(f"# FAILED {arm} i={i}",flush=True); continue
        s["arm"]=arm; s["i"]=i; s["port"]=port; out.append(s)
        print(f'{arm:6} i={i} decode={s["decode_tps"]:.3f} ttft={s["ttft_ms"]:.0f}ms tok={s["tokens"]} fin={s["finish"]} load={s["load1"]:.2f}',flush=True)
json.dump(out,open("/tmp/qa_ac33_raw.json","w"),indent=1)
print("# done")
