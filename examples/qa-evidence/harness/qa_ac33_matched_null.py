import json,sys
sys.path.insert(0,'/tmp'); from qa_perf import sample
# A/A: BOTH arms are :9621. True effect EXACTLY zero, same quiet window as the A/B.
print("# warmup"); sample(9621,"qwen-scatter")
out=[]
for i in range(8):
    for arm in ("A1","A2"):
        s=sample(9621,"qwen-scatter")
        if s is None: print(f"# FAILED {arm} i={i}",flush=True); continue
        s["arm"]=arm; s["i"]=i; out.append(s)
        print(f'{arm} i={i} decode={s["decode_tps"]:.3f} load={s["load1"]:.2f}',flush=True)
json.dump(out,open("/tmp/qa_aa_raw.json","w"),indent=1); print("# done")
