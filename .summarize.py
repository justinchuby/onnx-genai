import sys, statistics as st
from collections import defaultdict
path=sys.argv[1]
rows=defaultdict(lambda: defaultdict(list)); ort=defaultdict(lambda: defaultdict(list))
absol=defaultdict(lambda: defaultdict(list)); p90=defaultdict(lambda: defaultdict(list))
ap90=defaultdict(lambda: defaultdict(list))
wins=defaultdict(lambda:[0,0]); per_round=defaultdict(dict)
for line in open(path):
    f=line.strip().split(",")
    if len(f)<11: continue
    tag,rnd,case=f[0],int(f[1]),f[2]
    ours,ortp50,ratio,oursp90,ortp90,ratio90=map(float,f[3:9])
    rows[case][tag].append(ratio); p90[case][tag].append(ratio90)
    ort[case][tag].append(ortp50); absol[case][tag].append(ours); ap90[case][tag].append(oursp90)
    per_round[(case,rnd)][tag]=ratio
for (case,rnd),d in per_round.items():
    if "before" in d and "after" in d:
        wins[case][0 if d["after"]<d["before"] else 1]+=1
n=max(sum(v) for v in wins.values()) if wins else 0
print(f"{'case':28} {'bef p50':>8} {'aft p50':>8} {'bef p90':>8} {'aft p90':>8} {'drift':>7} {'us bef':>9} {'us aft':>9} {'won':>6}")
for case in sorted(rows):
    b=rows[case].get("before"); a=rows[case].get("after")
    if not b or not a: continue
    ob,oa=st.median(ort[case]["before"]),st.median(ort[case]["after"])
    drift=100*(oa-ob)/ob if ob else 0
    print(f"{case:28} {st.median(b):8.3f} {st.median(a):8.3f} {st.median(p90[case]['before']):8.3f} {st.median(p90[case]['after']):8.3f} {drift:+6.1f}% {st.median(absol[case]['before'])*1000:9.2f} {st.median(absol[case]['after'])*1000:9.2f} {wins[case][0]:3d}/{n}")
