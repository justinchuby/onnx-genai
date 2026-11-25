from pathlib import Path
import gzip,json,hashlib,math,statistics,itertools,sys
out=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent
med=statistics.median
def analyze(rows):
 assert len(rows)==3 and {r['repetition'] for r in rows}=={0,1,2};digests=set();groups={};paired={1:[],4:[]};processes=[]
 for process in rows:
  assert process['exit']==0 and process['final_used']==0 and process['same_workspace_owner'] and process['same_input_owner']
  assert all(math.isfinite(process[k]) and process[k]>0 for k in ['cpu_total_s','wall_total_s','maxrss_bytes']) and process['cpu_total_s']<=process['wall_total_s']*11
  digests.add(process['output_sha256']);records=process['records'];assert len(records)==24
  assert sum(r['wall_s'] for r in records)<=process['wall_total_s']
  seen=set()
  for r in records:
   c,index=r['construction'],r['index'];assert c in [1,4] and index in range(12) and (c,index) not in seen;seen.add((c,index));assert r['execution']==1 and r['physical_bytes']==16384
   assert r['logical_bytes']==(16384 if r['frozen'] and c==4 else 10240)
   expected_kind='work' if index<8 else 'empty' if index<10 else 'double';assert r['kind']==expected_kind and r['calls']=={'work':8192,'empty':0,'double':16384}[r['kind']]
   assert math.isfinite(r['wall_s']) and (r['wall_s']>=0 if r['kind']=='empty' else r['wall_s']>0)
   before,after=r['constructor_stats_before'],r['constructor_stats_after'];delta=[a-b for a,b in zip(after,before)];assert delta[0]>0 and delta[1]==0 if r['frozen'] else delta[1]>0 and delta[0]==0
   groups.setdefault((c,r['kind'],r['frozen']),[]).append(r)
  assert seen=={(c,i) for c,i in itertools.product([1,4],range(12))}
  for c in [1,4]:
   local=[r for r in records if r['construction']==c]
   for indices in [range(4),range(4,8)]:
    b=[r['wall_s'] for r in local if r['index'] in indices and r['frozen']];a=[r['wall_s'] for r in local if r['index'] in indices and not r['frozen']];assert len(b)==len(a)==2;paired[c].append(med(a)/med(b))
  processes.append({'rep':process['repetition'],'cpu_total_s':process['cpu_total_s'],'wall_total_s':process['wall_total_s']})
 assert len(digests)==1
 cells=[]
 for c in [1,4]:
  reasons=[];aa={};scaling={};empty_fraction={};work={}
  for frozen in [True,False]:
   label='baseline' if frozen else 'candidate';samples=groups[c,'work',frozen];assert len(samples)==12
   first=[];second=[]
   for process in rows:
    local=sorted([r for r in process['records'] if r['construction']==c and r['kind']=='work' and r['frozen']==frozen],key=lambda r:r['index']);assert len(local)==4;first.extend(r['wall_s'] for r in local[:2]);second.extend(r['wall_s'] for r in local[2:])
   values=[r['wall_s'] for r in samples];work[label]=med(values);aa[label]=med(second)/med(first)
   empty=med(r['wall_s'] for r in groups[c,'empty',frozen]);double=med(r['wall_s'] for r in groups[c,'double',frozen]);empty_fraction[label]=empty/work[label];scaling[label]=(double-empty)/(work[label]-empty)
   if not .95<=aa[label]<=1.05:reasons.append(label+' temporal A/A exceeds5%')
   if min(values)<.05:reasons.append(label+' work below50ms')
   if empty_fraction[label]>.01:reasons.append(label+' empty clock fraction exceeds1%')
   if not 1.7<=scaling[label]<=2.3:reasons.append(label+' double scaling outside15% bar')
  cells.append({'construction':c,'execution':1,'qualified':not reasons,'reasons':reasons,'aa':aa,'adjusted_double':scaling,'empty_fractions':empty_fraction,'work_wall_s':work,'paired_wall_ratios':paired[c],'candidate_over_baseline_wall':med(paired[c]),'per_arm_cpu':'not measured'})
 return {'scope':'diagnostic warm wall blocks sharing code, input owners and one physical workspace; not end-to-end throughput','processes':processes,'windows':72,'cells':cells,'all_outputs_bit_identical':True}
if __name__=='__main__':
 raw=gzip.decompress((out/'receipts.json.gz').read_bytes());meta=json.loads((out/'metadata.json').read_text());assert hashlib.sha256(raw).hexdigest()==meta['receipts_sha256'];result=analyze(json.loads(raw));(out/'results.json').write_text(json.dumps(result,indent=2))
 for c in result['cells']:print(c)
