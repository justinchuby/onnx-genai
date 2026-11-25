import gzip,hashlib,itertools,json,math,pathlib,statistics,sys
OUT=pathlib.Path(sys.argv[1] if len(sys.argv)>1 else '/tmp/nxrt-qmoe-contribution-study')
SHAPES=[(1,32,64,1,0,0),(1,64,128,1,1,0),(1,64,128,1,0,1),(8,128,256,1,0,0),(8,128,256,4,0,1),(8,128,256,4,1,1)]
def analyze(rows):
 expected={(rep,c,kind,arm) for rep,c,kind,arm in itertools.product(range(3),range(6),['setup','work','double'],['baseline','candidate','null']) if kind!='double' or arm!='null'}
 seen=set();outputs={};groups={}
 for r in rows:
  rep,c,kind,arm=[r[k] for k in ['rep','case','kind','arm']];key=(rep,c,kind,arm)
  assert key in expected and key not in seen;seen.add(key)
  m,h,i,width,dense,gated=SHAPES[c]
  assert (r['rows'],r['hidden'],r['inter'],r['rayon_width'],int(r['dense']),int(r['gated']))==(m,h,i,width,dense,gated)
  calls=0 if kind=='setup' else max(1,268435456//(m*h*i*2*(3 if gated else 2)))*(2 if kind=='double' else 1)
  assert r['calls']==calls and r['exit']==0 and r['final_used']==0
  assert all(math.isfinite(r[k]) and r[k]>0 for k in ['cpu_s','wall_s','maxrss_bytes'])
  assert r['cpu_s']<=r['wall_s']*11
  assert len(r['output_sha256'])==64 and outputs.setdefault(c,r['output_sha256'])==r['output_sha256']
  assert len(r['argmax'])==m and all(isinstance(a,int) and 0<=a<h for a in r['argmax'])
  assert r['derived_bytes']==0
  a,b=(max(h,i),max(2*i,h)) if gated else (h,i)
  workspace=(m*((a+b+2*h) if arm=='candidate' else (a+b))*4)
  assert r['workspace_bytes']==workspace,(c,arm,r['workspace_bytes'],workspace)
  assert r['host_limit_bytes']==(67108864 if dense else workspace)
  groups.setdefault((c,kind,arm),[]).append(r['cpu_s'])
 assert seen==expected and len(rows)==144
 cells=[]
 for c,shape in enumerate(SHAPES):
  work={arm:statistics.median(groups[c,'work',arm]) for arm in ['baseline','candidate','null']}
  setup={arm:statistics.median(groups[c,'setup',arm]) for arm in work}
  aa=work['null']/work['baseline'];fixed={arm:setup[arm]/work[arm] for arm in work}
  scaling={arm:(statistics.median(groups[c,'double',arm])-setup[arm])/(work[arm]-setup[arm]) if work[arm]>setup[arm] else None for arm in ['baseline','candidate']}
  reasons=[]
  if not .95<=aa<=1.05:reasons.append('baseline/null A/A exceeds 5%')
  if any(v>.25 for v in fixed.values()):reasons.append('setup exceeds 25% of process CPU')
  if any(v is None or not 1.7<=v<=2.3 for v in scaling.values()):reasons.append('double work fails 15% scaling bar')
  if min(min(groups[c,'work',a]) for a in work)<.05:reasons.append('work process CPU below 50ms')
  pairs=[]
  for rep in range(3):
   value=lambda a:next(r['cpu_s'] for r in rows if (r['case'],r['kind'],r['rep'],r['arm'])==(c,'work',rep,a))
   pairs.append(value('candidate')/value('baseline'))
  cells.append({'case':c,'shape':shape,'qualified':not reasons,'reasons':reasons,'cpu_work_s':work,'cpu_setup_s':setup,'baseline_null_aa':aa,'setup_fractions':fixed,'setup_adjusted_double':scaling,'candidate_over_baseline':statistics.median(pairs),'paired_ratios':pairs,'all_outputs_bit_identical':True})
 return {'schema':1,'processes':144,'cells':cells}
if __name__=='__main__':
 raw=gzip.decompress((OUT/'receipts.json.gz').read_bytes());meta=json.loads((OUT/'metadata.json').read_text());assert hashlib.sha256(raw).hexdigest()==meta['receipts_sha256']
 result=analyze(json.loads(raw));(OUT/'results.json').write_text(json.dumps(result,indent=2))
 print('qualified',sum(c['qualified'] for c in result['cells']),len(result['cells']))
 for c in result['cells']:print(c['shape'],'candidate/baseline',round(c['candidate_over_baseline'],4),'AA',round(c['baseline_null_aa'],4),c['reasons'])
