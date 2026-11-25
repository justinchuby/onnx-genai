import gzip,hashlib,itertools,json,math,pathlib,statistics,sys
OUT=pathlib.Path(sys.argv[1] if len(sys.argv)>1 else '/tmp/nxrt-qmoe-decoder-study')
SHAPES=[(1,32,64,1,0,0,1),(1,64,128,1,1,0,1),(1,64,128,1,0,1,1),(8,128,256,1,0,0,1),(8,128,256,4,0,1,4),(8,128,256,4,1,1,4),(8,128,256,4,0,1,1),(8,128,256,1,0,1,4)]
def capacity(m,h,i,width,gated,candidate):
 a,b=(max(h,i),max(2*i,h)) if gated else (h,i)
 base=m*(a+b+2*h)*4
 if not candidate:return base
 def phase(k,n):
  shards=(n+((n+width-1)//width)-1)//((n+width-1)//width)
  return ((m*n if m>1 and n>1 else 0)+shards*k)*4
 return base+max(phase(h,i),phase(i,h))
def analyze(rows):
 expected={(rep,c,kind,arm) for rep,c,kind,arm in itertools.product(range(3),range(8),['setup','work','double'],['baseline','candidate','null']) if kind!='double' or arm!='null'}
 seen=set();outputs={};groups={}
 for r in rows:
  rep,c,kind,arm=[r[k] for k in ['rep','case','kind','arm']];key=(rep,c,kind,arm)
  assert key in expected and key not in seen;seen.add(key)
  m,h,i,width,dense,gated,construction=SHAPES[c]
  assert (r['rows'],r['hidden'],r['inter'],r['rayon_width'],int(r['dense']),int(r['gated']),r['construction_width'])==(m,h,i,width,dense,gated,construction)
  calls=0 if kind=='setup' else max(1,268435456//(m*h*i*2*(3 if gated else 2)))*(2 if kind=='double' else 1)
  assert r['calls']==calls and r['exit']==0 and r['final_used']==0
  assert all(math.isfinite(r[k]) and r[k]>0 for k in ['cpu_s','wall_s','maxrss_bytes'])
  assert r['cpu_s']<=r['wall_s']*11
  assert len(r['output_sha256'])==64 and outputs.setdefault((m,h,i,gated,dense),r['output_sha256'])==r['output_sha256']
  assert len(r['argmax'])==m and all(isinstance(a,int) and 0<=a<h for a in r['argmax'])
  assert r['derived_bytes']==0
  logical=capacity(m,h,i,width,gated,arm=='candidate')
  initial=capacity(m,h,i,construction,gated,arm=='candidate')
  physical=max(logical,initial)
  assert (r['logical_workspace_bytes'],r['construction_workspace_bytes'],r['workspace_bytes'])==(logical,initial,physical),(c,arm,r,logical,initial,physical)
  assert r['host_limit_bytes']==(67108864 if dense else physical)
  for metric in ['cpu_s','wall_s']:groups.setdefault((c,kind,arm,metric),[]).append(r[metric])
 assert seen==expected and len(rows)==192
 cells=[]
 for c,shape in enumerate(SHAPES):
  metrics={}
  for metric in ['cpu_s','wall_s']:
   work={a:statistics.median(groups[c,'work',a,metric]) for a in ['baseline','candidate','null']}
   setup={a:statistics.median(groups[c,'setup',a,metric]) for a in work}
   aa=work['null']/work['baseline'];fixed={a:setup[a]/work[a] for a in work}
   scaling={a:(statistics.median(groups[c,'double',a,metric])-setup[a])/(work[a]-setup[a]) if work[a]>setup[a] else None for a in ['baseline','candidate']}
   reasons=[]
   if not .95<=aa<=1.05:reasons.append('baseline/null A/A exceeds 5%')
   if any(v>.25 for v in fixed.values()):reasons.append('setup exceeds 25% of process metric')
   if any(v is None or not 1.7<=v<=2.3 for v in scaling.values()):reasons.append('double work fails 15% scaling bar')
   if min(min(groups[c,'work',a,metric]) for a in work)<.05:reasons.append('work process metric below 50ms')
   pairs=[]
   for rep in range(3):
    value=lambda a:next(r[metric] for r in rows if (r['case'],r['kind'],r['rep'],r['arm'])==(c,'work',rep,a))
    pairs.append(value('candidate')/value('baseline'))
   metrics[metric]={'qualified':not reasons,'reasons':reasons,'work_s':work,'setup_s':setup,'baseline_null_aa':aa,'setup_fractions':fixed,'setup_adjusted_double':scaling,'candidate_over_baseline':statistics.median(pairs),'paired_ratios':pairs}
  cells.append({'case':c,'shape':shape,'metrics':metrics,'all_outputs_bit_identical':True})
 return {'schema':1,'processes':192,'cells':cells}
if __name__=='__main__':
 raw=gzip.decompress((OUT/'receipts.json.gz').read_bytes());meta=json.loads((OUT/'metadata.json').read_text());assert hashlib.sha256(raw).hexdigest()==meta['receipts_sha256']
 result=analyze(json.loads(raw));(OUT/'results.json').write_text(json.dumps(result,indent=2))
 for c in result['cells']:
  print(c['shape'], {m: {k:v[k] for k in ['qualified','candidate_over_baseline','baseline_null_aa','reasons']} for m,v in c['metrics'].items()})
