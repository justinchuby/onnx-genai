from pathlib import Path
import gzip,json,itertools,hashlib,statistics,math,sys
out=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent
SHAPES=[(1,64,128,1,0,1,1,0),(8,128,256,1,0,0,4,1),(8,128,256,4,1,0,4,1)]
ARMS=['baseline','candidate','null-baseline','null-candidate']
def capacity(m,h,i,width,gated,matmul):
 def phase(k,n):
  per=(n+width-1)//width;shards=(n+per-1)//per
  return ((m*n if m>1 and n>1 else 0)+shards*min(4,per)*k)*4
 if matmul:return phase(h,i)
 a,b=(max(h,i),max(2*i,h)) if gated else (h,i)
 return m*(a+b+h)*4+max(phase(h,i),phase(i,h))
def analyze(rows):
 expected={(rep,c,k,a) for rep,c,k,a in itertools.product(range(3),range(3),['setup','work','double'],ARMS) if k!='double' or a in ['baseline','candidate']}
 seen=set();groups={};outputs={}
 for r in rows:
  key=tuple(r[k] for k in ['rep','case','kind','arm']);assert key in expected and key not in seen;seen.add(key)
  m,h,i,width,dense,gated,construction,matmul=SHAPES[r['case']]
  assert (r['rows'],r['hidden'],r['inter'],r['rayon_width'],int(r['dense']),int(r['gated']),r['construction_width'],int(r['matmul']))==SHAPES[r['case']]
  calls=0 if r['kind']=='setup' else (4294967296//(m*h*i*(1 if matmul else 2*(3 if gated else 2))))*(2 if r['kind']=='double' else 1)
  assert r['calls']==calls and r['exit']==0 and r['final_used']==0
  assert all(math.isfinite(r[k]) and r[k]>0 for k in ['cpu_s','wall_s','maxrss_bytes']) and r['cpu_s']<=r['wall_s']*11
  assert outputs.setdefault(r['case'],r['output_sha256'])==r['output_sha256'] and len(r['output_sha256'])==64
  assert len(r['argmax'])==m and all(isinstance(a,int) and 0<=a<(i if matmul else h) for a in r['argmax'])
  frozen=r['arm'].endswith('baseline');logical=capacity(m,h,i,construction if frozen else width,gated,matmul);initial=capacity(m,h,i,construction,gated,matmul);peak=max(logical,initial)
  assert (r['logical_workspace_bytes'],r['construction_workspace_bytes'],r['workspace_bytes'])==(logical,initial,peak)
  assert r['host_limit_bytes']==(67108864 if dense else peak) and r['derived_bytes']==(h*i*4 if dense else 0)
  for metric in ['cpu_s','wall_s']:groups.setdefault((r['case'],r['kind'],r['arm'],metric),[]).append(r[metric])
 assert seen==expected and len(rows)==90
 cells=[]
 for c,shape in enumerate(SHAPES):
  metrics={}
  for metric in ['cpu_s','wall_s']:
   work={a:statistics.median(groups[c,'work',a,metric]) for a in ARMS};setup={a:statistics.median(groups[c,'setup',a,metric]) for a in ARMS}
   aa={a:work['null-'+a]/work[a] for a in ['baseline','candidate']}
   fraction={a:setup[a]/work[a] for a in ARMS}
   scaling={a:(statistics.median(groups[c,'double',a,metric])-setup[a])/(work[a]-setup[a]) for a in ['baseline','candidate']}
   reasons=[]
   if any(not .95<=v<=1.05 for v in aa.values()):reasons.append('an arm fails 5% A/A bar')
   if any(v>.25 for v in fraction.values()):reasons.append('setup exceeds 25%')
   if any(not 1.7<=v<=2.3 for v in scaling.values()):reasons.append('double work fails scaling')
   if any(min(groups[c,'work',a,metric])<.05 for a in ARMS):reasons.append('work below 50ms')
   pairs=[]
   for rep in range(3):
    value=lambda a:next(r[metric] for r in rows if (r['case'],r['kind'],r['rep'],r['arm'])==(c,'work',rep,a))
    pairs.append(value('candidate')/value('baseline'))
   metrics[metric]={'qualified':not reasons,'reasons':reasons,'aa':aa,'setup_fractions':fraction,'adjusted_double':scaling,'work_s':work,'setup_s':setup,'paired_ratios':pairs,'candidate_over_baseline':statistics.median(pairs)}
  cells.append({'case':c,'shape':shape,'metrics':metrics,'outputs_bit_identical':True})
 return {'processes':90,'scope':'diagnostic shared image, not original two-binary matrix','cells':cells}
if __name__=='__main__':
 raw=gzip.decompress((out/'receipts.json.gz').read_bytes());meta=json.loads((out/'metadata.json').read_text());assert hashlib.sha256(raw).hexdigest()==meta['receipts_sha256']
 result=analyze(json.loads(raw));(out/'results.json').write_text(json.dumps(result,indent=2))
 for c in result['cells']:print(c['shape'],{m:{k:v[k] for k in ['qualified','candidate_over_baseline','aa','reasons']} for m,v in c['metrics'].items()})
