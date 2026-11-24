import copy,gzip,hashlib,itertools,json,math,pathlib,statistics,sys
out=pathlib.Path(sys.argv[1] if len(sys.argv)>1 else '/tmp/nxrt-neon-unroll-process-focus')
SHAPES=[(1,4),(1,7),(1,8),(1,16),(1,257),(32,1024),(128,4099),(16,65539)]
def analyze(rows,meta):
 assert meta['schema']==4 and meta['repetitions']==3 and meta['work_elements']==134217728
 assert meta['arms']=={'0':'original_same_arithmetic','1':'unrolled_native','2':'unrolled_native_aa','3':'mlas','4':'original_outlined'}
 expected={(kind,rep,w,c,a) for kind,rep,w,c,a in itertools.product(['setup','work'],range(3),[8],[5],range(5))}
 expected|={('double',rep,w,c,a) for rep,w,c,a in itertools.product(range(3),[8],[5],[0,1])}
 seen=set();hashes={};groups={}
 for r in rows:
  kind,rep,w,c,a=[r[k] for k in ['kind','rep','width','case','arm']];key=(kind,rep,w,c,a)
  assert key in expected and key not in seen
  n,d=SHAPES[c];assert (r['rows'],r['cols'])==(n,d)
  calls=0 if kind=='setup' else max(1,meta['work_elements']//(n*d))*(2 if kind=='double' else 1)
  assert r['calls']==calls and r['exit']==0
  assert math.isfinite(r['cpu_s']) and math.isfinite(r['wall_s']) and r['cpu_s']>0 and r['wall_s']>0 and r['maxrss_bytes']>0
  assert r['cpu_s']<=r['wall_s']*(int(meta['hw.logicalcpu'])+1)
  assert 0<=r['worst_error']<=1e-6 and r['digest']>0 and not(r['fpcr']&((1<<1)|(3<<22)|(1<<24)|(1<<25)))
  assert r['workers']>=0 and (r['workers']>0)==(w>1 and n>=2 and n*d>=16384)
  assert r['hits']==(n*calls if a!=3 and d>=8 else 0)
  assert hashes.setdefault((c,a),r['digest'])==r['digest']
  groups.setdefault((kind,w,c,a),[]).append(r['cpu_s']);seen.add(key)
 assert seen==expected and len(rows)==36
 for c in [5]:assert len({hashes[c,a] for a in [0,1,2,4]})==1
 cells=[]
 for w,c in itertools.product([8],[5]):
  setup={str(a):statistics.median(groups['setup',w,c,a]) for a in range(5)}
  work={str(a):statistics.median(groups['work',w,c,a]) for a in range(5)}
  aa=work['1']/work['2'];fixed={str(a):setup[str(a)]/work[str(a)] for a in range(5)}
  scaling={}
  if c>=4:
   for a in [0,1]:
    double=statistics.median(groups['double',w,c,a]);denom=work[str(a)]-setup[str(a)]
    scaling[str(a)]={'double_cpu_s':double,'raw_double_over_work':double/work[str(a)],'setup_adjusted_work_scale':(double-setup[str(a)])/denom if denom>0 else None}
  reasons=[]
  if not .95<=aa<=1.05:reasons.append('candidate process A/A exceeds 5%')
  if any(fixed[str(a)]>.25 for a in [0,1]):reasons.append('setup exceeds 25% of whole-process CPU')
  if any(v['setup_adjusted_work_scale'] is None or not 1.7<=v['setup_adjusted_work_scale']<=2.3 for v in scaling.values()):reasons.append('doubled work fails 15% proportionality bar')
  ratios=[]
  for rep in range(3):
   f=lambda a:next(r['cpu_s'] for r in rows if (r['kind'],r['rep'],r['width'],r['case'],r['arm'])==('work',rep,w,c,a))
   ratios.append(f(0)/f(1))
  cells.append({'width':w,'case':c,'shape':SHAPES[c],'setup_cpu_s':setup,'work_cpu_s':work,'candidate_aa':aa,'fixed_fractions':fixed,'scaling':scaling,'qualified':not reasons,'reasons':reasons,'original_over_candidate_ratio':statistics.median(ratios),'paired_ratios':ratios,'original_over_outline':work['0']/work['4'],'outline_over_candidate':work['4']/work['1'],'mlas_over_candidate':work['3']/work['1']})
 return {'schema':1,'processes':len(rows),'cells':cells,'all_native_full_output_hashes_match':True}
if __name__=='__main__':
 meta=json.loads((out/'metadata.json').read_text());raw=gzip.decompress((out/'receipts.json.gz').read_bytes());assert hashlib.sha256(raw).hexdigest()==meta['receipts_sha256']
 result=analyze(json.loads(raw),meta);(out/'results.json').write_text(json.dumps(result,indent=2));print('qualified',sum(c['qualified'] for c in result['cells']),len(result['cells']))
 for c in result['cells']:print(c['width'],c['shape'],'ratio',round(c['original_over_candidate_ratio'],4),'AA',round(c['candidate_aa'],4),'fixed',round(c['fixed_fractions']['1'],4),c['reasons'])
