from pathlib import Path
import gzip,hashlib,itertools,json,math,statistics,struct,importlib.util
ROOT=Path('/tmp/nxrt-im2col-perf');sha=lambda b:hashlib.sha256(b).hexdigest()
def analyze(plan,rows,read_file,oracle):
 assert plan['completed'] and plan['formal_processes']==216 and len(rows)==216
 expected=set(itertools.product(plan['labels'],range(6),['setup','work','double'],range(3)));seen={};digests={};truth={}
 for r in rows:
  key=(r['label'],r['case'],r['kind'],r['rep']);assert key in expected and key not in seen;seen[key]=r
  arm=r['label'].removesuffix('-aa');assert r['arm']==arm
  n=plan['calls'][f'{arm}/{r["case"]}'];calls=0 if r['kind']=='setup' else n*(2 if r['kind']=='double' else 1);assert r['calls']==calls
  p=r['probe'];assert (p['arm'],p['case'],p['phase'],p['calls'],p['linked'])==(arm,r['case'],'reuse',calls,False)
  assert p['mlas_gemm_witnesses']==[0,0]
  if r['case'] in [1,2,4]:assert len(p['native_gemm_witnesses'])==2 and all(x>0 for x in p['native_gemm_witnesses'])
  assert r['exit']==0 and r['pid']>0 and r['rss_bytes']>0 and math.isfinite(r['cpu_s']) and 0<r['cpu_s']<=plan['logical_cpus']*r['wall_s']+.01
  for name in ['stdout','stderr']:assert sha(read_file(r[name]))==r[name+'_sha256']
  assert json.loads(read_file(r['stdout']))==p
  data=gzip.decompress(read_file(r['output']));assert sha(data)==r['output_sha256']
  digests.setdefault(r['case'],set()).add(sha(data));assert len(digests[r['case']])==1,'before/after/AA/work-count output bytes differ'
  identity=(arm,r['case'])
  if identity not in truth:
   expected_values=oracle(r['case']);values=[x[0] for x in struct.iter_unpack('<f',data)];assert len(values)==len(expected_values)
   assert all(math.isfinite(a) and abs(a-b)<=2e-5*(1+abs(b)) for a,b in zip(values,expected_values));truth[identity]=True
 assert set(seen)==expected
 cells=[];q=plan['qualifications']
 for case in range(6):
  arms={}
  for label in plan['labels']:
   slopes=[];fractions=[];scales=[];work=[];reasons=[]
   for rep in range(3):
    s=seen[label,case,'setup',rep];w=seen[label,case,'work',rep];d=seen[label,case,'double',rep];delta=w['cpu_s']-s['cpu_s'];slopes.append(delta/w['calls']);fractions.append(s['cpu_s']/w['cpu_s']);scales.append((d['cpu_s']-s['cpu_s'])/delta if delta>0 else None);work.append(w['cpu_s'])
   if any(x<=0 for x in slopes):reasons.append('nonpositive adjusted CPU')
   if any(x<q['minimum_work_cpu_s'] for x in work):reasons.append('short work CPU')
   if any(x>q['maximum_setup_fraction'] for x in fractions):reasons.append('large setup fraction')
   if any(x is None or not q['doubling_bounds'][0]<=x<=q['doubling_bounds'][1] for x in scales):reasons.append('double-work failed')
   arms[label]={'cpu_s_per_call':slopes,'median_cpu_us_per_call':statistics.median(slopes)*1e6,'range_cpu_us_per_call':[min(slopes)*1e6,max(slopes)*1e6],'setup_fractions':fractions,'double_scales':scales,'work_cpu_s':work,'reasons':reasons}
  for arm in ['after','before']:
   ratios=[a/b if b>0 else None for a,b in zip(arms[arm]['cpu_s_per_call'],arms[arm+'-aa']['cpu_s_per_call'])];reasons=arms[arm]['reasons']+['AA: '+s for s in arms[arm+'-aa']['reasons']]
   if any(x is None or not q['aa_bounds'][0]<=x<=q['aa_bounds'][1] for x in ratios):reasons.append('AA failed')
   arms[arm].update(aa_ratios=ratios,qualification_reasons=reasons,qualified=not reasons)
  a=arms['after']['cpu_s_per_call'];b=arms['before']['cpu_s_per_call'];ratios=[x/y if y>0 else None for x,y in zip(a,b)]
  cells.append({'case':case,'arms':arms,'paired_after_over_before':ratios,'median_ratio':statistics.median(ratios) if all(x is not None for x in ratios) else None,'qualified':arms['after']['qualified'] and arms['before']['qualified'],'nonoverlapping_ranges':max(a)<min(b) or max(b)<min(a)})
 return {'formal_processes':216,'distinct_verified_outputs':len(truth),'scope':plan['cpu_scope'],'cells':cells}
if __name__=='__main__':
 spec=importlib.util.spec_from_file_location('independent_oracle','/tmp/nxrt-conv-gap-analyze.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 plan=json.loads((ROOT/'plan.json').read_text());rows=json.loads((ROOT/'receipts.json').read_text());result=analyze(plan,rows,lambda name:(ROOT/name).read_bytes(),module.oracle);(ROOT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 for cell in result['cells']:print(cell['case'],*[round(cell['arms'][a]['median_cpu_us_per_call'],3) for a in ['after','before']],cell['median_ratio'],cell['qualified'],{a:cell['arms'][a]['qualification_reasons'] for a in ['after','before']})
