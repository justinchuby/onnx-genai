from pathlib import Path
import json,hashlib,gzip,math,statistics,itertools,sys,struct
ROOT=Path('/tmp/nxrt-conv-gap-study')
LABELS={'production':('default','native'),'production-aa':('default','native'),'generic':('linked','native'),'generic-aa':('linked','native'),'reference':('linked','reference'),'reference-aa':('linked','reference')}
PHASES=['reuse','new'];KINDS=['setup','work','double']
CASES=[([1,8,16,16],[16,8,3,3],1,1,1,1,False),([1,8,16,16],[16,8,3,3],1,2,1,2,False),([2,16,12,12],[24,8,3,3],2,1,1,1,False),([1,16,16,16],[16,1,3,3],16,1,1,1,False),([1,256,4,4],[128,256,1,1],1,1,1,0,False),([1,8,1,128],[16,8,1,5],1,1,2,2,True)]
sha=lambda b:hashlib.sha256(b).hexdigest()
def values(n,seed):
 state=seed;out=[]
 for _ in range(n):
  state=(state*6364136223846793005+1442695040888963407)&((1<<64)-1)
  v=struct.unpack('<f',struct.pack('<f',((state>>32)/((1<<32)-1)-0.5)))[0]*0.5;out.append(v)
 return out

def oracle(case):
 xs,ws,g,d,s,p,rank3=CASES[case];b,ic,ih,iw=xs;oc,ig,kh,kw=ws;oh=1 if rank3 else (ih+2*p-((kh-1)*d+1))//s+1;ow=(iw+2*p-((kw-1)*d+1))//s+1
 x=values(math.prod(xs),71);w=values(math.prod(ws),101);bias=values(oc,37);out=[]
 for bb in range(b):
  for cc in range(oc):
   gg=cc//(oc//g)
   for yy in range(oh):
    for zz in range(ow):
     total=bias[cc]
     for i in range(ig):
      for ky in range(kh):
       py=0 if rank3 else yy*s+ky*d-p
       if not 0<=py<ih:continue
       for kx in range(kw):
        px=zz*s+kx*d-p
        if 0<=px<iw:total+=x[((bb*ic+gg*ig+i)*ih+py)*iw+px]*w[((cc*ig+i)*kh+ky)*kw+kx]
     out.append(total)
 return out

def analyze(plan,rows,verify_files=False,read_file=None):
 if read_file is None:read_file=lambda p:(ROOT/p).read_bytes()
 assert plan['completed'] and plan['formal_processes']==648 and plan['repetitions']==3
 expected=set(itertools.product(LABELS,range(6),PHASES,KINDS,range(3)));seen={};digests={};oracles={}
 assert len(rows)==len(expected)
 for r in rows:
  key=(r['label'],r['case'],r['phase'],r['kind'],r['rep']);assert key in expected and key not in seen,key;seen[key]=r
  binary,arm=LABELS[r['label']];assert (r['binary'],r['arm'])==(binary,arm)
  n=plan['calls_by_arm'][f'{binary}/{arm}/{r["case"]}/{r["phase"]}'];calls=0 if r['kind']=='setup' else n*(2 if r['kind']=='double' else 1);assert r['calls']==calls and (calls>0)==(r['kind']!='setup')
  assert r['exit']==0 and r['pid']>0 and r['peak_rss_bytes']>0
  assert math.isfinite(r['cpu_s']) and math.isfinite(r['wall_s']) and r['cpu_s']>0 and r['wall_s']>0 and r['cpu_s']<=plan['logical_cpus']*r['wall_s']+.01
  p=r['probe'];assert (p['arm'],p['case'],p['phase'],p['calls'],p['linked'])==(arm,r['case'],r['phase'],calls,binary=='linked')
  assert math.isfinite(p['max_abs_f64_error']) and p['max_abs_f64_error']>=0
  assert len(p['native_gemm_witnesses'])==(2 if binary=='linked' else 1)
  if binary=='linked':
   assert p['mlas_gemm_witnesses'][0]==0
   if r['case'] in [1,2,4]:assert p['native_gemm_witnesses'][0]>0
  identity=(binary,arm,r['case']);digests.setdefault(identity,set()).add(r['output_sha256']);assert len(digests[identity])==1,'unstable output bytes within arm'
  assert len(r['output_sha256'])==64 and len(r['stdout_sha256'])==64 and len(r['stderr_sha256'])==64
  assert r['command']==[plan['binaries'][binary]['path'],arm,str(r['case']),r['phase'],str(calls),str(Path(plan['binaries'][binary]['path']).parent/r['output']).removesuffix('.gz')]
  if verify_files:
   assert sha(read_file(r['stdout']))==r['stdout_sha256'] and sha(read_file(r['stderr']))==r['stderr_sha256']
   data=gzip.decompress(read_file(r['output']));assert sha(data)==r['output_sha256'] and len(data)==math.prod(p['shape_y'])*4
   if identity not in oracles:
    truth=oracle(r['case']);actual=[x[0] for x in struct.iter_unpack('<f',data)];assert len(actual)==len(truth)
    assert all(math.isfinite(a) and abs(a-b)<=2e-5*(1+abs(b)) for a,b in zip(actual,truth));oracles[identity]=True
 assert set(seen)==expected
 cells=[];q=plan['qualifications']
 for case,phase in itertools.product(range(6),PHASES):
  arms={}
  for label in LABELS:
   slopes=[];fractions=[];scales=[];durations=[];rss=[]
   for rep in range(3):
    ss=seen[label,case,phase,'setup',rep];ww=seen[label,case,phase,'work',rep];dd=seen[label,case,phase,'double',rep];delta=ww['cpu_s']-ss['cpu_s']
    slopes.append(delta/ww['calls']);fractions.append(ss['cpu_s']/ww['cpu_s']);scales.append((dd['cpu_s']-ss['cpu_s'])/delta if delta>0 else None);durations.append(ww['cpu_s']);rss.extend([ss['peak_rss_bytes'],ww['peak_rss_bytes'],dd['peak_rss_bytes']])
   reasons=[]
   if any(x<=0 for x in slopes):reasons.append('nonpositive setup-adjusted body')
   if any(x<q['min_work_cpu_s'] for x in durations):reasons.append('short work CPU')
   if any(x>q['max_setup_fraction'] for x in fractions):reasons.append('setup exceeds fraction bar')
   lo,hi=q['double_scale_bounds']
   if any(x is None or not lo<=x<=hi for x in scales):reasons.append('double-work proportionality failed')
   arms[label]={'cpu_s_per_call':slopes,'median_cpu_us_per_call':statistics.median(slopes)*1e6,'range_cpu_us_per_call':[min(slopes)*1e6,max(slopes)*1e6],'setup_fractions':fractions,'double_scale':scales,'work_cpu_s':durations,'absolute_peak_rss_bytes_range':[min(rss),max(rss)],'reasons':reasons}
  for main in ['production','generic','reference']:
   aa=main+'-aa';ratios=[a/b if b>0 else None for a,b in zip(arms[main]['cpu_s_per_call'],arms[aa]['cpu_s_per_call'])];lo,hi=q['aa_ratio_bounds'];reasons=arms[main]['reasons']+['null: '+x for x in arms[aa]['reasons']]
   if not all(r is not None and lo<=r<=hi for r in ratios):reasons.append('AA outside 5% in at least one repetition')
   arms[main]['aa_ratios']=ratios;arms[main]['qualified']=not reasons;arms[main]['qualification_reasons']=reasons
  comparisons={}
  for a,b in [('production','reference'),('generic','reference')]:
   av=arms[a]['cpu_s_per_call'];bv=arms[b]['cpu_s_per_call'];ratios=[x/y if y>0 else None for x,y in zip(av,bv)];comparisons[a+'_over_'+b]={'paired_ratios':ratios,'median_ratio':statistics.median(ratios) if all(x is not None for x in ratios) else None,'qualified':arms[a]['qualified'] and arms[b]['qualified'],'nonoverlapping_ranges':min(av)>max(bv) or max(av)<min(bv)}
  cells.append({'case':case,'phase':phase,'arms':arms,'comparisons':comparisons})
 return {'schema':1,'cpu_scope':plan['cpu_scope'],'qualification_policy':'conservative: every repetition must pass AA, setup, duration and double-work bars; all failures retained','cells':cells,'formal_processes':len(rows),'verified_distinct_outputs':len(oracles) if verify_files else None}
if __name__=='__main__':
 plan=json.loads((ROOT/'plan.json').read_text());rows=json.loads((ROOT/'receipts.json').read_text());result=analyze(plan,rows,True);(ROOT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
 for cell in result['cells']:
  arms=cell['arms'];p=cell['comparisons']['production_over_reference'];g=cell['comparisons']['generic_over_reference'];print(cell['case'],cell['phase'],'us',*[round(arms[a]['median_cpu_us_per_call'],3) for a in ['production','generic','reference']],'prod/ref',round(p['median_ratio'],3),p['qualified'],'generic/ref',round(g['median_ratio'],3),g['qualified'],{a:arms[a]['qualification_reasons'] for a in ['production','generic','reference'] if not arms[a]['qualified']})
