from pathlib import Path
import os,json,hashlib,subprocess,time,gzip,fcntl,math,itertools
root=Path('/tmp/nxrt-qmoe-workspace');out=Path('/tmp/nxrt-im2col-perf');sha=lambda b:hashlib.sha256(b).hexdigest();build=json.loads((out/'build-metadata.json').read_text());assert sha((out/'probe').read_bytes())==build['binary_sha256'];assert sha((root/'crates/onnx-runtime-ep-cpu/src/kernels/conv_ref.rs').read_bytes())==build['source_sha256']['after.rs']
lock=Path('/tmp/onnx-genai-hostlock-darwin.flock');st=lock.stat();fds=[]
for v in os.listdir('/dev/fd'):
 try:
  f=int(v);s=os.fstat(f)
  if (s.st_dev,s.st_ino)==(st.st_dev,st.st_ino):fds.append(f)
 except (ValueError,OSError):pass
assert len(fds)==1;fcntl.flock(fds[0],fcntl.LOCK_EX|fcntl.LOCK_NB)
with lock.open('a+') as independent:
 try:fcntl.flock(independent,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:pass
 else:raise AssertionError('shared-lock exclusion failed')
env=os.environ|{'RAYON_NUM_THREADS':'1','ONNX_GENAI_MLAS_THREADPOOL_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'}
for k in ['NXRT_CPU_GEMM_BACKEND','NXRT_CPU_DISPATCH_LEDGER','ONNX_GENAI_PROFILE_OPS']:env.pop(k,None)
plan={'hardware':json.loads((out/'hardware.json').read_text()),'base':build['base'],'binary_sha256':build['binary_sha256'],'lock_inode':st.st_ino,'lock_exclusion':True,'logical_cpus':os.cpu_count(),'cpu_scope':'terminal wait4 child user+system, all threads; setup-adjusted complete execute, warmed kernel','target_cpu_s':.3,'repetitions':3,'labels':['after','before','after-aa','before-aa'],'cases':list(range(6)),'qualifications':{'minimum_work_cpu_s':.1,'maximum_setup_fraction':.25,'doubling_bounds':[1.7,2.3],'aa_bounds':[.95,1.05]},'foreign_cpu_instrument':'unavailable on Darwin; no assertion of absent foreign work','load_before':os.getloadavg(),'calibration':[],'calls':{},'completed':False};rows=[];counter=0

def one(label,case,kind,rep,calls,pilot=False):
 global counter
 counter+=1;arm=label.removesuffix('-aa');name=f'{counter:04d}-{label}-c{case}-{kind}-r{rep}';folder=out/('pilot' if pilot else 'raw');folder.mkdir(exist_ok=True);output=folder/(name+'.f32');stdout=folder/(name+'.stdout');stderr=folder/(name+'.stderr');cmd=[str(out/'probe'),arm,str(case),'reuse',str(calls),str(output)]
 with stdout.open('w') as f,stderr.open('w') as e:
  start=time.monotonic();child=subprocess.Popen(cmd,cwd=root,env=env,stdout=f,stderr=e,pass_fds=tuple(fds));pid,status,usage=os.wait4(child.pid,0);child.returncode=os.waitstatus_to_exitcode(status)
 wall=time.monotonic()-start;cpu=usage.ru_utime+usage.ru_stime
 assert child.returncode==0,(cmd,stderr.read_text());p=json.loads(stdout.read_text());assert (p['arm'],p['case'],p['phase'],p['calls'],p['linked'])==(arm,case,'reuse',calls,False)
 assert p['mlas_gemm_witnesses']==[0,0] and len(p['native_gemm_witnesses'])==2
 if case in [1,2,4]:assert all(x>0 for x in p['native_gemm_witnesses'])
 assert 0<cpu<=os.cpu_count()*wall+.01 and usage.ru_maxrss>0 and math.isfinite(cpu)
 data=output.read_bytes();compressed=output.with_suffix('.f32.gz');compressed.write_bytes(gzip.compress(data,mtime=0));output.unlink()
 return {'label':label,'arm':arm,'case':case,'kind':kind,'rep':rep,'calls':calls,'pilot':pilot,'command':cmd,'pid':pid,'exit':child.returncode,'cpu_s':cpu,'wall_s':wall,'rss_bytes':usage.ru_maxrss,'probe':p,'stdout':str(stdout.relative_to(out)),'stderr':str(stderr.relative_to(out)),'output':str(compressed.relative_to(out)),'stdout_sha256':sha(stdout.read_bytes()),'stderr_sha256':sha(stderr.read_bytes()),'output_sha256':sha(data)}
for case,arm in itertools.product(range(6),['after','before']):
 setup=one(arm,case,'setup',0,0,True);plan['calibration'].append(setup);n=64
 while True:
  trial=one(arm,case,'pilot',0,n,True);plan['calibration'].append(trial);delta=trial['cpu_s']-setup['cpu_s']
  if delta>.12:break
  n*=4;assert n<=1<<24
 plan['calls'][f'{arm}/{case}']=max(1,math.ceil(n*.3/delta))
(out/'plan.json').write_text(json.dumps(plan,indent=2))
labels=plan['labels'];kinds=['setup','work','double']
for rep in range(3):
 for case in range(6):
  for label in labels[rep:]+labels[:rep]:
   n=plan['calls'][f'{label.removesuffix("-aa")}/{case}']
   for kind in kinds[rep:]+kinds[:rep]:
    calls=0 if kind=='setup' else n*(2 if kind=='double' else 1);r=one(label,case,kind,rep,calls);rows.append(r);print('DONE',len(rows),label,case,kind,rep,flush=True)
assert len(rows)==216 and sha((out/'probe').read_bytes())==build['binary_sha256'];(out/'receipts.json').write_text(json.dumps(rows,indent=2));plan.update(completed=True,formal_processes=len(rows),load_after=os.getloadavg(),receipts_sha256=sha((out/'receipts.json').read_bytes()),runner_sha256=sha(Path(__file__).read_bytes()));(out/'plan.json').write_text(json.dumps(plan,indent=2));print('complete',len(rows),'formal',len(plan['calibration']),'calibration',flush=True)
