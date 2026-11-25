from pathlib import Path
import os,sys,json,hashlib,fcntl,subprocess,time,math,platform,itertools,gzip
root=Path('/tmp/nxrt-qmoe-workspace');out=Path('/tmp/nxrt-conv-gap-study');build=json.loads((out/'build-metadata.json').read_text());sha=lambda b:hashlib.sha256(b).hexdigest()
assert build.get('temporary_example_removed') and len(build['binaries'])==2
lock=Path('/tmp/onnx-genai-hostlock-darwin.flock');stat=lock.stat();held=[]
for value in os.listdir('/dev/fd'):
 try:
  fd=int(value);s=os.fstat(fd)
  if s.st_dev==stat.st_dev and s.st_ino==stat.st_ino:held.append(fd)
 except (ValueError,OSError):pass
assert len(held)==1,held
fcntl.flock(held[0],fcntl.LOCK_EX|fcntl.LOCK_NB)
with lock.open('a+') as probe:
 try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:pass
 else:raise SystemExit('independent shared-lock exclusion probe failed')
env=os.environ|{'RAYON_NUM_THREADS':'1','ONNX_GENAI_MLAS_THREADPOOL_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'}
for k in ['NXRT_CPU_GEMM_BACKEND','NXRT_CPU_DISPATCH_LEDGER','ONNX_GENAI_PROFILE_OPS']:env.pop(k,None)
labels=[('production','default','native'),('production-aa','default','native'),('generic','linked','native'),('generic-aa','linked','native'),('reference','linked','reference'),('reference-aa','linked','reference')]
metadata={'schema':1,'base':build['base'],'source_sha256':build['source_sha256'],'probe_sha256':build['probe_sha256'],'binaries':build['binaries'],'platform':platform.platform(),'logical_cpus':os.cpu_count(),'phases':['reuse','new'],'cases':list(range(6)),'labels':labels,'repetitions':3,'target_body_cpu_s':0.25,'qualifications':{'min_work_cpu_s':0.10,'max_setup_fraction':0.25,'aa_ratio_bounds':[0.95,1.05],'double_scale_bounds':[1.7,2.3],'physical_capacity_slack_s':0.01},'cpu_scope':'terminal wait4 child user+system CPU, all threads; includes setup/validation/file output','shared_lock':str(lock),'lock_inode':stat.st_ino,'inherited_lock_fd':held[0],'parent_pid':os.getppid(),'lock_exclusion_control':True,'load_before':os.getloadavg(),'foreign_cpu_instrument':'unavailable on Darwin; lock and AA do not prove absence of foreign CPU','system_framework_thread_policy':'Rust and MLAS pools fixed at one; VECLIB_MAXIMUM_THREADS=1; other framework workers not directly observed','environment_overrides':{'RAYON_NUM_THREADS':'1','ONNX_GENAI_MLAS_THREADPOOL_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'},'completed':False,'calibration':[]}
(out/'plan.json').write_text(json.dumps(metadata,indent=2));rows=[];counter=0
for label,data in build['binaries'].items():assert sha(Path(data['path']).read_bytes())==data['sha256']
for p,d in build['source_sha256'].items():assert sha((root/p).read_bytes())==d

def one(label,binary,arm,case,phase,kind,rep,calls,pilot=False):
 global counter
 counter+=1;name=f'{counter:04d}-{label}-c{case}-{phase}-{kind}-r{rep}';folder=out/('pilot' if pilot else 'raw');folder.mkdir(exist_ok=True);result=folder/(name+'.f32');stdout=folder/(name+'.stdout');stderr=folder/(name+'.stderr');cmd=[build['binaries'][binary]['path'],arm,str(case),phase,str(calls),str(result)]
 with stdout.open('w') as f,stderr.open('w') as e:
  started=time.monotonic();child=subprocess.Popen(cmd,cwd=root,env=env,stdout=f,stderr=e,pass_fds=(held[0],));pid,status,usage=os.wait4(child.pid,0);child.returncode=os.waitstatus_to_exitcode(status)
 wall=time.monotonic()-started;cpu=usage.ru_utime+usage.ru_stime
 record={'label':label,'binary':binary,'arm':arm,'case':case,'phase':phase,'kind':kind,'rep':rep,'calls':calls,'pid':pid,'exit':child.returncode,'cpu_s':cpu,'wall_s':wall,'peak_rss_bytes':usage.ru_maxrss if sys.platform=='darwin' else usage.ru_maxrss*1024,'stdout':str(stdout.relative_to(out)),'stderr':str(stderr.relative_to(out)),'output':str(result.relative_to(out)),'command':cmd,'pilot':pilot}
 if child.returncode:
  record['failure']=stderr.read_text()[-5000:];(out/'failed-process.json').write_text(json.dumps(record,indent=2));raise RuntimeError('Conv child failed: '+record['failure'])
 text=stdout.read_text().strip().splitlines();assert len(text)==1,text;probe=json.loads(text[0]);record['probe']=probe
 assert (probe['arm'],probe['case'],probe['phase'],probe['calls'],probe['linked'])==(arm,case,phase,calls,binary=='linked')
 assert cpu>0 and wall>0 and usage.ru_maxrss>0 and cpu<=metadata['logical_cpus']*wall+0.01
 assert math.isfinite(probe['max_abs_f64_error']) and probe['max_abs_f64_error']>=0
 data=result.read_bytes();assert len(data)==math.prod(probe['shape_y'])*4 and data
 digest=0xcbf29ce484222325
 for b in data:digest=((digest^b)*0x100000001b3)&((1<<64)-1)
 assert probe['digest']==f'{digest:016x}'
 record['output_sha256']=sha(data);record['stdout_sha256']=sha(stdout.read_bytes());record['stderr_sha256']=sha(stderr.read_bytes())
 # Preserve every raw output in compressed form; retain exact-byte hashes.
 zipped=result.with_suffix('.f32.gz');zipped.write_bytes(gzip.compress(data,mtime=0));result.unlink();record['output']=str(zipped.relative_to(out))
 if binary=='linked':
  assert len(probe['native_gemm_witnesses'])==2 and probe['mlas_gemm_witnesses'][0]==0
  if case in [1,2,4]:assert probe['native_gemm_witnesses'][0]>0
 else:assert len(probe['native_gemm_witnesses'])==1
 if pilot:metadata['calibration'].append(record);(out/'plan.json').write_text(json.dumps(metadata,indent=2))
 else:rows.append(record);(out/'receipts.json').write_text(json.dumps(rows,indent=2))
 return record

counts={}
for case,phase in itertools.product(range(6),metadata['phases']):
 for label,binary,arm in [labels[0],labels[2],labels[4]]:
  print('CALIBRATE',case,phase,label,flush=True);setup=one(label,binary,arm,case,phase,'setup',0,0,True);calls=64
  while True:
   work=one(label,binary,arm,case,phase,'pilot',0,calls,True);delta=work['cpu_s']-setup['cpu_s']
   if work['cpu_s']>=max(0.03,3*setup['cpu_s']) and delta>0:break
   calls*=4;assert calls<=1048576,'calibration cannot establish body CPU above setup'
  target=max(1,math.ceil(metadata['target_body_cpu_s']/(delta/calls)));assert target<=10000000
  counts[f'{binary}/{arm}/{case}/{phase}']=target
metadata['calls_by_arm']=counts;(out/'plan.json').write_text(json.dumps(metadata,indent=2));print('CALIBRATION COMPLETE',len(counts),flush=True)
for rep in range(3):
 axes=list(itertools.product(range(6),metadata['phases']));axes=axes[rep:]+axes[:rep]
 for case,phase in axes:
  ordered=labels[rep:]+labels[:rep]
  for slot,(label,binary,arm) in enumerate(ordered):
   calls=counts[f'{binary}/{arm}/{case}/{phase}'];kinds=['setup','work','double'] if (slot+rep)%2==0 else ['double','work','setup']
   for kind in kinds:one(label,binary,arm,case,phase,kind,rep,0 if kind=='setup' else calls*(2 if kind=='double' else 1))
  print('COMPLETE CELL',rep,case,phase,'processes',len(rows),flush=True)
  for p,d in build['source_sha256'].items():assert sha((root/p).read_bytes())==d
  for _,data in build['binaries'].items():assert sha(Path(data['path']).read_bytes())==data['sha256']
assert len(rows)==648
metadata['completed']=True;metadata['formal_processes']=len(rows);metadata['load_after']=os.getloadavg();metadata['runner_sha256']=sha(Path(__file__).read_bytes());metadata['receipts_sha256']=sha((out/'receipts.json').read_bytes());(out/'plan.json').write_text(json.dumps(metadata,indent=2));print('COMPLETE MATRIX',len(rows),flush=True)
