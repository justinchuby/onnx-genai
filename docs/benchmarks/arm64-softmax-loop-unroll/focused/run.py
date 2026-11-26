import itertools,csv,fcntl,gzip,hashlib,io,json,os,pathlib,platform,resource,statistics,subprocess,sys,time
root=pathlib.Path('/tmp/nxrt-qmoe-workspace')
out=pathlib.Path('/tmp/nxrt-neon-unroll-process-focus')
paths=['crates/onnx-runtime-ep-cpu/src/kernels/softmax_neon.rs','crates/onnx-runtime-ep-cpu/src/kernels/softmax_neon_unroll_tests.rs']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
sources={p:sha(root/p) for p in paths}
env=os.environ|{'CARGO_TARGET_DIR':'/tmp/nxrt-erf-nan-target','CARGO_INCREMENTAL':'0','CARGO_BUILD_JOBS':'1','ORT_ROOT':'/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt'}
lock_path=pathlib.Path('/tmp/onnx-genai-hostlock-darwin.flock')
with lock_path.open('a+') as probe:
 try: fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError: pass
 else: raise SystemExit('shared flock is not already held by the outer whole-study wrapper')
metadata={'schema':4,'protocol_phase':'independent-focused-repeat-width8-case5','source_sha256':sources,'parent':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'platform':platform.platform(),'rustc':subprocess.check_output(['rustc','-Vv'],text=True),'lock':str(lock_path),'lock_inode':lock_path.stat().st_ino,'lock_parent_pid':os.getppid(),'lock_busy_control':True,'load_before':os.getloadavg(),'foreign_cpu_instrument':'unavailable on Darwin; no claim of foreign CPU absence','receipts':[],'arms':{'0':'original_same_arithmetic','1':'unrolled_native','2':'unrolled_native_aa','3':'mlas','4':'original_outlined'},'cpu_unit':'seconds; terminal wait4 child user+system CPU, all threads, includes setup and validation','repetitions':3,'work_elements':134217728}
metadata.update(json.loads((out/'hardware.json').read_text()))
(out/'metadata.json').write_text(json.dumps(metadata,indent=2))
def run(cmd,name,identities=()):
 print('RUN',name,flush=True)
 with (out/name).open('w') as log: result=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
 text=(out/name).read_text()
 if result.returncode: print(text[-5000:],flush=True);raise SystemExit(result.returncode)
 for identity in identities:
  assert f'test {identity} ... ok' in text,(name,identity)
 print('PASS',name,text[-250:],flush=True)
 return text
identity='kernels::softmax::softmax_neon::unroll_tests::unrolled_rows_preserve_original_bits_and_inplace_tail_bounds'
run(['cargo','test','--offline','-p','onnx-runtime-ep-cpu','--lib',identity,'--','--exact','--nocapture'],'native-bits.log',[identity])
print('RUN optimized build',flush=True)
with (out/'cargo.jsonl').open('w') as log,(out/'build.log').open('w') as err:
 result=subprocess.run(['cargo','test','--offline','-p','onnx-runtime-ep-cpu','--release','--features','mlas','--lib','--no-run','--message-format=json-render-diagnostics'],cwd=root,env=env,stdout=log,stderr=err)
if result.returncode:print((out/'build.log').read_text()[-5000:],flush=True);raise SystemExit(result.returncode)
records=[json.loads(line) for line in (out/'cargo.jsonl').read_text().splitlines() if line.startswith('{')]
artifacts=[r for r in records if r.get('reason')=='compiler-artifact' and r.get('target',{}).get('name')=='onnx_runtime_ep_cpu' and r.get('executable') and r['profile']['test']]
assert len(artifacts)==1,artifacts
binary=pathlib.Path(artifacts[0]['executable']);metadata['binary']=str(binary);metadata['binary_sha256']=sha(binary)
(out/'binary').write_text(str(binary)+'\n');print('PASS optimized build',flush=True)
run([str(binary),'kernels::softmax','--nocapture'],'optimized-softmax.log',[identity])

test='kernels::softmax::softmax_neon::unroll_tests::softmax_unroll_whole_process'
listing=run([str(binary),test,'--exact','--ignored','--list'],'selector.log')
assert [x for x in listing.splitlines() if x.endswith(': test')]==[test+': test']
shapes=[(1,4),(1,7),(1,8),(1,16),(1,257),(32,1024),(128,4099),(16,65539)]
rows=[]
def one(rep,width,case,arm,kind):
 n,d=shapes[case];calls=0 if kind=='setup' else max(1,metadata['work_elements']//(n*d))*(2 if kind=='double' else 1)
 name=f'{rep}-w{width}-c{case}-a{arm}-{kind}'
 child_env=env|{'NXRT_UNROLL_WIDTH':str(width),'NXRT_UNROLL_CASE':str(case),'NXRT_UNROLL_ARM':str(arm),'NXRT_UNROLL_CALLS':str(calls)}
 with (out/(name+'.stdout')).open('w') as stdout,(out/(name+'.stderr')).open('w') as stderr:
  started=time.monotonic();p=subprocess.Popen([str(binary),test,'--exact','--ignored','--nocapture'],cwd=root,env=child_env,stdout=stdout,stderr=stderr)
  pid,status,usage=os.wait4(p.pid,0);p.returncode=os.waitstatus_to_exitcode(status)
 wall=time.monotonic()-started;cpu=usage.ru_utime+usage.ru_stime
 text=(out/(name+'.stdout')).read_text();err=(out/(name+'.stderr')).read_text()
 if p.returncode or f'test {test} ... ok' not in text:print(text[-2500:]+err[-2500:],flush=True);raise SystemExit('whole-process diagnostic failed or did not run')
 lines=[line.split(',')[1:] for line in text.splitlines() if line.startswith('UNROLL_PROCESS,')];assert len(lines)==1
 w,c,rn,rd,a,actual_calls,digest,worst,workers,hits,fpcr=lines[0]
 w,c,rn,rd,a,actual_calls,digest,workers,hits,fpcr=map(int,[w,c,rn,rd,a,actual_calls,digest,workers,hits,fpcr])
 assert (w,c,rn,rd,a,actual_calls)==(width,case,n,d,arm,calls)
 assert cpu>0 and wall>0 and cpu<=wall*(int(metadata['hw.logicalcpu'])+1) and usage.ru_maxrss>0
 assert digest>0 and 0<=float(worst)<=1e-6 and not(fpcr&((1<<1)|(3<<22)|(1<<24)|(1<<25)))
 assert (workers>0)==(width>1 and n>=2 and n*d>=16384)
 assert hits==(n*calls if arm!=3 and d>=8 else 0)
 row={'rep':rep,'width':width,'case':case,'rows':n,'cols':d,'arm':arm,'kind':kind,'calls':calls,'cpu_s':cpu,'wall_s':wall,'maxrss_bytes':usage.ru_maxrss,'digest':digest,'worst_error':float(worst),'workers':workers,'hits':hits,'fpcr':fpcr,'pid':pid,'exit':p.returncode}
 rows.append(row)
for rep in range(3):
 axes=[(8,5)]
 for width,case in axes:
  print('RUN fixed work',rep,width,case,flush=True)
  for slot in range(5):
   arm=(slot+rep)%5
   for kind in (['setup','work'] if (slot+rep)%2==0 else ['work','setup']):one(rep,width,case,arm,kind)
  print('PASS fixed work',rep,width,case,flush=True)
for rep in range(3):
 for width,case,arm in itertools.product([8],[5],[0,1]):
  one(rep,width,case,arm,'double')
 print('PASS work scaling',rep,flush=True)
assert len(rows)==36
assert {p:sha(root/p) for p in paths}==sources and sha(binary)==metadata['binary_sha256']
metadata['load_after']=os.getloadavg();metadata['runner_sha256']=sha(pathlib.Path(__file__))
raw=json.dumps(rows,sort_keys=True).encode();(out/'receipts.json.gz').write_bytes(gzip.compress(raw,mtime=0));metadata['receipts_sha256']=hashlib.sha256(raw).hexdigest();metadata['processes']=len(rows)
(out/'metadata.json').write_text(json.dumps(metadata,indent=2));print('COMPLETE whole-process matrix',len(rows),flush=True)
