import csv,fcntl,gzip,hashlib,io,json,os,pathlib,platform,resource,statistics,subprocess,sys,time
root=pathlib.Path('/tmp/nxrt-qmoe-workspace')
out=pathlib.Path('/tmp/nxrt-neon-unroll-clock')
paths=['crates/onnx-runtime-ep-cpu/src/kernels/softmax_neon.rs','crates/onnx-runtime-ep-cpu/src/kernels/softmax_neon_unroll_tests.rs']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
sources={p:sha(root/p) for p in paths}
env=os.environ|{'CARGO_TARGET_DIR':'/tmp/nxrt-erf-nan-target','CARGO_INCREMENTAL':'0','CARGO_BUILD_JOBS':'1','ORT_ROOT':'/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt'}
lock_path=pathlib.Path('/tmp/onnx-genai-hostlock-darwin.flock')
with lock_path.open('a+') as probe:
 try: fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError: pass
 else: raise SystemExit('shared flock is not already held by the outer whole-study wrapper')
metadata={'schema':2,'protocol_phase':'clock-process-cpu-and-physical-bound','source_sha256':sources,'parent':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'platform':platform.platform(),'rustc':subprocess.check_output(['rustc','-Vv'],text=True),'lock':str(lock_path),'lock_inode':lock_path.stat().st_ino,'lock_parent_pid':os.getppid(),'lock_busy_control':True,'load_before':os.getloadavg(),'foreign_cpu_instrument':'unavailable on Darwin; no claim of foreign CPU absence','receipts':[],'arms':{'0':'original_same_arithmetic','1':'unrolled_native','2':'unrolled_native_aa','3':'mlas','4':'original_outlined'},'cpu_unit':'ns; CLOCK_PROCESS_CPUTIME_ID, all threads','repetitions':15,'sweeps':3}
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
test='kernels::softmax::softmax_neon::unroll_tests::softmax_unroll_cpu_ab'
listing=run([str(binary),test,'--exact','--ignored','--list'],'selector.log')
assert [x for x in listing.splitlines() if x.endswith(': test')]==[test+': test']
rows=[];valids=[]
for sweep in range(3):
 for k in range(3):
  width=[1,4,8][(k+sweep)%3];name=f's{sweep}-w{width}'
  child_env=env|{'NXRT_UNROLL_WIDTH':str(width),'NXRT_UNROLL_ORDER':str(sweep)}
  print('RUN',name,flush=True)
  with (out/(name+'.stdout')).open('w') as stdout,(out/(name+'.stderr')).open('w') as stderr:
   start=time.monotonic();p=subprocess.Popen([str(binary),test,'--exact','--ignored','--nocapture'],cwd=root,env=child_env,stdout=stdout,stderr=stderr)
   pid,status,usage=os.wait4(p.pid,0);p.returncode=os.waitstatus_to_exitcode(status)
  receipt={'sweep':sweep,'width':width,'pid':pid,'exit':p.returncode,'wall_s':time.monotonic()-start,'cpu_s':usage.ru_utime+usage.ru_stime,'maxrss_bytes':usage.ru_maxrss,'load_after':os.getloadavg()}
  text=(out/(name+'.stdout')).read_text();error=(out/(name+'.stderr')).read_text()
  if p.returncode or f'test {test} ... ok' not in text:print(text[-3000:]+error[-3000:],flush=True);raise SystemExit('diagnostic failed or did not run')
  fpcr=[int(line.split(',')[1]) for line in text.splitlines() if line.startswith('UNROLL_FPCR,')];assert len(fpcr)==1 and not(fpcr[0]&((1<<1)|(3<<22)|(1<<24)|(1<<25)))
  cpus=[int(line.split(',')[1]) for line in text.splitlines() if line.startswith('UNROLL_CPUS,')];assert cpus==[int(metadata['hw.logicalcpu'])]
  receipt['cpus']=cpus[0];receipt['fpcr_start']=fpcr[0];receipt['fpcr_end_equality_assertion_passed']=True
  samples=[line.split(',')[1:] for line in text.splitlines() if line.startswith('UNROLL_SAMPLE,')]
  validations=[line.split(',')[1:] for line in text.splitlines() if line.startswith('UNROLL_VALID,')]
  assert len(samples)==1200 and len(validations)==80
  for row in samples: rows.append([sweep]+row)
  for row in validations: valids.append([sweep]+row)
  metadata['receipts'].append(receipt);print('PASS',name,receipt,flush=True)
  (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
assert {p:sha(root/p) for p in paths}==sources
assert sha(binary)==metadata['binary_sha256']
metadata['load_after']=os.getloadavg();metadata['runner_sha256']=sha(pathlib.Path(__file__))
stream=io.StringIO();writer=csv.writer(stream);writer.writerow(['sweep','width','serial','rows','cols','iters','rep','arm','cpu_ns','wall_ns','token']);writer.writerows(rows)
raw=stream.getvalue().encode();(out/'samples.csv.gz').write_bytes(gzip.compress(raw,mtime=0));metadata['raw_sha256']=hashlib.sha256(raw).hexdigest()
(out/'validations.json').write_text(json.dumps(valids));(out/'metadata.json').write_text(json.dumps(metadata,indent=2))
print('COMPLETE',len(rows),'samples',len(valids),'validations',flush=True)
