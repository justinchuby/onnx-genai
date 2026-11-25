from pathlib import Path
import os,subprocess,json,hashlib,re,gzip
root=Path('/tmp/nxrt-qmoe-workspace');out=Path(__file__).parent
paths=['crates/onnx-runtime-ep-cpu/src/lib.rs','crates/onnx-runtime-ep-cpu/src/kernels/weight_cache.rs','crates/onnx-runtime-ep-cpu/src/kernels/block_quantized_moe.rs']
original={p:(root/p).read_bytes() for p in paths};base='d42eb1d4aef7cf56148bc87ad91b3571b33060df'
sha=lambda b:hashlib.sha256(b).hexdigest()
ort='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt'
env=os.environ|{'ORT_ROOT':ort,'ONNX_GENAI_ORT_LIB':ort+'/lib/libonnxruntime.1.32.0.dylib','NXRT_ORT_LIB_DIR':ort+'/lib','NXRT_REQUIRE_ORT_TESTS':'1','CARGO_INCREMENTAL':'0','CARGO_BUILD_JOBS':'1','CARGO_PROFILE_DEV_DEBUG':'0','CARGO_PROFILE_TEST_DEBUG':'0','CARGO_TARGET_DIR':'/tmp/nxrt-qmoe-main-target'}
env['PATH']='/opt/homebrew/Caskroom/miniconda/base/bin:'+env['PATH'];rows=[]
try:
 for arm in ['baseline','candidate']:
  for p in paths:
   raw=subprocess.check_output(['git','show',base+':'+p],cwd=root) if arm=='baseline' else original[p]
   if p.endswith('/lib.rs'):raw+=b'\n'+(out/'allocator.rs').read_bytes()
   if p.endswith('/weight_cache.rs'):raw+=b'\n'+(out/'probe.rs').read_bytes()
   (root/p).write_bytes(raw)
   (out/(arm+'-'+Path(p).name+'.gz')).write_bytes(gzip.compress(raw,mtime=0))
  command=['cargo','test','--offline','-p','onnx-runtime-ep-cpu','--lib','kernels::weight_cache::allocation_probe::warm_group_heap_calls','--','--exact','--nocapture']
  print('RUN',arm,flush=True)
  r=subprocess.run(command,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);(out/(arm+'.log')).write_text(r.stdout)
  print('\n'.join(l for l in r.stdout.splitlines() if l.startswith(('CONTROL','GROUP')) or 'test result:' in l) if r.returncode==0 else r.stdout[-5000:],flush=True)
  assert r.returncode==0
  assert '1 passed; 0 failed' in r.stdout
  counts=[dict(zip(['members','iterations','allocations','frees'],map(int,m))) for m in re.findall(r'GROUP members=(\d+) iterations=(\d+) allocations=(\d+) frees=(\d+) issuer_delta=0 final_used=0',r.stdout)]
  assert [r['members'] for r in counts]==[0,1,2,3,4,8]
  for row in counts:
   if row['members']==0 or (arm=='candidate' and row['members']<=3):assert row['allocations']==row['frees']==0
   else:assert row['allocations']>0 and row['frees']>0
  rows.append({'arm':arm,'command':command,'counts':counts,'log_sha256':sha(r.stdout.encode()),'exit':r.returncode})
finally:
 for p,raw in original.items():(root/p).write_bytes(raw)
 assert all((root/p).read_bytes()==raw for p,raw in original.items())
 print('RESTORED exact production source bytes.',flush=True)
(out/'results.json').write_text(json.dumps({'scope':'native debug private WeightCache warm hit allocation/free calls; no timing or whole-kernel allocation claim','baseline':base,'candidate_source_sha256':{p:sha(b) for p,b in original.items()},'results':rows,'production_restored':True},indent=2))
