"""Darwin terminal-wait4 public-session CPU study; run under the shared flock."""
import fcntl,gzip,hashlib,itertools,json,math,os,pathlib,shutil,statistics,subprocess,time
ROOT=pathlib.Path('/tmp/nxrt-qmoe-workspace');OUT=pathlib.Path('/tmp/nxrt-qmoe-decoder-study')
sha=lambda data:hashlib.sha256(data).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
paths=[p for p in git('diff','--name-only').splitlines() if p.startswith('crates/')]
paths.append('crates/onnx-runtime-session/src/executor/workspace_context_tests.rs')
original={p:(ROOT/p).read_bytes() for p in paths}
base=git('rev-parse','HEAD')
env=os.environ|{'CARGO_TARGET_DIR':'/tmp/nxrt-erf-nan-target','CARGO_INCREMENTAL':'0','CARGO_BUILD_JOBS':'1','ORT_ROOT':'/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt'}
env['ONNX_GENAI_ORT_LIB']=env['ORT_ROOT']+'/lib/libonnxruntime.1.32.0.dylib'
for name in ['ONNX_GENAI_CPU_DECODE_THREADS','ONNX_GENAI_MOE_GROUPED_MIN_WORK','ONNX_GENAI_WEIGHT_OFFLOAD','ONNX_GENAI_WEIGHT_OFFLOAD_HOST_BYTES','ONNX_GENAI_WEIGHT_OFFLOAD_PREFETCH','ONNX_GENAI_QMOE_ROUTE_DUMP']:env.pop(name,None)
lock='/tmp/onnx-genai-hostlock-darwin.flock'
with open(lock,'a+') as attempt:
 try:fcntl.flock(attempt,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:pass
 else:raise AssertionError('whole-study wrapper does not own shared host flock')
hardware=json.loads(pathlib.Path('/tmp/nxrt-neon-unroll-process/hardware.json').read_text());assert hardware['hw.logicalcpu']=='10'
meta={'schema':1,'base':base,'source_sha256':{p:sha(v) for p,v in original.items()},'probe_sha256':sha((OUT/'probe/src/main.rs').read_bytes()),'rustc':subprocess.check_output(['rustc','-vV'],text=True),'hardware':hardware,'lock':lock,'lock_inode':os.stat(lock).st_ino,'lock_busy_control':True,'parent_pid':os.getpid(),'binaries':{},'foreign_core_monitor':'Unavailable on Darwin; no claim of an idle host','metric':'terminal child user+system CPU seconds, includes setup/work/validation/output IO/retirement; not pure inference latency'}
(OUT/'candidate.patch.gz').write_bytes(gzip.compress(subprocess.check_output(['git','diff','--binary',base],cwd=ROOT),mtime=0))
for label in ['baseline','candidate']:
 print('BUILD',label,flush=True)
 try:
  if label=='baseline':
   for p in paths:
    if p.endswith('/workspace_context_tests.rs'):continue
    (ROOT/p).write_bytes(subprocess.check_output(['git','show',base+':'+p],cwd=ROOT))
  with (OUT/(label+'-cargo.jsonl')).open('w') as log,(OUT/(label+'-build.log')).open('w') as err:
   subprocess.run(['cargo','build','--offline','--release','--manifest-path',str(OUT/'probe/Cargo.toml'),'--message-format=json-render-diagnostics'],cwd=ROOT,env=env,stdout=log,stderr=err,check=True)
  records=[json.loads(l) for l in (OUT/(label+'-cargo.jsonl')).read_text().splitlines() if l.startswith('{')]
  binaries=[r['executable'] for r in records if r.get('reason')=='compiler-artifact' and r.get('executable') and r['target']['name']=='nxrt-qmoe-fc-banks-probe'];assert len(binaries)==1
  shutil.copy2(binaries[0],OUT/(label+'.binary'))
  symbols=subprocess.check_output(['nm',str(OUT/(label+'.binary'))],text=True,stderr=subprocess.STDOUT)
  assert 'InferenceSession' in symbols and 'Mlas' not in symbols
  meta['binaries'][label]={'sha256':sha((OUT/(label+'.binary')).read_bytes()),'native':True,'symbol_count':len(symbols.splitlines())}
 finally:
  for p,v in original.items():(ROOT/p).write_bytes(v)
meta['probe_lock_sha256']=sha((OUT/'probe/Cargo.lock').read_bytes())
(OUT/'metadata.json').write_text(json.dumps(meta,indent=2))
# Both binaries use the same probe. null is a duplicate baseline process arm.
shapes=[(1,32,64,1,0,0,1),(1,64,128,1,1,0,1),(1,64,128,1,0,1,1),(8,128,256,1,0,0,1),(8,128,256,4,0,1,4),(8,128,256,4,1,1,4),(8,128,256,4,0,1,1),(8,128,256,1,0,1,4)]
protocol={'shapes_rows_hidden_inter_execution_width_dense_gate_construction_width':shapes,'arms':['baseline','candidate','null'],'repetitions':3,'work_macs':268435456,'k':2,'fixed_warm_calls':5,'setup_only_calls':0,'double_multiplier':2,'quality':{'aa':[.95,1.05],'setup_fraction_max':.25,'setup_adjusted_double':[1.7,2.3],'work_cpu_seconds_min':.05,'physical_cpu_wall_factor':11},'max_attempts':1,'quality_metrics':['cpu_s','wall_s'],'context_scope':'construction pool builds and initially prepares; execution pool prepares, warms and runs; physical peak and logical demand recorded separately','binary_scope':'two source-controlled optimized native builds; null runs baseline executable; this is not one-address-space A/B','cpu_scope':meta['metric']}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
rows=[]
for rep in range(3):
 for c,shape in enumerate(shapes):
  m,h,i,width,dense,gated,construction=shape;normal=max(1,protocol['work_macs']//(m*h*i*2*(3 if gated else 2)))
  # Rotate/reverse arms and work amounts, avoiding a fixed candidate position.
  kinds=['setup','work','double'];kinds=kinds[rep:]+kinds[:rep]
  for kind in kinds:
   arms=['baseline','candidate','null'] if kind!='double' else ['baseline','candidate']
   if (rep+c)%2:arms.reverse()
   for arm in arms:
    calls=0 if kind=='setup' else normal*(2 if kind=='double' else 1)
    stem=f'r{rep}-c{c}-{kind}-{arm}';output=OUT/(stem+'.f32');binary=OUT/(('baseline' if arm=='null' else arm)+'.binary')
    before_load=os.getloadavg();started=time.monotonic()
    with (OUT/(stem+'.stdout')).open('w') as stdout,(OUT/(stem+'.stderr')).open('w') as stderr:
     child=subprocess.Popen([str(binary),str(m),str(h),str(i),str(calls),str(width),str(output)],env=env|{'NXRT_GATHER_DENSE':str(dense),'NXRT_BLOCK_GATED':str(gated),'NXRT_CONSTRUCTION_WIDTH':str(construction)},stdout=stdout,stderr=stderr)
     pid,status,usage=os.wait4(child.pid,0);child.returncode=os.waitstatus_to_exitcode(status)
    wall=time.monotonic()-started;cpu=usage.ru_utime+usage.ru_stime
    assert child.returncode==0,(stem,(OUT/(stem+'.stderr')).read_text()[-4000:])
    receipt=json.loads((OUT/(stem+'.stdout')).read_text().strip().splitlines()[-1])
    assert receipt['construction_width']==construction
    assert receipt['calls']==calls and receipt['rayon_width']==width and receipt['final_used']==0
    assert (receipt['rows'],receipt['hidden'],receipt['inter'],int(receipt['dense']),int(receipt['gated']))==(m,h,i,dense,gated)
    raw=output.read_bytes();assert len(raw)==m*h*4
    witness=(OUT/(stem+'.f32.trace.json')).read_text();assert ('qmoe.streamed_rows' in witness)==(not dense)
    assert math.isfinite(cpu) and 0<cpu<=wall*11 and usage.ru_maxrss>0
    rows.append(receipt|{'rep':rep,'case':c,'kind':kind,'arm':arm,'exit':child.returncode,'cpu_s':cpu,'wall_s':wall,'maxrss_bytes':usage.ru_maxrss,'output_sha256':sha(raw),'load_before':before_load,'load_after':os.getloadavg()})
    (OUT/'receipts.json.gz').write_bytes(gzip.compress(json.dumps(rows,sort_keys=True).encode(),mtime=0))
    print('PROCESS',len(rows),stem,'CPU',round(cpu,4),flush=True)
assert len(rows)==192
assert sha((OUT/'probe/src/main.rs').read_bytes())==meta['probe_sha256']
assert sha((OUT/'probe/Cargo.lock').read_bytes())==meta['probe_lock_sha256']
assert all((ROOT/p).read_bytes()==v for p,v in original.items())
for label,b in meta['binaries'].items():assert sha((OUT/(label+'.binary')).read_bytes())==b['sha256']
meta['receipts_sha256']=sha(gzip.decompress((OUT/'receipts.json.gz').read_bytes()));meta['runner_sha256']=sha(pathlib.Path(__file__).read_bytes());meta['processes']=len(rows)
(OUT/'metadata.json').write_text(json.dumps(meta,indent=2))
print('DONE',len(rows),flush=True)
