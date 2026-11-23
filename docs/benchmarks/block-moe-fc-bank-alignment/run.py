import subprocess,os,pathlib,json,hashlib,shutil,statistics
p=pathlib.Path('/tmp/nxrt-bank-alignment');r=pathlib.Path('/tmp/nxrt-qmoe-workspace')
env=dict(os.environ,CARGO_TARGET_DIR='/tmp/nxrt-erf-nan-target',CARGO_INCREMENTAL='0',ORT_ROOT='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt');env['ONNX_GENAI_ORT_LIB']=env['ORT_ROOT']+'/lib/libonnxruntime.1.32.0.dylib'
for key in ['ONNX_GENAI_MOE_GROUPED_MIN_WORK','ONNX_GENAI_CPU_DECODE_THREADS']:env.pop(key,None)
shapes=[(8,128,256,4,0,1,'block'),(8,128,256,4,1,2,'block')]
protocol={'hypothesis':'Does requesting 128-byte backing alignment change the streamed shared-bank regression? Default backing may already be aligned; actual address modulus is recorded. This does not prove hardware false-sharing counters.','shapes':shapes,'arms':['before','banked','null','before128','banked128'],'processes_per_case':5,'construction_and_batch_start_rotations':[0,1,2,3,4],'batches':10,'target_cpu_ms':250,'pilot_minimum_cpu_ms':50,'duration_gate_ms':20,'cpu_wall_minimum':0.9,'control_and_AA_gate_percent':5,'attempts':1,'scope':'native Apple M1 Max, synthetic public strict prepare/run/output drop, one executable and CPU crate, separate books and funded owners; no model/CUDA/RSS claim','lock':'/tmp/onnx-genai-hostlock-darwin.flock'}
(p/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
with (p/'build.log').open('w') as log:result=subprocess.run(['cargo','build','--offline','--release','--manifest-path',str(p/'probe/Cargo.toml')],env=env,cwd=r,stdout=log,stderr=subprocess.STDOUT)
if result.returncode:print((p/'build.log').read_text()[-6000:],flush=True);raise SystemExit(result.returncode)
binary=p/'alignment-probe';shutil.copy2(pathlib.Path(env['CARGO_TARGET_DIR'])/'release/nxrt-bank-alignment-probe',binary)
sha=lambda b:hashlib.sha256(b).hexdigest()
files=subprocess.check_output(['git','diff','--name-only'],cwd=r,text=True).splitlines()+subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=r,text=True).splitlines();sources={f:sha((r/f).read_bytes()) for f in files}
proof={'candidate':subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip(),'baseline':'509144567bc6f74d899385aacee5717d20811cc0','binary_sha256':sha(binary.read_bytes()),'source_sha256':sources,'probe_sha256':sha((p/'probe/src/main.rs').read_bytes()),'cache_line_bytes':subprocess.check_output(['/usr/sbin/sysctl','-n','hw.cachelinesize'],text=True).strip(),'rustc':subprocess.check_output(['rustc','-vV'],text=True)}
(p/'provenance.json').write_text(json.dumps(proof,indent=2)+'\n')
records=[]
# Smoke runs exercise all five cold selections but are excluded from measurements.
for idx,shape in enumerate(shapes):
 m,k,n,t,d,g,op=shape
 result=subprocess.run([str(binary),str(m),str(k),str(n),'1',str(t),str(p/f'smoke-{idx}.f32')],env=dict(env,NXRT_GATHER_DENSE=str(d),NXRT_BLOCK_GATED=str(g),NXRT_BANK_OP=op,NXRT_PAIR_ROTATION='0'),capture_output=True,text=True)
 (p/f'smoke-{idx}.log').write_text(result.stdout+'\nSTDERR:\n'+result.stderr)
 if result.returncode:print(result.stdout+result.stderr,flush=True);raise SystemExit(result.returncode)
 assert len([x for x in result.stdout.splitlines() if x.startswith('{')])==5
 print(json.dumps({'smoke':idx,'status':'pass'}),flush=True)
for rotation in range(5):
 for idx,shape in enumerate(shapes):
  m,k,n,t,d,g,op=shape;prefix=f'process{rotation}-case{idx}'
  (p/f'{prefix}-host.log').write_text(subprocess.check_output(['ps','-Ao','pid=,pcpu=,comm='],text=True))
  result=subprocess.run([str(binary),str(m),str(k),str(n),'0',str(t),str(p/f'{prefix}.f32')],env=dict(env,NXRT_GATHER_DENSE=str(d),NXRT_BLOCK_GATED=str(g),NXRT_BANK_OP=op,NXRT_PAIR_ROTATION=str(rotation)),capture_output=True,text=True)
  (p/f'{prefix}.log').write_text(result.stdout+'\nSTDERR:\n'+result.stderr)
  if result.returncode:print(result.stdout+result.stderr,flush=True);raise SystemExit(result.returncode)
  arms=[json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')];assert len(arms)==5
  records.append({'rotation':rotation,'shape':shape,'arms':arms,'output_sha256':sha((p/f'{prefix}.f32').read_bytes())});(p/'runs.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps({'rotation':rotation,'case':idx,'complete':True}),flush=True)
summary=[]
ratio=lambda x,y:statistics.median(a/b for a,b in zip(x,y))
for shape in shapes:
 rows=[x for x in records if x['shape']==shape];assert len(rows)==5 and len({x['output_sha256'] for x in rows})==1
 cells=[]
 for row in rows:
  arms={x['arm']:x for x in row['arms']};b=arms['before']
  minimum=min(c/w for arm in arms.values() for c,w in zip(arm['cpu_batch_ms'],arm['wall_batch_ms']));duration=all(min(x['cpu_batch_ms'])>=20 and min(x['control_batch_ms'])>=20 for x in arms.values())
  controls={label:ratio(x['control_batch_ms'],b['control_batch_ms']) for label,x in arms.items()};aa=ratio(arms['null']['cpu_batch_ms'],b['cpu_batch_ms'])
  cell={'rotation':row['rotation'],'qualified':duration and minimum>=.9 and .95<=aa<=1.05 and all(.95<=v<=1.05 for v in controls.values()),'minimum_cpu_wall_ratio':minimum,'duration_pass':duration,'control_ratios':controls,'null_cpu_ratio':aa,'addresses_mod128':{label:x['workspace_address_mod128'] for label,x in arms.items()},'comparisons':{}}
  for x,y in [('banked','before'),('null','before'),('banked128','before128'),('banked128','banked'),('before128','before')]:
   cell['comparisons'][x+'/'+y]={metric:ratio(arms[x][metric+'_batch_ms'],arms[y][metric+'_batch_ms']) for metric in ['cpu','wall']}
  cells.append(cell)
 summary.append({'shape':shape,'qualified':all(x['qualified'] for x in cells),'processes':cells,'median_ratios':{key:{metric:statistics.median(x['comparisons'][key][metric] for x in cells) for metric in ['cpu','wall']} for key in cells[0]['comparisons']}})
assert all(sha((r/f).read_bytes())==v for f,v in sources.items()),'source changed during sweep'
(p/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'summary':summary}),flush=True)
