import subprocess,os,pathlib,json,hashlib,shutil,statistics
p=pathlib.Path('/tmp/nxrt-bank-paired');r=pathlib.Path('/tmp/nxrt-qmoe-workspace')
env=dict(os.environ,CARGO_TARGET_DIR='/tmp/nxrt-erf-nan-target',CARGO_INCREMENTAL='0',ORT_ROOT='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt')
env['ONNX_GENAI_ORT_LIB']=env['ORT_ROOT']+'/lib/libonnxruntime.1.32.0.dylib'
for key in ['ONNX_GENAI_MOE_GROUPED_MIN_WORK','ONNX_GENAI_CPU_DECODE_THREADS']:env.pop(key,None)
shapes=[(1,32,64,1,0,0,'block'),(8,128,256,1,0,0,'block'),(1,64,128,1,1,0,'block'),(8,128,256,4,0,1,'block'),(8,128,256,4,1,2,'block'),(8,128,256,4,0,3,'block'),(1,64,128,1,0,0,'float'),(8,128,256,4,0,1,'float')]
protocol={'shapes':shapes,'processes_per_case':3,'batches_per_arm_process':9,'process_construction_and_batch_start_rotation':[0,1,2],'target_cpu_ms':250,'excluded_pilot_minimum_cpu_ms':50,'minimum_batch_cpu_ms':20,'minimum_cpu_wall_ratio':0.9,'control_and_AA_max_deviation_percent':5,'attempts':1,'lock':'/tmp/onnx-genai-hostlock-darwin.flock','scope':'same process, same CPU crate, separate public sessions/providers/books per arm; strict prepare+run+output drop; no model throughput claim','limitations':'same-process pairing does not pin a CPU core or remove scheduler/DVFS effects; allocator placement and ASLR remain process-dependent'}
(p/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
with (p/'sweep-build.log').open('w') as log:result=subprocess.run(['cargo','build','--offline','--release','--manifest-path',str(p/'probe/Cargo.toml')],env=env,cwd=r,stdout=log,stderr=subprocess.STDOUT)
if result.returncode:print((p/'sweep-build.log').read_text()[-6000:],flush=True);raise SystemExit(result.returncode)
binary=p/'paired-probe';shutil.copy2(pathlib.Path(env['CARGO_TARGET_DIR'])/'release/nxrt-bank-paired-probe',binary)
sha=lambda data:hashlib.sha256(data).hexdigest()
symbols=subprocess.check_output(['nm',str(binary)],text=True,stderr=subprocess.STDOUT);assert 'probe_before_moe' in symbols and 'probe_before_block_quantized_moe' in symbols and 'Mlas' not in symbols
files=subprocess.check_output(['git','diff','--name-only'],cwd=r,text=True).splitlines()+subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=r,text=True).splitlines()
source_hashes={f:sha((r/f).read_bytes()) for f in files}
proof={'base':subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip(),'baseline':'509144567bc6f74d899385aacee5717d20811cc0','binary_sha256':sha(binary.read_bytes()),'source_sha256':source_hashes,'probe_sha256':sha((p/'probe/src/main.rs').read_bytes()),'rustc':subprocess.check_output(['rustc','-vV'],text=True),'symbol_count':len(symbols.splitlines()),'no_mlas_symbols':True}
(p/'provenance.json').write_text(json.dumps(proof,indent=2)+'\n')
records=[]
for rotation in range(3):
 for idx,shape in enumerate(shapes):
  m,k,n,t,d,g,op=shape;prefix=f'process{rotation}-case{idx}'
  (p/f'{prefix}-host.log').write_text(subprocess.check_output(['ps','-Ao','pid=,pcpu=,comm='],text=True))
  result=subprocess.run([str(binary),str(m),str(k),str(n),'0',str(t),str(p/f'{prefix}.f32')],env=dict(env,NXRT_GATHER_DENSE=str(d),NXRT_BLOCK_GATED=str(g),NXRT_BANK_OP=op,NXRT_PAIR_ROTATION=str(rotation)),capture_output=True,text=True)
  (p/f'{prefix}.log').write_text(result.stdout+'\nSTDERR:\n'+result.stderr)
  if result.returncode:print(result.stdout+result.stderr,flush=True);raise SystemExit(result.returncode)
  arms=[json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')];assert len(arms)==3
  records.append({'rotation':rotation,'shape':shape,'arms':arms,'output_sha256':sha((p/f'{prefix}.f32').read_bytes())})
  (p/'runs.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps({'rotation':rotation,'case':idx,'complete':True}),flush=True)
summary=[]
for idx,shape in enumerate(shapes):
 rows=[x for x in records if x['shape']==shape];assert len(rows)==3 and len({x['output_sha256'] for x in rows})==1
 cells=[]
 for row in rows:
  a={x['arm']:x for x in row['arms']};b=a['before'];c=a['banked'];n=a['null']
  paired=lambda x,y:statistics.median([v/w for v,w in zip(x,y)])
  cr=paired(c['cpu_batch_ms'],b['cpu_batch_ms']);nr=paired(n['cpu_batch_ms'],b['cpu_batch_ms'])
  controls=[paired(z['control_batch_ms'],b['control_batch_ms']) for z in [c,n]]
  minimum=min(v/w for z in a.values() for v,w in zip(z['cpu_batch_ms'],z['wall_batch_ms']))
  duration=all(min(z['cpu_batch_ms'])>=20 and min(z['control_batch_ms'])>=20 for z in a.values())
  cells.append({'rotation':row['rotation'],'banked_over_before':cr,'null_over_before':nr,'control_ratios':controls,'minimum_cpu_wall_ratio':minimum,'duration_pass':duration,'qualified':duration and minimum>=.9 and all(.95<=v<=1.05 for v in controls+[nr]),'cpu_us':{label:statistics.median(z['cpu_batch_ms'])/z['reps']*1000 for label,z in a.items()}})
 summary.append({'case':idx,'shape':shape,'processes':cells,'qualified':all(x['qualified'] for x in cells),'median_banked_ratio':statistics.median(x['banked_over_before'] for x in cells),'median_null_ratio':statistics.median(x['null_over_before'] for x in cells)})
assert all(sha((r/f).read_bytes())==value for f,value in source_hashes.items()),'source changed during measurement'
(p/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'summary':summary}),flush=True)
