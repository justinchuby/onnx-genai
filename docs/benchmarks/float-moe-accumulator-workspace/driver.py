import os,subprocess,pathlib,json,hashlib,shutil,math,statistics
root=pathlib.Path('/tmp/nxrt-float-moe-accumulator');os.chdir(root)
artifact=pathlib.Path('/tmp/nxrt-moe-accumulator-perf')
env=dict(os.environ,CARGO_TARGET_DIR='/tmp/nxrt-erf-nan-target',CARGO_INCREMENTAL='0',ORT_ROOT='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt')
env['ONNX_GENAI_ORT_LIB']=env['ORT_ROOT']+'/lib/libonnxruntime.1.32.0.dylib'
for key in ['ONNX_GENAI_MOE_GROUPED_MIN_WORK','ONNX_GENAI_CPU_DECODE_THREADS']:env.pop(key,None)
def git(*args):return subprocess.check_output(['git',*args],text=True).strip()
def sha(data):return hashlib.sha256(data).hexdigest()
base=git('rev-parse','HEAD^');paths=git('diff',base,'--name-only').splitlines();assert set(paths)=={'crates/onnx-runtime-ep-cpu/src/kernels/moe.rs','docs/memory/MEMORY_ARCHITECTURE.md'}
assert not git('status','--porcelain')
backup={p:(root/p).read_bytes() for p in paths}
(artifact/'candidate.patch').write_bytes(subprocess.check_output(['git','diff','--binary',base]))
probe=artifact/'probe';probe.mkdir(exist_ok=True)
deps=['onnx-runtime-ep-api','onnx-runtime-ep-cpu','onnx-runtime-ir','onnx-runtime-loader','onnx-runtime-memory-api','onnx-runtime-memory-governor','onnx-runtime-session']
(probe/'Cargo.toml').write_text('[package]\nname="nxrt-moe-accumulator-probe"\nversion="0.0.0"\nedition="2024"\n[dependencies]\nlibc="0.2"\nrayon="1"\n'+''.join(f'{p}={{path="{root}/crates/{p}"}}\n' for p in deps))
(probe/'src').mkdir(exist_ok=True);shutil.copy2(artifact/'probe.rs',probe/'src/main.rs')
provenance={'base':base,'source_path':str(root),'candidate_sha256':{p:sha(d) for p,d in backup.items()},'probe_sha256':sha((artifact/'probe.rs').read_bytes()),'lock':'/tmp/onnx-genai-hostlock-darwin.flock','rustc':subprocess.check_output(['rustc','-vV'],text=True),'binaries':{}}
try:
 for label in ['baseline','after']:
  for p,data in backup.items():(root/p).write_bytes(subprocess.check_output(['git','show',f'{base}:{p}']) if label=='baseline' else data)
  for stage,cmd in [('clean',['cargo','clean','--offline','--profile','release','-p','onnx-runtime-ep-cpu','-p','onnx-runtime-session']),('build',['cargo','build','--offline','--release','--manifest-path',str(probe/'Cargo.toml')])]:
   print(json.dumps({'stage':stage,'arm':label}),flush=True)
   with (artifact/f'{label}-{stage}.log').open('w') as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
   if r.returncode:print((artifact/f'{label}-{stage}.log').read_text()[-5000:],flush=True);raise RuntimeError(f'{label} {stage}')
  target=artifact/f'{label}-probe';shutil.copy2(pathlib.Path(env['CARGO_TARGET_DIR'])/'release/nxrt-moe-accumulator-probe',target)
  symbols=subprocess.check_output(['nm',str(target)],text=True,stderr=subprocess.STDOUT)
  assert 'InferenceSession' in symbols and any(line.split()[-1:]==['_main'] for line in symbols.splitlines()) and 'Mlas' not in symbols
  provenance['binaries'][label]={'sha256':sha(target.read_bytes()),'symbol_count':len(symbols.splitlines()),'mlas_symbols':0}
  (artifact/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
finally:
 for p,data in backup.items():(root/p).write_bytes(data)
 assert all((root/p).read_bytes()==data for p,data in backup.items())
print('source restored',flush=True)
shutil.copy2(artifact/'baseline-probe',artifact/'null-probe')
# Default per-expert decode/prefill, forced batched gather, natural crossover.
shapes=[(1,64,64,1,None,'identity'),(8,256,256,1,None,'identity'),(512,256,256,4,'0','identity'),(512,1024,1024,4,None,'identity'),(1,64,128,1,None,'swiglu1'),(8,256,32,1,None,'swiglu0'),(512,256,256,4,'0','swiglu2'),(8,64,128,4,'0','silu3')]
records=[]
def run(label,shape,reps,tag):
 m,k,n,t,threshold,variant=shape;prefix=f'{tag}-{label}-m{m}-h{k}-n{n}-t{t}-gate{threshold}-{variant}'
 output=artifact/f'{prefix}.f32';runenv=dict(env,NXRT_FC_VARIANT=variant)
 if threshold is not None:runenv['ONNX_GENAI_MOE_GROUPED_MIN_WORK']=threshold
 (artifact/f'{prefix}-host.log').write_text(subprocess.check_output(['ps','-Ao','pid=,pcpu=,comm='],text=True))
 r=subprocess.run([str(artifact/f'{label}-probe'),str(m),str(k),str(n),str(reps),str(t),str(output)],env=runenv,capture_output=True,text=True)
 (artifact/f'{prefix}.log').write_text(r.stdout+'\nSTDERR:\n'+r.stderr)
 if r.returncode:print(r.stdout[-2000:]+r.stderr[-4000:],flush=True);raise RuntimeError(prefix)
 result=json.loads(r.stdout.strip().splitlines()[-1]);result.update(label=label,tag=tag,shape=shape,output_sha256=sha(output.read_bytes()))
 result['cpu_ms_per_call']=statistics.median(result['cpu_batch_ms'])/reps
 result['wall_ms_per_call']=statistics.median(result['wall_batch_ms'])/reps
 return result
reps={}
for shape in shapes:
 pilot=run('baseline',shape,8,'pilot-excluded');(artifact/f'pilot-{shape[0]}-{shape[1]}-{shape[3]}-{shape[5]}.json').write_text(json.dumps(pilot,indent=2)+'\n');reps[shape]=max(1,math.ceil(100/pilot['cpu_ms_per_call']))
print(json.dumps({'stage':'pilots','reps':{str(s):n for s,n in reps.items()}}),flush=True)
for index,order in enumerate([['baseline','after','null'],['after','null','baseline'],['null','baseline','after']]):
 for shape in shapes:
  for label in order:
   row=run(label,shape,reps[shape],f'process{index}');records.append(row)
   (artifact/'runs.json').write_text(json.dumps(records,indent=2)+'\n')
   print(json.dumps({'stage':'run','index':index,'label':label,'shape':shape,'cpu_us':1000*row['cpu_ms_per_call']}),flush=True)
summary=[]
for shape in shapes:
 chosen=[r for r in records if tuple(r['shape'])==shape];assert len(chosen)==9
 assert len({r['output_sha256'] for r in chosen})==1 and len({tuple(r['argmax']) for r in chosen})==1
 cell={'shape':shape,'full_output_equal':True,'argmax_equal':True,'arms':{}}
 for label in ['baseline','after','null']:
  rows=[r for r in chosen if r['label']==label];cpu=[r['cpu_ms_per_call']*1000 for r in rows];wall=[r['wall_ms_per_call']*1000 for r in rows]
  cell['arms'][label]={'cpu_us_median':statistics.median(cpu),'cpu_us_range':[min(cpu),max(cpu)],'wall_us_median':statistics.median(wall),'wall_us_range':[min(wall),max(wall)],'control_ms_median':statistics.median([statistics.median(r['control_batch_ms']) for r in rows]),'workspace_bytes':sorted({r['workspace_requirement_bytes'] for r in rows})}
  assert all(r['workspace_requirement_bytes']==r['funded_workspace_bytes'] for r in rows)
 b=cell['arms']['baseline'];a=cell['arms']['after'];n=cell['arms']['null']
 cell.update(after_over_baseline=a['cpu_us_median']/b['cpu_us_median'],null_over_baseline=n['cpu_us_median']/b['cpu_us_median'],control_after_over_baseline=a['control_ms_median']/b['control_ms_median'],control_null_over_baseline=n['control_ms_median']/b['control_ms_median'])
 cell['duration_pass']=all(min(r['cpu_batch_ms'])>=20 and min(r['control_batch_ms'])>=20 for r in chosen)
 cell['control_5pct_pass']=all(.95<=cell[k]<=1.05 for k in ['control_after_over_baseline','control_null_over_baseline'])
 summary.append(cell)
(artifact/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'stage':'summary','cells':summary}),flush=True)
