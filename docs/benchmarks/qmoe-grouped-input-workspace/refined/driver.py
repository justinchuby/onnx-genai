import os,subprocess,pathlib,json,hashlib,shutil,math,statistics,time
root=pathlib.Path('/tmp/nxrt-qmoe-workspace');os.chdir(root)
artifact=pathlib.Path('/tmp/nxrt-qmoe-gather-refined-session-perf');artifact.mkdir(exist_ok=True)
env=os.environ.copy();env.update(CARGO_TARGET_DIR='/tmp/nxrt-erf-nan-target',CARGO_INCREMENTAL='0',ORT_ROOT='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt')
env['ONNX_GENAI_ORT_LIB']=env['ORT_ROOT']+'/lib/libonnxruntime.1.32.0.dylib'
def git(*a):return subprocess.check_output(['git',*a],text=True).strip()
def sha(data):return hashlib.sha256(data).hexdigest()
def audit_symbols(symbols):
 assert any(line.split()[-1:] == ['_main'] for line in symbols.splitlines()) and 'InferenceSession' in symbols,'load-bearing symbol audit failed'
 assert 'Mlas' not in symbols,'unexpected MLAS symbols in default native probe'
for invalid in ['', '0000000000000000 T _main']:
 try:audit_symbols(invalid)
 except AssertionError:pass
 else:raise RuntimeError('symbol audit negative control failed')
head=git('rev-parse','HEAD');base='911fcbd78ee1eb0a82b471b164aa02cff2d33c32';assert not git('status','--porcelain')
paths=git('diff','--name-only',base,head).splitlines();backup={p:(root/p).read_bytes() for p in paths}
manifest=root/'crates/onnx-runtime-session/Cargo.toml';manifest_bytes=manifest.read_bytes();lock=root/'Cargo.lock';lock_bytes=lock.read_bytes()
example=root/'crates/onnx-runtime-session/examples/nxrt_qmoe_grouped_input_probe.rs';assert not example.exists()
probe=pathlib.Path('/tmp/nxrt-qmoe-gather-session-probe.rs').read_bytes()
for p,data in backup.items():
 target=artifact/'candidate-source'/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
provenance={'head':head,'base':base,'source_path':str(root),'source_sha256':{p:sha(d) for p,d in backup.items()},'probe_sha256':sha(probe),'temporary_dev_dependency':'libc=0.2, identical in both arms, manifest/lock restored','lock':'/tmp/onnx-genai-hostlock-darwin.flock','compiler':subprocess.check_output(['rustc','-vV'],text=True),'binaries':{}}
(artifact/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
try:
 for label,ref in [('baseline',base),('after',head)]:
  for p,data in backup.items():
   (root/p).write_bytes(subprocess.check_output(['git','show',f'{ref}:{p}']) if label=='baseline' else data)
  manifest.write_bytes(manifest_bytes.replace(b'[dev-dependencies]\n',b'[dev-dependencies]\nlibc = "0.2"\n',1));lock.write_bytes(lock_bytes)
  example.parent.mkdir(exist_ok=True);example.write_bytes(probe)
  for stage,cmd in [('clean',['cargo','clean','--offline','--profile','release','-p','onnx-runtime-ep-api','-p','onnx-runtime-ep-cpu','-p','onnx-runtime-session']),('build',['cargo','build','--offline','--release','-p','onnx-runtime-session','--example','nxrt_qmoe_grouped_input_probe'])]:
   print(json.dumps({'stage':stage,'arm':label,'source':ref}),flush=True)
   with (artifact/f'{label}-{stage}.log').open('w') as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
   if r.returncode:print((artifact/f'{label}-{stage}.log').read_text()[-12000:],flush=True);raise RuntimeError(f'{label} {stage} failed')
  executable=pathlib.Path(env['CARGO_TARGET_DIR'])/'release/examples/nxrt_qmoe_grouped_input_probe';target=artifact/f'{label}-probe';shutil.copy2(executable,target)
  symbols=subprocess.check_output(['nm',str(target)],text=True,stderr=subprocess.STDOUT);(artifact/f'{label}-symbols.log').write_text(symbols)
  audit_symbols(symbols)
  provenance['binaries'][label]={'sha256':sha(target.read_bytes()),'symbols':len(symbols.splitlines()),'mlas_symbols':0}
  (artifact/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
finally:
 for p,data in backup.items():(root/p).write_bytes(data)
 manifest.write_bytes(manifest_bytes);lock.write_bytes(lock_bytes)
 if example.exists():example.unlink()
 assert not git('status','--porcelain'),'source restoration failed'
print(json.dumps({'stage':'restored','head':git('rev-parse','HEAD'),'tree':git('rev-parse','HEAD^{tree}')}),flush=True)
shutil.copy2(artifact/'baseline-probe',artifact/'null-probe');assert sha((artifact/'baseline-probe').read_bytes())==sha((artifact/'null-probe').read_bytes())
shapes=[(1,64,64,1,0),(8,256,256,1,0),(8,256,256,4,0)]
records=[]
def run(label,shape,reps,tag):
 m,k,n,threads,dense=shape;prefix=f'{tag}-{label}-m{m}-k{k}-n{n}-t{threads}-dense{dense}';output=artifact/f'{prefix}.f32'
 # This is supplementary process context, not an idle-host claim.
 (artifact/f'{prefix}-host-context.log').write_text(subprocess.check_output(['ps','-Ao','pid=,pcpu=,comm='],text=True))
 r=subprocess.run([str(artifact/f'{label}-probe'),str(m),str(k),str(n),str(reps),str(threads),str(output)],env=dict(env,NXRT_GATHER_DENSE=str(dense)),capture_output=True,text=True)
 (artifact/f'{prefix}.log').write_text(r.stdout+'\nSTDERR:\n'+r.stderr)
 if r.returncode:print(r.stdout[-2000:]+r.stderr[-8000:],flush=True);raise RuntimeError(f'run failed: {prefix}')
 result=json.loads(r.stdout.strip().splitlines()[-1]);result.update(label=label,tag=tag,output_sha256=sha(output.read_bytes()),output_path=str(output))
 result['cpu_ms_per_call']=statistics.median(result['cpu_batch_ms'])/reps;result['wall_ms_per_call']=statistics.median(result['wall_batch_ms'])/reps
 result['control_median_ms']=statistics.median(result['control_batch_ms']);result['batch_cpu_min_ms']=min(result['cpu_batch_ms'])
 return result
common={}
for shape in shapes:
 pilot=run('baseline',shape,256,'pilot-excluded');(artifact/f'pilot-m{shape[0]}-k{shape[1]}-t{shape[3]}-dense{shape[4]}.json').write_text(json.dumps(pilot,indent=2)+'\n')
 common[shape]=max(1,math.ceil(100/pilot['cpu_ms_per_call']))
print(json.dumps({'stage':'pilots','common_reps':{str(k):v for k,v in common.items()}}),flush=True)
orders=[['baseline','after','null'],['after','null','baseline'],['null','baseline','after']]
for index,order in enumerate(orders):
 for shape in shapes:
  for label in order:
   result=run(label,shape,common[shape],f'process{index}')
   records.append(result);(artifact/'runs.json').write_text(json.dumps(records,indent=2)+'\n')
   print(json.dumps({'stage':'run','index':index,'label':label,'shape':shape,'cpu_ms':result['cpu_ms_per_call'],'control_ms':result['control_median_ms'],'min_batch_cpu_ms':result['batch_cpu_min_ms']}),flush=True)
summary=[]
for shape in shapes:
 chosen=[r for r in records if (r['rows'],r['k'],r['n'],r['threads'],int(r['dense']))==shape]
 assert len(chosen)==9 and len({r['output_sha256'] for r in chosen})==1 and len({tuple(r['argmax']) for r in chosen})==1,'full output mismatch'
 groups={label:[r for r in chosen if r['label']==label] for label in ['baseline','after','null']}
 cell={'shape':shape,'full_output_bytes_equal':True,'row_argmax_equal':True,'arms':{}}
 for label,rows in groups.items():
  cpu=[r['cpu_ms_per_call'] for r in rows];wall=[r['wall_ms_per_call'] for r in rows]
  cell['arms'][label]={'cpu_ms_median':statistics.median(cpu),'cpu_ms_range':[min(cpu),max(cpu)],'wall_ms_median':statistics.median(wall),'wall_ms_range':[min(wall),max(wall)],'control_ms_median':statistics.median([r['control_median_ms'] for r in rows]),'requirement_bytes':sorted({r['workspace_requirement_bytes'] for r in rows}),'funded_bytes':sorted({r['funded_workspace_bytes'] for r in rows})}
 base_arm=cell['arms']['baseline'];after=cell['arms']['after'];null=cell['arms']['null']
 assert base_arm['requirement_bytes']==[0] and after['requirement_bytes']==[0 if shape[0]<=1 else shape[0]*shape[1]*4], 'grouped input must add precisely one rows*hidden f32 lane'
 assert all(arm['requirement_bytes']==arm['funded_bytes'] for arm in cell['arms'].values())
 cell['after_over_baseline']=after['cpu_ms_median']/base_arm['cpu_ms_median'];cell['null_over_baseline']=null['cpu_ms_median']/base_arm['cpu_ms_median']
 cell['control_after_over_baseline']=after['control_ms_median']/base_arm['control_ms_median'];cell['control_null_over_baseline']=null['control_ms_median']/base_arm['control_ms_median']
 cell['minimum_duration_pass']=all(r['batch_cpu_min_ms']>=20 and min(r['control_batch_ms'])>=20 for r in chosen)
 cell['control_median_5pct_pass']=all(.95<=cell[key]<=1.05 for key in ['control_after_over_baseline','control_null_over_baseline'])
 summary.append(cell)
(artifact/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'stage':'summary','cells':summary}),flush=True)
