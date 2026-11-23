import os,subprocess,pathlib,json,hashlib,shutil,math,statistics
root=pathlib.Path('/tmp/nxrt-qmoe-workspace');os.chdir(root)
artifact=pathlib.Path('/tmp/nxrt-block-moe-accumulator')
env=dict(os.environ,CARGO_TARGET_DIR='/tmp/nxrt-erf-nan-target',CARGO_INCREMENTAL='0',ORT_ROOT='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt')
env['ONNX_GENAI_ORT_LIB']=env['ORT_ROOT']+'/lib/libonnxruntime.1.32.0.dylib'
for key in ['ONNX_GENAI_MOE_GROUPED_MIN_WORK','ONNX_GENAI_CPU_DECODE_THREADS']:env.pop(key,None)
def git(*args):return subprocess.check_output(['git',*args],text=True).strip()
def sha(data):return hashlib.sha256(data).hexdigest()
artifact=pathlib.Path('/tmp/nxrt-block-moe-accumulator/tiny-diagnostic');prior=artifact.parent
provenance=json.loads((prior/'provenance.json').read_text())
for label in ['baseline','after','null']:
 key='baseline' if label=='null' else label
 p=prior/f'{label}-probe';assert sha(p.read_bytes())==provenance['binaries'][key]['sha256'];shutil.copy2(p,artifact/f'{label}-probe')
(artifact/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
# Standalone native session, including strict prepare/run/output drop.
# The first three shapes isolate the new single-row lane round trip.
shapes=[(1,32,64,1,0,0),(1,64,128,1,1,0)]
orders=[['baseline','after','null'],['after','null','baseline'],['null','baseline','after'],['baseline','after','null'],['after','null','baseline']]
(artifact/'protocol.json').write_text(json.dumps({'shapes':shapes,'orders':orders,'processes_per_arm_shape':5,'batches':9,'target_cpu_ms':500,'minimum_cpu_ms':20,'minimum_cpu_wall_ratio':.9,'control_gate_percent':5,'attempts':1,'reason':'one predeclared diagnostic of the original two unqualified tiny cells, longer CPU batches and more independent processes; identical previously built executables'},indent=2)+'\n')
records=[]
def run(label,shape,reps,tag):
 m,k,n,threads,dense,gated=shape;prefix=f'{tag}-{label}-m{m}-h{k}-n{n}-t{threads}-dense{dense}-gated{gated}';output=artifact/f'{prefix}.f32'
 (artifact/f'{prefix}-host.log').write_text(subprocess.check_output(['ps','-Ao','pid=,pcpu=,comm='],text=True))
 result=subprocess.run([str(artifact/f'{label}-probe'),str(m),str(k),str(n),str(reps),str(threads),str(output)],env=dict(env,NXRT_GATHER_DENSE=str(dense),NXRT_BLOCK_GATED=str(gated)),capture_output=True,text=True)
 (artifact/f'{prefix}.log').write_text(result.stdout+'\nSTDERR:\n'+result.stderr)
 if result.returncode:print(result.stdout[-2000:]+result.stderr[-4000:],flush=True);raise RuntimeError(prefix)
 row=json.loads(result.stdout.strip().splitlines()[-1]);row.update(label=label,tag=tag,shape=shape,output_sha256=sha(output.read_bytes()))
 row['cpu_ms_per_call']=statistics.median(row['cpu_batch_ms'])/reps;row['wall_ms_per_call']=statistics.median(row['wall_batch_ms'])/reps
 return row
reps={}
for shape in shapes:
 pilot=run('baseline',shape,8,'pilot-excluded');(artifact/f'pilot-{shape[0]}-{shape[1]}-{shape[3]}-{shape[4]}-{shape[5]}.json').write_text(json.dumps(pilot,indent=2)+'\n');reps[shape]=max(1,math.ceil(500/pilot['cpu_ms_per_call']))
print(json.dumps({'stage':'pilots','reps':{str(s):n for s,n in reps.items()}}),flush=True)
for index,order in enumerate(orders):
 for shape in shapes:
  for label in order:
   row=run(label,shape,reps[shape],f'process{index}');records.append(row)
   (artifact/'runs.json').write_text(json.dumps(records,indent=2)+'\n')
   print(json.dumps({'stage':'run','index':index,'label':label,'shape':shape,'cpu_us':1000*row['cpu_ms_per_call']}),flush=True)
summary=[]
for shape in shapes:
 chosen=[r for r in records if tuple(r['shape'])==shape];assert len(chosen)==15
 assert len({r['output_sha256'] for r in chosen})==1 and len({tuple(r['argmax']) for r in chosen})==1
 cell={'shape':shape,'full_output_equal':True,'argmax_equal':True,'arms':{}}
 for label in ['baseline','after','null']:
  rows=[r for r in chosen if r['label']==label];cpu=[r['cpu_ms_per_call']*1000 for r in rows];wall=[r['wall_ms_per_call']*1000 for r in rows]
  cell['arms'][label]={'cpu_us_median':statistics.median(cpu),'cpu_us_range':[min(cpu),max(cpu)],'wall_us_median':statistics.median(wall),'wall_us_range':[min(wall),max(wall)],'control_ms_median':statistics.median([statistics.median(r['control_batch_ms']) for r in rows]),'workspace_bytes':sorted({r['workspace_requirement_bytes'] for r in rows})}
  assert all(r['workspace_requirement_bytes']==r['funded_workspace_bytes'] for r in rows)
 b=cell['arms']['baseline'];a=cell['arms']['after'];n=cell['arms']['null'];m,h,*_=shape
 assert len(b['workspace_bytes'])==len(a['workspace_bytes'])==1
 assert a['workspace_bytes'][0]-b['workspace_bytes'][0]==m*h*4
 cell.update(after_over_baseline=a['cpu_us_median']/b['cpu_us_median'],null_over_baseline=n['cpu_us_median']/b['cpu_us_median'],control_after_over_baseline=a['control_ms_median']/b['control_ms_median'],control_null_over_baseline=n['control_ms_median']/b['control_ms_median'])
 cell['minimum_cpu_wall_ratio']=min(c/w for r in chosen for c,w in zip(r['cpu_batch_ms'],r['wall_batch_ms']))
 cell['duration_pass']=all(min(r['cpu_batch_ms'])>=20 and min(r['control_batch_ms'])>=20 for r in chosen)
 cell['control_5pct_pass']=all(.95<=cell[k]<=1.05 for k in ['control_after_over_baseline','control_null_over_baseline'])
 summary.append(cell)
(artifact/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'stage':'summary','cells':summary}),flush=True)
