import os,subprocess,pathlib,json,hashlib,shutil,math,statistics,time
root=pathlib.Path('/tmp/nxrt-compact-wire-grants');os.chdir(root)
artifact=pathlib.Path('/tmp/nxrt-moe-workspace-perf')
env=os.environ.copy();env.update(CARGO_TARGET_DIR='/tmp/nxrt-erf-nan-target',CARGO_INCREMENTAL='0',ORT_ROOT='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt')
env['ONNX_GENAI_ORT_LIB']=env['ORT_ROOT']+'/lib/libonnxruntime.1.32.0.dylib'
def git(*a):return subprocess.check_output(['git',*a],text=True).strip()
def sha(data):return hashlib.sha256(data).hexdigest()
provenance=json.loads((artifact/'provenance.json').read_text())
assert git('rev-parse','HEAD')==provenance['head'] and not git('status','--porcelain')
for label in ['baseline','after']:
 assert sha((artifact/f'{label}-probe').read_bytes())==provenance['binaries'][label]['sha256']
print(json.dumps({'stage':'restored','head':git('rev-parse','HEAD'),'tree':git('rev-parse','HEAD^{tree}')}),flush=True)
shutil.copy2(artifact/'baseline-probe',artifact/'null-probe');assert sha((artifact/'baseline-probe').read_bytes())==sha((artifact/'null-probe').read_bytes())
shapes=[(1,32,32,1),(1,256,512,1),(4,256,512,1),(4,256,512,4)]
records=[]
def run(label,shape,reps,tag):
 m,k,n,threads=shape;prefix=f'{tag}-{label}-m{m}-k{k}-n{n}-t{threads}';output=artifact/f'{prefix}.f32'
 # This is supplementary process context, not an idle-host claim.
 (artifact/f'{prefix}-host-context.log').write_text(subprocess.check_output(['ps','-Ao','pid=,pcpu=,comm='],text=True))
 r=subprocess.run([str(artifact/f'{label}-probe'),str(m),str(k),str(n),str(reps),str(threads),str(output)],env=env,capture_output=True,text=True)
 (artifact/f'{prefix}.log').write_text(r.stdout+'\nSTDERR:\n'+r.stderr)
 if r.returncode:print(r.stdout[-2000:]+r.stderr[-8000:],flush=True);raise RuntimeError(f'run failed: {prefix}')
 result=json.loads(r.stdout.strip().splitlines()[-1]);result.update(label=label,tag=tag,output_sha256=sha(output.read_bytes()),output_path=str(output))
 result['cpu_ms_per_call']=statistics.median(result['cpu_batch_ms'])/reps;result['wall_ms_per_call']=statistics.median(result['wall_batch_ms'])/reps
 result['control_median_ms']=statistics.median(result['control_batch_ms']);result['batch_cpu_min_ms']=min(result['cpu_batch_ms'])
 return result
common={}
for shape in shapes:
 pilot=run('baseline',shape,1,'pilot-excluded');(artifact/f'pilot-m{shape[0]}-k{shape[1]}-t{shape[3]}.json').write_text(json.dumps(pilot,indent=2)+'\n')
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
 chosen=[r for r in records if (r['rows'],r['k'],r['n'],r['threads'])==shape]
 assert len(chosen)==9 and len({r['output_sha256'] for r in chosen})==1 and len({tuple(r['argmax']) for r in chosen})==1,'full output mismatch'
 groups={label:[r for r in chosen if r['label']==label] for label in ['baseline','after','null']}
 cell={'shape':shape,'full_output_bytes_equal':True,'row_argmax_equal':True,'arms':{}}
 for label,rows in groups.items():
  cpu=[r['cpu_ms_per_call'] for r in rows];wall=[r['wall_ms_per_call'] for r in rows]
  cell['arms'][label]={'cpu_ms_median':statistics.median(cpu),'cpu_ms_range':[min(cpu),max(cpu)],'wall_ms_median':statistics.median(wall),'wall_ms_range':[min(wall),max(wall)],'control_ms_median':statistics.median([r['control_median_ms'] for r in rows]),'requirement_bytes':sorted({r['workspace_requirement_bytes'] for r in rows}),'funded_bytes':sorted({r['funded_workspace_bytes'] for r in rows})}
 base_arm=cell['arms']['baseline'];after=cell['arms']['after'];null=cell['arms']['null']
 cell['after_over_baseline']=after['cpu_ms_median']/base_arm['cpu_ms_median'];cell['null_over_baseline']=null['cpu_ms_median']/base_arm['cpu_ms_median']
 cell['control_after_over_baseline']=after['control_ms_median']/base_arm['control_ms_median'];cell['control_null_over_baseline']=null['control_ms_median']/base_arm['control_ms_median']
 cell['minimum_duration_pass']=all(r['batch_cpu_min_ms']>=20 and min(r['control_batch_ms'])>=20 for r in chosen)
 cell['control_median_5pct_pass']=all(.95<=cell[key]<=1.05 for key in ['control_after_over_baseline','control_null_over_baseline'])
 summary.append(cell)
(artifact/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'stage':'summary','cells':summary}),flush=True)
