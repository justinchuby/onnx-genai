import os,pathlib,subprocess,hashlib,shutil,json,statistics,math
root=pathlib.Path('/tmp/nxrt-qmoe-workspace');os.chdir(root)
out=pathlib.Path('/tmp/nxrt-host-warm-fit-perf')
env=dict(os.environ,CARGO_TARGET_DIR='/tmp/nxrt-erf-nan-target',CARGO_INCREMENTAL='0',ORT_ROOT='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt')
env['ONNX_GENAI_ORT_LIB']=env['ORT_ROOT']+'/lib/libonnxruntime.1.32.0.dylib'
paths=['crates/onnx-runtime-memory-governor/src/manager.rs','crates/onnx-runtime-memory-governor/src/workspace.rs']
candidate={p:(root/p).read_bytes() for p in paths};baseline={p:subprocess.check_output(['git','show','HEAD:'+p]) for p in paths}
def sha(b):return hashlib.sha256(b).hexdigest()
prov={'base':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'candidate':{p:sha(b) for p,b in candidate.items()},'probe_sha256':sha((out/'probe.rs').read_bytes()),'compiler':subprocess.check_output(['rustc','-vV'],text=True),'binaries':{}}
(out/'candidate.patch').write_bytes(subprocess.check_output(['git','diff','--binary','--',*paths]))
try:
 for label,source in [('baseline',baseline),('after',candidate)]:
  for p,b in source.items():(root/p).write_bytes(b)
  print('build '+label,flush=True)
  with (out/f'build-{label}.log').open('w') as log:
   r=subprocess.run(['cargo','build','--offline','--release','--manifest-path',str(out/'probe/Cargo.toml')],env=env,stdout=log,stderr=subprocess.STDOUT)
  if r.returncode:raise RuntimeError((out/f'build-{label}.log').read_text()[-4000:])
  shutil.copyfile(pathlib.Path(env['CARGO_TARGET_DIR'])/'release/nxrt-host-warm-fit-probe',out/f'{label}-probe')
finally:
 for p,b in candidate.items():(root/p).write_bytes(b)
for p,b in candidate.items():assert (root/p).read_bytes()==b
shutil.copyfile(out/'baseline-probe',out/'null-probe')
for label in ['baseline','after','null']:
 b=out/f'{label}-probe';b.chmod(0o755)
 symbols=subprocess.check_output(['nm',str(b)],text=True,stderr=subprocess.STDOUT)
 assert 'Mlas' not in symbols and 'HostWorkspacePool' in symbols
 prov['binaries'][label]={'sha256':sha(b.read_bytes()),'size_bytes':b.stat().st_size,'mlas_symbols':0}
assert prov['binaries']['baseline']==prov['binaries']['null']
(out/'provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
cases=[(mode,n) for mode in ['pool','ep'] for n in [64,768,1048576]]
orders=[['baseline','after','null'],['after','null','baseline'],['null','baseline','after']]
(out/'protocol.json').write_text(json.dumps({'cases':cases,'orders':orders,'processes_per_arm_case':3,'batches':9,'target_cpu_ms':100,'minimum_cpu_ms':20,'control_gate_percent':5,'minimum_cpu_wall_ratio':.9,'attempts':1,'payload':'not touched; allocator/ownership mechanism only, not inference throughput'},indent=2)+'\n')
def run(label,case,reps,cr,tag):
 mode,n=case;prefix=f'{tag}-{label}-{mode}-{n}'
 (out/f'{prefix}-host.log').write_text(subprocess.check_output(['ps','-Ao','pid=,pcpu=,comm='],text=True))
 r=subprocess.run([str(out/f'{label}-probe'),mode,str(n),str(reps),str(cr)],env=env,capture_output=True,text=True)
 (out/f'{prefix}.log').write_text(r.stdout+'\nSTDERR:\n'+r.stderr)
 if r.returncode:raise RuntimeError(prefix+' '+r.stderr[-3000:])
 row=json.loads(r.stdout.strip().splitlines()[-1]);row.update(label=label,tag=tag)
 assert row['warm_issuer_calls']==0 and row['same_owner'] and row['funded_bytes']==n
 row['cpu_ns']=statistics.median(row['cpu_batch_ms'])*1e6/reps
 row['wall_ns']=statistics.median(row['wall_batch_ms'])*1e6/reps
 return row
pilots={}
for case in cases:
 row=run('baseline',case,10000,10000000,'pilot-excluded')
 reps=max(1,math.ceil(100/(row['cpu_ns']/1e6)));cr=max(1,math.ceil(100*10000000/statistics.median(row['control_batch_ms'])))
 pilots[case]=(reps,cr)
print(json.dumps({'pilots':{str(k):v for k,v in pilots.items()}}),flush=True)
rows=[]
for i,order in enumerate(orders):
 for case in cases:
  for label in order:
   row=run(label,case,*pilots[case],f'process{i}');rows.append(row)
   (out/'runs.json').write_text(json.dumps(rows,indent=2)+'\n')
   print(json.dumps({'process':i,'case':case,'arm':label,'cpu_ns':row['cpu_ns']}),flush=True)
summary=[]
for mode,n in cases:
 chosen=[r for r in rows if r['mode']==mode and r['bytes']==n];assert len(chosen)==9
 assert len({r['control_result'] for r in chosen})==1
 cell={'mode':mode,'bytes':n,'arms':{}}
 for label in orders[0]:
  rs=[r for r in chosen if r['label']==label]
  arm={}
  for key in ['cpu_ns','wall_ns']:
   xs=[r[key] for r in rs];arm[key]={'median':statistics.median(xs),'min':min(xs),'max':max(xs)}
  arm['control_ms']=statistics.median(statistics.median(r['control_batch_ms']) for r in rs)
  arm['minimum_cpu_wall_ratio']=min(c/w for r in rs for c,w in zip(r['cpu_batch_ms'],r['wall_batch_ms']))
  arm['minimum_ms']=min(min(r[k]) for r in rs for k in ['cpu_batch_ms','control_batch_ms'])
  cell['arms'][label]=arm
 b=cell['arms']['baseline'];a=cell['arms']['after'];z=cell['arms']['null']
 cell['after_percent']=100*(a['cpu_ns']['median']/b['cpu_ns']['median']-1)
 cell['null_percent']=100*(z['cpu_ns']['median']/b['cpu_ns']['median']-1)
 cell['control_ratios']={label:cell['arms'][label]['control_ms']/b['control_ms'] for label in ['after','null']}
 cell['gate_pass']=all(v['minimum_ms']>=20 and v['minimum_cpu_wall_ratio']>=.9 for v in cell['arms'].values()) and all(.95<=x<=1.05 for x in cell['control_ratios'].values())
 summary.append(cell)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
