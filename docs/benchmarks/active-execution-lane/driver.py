import os,subprocess,pathlib,json,hashlib,shutil,math,statistics
root=pathlib.Path('/tmp/nxrt-qmoe-workspace');os.chdir(root)
artifact=pathlib.Path('/tmp/nxrt-active-lane-perf');prior=pathlib.Path('/tmp/nxrt-qmoe-contribution-perf')
env=dict(os.environ,CARGO_TARGET_DIR='/tmp/nxrt-erf-nan-target',CARGO_INCREMENTAL='0',ORT_ROOT='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt');env['ONNX_GENAI_ORT_LIB']=env['ORT_ROOT']+'/lib/libonnxruntime.1.32.0.dylib'
for key in ['ONNX_GENAI_MOE_GROUPED_MIN_WORK','ONNX_GENAI_CPU_DECODE_THREADS','NXRT_ACTIVE_SCOPE']:env.pop(key,None)
def sha(data):return hashlib.sha256(data).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],text=True).strip()
base=git('rev-parse','HEAD');changed=git('diff','--name-only').splitlines();new=git('ls-files','--others','--exclude-standard','--','crates/onnx-runtime-session/src').splitlines();paths=changed+new
assert len(paths)==7 and all(p.startswith('crates/onnx-runtime-session/src/') for p in paths)
source={p:sha((root/p).read_bytes()) for p in paths}
patch=subprocess.check_output(['git','diff','--binary'])
for p in new:
 r=subprocess.run(['git','diff','--no-index','--binary','--','/dev/null',p],capture_output=True);assert r.returncode==1;patch+=r.stdout
(artifact/'candidate.patch').write_bytes(patch)
old_prov=json.loads((prior/'provenance.json').read_text())
for label,original in [('original','baseline'),('routed','after'),('null','baseline')]:
 src=prior/f'{original}-probe';assert sha(src.read_bytes())==old_prov['binaries'][original]['sha256'];shutil.copy2(src,artifact/f'{label}-probe')
probe=artifact/'probe';(probe/'src').mkdir(parents=True,exist_ok=True)
deps=['onnx-runtime-ep-api','onnx-runtime-ep-cpu','onnx-runtime-ir','onnx-runtime-loader','onnx-runtime-memory-api','onnx-runtime-memory-governor','onnx-runtime-session']
(probe/'Cargo.toml').write_text('[package]\nname="nxrt-active-lane-probe"\nversion="0.0.0"\nedition="2024"\n[dependencies]\nlibc="0.2"\nrayon="1"\n'+''.join(f'{p}={{path="{root}/crates/{p}"}}\n' for p in deps))
shutil.copy2(artifact/'probe.rs',probe/'src/main.rs')
old_probe=(prior/'probe.rs').read_bytes();new_probe=(artifact/'probe.rs').read_bytes();assert old_probe[:old_probe.index(b'fn execute(')]==new_probe[:new_probe.index(b'trait ProbeRunner')]
provenance={'base':base,'candidate_sha256':source,'probe_sha256':sha(new_probe),'original_probe_sha256':sha(old_probe),'model_prefix_equal':True,'old_binary_provenance':old_prov,'host_lock':'/tmp/onnx-genai-hostlock-darwin.flock','compiler':subprocess.check_output(['rustc','-vV'],text=True),'binaries':{}}
print('build current runtime probe',flush=True)
with (artifact/'build.log').open('w') as log:r=subprocess.run(['cargo','build','--offline','--release','--manifest-path',str(probe/'Cargo.toml')],env=env,stdout=log,stderr=subprocess.STDOUT)
if r.returncode:print((artifact/'build.log').read_text()[-5000:],flush=True);raise SystemExit(r.returncode)
shutil.copy2(pathlib.Path(env['CARGO_TARGET_DIR'])/'release/nxrt-active-lane-probe',artifact/'current-probe')
for label in ['original','routed','null','current']:
 p=artifact/f'{label}-probe';symbols=subprocess.check_output(['nm',str(p)],text=True,stderr=subprocess.STDOUT);assert 'InferenceSession' in symbols and 'Mlas' not in symbols
 provenance['binaries'][label]={'sha256':sha(p.read_bytes()),'symbol_count':len(symbols.splitlines()),'mlas_symbols':0}
assert provenance['binaries']['original']['sha256']==provenance['binaries']['null']['sha256']
(artifact/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
shapes=[(1,32,32,1,0),(1,64,64,1,0),(1,256,256,1,0)]
labels=['original','routed','unscoped','scoped','null'];orders=[labels[i:]+labels[:i] for i in range(5)]
(artifact/'protocol.json').write_text(json.dumps({'shapes':shapes,'orders':orders,'processes_per_shape_arm':5,'batches':9,'target_batch_cpu_ms':100,'minimum_batch_cpu_ms':20,'control_gate_percent':5,'attempts':1,'lifecycle':'scope creation and finish measured separately and added to total request CPU/wall work; counters have finite resolution and include timer overhead','source':'original/null identical old baseline; routed old candidate; unscoped/scoped same new binary with runtime mode selected outside loops'},indent=2)+'\n')
def run(label,shape,reps,tag):
 m,h,i,t,dense=shape;prefix=f'{tag}-{label}-m{m}-h{h}-n{i}-t{t}';output=artifact/f'{prefix}.f32';binary='current' if label in ['unscoped','scoped'] else label
 (artifact/f'{prefix}-host.log').write_text(subprocess.check_output(['ps','-Ao','pid=,pcpu=,comm='],text=True))
 r=subprocess.run([str(artifact/f'{binary}-probe'),str(m),str(h),str(i),str(reps),str(t),str(output)],env=dict(env,NXRT_GATHER_DENSE=str(dense),NXRT_ACTIVE_SCOPE='1' if label=='scoped' else '0'),capture_output=True,text=True)
 (artifact/f'{prefix}.log').write_text(r.stdout+'\nSTDERR:\n'+r.stderr)
 if r.returncode:print(r.stdout[-2000:]+r.stderr[-4000:],flush=True);raise RuntimeError(prefix)
 row=json.loads(r.stdout.strip().splitlines()[-1]);row.update(label=label,shape=shape,tag=tag,output_sha256=sha(output.read_bytes()))
 row['loop_cpu_us']=statistics.median(row['cpu_batch_ms'])*1000/reps;row['loop_wall_us']=statistics.median(row['wall_batch_ms'])*1000/reps
 row['total_cpu_us_per_call']=(sum(row['cpu_batch_ms'])+row.get('scope_setup_cpu_ms',0)+row.get('scope_finish_cpu_ms',0))*1000/(len(row['cpu_batch_ms'])*reps)
 row['total_wall_us_per_call']=(sum(row['wall_batch_ms'])+row.get('scope_setup_wall_ms',0)+row.get('scope_finish_wall_ms',0))*1000/(len(row['wall_batch_ms'])*reps)
 assert row['workspace_requirement_bytes']==row['funded_workspace_bytes'];return row
reps={}
for shape in shapes:
 pilot=run('original',shape,8,'pilot-excluded');reps[shape]=max(1,math.ceil(100/(pilot['loop_cpu_us']/1000)))
 (artifact/f'pilot-h{shape[1]}.json').write_text(json.dumps(pilot,indent=2)+'\n')
print(json.dumps({'stage':'pilots','reps':{str(s):n for s,n in reps.items()}}),flush=True)
records=[]
for idx,order in enumerate(orders):
 for shape in shapes:
  for label in order:
   row=run(label,shape,reps[shape],f'process{idx}');records.append(row);(artifact/'runs.json').write_text(json.dumps(records,indent=2)+'\n')
   print(json.dumps({'stage':'run','index':idx,'shape':shape,'label':label,'loop_cpu_us':row['loop_cpu_us'],'total_cpu_us_per_call':row['total_cpu_us_per_call']}),flush=True)
summary=[]
for shape in shapes:
 chosen=[r for r in records if tuple(r['shape'])==shape];assert len(chosen)==25
 assert len({r['output_sha256'] for r in chosen})==len({tuple(r['argmax']) for r in chosen})==1
 cell={'shape':shape,'full_output_equal':True,'argmax_equal':True,'arms':{}}
 for label in labels:
  rows=[r for r in chosen if r['label']==label];arm={}
  for metric in ['loop_cpu_us','loop_wall_us','total_cpu_us_per_call','total_wall_us_per_call']:
   vals=[r[metric] for r in rows];arm[metric+'_median']=statistics.median(vals);arm[metric+'_range']=[min(vals),max(vals)]
  arm['control_ms_median']=statistics.median([statistics.median(r['control_batch_ms']) for r in rows]);arm['workspace_bytes']=sorted({r['workspace_requirement_bytes'] for r in rows})
  arm['setup_cpu_ms']=[r.get('scope_setup_cpu_ms',0) for r in rows];arm['finish_cpu_ms']=[r.get('scope_finish_cpu_ms',0) for r in rows]
  cell['arms'][label]=arm
 o=cell['arms']['original'];u=cell['arms']['unscoped'];s=cell['arms']['scoped']
 cell['scoped_over_original']=s['total_cpu_us_per_call_median']/o['total_cpu_us_per_call_median'];cell['scoped_over_unscoped']=s['total_cpu_us_per_call_median']/u['total_cpu_us_per_call_median'];cell['null_over_original']=cell['arms']['null']['total_cpu_us_per_call_median']/o['total_cpu_us_per_call_median']
 cell['duration_pass']=all(min(r['cpu_batch_ms'])>=20 and min(r['control_batch_ms'])>=20 for r in chosen)
 cell['control_ratios']={label:cell['arms'][label]['control_ms_median']/o['control_ms_median'] for label in labels}
 cell['control_5pct_pass']=all(.95<=ratio<=1.05 for ratio in cell['control_ratios'].values());summary.append(cell)
assert all(sha((root/p).read_bytes())==digest for p,digest in source.items())
(artifact/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'stage':'summary','cells':summary}),flush=True)
