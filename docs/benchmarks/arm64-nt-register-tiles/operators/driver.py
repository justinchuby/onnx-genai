import os,subprocess,pathlib,json,hashlib,shutil,statistics,math
root=pathlib.Path('/tmp/nxrt-compact-wire-grants');os.chdir(root);raw=pathlib.Path('/tmp/nxrt-neon-nt-operators');raw.mkdir(exist_ok=True)
env=os.environ.copy();env.update(CARGO_TARGET_DIR='/tmp/nxrt-erf-nan-target',CARGO_INCREMENTAL='0',ORT_ROOT='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt',RAYON_NUM_THREADS='1',ONNX_GENAI_HOST_ALLOC_CACHE_FLOOR_BYTES='1048576',ONNX_GENAI_MLAS_THREADPOOL_THREADS='1')
def git(*a):return subprocess.check_output(['git',*a],text=True).strip()
def sha(b):return hashlib.sha256(b).hexdigest()
head=git('rev-parse','HEAD');base='bb1b14cf3f564e7b41ac6b5cc30df19a77b0105c';assert not git('status','--porcelain');saved={p:(root/p).read_bytes() for p in git('diff','--name-only',base,head).splitlines()}
probe=pathlib.Path('/tmp/nxrt-neon-nt-operators.rs').read_bytes();example=root/'crates/onnx-runtime-ep-cpu/examples/nxrt_neon_nt_operators.rs';assert not example.exists()
proof={'head':head,'base':base,'source_sha256':{p:sha(b) for p,b in saved.items()},'probe_sha256':sha(probe),'compiler':subprocess.check_output(['rustc','-vV'],text=True),'shared_lock':'/tmp/onnx-genai-hostlock-darwin.flock','binaries':{}}
try:
 for label,ref in [('baseline',base),('after',head)]:
  for p,b in saved.items():
   if label=='after':(root/p).write_bytes(b)
   elif subprocess.run(['git','cat-file','-e',f'{ref}:{p}'],stderr=subprocess.DEVNULL).returncode==0:(root/p).write_bytes(subprocess.check_output(['git','show',f'{ref}:{p}']))
   elif (root/p).exists():(root/p).unlink()
  example.write_bytes(probe)
  for stage,args in [('clean',['cargo','clean','--offline','--profile','release','-p','onnx-runtime-ep-cpu']),('build',['cargo','build','--offline','--release','-p','onnx-runtime-ep-cpu','--example','nxrt_neon_nt_operators'])]:
   print(json.dumps({'stage':stage,'label':label}),flush=True)
   with (raw/f'{label}-{stage}.log').open('w') as f:r=subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT)
   if r.returncode:print((raw/f'{label}-{stage}.log').read_text()[-12000:],flush=True);raise RuntimeError((label,stage))
  exe=pathlib.Path(env['CARGO_TARGET_DIR'])/'release/examples/nxrt_neon_nt_operators';target=raw/f'{label}-probe';shutil.copy2(exe,target);symbols=subprocess.check_output(['nm',str(target)],text=True);(raw/f'{label}-symbols.log').write_text(symbols)
  assert any(l.split()[-1:]==['_main'] for l in symbols.splitlines()) and 'Mlas' not in symbols
  assert ('gemm_bt_neon' in symbols)==(label=='after')
  proof['binaries'][label]={'sha256':sha(target.read_bytes()),'symbols':len(symbols.splitlines()),'mlas_symbols':0,'neon_nt_symbol':'gemm_bt_neon' in symbols}
  (raw/'provenance.json').write_text(json.dumps(proof,indent=2)+'\n')
finally:
 for p,b in saved.items():(root/p).write_bytes(b)
 if example.exists():example.unlink()
 assert not git('status','--porcelain')
shutil.copy2(raw/'baseline-probe',raw/'null-probe')
cases=[('matmul','streamed',4,1024,2048),('moe','warm',16,512,512)]
def run(label,case,reps,tag):
 kind,mode,m,k,n=case;prefix=f'{tag}-{label}-{kind}-{mode}-m{m}-k{k}-n{n}';output=raw/f'{prefix}.f32'
 (raw/f'{prefix}-host.log').write_text(subprocess.check_output(['ps','-Ao','pid=,pcpu=,comm='],text=True))
 r=subprocess.run([str(raw/f'{label}-probe'),kind,mode,str(m),str(k),str(n),str(reps),str(output)],env=env,capture_output=True,text=True);(raw/f'{prefix}.log').write_text(r.stdout+'\nSTDERR:\n'+r.stderr)
 if r.returncode:print(r.stdout+r.stderr,flush=True);raise RuntimeError(prefix)
 d=json.loads(r.stdout.strip().splitlines()[-1]);assert d['threads']==1;d.update(label=label,tag=tag,output_sha256=sha(output.read_bytes()));return d
counts={}
for case in cases:
 p=run('baseline',case,4,'pilot-excluded');counts[case]=max(1,math.ceil(100/p['cpu_ms_per_call']))
records=[]
for i,order in enumerate([['baseline','after','null'],['after','null','baseline'],['null','baseline','after']]):
 for case in cases:
  for label in order:
   d=run(label,case,counts[case],f'process{i}');records.append(d);(raw/'runs.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps({'stage':'run','case':case,'label':label,'cpu_ms':d['cpu_ms_per_call']}),flush=True)
summary=[]
for case in cases:
 chosen=[r for r in records if (r['kind'],r['mode'],r['rows'],r['k'],r['n'])==case];assert len(chosen)==9 and len({r['output_sha256'] for r in chosen})==1 and len({tuple(r['argmax']) for r in chosen})==1
 c={'case':case,'outputs_equal':True,'arms':{}}
 for label in ['baseline','after','null']:
  a=[r for r in chosen if r['label']==label];cpu=[r['cpu_ms_per_call'] for r in a];wall=[r['wall_ms_per_call'] for r in a];c['arms'][label]={'cpu_ms':statistics.median(cpu),'cpu_range':[min(cpu),max(cpu)],'wall_ms':statistics.median(wall),'wall_range':[min(wall),max(wall)],'control_ms':statistics.median(statistics.median(r['control_batch_ms']) for r in a)}
 for label in ['after','null']:
  c[label+'_cpu_ratio']=c['arms'][label]['cpu_ms']/c['arms']['baseline']['cpu_ms'];c[label+'_control_ratio']=c['arms'][label]['control_ms']/c['arms']['baseline']['control_ms']
 c['minimum_duration_pass']=all(min(r['cpu_samples_ms_per_call'])*r['reps']>=20 and min(r['control_batch_ms'])>=20 for r in chosen);c['control_5pct_pass']=all(.95<=c[x+'_control_ratio']<=1.05 for x in ['after','null']);summary.append(c)
(raw/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'stage':'summary','cells':summary}),flush=True)
