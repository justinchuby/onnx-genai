import os,subprocess,pathlib,json,hashlib,shutil,statistics,math,re
root=pathlib.Path('/tmp/nxrt-compact-wire-grants');os.chdir(root)
raw=pathlib.Path('/tmp/nxrt-neon-eight-perf');raw.mkdir(exist_ok=True)
env=os.environ.copy();env.update(CARGO_TARGET_DIR='/tmp/nxrt-erf-nan-target',CARGO_INCREMENTAL='0',ORT_ROOT='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt',RAYON_NUM_THREADS='1',MLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',ONNX_GENAI_MLAS_THREADPOOL_THREADS='1')
def git(*a):return subprocess.check_output(['git',*a],text=True).strip()
def sha(b):return hashlib.sha256(b).hexdigest()
def audit_symbols(symbols):
 lines=symbols.splitlines()
 assert any(l.split()[-1:]==['_main'] for l in lines), 'missing main symbol'
 assert any('gemm_bt_serial_with_backend' in l for l in lines), 'missing serial entry symbol'
 assert any('nxrt_neon_nt_probe7control' in l for l in lines), 'missing unchanged control symbol'
 return len(lines)
for invalid in ['', '0000000000000000 T _main']:
 try:audit_symbols(invalid)
 except AssertionError:pass
 else:raise RuntimeError('symbol-audit negative control failed')
head=git('rev-parse','HEAD');base='911fcbd78ee1eb0a82b471b164aa02cff2d33c32';assert not git('status','--porcelain')
paths=git('diff','--name-only',base,head).splitlines();saved={p:(root/p).read_bytes() for p in paths}
probe=pathlib.Path('/tmp/nxrt-neon-nt-probe.rs').read_bytes();example=root/'crates/onnx-runtime-ep-cpu/examples/nxrt_neon_nt_probe.rs';assert not example.exists()
provenance={'head':head,'base':base,'source_path':str(root),'source_sha256':{p:sha(b) for p,b in saved.items()},'probe_sha256':sha(probe),'shared_lock':'/tmp/onnx-genai-hostlock-darwin.flock','compiler':subprocess.check_output(['rustc','-vV'],text=True),'binaries':{},'environment_overrides':{k:env[k] for k in ['RAYON_NUM_THREADS','MLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','ONNX_GENAI_MLAS_THREADPOOL_THREADS']}}
for p,b in saved.items():f=raw/'candidate-source'/p;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(b)
(raw/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
try:
 for label,ref,mlas in [('baseline',base,False),('after',head,False),('research',head,True)]:
  cache=pathlib.Path('/tmp/nxrt-neon-nt-perf')
  reuse=False
  if reuse:
   old=json.loads((cache/'provenance.json').read_text())
   assert old['base']==base and old['probe_sha256']==provenance['probe_sha256'] and old['compiler']==provenance['compiler']
   assert sha((cache/'baseline-probe').read_bytes())==old['binaries']['baseline']['sha256']
   provenance['reused_baseline']={'path':str(cache),'sha256':old['binaries']['baseline']['sha256'],'revision':base}
  if not reuse:
    for p,b in saved.items():
     if label=='baseline':
      exists=subprocess.run(['git','cat-file','-e',f'{ref}:{p}'],stderr=subprocess.DEVNULL).returncode==0
      if exists:(root/p).write_bytes(subprocess.check_output(['git','show',f'{ref}:{p}']))
      elif (root/p).exists():(root/p).unlink()
     else:(root/p).write_bytes(b)
    example.parent.mkdir(exist_ok=True);example.write_bytes(probe)
    for stage,args in [('clean',['cargo','clean','--offline','--profile','release','-p','onnx-runtime-ep-api','-p','onnx-runtime-ep-cpu','-p','onnx-runtime-session']),('build',['cargo','build','--offline','--release','-p','onnx-runtime-ep-cpu','--example','nxrt_neon_nt_probe']+(['--features','mlas'] if mlas else []))]:
     print(json.dumps({'stage':stage,'label':label,'revision':ref}),flush=True)
     with (raw/f'{label}-{stage}.log').open('w') as f:r=subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT)
     if r.returncode:print((raw/f'{label}-{stage}.log').read_text()[-12000:],flush=True);raise RuntimeError((label,stage))
  executable=cache/'baseline-probe' if reuse else pathlib.Path(env['CARGO_TARGET_DIR'])/'release/examples/nxrt_neon_nt_probe';target=raw/f'{label}-probe';shutil.copy2(executable,target)
  symbols=subprocess.check_output(['nm',str(target)],text=True,stderr=subprocess.STDOUT);(raw/f'{label}-symbols.log').write_text(symbols)
  audit_symbols(symbols)
  mlas_count=sum('Mlas' in l for l in symbols.splitlines());neon_count=sum('gemm_bt_neon' in l and 'block' in l for l in symbols.splitlines())
  assert (mlas_count>0)==mlas and neon_count>0,(label,mlas_count,neon_count)
  provenance['binaries'][label]={'sha256':sha(target.read_bytes()),'symbols':len(symbols.splitlines()),'mlas_symbols':mlas_count,'nt_neon_symbols':neon_count}
  if neon_count:
   assembly=subprocess.check_output(['otool','-tvV',str(target)],text=True);lines=assembly.splitlines();starts=[i for i,l in enumerate(lines) if l.endswith(':') and 'gemm_bt_neon' in l and 'block' in l];assert starts
   sections=[]
   for i in starts:
    end=next((j for j in range(i+1,len(lines)) if lines[j].endswith(':') and not lines[j][0].isspace()),len(lines));sections.extend(lines[i:end])
   section='\n'.join(sections);(raw/f'{label}-nt-neon.disasm').write_text(section+'\n')
   counts={name:len(re.findall(r'\b'+name+r'\b',section)) for name in ['fmul','fadd','fmla','fmadd','faddp','trn1','trn2','zip1','zip2']}
   assert counts['fmul'] and counts['fadd'] and not any(counts[x] for x in ['fmla','fmadd','faddp']),counts
   assert sum(counts[x] for x in ['trn1','trn2','zip1','zip2'])>0,counts
   provenance['binaries'][label]['instruction_counts']=counts
   provenance['binaries'][label]['vector_stack_accesses']=len(re.findall(r'\b(?:ldr|str|ldp|stp)\s+q[^\n]*\[sp',section))
  (raw/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
finally:
 for p,b in saved.items():(root/p).write_bytes(b)
 if example.exists():example.unlink()
 assert not git('status','--porcelain'),'source restoration failed'
assert provenance['binaries']['after']['instruction_counts']['fmul']>provenance['binaries']['baseline']['instruction_counts']['fmul'], 'missing widened arithmetic witness'
print(json.dumps({'stage':'restored','head':git('rev-parse','HEAD'),'tree':git('rev-parse','HEAD^{tree}')}),flush=True)
shutil.copy2(raw/'baseline-probe',raw/'null-probe');assert sha((raw/'baseline-probe').read_bytes())==sha((raw/'null-probe').read_bytes())
shapes=[(1,4096,1),(1,4096,64),(4,1024,64),(8,512,64),(16,512,64),(32,512,64),(9,35,7),(4,4096,4),(8,4096,4),(4,1,4)]
def run(label,shape,reps,tag,paired=False):
 m,k,n=shape;prefix=f'{tag}-{label}-m{m}-k{k}-n{n}';output=raw/prefix
 (raw/f'{prefix}-host.log').write_text(subprocess.check_output(['ps','-Ao','pid=,pcpu=,comm='],text=True))
 r=subprocess.run([str(raw/f'{label}-probe'),'paired' if paired else 'generic',str(m),str(k),str(n),str(reps),str(output)],env=env,capture_output=True,text=True)
 (raw/f'{prefix}.log').write_text(r.stdout+'\nSTDERR:\n'+r.stderr)
 if r.returncode:print(r.stdout+r.stderr,flush=True);raise RuntimeError(prefix)
 result=json.loads(r.stdout.strip().splitlines()[-1]);result.update(label=label,tag=tag,output_sha256=sha(pathlib.Path(str(output)+'.native.f32').read_bytes()))
 assert statistics.median(result['cpu_batch_ms'])/statistics.median(result['wall_batch_ms'])<=1.2, 'serial native process has extra CPU activity'
 if paired:assert statistics.median(result['reference_cpu_batch_ms'])/statistics.median(result['reference_wall_batch_ms'])<=1.2, 'serial reference process has extra CPU activity'
 result['cpu_ms_per_call']=statistics.median(result['cpu_batch_ms'])/reps;result['wall_ms_per_call']=statistics.median(result['wall_batch_ms'])/reps;result['control_ms']=statistics.median(result['control_batch_ms'])
 return result
def calibrate(label,shape,paired=False):
 count=256
 for attempt in range(8):
  pilot=run(label,shape,count,f'pilot-batched-excluded{attempt}',paired)
  (raw/f'pilot-{label}-{shape}-{attempt}.json').write_text(json.dumps(pilot,indent=2)+'\n')
  duration=statistics.median(pilot['cpu_batch_ms'])
  if paired:duration=min(duration,statistics.median(pilot['reference_cpu_batch_ms']))
  if duration>=0.1:return max(1,math.ceil(100/(duration/count)))
  count*=4
 raise RuntimeError('pilot cannot resolve a CPU-time batch; increase pilot duration')
reps={shape:calibrate('baseline',shape) for shape in shapes}
records=[]
for index,order in enumerate([['baseline','after','null'],['after','null','baseline'],['null','baseline','after']]):
 for shape in shapes:
  for label in order:
   r=run(label,shape,reps[shape],f'process{index}');records.append(r);(raw/'runs.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps({'stage':'run','label':label,'shape':shape,'cpu_ms':r['cpu_ms_per_call'],'control_ms':r['control_ms']}),flush=True)
summary=[]
for shape in shapes:
 selected=[r for r in records if (r['m'],r['k'],r['n'])==shape];assert len(selected)==9 and len({r['output_sha256'] for r in selected})==1 and len({tuple(r['argmax']) for r in selected})==1
 cell={'shape':shape,'outputs_equal':True,'arms':{}}
 for label in ['baseline','after','null']:
  chosen=[r for r in selected if r['label']==label];cpu=[r['cpu_ms_per_call'] for r in chosen];wall=[r['wall_ms_per_call'] for r in chosen];cell['arms'][label]={'cpu_ms':statistics.median(cpu),'cpu_range':[min(cpu),max(cpu)],'wall_ms':statistics.median(wall),'wall_range':[min(wall),max(wall)],'control_ms':statistics.median(r['control_ms'] for r in chosen)}
 for label in ['after','null']:
  cell[label+'_cpu_ratio']=cell['arms'][label]['cpu_ms']/cell['arms']['baseline']['cpu_ms'];cell[label+'_control_ratio']=cell['arms'][label]['control_ms']/cell['arms']['baseline']['control_ms']
 cell['minimum_duration_pass']=all(min(r['cpu_batch_ms'])>=20 and min(r['control_batch_ms'])>=20 for r in selected);cell['control_5pct_pass']=all(.95<=cell[x+'_control_ratio']<=1.05 for x in ['after','null']);summary.append(cell)
(raw/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'stage':'summary','cells':summary}),flush=True)
reference=[]
for shape in [(8,512,64),(16,512,64),(32,512,64)]:
 common=calibrate('research',shape,True)
 for index in range(3):
  r=run('research',shape,common,f'reference{index}',True);reference.append(r);(raw/'reference-runs.json').write_text(json.dumps(reference,indent=2)+'\n');print(json.dumps({'stage':'reference','shape':shape,'native_ms':r['cpu_ms_per_call'],'mlas_ms':statistics.median(r['reference_cpu_batch_ms'])/common}),flush=True)
