import pathlib,json,subprocess,os,hashlib,statistics
raw=pathlib.Path('/tmp/nxrt-moe-window-perf');out=pathlib.Path('/tmp/nxrt-moe-window-focused-perf');out.mkdir()
proof=json.loads((raw/'provenance.json').read_text());original=json.loads((raw/'runs.json').read_text())
def sha(b):return hashlib.sha256(b).hexdigest()
for label in ['baseline','after','null']:
 assert sha((raw/f'{label}-probe').read_bytes())==proof['binaries']['baseline' if label=='null' else label]['sha256']
cases=[(1,256,512,1,0),(4,256,512,4,0),(1,32,32,1,1)]
common={case:next(r['reps'] for r in original if (r['rows'],r['k'],r['n'],r['threads'],int(r['dense']))==case and not r['partial']) for case in cases}
proof.update(purpose='one independent focused repeat of the three concerning cells',parent_raw=str(raw),common_reps={str(k):v for k,v in common.items()})
(out/'provenance.json').write_text(json.dumps(proof,indent=2)+'\n')
env=os.environ.copy();env.update(CARGO_TARGET_DIR='/tmp/nxrt-erf-nan-target',CARGO_INCREMENTAL='0',ORT_ROOT='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt')
env['ONNX_GENAI_ORT_LIB']=env['ORT_ROOT']+'/lib/libonnxruntime.1.32.0.dylib'
records=[]
for i,order in enumerate([['baseline','after','null'],['after','null','baseline'],['null','baseline','after']]):
 for case in cases:
  m,k,n,t,dense=case
  for label in order:
   prefix=f'focused{i}-{label}-m{m}-k{k}-n{n}-t{t}-dense{dense}';p=out/f'{prefix}.f32'
   (out/f'{prefix}-host.log').write_text(subprocess.check_output(['ps','-Ao','pid=,pcpu=,comm='],text=True))
   r=subprocess.run([str(raw/f'{label}-probe'),str(m),str(k),str(n),str(common[case]),str(t),str(p)],env=dict(env,NXRT_GATHER_DENSE=str(dense),NXRT_WINDOW_PARTIAL='0'),capture_output=True,text=True)
   (out/f'{prefix}.log').write_text(r.stdout+'\nSTDERR:\n'+r.stderr);assert r.returncode==0,(prefix,r.stdout,r.stderr)
   d=json.loads(r.stdout.strip().splitlines()[-1]);d.update(label=label,process=i,output_sha256=sha(p.read_bytes()));records.append(d)
   (out/'runs.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps({'case':case,'process':i,'label':label,'cpu_ms':statistics.median(d['cpu_batch_ms'])/d['reps']}),flush=True)
summary=[]
for case in cases:
 chosen=[r for r in records if (r['rows'],r['k'],r['n'],r['threads'],int(r['dense']))==case]
 assert len(chosen)==9 and len({r['output_sha256'] for r in chosen})==1 and len({tuple(r['argmax']) for r in chosen})==1
 s={'shape':case,'outputs_equal':True,'arms':{}}
 for label in ['baseline','after','null']:
  a=[r for r in chosen if r['label']==label];cpu=[statistics.median(r['cpu_batch_ms'])/r['reps'] for r in a];wall=[statistics.median(r['wall_batch_ms'])/r['reps'] for r in a]
  s['arms'][label]={'cpu_ms':statistics.median(cpu),'cpu_range':[min(cpu),max(cpu)],'wall_ms':statistics.median(wall),'wall_range':[min(wall),max(wall)],'control_ms':statistics.median(statistics.median(r['control_batch_ms']) for r in a)}
 for label in ['after','null']:
  s[label+'_cpu_ratio']=s['arms'][label]['cpu_ms']/s['arms']['baseline']['cpu_ms'];s[label+'_control_ratio']=s['arms'][label]['control_ms']/s['arms']['baseline']['control_ms']
 s['control_pass']=all(.95<=s[x+'_control_ratio']<=1.05 for x in ['after','null']);s['duration_pass']=all(min(x['cpu_batch_ms'])>=20 and min(x['control_batch_ms'])>=20 for x in chosen)
 assert all(r['workspace_requirement_bytes']==r['funded_workspace_bytes'] for r in chosen);summary.append(s)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
