import ast, os, pathlib, subprocess, hashlib, json, statistics
root=pathlib.Path('/tmp/nxrt-compact-wire-grants');os.chdir(root)
def sha(b): return hashlib.sha256(b).hexdigest()
env=os.environ.copy();env.update(CARGO_TARGET_DIR='/tmp/nxrt-erf-nan-target',CARGO_INCREMENTAL='0',ORT_ROOT='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt',RAYON_NUM_THREADS='1',MLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',ONNX_GENAI_MLAS_THREADPOOL_THREADS='1')
env['ONNX_GENAI_ORT_LIB']=env['ORT_ROOT']+'/lib/libonnxruntime.1.32.0.dylib'
for kind,driver,shapes in [('serial','/tmp/nxrt-neon-eight-perf-driver.py',[(8,512,64),(32,512,64),(4,4096,4)]),('session','/tmp/nxrt-neon-eight-session-driver.py',[(16,512,512,1,1)])]:
 raw=pathlib.Path('/tmp/nxrt-neon-eight-'+('perf' if kind=='serial' else 'session-perf'))
 provenance=json.loads((raw/'provenance.json').read_text())
 for label in ['baseline','after','null']:
  expected=provenance['binaries']['baseline' if label=='null' else label]['sha256']
  assert sha((raw/f'{label}-probe').read_bytes())==expected
 source=pathlib.Path(driver).read_text();module=ast.parse(source)
 fn=next(x for x in module.body if isinstance(x,ast.FunctionDef) and x.name=='run')
 ns=dict(globals(),raw=raw,artifact=raw)
 exec(compile(ast.Module(body=[fn],type_ignores=[]),driver,'exec'),ns)
 original=json.loads((raw/'runs.json').read_text());records=[]
 shapeof=(lambda r:(r['m'],r['k'],r['n'])) if kind=='serial' else (lambda r:(r['rows'],r['k'],r['n'],r['threads'],int(r['dense'])))
 for index,order in enumerate([['baseline','after','null'],['after','null','baseline'],['null','baseline','after']]):
  for shape in shapes:
   reps=next(r['reps'] for r in original if shapeof(r)==shape)
   for label in order:
    result=ns['run'](label,shape,reps,f'focused{index}');records.append(result)
    (raw/'focused-runs.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps({'kind':kind,'index':index,'shape':shape,'label':label,'cpu_ms':result['cpu_ms_per_call']}),flush=True)
 summary=[]
 for shape in shapes:
  selected=[r for r in records if shapeof(r)==shape]
  assert len(selected)==9 and len({r['output_sha256'] for r in selected})==1
  assert {r['output_sha256'] for r in selected}=={r['output_sha256'] for r in original if shapeof(r)==shape}
  assert len({tuple(r['argmax']) for r in selected})==1
  arms={}
  for label in ['baseline','after','null']:
   rows=[r for r in selected if r['label']==label]
   arms[label]={'cpu_ms':statistics.median(r['cpu_ms_per_call'] for r in rows),'cpu_range':[min(r['cpu_ms_per_call'] for r in rows),max(r['cpu_ms_per_call'] for r in rows)],'wall_ms':statistics.median(r['wall_ms_per_call'] for r in rows),'control_ms':statistics.median(statistics.median(r['control_batch_ms']) for r in rows)}
  cell={'shape':shape,'outputs_equal':True,'arms':arms}
  for label in ['after','null']:
   cell[label+'_cpu_ratio']=arms[label]['cpu_ms']/arms['baseline']['cpu_ms'];cell[label+'_control_ratio']=arms[label]['control_ms']/arms['baseline']['control_ms']
  cell['minimum_duration_pass']=all(min(r['cpu_batch_ms'])>=20 and min(r['control_batch_ms'])>=20 for r in selected)
  cell['control_5pct_pass']=all(.95<=cell[label+'_control_ratio']<=1.05 for label in ['after','null'])
  if kind=='session':
   assert all(r['workspace_requirement_bytes']==r['funded_workspace_bytes']==73728 for r in selected)
  summary.append(cell)
 (raw/'focused-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps({'kind':kind,'summary':summary}),flush=True)
