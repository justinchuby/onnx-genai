import pathlib,json,subprocess,os,hashlib,statistics
raw=pathlib.Path('/tmp/nxrt-moe-gather-final-perf');out=pathlib.Path('/tmp/nxrt-moe-gather-final-t4-perf');out.mkdir()
proof=json.loads((raw/'provenance.json').read_text())
def sha(b):return hashlib.sha256(b).hexdigest()
for label in ['baseline','after','null']:
 expected=proof['binaries']['baseline' if label=='null' else label]['sha256']
 assert sha((raw/f'{label}-probe').read_bytes())==expected
proof.update(purpose='one focused independent repeat of the failed four-thread control cell',parent_raw=str(raw),common_reps=82)
(out/'provenance.json').write_text(json.dumps(proof,indent=2)+'\n')
env=os.environ.copy();env['NXRT_GATHER_DENSE']='0'
records=[]
for i,order in enumerate([['baseline','after','null'],['after','null','baseline'],['null','baseline','after']]):
 for label in order:
  prefix=f'focused{i}-{label}';p=out/f'{prefix}.f32'
  (out/f'{prefix}-host.log').write_text(subprocess.check_output(['ps','-Ao','pid=,pcpu=,comm='],text=True))
  r=subprocess.run([str(raw/f'{label}-probe'),'4','256','512','82','4',str(p)],env=env,capture_output=True,text=True)
  (out/f'{prefix}.log').write_text(r.stdout+'\nSTDERR:\n'+r.stderr)
  assert r.returncode==0,(prefix,r.stdout,r.stderr)
  d=json.loads(r.stdout.strip().splitlines()[-1]);d.update(label=label,process=i,output_sha256=sha(p.read_bytes()))
  records.append(d);(out/'runs.json').write_text(json.dumps(records,indent=2)+'\n')
  print(json.dumps({'process':i,'label':label,'cpu_ms':statistics.median(d['cpu_batch_ms'])/82}),flush=True)
assert len({x['output_sha256'] for x in records})==1 and len({tuple(x['argmax']) for x in records})==1
summary={'shape':[4,256,512,4,0],'outputs_equal':True,'arms':{}}
for label in ['baseline','after','null']:
 a=[r for r in records if r['label']==label];cpu=[statistics.median(r['cpu_batch_ms'])/82 for r in a];wall=[statistics.median(r['wall_batch_ms'])/82 for r in a]
 summary['arms'][label]={'cpu_ms':statistics.median(cpu),'cpu_range':[min(cpu),max(cpu)],'wall_ms':statistics.median(wall),'wall_range':[min(wall),max(wall)],'control_ms':statistics.median(statistics.median(r['control_batch_ms']) for r in a)}
for label in ['after','null']:
 summary[label+'_cpu_ratio']=summary['arms'][label]['cpu_ms']/summary['arms']['baseline']['cpu_ms'];summary[label+'_control_ratio']=summary['arms'][label]['control_ms']/summary['arms']['baseline']['control_ms']
summary['control_pass']=all(.95<=summary[x+'_control_ratio']<=1.05 for x in ['after','null']);summary['duration_pass']=all(min(x['cpu_batch_ms'])>=20 and min(x['control_batch_ms'])>=20 for x in records)
assert all(x['workspace_requirement_bytes']==x['funded_workspace_bytes'] for x in records)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
