import pathlib,json,hashlib,itertools
original=pathlib.Path('/tmp/nxrt-float-moe-fc-perf');script=(original/'driver.py').read_text()
exec(script[:script.index("base=git('rev-parse'")])
provenance=json.loads((original/'provenance.json').read_text())
assert all(sha((root/p).read_bytes())==digest for p,digest in provenance['candidate_sha256'].items())
assert all(sha((original/f'{label}-probe').read_bytes())==row['sha256'] for label,row in provenance['binaries'].items())
assert sha((original/'null-probe').read_bytes())==provenance['binaries']['baseline']['sha256']
artifact=original/'diagnostic';artifact.mkdir(exist_ok=True)
for label in ['baseline','after','null']:
 path=artifact/f'{label}-probe'
 if not path.exists():path.symlink_to(original/f'{label}-probe')
start=script.index('def run(');end=script.index('\nreps={}',start);exec(script[start:end])
old=json.loads((original/'runs.json').read_text())
shapes=[(512,256,256,4,'0','identity'),(512,256,256,4,'0','swiglu2'),(8,64,128,4,'0','silu3')]
orders=[['baseline','after','null'],['after','null','baseline'],['null','baseline','after'],['null','after','baseline'],['after','baseline','null'],['baseline','null','after']]
reps={shape:2*next(r['reps'] for r in old if tuple(r['shape'])==shape) for shape in shapes}
(artifact/'protocol.json').write_text(json.dumps({'source_and_binaries':'Unchanged from original eight-cell sweep; hashes verified before this fixed diagnostic','reason':'One Relu control failed the 5% gate; forced parallel cells have large baseline/null differences','orders':orders,'repetitions':{str(s):n for s,n in reps.items()},'attempts':1,'rule':'Report both studies; do not replace original records or repeat until favourable.'},indent=2)+'\n')
records=[]
for i,order in enumerate(orders):
 for shape in shapes:
  for label in order:
   row=run(label,shape,reps[shape],f'process{i}');records.append(row)
   (artifact/'runs.json').write_text(json.dumps(records,indent=2)+'\n')
   print(json.dumps({'stage':'diagnostic','index':i,'label':label,'shape':shape,'cpu_us':1000*row['cpu_ms_per_call'],'wall_us':1000*row['wall_ms_per_call']}),flush=True)
exec(script[script.index('summary=[]'):])
