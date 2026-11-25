import hashlib,json,pathlib,runpy
r=json.loads(pathlib.Path('/tmp/nxrt-qmoe-decoder-validation/validation.json').read_text())
assert {x['name'] for x in r['results']}=={'api','native','mlas','layout','session','plugin','clippy','clippy-mlas','final-qmoe'}
assert all(x['exit']==0 for x in r['results'])
root=pathlib.Path('/tmp/nxrt-qmoe-workspace')
assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in r['source_sha256'].items())
runpy.run_path('/tmp/nxrt-qmoe-decoder-study/run.py',run_name='__main__')
