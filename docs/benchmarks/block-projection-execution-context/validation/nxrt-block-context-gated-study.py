import hashlib,json,pathlib,runpy
root=pathlib.Path('/tmp/nxrt-qmoe-workspace');r=json.loads(pathlib.Path('/tmp/nxrt-block-context-validation/validation.json').read_text())
assert {x['name'] for x in r['results']}=={'list-context','context','native','mlas','layout','session','clippy','clippy-mlas'}
assert all(x['exit']==0 for x in r['results'])
assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in r['source_sha256'].items())
runpy.run_path('/tmp/nxrt-block-context-study/run.py',run_name='__main__')
