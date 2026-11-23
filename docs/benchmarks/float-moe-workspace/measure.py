import pathlib, json
script=pathlib.Path('/tmp/nxrt-float-moe-perf/driver.py').read_text()
exec(script[:script.index("base=git('rev-parse'")])
provenance=json.loads((artifact/'provenance.json').read_text())
assert all(sha((root/p).read_bytes())==digest for p,digest in provenance['candidate_sha256'].items())
assert all(sha((artifact/f'{label}-probe').read_bytes())==info['sha256'] for label,info in provenance['binaries'].items())
exec(script[script.index("shutil.copy2(artifact/'baseline-probe',artifact/'null-probe')"):])
