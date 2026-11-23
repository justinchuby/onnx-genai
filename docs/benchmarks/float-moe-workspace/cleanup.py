import pathlib,json,hashlib
root=pathlib.Path('/tmp/nxrt-float-moe-perf');p=json.loads((root/'provenance.json').read_text());known={b['sha256'] for b in p['binaries'].values()};removed=[]
paths=[root/f'{label}-probe' for label in ['baseline','after','null']]
paths.append(pathlib.Path('/tmp/nxrt-erf-nan-target/release/nxrt-float-moe-probe'))
paths.extend(pathlib.Path('/tmp/nxrt-erf-nan-target/release/deps').glob('nxrt_float_moe_probe-*'))
for path in paths:
 if not path.is_file() or path.suffix=='.d':continue
 data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
 if digest not in known:continue
 removed.append({'path':str(path),'logical_bytes':len(data),'sha256':digest});path.unlink()
result={'removed_verified_probe_copies':removed,'logical_bytes':sum(p['logical_bytes'] for p in removed),'stable_lock_preserved':True,'sources_logs_raw_outputs_preserved':True}
(root/'cleanup.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
