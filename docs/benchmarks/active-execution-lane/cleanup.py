import pathlib,json,hashlib,subprocess,os,sys
root=pathlib.Path('/tmp/nxrt-active-lane-perf');out=pathlib.Path('/tmp/nxrt-qmoe-workspace/docs/benchmarks/active-execution-lane');prov=json.loads((root/'provenance.json').read_text());removed=[]
for label in ['original','routed','null','current']:
 p=root/f'{label}-probe';assert not p.is_symlink();digest=hashlib.sha256(p.read_bytes()).hexdigest();assert digest==prov['binaries'][label]['sha256'];removed.append({'path':str(p),'sha256':digest,'logical_bytes':p.stat().st_size});p.unlink()
env=dict(os.environ,CARGO_TARGET_DIR='/tmp/nxrt-erf-nan-target')
r=subprocess.run(['cargo','clean','--offline','--release','--manifest-path',str(root/'probe/Cargo.toml'),'-p','nxrt-active-lane-probe'],env=env,capture_output=True,text=True)
receipt={'removed':removed,'logical_bytes':sum(p['logical_bytes'] for p in removed),'probe_cache_clean_exit':r.returncode,'probe_cache_clean_output':r.stdout+r.stderr,'retained':'Original QMoE baseline/candidate binaries in their separate folder remain for ongoing work; raw outputs, batch records and sources remain. No host-lock inode or shared library cache is deleted. Logical byte sizes are not measured physical disk savings.'}
(out/'cleanup.json').write_text(json.dumps(receipt,indent=2)+'\n');(out/'cleanup.py').write_bytes(pathlib.Path(__file__).read_bytes())
p=out/'source.json';s=json.loads(p.read_text());s['raw_binaries']='Hash-verified attribution copies and this temporary probe package cache removed under the shared flock; see cleanup.json. Original QMoE binaries remain in their separate folder.';p.write_text(json.dumps(s,indent=2)+'\n')
p=out/'README.md';s=p.read_text();s+='\n## Cleanup\n\nThe four hash-verified attribution executable copies and this temporary probe package cache were removed under the shared host lock after the fixed study. Raw records/output files and source/binary hashes remain; original QMoE binaries in their separate folder remain available for ongoing work. cleanup.json records the exact scope and logical sizes.\n';p.write_text(s)
print(json.dumps(receipt),flush=True);sys.exit(r.returncode)
