from pathlib import Path
import json,hashlib,shutil
root=Path('/tmp/nxrt-float-moe-fc-perf');provenance=json.loads((root/'provenance.json').read_text());known={row['sha256'] for row in provenance['binaries'].values()}
removed=[]
for label in ['baseline','after','null']:
 alias=root/'diagnostic'/f'{label}-probe'
 if alias.is_symlink():alias.unlink()
paths=[root/f'{label}-probe' for label in ['baseline','after','null']]+[Path('/tmp/nxrt-erf-nan-target/release/nxrt-float-moe-fc-probe')]+list(Path('/tmp/nxrt-erf-nan-target/release/deps').glob('nxrt_float_moe_fc_probe-*'))
for path in paths:
 if not path.is_file() or path.suffix=='.d':continue
 data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
 if digest not in known:continue
 removed.append({'path':str(path),'sha256':digest,'logical_bytes':len(data)});path.unlink()
log=Path('/tmp/nxrt-float-moe-fc-miri.log').read_text();assert 'test zeroed_prefix_keeps_whole_bank_initialization_separate ... ok' in log and 'test result: ok. 1 passed; 0 failed' in log
miri=Path('/tmp/nxrt-miri-moe-prefix-target');assert miri.resolve()==Path('/private/tmp/nxrt-miri-moe-prefix-target') and not miri.is_symlink()
miri_bytes=sum(p.stat().st_size for p in miri.rglob('*') if p.is_file())
shutil.rmtree(miri)
record={'removed_probe_copies':removed,'removed_miri_target':str(miri),'miri_logical_bytes':miri_bytes,'total_logical_bytes':sum(p['logical_bytes'] for p in removed)+miri_bytes,'physical_disk_saving_not_measured':True,'source_logs_raw_outputs_preserved':True,'stable_lock_inode_preserved':True}
(root/'cleanup.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
