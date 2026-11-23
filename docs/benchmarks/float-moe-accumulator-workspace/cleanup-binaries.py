import pathlib,json,hashlib
root=pathlib.Path('/tmp/nxrt-moe-accumulator-perf');prov=json.loads((root/'provenance.json').read_text());removed=[]
for label in ['baseline','after','null']:
 p=root/f'{label}-probe';expected=prov['binaries']['baseline' if label=='null' else label]['sha256'];actual=hashlib.sha256(p.read_bytes()).hexdigest();assert actual==expected
 link=root/'diagnostic'/p.name;assert link.is_symlink() and link.resolve()==p.resolve();link.unlink()
 removed.append({'path':str(p),'logical_bytes':p.stat().st_size,'sha256':actual});p.unlink()
receipt={'removed':removed,'logical_bytes':sum(p['logical_bytes'] for p in removed),'retained':'Raw outputs, batch records, diagnostics, source/probe hashes and source checkpoints remain; these bytes are logical sizes, not measured filesystem savings.'}
(root/'binary-cleanup.json').write_text(json.dumps(receipt,indent=2)+'\n')
out=pathlib.Path('/tmp/nxrt-float-moe-accumulator/docs/benchmarks/float-moe-accumulator-workspace');(out/'binary-cleanup.json').write_text(json.dumps(receipt,indent=2)+'\n');(out/'cleanup-binaries.py').write_bytes(pathlib.Path(__file__).read_bytes());print(json.dumps(receipt))
