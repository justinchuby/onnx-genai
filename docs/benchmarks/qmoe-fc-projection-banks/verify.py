"""Verify complete retained data and optionally its measured candidate sources."""
from pathlib import Path
import base64,gzip,hashlib,importlib.util,json,math,struct,sys
folder=Path(__file__).resolve().parent
sha=lambda data:hashlib.sha256(data).hexdigest()
manifest=json.loads((folder/'manifest.json').read_text())
actual={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts}
assert actual==set(manifest)
for name,digest in manifest.items():assert sha((folder/name).read_bytes())==digest,name
study=folder/'study';meta=json.loads((study/'metadata.json').read_text())
raw=gzip.decompress((study/'receipts.json.gz').read_bytes());assert sha(raw)==meta['receipts_sha256'];rows=json.loads(raw)
spec=importlib.util.spec_from_file_location('qmoe_reader',study/'analyze.py');reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
assert json.loads(json.dumps(reader.analyze(rows)))==json.loads((study/'results.json').read_text())
outputs=json.loads(gzip.decompress((study/'full-outputs.json.gz').read_bytes()))
logs=json.loads(gzip.decompress((study/'process-logs.json.gz').read_bytes()))
traces=json.loads(gzip.decompress((study/'dispatch-traces.json.gz').read_bytes()))
assert len(rows)==len(outputs)==len(traces)==144 and len(logs)==288
for row in rows:
 name=f"r{row['rep']}-c{row['case']}-{row['kind']}-{row['arm']}"
 raw=base64.b64decode(outputs[name+'.f32']);assert sha(raw)==row['output_sha256']
 assert len(raw)==row['rows']*row['hidden']*4
 values=struct.unpack('<'+'f'*(len(raw)//4),raw);assert all(math.isfinite(v) for v in values) and any(v!=0 for v in values)
 argmax=[max(range(row['hidden']),key=lambda c:(values[r*row['hidden']+c],-c)) for r in range(row['rows'])]
 assert argmax==row['argmax']
 receipt=json.loads(logs[name+'.stdout'].strip().splitlines()[-1]);assert all(row[k]==v for k,v in receipt.items())
 events=traces[name+'.f32.trace.json'];variants=[e.get('args',{}).get('kernel_variant') for e in events]
 assert ('qmoe.streamed_rows' in variants)==(not row['dense'])
for path,digest in meta['source_sha256'].items():assert sha(gzip.decompress((study/(Path(path).name+'.gz')).read_bytes()))==digest
assert sha((study/'run.py').read_bytes())==meta['runner_sha256']
assert sha((study/'probe/main.rs').read_bytes())==meta['probe_sha256']
assert sha((study/'probe/Cargo.lock').read_bytes())==meta['probe_lock_sha256']
validation=json.loads((folder/'validation/validation.json').read_text());assert validation['source_sha256']==meta['source_sha256']
for check in validation['checks']:
 raw=gzip.decompress((folder/'validation'/(check['name']+'.log.gz')).read_bytes());assert sha(raw)==check['sha256']
 if check['name'] in ['full','oracle']:
  assert check['counts'] and all(status=='ok' and int(p)>0 and int(f)==0 for status,p,f,i in check['counts'])
full=gzip.decompress((folder/'validation/full.log.gz').read_bytes()).decode()
for identity in ['expert_banks_use_one_peak_lane_without_warm_workspace_issuer_calls','expert_workspace_metadata_checks_presence_geometry_dtype_and_overflow','transient_expert_grants_cover_complete_sets_and_stream_all_gates']:
 assert full.count(f'test kernels::qmoe::tests::{identity} ... ok')==2,identity
controls=json.loads((study/'reader-controls.json').read_text());assert controls['accepted_complete_matrix'] and len(controls['controls'])==10 and all(c['rejected'] for c in controls['controls'])
assert controls['reader_sha256']==sha((study/'analyze.py').read_bytes())
if len(sys.argv)>1:
 for path,digest in meta['source_sha256'].items():assert sha((Path(sys.argv[1])/path).read_bytes())==digest,path
print(f'Verified {len(manifest)} payloads, 144 complete outputs/dispatch traces and final validation.')
