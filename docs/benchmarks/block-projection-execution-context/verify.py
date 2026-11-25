"""Verify retained data, controls and optional measured source checkout."""
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
assert len(rows)==len(outputs)==len(traces)==240 and len(logs)==480
for row in rows:
 name=f"r{row['rep']}-c{row['case']}-{row['kind']}-{row['arm']}"
 raw=base64.b64decode(outputs[name+'.f32']);assert sha(raw)==row['output_sha256']
 width=row['inter'] if row['matmul'] else row['hidden']
 assert len(raw)==row['rows']*width*4
 values=struct.unpack('<'+'f'*(len(raw)//4),raw);assert all(math.isfinite(v) for v in values) and any(v!=0 for v in values)
 argmax=[max(range(width),key=lambda c:(values[r*width+c],-c)) for r in range(row['rows'])]
 assert argmax==row['argmax']
 receipt=json.loads(logs[name+'.stdout'].strip().splitlines()[-1]);assert all(row[k]==v for k,v in receipt.items())
 variants=[e.get('args',{}).get('kernel_variant') for e in traces[name+'.f32.trace.json']]
 assert (('block_matmul.streamed_rows' if row['matmul'] else 'block_moe.streamed_rows') in variants)==(not row['dense'])
for path,digest in meta['source_sha256'].items():assert sha(gzip.decompress((study/'sources'/(path+'.gz')).read_bytes()))==digest
assert sha((study/'run.py').read_bytes())==meta['runner_sha256']
assert sha((study/'probe/main.rs').read_bytes())==meta['probe_sha256']
assert sha((study/'probe/Cargo.lock').read_bytes())==meta['probe_lock_sha256']
validation=json.loads((folder/'validation/validation.json').read_text());assert validation['source_sha256']==meta['source_sha256']
assert len(validation['results'])==8 and all(r['exit']==0 for r in validation['results'])
for r in validation['results']:
 raw=gzip.decompress((folder/'validation'/(r['name']+'.log.gz')).read_bytes());assert sha(raw)==r['log_sha256']
 if 'test' in r['command'] and '--list' not in r['command']:assert r['counts'] and all(int(p)>0 and int(f)==0 for p,f,i in r['counts'])
oracle=json.loads((folder/'oracle-validation/validation.json').read_text())
assert oracle['runtime_source_sha256']==meta['source_sha256'] and len(oracle['results'])==5 and all(r['exit']==0 for r in oracle['results'])
for r in oracle['results']:
 assert sha(gzip.decompress((folder/'oracle-validation'/(r['name']+'.log.gz')).read_bytes()))==r['log_sha256']
 if r['name']=='host':assert r['counts']==[['2','0','0']]
 if r['name']=='gpu-fixtures-list':assert r['listed_tests']>0 and not r['gpu_execution']
for p,h in oracle['source_sha256'].items():assert sha(gzip.decompress((folder/'oracle-validation/sources'/(p+'.gz')).read_bytes()))==h
native=gzip.decompress((folder/'validation/native.log.gz').read_bytes()).decode()
mlas=gzip.decompress((folder/'validation/mlas.log.gz').read_bytes()).decode()
for kind in ['matmul','moe']:
 identity='test kernels::block_quantized_'+kind+'::tests::execution_refinement_selects_current_pool_and_consumes_plan ... ok'
 assert identity in native and identity in mlas
standalone=json.loads((folder/'standalone-oracle-validation/validation.json').read_text())
assert standalone['source_sha256']==oracle['source_sha256'] and all(r['exit']==0 for r in standalone['results'])
assert json.loads((folder/'standalone-oracle-validation/head.json').read_text())['candidate_restored']
for r in standalone['results']:
 assert sha(gzip.decompress((folder/'standalone-oracle-validation'/(r['name']+'.log.gz')).read_bytes()))==r['log_sha256']
 if r['name']=='host':assert r['counts']==[['2','0','0']]
for p,h in standalone['runtime_source_sha256'].items():assert sha(gzip.decompress((folder/'standalone-oracle-validation/sources'/(p+'.gz')).read_bytes()))==h
controls=json.loads((study/'reader-controls.json').read_text());assert controls['accepted_complete_matrix'] and controls['synthetic_all_qualified']
assert len(controls['controls'])==13 and all(c['rejected'] for c in controls['controls'])
assert len(controls['quality_controls'])==2 and all(c['failed_aa_remains_unqualified'] for c in controls['quality_controls'])
assert controls['reader_sha256']==sha((study/'analyze.py').read_bytes())
if len(sys.argv)>1:
 for path,digest in (meta['source_sha256']|oracle['source_sha256']).items():assert sha((Path(sys.argv[1])/path).read_bytes())==digest,path
print(f'Verified {len(manifest)} payloads, 240 full outputs and dispatch traces, CPU/wall controls and validation.')
