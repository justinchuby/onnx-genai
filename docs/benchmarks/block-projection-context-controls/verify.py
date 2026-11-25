from pathlib import Path
import gzip,base64,hashlib,importlib.util,json,math,struct,sys
folder=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
manifest=json.loads((folder/'manifest.json').read_text());actual={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts};assert actual==set(manifest)
for p,h in manifest.items():assert sha((folder/p).read_bytes())==h,p
s=folder/'shared-image';meta=json.loads((s/'metadata.json').read_text());raw=gzip.decompress((s/'receipts.json.gz').read_bytes());assert sha(raw)==meta['receipts_sha256'];rows=json.loads(raw)
spec=importlib.util.spec_from_file_location('reader',s/'analyze.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert json.loads(json.dumps(m.analyze(rows)))==json.loads((s/'results.json').read_text())
assert sha((s/'run.py').read_bytes())==meta['runner_sha256'];assert len(meta['single_decode_symbol'])==1
seal=json.loads((s/'binary-seal.json').read_text());assert seal['sha256']==meta['binary_sha256'] and seal['verified_after_terminal_matrix']
outputs=json.loads(gzip.decompress((s/'full-outputs.json.gz').read_bytes()));logs=json.loads(gzip.decompress((s/'process-logs.json.gz').read_bytes()));traces=json.loads(gzip.decompress((s/'dispatch-traces.json.gz').read_bytes()))
assert len(rows)==len(outputs)==len(traces)==90 and len(logs)==180
old=folder.parent/'block-projection-execution-context/study';previous=json.loads(gzip.decompress((old/'receipts.json.gz').read_bytes()));mapping={0:0,1:9,2:7}
for r in rows:
 name=f"r{r['rep']}-c{r['case']}-{r['kind']}-{r['arm']}";raw=base64.b64decode(outputs[name+'.f32']);assert sha(raw)==r['output_sha256'];width=r['inter'] if r['matmul'] else r['hidden'];assert len(raw)==r['rows']*width*4
 values=struct.unpack('<'+'f'*(len(raw)//4),raw);assert all(math.isfinite(v) for v in values) and any(v!=0 for v in values)
 assert [max(range(width),key=lambda c:(values[j*width+c],-c)) for j in range(r['rows'])]==r['argmax']
 receipt=json.loads(logs[name+'.stdout'].strip().splitlines()[-1]);assert all(r[k]==v for k,v in receipt.items())
 variants=[e.get('args',{}).get('kernel_variant') for e in traces[name+'.f32.trace.json']];assert (('block_matmul.streamed_rows' if r['matmul'] else 'block_moe.streamed_rows') in variants)==(not r['dense'])
 assert {p['output_sha256'] for p in previous if p['case']==mapping[r['case']]}=={r['output_sha256']}
controls=json.loads((s/'reader-controls.json').read_text());assert controls['accepted_complete_matrix'] and controls['synthetic_all_qualified'] and controls['reader_sha256']==sha((s/'analyze.py').read_bytes());assert len(controls['controls'])==11 and all(c['rejected'] for c in controls['controls']);assert len(controls['quality_controls'])==4 and all(c['failed_aa_remains_unqualified'] for c in controls['quality_controls'])
constants=json.loads((folder/'assembly/constant-payloads.json').read_text());assert constants['baseline']['sha256']==constants['candidate']['sha256']
asm=json.loads((folder/'assembly/assembly.json').read_text())
for label in ['baseline','candidate']:
 for kind in ['decode','project']:
  assert asm[label][kind]['present'] and asm[label][kind]['instructions']>0 and sha((folder/'assembly'/(label+'-'+kind+'.asm')).read_bytes())==asm[label][kind]['sha256']
v=folder/'cache-validation';validation=json.loads((v/'validation.json').read_text());assert len(validation['results'])==8 and all(r['exit']==0 for r in validation['results'])
for r in validation['results']:
 assert sha(gzip.decompress((v/(r['name']+'.log.gz')).read_bytes()))==r['log_sha256']
 if 'test' in r['command']:assert r['counts'] and all(int(p)>0 and int(f)==0 for p,f,i in r['counts'])
for p,h in validation['source_sha256'].items():
 assert sha(gzip.decompress((v/'sources'/(p+'.gz')).read_bytes()))==h
 if len(sys.argv)>1:assert sha((Path(sys.argv[1])/p).read_bytes())==h,p
print(f'Verified {len(manifest)} payloads,90 complete shared-image outputs, original-fixture equality and cache-fix validation.')
