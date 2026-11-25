"""Verify archived route-count state and allocation evidence without executing it."""
from pathlib import Path
import gzip,hashlib,json,re,subprocess,sys
folder=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
manifest=json.loads((folder/'manifest.json').read_text());actual={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts};assert actual==set(manifest)
for p,h in manifest.items():assert sha((folder/p).read_bytes())==h,p
probe=folder/'probe';result=json.loads((probe/'results.json').read_text());assert result['exit']==0 and result['production_restored'] and result['all_route_snapshots_identical'];assert re.fullmatch('[0-9a-f]{64}',result['test_binary_sha256'])
raw=gzip.decompress((probe/'probe.log.gz').read_bytes());assert sha(raw)==result['log_sha256'] and b'1 passed; 0 failed' in raw and b'CONTROL allocations=1 frees=1' in raw
rows=[]
for values in re.findall(rb'CASE experts=(\d+) rows=(\d+) k=(\d+) active=(\d+) unique=(\d+) calls=(\d+) baseline_alloc=(\d+) baseline_free=(\d+) candidate_alloc=(\d+) candidate_free=(\d+)',raw):
 rows.append(dict(zip(['experts','rows','k','active','unique','calls','baseline_alloc','baseline_free','candidate_alloc','candidate_free'],map(int,values))))
assert rows==result['cases'] and len(rows)==15
assert {(r['experts'],r['rows'],r['k']) for r in rows}=={(e,n,k) for e in [1,7,65] for n in [1,8,64] for k in {1,min(3,e)}}
for r in rows:
 assert r['calls']==2048 and r['active']==r['rows']*r['k']
 assert r['unique']==len({(row+r['k']-1-rank)%r['experts'] for row in range(r['rows']) for rank in range(r['k'])})
 assert r['baseline_alloc']==r['baseline_free']>0 and r['candidate_alloc']==r['candidate_free']==0
reference=(probe/'reference.rs').read_bytes();assert sha(reference)==result['reference_sha256']
metric='crates/onnx-runtime-ep-cpu/src/weight_offload.rs'
for p,h in result['source_sha256'].items():
 source=gzip.decompress((probe/(Path(p).name+'.gz')).read_bytes())
 suffix=b'\n'+(probe/'allocator.rs').read_bytes() if p.endswith('/lib.rs') else b'\n'+reference+b'\n'+(probe/'probe.rs').read_bytes() if p==metric else b''
 if suffix:assert source.endswith(suffix);source=source[:-len(suffix)]
 assert sha(source)==h,p
 if len(sys.argv)>1:assert (Path(sys.argv[1])/p).read_bytes()==source,p
if len(sys.argv)>1:
 old=subprocess.check_output(['git','-C',str(Path(sys.argv[1]).resolve()),'show',result['base']+':'+metric],text=True);start=old.index('    pub fn record_routes(');old=old[start:old.index('\n    fn snapshot(',start)].replace('pub fn record_routes(', 'fn reference_record_routes(')
 assert reference==('\n#[cfg(test)]\nimpl WeightOffloadMetrics {\n'+old+'\n}\n').encode()
prototype=folder/'iterator-prototype';prior=json.loads((prototype/'results.json').read_text())
assert prior['production_restored'] and prior['exit']==0 and prior['all_route_snapshots_identical']
assert len(prior['cases'])==15 and all(r['candidate_alloc']==r['candidate_free']==0 for r in prior['cases'])
assert sha(gzip.decompress((prototype/'probe.log.gz').read_bytes()))==prior['log_sha256']
v=folder/'validation';validation=json.loads((v/'validation.json').read_text());assert len(validation['results'])==8 and all(r['exit']==0 for r in validation['results'])
for p,h in validation['source_sha256'].items():assert result['source_sha256'][p]==h,p
for r in validation['results']:
 log=gzip.decompress((v/(r['name']+'.log.gz')).read_bytes());assert sha(log)==r['log_sha256']
 if 'counts' in r:assert r['counts'] and all(int(p)>0 and int(f)==0 for p,f,i in r['counts'])
 if r['name']=='metrics':
  for name in ['derived_route_counts_preserve_global_and_layer_snapshots','route_metrics_keep_zero_members_and_saturating_per_layer_totals']:assert ('test weight_offload::tests::'+name+' ... ok').encode() in log
 if r['name'] in ['qmoe','qmoe-mlas']:
  for name in ['prefetch_pipeline_is_byte_and_stat_identical_to_serial','cached_and_uncached_routes_and_logits_are_bit_identical','route_first_preserves_exact_accumulation_order_for_top_k']:assert ('test kernels::qmoe::tests::'+name+' ... ok').encode() in log
print(f'Verified {len(manifest)} sealed payloads, all 15 route-count cases, original metrics body and final validation/source restoration.')
