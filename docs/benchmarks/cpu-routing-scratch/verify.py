"""Verify archived source, numerical validation and allocation observations."""
from pathlib import Path
import gzip,hashlib,itertools,json,re,subprocess,sys
folder=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
manifest=json.loads((folder/'manifest.json').read_text())
actual={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts}
assert actual==set(manifest)
for p,h in manifest.items():assert sha((folder/p).read_bytes())==h,p
probe=folder/'probe';result=json.loads((probe/'results.json').read_text())
assert result['production_restored'] and result['reference_matches_main_except_name'] and result['exit']==0
assert re.fullmatch('[0-9a-f]{64}',result['test_binary_sha256'])
raw=gzip.decompress((probe/'probe.log.gz').read_bytes())
assert sha(raw)==result['log_sha256'] and b'1 passed; 0 failed' in raw
assert b'CONTROL allocations=1 frees=1' in raw
rows=[]
for values in re.findall(rb'CASE experts=(\d+) k=(\d+) normalize=(true|false) aggregation=(true|false) calls=(\d+) baseline_alloc=(\d+) baseline_free=(\d+) candidate_alloc=(\d+) candidate_free=(\d+)',raw):
 e,k,n,a,c,ba,bf,ca,cf=values
 rows.append(dict(experts=int(e),k=int(k),normalize=n==b'true',aggregation=a==b'true',calls=int(c),baseline_alloc=int(ba),baseline_free=int(bf),candidate_alloc=int(ca),candidate_free=int(cf)))
assert rows==result['cases'] and len(rows)==28
expected={(e,k,n,a) for e in [1,7,65,257] for k in {1,e} for n,a in itertools.product([False,True],repeat=2)}
assert {(r['experts'],r['k'],r['normalize'],r['aggregation']) for r in rows}==expected
for r in rows:
 assert r['calls']==2048 and r['baseline_alloc']==r['baseline_free']>0
 assert r['candidate_alloc']==r['candidate_free']==0
sources={}
for p,h in result['source_sha256'].items():
 source=gzip.decompress((probe/(Path(p).name+'.gz')).read_bytes())
 suffix_name='allocator.rs' if p.endswith('/lib.rs') else 'probe.rs' if p.endswith('/moe.rs') else None
 if suffix_name:
  suffix=b'\n'+(probe/suffix_name).read_bytes();assert source.endswith(suffix);source=source[:-len(suffix)]
 assert sha(source)==h,p;sources[p]=source
 if len(sys.argv)>1:assert (Path(sys.argv[1])/p).read_bytes()==source,p
moe='crates/onnx-runtime-ep-cpu/src/kernels/moe.rs';s=sources[moe].decode();start=s.index('pub(super) fn reference_routing_weights(');reference=s[start:s.index('\n#[cfg(test)]\nmod routing_scratch_tests',start)].strip()
assert sha(reference.encode())==result['reference_sha256']
if len(sys.argv)>1:
 old=subprocess.check_output(['git','-C',str(Path(sys.argv[1]).resolve()),'show',result['base']+':'+moe],text=True)
 start=old.index('pub(super) fn routing_weights(');old=old[start:old.index('\nfn activate(',start)].replace('fn routing_weights(', 'fn reference_routing_weights(').strip()
 assert old==reference
v=folder/'validation';validation=json.loads((v/'validation.json').read_text())
assert len(validation['results'])==6 and all(r['exit']==0 for r in validation['results'])
for p,h in validation['source_sha256'].items():assert result['source_sha256'][p]==h,p
for r in validation['results']:
 log=gzip.decompress((v/(r['name']+'.log.gz')).read_bytes());assert sha(log)==r['log_sha256']
 if r['name'] in ['native','mlas','oracles']:assert r['counts'] and all(int(p)>0 and int(f)==0 for p,f,i in r['counts'])
 if r['name'] in ['native','mlas']:
  for name in ['reused_routing_preserves_reference_bits_across_geometry_and_special_values','routing_keeps_total_order_ties_signed_zero_and_zero_denominator']:
   assert ('test kernels::moe::routing_scratch_tests::'+name+' ... ok').encode() in log
print(f'Verified {len(manifest)} sealed payloads, 28 paired allocation cases, frozen reference and final source restoration/validation.')
