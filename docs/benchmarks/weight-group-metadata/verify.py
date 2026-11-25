"""Verify the sealed evidence; never run a build or benchmark."""
from pathlib import Path
import gzip,hashlib,json,re,sys,subprocess
folder=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
manifest=json.loads((folder/'manifest.json').read_text())
actual={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts}
assert actual==set(manifest)
for p,h in manifest.items():assert sha((folder/p).read_bytes())==h,p
probe=folder/'probe';result=json.loads((probe/'results.json').read_text())
assert result['production_restored'] and result['baseline']=='d42eb1d4aef7cf56148bc87ad91b3571b33060df'
assert [r['arm'] for r in result['results']]==['baseline','candidate']
for arm in result['results']:
 log=gzip.decompress((probe/(arm['arm']+'.log.gz')).read_bytes())
 assert arm['exit']==0 and sha(log)==arm['log_sha256']
 assert b'1 passed; 0 failed' in log and b'CONTROL allocations=1 frees=1' in log
 parsed=[dict(zip(['members','iterations','allocations','frees'],map(int,m))) for m in re.findall(rb'GROUP members=(\d+) iterations=(\d+) allocations=(\d+) frees=(\d+) issuer_delta=0 final_used=0',log)]
 assert parsed==arm['counts'] and [r['members'] for r in parsed]==[0,1,2,3,4,8]
 for row in parsed:
  assert row['iterations']==2048
  if row['members']==0 or (arm['arm']=='candidate' and row['members']<=3):assert row['allocations']==row['frees']==0
  else:assert row['allocations']==row['frees']>0
 for source in ['lib.rs','weight_cache.rs','block_quantized_moe.rs']:
  raw=gzip.decompress((probe/(arm['arm']+'-'+source+'.gz')).read_bytes())
  if source in ['lib.rs','weight_cache.rs']:
   suffix=b'\n'+(probe/('allocator.rs' if source=='lib.rs' else 'probe.rs')).read_bytes()
   assert raw.endswith(suffix);raw=raw[:-len(suffix)]
  path='crates/onnx-runtime-ep-cpu/src/'+('' if source=='lib.rs' else 'kernels/')+source
  if arm['arm']=='candidate':
   assert sha(raw)==result['candidate_source_sha256'][path]
  elif len(sys.argv)>1:
   assert raw==subprocess.check_output(['git','-C',str(Path(sys.argv[1]).resolve()),'show',result['baseline']+':'+path])
initial=json.loads((folder/'initial-probe/results.json').read_text())
assert initial['results'][0]==result['results'][0], 'unchanged baseline must be reused'
for arm in initial['results']:
 raw=gzip.decompress((folder/'initial-probe'/(arm['arm']+'.log.gz')).read_bytes())
 assert sha(raw)==arm['log_sha256'] and arm['exit']==0
 if arm['arm']=='candidate':
  assert all(r['allocations']==r['frees']==6144 for r in arm['counts'] if r['members']>3)
assert result['baseline_reused_from']
validation=json.loads((folder/'validation/validation.json').read_text())
assert len(validation['results'])==5 and all(r['exit']==0 for r in validation['results'])
for record in validation['results']:
 raw=gzip.decompress((folder/'validation'/(record['name']+'.log.gz')).read_bytes())
 assert sha(raw)==record['log_sha256']
 if record['name'] in ['native','mlas']:
  assert record['counts'] and all(int(p)>0 and int(f)==0 for p,f,i in record['counts'])
  assert b'test kernels::weight_cache::tests::larger_projection_groups_keep_atomic_admission_and_reader_lifetimes ... ok' in raw
for p,h in validation['source_sha256'].items():assert result['candidate_source_sha256'][p]==h
if len(sys.argv)>1:
 for p,h in result['candidate_source_sha256'].items():assert sha((Path(sys.argv[1])/p).read_bytes())==h,p
print(f'Verified {len(manifest)} sealed payloads, both six-group allocation probes, and final validation/source restoration.')
