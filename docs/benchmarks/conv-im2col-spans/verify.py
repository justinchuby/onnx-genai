"""Verify retained CPU/bit-equivalence evidence without executing native code."""
from pathlib import Path,PurePosixPath
import gzip,hashlib,importlib.util,json,sys,tarfile
ROOT=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def load(name):return json.loads((ROOT/name).read_bytes())
def gz(name):return gzip.decompress((ROOT/name).read_bytes())
def module(name):
 spec=importlib.util.spec_from_file_location('im2col_'+name,ROOT/(name+'.py'));value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

def verify():
 manifest=load('manifest.json');files={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest.json'};assert set(manifest)==files
 for name,digest in manifest.items():assert sha((ROOT/name).read_bytes())==digest,name
 build=load('build-metadata.json');plan=json.loads(gz('plan.json.gz'));rows=json.loads(gz('receipts.json.gz'))
 cleanup=load('cleanup.json');assert cleanup['removed_private_probe_sha256']==build['binary_sha256'] and cleanup['matrix_completed'] and cleanup['formal_processes']==216 and cleanup['all_child_receipts_exit_zero'] and cleanup['cargo_caches_and_lock_preserved']
 assert plan['binary_sha256']==build['binary_sha256'] and plan['lock_exclusion'] and build['temporary_module_and_example_removed']
 assert sha(gz('receipts.json.gz'))==plan['receipts_sha256'] and sha((ROOT/'run.py').read_bytes())==plan['runner_sha256']
 for name in ['before.rs','after.rs']:assert sha(gz(name+'.gz'))==build['source_sha256'][name]
 assert sha((ROOT/'probe.rs').read_bytes())==build['source_sha256']['probe.rs']
 if len(sys.argv)>1:assert sha((Path(sys.argv[1])/'crates/onnx-runtime-ep-cpu/src/kernels/conv_ref.rs').read_bytes())==build['source_sha256']['after.rs']
 symbols=gz('symbols.log.gz').decode();assert '_main' in symbols and 'im2col_before' in symbols and 'mlas_sys' not in symbols and not any(line.split()[-1].startswith('_Mlas') for line in symbols.splitlines() if line.split())
 validation=load('validation.json');assert validation['source_sha256']==build['source_sha256']['after.rs'] and len(validation['results'])==11
 for r in validation['results']:
  data=gz('validation-'+r['name']+'.log.gz');assert r['exit']==0 and sha(data)==r['log_sha256']
  if 'counts' in r:assert all(int(p)>0 and int(f)==0 for p,f,i in r['counts'])
 integration=load('integration-validation.json');assert integration['source_sha256']==build['source_sha256']['after.rs'] and len(integration['results'])==12
 assert len(integration['initial_failed_checks'])==1
 for failed in integration['initial_failed_checks']:
  assert failed['exit']!=0 and sha(gz('integration-initial-'+failed['name']+'.log.gz'))==failed['log_sha256']
 for r in integration['results']:
  assert r['exit']==0 and sha(gz('integration-'+r['name']+'.log.gz'))==r['log_sha256']
  if 'counts' in r:assert all(int(p)>0 and int(f)==0 for p,f,i in r['counts'])
 negative=load('negative.json');assert negative['source_restored_sha256']==build['source_sha256']['after.rs'] and len(negative['results'])==2
 for r in negative['results']:
  assert r['exit']!=0 and (r['selected'],r['passed'],r['failed'])==(1,0,1)
  assert sha(gz('negative-'+r['name']+'.log.gz'))==r['log_sha256']
 with tarfile.open(ROOT/'raw-data.tar.gz','r:gz') as archive:
  members=archive.getmembers();names=[m.name for m in members];assert len(names)==len(set(names))
  assert all(m.isfile() and PurePosixPath(m.name).parts[0] in ['raw','pilot'] and '..' not in PurePosixPath(m.name).parts for m in members)
  assert set(names)=={r[k] for r in rows+plan['calibration'] for k in ['stdout','stderr','output']}
  def read_file(name):
   f=archive.extractfile(name);assert f is not None;return f.read()
  for r in plan['calibration']:
   assert r['exit']==0 and 0<r['cpu_s']<=plan['logical_cpus']*r['wall_s']+.01
   assert sha(read_file(r['stdout']))==r['stdout_sha256'] and sha(read_file(r['stderr']))==r['stderr_sha256']
   assert sha(gzip.decompress(read_file(r['output'])))==r['output_sha256']
  reader=module('analyze');oracle=module('oracle').oracle
  cache={i:oracle(i) for i in range(6)}
  summary=reader.analyze(plan,rows,read_file,lambda i:cache[i]);assert summary==load('summary.json')
  controls=module('reader_controls').controls(plan,rows,reader,read_file,lambda i:cache[i]);assert controls==load('reader-controls.json') and len(controls)==13
 print('PASS: sealed source, 11 validation commands, two rejected implementation mutants, 216 formal processes, 12 f64 outputs and 13 reader controls')
if __name__=='__main__':verify()
