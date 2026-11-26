"""Verify historical data without extracting archive paths or running binaries."""
from pathlib import Path,PurePosixPath
import gzip,hashlib,importlib.util,json,re,sys,tarfile
ROOT=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def load(name):return json.loads((ROOT/name).read_bytes())
def loadgz(name):return json.loads(gzip.decompress((ROOT/name).read_bytes()))
def module(name):
 spec=importlib.util.spec_from_file_location('conv_gap_'+name,ROOT/(name+'.py'))
 value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

def main():
 manifest=load('manifest.json')
 files={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest.json'}
 assert files==set(manifest),'sealed artifact file inventory changed'
 for name,digest in manifest.items():assert sha((ROOT/name).read_bytes())==digest,name
 plan=loadgz('plan.json.gz');rows=loadgz('receipts.json.gz');build=load('build-metadata.json')
 assert sha(gzip.decompress((ROOT/'receipts.json.gz').read_bytes()))==plan['receipts_sha256']
 assert sha((ROOT/'probe.rs').read_bytes())==plan['probe_sha256']==build['probe_sha256']
 assert sha((ROOT/'run.py').read_bytes())==plan['runner_sha256']
 assert plan['source_sha256']==build['source_sha256'] and plan['binaries']==build['binaries']
 assert build['temporary_example_removed'] and plan['lock_exclusion_control'] and plan['lock_inode']==build['lock_inode']
 for name,digest in build['source_sha256'].items():
  assert sha(gzip.decompress((ROOT/'source'/(name+'.gz')).read_bytes()))==digest,name
  if len(sys.argv)>1:assert sha((Path(sys.argv[1])/name).read_bytes())==digest,name
 for label in ['default','linked']:
  symbols=gzip.decompress((ROOT/(label+'-symbols.log.gz')).read_bytes()).decode()
  lines=symbols.splitlines();count=sum(bool(re.search(r'mlas_sys|\b_Mlas\w*',s)) for s in lines)
  assert len(lines)==build['binaries'][label]['symbol_lines']>100 and any(re.search(r'\b_main$',s) for s in lines)
  assert count==build['binaries'][label]['mlas_symbols'] and ((count==0)==(label=='default'))
 cleanup=load('cleanup.json')
 assert cleanup['matrix_completed'] and cleanup['formal_processes']==648 and cleanup['prior_archive_verifier_exit']==0 and cleanup['no_matching_executable_in_process_snapshot']
 assert cleanup['shared_lock_inode_at_cleanup']>0 and cleanup['cargo_dependency_caches_preserved']
 assert {x['path']:x['sha256'] for x in cleanup['removed_private_consumed_executables']}=={x['path']:x['sha256'] for x in build['binaries'].values()}
 inspection=load('inspection.json');disasm=(ROOT/'nchwc-block-size.disasm').read_bytes()
 assert inspection['binary_sha256']==build['binaries']['linked']['sha256'] and inspection['disassembly_sha256']==sha(disasm)
 assert inspection['symbol'].encode() in disasm and re.search(rb'mov\s+w0, #0x1',disasm) and b'ret' in disasm
 with tarfile.open(ROOT/'raw-data.tar.gz','r:gz') as archive:
  members=archive.getmembers();names=[m.name for m in members]
  assert len(names)==len(set(names)) and all(m.isfile() and PurePosixPath(m.name).parts[0] in ['raw','pilot'] and '..' not in PurePosixPath(m.name).parts for m in members)
  expected={r[k] for r in rows+plan['calibration'] for k in ['stdout','stderr','output']}
  assert set(names)==expected and len(plan['calibration'])==174
  def read_file(name):
   f=archive.extractfile(name);assert f is not None
   return f.read()
  for r in rows+plan['calibration']:
   assert r['exit']==0 and r['cpu_s']>0 and r['wall_s']>0 and r['cpu_s']<=plan['logical_cpus']*r['wall_s']+.01
   for field in ['stdout','stderr']:assert sha(read_file(r[field]))==r[field+'_sha256']
   assert json.loads(read_file(r['stdout']))==r['probe']
   assert sha(gzip.decompress(read_file(r['output'])))==r['output_sha256']
  reader=module('analyze');summary=reader.analyze(plan,rows,True,read_file)
 assert summary==load('summary.json')==loadgz('initial-summary.json.gz')
 assert summary['formal_processes']==648 and summary['verified_distinct_outputs']==18
 for key,count in [('production_over_reference',10),('generic_over_reference',12)]:
  assert sum(c['comparisons'][key]['qualified'] for c in summary['cells'])==count
 for c in summary['cells']:
  assert c['comparisons']['production_over_reference']['qualified']==(c['case']!=1)
 controls=module('reader_controls').run_controls(plan,rows,reader)
 assert controls==load('reader-controls.json') and len(controls)==13 and sum(not c['synthetic'] for c in controls)==1
 print('PASS: sealed files, 648 formal + 174 calibration receipts, 18 f64 outputs, 13 controls; production 10/12, Generic 12/12 comparisons qualified')
if __name__=='__main__':main()
