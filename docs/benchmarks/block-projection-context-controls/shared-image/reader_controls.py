from pathlib import Path
import json,gzip,hashlib,copy,importlib.util
root=Path(__file__).parent;spec=importlib.util.spec_from_file_location('reader',root/'analyze.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rows=json.loads(gzip.decompress((root/'receipts.json.gz').read_bytes()));assert len(m.analyze(rows)['cells'])==3
fixture=copy.deepcopy(rows)
for r in fixture:
 for metric in ['cpu_s','wall_s']:r[metric]={'setup':1.,'work':10.,'double':19.}[r['kind']]
assert all(v['qualified'] for c in m.analyze(fixture)['cells'] for v in c['metrics'].values())
controls=[]
def reject(name,mutate):
 data=copy.deepcopy(fixture);mutate(data)
 try:m.analyze(data)
 except (AssertionError,KeyError,ValueError,TypeError,IndexError):controls.append({'name':name,'rejected':True})
 else:raise AssertionError(name)
reject('empty',lambda r:r.clear());reject('missing',lambda r:r.pop());reject('duplicate',lambda r:r.__setitem__(0,r[1]))
reject('wrong executable arm',lambda r:r[0].__setitem__('arm','foreign'))
reject('wrong calls',lambda r:r[0].__setitem__('calls',99999))
reject('wrong capacity',lambda r:r[0].__setitem__('workspace_bytes',0))
reject('wrong logical capacity',lambda r:r[0].__setitem__('logical_workspace_bytes',0))
reject('wrong constructor width',lambda r:r[0].__setitem__('construction_width',99))
reject('unrefunded quota',lambda r:r[0].__setitem__('final_used',1))
reject('impossible cpu',lambda r:r[0].__setitem__('cpu_s',10000.))
reject('bad output',lambda r:r[0].__setitem__('output_sha256','0'*64))
quality=[]
for arm in ['baseline','candidate']:
 for metric in ['cpu_s','wall_s']:
  data=copy.deepcopy(fixture)
  for r in data:
   if r['kind']=='work' and r['arm']=='null-'+arm:r[metric]*=1.1
  assert all(not c['metrics'][metric]['qualified'] for c in m.analyze(data)['cells'])
  quality.append({'arm':arm,'metric':metric,'failed_aa_remains_unqualified':True})
(root/'reader-controls.json').write_text(json.dumps({'accepted_complete_matrix':True,'synthetic_all_qualified':True,'controls':controls,'quality_controls':quality,'reader_sha256':hashlib.sha256((root/'analyze.py').read_bytes()).hexdigest()},indent=2));print('Accepted full fixture; rejected',len(controls),'invalid variants and preserved',len(quality),'failed A/A cases.')
