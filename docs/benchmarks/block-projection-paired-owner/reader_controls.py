from pathlib import Path
import json,gzip,hashlib,copy,importlib.util
root=Path(__file__).parent;spec=importlib.util.spec_from_file_location('reader',root/'analyze.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rows=json.loads(gzip.decompress((root/'receipts.json.gz').read_bytes()));assert len(m.analyze(rows)['cells'])==2
fixture=copy.deepcopy(rows)
for p in fixture:
 p['wall_total_s']=100.;p['cpu_total_s']=90.
 for r in p['records']:r['wall_s']={'empty':0.,'work':1.,'double':2.}[r['kind']]
assert all(c['qualified'] for c in m.analyze(fixture)['cells']);controls=[]
def reject(name,mutate):
 data=copy.deepcopy(fixture);mutate(data)
 try:m.analyze(data)
 except (AssertionError,KeyError,TypeError,ValueError,IndexError):controls.append({'name':name,'rejected':True})
 else:raise AssertionError(name)
reject('missing process',lambda r:r.pop());reject('missing window',lambda r:r[0]['records'].pop());reject('duplicate window',lambda r:r[0]['records'].__setitem__(0,r[0]['records'][1]))
reject('different workspace owner',lambda r:r[0].__setitem__('same_workspace_owner',False));reject('different input owner',lambda r:r[0].__setitem__('same_input_owner',False))
reject('wrong logical demand',lambda r:r[0]['records'][0].__setitem__('logical_bytes',0));reject('wrong physical demand',lambda r:r[0]['records'][0].__setitem__('physical_bytes',0))
reject('zero timed work',lambda r:r[0]['records'][0].__setitem__('wall_s',0));reject('negative empty timer',lambda r:next(x for x in r[0]['records'] if x['kind']=='empty').__setitem__('wall_s',-1))
reject('wrong fixed calls',lambda r:r[0]['records'][0].__setitem__('calls',1));reject('bad final output',lambda r:r[0].__setitem__('output_sha256','0'*64))
quality=[]
for frozen in [True,False]:
 data=copy.deepcopy(fixture)
 for p in data:
  for c in [1,4]:
   selected=sorted([r for r in p['records'] if r['construction']==c and r['kind']=='work' and r['frozen']==frozen],key=lambda r:r['index'])
   for r in selected[2:]:r['wall_s']*=1.2
 assert all(not c['qualified'] and any('A/A' in r for r in c['reasons']) for c in m.analyze(data)['cells']);quality.append({'frozen':frozen,'failed_aa_preserved':True})
(root/'reader-controls.json').write_text(json.dumps({'synthetic_qualified':True,'controls':controls,'quality_controls':quality,'reader_sha256':hashlib.sha256((root/'analyze.py').read_bytes()).hexdigest(),'zero_empty_only':True},indent=2))
(root/'reader-correction.json').write_text(json.dumps({'initial_reader':'analyze-initial.py','initial_issue':'asserted positive elapsed time for zero-call empty windows; some are below one Instant tick and return zero','correction':'allow zero only for empty windows; negative empty and zero timed work remain rejected; no sample rerun, unchanged 50ms work floor and all qualification thresholds','timing_attempts':1},indent=2))
print('Qualified full synthetic fixture; rejected',len(controls),'malformed variants; kept both arms failed A/A unqualified.')
