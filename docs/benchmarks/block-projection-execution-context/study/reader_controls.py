import copy,gzip,hashlib,importlib.util,json,pathlib
root=pathlib.Path(__file__).parent
spec=importlib.util.spec_from_file_location('reader',root/'analyze.py');reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
rows=json.loads(gzip.decompress((root/'receipts.json.gz').read_bytes()))
assert len(reader.analyze(rows)['cells'])==10
controls=[]
def reject(name,mutate):
 data=copy.deepcopy(rows);mutate(data)
 try:reader.analyze(data)
 except (AssertionError,KeyError,ValueError,IndexError,TypeError):controls.append({'name':name,'rejected':True})
 else:raise AssertionError(name+' was accepted')
reject('empty',lambda r:r.clear())
reject('missing process',lambda r:r.pop())
reject('duplicate process',lambda r:r.__setitem__(0,r[1]))
reject('wrong execution width',lambda r:r[0].__setitem__('rayon_width',99))
reject('wrong construction width',lambda r:r[0].__setitem__('construction_width',99))
reject('wrong logical demand',lambda r:r[0].__setitem__('logical_workspace_bytes',0))
reject('wrong construction demand',lambda r:r[0].__setitem__('construction_workspace_bytes',0))
reject('wrong physical peak',lambda r:r[0].__setitem__('workspace_bytes',0))
reject('missing retained refund',lambda r:r[0].__setitem__('final_used',1))
reject('physical CPU impossibility',lambda r:r[0].__setitem__('cpu_s',r[0]['wall_s']*100))
reject('nonfinite wall',lambda r:r[0].__setitem__('wall_s',float('nan')))
reject('wrong fixed work',lambda r:r[0].__setitem__('calls',123456789))
reject('bad output',lambda r:r[0].__setitem__('output_sha256','0'*64))
synthetic=copy.deepcopy(rows)
for r in synthetic:
 for metric in ['cpu_s','wall_s']:r[metric]={'setup':1.,'work':10.,'double':19.}[r['kind']]
assert all(m['qualified'] for c in reader.analyze(synthetic)['cells'] for m in c['metrics'].values())
quality_controls=[]
for metric in ['cpu_s','wall_s']:
 sample=copy.deepcopy(synthetic)
 for r in sample:
  if r['kind']=='work' and r['arm']=='null':r[metric]*=1.1
 cells=reader.analyze(sample)['cells']
 assert all(not c['metrics'][metric]['qualified'] and 'baseline/null A/A exceeds 5%' in c['metrics'][metric]['reasons'] for c in cells)
 quality_controls.append({'metric':metric,'failed_aa_remains_unqualified':True})
(root/'reader-controls.json').write_text(json.dumps({'accepted_complete_matrix':True,'reader_sha256':hashlib.sha256((root/'analyze.py').read_bytes()).hexdigest(),'controls':controls,'synthetic_all_qualified':True,'quality_controls':quality_controls},indent=2))
print('Accepted complete matrix; rejected',len(controls),'malformed variants')
