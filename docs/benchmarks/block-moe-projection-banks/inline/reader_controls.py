import copy,hashlib,importlib.util,itertools,json,pathlib
OUT=pathlib.Path('/tmp/nxrt-block-banks-inline-study');spec=importlib.util.spec_from_file_location('reader',OUT/'analyze.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
rows=[]
for rep,c,kind,arm in itertools.product(range(3),range(6),['setup','work','double'],['baseline','candidate','null']):
 if kind=='double' and arm=='null':continue
 m,h,i,width,dense,gated=module.SHAPES[c];calls=0 if kind=='setup' else max(1,268435456//(m*h*i*2*(3 if gated else 2)))*(2 if kind=='double' else 1)
 decode=lambda k,n:((m*n if m>1 and n>1 else 0)+width*min(4,n//width)*k)*4
 maximum=max(decode(h,i),decode(i,h));a,b=(max(h,i),max(2*i,h)) if gated else (h,i)
 cpu={'setup':1.,'work':10.,'double':19.}[kind]
 rows.append({'rep':rep,'case':c,'kind':kind,'arm':arm,'rows':m,'hidden':h,'inter':i,'rayon_width':width,'dense':bool(dense),'gated':bool(gated),'calls':calls,'exit':0,'final_used':0,'cpu_s':cpu,'wall_s':cpu,'maxrss_bytes':100,'output_sha256':'a'*64,'argmax':[0]*m,'workspace_bytes':maximum+m*((a+b+h) if arm=='candidate' else 2*h)*4,'derived_bytes':((2 if m==1 else 4)*h*i*(3 if gated else 2)*4 if dense else 0)})
assert len(rows)==144 and all(c['qualified'] for c in module.analyze(rows)['cells'])
controls=[]
def reject(name,mutate):
 sample=copy.deepcopy(rows);mutate(sample)
 try:module.analyze(sample)
 except (AssertionError,KeyError):controls.append({'name':name,'rejected':True})
 else:raise AssertionError('accepted malformed synthetic input: '+name)
reject('vacuity_only_setup',lambda r:r.__setitem__(slice(None),[x for x in r if x['kind']=='setup']))
reject('duplicate_identity',lambda r:r.append(r[0]))
reject('missing_process',lambda r:r.pop())
reject('zero_calls_labeled_work',lambda r:next(x for x in r if x['kind']=='work').__setitem__('calls',0))
reject('impossible_cpu_capacity',lambda r:r[0].__setitem__('cpu_s',1000.))
reject('unreleased_backing',lambda r:r[0].__setitem__('final_used',1))
reject('unstable_output_digest',lambda r:r[0].__setitem__('output_sha256','b'*64))
reject('unfunded_workspace',lambda r:r[0].__setitem__('workspace_bytes',1))
reject('wrong_dense_payload_funding',lambda r:next(x for x in r if x['dense']).__setitem__('derived_bytes',1))
reject('missing_width_witness',lambda r:r[0].pop('rayon_width'))
sample=copy.deepcopy(rows)
for r in sample:
 if r['kind']=='work' and r['arm']=='null':r['cpu_s']*=1.1
assert all(not c['qualified'] and 'baseline/null A/A exceeds 5%' in c['reasons'] for c in module.analyze(sample)['cells'])
(OUT/'reader-controls.json').write_text(json.dumps({'scope':'synthetic, no performance samples','accepted_complete_matrix':True,'processes':144,'controls':controls,'failed_aa_cells_remain_unqualified':True,'reader_sha256':hashlib.sha256((OUT/'analyze.py').read_bytes()).hexdigest()},indent=2))
print('Accepted full fixture; rejected',len(controls),'malformed/vacuity cases; retained failed A/A cells')
