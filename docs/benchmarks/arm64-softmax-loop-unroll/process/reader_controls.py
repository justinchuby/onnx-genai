import copy,hashlib,importlib.util,itertools,json,pathlib
out=pathlib.Path('/tmp/nxrt-neon-unroll-process');spec=importlib.util.spec_from_file_location('reader',out/'analyze.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
rows=[];meta={'schema':3,'repetitions':3,'work_elements':134217728,'hw.logicalcpu':'10','arms':{'0':'original_same_arithmetic','1':'unrolled_native','2':'unrolled_native_aa','3':'mlas','4':'original_outlined'}}
for kind,rep,w,c,a in itertools.product(['setup','work','double'],range(3),[1,8],range(8),range(5)):
 if kind=='double' and (c<4 or a not in [0,1]):continue
 n,d=module.SHAPES[c];calls=0 if kind=='setup' else max(1,meta['work_elements']//(n*d))*(2 if kind=='double' else 1)
 cpu={'setup':1.,'work':10.,'double':19.}[kind]
 rows.append({'kind':kind,'rep':rep,'width':w,'case':c,'arm':a,'rows':n,'cols':d,'calls':calls,'exit':0,'cpu_s':cpu,'wall_s':cpu,'maxrss_bytes':100,'worst_error':0.,'digest':1000 if a==3 else 999,'fpcr':0,'workers':int(w>1 and n>=2 and n*d>=16384),'hits':n*calls if a!=3 and d>=8 else 0})
assert len(rows)==528 and len(module.analyze(rows,meta)['cells'])==16
controls=[]
def refuse(name,mutate):
 sample=copy.deepcopy(rows);mutate(sample)
 try:module.analyze(sample,meta)
 except (AssertionError,KeyError):controls.append({'name':name,'rejected':True})
 else:raise AssertionError(name+' accepted malformed synthetic input')
refuse('vacuity_only_setup_processes',lambda r:r.__setitem__(slice(None),[x for x in r if x['kind']=='setup']))
refuse('missing_process',lambda r:r.pop())
refuse('duplicate_process_identity',lambda r:r.append(r[0]))
refuse('work_labeled_with_zero_calls',lambda r:next(x for x in r if x['kind']=='work').__setitem__('calls',0))
refuse('no_native_work_witness',lambda r:next(x for x in r if x['kind']=='work' and x['case']==4 and x['arm']==1).__setitem__('hits',0))
refuse('impossible_cpu_capacity',lambda r:r[0].__setitem__('cpu_s',1000.))
refuse('unstable_full_output_digest',lambda r:r[0].__setitem__('digest',123))
refuse('missing_validation_bound',lambda r:r[0].pop('worst_error'))
refuse('missing_parallel_worker',lambda r:next(x for x in r if x['width']==8 and x['case']==5).__setitem__('workers',0))
(out/'reader-controls.json').write_text(json.dumps({'scope':'synthetic controls, not performance measurements','accepted_complete_matrix':True,'processes':528,'controls':controls,'reader_sha256':hashlib.sha256((out/'analyze.py').read_bytes()).hexdigest()},indent=2));print('Accepted complete fixture and rejected',len(controls),'named malformed/vacuity controls')
