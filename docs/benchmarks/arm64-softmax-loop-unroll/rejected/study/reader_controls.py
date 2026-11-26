import copy,csv,hashlib,io,itertools,json,pathlib
out=pathlib.Path('/tmp/nxrt-neon-unroll-study')
source=(out/'analyze.py').read_text();ns={};exec(compile(source[:source.index('raw=gzip.decompress')],str(out/'analyze.py'),'exec'),ns)
shapes=ns['SHAPES'];fields=ns['FIELDS'];analyze=ns['analyze']
rows=[];valid=[]
for s,w,serial in itertools.product(range(3),[1,4,8],['true','false']):
 for n,d in shapes:
  for arm in range(4):
   digest=999 if arm<3 else 1000
   workers=int(serial=='false' and w>1 and n>=2 and n*d>=16384)
   valid.append([s,str(w),serial,str(n),str(d),str(arm),str(digest),'0',str(workers)])
   for rep in range(15): rows.append([s,w,serial,n,d,min(8192,max(1,1048576//(n*d))),rep,arm,1000,2000,digest])
meta={'schema':1,'repetitions':15,'sweeps':3,'arms':{'0':'original_same_arithmetic','1':'unrolled_native','2':'unrolled_native_aa','3':'mlas'},'receipts':[{'sweep':s,'width':w,'exit':0,'cpu_s':1,'wall_s':1,'maxrss_bytes':100,'fpcr_start':0,'fpcr_end_equality_assertion_passed':True} for s,w in itertools.product(range(3),[1,4,8])]}
def pack(rows):
 f=io.StringIO();wr=csv.writer(f);wr.writerow(fields);wr.writerows(rows);return f.getvalue().encode()
def attempt(rows,valid,meta):
 raw=pack(rows);meta=copy.deepcopy(meta);meta['raw_sha256']=hashlib.sha256(raw).hexdigest();return analyze(raw,meta,valid)
assert attempt(rows,valid,meta)['samples']==8640
controls=[]
def refuse(name,r=rows,v=valid,m=meta):
 try:attempt(r,v,m)
 except AssertionError:controls.append({'name':name,'rejected':True})
 else:raise AssertionError(name+' unexpectedly accepted malformed synthetic input')
refuse('vacuity_no_timed_work',r=[])
refuse('missing_one_sample',r=rows[:-1])
refuse('duplicate_sample_identity',r=rows+[rows[0]])
r=copy.deepcopy(rows);r[0][8]=0;refuse('zero_process_cpu',r=r)
r=copy.deepcopy(rows);r[0][-1]+=1;refuse('unstable_full_call_token',r=r)
refuse('missing_validation',v=valid[:-1])
v=copy.deepcopy(valid);v[0][-1]='1';refuse('unexpected_worker_participation',v=v)
m=copy.deepcopy(meta);m['arms']['0']='candidate_alias';refuse('arm_label_drift',m=m)
(out/'reader-controls.json').write_text(json.dumps({'scope':'synthetic parser controls, not measurements','accepted_complete_fixture':True,'complete_samples':8640,'controls':controls,'analyzer_sha256':hashlib.sha256((out/'analyze.py').read_bytes()).hexdigest()},indent=2))
print('Parser accepted complete synthetic matrix and rejected',len(controls),'named malformed/vacuity controls.')
