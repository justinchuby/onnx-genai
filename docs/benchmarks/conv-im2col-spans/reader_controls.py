import copy,json

def controls(plan,rows,reader,read_file,oracle):
 value=reader.analyze(plan,rows,read_file,oracle);assert value['formal_processes']==216 and value['distinct_verified_outputs']==12
 receipts=[{'name':'complete_real_data','synthetic':False,'accepted':True}]
 def reject(name,mutate):
  data=copy.deepcopy(rows);mutate(data)
  try:reader.analyze(plan,data,read_file,oracle)
  except (AssertionError,KeyError,ValueError,TypeError):receipts.append({'name':name,'synthetic':True,'rejected':True})
  else:raise AssertionError('invalid synthetic corpus accepted: '+name)
 reject('missing_process',lambda r:r.pop())
 reject('duplicate_identity',lambda r:r.__setitem__(0,copy.deepcopy(r[1])))
 reject('zero_calls_in_work',lambda r:next(x for x in r if x['kind']=='work').__setitem__('calls',0))
 reject('impossible_cpu',lambda r:r[0].__setitem__('cpu_s',r[0]['wall_s']*(plan['logical_cpus']+1)+1))
 reject('missing_native_gemm',lambda r:next(x for x in r if x['case']==1)['probe'].__setitem__('native_gemm_witnesses',[0,0]))
 reject('mlas_routes_in_default',lambda r:r[0]['probe'].__setitem__('mlas_gemm_witnesses',[1,0]))
 reject('output_bytes_changed',lambda r:r[0].__setitem__('output_sha256','0'*64))
 reject('stdout_bytes_changed',lambda r:r[0].__setitem__('stdout_sha256','0'*64))
 for kind in ['aa','setup','double','nonpositive']:
  data=copy.deepcopy(rows)
  def row(k):return next(x for x in data if (x['label'],x['case'],x['kind'],x['rep'])==('after',0,k,0))
  s,w,d=[row(k) for k in ['setup','work','double']]
  if kind=='aa':
   for x in [w,d]:x['cpu_s']=s['cpu_s']+1.3*(x['cpu_s']-s['cpu_s'])
  elif kind=='setup':s['cpu_s']=w['cpu_s']*.9;s['wall_s']=max(s['wall_s'],s['cpu_s'])
  elif kind=='double':d['cpu_s']=w['cpu_s']
  else:w['cpu_s']=s['cpu_s']
  value=reader.analyze(plan,data,read_file,oracle);assert not value['cells'][0]['qualified'];json.dumps(value,allow_nan=False);receipts.append({'name':kind,'synthetic':True,'qualified':False})
 return receipts
