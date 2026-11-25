import copy,json

def run_controls(plan,rows,module):
 receipts=[]
 base=module.analyze(plan,rows);assert len(base['cells'])==12 and base['formal_processes']==648
 receipts.append({'control':'complete_real_metadata','accepted':True,'synthetic':False})
 def refuse(name,mutate):
  changed=copy.deepcopy(rows);mutate(changed)
  try:module.analyze(plan,changed)
  except (AssertionError,KeyError,ValueError,TypeError):receipts.append({'control':name,'rejected':True,'synthetic':True})
  else:raise AssertionError('invalid synthetic corpus accepted: '+name)
 refuse('missing_process',lambda r:r.pop())
 refuse('duplicate_identity',lambda r:r.__setitem__(0,copy.deepcopy(r[1])))
 refuse('zero_work_mislabeled',lambda r:next(x for x in r if x['kind']=='work').__setitem__('calls',0))
 refuse('impossible_cpu',lambda r:r[0].__setitem__('cpu_s',(plan['logical_cpus']+1)*r[0]['wall_s']+1))
 refuse('wrong_build_claim',lambda r:r[0]['probe'].__setitem__('linked',not r[0]['probe']['linked']))
 refuse('native_gemm_witness_missing',lambda r:next(x for x in r if x['binary']=='linked' and x['case']==1)['probe']['native_gemm_witnesses'].__setitem__(0,0))
 refuse('native_routes_to_reference',lambda r:next(x for x in r if x['binary']=='linked')['probe']['mlas_gemm_witnesses'].__setitem__(0,1))
 refuse('unstable_output',lambda r:r[0].__setitem__('output_sha256','0'*64))
 # These retain valid corpus shape/clocks but must disqualify the affected cell.
 for kind in ['aa','setup','double','nonpositive']:
  data=copy.deepcopy(rows);target=lambda k:next(x for x in data if (x['label'],x['case'],x['phase'],x['kind'],x['rep'])==('production',0,'reuse',k,0));ss=target('setup');ww=target('work');dd=target('double')
  if kind=='aa':
   dd['cpu_s']=ss['cpu_s']+1.2*(dd['cpu_s']-ss['cpu_s']);ww['cpu_s']=ss['cpu_s']+1.2*(ww['cpu_s']-ss['cpu_s'])
  elif kind=='setup':ss['cpu_s']=ww['cpu_s']*.9;ss['wall_s']=max(ss['wall_s'],ss['cpu_s']*1.1)
  elif kind=='double':dd['cpu_s']=ww['cpu_s']
  else:ww['cpu_s']=ss['cpu_s']
  result=module.analyze(plan,data);cell=next(c for c in result['cells'] if c['case']==0 and c['phase']=='reuse');assert not cell['arms']['production']['qualified'];json.dumps(result,allow_nan=False);receipts.append({'control':'qualification_'+kind,'qualified':False,'synthetic':True,'reasons':cell['arms']['production']['qualification_reasons']})
 return receipts
