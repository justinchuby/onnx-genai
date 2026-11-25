import os,pathlib,re,subprocess,sys,json,hashlib
root=pathlib.Path('/tmp/nxrt-qmoe-workspace');out=pathlib.Path('/tmp/nxrt-conv-driver-validation');out.mkdir(exist_ok=True)
ns={};exec(pathlib.Path('/tmp/nxrt-prefetch-groups-check.py').read_text().split('commands=')[0],ns);env=ns['env'];env['RAYON_NUM_THREADS']='1'
commands=[('list',['test','--offline','-p','onnx-runtime-ep-cpu','--lib','kernels::conv::tests::','--','--list'],{}),('conv',['test','--offline','-p','onnx-runtime-ep-cpu','--lib','kernels::conv::tests::'],{}),('conv-generic',['test','--offline','-p','onnx-runtime-ep-cpu','--lib','kernels::conv::tests::gemm_driver_matches_exact_scalar_values_across_groups_batches_and_empty_axes','--','--exact'],{'NXRT_CPU_GEMM_BACKEND':'generic'}),('allocation',['test','--offline','-p','onnx-runtime-ep-cpu','--test','conv_im2col_allocation','--','--nocapture'],{}),('allocation-generic',['test','--offline','-p','onnx-runtime-ep-cpu','--test','conv_im2col_allocation','--','--nocapture'],{'NXRT_CPU_GEMM_BACKEND':'generic'}),('clippy',['clippy','--offline','-p','onnx-runtime-ep-cpu','--all-targets','--','-D','warnings'],{})]
paths=['crates/onnx-runtime-ep-cpu/src/kernels/conv_ref.rs','crates/onnx-runtime-ep-cpu/tests/conv_im2col_allocation.rs'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();sources={p:sha(root/p) for p in paths};results=[]
for label,args,extra in commands:
 print('RUN',label,flush=True)
 with (out/(label+'.log')).open('w') as f:r=subprocess.run(['cargo',*args],cwd=root,env=env|extra,stdout=f,stderr=subprocess.STDOUT)
 s=(out/(label+'.log')).read_text();record={'name':label,'command':['cargo',*args],'extra_env':extra,'exit':r.returncode,'log_sha256':sha(out/(label+'.log'))};results.append(record)
 print(s[-4000:] if r.returncode else '\n'.join(l for l in s.splitlines() if 'test result:' in l or 'Finished' in l or 'col_allocations=' in l),flush=True)
 if label=='list' and r.returncode==0:assert 'kernels::conv::tests::gemm_driver_matches_exact_scalar_values_across_groups_batches_and_empty_axes: test' in s
 if label!='list' and 'test' in args and r.returncode==0:
  counts=re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;',s);assert counts and all(int(p)>0 and int(f)==0 for p,f,i in counts);record['counts']=counts
 (out/'validation.json').write_text(json.dumps({'source_sha256':sources,'results':results},indent=2))
 if r.returncode:sys.exit(r.returncode)
assert sources=={p:sha(root/p) for p in paths}
