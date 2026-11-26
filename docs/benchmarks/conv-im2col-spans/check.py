import os,pathlib,subprocess,json,re,hashlib,sys
root=pathlib.Path('/tmp/nxrt-qmoe-workspace');out=pathlib.Path('/tmp/nxrt-im2col-validation');out.mkdir(exist_ok=True)
ort='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt'
env=os.environ|{'ORT_ROOT':ort,'ONNX_GENAI_ORT_LIB':ort+'/lib/libonnxruntime.1.32.0.dylib','NXRT_ORT_LIB_DIR':ort+'/lib','NXRT_REQUIRE_ORT_TESTS':'1','CARGO_INCREMENTAL':'0','CARGO_BUILD_JOBS':'1','CARGO_PROFILE_DEV_DEBUG':'0','CARGO_PROFILE_TEST_DEBUG':'0','CARGO_TARGET_DIR':'/tmp/nxrt-qmoe-main-target','RAYON_NUM_THREADS':'1','ONNX_GENAI_MLAS_THREADPOOL_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'}
env['PATH']='/opt/homebrew/Caskroom/miniconda/base/bin:'+env['PATH']
for name in ['NXRT_CPU_GEMM_BACKEND','NXRT_CPU_DISPATCH_LEDGER','ONNX_GENAI_PROFILE_OPS']:env.pop(name,None)
base=['--offline','-p','onnx-runtime-ep-cpu'];test='im2col_spans_preserve_every_sample_bit_and_overwrite_padding'
commands=[('list-native',['test',*base,'--lib','kernels::conv::tests::','--','--list']),('conv-native',['test',*base,'--lib','kernels::conv::tests::']),('gemm-generic',['test',*base,'--lib','kernels::conv::tests::gemm_driver_matches_exact_scalar_values_across_groups_batches_and_empty_axes','--','--exact']),('allocation',['test',*base,'--test','conv_im2col_allocation']),('pair-default',['test',*base,'--test','conv_same_binary']),('list-linked-native',['test',*base,'--features','mlas','--lib','kernels::conv_native::tests::','--','--list']),('conv-linked-native',['test',*base,'--features','mlas','--lib','kernels::conv_native::tests::']),('conv-reference',['test',*base,'--features','mlas','--lib','kernels::conv::tests::']),('pair-linked',['test',*base,'--features','mlas','--test','conv_same_binary']),('clippy',['clippy',*base,'--all-targets','--','-D','warnings']),('clippy-linked',['clippy',*base,'--features','mlas','--all-targets','--','-D','warnings'])]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();path=root/'crates/onnx-runtime-ep-cpu/src/kernels/conv_ref.rs';source=sha(path);results=[]
for label,args in commands:
 print('RUN',label,flush=True);extra={'NXRT_CPU_GEMM_BACKEND':'generic'} if label=='gemm-generic' else {}
 with (out/(label+'.log')).open('w') as f:r=subprocess.run(['cargo',*args],cwd=root,env=env|extra,stdout=f,stderr=subprocess.STDOUT)
 text=(out/(label+'.log')).read_text();record={'name':label,'command':['cargo',*args],'environment_overrides':extra,'exit':r.returncode,'log_sha256':sha(out/(label+'.log'))};results.append(record)
 print(text[-4500:] if r.returncode else '\n'.join(x for x in text.splitlines() if 'test result:' in x or 'Finished' in x),flush=True)
 if r.returncode==0:
  if label.startswith('list'):assert test+': test' in text
  elif args[0]=='test':
   counts=re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;',text);assert counts and all(int(p)>0 and int(f)==0 for p,f,i in counts);record['counts']=counts
 (out/'validation.json').write_text(json.dumps({'source_sha256':source,'results':results},indent=2))
 if r.returncode:sys.exit(r.returncode)
assert source==sha(path)
