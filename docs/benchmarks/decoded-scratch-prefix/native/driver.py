import os,pathlib,re,subprocess,sys,json,hashlib
root=pathlib.Path('/tmp/nxrt-qmoe-workspace');out=pathlib.Path('/tmp/nxrt-scratch-prefix-validation');out.mkdir(exist_ok=True)
ort='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt'
env=os.environ|{'ORT_ROOT':ort,'ONNX_GENAI_ORT_LIB':ort+'/lib/libonnxruntime.1.32.0.dylib','NXRT_ORT_LIB_DIR':ort+'/lib','NXRT_REQUIRE_ORT_TESTS':'1','CARGO_INCREMENTAL':'0','CARGO_BUILD_JOBS':'1','CARGO_PROFILE_DEV_DEBUG':'0','CARGO_PROFILE_TEST_DEBUG':'0','CARGO_TARGET_DIR':'/tmp/nxrt-qmoe-main-target'}
env['PATH']='/opt/homebrew/Caskroom/miniconda/base/bin:'+env['PATH']
commands=[('list',['test','--offline','-p','onnx-runtime-ep-cpu','--lib','kernels::decoded_projection::tests::','--','--list']),('projection',['test','--offline','-p','onnx-runtime-ep-cpu','--lib','kernels::decoded_projection::tests::']),('block-matmul',['test','--offline','-p','onnx-runtime-ep-cpu','--lib','kernels::block_quantized_matmul::tests::']),('block-moe',['test','--offline','-p','onnx-runtime-ep-cpu','--lib','kernels::block_quantized_moe::tests::']),('qmoe',['test','--offline','-p','onnx-runtime-ep-cpu','--lib','kernels::qmoe::tests::']),('layout',['test','--offline','-p','onnx-runtime-ep-cpu','--test','decoded_projection_layout']),('oracle',['test','--offline','-p','onnx-runtime-ep-cuda','--test','qmoe_cpu_oracle']),('clippy',['clippy','--offline','-p','onnx-runtime-ep-cpu','--all-targets','--','-D','warnings'])]
paths=['crates/onnx-runtime-ep-cpu/src/kernels/'+p for p in ['decoded_projection.rs','block_quantized_matmul.rs','block_quantized_moe.rs','qmoe.rs']]+['crates/onnx-runtime-ep-cpu/tests/decoded_projection_layout.rs'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();source={p:sha(root/p) for p in paths};results=[]
for label,args in commands:
 print('RUN',label,flush=True)
 with (out/(label+'.log')).open('w') as f:r=subprocess.run(['cargo',*args],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
 s=(out/(label+'.log')).read_text();record={'name':label,'command':['cargo',*args],'exit':r.returncode,'log_sha256':sha(out/(label+'.log'))};results.append(record)
 print(s[-5000:] if r.returncode else '\n'.join(l for l in s.splitlines() if 'test result:' in l or 'Finished' in l or l.endswith(': test')),flush=True)
 if label=='list' and r.returncode==0:assert 'kernels::decoded_projection::tests::projection_scratch_is_lazy_once_per_dispatch_and_only_borrows_the_needed_prefix: test' in s
 if not label.startswith('list') and 'test' in args and not r.returncode:
  counts=re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;',s);assert counts and all(int(p)>0 and int(f)==0 for p,f,i in counts);record['counts']=counts
 (out/'validation.json').write_text(json.dumps({'source_sha256':source,'results':results},indent=2))
 if r.returncode:sys.exit(r.returncode)
assert source=={p:sha(root/p) for p in paths}
