import os,pathlib,re,subprocess,sys,json,hashlib
root=pathlib.Path('/tmp/nxrt-qmoe-workspace');out=pathlib.Path('/tmp/nxrt-block-context-oracle-validation');out.mkdir(exist_ok=True)
ort='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt'
env=os.environ|{'ORT_ROOT':ort,'ONNX_GENAI_ORT_LIB':ort+'/lib/libonnxruntime.1.32.0.dylib','NXRT_ORT_LIB_DIR':ort+'/lib','NXRT_REQUIRE_ORT_TESTS':'1','CARGO_INCREMENTAL':'0','CARGO_BUILD_JOBS':'1','CARGO_PROFILE_DEV_DEBUG':'0','CARGO_PROFILE_TEST_DEBUG':'0','CARGO_TARGET_DIR':'/tmp/nxrt-qmoe-main-target'}
env['PATH']='/opt/homebrew/Caskroom/miniconda/base/bin:'+env['PATH']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checked=json.loads(pathlib.Path('/tmp/nxrt-block-context-validation/validation.json').read_text())['source_sha256']
assert all(sha(root/p)==h for p,h in checked.items())
paths=['crates/onnx-runtime-ep-cuda/tests/'+p for p in ['block_quantized_cpu_oracle.rs','block_quantized_matmul_gpu.rs','block_quantized_moe_gpu.rs']]
source={p:sha(root/p) for p in paths};results=[]
commands=[
 ('source-guard',['python3','.github/scripts/verify_cuda_test_honesty.py','--source-scan']),
 ('host-list',['cargo','test','--offline','-p','onnx-runtime-ep-cuda','--test','block_quantized_cpu_oracle','--','--list']),
 ('host',['cargo','test','--offline','-p','onnx-runtime-ep-cuda','--test','block_quantized_cpu_oracle']),
 ('gpu-fixtures-list',['cargo','test','--offline','-p','onnx-runtime-ep-cuda','--test','block_quantized_matmul_gpu','--test','block_quantized_moe_gpu','--features','gpu-tests','--','--list']),
 ('clippy-fixtures',['cargo','clippy','--offline','-p','onnx-runtime-ep-cuda','--test','block_quantized_cpu_oracle','--test','block_quantized_matmul_gpu','--test','block_quantized_moe_gpu','--','-D','warnings']),
]
for name,c in commands:
 print('RUN',name,flush=True)
 with (out/(name+'.log')).open('w') as f:r=subprocess.run(c,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
 s=(out/(name+'.log')).read_text();print(s[-5500:] if r.returncode else '\n'.join(l for l in s.splitlines() if 'test result:' in l or 'Finished' in l or l.endswith(': test') or 'scan' in l.lower()),flush=True)
 record={'name':name,'command':c,'exit':r.returncode,'log_sha256':sha(out/(name+'.log'))};results.append(record)
 if r.returncode==0:
  if name=='host-list':assert s.count(': test')==2
  if name=='host':
   counts=re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;',s);assert counts==[('2','0','0')];record['counts']=counts
  if name=='gpu-fixtures-list':assert s.count(': test')>0 and 'block_quantized_gemv_random_supported_formats_match_cpu: test' in s and 'block_quantized_moe_matches_cpu_across_activations: test' in s;record['listed_tests']=s.count(': test');record['gpu_execution']=False
 (out/'validation.json').write_text(json.dumps({'source_sha256':source,'runtime_source_sha256':checked,'results':results},indent=2))
 if r.returncode:sys.exit(r.returncode)
assert source=={p:sha(root/p) for p in paths} and all(sha(root/p)==h for p,h in checked.items())
print('No GPU execution; both real host oracles and fixture compilation validated.',flush=True)
