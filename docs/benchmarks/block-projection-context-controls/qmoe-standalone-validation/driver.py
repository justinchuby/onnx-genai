from pathlib import Path
import os,subprocess,json,re,hashlib,sys
root=Path('/tmp/nxrt-qmoe-workspace');out=Path('/tmp/nxrt-qmoe-ready-standalone-validation');out.mkdir(exist_ok=True)
ort='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt'
env=os.environ|{'ORT_ROOT':ort,'ONNX_GENAI_ORT_LIB':ort+'/lib/libonnxruntime.1.32.0.dylib','NXRT_ORT_LIB_DIR':ort+'/lib','NXRT_REQUIRE_ORT_TESTS':'1','CARGO_INCREMENTAL':'0','CARGO_BUILD_JOBS':'1','CARGO_PROFILE_DEV_DEBUG':'0','CARGO_PROFILE_TEST_DEBUG':'0','CARGO_TARGET_DIR':'/tmp/nxrt-qmoe-main-target'}
env['PATH']='/opt/homebrew/Caskroom/miniconda/base/bin:'+env['PATH']
source_path='crates/onnx-runtime-ep-cpu/src/kernels/qmoe.rs';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();source={source_path:sha(root/source_path)}
expected=json.loads(Path('/tmp/nxrt-block-context-cache-validation/validation.json').read_text())['source_sha256'][source_path];assert source[source_path]==expected
commands=[
 ('list',['test','--offline','-p','onnx-runtime-ep-cpu','--lib','qmoe','--','--list']),
 ('qmoe',['test','--offline','-p','onnx-runtime-ep-cpu','--lib','qmoe']),
 ('qmoe-mlas',['test','--offline','-p','onnx-runtime-ep-cpu','--lib','qmoe','--features','mlas']),
 ('session',['test','--offline','-p','onnx-runtime-session','--lib']),
 ('plugin-cache',['test','--offline','-p','onnx-runtime-ep-plugin','--lib','workspace_plan_cache_tests']),
 ('host-oracle',['test','--offline','-p','onnx-runtime-ep-cuda','--test','qmoe_cpu_oracle']),
 ('clippy',['clippy','--offline','-p','onnx-runtime-ep-cpu','-p','onnx-runtime-ep-cpu-plugin','-p','onnx-runtime-session','--all-targets','--','-D','warnings']),
 ('clippy-mlas',['clippy','--offline','-p','onnx-runtime-ep-cpu','-p','onnx-runtime-ep-cpu-plugin','-p','onnx-runtime-session','--all-targets','--features','mlas','--','-D','warnings']),
];results=[]
for label,args in commands:
 print('RUN',label,flush=True);command=['cargo',*args]
 with (out/(label+'.log')).open('w') as f:r=subprocess.run(command,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
 text=(out/(label+'.log')).read_text();record={'name':label,'command':command,'exit':r.returncode,'log_sha256':sha(out/(label+'.log'))};results.append(record)
 print(text[-5000:] if r.returncode else '\n'.join(l for l in text.splitlines() if 'test result:' in l or 'Finished' in l),flush=True)
 if r.returncode==0:
  if '--list' in args:assert 'kernels::qmoe::tests::decoder_workspace_keeps_one_owner_when_execution_pool_changes: test' in text
  elif 'test' in args:
   counts=re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;',text);assert counts and all(int(p)>0 and int(f)==0 for p,f,i in counts);record['counts']=counts
   if label in ['qmoe','qmoe-mlas']:assert 'test kernels::qmoe::tests::decoder_workspace_keeps_one_owner_when_execution_pool_changes ... ok' in text
 (out/'validation.json').write_text(json.dumps({'source_sha256':source,'results':results},indent=2))
 if r.returncode:sys.exit(r.returncode)
assert source[source_path]==sha(root/source_path)
print('Standalone QMoE source matched experimental validation; all isolated checks passed.',flush=True)
