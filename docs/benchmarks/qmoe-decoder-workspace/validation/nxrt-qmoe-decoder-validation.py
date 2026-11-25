import hashlib,json,os,pathlib,re,subprocess,sys
ROOT=pathlib.Path('/tmp/nxrt-qmoe-workspace');OUT=pathlib.Path('/tmp/nxrt-qmoe-decoder-validation');OUT.mkdir(exist_ok=True)
ort='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt'
env=os.environ|{'ORT_ROOT':ort,'ONNX_GENAI_ORT_LIB':ort+'/lib/libonnxruntime.1.32.0.dylib','NXRT_ORT_LIB_DIR':ort+'/lib','NXRT_REQUIRE_ORT_TESTS':'1','CARGO_INCREMENTAL':'0','CARGO_BUILD_JOBS':'1','CARGO_PROFILE_DEV_DEBUG':'0','CARGO_PROFILE_TEST_DEBUG':'0','CARGO_TARGET_DIR':'/tmp/nxrt-qmoe-main-target'}
env['PATH']='/opt/homebrew/Caskroom/miniconda/base/bin:'+env['PATH']
paths=subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines()+['crates/onnx-runtime-session/src/executor/workspace_context_tests.rs']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();source={p:sha(ROOT/p) for p in paths}
commands=[('api',['test','--offline','-p','onnx-runtime-ep-api','--lib']),('native',['test','--offline','-p','onnx-runtime-ep-cpu','--lib']),('mlas',['test','--offline','-p','onnx-runtime-ep-cpu','--lib','--features','mlas']),('layout',['test','--offline','-p','onnx-runtime-ep-cpu','--test','decoded_projection_layout']),('session',['test','--offline','-p','onnx-runtime-session','--lib']),('plugin',['test','--offline','-p','onnx-runtime-ep-plugin','--lib']),('clippy',['clippy','--offline','-p','onnx-runtime-ep-api','-p','onnx-runtime-ep-cpu','-p','onnx-runtime-ep-plugin','-p','onnx-runtime-ep-cpu-plugin','-p','onnx-runtime-session','--all-targets','--','-D','warnings']),('clippy-mlas',['clippy','--offline','-p','onnx-runtime-ep-cpu','-p','onnx-runtime-ep-cpu-plugin','-p','onnx-runtime-session','--all-targets','--features','mlas','--','-D','warnings'])]
initial=json.loads((OUT/'initial-validation.json').read_text())
assert all(source[p]==h for p,h in initial['source_sha256'].items() if p not in ['crates/onnx-runtime-ep-plugin/src/compute.rs','crates/onnx-runtime-ep-cpu/src/kernels/qmoe.rs'])
plugin=(ROOT/'crates/onnx-runtime-ep-plugin/src/compute.rs').read_text().split('#[cfg(test)]\nmod tests')[0]
assert hashlib.sha256(plugin.encode()).hexdigest()==json.loads((OUT/'plugin-runtime.json').read_text())['sha256']
qmoe=(ROOT/'crates/onnx-runtime-ep-cpu/src/kernels/qmoe.rs').read_text().split('#[cfg(test)]\nmod tests')[0]
assert hashlib.sha256(qmoe.encode()).hexdigest()==json.loads((OUT/'qmoe-runtime.json').read_text())['sha256']
commands=[('final-qmoe',['test','--offline','-p','onnx-runtime-ep-cpu','--lib','kernels::qmoe::tests::decoder_workspace_keeps_one_owner_when_execution_pool_changes','--','--exact'])]+commands[6:]
second=json.loads((OUT/'second-validation.json').read_text())
results=[r for r in second['results'] if r['exit']==0]
for name,args in commands:
 print('RUN',name,flush=True)
 with (OUT/(name+'.log')).open('w') as log:r=subprocess.run(['cargo',*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
 t=(OUT/(name+'.log')).read_text();result={'name':name,'command':['cargo',*args],'exit':r.returncode,'log_sha256':sha(OUT/(name+'.log'))}
 print(t[-6000:] if r.returncode else '\n'.join(l for l in t.splitlines() if 'test result:' in l or 'Finished' in l),flush=True)
 results.append(result);(OUT/'validation.json').write_text(json.dumps({'source_sha256':source,'results':results},indent=2))
 if r.returncode:sys.exit(r.returncode)
 if 'test' in args:
  counts=re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;',t)
  assert counts and all(int(p)>0 and int(f)==0 for p,f,i in counts),(name,t[-5000:])
  result['counts']=counts
  if name in ['native','mlas']:
   for identity in ['decoder_workspace_keeps_one_owner_when_execution_pool_changes','explicit_decoder_plan_matches_dynamic_across_selection_and_execution_pools']:
    assert 'test kernels::qmoe::tests::'+identity+' ... ok' in t
assert source=={p:sha(ROOT/p) for p in paths}
(OUT/'validation.json').write_text(json.dumps({'source_sha256':source,'results':results},indent=2))
print('All source checks passed; source hashes unchanged.',flush=True)
