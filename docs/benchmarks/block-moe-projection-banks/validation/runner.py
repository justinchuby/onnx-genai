import hashlib,json,os,pathlib,re,subprocess,sys
ROOT=pathlib.Path('/tmp/nxrt-qmoe-workspace');OUT=pathlib.Path('/tmp/nxrt-block-banks-final-validation');OUT.mkdir(exist_ok=True)
ort='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt'
env=os.environ|{'ORT_ROOT':ort,'ONNX_GENAI_ORT_LIB':ort+'/lib/libonnxruntime.1.32.0.dylib','NXRT_ORT_LIB_DIR':ort+'/lib','NXRT_REQUIRE_ORT_TESTS':'1','CARGO_INCREMENTAL':'0','CARGO_BUILD_JOBS':'1','CARGO_PROFILE_DEV_DEBUG':'0','CARGO_PROFILE_TEST_DEBUG':'0','CARGO_TARGET_DIR':'/tmp/nxrt-qmoe-main-target'}
env['PATH']='/opt/homebrew/Caskroom/miniconda/base/bin:'+env['PATH']
paths=['crates/onnx-runtime-ep-cpu/src/kernels/moe.rs','crates/onnx-runtime-ep-cpu/src/kernels/block_quantized_moe.rs'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();source={p:sha(ROOT/p) for p in paths}
meta=json.loads(pathlib.Path('/tmp/nxrt-block-banks-inline-study/metadata.json').read_text());assert source==meta['source_sha256']
commands=[('native',['test','--offline','-p','onnx-runtime-ep-cpu','--lib']),('mlas',['test','--offline','-p','onnx-runtime-ep-cpu','--lib','--features','mlas']),('release-native',['test','--offline','-p','onnx-runtime-ep-cpu','--release','--lib']),('release-mlas',['test','--offline','-p','onnx-runtime-ep-cpu','--release','--lib','--features','mlas']),('layout',['test','--offline','-p','onnx-runtime-ep-cpu','--test','decoded_projection_layout']),('session',['test','--offline','-p','onnx-runtime-session','--lib']),('clippy',['clippy','--offline','-p','onnx-runtime-ep-cpu','-p','onnx-runtime-ep-cpu-plugin','-p','onnx-runtime-session','--all-targets','--','-D','warnings']),('clippy-mlas',['clippy','--offline','-p','onnx-runtime-ep-cpu','-p','onnx-runtime-ep-cpu-plugin','-p','onnx-runtime-session','--all-targets','--features','mlas','--','-D','warnings'])]
results=[]
for name,args in commands:
 print('RUN',name,flush=True);local=env|({'CARGO_TARGET_DIR':'/tmp/nxrt-erf-nan-target'} if name.startswith('release') else {})
 with (OUT/(name+'.log')).open('w') as log:r=subprocess.run(['cargo',*args],cwd=ROOT,env=local,stdout=log,stderr=subprocess.STDOUT)
 text=(OUT/(name+'.log')).read_text();result={'name':name,'command':['cargo',*args],'exit':r.returncode,'log_sha256':sha(OUT/(name+'.log'))}
 if 'test' in args:
  counts=re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;',text)
  assert counts and all(int(p)>0 and int(f)==0 for p,f,i in counts),(name,text[-5000:])
  result['counts']=counts
  if name in ['native','mlas','release-native','release-mlas']:
   for identity in ['kernels::block_quantized_moe::tests::streamed_fc_banks_match_owned_driver_bits_across_gates_zero_axes_and_dirty_reuse','kernels::block_quantized_moe::tests::streamed_workspace_covers_projection_banks_and_accumulator_without_warm_issuer_calls']:
    assert f'test {identity} ... ok' in text,(name,identity,text[-5000:])
 print(text[-5000:] if r.returncode else '\n'.join(l for l in text.splitlines() if 'test result:' in l or 'Finished' in l),flush=True)
 results.append(result);(OUT/'validation.json').write_text(json.dumps({'source_sha256':source,'results':results},indent=2))
 if r.returncode:sys.exit(r.returncode)
assert source=={p:sha(ROOT/p) for p in paths}
print('All final-source checks passed; measured source hashes unchanged.',flush=True)
