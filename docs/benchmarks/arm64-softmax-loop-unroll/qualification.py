import os,pathlib,subprocess,json
root=pathlib.Path('/tmp/nxrt-qmoe-workspace');out=pathlib.Path('/tmp/nxrt-neon-unroll-process')
ort='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt'
env=os.environ|{'CARGO_TARGET_DIR':'/tmp/nxrt-erf-nan-target','CARGO_INCREMENTAL':'0','CARGO_BUILD_JOBS':'1','ORT_ROOT':ort,'ONNX_GENAI_ORT_LIB':ort+'/lib/libonnxruntime.1.32.0.dylib','NXRT_ORT_LIB_DIR':ort+'/lib','NXRT_REQUIRE_ORT_TESTS':'1'}
def run(cmd,name,ids=()):
 print('RUN',name,flush=True)
 with (out/name).open('w') as log:r=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
 text=(out/name).read_text()
 if r.returncode:print(text[-6000:],flush=True);raise SystemExit(r.returncode)
 for identity in ids:assert f'test {identity} ... ok' in text,(name,identity)
 print('PASS',name,text[-350:],flush=True)
identity='kernels::softmax::softmax_neon::unroll_tests::unrolled_rows_preserve_original_bits_and_inplace_tail_bounds'
print('RUN optimized fixture rebuild',flush=True)
with (out/'validation-build.jsonl').open('w') as log,(out/'validation-build.log').open('w') as err:
    subprocess.run(['cargo','test','--offline','-p','onnx-runtime-ep-cpu','--release','--features','mlas','--lib','--no-run','--message-format=json-render-diagnostics'],cwd=root,env=env,stdout=log,stderr=err,check=True)
records=[json.loads(line) for line in (out/'validation-build.jsonl').read_text().splitlines() if line.startswith('{')]
artifacts=[r for r in records if r.get('reason')=='compiler-artifact' and r.get('target',{}).get('name')=='onnx_runtime_ep_cpu' and r.get('executable') and r['profile']['test']]
assert len(artifacts)==1
binary=artifacts[0]['executable']
run([binary,'kernels::moe::tests::gated_row_bias_states_preserve_layouts_and_signed_zero','--exact','--nocapture'],'moe-oracle-exact.log',['kernels::moe::tests::gated_row_bias_states_preserve_layouts_and_signed_zero'])
run(['cargo','test','--offline','-p','onnx-runtime-ep-cpu','--lib'],'native-full.log',[identity])
run([binary],'mlas-full.log',[identity])
run(['cargo','test','--offline','-p','onnx-runtime-ep-cpu','--release','--features','mlas','--test','native_vs_mlas_differential','softmax'],'differential.log',['softmax_rows_native_and_mlas_agree','softmax_stays_normalized_on_extreme_rows'])
run(['cargo','clippy','--offline','-p','onnx-runtime-ep-cpu','-p','onnx-runtime-ep-cpu-plugin','--all-targets','--','-D','warnings'],'native-clippy.log')
run(['cargo','clippy','--offline','-p','onnx-runtime-ep-cpu','-p','onnx-runtime-ep-cpu-plugin','--all-targets','--features','mlas','--','-D','warnings'],'mlas-clippy.log')
