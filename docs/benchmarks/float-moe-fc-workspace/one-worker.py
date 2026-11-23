import os,subprocess,pathlib
name='kernels::moe::tests::execution_workspace_bounds_cover_threshold_skew_and_pool_changes'
env=dict(os.environ,CARGO_TARGET_DIR='/tmp/nxrt-erf-nan-target',CARGO_INCREMENTAL='0',ORT_ROOT='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt',ONNX_GENAI_CPU_DECODE_THREADS='1')
result=subprocess.run(['cargo','test','--offline','-p','onnx-runtime-ep-cpu','--features','mlas','--lib',name,'--','--exact'],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
pathlib.Path('/tmp/nxrt-float-moe-fc-validation/mlas-one-worker.log').write_text(result.stdout)
print(result.stdout[-2500:]);assert result.returncode==0 and f'test {name} ... ok' in result.stdout and 'test result: ok. 1 passed; 0 failed' in result.stdout
