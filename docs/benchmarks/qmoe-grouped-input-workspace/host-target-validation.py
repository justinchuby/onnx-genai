import os,pathlib,subprocess,json
os.environ.update(CARGO_TARGET_DIR='/tmp/nxrt-erf-nan-target',CARGO_INCREMENTAL='0',ORT_ROOT='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt')
for label,args in [('source-guard',['python','.github/scripts/verify_cuda_test_honesty.py','--source-scan']),('host-default',['cargo','test','--offline','-p','onnx-runtime-ep-cuda','--test','qmoe_cpu_oracle','--test','qmoe_gpu','--quiet']),('host-gpu-feature',['cargo','test','--offline','-p','onnx-runtime-ep-cuda','--test','qmoe_cpu_oracle','--features','gpu-tests','--quiet']),('host-clippy',['cargo','clippy','--offline','-p','onnx-runtime-ep-cuda','--all-targets','--','-D','warnings'])]:
 p=pathlib.Path('/tmp/nxrt-qmoe-'+label+'.log');print(json.dumps({'step':label,'log':str(p)}),flush=True)
 with p.open('w') as log:r=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
 print(p.read_text()[-1800:],flush=True)
 if r.returncode:raise SystemExit(r.returncode)
