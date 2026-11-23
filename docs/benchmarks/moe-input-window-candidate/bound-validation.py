import os,subprocess,pathlib,json
os.environ.update(CARGO_TARGET_DIR='/tmp/nxrt-erf-nan-target',CARGO_INCREMENTAL='0',ORT_ROOT='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt')
steps=[('native-bound',['cargo','test','--offline','-p','onnx-runtime-ep-cpu','--lib','dense_resident_experts_do_not_initialize_projection_scratch','--quiet']),('mlas-bound',['cargo','test','--offline','-p','onnx-runtime-ep-cpu','--features','mlas','--lib','dense_resident_experts_do_not_initialize_projection_scratch','--quiet']),('pool',['cargo','test','--offline','-p','onnx-runtime-memory-governor','--lib','warm_runs_reuse_peak_capacity_and_its_exact_original_grant','--quiet']),('carrier',['cargo','test','--offline','-p','onnx-runtime-ep-cpu','--test','workspace_owner_allocation','--quiet'])]
for label,args in steps:
 p=pathlib.Path('/tmp/nxrt-moe-window-'+label+'.log');print(json.dumps({'step':label}),flush=True)
 with p.open('w') as f:r=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
 output=p.read_text();print(output[-1500:],flush=True)
 if r.returncode:raise SystemExit(r.returncode)
 assert '1 passed' in output,label
