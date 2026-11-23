import os,pathlib,subprocess,json
os.environ['PATH']='/opt/homebrew/Caskroom/miniconda/base/bin:'+os.environ['PATH']
os.environ.update(CARGO_TARGET_DIR='/tmp/nxrt-erf-nan-target',CARGO_INCREMENTAL='0',ORT_ROOT='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt')
for label,args in [('integration-qmoe', ['cargo', 'test', '--offline', '-p', 'onnx-runtime-ep-cpu', '--lib', 'qmoe', '--quiet']), ('integration-mlas-qmoe', ['cargo', 'test', '--offline', '-p', 'onnx-runtime-ep-cpu', '--features', 'mlas', '--lib', 'qmoe', '--quiet']), ('integration-session', ['cargo', 'test', '--offline', '-p', 'onnx-runtime-session', '--lib', '--quiet']), ('integration-mlas-session', ['cargo', 'test', '--offline', '-p', 'onnx-runtime-session', '--features', 'mlas', '--lib', '--quiet']), ('integration-cpu-oracle', ['cargo', 'test', '--offline', '-p', 'onnx-runtime-ep-cuda', '--test', 'qmoe_gpu', '--quiet']), ('integration-clippy', ['cargo', 'clippy', '--offline', '-p', 'onnx-runtime-ep-cpu', '-p', 'onnx-runtime-ep-cuda', '-p', 'onnx-runtime-session', '--all-targets', '--', '-D', 'warnings'])]:
 p=pathlib.Path('/tmp/nxrt-qmoe-workspace-'+label+'.log');print(json.dumps({'step':label,'log':str(p)}),flush=True)
 with p.open('w') as log:r=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
 print(p.read_text()[-2000:],flush=True)
 if r.returncode:raise SystemExit(r.returncode)
