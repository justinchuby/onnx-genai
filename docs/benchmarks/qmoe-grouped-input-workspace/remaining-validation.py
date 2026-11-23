import os,pathlib,subprocess,json
os.environ['PATH']='/opt/homebrew/Caskroom/miniconda/base/bin:'+os.environ['PATH']
os.environ.update(CARGO_TARGET_DIR='/tmp/nxrt-erf-nan-target',CARGO_INCREMENTAL='0',ORT_ROOT='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt')
for label,args in [('qmoe-final', ['cargo', 'test', '--offline', '-p', 'onnx-runtime-ep-cpu', '--lib', 'qmoe', '--quiet']), ('cpu-oracle', ['cargo', 'test', '--offline', '-p', 'onnx-runtime-ep-cuda', '--test', 'qmoe_gpu', '--quiet']), ('session-integration-final', ['cargo', 'test', '--offline', '-p', 'onnx-runtime-session', '--tests', '--quiet']), ('clippy', ['cargo', 'clippy', '--offline', '-p', 'onnx-runtime-ep-cpu', '-p', 'onnx-runtime-ep-cuda', '-p', 'onnx-runtime-session', '--all-targets', '--', '-D', 'warnings']), ('mlas-clippy', ['cargo', 'clippy', '--offline', '-p', 'onnx-runtime-ep-cpu', '-p', 'onnx-runtime-session', '--all-targets', '--features', 'mlas', '--', '-D', 'warnings'])]:
 p=pathlib.Path('/tmp/nxrt-qmoe-workspace-'+label+'.log');print(json.dumps({'step':label,'log':str(p)}),flush=True)
 with p.open('w') as log:r=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
 print(p.read_text()[-2000:],flush=True)
 if r.returncode:raise SystemExit(r.returncode)
