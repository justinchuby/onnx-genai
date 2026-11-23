import os, subprocess, pathlib, sys
root=pathlib.Path('/tmp/nxrt-float-moe-workspace'); os.chdir(root)
env=dict(os.environ,CARGO_TARGET_DIR='/tmp/nxrt-erf-nan-target',CARGO_INCREMENTAL='0',ORT_ROOT='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt')
out=pathlib.Path('/tmp/nxrt-float-moe-validation');out.mkdir(exist_ok=True)
commands=[('native-cpu',['test','-p','onnx-runtime-ep-cpu','--lib']),('native-session',['test','-p','onnx-runtime-session','--lib']),('mlas-moe',['test','-p','onnx-runtime-ep-cpu','--features','mlas','--lib','kernels::moe::tests']),('clippy',['clippy','-p','onnx-runtime-ep-cpu','-p','onnx-runtime-session','--all-targets','--','-D','warnings']),('mlas-clippy',['clippy','-p','onnx-runtime-ep-cpu','--features','mlas','--all-targets','--','-D','warnings'])]
for name,args in commands:
 print(name,flush=True)
 with (out/f'{name}.log').open('w') as log:r=subprocess.run(['cargo','--offline',*args],env=env,stdout=log,stderr=subprocess.STDOUT)
 text=(out/f'{name}.log').read_text();print(text[-3500:] if r.returncode else '\n'.join(line for line in text.splitlines() if 'test result:' in line or 'Finished' in line),flush=True)
 if r.returncode:sys.exit(r.returncode)
