import os,subprocess,pathlib,sys
os.chdir('/tmp/nxrt-float-moe-accumulator')
env=dict(os.environ,CARGO_TARGET_DIR='/tmp/nxrt-erf-nan-target',CARGO_INCREMENTAL='0',ORT_ROOT='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt')
out=pathlib.Path('/tmp/nxrt-moe-accumulator-integration');out.mkdir(exist_ok=True)
commands=[('layout',['test','-p','onnx-runtime-ep-cpu','--test','decoded_projection_layout']),('native',['test','-p','onnx-runtime-ep-cpu','--lib']),('session',['test','-p','onnx-runtime-session','--lib']),('mlas',['test','-p','onnx-runtime-ep-cpu','--features','mlas','--lib']),('clippy',['clippy','-p','onnx-runtime-ep-cpu','-p','onnx-runtime-session','--all-targets','--','-D','warnings']),('mlas-clippy',['clippy','-p','onnx-runtime-ep-cpu','--features','mlas','--all-targets','--','-D','warnings'])]
for name,args in commands:
 print(name,flush=True)
 with (out/f'{name}.log').open('w') as log:r=subprocess.run(['cargo','--offline',*args],env=env,stdout=log,stderr=subprocess.STDOUT)
 text=(out/f'{name}.log').read_text();print(text[-4000:] if r.returncode else '\n'.join(l for l in text.splitlines() if 'test result:' in l or 'Finished' in l),flush=True)
 if r.returncode:sys.exit(r.returncode)
