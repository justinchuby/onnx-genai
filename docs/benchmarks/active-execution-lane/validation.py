import os,subprocess,pathlib,sys
os.chdir('/tmp/nxrt-qmoe-workspace')
env=dict(os.environ,CARGO_TARGET_DIR='/tmp/nxrt-erf-nan-target',CARGO_INCREMENTAL='0',ORT_ROOT='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt');env['ONNX_GENAI_ORT_LIB']=env['ORT_ROOT']+'/lib/libonnxruntime.1.32.0.dylib'
out=pathlib.Path('/tmp/nxrt-active-scope-validation');out.mkdir(exist_ok=True)
for name,args in [('session',['test','-p','onnx-runtime-session','--lib']),('docs',['test','-p','onnx-runtime-session','--doc','ExecutionScope']),('clippy',['clippy','-p','onnx-runtime-session','--all-targets','--','-D','warnings']),('engine',['check','-p','onnx-genai-engine','--lib'])]:
 print(name,flush=True)
 with (out/f'{name}.log').open('w') as log:r=subprocess.run(['cargo','--offline',*args],env=env,stdout=log,stderr=subprocess.STDOUT)
 data=(out/f'{name}.log').read_text();print(data[-5000:] if r.returncode else '\n'.join(l for l in data.splitlines() if 'test result:' in l or 'Finished' in l),flush=True)
 if r.returncode:sys.exit(r.returncode)
