from pathlib import Path
import os,subprocess
os.chdir('/tmp/nxrt-qmoe-workspace');out=Path('/tmp/nxrt-block-fc-workspace');env=dict(os.environ,CARGO_TARGET_DIR='/tmp/nxrt-erf-nan-target',CARGO_INCREMENTAL='0',ORT_ROOT='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt')
for tag,tail in [('math-list',['--','--list']),('math-tests',[])]:
 with (out/f'{tag}.log').open('w') as log:r=subprocess.run(['cargo','--offline','test','--release','-p','onnx-runtime-ep-cpu','--lib','kernels::standard_math::tests::',*tail],env=env,stdout=log,stderr=subprocess.STDOUT)
 s=(out/f'{tag}.log').read_text()
 if r.returncode:print(s[-3500:],flush=True);raise SystemExit(r.returncode)
 if tag=='math-list':assert sum(l.endswith(': test') for l in s.splitlines())>0
 print('\n'.join(l for l in s.splitlines() if 'test result:' in l),flush=True)
