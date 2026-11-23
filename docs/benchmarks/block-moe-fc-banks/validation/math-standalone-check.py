from pathlib import Path
import os,subprocess
os.chdir('/tmp/nxrt-float-moe-workspace');out=Path('/tmp/nxrt-block-fc-workspace');env=dict(os.environ,CARGO_TARGET_DIR='/tmp/nxrt-erf-nan-target',CARGO_INCREMENTAL='0',ORT_ROOT='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt')
for name,args in [('standalone-math-list',['test','-p','onnx-runtime-ep-cpu','--lib','kernels::standard_math::tests::','--','--list']),('standalone-math-tests',['test','-p','onnx-runtime-ep-cpu','--lib','kernels::standard_math::tests::']),('standalone-math-clippy',['clippy','-p','onnx-runtime-ep-cpu','--all-targets','--','-D','warnings'])]:
 print(name,flush=True)
 with (out/f'{name}.log').open('w') as log:r=subprocess.run(['cargo','--offline',*args],env=env,stdout=log,stderr=subprocess.STDOUT)
 s=(out/f'{name}.log').read_text()
 if r.returncode:print(s[-4500:],flush=True);raise SystemExit(r.returncode)
 if name.endswith('list'):assert sum(l.endswith(': test') for l in s.splitlines())==2
 if name.endswith('tests'):assert 'test kernels::standard_math::tests::scan_preserves_small_prefix_and_handles_interleaved_lanes ... ok' in s and 'test kernels::standard_math::tests::gated_activation_rejects_broadcasting_before_writing ... ok' in s
 print('\n'.join(l for l in s.splitlines() if 'test result:' in l),flush=True)
