import os,subprocess
from pathlib import Path
os.chdir('/tmp/nxrt-qmoe-workspace');out=Path('/tmp/nxrt-block-fc-workspace');env=dict(os.environ,CARGO_TARGET_DIR='/tmp/nxrt-erf-nan-target',CARGO_INCREMENTAL='0',ORT_ROOT='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt')
for module in ['block_quantized_moe','moe']:
 for kind,tail in [('list',['--','--list']),('tests',[])]:
  tag=f'{module}-{kind}';print(tag,flush=True)
  with (out/f'{tag}.log').open('w') as log:r=subprocess.run(['cargo','--offline','test','--release','-p','onnx-runtime-ep-cpu','--lib',f'kernels::{module}::tests::',*tail],env=env,stdout=log,stderr=subprocess.STDOUT)
  s=(out/f'{tag}.log').read_text()
  if r.returncode:print(s[-6500:],flush=True);raise SystemExit(r.returncode)
  if kind=='list':
   names=[l for l in s.splitlines() if l.endswith(': test')];assert names;print('selected',len(names),flush=True)
   if module=='block_quantized_moe':assert any('expert_banks_cover_fc_phases_and_accumulator_without_warm_issuer_calls' in l for l in names)
  else:print('\n'.join(l for l in s.splitlines() if 'test result:' in l),flush=True)
