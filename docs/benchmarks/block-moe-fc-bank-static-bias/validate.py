import os,subprocess
from pathlib import Path
os.chdir('/tmp/nxrt-qmoe-workspace');out=Path('/tmp/nxrt-bank-static-bias');env=dict(os.environ,CARGO_TARGET_DIR='/tmp/nxrt-erf-nan-target',CARGO_INCREMENTAL='0',ORT_ROOT='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt')
for mode,features in [('native',[]),('mlas',['--features','mlas'])]:
 for module in ['block_quantized_moe','moe','block_quantized_matmul']:
  for kind,tail in [('list',['--','--list']),('tests',[])]:
   name=f'{mode}-{module}-{kind}';print(name,flush=True)
   with (out/f'{name}.log').open('w') as log:r=subprocess.run(['cargo','--offline','test','-p','onnx-runtime-ep-cpu',*features,'--lib',f'kernels::{module}::tests::',*tail],env=env,stdout=log,stderr=subprocess.STDOUT)
   s=(out/f'{name}.log').read_text()
   if r.returncode:print(s[-6500:],flush=True);raise SystemExit(r.returncode)
   if kind=='list':assert any(l.endswith(': test') for l in s.splitlines())
   elif module=='block_quantized_moe':assert 'test kernels::block_quantized_moe::tests::expert_banks_cover_fc_phases_and_accumulator_without_warm_issuer_calls ... ok' in s
   print('\n'.join(l for l in s.splitlines() if 'test result:' in l),flush=True)
 print(mode,'clippy',flush=True)
 with (out/f'{mode}-clippy.log').open('w') as log:r=subprocess.run(['cargo','--offline','clippy','-p','onnx-runtime-ep-cpu',*features,'--all-targets','--','-D','warnings'],env=env,stdout=log,stderr=subprocess.STDOUT)
 if r.returncode:print((out/f'{mode}-clippy.log').read_text()[-5000:],flush=True);raise SystemExit(r.returncode)

subprocess.run(["cargo","fmt","--all","--","--check"],check=True)
subprocess.run(["git","diff","--check"],check=True)
