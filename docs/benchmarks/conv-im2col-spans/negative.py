from pathlib import Path
import os,json,subprocess,hashlib,re
root=Path('/tmp/nxrt-qmoe-workspace');out=Path('/tmp/nxrt-im2col-negative');out.mkdir(exist_ok=True);p=root/'crates/onnx-runtime-ep-cpu/src/kernels/conv_ref.rs';original=p.read_text();sha=lambda b:hashlib.sha256(b).hexdigest()
ns={};exec(Path('/tmp/nxrt-im2col-check.py').read_text().split('base=')[0],ns);env=ns['env'];cmd=['cargo','test','--offline','-p','onnx-runtime-ep-cpu','--lib','kernels::conv::tests::im2col_spans_preserve_every_sample_bit_and_overwrite_padding','--','--exact','--nocapture'];records=[]
start=original.index('    let (pt, pl)',original.index('fn im2col('));end=original.index('\n}\n',start)
mutants={'negative_zero_padding':original.replace('row[..begin].fill(0.0);','row[..begin].fill(-0.0);'),'vacuity_no_writes':original[:start]+'    let _ = (input, c, ih, iw, kh, kw, strides, dilations, pads, oh, ow, col);'+original[end:]}
try:
 for name,changed in mutants.items():
  assert changed!=original;p.write_text(changed)
  with (out/(name+'.log')).open('w') as f:r=subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
  s=(out/(name+'.log')).read_text();assert r.returncode!=0 and 'running 1 test' in s and '0 passed; 1 failed;' in s and 'assertion `left == right` failed' in s,(name,r.returncode,s[-3000:]);records.append({'name':name,'command':cmd,'exit':r.returncode,'selected':1,'passed':0,'failed':1,'mutant_source_sha256':sha(changed.encode()),'log_sha256':sha((out/(name+'.log')).read_bytes())});print('REJECTED',name,flush=True)
finally:p.write_text(original)
assert p.read_text()==original; (out/'negative.json').write_text(json.dumps({'source_restored_sha256':sha(p.read_bytes()),'results':records},indent=2)+'\n')
