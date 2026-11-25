from pathlib import Path
import hashlib,json,subprocess
root=Path('/tmp/nxrt-qmoe-workspace');out=Path('/tmp/nxrt-conv-pair-negative');out.mkdir(exist_ok=True)
ns={};exec(Path('/tmp/nxrt-conv-pair-check.py').read_text().split('commands=')[0],ns);env=ns['env']|{'NXRT_CPU_GEMM_BACKEND':'mlas'}
paths=['crates/onnx-runtime-ep-cpu/'+p for p in ['src/backend_ab.rs','src/kernels/conv_ref.rs','src/kernels/mod.rs','tests/conv_same_binary.rs']];original={p:(root/p).read_bytes() for p in paths};sha=lambda b:hashlib.sha256(b).hexdigest();records=[]
variants=[('alias-reference',paths[0],'native_generic_gemm: Box::new(crate::kernels::conv_native::ConvFactory),','native_generic_gemm: Box::new(crate::kernels::conv::ConvFactory),','native and reference must be independent concrete kernels'),('ambient-mlas',paths[1],'let backend = crate::backend::CpuBackend::Generic;','let backend = crate::backend::CpuBackend::auto_detect();','native Conv must not route GEMM to MLAS')]
cmd=['cargo','test','--offline','-p','onnx-runtime-ep-cpu','--features','mlas','--test','conv_same_binary','linked::linked_conv_factories_are_independent_and_match_f64_on_real_execute','--','--exact','--nocapture']
try:
 for name,p,a,b,message in variants:
  for path,data in original.items():(root/path).write_bytes(data)
  s=original[p].decode();assert s.count(a)==1;mutant=s.replace(a,b);(root/p).write_text(mutant);(out/(name+'.rs')).write_text(mutant)
  with (out/(name+'.log')).open('w') as f:r=subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
  log=(out/(name+'.log')).read_bytes();record={'name':name,'source_path':p,'original_sha256':sha(original[p]),'mutant_sha256':sha(mutant.encode()),'command':cmd,'extra_env':{'NXRT_CPU_GEMM_BACKEND':'mlas'},'exit':r.returncode,'log_sha256':sha(log),'expected_failure':r.returncode!=0 and b'0 passed; 1 failed' in log and b'error[E' not in log and message.encode() in log};records.append(record);(out/'controls.json').write_text(json.dumps(records,indent=2));print(name,log.decode()[-2000:],flush=True);assert record['expected_failure']
finally:
 for p,data in original.items():(root/p).write_bytes(data)
 restored={p:sha((root/p).read_bytes()) for p in paths};assert restored=={p:sha(data) for p,data in original.items()}
 (out/'restoration.json').write_text(json.dumps(restored,indent=2))
