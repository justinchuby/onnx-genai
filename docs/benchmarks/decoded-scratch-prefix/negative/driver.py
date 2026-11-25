from pathlib import Path
import subprocess,json,hashlib
root=Path('/tmp/nxrt-qmoe-workspace');p=root/'crates/onnx-runtime-ep-cpu/src/kernels/decoded_projection.rs';original=p.read_bytes();s=original.decode();out=Path('/tmp/nxrt-scratch-prefix-negative');out.mkdir(exist_ok=True)
a='Ok(scratch.initialized_prefix(elements)?.split_at_mut(columns))';b='let values = scratch.initialized_prefix(available)?; Ok(values[..elements].split_at_mut(columns))';assert s.count(a)==1;mutant=s.replace(a,b)
ns={};exec(Path('/tmp/nxrt-prefetch-groups-check.py').read_text().split('commands=')[0],ns)
cmd=['cargo','test','--offline','-p','onnx-runtime-ep-cpu','--lib','kernels::qmoe::tests::decoder_workspace_keeps_one_owner_when_execution_pool_changes','--','--exact','--nocapture']
sha=lambda b:hashlib.sha256(b).hexdigest()
try:
 p.write_text(mutant);(out/'original.rs').write_bytes(original);(out/'mutant.rs').write_text(mutant)
 with (out/'negative.log').open('w') as f:r=subprocess.run(cmd,cwd=root,env=ns['env'],stdout=f,stderr=subprocess.STDOUT)
 log=(out/'negative.log').read_bytes();receipt={'command':cmd,'exit':r.returncode,'original_sha256':sha(original),'mutant_sha256':sha(mutant.encode()),'log_sha256':sha(log),'selected_test_failure':r.returncode!=0 and b'0 passed; 1 failed' in log and b'error[E' not in log and b'only actual grouped projections may initialize decoder prefixes' in log}
 (out/'receipt.json').write_text(json.dumps(receipt,indent=2));print(log.decode()[-2400:],flush=True);assert receipt['selected_test_failure']
finally:
 p.write_bytes(original);assert p.read_bytes()==original
 (out/'restoration.json').write_text(json.dumps({'restored_sha256':sha(p.read_bytes())},indent=2))
