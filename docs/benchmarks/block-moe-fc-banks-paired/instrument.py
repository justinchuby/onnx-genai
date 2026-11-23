from pathlib import Path
import subprocess,hashlib,json
r=Path('/tmp/nxrt-qmoe-workspace'); k=r/'crates/onnx-runtime-ep-cpu/src/kernels'
proof={}
for name in ['moe','block_quantized_moe']:
 raw=subprocess.check_output(['git','show',f'509144567:crates/onnx-runtime-ep-cpu/src/kernels/{name}.rs'],cwd=r)
 prefix=raw.split(b'#[cfg(test)]\nmod tests {')[0]
 # Preserve the old Block driver and its old Float helper implementation together.
 if name=='block_quantized_moe':prefix=prefix.replace(b'super::moe::',b'super::probe_before_moe::')
 (k/f'probe_before_{name}.rs').write_bytes(prefix)
 proof[name]={'original_sha256':hashlib.sha256(raw).hexdigest(),'copied_sha256':hashlib.sha256(prefix).hexdigest(),'only_import_redirect':name=='block_quantized_moe','tests_module_omitted':len(prefix)<len(raw)}
p=r/'crates/onnx-runtime-ep-cpu/Cargo.toml';s=p.read_text();s=s.replace('gqa_phase_profile = []','gqa_phase_profile = []\n# Isolated same-process performance attribution experiment; never enabled by default.\nexpert_bank_pair_probe = []');p.write_text(s)
p=k/'mod.rs';s=p.read_text();s+='\n#[cfg(feature = "expert_bank_pair_probe")]\nmod probe_before_moe;\n#[cfg(feature = "expert_bank_pair_probe")]\nmod probe_before_block_quantized_moe;\n#[cfg(feature = "expert_bank_pair_probe")]\npub(crate) mod expert_bank_probe;\n';p.write_text(s)
Path('/tmp/nxrt-bank-paired/source-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
