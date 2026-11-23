from pathlib import Path
import subprocess
r=Path('/tmp/nxrt-qmoe-workspace');k=r/'crates/onnx-runtime-ep-cpu/src/kernels'
subprocess.run(['git','apply','/tmp/nxrt-bank-alignment/instrumentation-base.patch'],cwd=r,check=True)
for name in ['moe','block_quantized_moe']:
 raw=subprocess.check_output(['git','show',f'71b105629:crates/onnx-runtime-ep-cpu/src/kernels/{name}.rs'],cwd=r).split(b'#[cfg(test)]\nmod tests {')[0]
 if name=='block_quantized_moe':raw=raw.replace(b'super::moe::',b'super::probe_original_banks_moe::')
 (k/f'probe_original_banks_{name}.rs').write_bytes(raw)
p=k/'mod.rs';p.write_text(p.read_text()+'\n#[cfg(feature="expert_bank_pair_probe")]\nmod probe_original_banks_moe;\n#[cfg(feature="expert_bank_pair_probe")]\nmod probe_original_banks_block_quantized_moe;\n')
p=k/'expert_bank_probe.rs';s=p.read_text().replace('    Before,','    Before,\n    OriginalBanks,');s=s.replace('        ExpertBankProbeArm::Banked => (','        ExpertBankProbeArm::OriginalBanks => (\n            Box::new(CountedFactory { factory: super::probe_original_banks_block_quantized_moe::BlockQuantizedMoEFactory::new(weights, limit), creates: creates.clone() }),\n            Box::new(CountedFactory { factory: super::probe_original_banks_moe::MoEFactory, creates }),\n        ),\n        ExpertBankProbeArm::Banked => (');p.write_text(s)
p=Path('/tmp/nxrt-bank-static-bias/probe');(p/'src').mkdir(parents=True,exist_ok=True);(p/'Cargo.toml').write_text(Path('/tmp/nxrt-bank-paired/probe/Cargo.toml').read_text().replace('nxrt-bank-paired-probe','nxrt-bank-static-bias-probe'))
s=Path('/tmp/nxrt-bank-paired/probe/src/main.rs').read_text().replace('for j in 0..3 {','for j in 0..4 {').replace('(j+rotation)%3','(j+rotation)%4').replace('(j+batch+rotation)%3','(j+batch+rotation)%4').replace('for batch in 0..9 {','for batch in 0..8 {')
s=s.replace('[("before",ExpertBankProbeArm::Before),("banked",ExpertBankProbeArm::Banked),("null",ExpertBankProbeArm::Before)]','[("before",ExpertBankProbeArm::Before),("original",ExpertBankProbeArm::OriginalBanks),("static",ExpertBankProbeArm::Banked),("null",ExpertBankProbeArm::Before)]');s=s.replace('"before"=>0,"banked"=>1,_=>2','"before"=>0,"original"=>1,"static"=>2,_=>3');s=s.replace('assert_eq!(arms[0].requirement,arms[2].requirement);','assert_eq!(arms[0].requirement,arms[3].requirement);assert_eq!(arms[1].requirement,arms[2].requirement);');s=s.replace('arms[1].requirement-arms[0].requirement','arms[2].requirement-arms[0].requirement');(p/'src/main.rs').write_text(s)
