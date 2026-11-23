from pathlib import Path
import subprocess
r=Path('/tmp/nxrt-qmoe-workspace');k=r/'crates/onnx-runtime-ep-cpu/src/kernels'
subprocess.run(['git','apply','/tmp/nxrt-bank-alignment/instrumentation-base.patch'],cwd=r,check=True)
for name in ['moe','block_quantized_moe']:
 raw=subprocess.check_output(['git','show',f'f0a2cd115:crates/onnx-runtime-ep-cpu/src/kernels/{name}.rs'],cwd=r).split(b'#[cfg(test)]\nmod tests {')[0]
 if name=='block_quantized_moe':raw=raw.replace(b'super::moe::',b'super::probe_original_banks_moe::')
 (k/f'probe_original_banks_{name}.rs').write_bytes(raw)
p=k/'mod.rs';p.write_text(p.read_text()+'\n#[cfg(feature="expert_bank_pair_probe")]\nmod probe_original_banks_moe;\n#[cfg(feature="expert_bank_pair_probe")]\nmod probe_original_banks_block_quantized_moe;\n')
p=k/'expert_bank_probe.rs';s=p.read_text().replace('    Before,','    Before,\n    OriginalBanks,');s=s.replace('        ExpertBankProbeArm::Banked => (','        ExpertBankProbeArm::OriginalBanks => (\n            Box::new(CountedFactory { factory: super::probe_original_banks_block_quantized_moe::BlockQuantizedMoEFactory::new(weights, limit), creates: creates.clone() }),\n            Box::new(CountedFactory { factory: super::probe_original_banks_moe::MoEFactory, creates }),\n        ),\n        ExpertBankProbeArm::Banked => (');p.write_text(s)
p=Path('/tmp/nxrt-bank-inplace-gate/probe');(p/'src').mkdir(parents=True,exist_ok=True);(p/'Cargo.toml').write_text(Path('/tmp/nxrt-bank-static-bias/probe/Cargo.toml').read_text().replace('nxrt-bank-static-bias-probe','nxrt-bank-inplace-gate-probe'))
s=Path('/tmp/nxrt-bank-static-bias/probe/src/main.rs').read_text().replace('"static",ExpertBankProbeArm::Banked','"compact",ExpertBankProbeArm::Banked').replace('"static"=>2','"compact"=>2')
s=s.replace('assert_eq!(arms[0].requirement,arms[3].requirement);assert_eq!(arms[1].requirement,arms[2].requirement);','''assert_eq!(arms[0].requirement,arms[3].requirement);
    let shrink = if gate!=0 { m*(k.max(n)+k.max(2*n)-k-2*n)*4 } else { 0 };
    assert_eq!(arms[1].requirement-arms[2].requirement,shrink as u64);''')
s=s.replace('let delta = if is_float {0}else{m*(widths-k)*4};\n    assert_eq!(arms[2].requirement-arms[0].requirement,delta as u64);','''if is_float {
        assert_eq!(arms[0].requirement-arms[2].requirement,shrink as u64);
    } else {
        let delta=m*(if gate!=0 {2*n}else{n})*4;
        assert_eq!(arms[2].requirement-arms[0].requirement,delta as u64);
    }''')
s=s.replace('    let widths = if gate!=0 {k.max(n)+k.max(2*n)}else{k+n};\n','');(p/'src/main.rs').write_text(s)
