from pathlib import Path
import subprocess,re
p=Path('/tmp/nxrt-block-fc-workspace');root=Path('/tmp/nxrt-qmoe-workspace');file='crates/onnx-runtime-ep-cpu/src/kernels/standard_math.rs'
old=subprocess.check_output(['git','show','f13b60689:'+file],cwd=root,text=True)
match=re.search(r'\(\((0\.797[0-9_]*) as \$float\)',old);assert match
current=(root/file).read_text();value=re.search(r'const GELU_TANH_SCALE: f64 = ([0-9_.]+);',current).group(1)
source=f'''const SCALE: f64 = {value};
fn main() {{
 let old32 = {match.group(1)} as f32; let old64 = {match.group(1)} as f64;
 assert_eq!(old32.to_bits(), (SCALE as f32).to_bits());
 assert_eq!(old64.to_bits(), SCALE.to_bits());
 println!("coefficient f32_bits={{:#x}} f64_bits={{:#x}}", old32.to_bits(), old64.to_bits());
}}
'''
(p/'constant-bits.rs').write_text(source)
subprocess.run(['rustc','--edition=2024',str(p/'constant-bits.rs'),'-o',str(p/'constant-bits')],check=True)
result=subprocess.check_output([str(p/'constant-bits')],text=True);(p/'constant-bits.log').write_text(result);print(result,flush=True)
(p/'constant-bits').unlink()
