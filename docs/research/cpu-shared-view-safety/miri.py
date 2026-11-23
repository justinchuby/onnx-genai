import pathlib,subprocess,os,hashlib,json,sys
root=pathlib.Path('/tmp/nxrt-cpu-float-unaligned');os.chdir(root)
out=pathlib.Path('/tmp/nxrt-shared-view-validation');out.mkdir(exist_ok=True)
env=dict(os.environ,CARGO_TARGET_DIR='/tmp/nxrt-miri-shared-view-target',CARGO_INCREMENTAL='0',ORT_ROOT='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt');env.pop('MIRIFLAGS',None)
path=root/'crates/onnx-runtime-ep-cpu/src/kernels/mod.rs';candidate=path.read_bytes();text=candidate.decode();marker='    #[test]\n    fn dense_roundtrip_contiguous()';start=text.index('    #[test]\n    fn shared_numeric_accessors_accept_unaligned_origins_and_strides()');end=text.index(marker,start)
base=subprocess.check_output(['git','show','HEAD:crates/onnx-runtime-ep-cpu/src/kernels/mod.rs']).decode();assert marker in base
red=base.replace(marker,text[start:end]+marker);assert red.encode()!=candidate
receipt={'base':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'candidate_mod_sha256':hashlib.sha256(candidate).hexdigest(),'miri_flags':'default; inherited MIRIFLAGS removed','tests':[]}
def run(label,name):
 with (out/f'{label}.log').open('w') as log:r=subprocess.run(['cargo','+nightly','miri','test','--offline','-p','onnx-runtime-ep-cpu','--lib',name,'--','--exact'],env=env,stdout=log,stderr=subprocess.STDOUT)
 return r,(out/f'{label}.log').read_text()
name='kernels::tests::shared_numeric_accessors_accept_unaligned_origins_and_strides'
try:
 path.write_text(red);r,data=run('miri-before',name)
 assert r.returncode!=0 and 'Undefined Behavior' in data and 'alignment' in data and name in data,data[-4000:]
 receipt['before']='Expected undefined behavior detected in the named regression test';print(receipt['before'],flush=True)
finally:
 path.write_bytes(candidate);assert path.read_bytes()==candidate
for label,name in [
 ('miri-shared-numeric','kernels::tests::shared_numeric_accessors_accept_unaligned_origins_and_strides'),
 ('miri-direct-borrow','kernels::tests::direct_f32_borrow_checks_alignment_but_accepts_empty_byte_storage'),
 ('miri-arg-unaligned','kernels::selection::tests::arg_reductions_accept_unaligned_contiguous_f32_logits'),
 ('miri-arg-aligned','kernels::selection::tests::argmax_contiguous_f32_borrows_logits')]:
 r,data=run(label,name);assert r.returncode==0 and f'test {name} ... ok' in data,data[-5000:]
 receipt['tests'].append({'name':name,'passed':True});print(label+' passed',flush=True)
receipt['candidate_restored']=path.read_bytes()==candidate
(out/'miri-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
