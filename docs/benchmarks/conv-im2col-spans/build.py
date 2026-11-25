from pathlib import Path
import os,subprocess,json,hashlib,shutil
root=Path('/tmp/nxrt-qmoe-workspace');out=Path('/tmp/nxrt-im2col-perf');out.mkdir(exist_ok=True);sha=lambda b:hashlib.sha256(b).hexdigest()
mod=root/'crates/onnx-runtime-ep-cpu/src/kernels/mod.rs';prior=mod.read_bytes();example=root/'crates/onnx-runtime-ep-cpu/examples/nxrt_im2col_spans.rs';assert not example.exists()
probe=Path('/tmp/nxrt-conv-gap-probe.rs').read_text();begin=probe.index('    let factories: Vec<Box<dyn KernelFactory>> = {');end=probe.index('    let xs =',begin)
probe=probe[:begin]+'''    let factories: Vec<Box<dyn KernelFactory>> = vec![
        Box::new(onnx_runtime_ep_cpu::kernels::conv::ConvFactory),
        Box::new(onnx_runtime_ep_cpu::kernels::im2col_before::ConvFactory),
    ];
'''+probe[end:];probe=probe.replace('"native" => 0','"after" => 0').replace('"reference" => 1','"before" => 1').replace('native|reference','after|before').replace('use native or reference','use after or before').replace('factories.len() == 2\n    );','false\n    );')
probe=probe.replace('    for factory in &factories {','    let mut first_bits = None;\n    for factory in &factories {').replace('        first_ns.push(t.elapsed().as_nanos());','        first_ns.push(t.elapsed().as_nanos());\n        let bits = output.iter().map(|v| v.to_bits()).collect::<Vec<_>>();\n        if let Some(previous) = &first_bits { assert_eq!(&bits, previous, "before/after complete outputs must match bits"); } else { first_bits = Some(bits); }')
(out/'probe.rs').write_text(probe);shutil.copyfile('/tmp/nxrt-im2col-before.rs',out/'before.rs');shutil.copyfile(root/'crates/onnx-runtime-ep-cpu/src/kernels/conv_ref.rs',out/'after.rs')
ort='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt';env=os.environ|{'ORT_ROOT':ort,'ONNX_GENAI_ORT_LIB':ort+'/lib/libonnxruntime.1.32.0.dylib','NXRT_ORT_LIB_DIR':ort+'/lib','CARGO_INCREMENTAL':'0','CARGO_BUILD_JOBS':'1','CARGO_TARGET_DIR':'/tmp/nxrt-erf-nan-target','RAYON_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'};env['PATH']='/opt/homebrew/Caskroom/miniconda/base/bin:'+env['PATH']
for k in ['NXRT_CPU_GEMM_BACKEND','NXRT_CPU_DISPATCH_LEDGER','ONNX_GENAI_PROFILE_OPS']:env.pop(k,None)
cmd=['cargo','build','--offline','--release','-p','onnx-runtime-ep-cpu','--example','nxrt_im2col_spans','--message-format=json-render-diagnostics']
try:
 mod.write_bytes(prior+b'\n#[path = "/tmp/nxrt-im2col-perf/before.rs"]\npub mod im2col_before;\n');example.write_text(probe)
 with (out/'cargo.jsonl').open('w') as f,(out/'build.log').open('w') as e:r=subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=e)
 if r.returncode:print((out/'build.log').read_text()[-5000:]);raise SystemExit(r.returncode)
 records=[json.loads(x) for x in (out/'cargo.jsonl').read_text().splitlines() if x.startswith('{')];paths=[x['executable'] for x in records if x.get('reason')=='compiler-artifact' and x.get('target',{}).get('name')=='nxrt_im2col_spans' and x.get('executable')];assert len(paths)==1;shutil.copy2(paths[0],out/'probe')
 symbols=subprocess.check_output(['nm','-a',str(out/'probe')],text=True);(out/'symbols.log').write_text(symbols);assert '_main' in symbols and 'im2col_before' in symbols
 metadata={'base':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'command':cmd,'compiler':subprocess.check_output(['rustc','-Vv'],text=True),'source_sha256':{p:sha((out/p).read_bytes()) for p in ['before.rs','after.rs','probe.rs']},'binary_sha256':sha((out/'probe').read_bytes()),'module_before_sha256':sha(prior),'temporary_registration':'pub module compiles frozen pre-change source alongside production after source; default backend for both','environment_overrides':{k:env[k] for k in ['RAYON_NUM_THREADS','VECLIB_MAXIMUM_THREADS','CARGO_TARGET_DIR']}}
finally:
 mod.write_bytes(prior)
 if example.exists():example.unlink()
assert mod.read_bytes()==prior and not example.exists();metadata['temporary_module_and_example_removed']=True;(out/'build-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');print('release before/after probe built and temporary registration restored',flush=True)
