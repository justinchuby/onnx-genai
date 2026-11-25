from pathlib import Path
import os,subprocess,json,hashlib,gzip,shutil
root=Path('/tmp/nxrt-qmoe-workspace');out=Path('/tmp/nxrt-block-context-shared-image');out.mkdir(exist_ok=True)
paths=['crates/onnx-runtime-ep-cpu/src/kernels/block_quantized_'+k+'.rs' for k in ['matmul','moe']]
original={p:(root/p).read_bytes() for p in paths};base='61c47ee27c19978a441443ed4e42995a7a172e15'
sha=lambda b:hashlib.sha256(b).hexdigest()
def section(source,marker):
 start=source.index(marker);a=source.index('{',start);depth=1;i=a+1
 # These blocks contain no braces in comments/string literals whose net depth
 # differs. Validate extraction by both boundary and subsequent compilation.
 while depth:
  depth+=(source[i]=='{')-(source[i]=='}');i+=1
 return source[start:i]
try:
 for kind,p in zip(['matmul','moe'],paths):
  old=subprocess.check_output(['git','show',base+':'+p],cwd=root).decode();s=original[p].decode()
  kernel='BlockQuantized'+('MatMul' if kind=='matmul' else 'MoE')+'Kernel';factory=kernel.replace('Kernel','Factory');frozen_kernel='Frozen'+kernel;frozen_factory='Frozen'+factory
  pieces=[section(old,'pub struct '+kernel+' {'),section(old,'pub struct '+factory+' {'),section(old,'impl KernelFactory for '+factory+' {'),section(old,'impl Kernel for '+kernel+' {'),section(old,'impl '+kernel+' {')]
  frozen='\n\n'.join(pieces).replace(kernel,frozen_kernel).replace(factory,frozen_factory)
  marker='impl KernelFactory for '+factory+' {';start=s.index(marker);a=s.index('        ',s.index('fn create(',start))
  # Locate the function body, not its signature continuation.
  body=s.index(' -> Result<Box<dyn Kernel>> {',start)+len(' -> Result<Box<dyn Kernel>> {')
  branch='''
        // Diagnostic-only cold selector. Removed after building this image.
        if std::env::var_os("NXRT_DIAGNOSTIC_FROZEN_PLAN").is_some() {
            return '''+frozen_factory+''' { memory: self.memory.clone(), cache_bytes: self.cache_bytes }.create(node, _input_shapes);
        }
'''
  s=s[:body]+branch+s[body:]
  marker='#[cfg(test)]\nmod tests {';i=s.index(marker);s=s[:i]+frozen+'\n\n'+s[i:]
  (root/p).write_text(s)
  target=out/(kind+'-instrumented.rs.gz');target.write_bytes(gzip.compress(s.encode(),mtime=0))
 ort='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt'
 env=os.environ|{'ORT_ROOT':ort,'ONNX_GENAI_ORT_LIB':ort+'/lib/libonnxruntime.1.32.0.dylib','CARGO_TARGET_DIR':'/tmp/nxrt-erf-nan-target','CARGO_INCREMENTAL':'0','CARGO_BUILD_JOBS':'1'}
 with (out/'build.jsonl').open('w') as log,(out/'build.log').open('w') as err:
  r=subprocess.run(['cargo','build','--offline','--release','--manifest-path','/tmp/nxrt-block-context-study/probe/Cargo.toml','--message-format=json-render-diagnostics'],cwd=root,env=env,stdout=log,stderr=err)
 if r.returncode:print((out/'build.log').read_text()[-6000:],flush=True);raise SystemExit(r.returncode)
 records=[json.loads(l) for l in (out/'build.jsonl').read_text().splitlines() if l.startswith('{')]
 bins=[r['executable'] for r in records if r.get('reason')=='compiler-artifact' and r.get('executable') and r['target']['name']=='nxrt-qmoe-fc-banks-probe'];assert len(bins)==1
 shutil.copy2(bins[0],out/'shared.binary')
 symbols=subprocess.check_output(['nm','-n',str(out/'shared.binary')],text=True)
 decode=[l for l in symbols.splitlines() if l.split()[-1].endswith('decode_mxfp4_block') and len(l.split())>=3]
 assert len(decode)==1,decode
 assert 'FrozenBlockQuantizedMoEKernel' in symbols and 'FrozenBlockQuantizedMatMulKernel' in symbols
 meta={'source_sha256':{p:sha(v) for p,v in original.items()},'baseline':base,'binary_sha256':sha((out/'shared.binary').read_bytes()),'single_decode_symbol':decode,'diagnostic_only':True,'runtime_selector':'factory creation only; absent=actual execution context, present=frozen constructor context','probe_sha256':sha(Path('/tmp/nxrt-block-context-study/probe/src/main.rs').read_bytes())}
 (out/'metadata.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2),flush=True)
finally:
 for p,v in original.items():(root/p).write_bytes(v)
assert all((root/p).read_bytes()==v for p,v in original.items())
print('Restored production source exactly; diagnostic selector is not part of candidate.',flush=True)
