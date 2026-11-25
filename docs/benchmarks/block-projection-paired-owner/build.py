from pathlib import Path
import os,subprocess,json,gzip,hashlib,shutil
root=Path('/tmp/nxrt-qmoe-workspace');out=Path(__file__).parent;p='crates/onnx-runtime-ep-cpu/src/kernels/block_quantized_matmul.rs';original=(root/p).read_bytes();base='61c47ee27c19978a441443ed4e42995a7a172e15';sha=lambda b:hashlib.sha256(b).hexdigest()
def section(source,marker):
 start=source.index(marker);a=source.index('{',start);depth=1;i=a+1
 while depth:depth+=(source[i]=='{')-(source[i]=='}');i+=1
 return source[start:i]
try:
 old=subprocess.check_output(['git','show',base+':'+p],cwd=root).decode();s=original.decode();kernel='BlockQuantizedMatMulKernel';factory='BlockQuantizedMatMulFactory'
 frozen='\n\n'.join(section(old,m) for m in ['pub struct '+kernel+' {','pub struct '+factory+' {','impl KernelFactory for '+factory+' {','impl Kernel for '+kernel+' {','impl '+kernel+' {']).replace(kernel,'Frozen'+kernel).replace(factory,'Frozen'+factory)
 helpers='''
thread_local! {
 static DIAGNOSTIC_FROZEN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
 static DIAGNOSTIC_FACTORIES: std::cell::Cell<(usize,usize)> = const { std::cell::Cell::new((0,0)) };
}
pub fn diagnostic_select_frozen(frozen:bool) { DIAGNOSTIC_FROZEN.with(|v|v.set(frozen)); }
pub fn diagnostic_factory_stats()->(usize,usize) { DIAGNOSTIC_FACTORIES.with(std::cell::Cell::get) }
'''
 start=s.index('impl KernelFactory for '+factory+' {');body=s.index(' -> Result<Box<dyn Kernel>> {',start)+len(' -> Result<Box<dyn Kernel>> {')
 branch='''
        if DIAGNOSTIC_FROZEN.with(std::cell::Cell::get) {
            DIAGNOSTIC_FACTORIES.with(|v| { let (a,b)=v.get();v.set((a+1,b)); });
            return FrozenBlockQuantizedMatMulFactory { memory:self.memory.clone(),cache_bytes:self.cache_bytes }.create(node,_input_shapes);
        }
        DIAGNOSTIC_FACTORIES.with(|v| { let (a,b)=v.get();v.set((a,b+1)); });
'''
 s=s[:body]+branch+s[body:];i=s.index('#[cfg(test)]\nmod tests {');s=s[:i]+helpers+frozen+'\n\n'+s[i:];(root/p).write_text(s);(out/'instrumented.rs.gz').write_bytes(gzip.compress(s.encode(),mtime=0))
 ort='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt';env=os.environ|{'ORT_ROOT':ort,'ONNX_GENAI_ORT_LIB':ort+'/lib/libonnxruntime.1.32.0.dylib','CARGO_TARGET_DIR':'/tmp/nxrt-erf-nan-target','CARGO_INCREMENTAL':'0','CARGO_BUILD_JOBS':'1'}
 with (out/'build.jsonl').open('w') as f,(out/'build.log').open('w') as e:r=subprocess.run(['cargo','build','--offline','--release','--manifest-path',str(out/'probe/Cargo.toml'),'--message-format=json-render-diagnostics'],cwd=root,env=env,stdout=f,stderr=e)
 if r.returncode:print((out/'build.log').read_text()[-6000:],flush=True);raise SystemExit(r.returncode)
 records=[json.loads(l) for l in (out/'build.jsonl').read_text().splitlines() if l.startswith('{')];bins=[r['executable'] for r in records if r.get('reason')=='compiler-artifact' and r.get('executable') and r['target']['name']=='nxrt-qmoe-fc-banks-probe'];assert len(bins)==1;shutil.copy2(bins[0],out/'paired.binary')
 symbols=subprocess.check_output(['nm','-n',str(out/'paired.binary')],text=True);decoders=[l for l in symbols.splitlines() if len(l.split())>=3 and l.split()[-1].endswith('decode_mxfp4_block')];assert len(decoders)==1
 (out/'metadata.json').write_text(json.dumps({'source_base':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'source_sha256':{p:sha(original)},'binary_sha256':sha((out/'paired.binary').read_bytes()),'probe_sha256':sha((out/'probe/src/main.rs').read_bytes()),'probe_lock_sha256':sha((out/'probe/Cargo.lock').read_bytes()),'single_decode_symbol':decoders,'selector':'thread-local factory-only; no process environment mutation','scope':'diagnostic, never shipped'},indent=2));print('Built paired diagnostic with one decoder symbol.',flush=True)
finally:(root/p).write_bytes(original)
assert (root/p).read_bytes()==original
print('Restored production source exactly.',flush=True)
