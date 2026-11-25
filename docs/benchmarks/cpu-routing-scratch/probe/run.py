from pathlib import Path
import gzip,hashlib,json,os,subprocess,re
root=Path('/tmp/nxrt-qmoe-workspace');out=Path(__file__).parent
paths=['crates/onnx-runtime-ep-cpu/src/lib.rs','crates/onnx-runtime-ep-cpu/src/kernels/moe.rs','crates/onnx-runtime-ep-cpu/src/kernels/qmoe.rs','crates/onnx-runtime-ep-cpu/src/kernels/block_quantized_moe.rs']
source={p:(root/p).read_bytes() for p in paths};sha=lambda b:hashlib.sha256(b).hexdigest()
ort='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt'
env=os.environ|{'ORT_ROOT':ort,'ONNX_GENAI_ORT_LIB':ort+'/lib/libonnxruntime.1.32.0.dylib','NXRT_ORT_LIB_DIR':ort+'/lib','NXRT_REQUIRE_ORT_TESTS':'1','CARGO_INCREMENTAL':'0','CARGO_BUILD_JOBS':'1','CARGO_PROFILE_DEV_DEBUG':'0','CARGO_PROFILE_TEST_DEBUG':'0','CARGO_TARGET_DIR':'/tmp/nxrt-qmoe-main-target'}
env['PATH']='/opt/homebrew/Caskroom/miniconda/base/bin:'+env['PATH']
base='d39962a67d8f0b3166036341028f6834204f331a'
moe='crates/onnx-runtime-ep-cpu/src/kernels/moe.rs'
old=subprocess.check_output(['git','show',base+':'+moe],cwd=root,text=True);start=old.index('pub(super) fn routing_weights(');old=old[start:old.index('\nfn activate(',start)].replace('fn routing_weights(', 'fn reference_routing_weights(').strip()
new=source[moe].decode();start=new.index('pub(super) fn reference_routing_weights(');new=new[start:new.index('\n#[cfg(test)]\nmod routing_scratch_tests',start)].strip();assert new==old
reference_sha256=sha(old.encode())
command=['cargo','test','--offline','-p','onnx-runtime-ep-cpu','--lib','kernels::moe::routing_allocation_probe::repeated_rows_reuse_routing_metadata','--','--exact','--nocapture']
try:
 for p,raw in source.items():
  if p.endswith('/lib.rs'):raw+=b'\n'+(out/'allocator.rs').read_bytes()
  if p.endswith('/moe.rs'):raw+=b'\n'+(out/'probe.rs').read_bytes()
  (root/p).write_bytes(raw);(out/(Path(p).name+'.gz')).write_bytes(gzip.compress(raw,mtime=0))
 print('RUN same-image old-reference and reusable scratch allocation probe',flush=True)
 result=subprocess.run(command,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);(out/'probe.log').write_text(result.stdout)
 print('\n'.join(l for l in result.stdout.splitlines() if l.startswith(('CONTROL','CASE')) or 'test result:' in l) if result.returncode==0 else result.stdout[-5000:],flush=True)
 assert result.returncode==0 and '1 passed; 0 failed' in result.stdout
 rows=[]
 for values in re.findall(r'CASE experts=(\d+) k=(\d+) normalize=(true|false) aggregation=(true|false) calls=(\d+) baseline_alloc=(\d+) baseline_free=(\d+) candidate_alloc=(\d+) candidate_free=(\d+)',result.stdout):
  e,k,n,a,c,ba,bf,ca,cf=values;row=dict(experts=int(e),k=int(k),normalize=n=='true',aggregation=a=='true',calls=int(c),baseline_alloc=int(ba),baseline_free=int(bf),candidate_alloc=int(ca),candidate_free=int(cf));rows.append(row)
  assert row['calls']==2048 and row['baseline_alloc']==row['baseline_free']>0 and row['candidate_alloc']==row['candidate_free']==0
 assert len(rows)==28
 match=re.search(r'Running unittests.*\(([^)]+)\)',result.stdout);assert match
 binary=Path(match.group(1));assert binary.is_file();binary_sha256=sha(binary.read_bytes())
finally:
 for p,raw in source.items():(root/p).write_bytes(raw)
 assert all((root/p).read_bytes()==raw for p,raw in source.items())
 print('RESTORED exact production source bytes.',flush=True)
(out/'results.json').write_text(json.dumps({'scope':'macOS arm64 native debug same-image private row-routing allocation/free calls; not timing, whole-kernel allocations or governed workspace','base':'d39962a67d8f0b3166036341028f6834204f331a','source_sha256':{p:sha(b) for p,b in source.items()},'command':command,'exit':result.returncode,'log_sha256':sha(result.stdout.encode()),'cases':rows,'production_restored':True,'reference_matches_main_except_name':True,'reference_sha256':reference_sha256,'test_binary_sha256':binary_sha256},indent=2))
