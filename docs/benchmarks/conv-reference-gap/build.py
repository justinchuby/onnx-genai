from pathlib import Path
import os,subprocess,json,hashlib,shutil,platform,fcntl
root=Path('/tmp/nxrt-qmoe-workspace');out=Path('/tmp/nxrt-conv-gap-study');out.mkdir(exist_ok=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
paths=['crates/onnx-runtime-ep-cpu/'+p for p in ['src/kernels/conv_ref.rs','src/kernels/conv.rs','src/kernels/matmul.rs','src/backend_ab.rs','src/kernels/mod.rs','Cargo.toml']]+['Cargo.toml','Cargo.lock']
source={p:sha((root/p).read_bytes()) for p in paths};probe=Path('/tmp/nxrt-conv-gap-probe.rs').read_bytes();example=root/'crates/onnx-runtime-ep-cpu/examples/nxrt_conv_gap.rs';assert not example.exists()
ort='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt'
env=os.environ|{'ORT_ROOT':ort,'ONNX_GENAI_ORT_LIB':ort+'/lib/libonnxruntime.1.32.0.dylib','NXRT_ORT_LIB_DIR':ort+'/lib','CARGO_INCREMENTAL':'0','CARGO_BUILD_JOBS':'1','CARGO_TARGET_DIR':'/tmp/nxrt-erf-nan-target','RAYON_NUM_THREADS':'1','ONNX_GENAI_MLAS_THREADPOOL_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'}
env['PATH']='/opt/homebrew/Caskroom/miniconda/base/bin:'+env['PATH']
for k in ['NXRT_CPU_GEMM_BACKEND','NXRT_CPU_DISPATCH_LEDGER','ONNX_GENAI_PROFILE_OPS']:env.pop(k,None)
lock=Path('/tmp/onnx-genai-hostlock-darwin.flock');st=lock.stat();held=[]
for s in os.listdir('/dev/fd'):
 try:
  fd=int(s);f=os.fstat(fd)
  if f.st_dev==st.st_dev and f.st_ino==st.st_ino:held.append(fd)
 except (ValueError,OSError):pass
assert len(held)==1,held
fcntl.flock(held[0],fcntl.LOCK_EX|fcntl.LOCK_NB)
metadata={'base':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'source_sha256':source,'probe_sha256':sha(probe),'compiler':subprocess.check_output(['rustc','-Vv'],text=True),'platform':platform.platform(),'shared_lock':str(lock),'lock_inode':st.st_ino,'inherited_lock_fd':held[0],'parent_pid':os.getppid(),'binaries':{},'environment_overrides':{k:env[k] for k in ['RAYON_NUM_THREADS','ONNX_GENAI_MLAS_THREADPOOL_THREADS','VECLIB_MAXIMUM_THREADS','CARGO_TARGET_DIR','CARGO_INCREMENTAL','CARGO_BUILD_JOBS']}}
(out/'probe.rs').write_bytes(probe);(out/'build-metadata.json').write_text(json.dumps(metadata,indent=2))
try:
 example.write_bytes(probe)
 for label,features in [('default',[]),('linked',['--features','mlas'])]:
  cmd=['cargo','build','--offline','--release','-p','onnx-runtime-ep-cpu','--example','nxrt_conv_gap',*features,'--message-format=json-render-diagnostics'];print('BUILD',label,flush=True)
  with (out/(label+'-cargo.jsonl')).open('w') as f,(out/(label+'-build.log')).open('w') as e:r=subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=e)
  if r.returncode:print((out/(label+'-build.log')).read_text()[-5000:],flush=True);raise SystemExit(r.returncode)
  records=[json.loads(x) for x in (out/(label+'-cargo.jsonl')).read_text().splitlines() if x.startswith('{')];binaries=[x['executable'] for x in records if x.get('reason')=='compiler-artifact' and x.get('target',{}).get('name')=='nxrt_conv_gap' and x.get('executable')];assert len(binaries)==1,binaries
  dest=out/(label+'-probe');shutil.copy2(binaries[0],dest);symbols=subprocess.check_output(['nm','-a',str(dest)],text=True);(out/(label+'-symbols.log')).write_text(symbols);lines=symbols.splitlines();assert len(lines)>100 and any(x.split()[-1:] == ['_main'] for x in lines)
  mlas=[x for x in lines if 'mlas_sys' in x or (x.split() and x.split()[-1].startswith('_Mlas'))];assert bool(mlas)==(label=='linked'),(label,mlas[:10]);metadata['binaries'][label]={'sha256':sha(dest.read_bytes()),'path':str(dest),'command':cmd,'mlas_symbols':len(mlas),'symbol_lines':len(lines)};(out/'build-metadata.json').write_text(json.dumps(metadata,indent=2));print('BUILT',label,metadata['binaries'][label]['sha256'],flush=True)
 assert source=={p:sha((root/p).read_bytes()) for p in paths}
finally:
 if example.exists():assert example.read_bytes()==probe;example.unlink()
 metadata['temporary_example_removed']=not example.exists();(out/'build-metadata.json').write_text(json.dumps(metadata,indent=2))
