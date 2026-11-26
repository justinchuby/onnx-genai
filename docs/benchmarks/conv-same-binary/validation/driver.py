from pathlib import Path
import os,subprocess,json,hashlib,re,sys
root=Path('/tmp/nxrt-qmoe-workspace');out=Path('/tmp/nxrt-conv-pair-validation');out.mkdir(exist_ok=True)
ort='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt'
env=os.environ|{'ORT_ROOT':ort,'ONNX_GENAI_ORT_LIB':ort+'/lib/libonnxruntime.1.32.0.dylib','NXRT_ORT_LIB_DIR':ort+'/lib','NXRT_REQUIRE_ORT_TESTS':'1','CARGO_INCREMENTAL':'0','CARGO_BUILD_JOBS':'1','CARGO_PROFILE_DEV_DEBUG':'0','CARGO_PROFILE_TEST_DEBUG':'0','CARGO_TARGET_DIR':'/tmp/nxrt-qmoe-main-target','RAYON_NUM_THREADS':'1','ONNX_GENAI_MLAS_THREADPOOL_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'}
env['PATH']='/opt/homebrew/Caskroom/miniconda/base/bin:'+env['PATH']
commands=[('default-conv',['test','--offline','-p','onnx-runtime-ep-cpu','--lib','kernels::conv::tests::'],{}),('default-registry',['test','--offline','-p','onnx-runtime-ep-cpu','--test','conv_same_binary','--','--nocapture'],{}),('allocation',['test','--offline','-p','onnx-runtime-ep-cpu','--test','conv_im2col_allocation'],{}),('pair-list',['test','--offline','-p','onnx-runtime-ep-cpu','--features','mlas','--test','conv_same_binary','--','--list'],{}),('pair',['test','--offline','-p','onnx-runtime-ep-cpu','--features','mlas','--test','conv_same_binary','--','--nocapture'],{'NXRT_CPU_GEMM_BACKEND':'mlas'}),('native-linked',['test','--offline','-p','onnx-runtime-ep-cpu','--features','mlas','--lib','kernels::conv_native::tests::'],{'NXRT_CPU_GEMM_BACKEND':'mlas'}),('clippy',['clippy','--offline','-p','onnx-runtime-ep-cpu','--all-targets','--','-D','warnings'],{}),('clippy-mlas',['clippy','--offline','-p','onnx-runtime-ep-cpu','--features','mlas','--all-targets','--','-D','warnings'],{})]
paths=['crates/onnx-runtime-ep-cpu/'+p for p in ['src/backend_ab.rs','src/kernels/conv_ref.rs','src/kernels/mod.rs','tests/conv_same_binary.rs']];sha=lambda b:hashlib.sha256(b).hexdigest();sources={p:sha((root/p).read_bytes()) for p in paths};results=[]
for name,args,extra in commands:
 print('RUN',name,flush=True)
 with (out/(name+'.log')).open('w') as f:r=subprocess.run(['cargo',*args],cwd=root,env=env|extra,stdout=f,stderr=subprocess.STDOUT)
 log=(out/(name+'.log')).read_text();record={'name':name,'command':['cargo',*args],'extra_env':extra,'exit':r.returncode,'log_sha256':sha(log.encode())};results.append(record)
 print(log[-4500:] if r.returncode else '\n'.join(l for l in log.splitlines() if 'test result:' in l or 'Finished' in l or 'compared ' in l or l.endswith(': test')),flush=True)
 if name=='pair-list' and r.returncode==0:assert 'linked::linked_conv_factories_are_independent_and_match_f64_on_real_execute: test' in log
 elif 'test' in args and r.returncode==0:
  counts=re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;',log);assert counts and all(int(p)>0 and int(f)==0 for p,f,i in counts);record['counts']=counts
 (out/'validation.json').write_text(json.dumps({'source_sha256':sources,'results':results},indent=2))
 if r.returncode:sys.exit(r.returncode)
assert sources=={p:sha((root/p).read_bytes()) for p in paths}
