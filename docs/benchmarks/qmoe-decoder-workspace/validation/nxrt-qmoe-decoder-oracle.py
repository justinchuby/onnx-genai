import hashlib,json,os,pathlib,re,subprocess
root=pathlib.Path('/tmp/nxrt-qmoe-workspace');out=pathlib.Path('/tmp/nxrt-qmoe-decoder-validation')
ort='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt'
env=os.environ|{'ORT_ROOT':ort,'ONNX_GENAI_ORT_LIB':ort+'/lib/libonnxruntime.1.32.0.dylib','NXRT_ORT_LIB_DIR':ort+'/lib','NXRT_REQUIRE_ORT_TESTS':'1','CARGO_INCREMENTAL':'0','CARGO_BUILD_JOBS':'1','CARGO_PROFILE_DEV_DEBUG':'0','CARGO_PROFILE_TEST_DEBUG':'0','CARGO_TARGET_DIR':'/tmp/nxrt-qmoe-main-target'}
env['PATH']='/opt/homebrew/Caskroom/miniconda/base/bin:'+env['PATH']
command=['cargo','test','--offline','-p','onnx-runtime-ep-cuda','--test','qmoe_cpu_oracle']
with (out/'cpu-oracle.log').open('w') as f:r=subprocess.run(command,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
s=(out/'cpu-oracle.log').read_text();print(s[-3500:],flush=True);assert r.returncode==0
counts=re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;',s);assert counts and all(int(p)>0 and int(f)==0 for p,f,i in counts)
(out/'cpu-oracle.json').write_text(json.dumps({'command':command,'counts':counts,'exit':r.returncode,'sha256':hashlib.sha256(s.encode()).hexdigest(),'gpu_execution':False},indent=2))
