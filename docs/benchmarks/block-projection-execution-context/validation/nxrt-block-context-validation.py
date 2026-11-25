import os,pathlib,re,subprocess,sys,json
root=pathlib.Path('/tmp/nxrt-qmoe-workspace');out=pathlib.Path('/tmp/nxrt-block-context-validation');out.mkdir(exist_ok=True)
ort='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt'
env=os.environ|{'ORT_ROOT':ort,'ONNX_GENAI_ORT_LIB':ort+'/lib/libonnxruntime.1.32.0.dylib','NXRT_ORT_LIB_DIR':ort+'/lib','NXRT_REQUIRE_ORT_TESTS':'1','CARGO_INCREMENTAL':'0','CARGO_BUILD_JOBS':'1','CARGO_PROFILE_DEV_DEBUG':'0','CARGO_PROFILE_TEST_DEBUG':'0','CARGO_TARGET_DIR':'/tmp/nxrt-qmoe-main-target'}
env['PATH']='/opt/homebrew/Caskroom/miniconda/base/bin:'+env['PATH']
commands=[
 ('list-context',['test','--offline','-p','onnx-runtime-session','--lib','workspace_context_tests','--','--list']),
 ('context',['test','--offline','-p','onnx-runtime-session','--lib','workspace_context_tests']),
 ('native',['test','--offline','-p','onnx-runtime-ep-cpu','--lib']),
 ('mlas',['test','--offline','-p','onnx-runtime-ep-cpu','--lib','--features','mlas']),
 ('layout',['test','--offline','-p','onnx-runtime-ep-cpu','--test','decoded_projection_layout']),
 ('session',['test','--offline','-p','onnx-runtime-session','--lib']),
 ('clippy',['clippy','--offline','-p','onnx-runtime-ep-cpu','-p','onnx-runtime-ep-cpu-plugin','-p','onnx-runtime-session','--all-targets','--','-D','warnings']),
 ('clippy-mlas',['clippy','--offline','-p','onnx-runtime-ep-cpu','-p','onnx-runtime-ep-cpu-plugin','-p','onnx-runtime-session','--all-targets','--features','mlas','--','-D','warnings']),
]
import hashlib
paths=[p for p in subprocess.check_output(['git','diff','--name-only'],cwd=root,text=True).splitlines() if p.startswith('crates/')]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();source={p:sha(root/p) for p in paths};results=[]
for label,args in commands:
 c=['cargo',*args];print('RUN',label,flush=True)
 with (out/(label+'.log')).open('w') as f:r=subprocess.run(c,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
 s=(out/(label+'.log')).read_text();record={'name':label,'command':c,'exit':r.returncode,'log_sha256':sha(out/(label+'.log'))};results.append(record)
 print(s[-5000:] if r.returncode else '\n'.join(l for l in s.splitlines() if 'test result:' in l or 'Finished' in l or l.endswith(': test')),flush=True)
 if 'test' in args:
  if '--list' in args:
   assert s.count(': test')==3,(label,s[-5000:])
  else:
   counts=re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;',s);record['counts']=counts
   if r.returncode==0:assert counts and all(int(p)>0 and int(f)==0 for p,f,i in counts)
 (out/'validation.json').write_text(json.dumps({'source_sha256':source,'results':results},indent=2))
 if r.returncode:sys.exit(r.returncode)
assert source=={p:sha(root/p) for p in paths}
print('All final source hashes unchanged.',flush=True)
