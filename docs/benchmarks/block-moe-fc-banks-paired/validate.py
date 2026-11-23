import subprocess,os,pathlib,json,hashlib
p=pathlib.Path('/tmp/nxrt-bank-paired');r=pathlib.Path('/tmp/nxrt-qmoe-workspace')
env=dict(os.environ,CARGO_TARGET_DIR='/tmp/nxrt-erf-nan-target',CARGO_INCREMENTAL='0',ORT_ROOT='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt')
env['ONNX_GENAI_ORT_LIB']=env['ORT_ROOT']+'/lib/libonnxruntime.1.32.0.dylib'
for x in ['ONNX_GENAI_MOE_GROUPED_MIN_WORK','ONNX_GENAI_CPU_DECODE_THREADS']:env.pop(x,None)
with (p/'build.log').open('w') as log:
 result=subprocess.run(['cargo','build','--offline','--release','--manifest-path',str(p/'probe/Cargo.toml')],env=env,cwd=r,stdout=log,stderr=subprocess.STDOUT)
if result.returncode:print((p/'build.log').read_text()[-6500:],flush=True);raise SystemExit(result.returncode)
binary=pathlib.Path(env['CARGO_TARGET_DIR'])/'release/nxrt-bank-paired-probe'
shapes=[(1,32,64,1,0,0,'block'),(8,128,256,1,0,0,'block'),(1,64,128,1,1,0,'block'),(8,128,256,4,0,1,'block'),(8,128,256,4,1,2,'block'),(8,128,256,4,0,3,'block'),(1,64,128,1,0,0,'float'),(8,128,256,4,0,1,'float')]
records=[]
for idx,(m,k,n,t,d,g,op) in enumerate(shapes):
 result=subprocess.run([str(binary),str(m),str(k),str(n),'1',str(t),str(p/f'smoke-{idx}.f32')],env=dict(env,NXRT_GATHER_DENSE=str(d),NXRT_BLOCK_GATED=str(g),NXRT_BANK_OP=op,NXRT_PAIR_ROTATION=str(idx%3)),capture_output=True,text=True)
 (p/f'smoke-{idx}.log').write_text(result.stdout+'\nSTDERR:\n'+result.stderr)
 if result.returncode:print(result.stdout+result.stderr,flush=True);raise SystemExit(result.returncode)
 rows=[json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')];assert len(rows)==3
 records.append({'shape':shapes[idx],'rows':rows,'output_sha256':hashlib.sha256((p/f'smoke-{idx}.f32').read_bytes()).hexdigest()});print(json.dumps({'smoke':idx,'status':'pass'}),flush=True)
(p/'smoke.json').write_text(json.dumps(records,indent=2)+'\n')
print('Smoke validation only; timings are intentionally below measurement gates.',flush=True)
