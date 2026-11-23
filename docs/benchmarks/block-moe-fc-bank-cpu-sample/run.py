import subprocess,os,pathlib,json,hashlib,shutil
p=pathlib.Path('/tmp/nxrt-bank-cpu-sample');r=pathlib.Path('/tmp/nxrt-qmoe-workspace');env=dict(os.environ,CARGO_TARGET_DIR='/tmp/nxrt-erf-nan-target',CARGO_INCREMENTAL='0',ORT_ROOT='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt');env['ONNX_GENAI_ORT_LIB']=env['ORT_ROOT']+'/lib/libonnxruntime.1.32.0.dylib'
for key in ['ONNX_GENAI_MOE_GROUPED_MIN_WORK','ONNX_GENAI_CPU_DECODE_THREADS']:env.pop(key,None)
env.update(NXRT_GATHER_DENSE='0',NXRT_BLOCK_GATED='1',NXRT_BANK_OP='block')
protocol={'scope':'CPU stack diagnosis only, not a performance measurement; separate sampler overhead from prior formal benchmarks','case':[8,128,256,4,0,1,'block'],'orders':[['before','banked','null'],['banked','null','before']],'steady_window_wall_seconds':15,'sample_duration_seconds':5,'sample_interval_ms':1,'sample_command':['/usr/bin/sample','EXACT_OWN_CHILD_PID','5','1','-mayDie','-file','OUTPUT'],'lock':'/tmp/onnx-genai-hostlock-darwin.flock','sampler_only_targets_own_synthetic_child':True}
(p/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
with (p/'build.log').open('w') as out:result=subprocess.run(['cargo','build','--offline','--release','--manifest-path',str(p/'probe/Cargo.toml')],cwd=r,env=env,stdout=out,stderr=subprocess.STDOUT)
if result.returncode:print((p/'build.log').read_text()[-5000:],flush=True);raise SystemExit(result.returncode)
binary=p/'sample-probe';shutil.copy2(pathlib.Path(env['CARGO_TARGET_DIR'])/'release/nxrt-bank-cpu-sample',binary)
sha=lambda b:hashlib.sha256(b).hexdigest();files=subprocess.check_output(['git','diff','--name-only'],cwd=r,text=True).splitlines()+subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=r,text=True).splitlines();source={f:sha((r/f).read_bytes()) for f in files}
(p/'provenance.json').write_text(json.dumps({'candidate':subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip(),'baseline':'509144567bc6f74d899385aacee5717d20811cc0','binary_sha256':sha(binary.read_bytes()),'source_sha256':source,'probe_sha256':sha((p/'probe/src/main.rs').read_bytes()),'rustc':subprocess.check_output(['rustc','-vV'],text=True)},indent=2)+'\n')
records=[]
for rep,order in enumerate(protocol['orders']):
 for label in order:
  prefix=f'round{rep}-{label}';output=p/f'{prefix}.f32';stderr=(p/f'{prefix}-stderr.log').open('w')
  child=subprocess.Popen([str(binary),label,str(output)],cwd=r,env=env,stdout=subprocess.PIPE,stderr=stderr,text=True)
  ready=child.stdout.readline();assert ready.startswith('READY '),(ready,child.poll());assert child.poll() is None
  (p/f'{prefix}-host.log').write_text(subprocess.check_output(['ps','-Ao','pid=,pcpu=,comm='],text=True))
  sampled=subprocess.run(['/usr/bin/sample',str(child.pid),'5','1','-mayDie','-file',str(p/f'{prefix}-sample.txt')],capture_output=True,text=True)
  (p/f'{prefix}-sampler.log').write_text(sampled.stdout+'\nSTDERR:\n'+sampled.stderr)
  rest=child.stdout.read();code=child.wait();stderr.close();(p/f'{prefix}-stdout.log').write_text(ready+rest)
  assert code==0,(prefix,code,(p/f'{prefix}-stderr.log').read_text());assert sampled.returncode==0,(prefix,sampled.stdout,sampled.stderr)
  assert 'DONE ' in rest;sample=(p/f'{prefix}-sample.txt').read_text();assert 'Call graph:' in sample and 'onnx' in sample,(prefix,sample[:1500])
  records.append({'round':rep,'arm':label,'pid':child.pid,'ready':json.loads(ready.removeprefix('READY ')),'output_sha256':sha(output.read_bytes()),'sample_sha256':sha((p/f'{prefix}-sample.txt').read_bytes())});(p/'runs.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps({'round':rep,'arm':label,'sample_complete':True}),flush=True)
assert len({x['output_sha256'] for x in records})==1
assert all(sha((r/f).read_bytes())==v for f,v in source.items())
print('All six samples captured; full outputs and ownership checks agree. Sampled call counts are not throughput evidence.',flush=True)
