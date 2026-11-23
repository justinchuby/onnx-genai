import subprocess,os,pathlib,json,hashlib,shutil,statistics
p=pathlib.Path('/tmp/nxrt-bank-inplace-gate');r=pathlib.Path('/tmp/nxrt-qmoe-workspace');env=dict(os.environ,CARGO_TARGET_DIR='/tmp/nxrt-erf-nan-target',CARGO_INCREMENTAL='0',ORT_ROOT='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt');env['ONNX_GENAI_ORT_LIB']=env['ORT_ROOT']+'/lib/libonnxruntime.1.32.0.dylib'
for key in ['ONNX_GENAI_MOE_GROUPED_MIN_WORK','ONNX_GENAI_CPU_DECODE_THREADS']:env.pop(key,None)
shapes=[(1,32,64,1,0,0,'block'),(8,128,256,1,0,0,'block'),(1,64,128,1,1,0,'block'),(8,128,256,4,0,1,'block'),(8,128,256,4,1,2,'block'),(8,128,256,4,0,3,'block'),(1,64,128,1,0,0,'float'),(8,128,256,4,0,1,'float')]
protocol={'purpose':'Evaluate serial in-place activation and its smaller actual backing against prior static-bias banks and the old allocation-based driver. Full prior eight synthetic cases, not CUDA/model throughput or a rebased-main adoption audit.','shapes':shapes,'arms':{'before':'old kernels from 509144567','original':'prior static-bias banks from f0a2cd115','compact':'in-place activation/minimum serial banks at recorded candidate head','null':'identical old factory as before'},'processes_per_case':4,'rotations':[0,1,2,3],'batches':8,'target_cpu_ms':250,'excluded_pilot_minimum_cpu_ms':50,'minimum_batch_cpu_ms':20,'minimum_cpu_wall_ratio':0.9,'control_and_AA_gate_percent':5,'attempts':1,'lock':'/tmp/onnx-genai-hostlock-darwin.flock','sampling':False,'scope':'native Apple M1 Max, same executable/CPU crate, public strict prepare/run/output drop, independent funded pools/books; no CUDA/model/RSS claim'}
(p/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
with (p/'sweep-build.log').open('w') as log:result=subprocess.run(['cargo','build','--offline','--release','--manifest-path',str(p/'probe/Cargo.toml')],cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT)
if result.returncode:print((p/'sweep-build.log').read_text()[-6000:],flush=True);raise SystemExit(result.returncode)
binary=p/'inplace-gate-probe';shutil.copy2(pathlib.Path(env['CARGO_TARGET_DIR'])/'release/nxrt-bank-inplace-gate-probe',binary)
sha=lambda b:hashlib.sha256(b).hexdigest();files=subprocess.check_output(['git','diff','--name-only'],cwd=r,text=True).splitlines()+subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=r,text=True).splitlines();source={f:sha((r/f).read_bytes()) for f in files}
nm=subprocess.check_output(['nm',str(binary)],text=True);assert 'probe_before_block_quantized_moe' in nm and 'probe_original_banks_block_quantized_moe' in nm and 'Mlas' not in nm
symbols=[line.split()[-1] for line in nm.splitlines() if 'compact_gated_activation' in line]
for idx,symbol in enumerate(symbols):
 (p/f'compact-assembly-{idx}.txt').write_text(subprocess.check_output(['/opt/homebrew/opt/llvm/bin/llvm-objdump','--disassemble','--disassemble-symbols='+symbol,str(binary)],text=True))
(p/'provenance.json').write_text(json.dumps({'candidate':subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip(),'old_bank_head':subprocess.check_output(['git','rev-parse','f0a2cd115'],cwd=r,text=True).strip(),'baseline':'509144567bc6f74d899385aacee5717d20811cc0','binary_sha256':sha(binary.read_bytes()),'source_sha256':source,'probe_sha256':sha((p/'probe/src/main.rs').read_bytes()),'compact_specializations':symbols,'rustc':subprocess.check_output(['rustc','-vV'],text=True)},indent=2)+'\n')
records=[]
def run(shape,rotation,reps,prefix):
 m,k,n,t,d,g,op=shape;out=p/f'{prefix}.f32'
 (p/f'{prefix}-host.log').write_text(subprocess.check_output(['ps','-Ao','pid=,pcpu=,comm='],text=True))
 result=subprocess.run([str(binary),str(m),str(k),str(n),str(reps),str(t),str(out)],env=dict(env,NXRT_GATHER_DENSE=str(d),NXRT_BLOCK_GATED=str(g),NXRT_BANK_OP=op,NXRT_PAIR_ROTATION=str(rotation)),capture_output=True,text=True)
 (p/f'{prefix}.log').write_text(result.stdout+'\nSTDERR:\n'+result.stderr)
 if result.returncode:print(result.stdout+result.stderr,flush=True);raise SystemExit(result.returncode)
 arms=[json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')];assert len(arms)==4
 return {'rotation':rotation,'shape':shape,'arms':arms,'output_sha256':sha(out.read_bytes())}
for idx,shape in enumerate(shapes):run(shape,0,1,f'smoke-{idx}');print(json.dumps({'smoke':idx,'status':'pass'}),flush=True)
for rotation in range(4):
 for idx,shape in enumerate(shapes):
  records.append(run(shape,rotation,0,f'process{rotation}-case{idx}'));(p/'runs.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps({'rotation':rotation,'case':idx,'complete':True}),flush=True)
ratio=lambda x,y:statistics.median(a/b for a,b in zip(x,y));summary=[]
for shape in shapes:
 rows=[x for x in records if x['shape']==shape];assert len(rows)==4 and len({x['output_sha256'] for x in rows})==1;cells=[]
 for row in rows:
  arms={x['arm']:x for x in row['arms']};b=arms['before'];minimum=min(c/w for x in arms.values() for c,w in zip(x['cpu_batch_ms'],x['wall_batch_ms']));duration=all(min(x['cpu_batch_ms'])>=20 and min(x['control_batch_ms'])>=20 for x in arms.values());controls={label:ratio(x['control_batch_ms'],b['control_batch_ms']) for label,x in arms.items()};aa=ratio(arms['null']['cpu_batch_ms'],b['cpu_batch_ms']);cell={'rotation':row['rotation'],'qualified':duration and minimum>=.9 and .95<=aa<=1.05 and all(.95<=v<=1.05 for v in controls.values()),'duration_pass':duration,'minimum_cpu_wall_ratio':minimum,'control_ratios':controls,'AA_cpu_ratio':aa,'comparisons':{}}
  for x,y in [('compact','original'),('original','before'),('compact','before'),('null','before')]:cell['comparisons'][x+'/'+y]={metric:ratio(arms[x][metric+'_batch_ms'],arms[y][metric+'_batch_ms']) for metric in ['cpu','wall']}
  m,h,i,t,d,g,op=shape; shrink=0 if g==0 else m*(max(h,i)+max(2*i,h)-h-2*i)*4
  assert arms['original']['workspace_bytes']-arms['compact']['workspace_bytes']==shrink;cells.append(cell)
 summary.append({'shape':shape,'qualified':all(x['qualified'] for x in cells),'processes':cells,'median_ratios':{key:{metric:statistics.median(x['comparisons'][key][metric] for x in cells) for metric in ['cpu','wall']} for key in cells[0]['comparisons']}})
assert all(sha((r/f).read_bytes())==v for f,v in source.items()),'source changed during measurement'
(p/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'summary':summary}),flush=True)
