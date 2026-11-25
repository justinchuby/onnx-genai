"""Diagnostic shared-code-image control, not a replacement for the first matrix."""
from pathlib import Path
import os,json,gzip,hashlib,subprocess,time,fcntl
out=Path(__file__).parent
meta=json.loads((out/'metadata.json').read_text());sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha((out/'shared.binary').read_bytes())==meta['binary_sha256'] and len(meta['single_decode_symbol'])==1
lock='/tmp/onnx-genai-hostlock-darwin.flock'
with open(lock,'a+') as f:
 try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:pass
 else:raise AssertionError('shared host lock not held by wrapper')
shapes=[(1,64,128,1,0,1,1,0),(8,128,256,1,0,0,4,1),(8,128,256,4,1,0,4,1)]
protocol={'scope':'shared executable/code/constant layout causal control; no production change','shapes_rows_hidden_inter_execution_dense_gate_construction_matmul':shapes,'repetitions':3,'work_macs':4294967296,'arms':['baseline','candidate','null-baseline','null-candidate'],'kinds':['setup','work','double'],'double_arms':['baseline','candidate'],'quality':{'aa':[.95,1.05],'setup_fraction_max':.25,'adjusted_double':[1.7,2.3],'work_s_min':.05,'physical_cpu_wall_factor':11},'attempts':1,'selector_scope':'Factory creation only; no selector branch on warm dispatch','binary_sha256':meta['binary_sha256']}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2));rows=[]
ort='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt'
env=os.environ|{'ORT_ROOT':ort,'ONNX_GENAI_ORT_LIB':ort+'/lib/libonnxruntime.1.32.0.dylib'}
for name in ['ONNX_GENAI_CPU_DECODE_THREADS','ONNX_GENAI_MOE_GROUPED_MIN_WORK','ONNX_GENAI_WEIGHT_OFFLOAD','ONNX_GENAI_WEIGHT_OFFLOAD_HOST_BYTES','ONNX_GENAI_WEIGHT_OFFLOAD_PREFETCH','ONNX_GENAI_QMOE_ROUTE_DUMP','NXRT_DIAGNOSTIC_FROZEN_PLAN']:env.pop(name,None)
for rep in range(3):
 for case,shape in enumerate(shapes):
  m,h,i,width,dense,gated,construction,matmul=shape
  normal=protocol['work_macs']//(m*h*i*(1 if matmul else 2*(3 if gated else 2)))
  kinds=protocol['kinds'][rep:]+protocol['kinds'][:rep]
  for kind in kinds:
   arms=protocol['double_arms'][:] if kind=='double' else protocol['arms'][:]
   if (case+rep)%2:arms.reverse()
   for arm in arms:
    calls=0 if kind=='setup' else normal*(2 if kind=='double' else 1)
    local=env|{'NXRT_GATHER_DENSE':str(dense),'NXRT_BLOCK_GATED':str(gated),'NXRT_CONSTRUCTION_WIDTH':str(construction),'NXRT_BLOCK_MATMUL':str(matmul)}
    if arm.endswith('baseline'):local['NXRT_DIAGNOSTIC_FROZEN_PLAN']='1'
    stem=f'r{rep}-c{case}-{kind}-{arm}';output=out/(stem+'.f32');started=time.monotonic()
    with (out/(stem+'.stdout')).open('w') as f,(out/(stem+'.stderr')).open('w') as e:
     child=subprocess.Popen([str(out/'shared.binary'),str(m),str(h),str(i),str(calls),str(width),str(output)],env=local,stdout=f,stderr=e)
     pid,status,usage=os.wait4(child.pid,0);child.returncode=os.waitstatus_to_exitcode(status)
    wall=time.monotonic()-started;assert child.returncode==0,(stem,(out/(stem+'.stderr')).read_text()[-4000:])
    r=json.loads((out/(stem+'.stdout')).read_text().strip().splitlines()[-1]);assert r['calls']==calls and r['final_used']==0
    trace=(out/(stem+'.f32.trace.json')).read_text();assert (('block_matmul.streamed_rows' if matmul else 'block_moe.streamed_rows') in trace)==(not dense)
    rows.append(r|{'rep':rep,'case':case,'kind':kind,'arm':arm,'cpu_s':usage.ru_utime+usage.ru_stime,'wall_s':wall,'maxrss_bytes':usage.ru_maxrss,'exit':0,'output_sha256':sha(output.read_bytes())})
    (out/'receipts.json.gz').write_bytes(gzip.compress(json.dumps(rows,sort_keys=True).encode(),mtime=0));print('PROCESS',len(rows),stem,'CPU',round(rows[-1]['cpu_s'],4),flush=True)
assert len(rows)==90
meta['receipts_sha256']=sha(gzip.decompress((out/'receipts.json.gz').read_bytes()));meta['runner_sha256']=sha(Path(__file__).read_bytes());(out/'metadata.json').write_text(json.dumps(meta,indent=2))
print('DONE',len(rows),flush=True)
