from pathlib import Path
import subprocess,os,time,json,hashlib,gzip,fcntl,struct,math
out=Path(__file__).parent;meta=json.loads((out/'metadata.json').read_text());sha=lambda b:hashlib.sha256(b).hexdigest();assert sha((out/'paired.binary').read_bytes())==meta['binary_sha256'];assert sha((out/'probe/src/main.rs').read_bytes())==meta['probe_sha256']
with open('/tmp/onnx-genai-hostlock-darwin.flock','a+') as f:
 try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:pass
 else:raise AssertionError('wrapper must own shared host lock')
protocol={'scope':'same-process warm wall blocks, not per-arm CPU or end-to-end process latency','repetitions':3,'normal_calls':8192,'double_calls':16384,'geometry':[8,128,256],'execution_width':1,'construction_widths':[1,4],'normal_order':'ABBA BAAB, reversed on odd repetitions','physical_bytes':16384,'shared_input_tensors':True,'shared_physical_owner':True,'factory_selection':'thread local, cold only; no process env mutation','cold_build_warm5_drop_and_validation':'outside each wall window, included in whole-process CPU/wall envelope','quality':{'aa':[.95,1.05],'adjusted_double':[1.7,2.3],'work_wall_s_min':.05,'empty_fraction_max':.01,'physical_cpu_wall_factor':11},'attempts':1,'max_cells':2,'no_gpu':True}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2));ort='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt';env=os.environ|{'ORT_ROOT':ort,'ONNX_GENAI_ORT_LIB':ort+'/lib/libonnxruntime.1.32.0.dylib'}
for k in ['NXRT_DIAGNOSTIC_FROZEN_PLAN','ONNX_GENAI_CPU_DECODE_THREADS','ONNX_GENAI_MOE_GROUPED_MIN_WORK','ONNX_GENAI_WEIGHT_OFFLOAD','ONNX_GENAI_WEIGHT_OFFLOAD_PREFETCH','ONNX_GENAI_WEIGHT_OFFLOAD_HOST_BYTES']:env.pop(k,None)
rows=[]
for rep in range(3):
 name=f'r{rep}';output=out/(name+'.f32');started=time.monotonic()
 with (out/(name+'.stdout')).open('w') as f,(out/(name+'.stderr')).open('w') as e:
  child=subprocess.Popen([str(out/'paired.binary'),str(rep),str(protocol['normal_calls']),str(output)],env=env,stdout=f,stderr=e)
  pid,status,usage=os.wait4(child.pid,0);child.returncode=os.waitstatus_to_exitcode(status)
 wall=time.monotonic()-started
 assert child.returncode==0,(name,(out/(name+'.stderr')).read_text()[-5000:])
 receipt=json.loads((out/(name+'.stdout')).read_text().strip().splitlines()[-1]);assert receipt['repetition']==rep and len(receipt['records'])==24 and receipt['final_used']==0 and receipt['same_workspace_owner'] and receipt['same_input_owner']
 assert 0<usage.ru_utime+usage.ru_stime<=wall*11 and sum(r['wall_s'] for r in receipt['records'])<=wall
 raw=output.read_bytes();assert len(raw)==8*256*4;values=struct.unpack('<2048f',raw);assert all(math.isfinite(v) for v in values) and any(v!=0 for v in values)
 rows.append(receipt|{'cpu_total_s':usage.ru_utime+usage.ru_stime,'wall_total_s':wall,'maxrss_bytes':usage.ru_maxrss,'exit':0,'output_sha256':sha(raw)});(out/'receipts.json.gz').write_bytes(gzip.compress(json.dumps(rows,sort_keys=True).encode(),mtime=0));print('PROCESS',rep,'CPUtotal',rows[-1]['cpu_total_s'],'walltotal',wall,flush=True)
assert len(rows)==3 and len({r['output_sha256'] for r in rows})==1
meta['receipts_sha256']=sha(gzip.decompress((out/'receipts.json.gz').read_bytes()));meta['runner_sha256']=sha(Path(__file__).read_bytes());meta['binary_after_sha256']=sha((out/'paired.binary').read_bytes());assert meta['binary_after_sha256']==meta['binary_sha256'];(out/'metadata.json').write_text(json.dumps(meta,indent=2));print('DONE72 diagnostic wall windows; no per-arm CPU claim.',flush=True)
