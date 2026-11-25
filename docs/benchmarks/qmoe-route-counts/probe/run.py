from pathlib import Path
import gzip,hashlib,json,os,subprocess,re
root=Path('/tmp/nxrt-qmoe-workspace');out=Path(__file__).parent
base='bf0e6683be387d1b632b9cb522f058e5f81499db';metric='crates/onnx-runtime-ep-cpu/src/weight_offload.rs'
paths=['crates/onnx-runtime-ep-cpu/src/lib.rs',metric,'crates/onnx-runtime-ep-cpu/src/kernels/qmoe.rs'];source={p:(root/p).read_bytes() for p in paths};sha=lambda b:hashlib.sha256(b).hexdigest()
old=subprocess.check_output(['git','show',base+':'+metric],cwd=root,text=True);start=old.index('    pub fn record_routes(');old=old[start:old.index('\n    fn snapshot(',start)].replace('pub fn record_routes(', 'fn reference_record_routes(')
reference='\n#[cfg(test)]\nimpl WeightOffloadMetrics {\n'+old+'\n}\n';(out/'reference.rs').write_text(reference)
ort='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt'
env=os.environ|{'ORT_ROOT':ort,'ONNX_GENAI_ORT_LIB':ort+'/lib/libonnxruntime.1.32.0.dylib','NXRT_ORT_LIB_DIR':ort+'/lib','NXRT_REQUIRE_ORT_TESTS':'1','CARGO_INCREMENTAL':'0','CARGO_BUILD_JOBS':'1','CARGO_PROFILE_DEV_DEBUG':'0','CARGO_PROFILE_TEST_DEBUG':'0','CARGO_TARGET_DIR':'/tmp/nxrt-qmoe-main-target'}
env['PATH']='/opt/homebrew/Caskroom/miniconda/base/bin:'+env['PATH']
command=['cargo','test','--offline','-p','onnx-runtime-ep-cpu','--lib','weight_offload::route_count_allocation_probe::route_statistics_borrow_existing_groups_without_temporary_counts','--','--exact','--nocapture']
try:
 for p,raw in source.items():
  if p.endswith('/lib.rs'):raw+=b'\n'+(out/'allocator.rs').read_bytes()
  if p==metric:raw+=b'\n'+reference.encode()+b'\n'+(out/'probe.rs').read_bytes()
  (root/p).write_bytes(raw);(out/(Path(p).name+'.gz')).write_bytes(gzip.compress(raw,mtime=0))
 print('RUN same-image original route metrics/count-map vs borrowed task group counts',flush=True)
 result=subprocess.run(command,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);(out/'probe.log').write_text(result.stdout)
 print('\n'.join(l for l in result.stdout.splitlines() if l.startswith(('CONTROL','CASE')) or 'test result:' in l) if result.returncode==0 else result.stdout[-5000:],flush=True)
 assert result.returncode==0 and '1 passed; 0 failed' in result.stdout
 rows=[]
 for values in re.findall(r'CASE experts=(\d+) rows=(\d+) k=(\d+) active=(\d+) unique=(\d+) calls=(\d+) baseline_alloc=(\d+) baseline_free=(\d+) candidate_alloc=(\d+) candidate_free=(\d+)',result.stdout):
  row=dict(zip(['experts','rows','k','active','unique','calls','baseline_alloc','baseline_free','candidate_alloc','candidate_free'],map(int,values)));rows.append(row)
  assert row['calls']==2048 and row['active']==row['rows']*row['k'] and row['baseline_alloc']==row['baseline_free']>0 and row['candidate_alloc']==row['candidate_free']==0
 assert len(rows)==15
 match=re.search(r'Running unittests.*\(([^)]+)\)',result.stdout);assert match;binary=Path(match.group(1));assert binary.is_file();binary_sha256=sha(binary.read_bytes())
finally:
 for p,raw in source.items():(root/p).write_bytes(raw)
 assert all((root/p).read_bytes()==raw for p,raw in source.items())
 print('RESTORED exact production source bytes.',flush=True)
(out/'results.json').write_text(json.dumps({'scope':'native macOS arm64 debug same-image redundant route-count map + warmed metrics vs borrowed group counts + warmed metrics; no timing or whole-kernel allocation claim','base':base,'source_sha256':{p:sha(b) for p,b in source.items()},'command':command,'exit':result.returncode,'log_sha256':sha(result.stdout.encode()),'cases':rows,'production_restored':True,'reference_sha256':sha(reference.encode()),'test_binary_sha256':binary_sha256,'all_route_snapshots_identical':True},indent=2))
