from pathlib import Path
import gzip,hashlib,json,os,subprocess,re,ast
root=Path('/tmp/nxrt-qmoe-workspace');out=Path(__file__).parent
base='b249721e1c47a31b8078ae3ab3e1dc68bf5ad645';path='crates/onnx-runtime-ep-cpu/src/kernels/moe.rs';original=(root/path).read_bytes();sha=lambda b:hashlib.sha256(b).hexdigest()
old=subprocess.check_output(['git','show',base+':'+path],cwd=root,text=True);start=old.index('    pub(super) fn weights(');body=old[start:old.index('\n}\n\n// Frozen',start)];old_body=body
needle='''        self.indices
            .sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]).then_with(|| a.cmp(&b)));'''
replacement='''        self.indices.sort_unstable_by(|&a, &b| {
            ROUTE_ORDER_COMPARISONS.with(|count| count.set(count.get() + 1));
            logits[b].total_cmp(&logits[a]).then_with(|| a.cmp(&b))
        });'''
assert body.count(needle)==1;body=body.replace(needle,replacement).replace('pub(super) fn weights(', 'fn fullsort_weights(')
reference='\n#[cfg(test)]\nthread_local! { static ROUTE_ORDER_COMPARISONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }\n#[cfg(test)]\nimpl RoutingScratch {\n'+body+'\n}\n';(out/'reference.rs').write_text(reference)
needle='''        let compare =
            |&a: &usize, &b: &usize| logits[b].total_cmp(&logits[a]).then_with(|| a.cmp(&b));'''
replacement='''        let compare = |&a: &usize, &b: &usize| {
            ROUTE_ORDER_COMPARISONS.with(|count| count.set(count.get() + 1));
            logits[b].total_cmp(&logits[a]).then_with(|| a.cmp(&b))
        };'''
s=original.decode();assert s.count(needle)==1;s=s.replace(needle,replacement)+'\n'+reference+'\n'+(out/'probe.rs').read_text()
(out/'instrumented.rs.gz').write_bytes(gzip.compress(s.encode(),mtime=0));(out/'candidate.rs.gz').write_bytes(gzip.compress(original,mtime=0));(out/'baseline.rs.gz').write_bytes(gzip.compress(old.encode(),mtime=0))
ort='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt';env=os.environ|{'ORT_ROOT':ort,'ONNX_GENAI_ORT_LIB':ort+'/lib/libonnxruntime.1.32.0.dylib','NXRT_ORT_LIB_DIR':ort+'/lib','NXRT_REQUIRE_ORT_TESTS':'1','CARGO_INCREMENTAL':'0','CARGO_BUILD_JOBS':'1','CARGO_PROFILE_DEV_DEBUG':'0','CARGO_PROFILE_TEST_DEBUG':'0','CARGO_TARGET_DIR':'/tmp/nxrt-qmoe-main-target'};env['PATH']='/opt/homebrew/Caskroom/miniconda/base/bin:'+env['PATH']
command=['cargo','test','--offline','-p','onnx-runtime-ep-cpu','--lib','kernels::moe::selection_comparison_probe::count_ordering_work_without_assuming_latency','--','--exact','--nocapture']
try:
 (root/path).write_text(s);print('RUN exact comparison-count diagnostic; no timing claim',flush=True)
 result=subprocess.run(command,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);(out/'probe.log').write_text(result.stdout)
 assert result.returncode==0 and '1 passed; 0 failed' in result.stdout,result.stdout[-5000:]
 rows=[];outputs={}
 for e,k,p,b,c in re.findall(r'CASE experts=(\d+) k=(\d+) pattern=(\w+) baseline_comparisons=(\d+) candidate_comparisons=(\d+)',result.stdout):rows.append(dict(experts=int(e),k=int(k),pattern=p,baseline_comparisons=int(b),candidate_comparisons=int(c)))
 for e,k,p,pairs in re.findall(r'OUTPUT experts=(\d+) k=(\d+) pattern=(\w+) pairs=(.*)',result.stdout):outputs[f'{e}:{k}:{p}']=ast.literal_eval(pairs)
 assert len(rows)==len(outputs)==160
 controls=[r for r in rows if r['experts']>=2 and r['k']>=r['experts']];assert controls and all(r['baseline_comparisons']==r['candidate_comparisons']>0 for r in controls)
 assert all(r['candidate_comparisons']==0 for r in rows if r['k']==0)
 for row in rows:
  pairs=outputs[f"{row['experts']}:{row['k']}:{row['pattern']}"];assert len(pairs)==min(row['experts'],row['k']) and len({p[0] for p in pairs})==len(pairs)
 match=re.search(r'Running unittests.*\(([^)]+)\)',result.stdout);assert match;binary=Path(match.group(1));assert binary.is_file();binary_sha256=sha(binary.read_bytes())
finally:
 (root/path).write_bytes(original);assert (root/path).read_bytes()==original;print('RESTORED exact candidate source.',flush=True)
summary={'less':sum(r['candidate_comparisons']<r['baseline_comparisons'] for r in rows),'same':sum(r['candidate_comparisons']==r['baseline_comparisons'] for r in rows),'more':sum(r['candidate_comparisons']>r['baseline_comparisons'] for r in rows)}
(out/'results.json').write_text(json.dumps({'scope':'instrumented ordering-comparator invocations, not elapsed time, CPU time, allocations or throughput','base':base,'source_path':path,'candidate_sha256':sha(original),'baseline_sha256':sha(old.encode()),'frozen_method_sha256':sha(old_body.encode()),'command':command,'exit':result.returncode,'log_sha256':sha(result.stdout.encode()),'binary_sha256':binary_sha256,'cases':rows,'summary':summary,'full_outputs':outputs,'candidate_restored':True},indent=2));print(json.dumps(summary),flush=True)
for r in rows:
 if r['experts']==256 and r['k'] in [1,128,255,256] and r['pattern'] in ['random','descending','ties']:print(r,flush=True)
