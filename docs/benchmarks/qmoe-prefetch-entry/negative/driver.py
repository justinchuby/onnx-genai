from pathlib import Path
import os,subprocess,json,gzip,hashlib,re
root=Path('/tmp/nxrt-qmoe-workspace');out=Path('/tmp/nxrt-prefetch-entry-negative');out.mkdir(exist_ok=True)
p='crates/onnx-runtime-ep-cpu/src/kernels/qmoe.rs';original=(root/p).read_bytes();sha=lambda b:hashlib.sha256(b).hexdigest();validation=json.loads(Path('/tmp/nxrt-prefetch-entry-validation/validation.json').read_text());assert len(validation['results'])==6 and all(r['exit']==0 for r in validation['results']);assert validation['source_sha256'][p]==sha(original)
needle='''        if !self.prefetch_eligibility().allows_pipeline() {
            return Ok(None);
        }
''';text=original.decode();assert text.count(needle)==1;mutant=text.replace(needle,'');(out/'mutant.rs.gz').write_bytes(gzip.compress(mutant.encode(),mtime=0));(out/'original.rs.gz').write_bytes(gzip.compress(original,mtime=0))
ort='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt';env=os.environ|{'ORT_ROOT':ort,'ONNX_GENAI_ORT_LIB':ort+'/lib/libonnxruntime.1.32.0.dylib','NXRT_ORT_LIB_DIR':ort+'/lib','NXRT_REQUIRE_ORT_TESTS':'1','CARGO_INCREMENTAL':'0','CARGO_BUILD_JOBS':'1','CARGO_PROFILE_DEV_DEBUG':'0','CARGO_PROFILE_TEST_DEBUG':'0','CARGO_TARGET_DIR':'/tmp/nxrt-qmoe-main-target'};env['PATH']='/opt/homebrew/Caskroom/miniconda/base/bin:'+env['PATH']
name='kernels::qmoe::tests::prefetch_scope_rejects_partial_banks_before_payload_lock_or_expert_inputs';command=['cargo','test','--offline','-p','onnx-runtime-ep-cpu','--lib',name,'--','--exact','--nocapture']
try:
 (root/p).write_text(mutant);print('RUN guard deletion negative control; expect the direct-scope test to fail',flush=True)
 result=subprocess.run(command,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);(out/'negative.log').write_text(result.stdout);print(result.stdout[-4000:],flush=True)
 assert result.returncode!=0 and 'test result: FAILED. 0 passed; 1 failed; 0 ignored;' in result.stdout
 assert name+' ... FAILED' in result.stdout and 'governed=false, foreign=false' in result.stdout
 assert 'error[E' not in result.stdout
finally:
 (root/p).write_bytes(original);assert (root/p).read_bytes()==original;print('RESTORED exact validated source.',flush=True)
(out/'receipt.json').write_text(json.dumps({'source_path':p,'source_sha256':sha(original),'mutant_sha256':sha(mutant.encode()),'mutation':'remove only the entry eligibility guard; caller already has scheduling-only condition','command':command,'exit':result.returncode,'expected_test_failure':True,'compile_error':False,'log_sha256':sha(result.stdout.encode()),'restored':True},indent=2))
