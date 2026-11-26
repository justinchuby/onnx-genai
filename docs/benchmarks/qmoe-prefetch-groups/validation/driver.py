import os,pathlib,re,subprocess,sys,json,hashlib
root=pathlib.Path('/tmp/nxrt-qmoe-workspace');out=pathlib.Path('/tmp/nxrt-prefetch-groups-validation');out.mkdir(exist_ok=True)
ort='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt'
env=os.environ|{'ORT_ROOT':ort,'ONNX_GENAI_ORT_LIB':ort+'/lib/libonnxruntime.1.32.0.dylib','NXRT_ORT_LIB_DIR':ort+'/lib','NXRT_REQUIRE_ORT_TESTS':'1','CARGO_INCREMENTAL':'0','CARGO_BUILD_JOBS':'1','CARGO_PROFILE_DEV_DEBUG':'0','CARGO_PROFILE_TEST_DEBUG':'0','CARGO_TARGET_DIR':'/tmp/nxrt-qmoe-main-target'}
env['PATH']='/opt/homebrew/Caskroom/miniconda/base/bin:'+env['PATH']
commands=[('list-qmoe',['test','--offline','-p','onnx-runtime-ep-cpu','--lib','kernels::qmoe::tests::','--','--list']),('qmoe',['test','--offline','-p','onnx-runtime-ep-cpu','--lib','kernels::qmoe::tests::']),('qmoe-mlas',['test','--offline','-p','onnx-runtime-ep-cpu','--lib','--features','mlas','kernels::qmoe::tests::']),('oracle',['test','--offline','-p','onnx-runtime-ep-cuda','--test','qmoe_cpu_oracle']),('clippy',['clippy','--offline','-p','onnx-runtime-ep-cpu','--all-targets','--','-D','warnings']),('clippy-mlas',['clippy','--offline','-p','onnx-runtime-ep-cpu','--all-targets','--features','mlas','--','-D','warnings'])]
paths=['crates/onnx-runtime-ep-cpu/src/kernels/qmoe.rs'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();source={p:sha(root/p) for p in paths};results=[]
for label,args in commands:
 print('RUN',label,flush=True)
 with (out/(label+'.log')).open('w') as f:r=subprocess.run(['cargo',*args],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
 s=(out/(label+'.log')).read_text();record={'name':label,'command':['cargo',*args],'exit':r.returncode,'log_sha256':sha(out/(label+'.log'))};results.append(record)
 print(s[-5000:] if r.returncode else '\n'.join(l for l in s.splitlines() if 'test result:' in l or 'Finished' in l or l.endswith(': test')),flush=True)
 if label=='list-qmoe' and not r.returncode:
  for name in ['prefetch_pipeline_is_byte_and_stat_identical_to_serial','cached_and_uncached_routes_and_logits_are_bit_identical','route_first_preserves_exact_accumulation_order_for_top_k','prefetch_scope_rejects_partial_banks_before_payload_lock_or_grants','prefetch_scope_streaming_funds_one_window_without_registry_lock','prefetch_scope_warm_hits_pin_payloads_without_new_issuer_calls','prefetch_group_driver_preserves_sparse_ids_and_two_live_payload_bound','prefetch_group_driver_propagates_callback_failures_and_retires_pending_weights','sparse_prefetch_expert_keys_match_serial_across_capacity_modes']:assert 'kernels::qmoe::tests::'+name+': test' in s
 if not label.startswith('list') and 'test' in args and not r.returncode:
  counts=re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;',s);assert counts and all(int(p)>0 and int(f)==0 for p,f,i in counts);record['counts']=counts
 (out/'validation.json').write_text(json.dumps({'source_sha256':source,'results':results},indent=2))
 if r.returncode:sys.exit(r.returncode)
assert source=={p:sha(root/p) for p in paths}
