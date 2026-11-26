"""Verify prefetch-entry behavior evidence and guard-mutation restoration."""
from pathlib import Path
import gzip,hashlib,json,subprocess,sys
folder=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
manifest=json.loads((folder/'manifest.json').read_text());actual={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts};assert actual==set(manifest)
for p,h in manifest.items():assert sha((folder/p).read_bytes())==h,p
n=folder/'negative';receipt=json.loads((n/'receipt.json').read_text());assert receipt['restored'] and receipt['expected_test_failure'] and not receipt['compile_error'] and receipt['exit']!=0
original=gzip.decompress((n/'original.rs.gz').read_bytes());mutant=gzip.decompress((n/'mutant.rs.gz').read_bytes());assert sha(original)==receipt['source_sha256'] and sha(mutant)==receipt['mutant_sha256']
guard='''        if !self.prefetch_eligibility().allows_pipeline() {
            return Ok(None);
        }
''';s=original.decode();assert s.count(guard)==1 and s.replace(guard,'').encode()==mutant
raw=gzip.decompress((n/'negative.log.gz').read_bytes());assert sha(raw)==receipt['log_sha256'];assert b'test result: FAILED. 0 passed; 1 failed; 0 ignored;' in raw and b'governed=false, foreign=false' in raw and b'error[E' not in raw
name='kernels::qmoe::tests::prefetch_scope_rejects_partial_banks_before_payload_lock_or_expert_inputs';assert (name+' ... FAILED').encode() in raw
v=folder/'validation';validation=json.loads((v/'validation.json').read_text());assert len(validation['results'])==6 and all(r['exit']==0 for r in validation['results']);assert validation['source_sha256'][receipt['source_path']]==sha(original)
required=['prefetch_scope_rejects_partial_banks_before_payload_lock_or_expert_inputs','prefetch_scope_streaming_funds_one_window_without_registry_lock','prefetch_scope_warm_hits_pin_payloads_without_new_issuer_calls','prefetch_pipeline_is_byte_and_stat_identical_to_serial','selected_prefetch_grants_refund_each_retained_payload_independently']
for r in validation['results']:
 log=gzip.decompress((v/(r['name']+'.log.gz')).read_bytes());assert sha(log)==r['log_sha256']
 if 'counts' in r:assert r['counts'] and all(int(p)>0 and int(f)==0 for p,f,i in r['counts'])
 if r['name'] in ['qmoe','qmoe-mlas']:
  assert r['counts']==[['65','0','1']]
  for test in required:assert ('test kernels::qmoe::tests::'+test+' ... ok').encode() in log
# Audit that this is a moved decision, with no other runtime/kernel changes.
base=gzip.decompress((folder/'baseline.rs.gz').read_bytes());t=s
a=t.index('    #[test]\n    fn prefetch_scope_rejects_partial_banks_before_payload_lock_or_expert_inputs() {');b=t.index('    #[test]\n    fn selected_prefetch_grants_refund_each_retained_payload_independently()',a);t=t[:a]+t[b:]
doc='''    /// Prepare an eligible pipeline scope while this operation freezes bank
    /// membership. Reject capacity-ineligible operations before consuming the
    /// selected experts, locking payloads, pinning hits or requesting grants.
''';assert t.count(doc)==1;t=t.replace(doc,'').replace(guard,'')
new='''            let prefetch_memory = if self.prefetch_experts && ordered.len() > 1 {'''
old='''            let prefetch_memory = if self.prefetch_experts
                && ordered.len() > 1
                && cache_operation.prefetch_eligibility().allows_pipeline()
            {''';assert t.count(new)==1;t=t.replace(new,old);assert t.encode()==base
prefix=s[:s.index('#[cfg(test)]\nmod tests {')];assert prefix.count('.prefetch_eligibility().allows_pipeline()')==1
method=prefix[prefix.index('    fn prefetch_memory('):];assert method.index(guard)<method.index('if !self.cache.memory.is_governed()')<method.index('for expert in experts')
seal=json.loads((folder/'source-seal.json').read_text());assert seal['final_sha256']==sha(original) and seal['runtime_change']=='move existing eligibility check from caller to prefetch funding entry'
if len(sys.argv)>1:
 root=Path(sys.argv[1]);assert (root/receipt['source_path']).read_bytes()==original
 assert subprocess.check_output(['git','-C',str(root.resolve()),'show',seal['base']+':'+receipt['source_path']])==base
print(f'Verified {len(manifest)} sealed payloads,65 QMoE tests per mode, exact guard-only control and source restoration.')
