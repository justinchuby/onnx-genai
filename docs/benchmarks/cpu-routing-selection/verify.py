"""Verify the complete non-timing study and test-only final change."""
from pathlib import Path
import ast,gzip,hashlib,json,re,struct,subprocess,sys
folder=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
manifest=json.loads((folder/'manifest.json').read_text());actual={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts};assert actual==set(manifest)
for p,h in manifest.items():assert sha((folder/p).read_bytes())==h,p
study=folder/'study';data=json.loads(gzip.decompress((study/'results.json.gz').read_bytes()));assert data['exit']==0 and data['candidate_restored'];assert re.fullmatch('[0-9a-f]{64}',data['binary_sha256'])
log=gzip.decompress((study/'probe.log.gz').read_bytes());assert sha(log)==data['log_sha256'] and b'1 passed; 0 failed' in log
rows=[];outputs={}
for e,k,p,b,c in re.findall(rb'CASE experts=(\d+) k=(\d+) pattern=(\w+) baseline_comparisons=(\d+) candidate_comparisons=(\d+)',log):rows.append(dict(experts=int(e),k=int(k),pattern=p.decode(),baseline_comparisons=int(b),candidate_comparisons=int(c)))
for e,k,p,pairs in re.findall(rb'OUTPUT experts=(\d+) k=(\d+) pattern=(\w+) pairs=(.*)',log):outputs[f'{int(e)}:{int(k)}:{p.decode()}']=[list(pair) for pair in ast.literal_eval(pairs.decode())]
assert rows==data['cases'] and outputs==data['full_outputs'] and len(rows)==len(outputs)==160
expected={(e,k,p) for e in [0,1,2,8,64,256,1024] for k in {0,1,2,e//2,max(0,e-1),e,e+3} for p in ['random','descending','ties','nonfinite']}
assert {(r['experts'],r['k'],r['pattern']) for r in rows}==expected
summary={'less':sum(r['candidate_comparisons']<r['baseline_comparisons'] for r in rows),'same':sum(r['candidate_comparisons']==r['baseline_comparisons'] for r in rows),'more':sum(r['candidate_comparisons']>r['baseline_comparisons'] for r in rows)}
assert summary==data['summary']=={'less':39,'same':80,'more':41}
controls=[r for r in rows if r['experts']>=2 and r['k']>=r['experts']];assert controls and all(r['baseline_comparisons']==r['candidate_comparisons']>0 for r in controls)
assert all(r['candidate_comparisons']==0 for r in rows if r['k']==0)
def bits(value):return struct.unpack('<I',struct.pack('<f',value))[0]
def total_key(word):return (~word)&0xffffffff if word&0x80000000 else word|0x80000000
for r in rows:
 e,k,p=r['experts'],r['k'],r['pattern'];words=[];seed=0xa713481b
 for i in range(e):
  if p=='random':
   seed=(seed*1664525+1013904223)&0xffffffff;value=struct.unpack('<f',struct.pack('<I',0x3f800000|(seed&0x007fffff)))[0]-1.;word=bits(value)
  elif p=='descending':word=bits(e-i)
  elif p=='ties':word=bits(1.)
  else:word=[0x7fc00032,0xffc00021,0x7f800000,0xff800000,0,0x80000000,0x7f7fffff,0xff7fffff][i%8]
  words.append(word)
 selected=sorted(range(e),key=lambda i:(-total_key(words[i]),i))[:min(k,e)]
 pairs=outputs[f'{e}:{k}:{p}'];assert [pair[0] for pair in pairs]==selected
 assert all(0<=word<=0xffffffff for _,word in pairs)
baseline=gzip.decompress((study/'baseline.rs.gz').read_bytes());candidate=gzip.decompress((study/'candidate.rs.gz').read_bytes());assert sha(baseline)==data['baseline_sha256'] and sha(candidate)==data['candidate_sha256']
base_text=baseline.decode();start=base_text.index('    pub(super) fn weights(');body=base_text[start:base_text.index('\n}\n\n// Frozen',start)];assert sha(body.encode())==data['frozen_method_sha256']
reference=(study/'reference.rs').read_text()
# Compare the complete copied method after removing only its diagnostic edits.
original='''        self.indices
            .sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]).then_with(|| a.cmp(&b)));'''
counted='''        self.indices.sort_unstable_by(|&a, &b| {
            ROUTE_ORDER_COMPARISONS.with(|count| count.set(count.get() + 1));
            logits[b].total_cmp(&logits[a]).then_with(|| a.cmp(&b))
        });'''
expected_reference='\n#[cfg(test)]\nthread_local! { static ROUTE_ORDER_COMPARISONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }\n#[cfg(test)]\nimpl RoutingScratch {\n'+body.replace(original,counted).replace('pub(super) fn weights(', 'fn fullsort_weights(')+'\n}\n'
assert reference==expected_reference
old='''        let compare =
            |&a: &usize, &b: &usize| logits[b].total_cmp(&logits[a]).then_with(|| a.cmp(&b));'''
new='''        let compare = |&a: &usize, &b: &usize| {
            ROUTE_ORDER_COMPARISONS.with(|count| count.set(count.get() + 1));
            logits[b].total_cmp(&logits[a]).then_with(|| a.cmp(&b))
        };'''
assert candidate.decode().count(old)==1
assert gzip.decompress((study/'instrumented.rs.gz').read_bytes()).decode()==candidate.decode().replace(old,new)+'\n'+reference+'\n'+(study/'probe.rs').read_text()
receipt=json.loads((folder/'final-source.json').read_text());final=gzip.decompress((folder/'final-source.rs.gz').read_bytes());assert not receipt['runtime_candidate_adopted'] and sha(final)==receipt['final_source_sha256']
text=final.decode();a=text.index('    #[test]\n    fn routing_matches_sorted_reference_for_permuted_bits_and_selection_boundaries() {');b=text.index('    #[test]\n    fn routing_keeps_total_order_ties_signed_zero_and_zero_denominator()',a);assert (text[:a]+text[b:]).encode()==baseline
if len(sys.argv)>1:
 root=Path(sys.argv[1]);assert (root/data['source_path']).read_bytes()==final
 assert subprocess.check_output(['git','-C',str(root.resolve()),'show',data['base']+':'+data['source_path']])==baseline
for label,count in [('candidate-validation',6),('final-validation',4)]:
 v=folder/label;validation=json.loads((v/'validation.json').read_text());assert len(validation['results'])==count and all(r['exit']==0 for r in validation['results'])
 expected_hash=data['candidate_sha256'] if label=='candidate-validation' else receipt['final_source_sha256'];assert validation['source_sha256'][data['source_path']]==expected_hash
 for r in validation['results']:
  raw=gzip.decompress((v/(r['name']+'.log.gz')).read_bytes());assert sha(raw)==r['log_sha256']
  if 'counts' in r:assert r['counts'] and all(int(p)>0 and int(f)==0 for p,f,i in r['counts'])
  if label=='final-validation' and r['name'].startswith('routing'):
   assert r['counts']==[['3','0','0']]
   assert b'test kernels::moe::routing_scratch_tests::routing_matches_sorted_reference_for_permuted_bits_and_selection_boundaries ... ok' in raw
print(f'Verified {len(manifest)} sealed payloads,160 complete ordering cases, independent prefixes and unchanged production code.')
