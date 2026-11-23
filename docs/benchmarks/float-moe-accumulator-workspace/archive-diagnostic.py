import pathlib,json,shutil,hashlib
root=pathlib.Path('/tmp/nxrt-moe-accumulator-perf');src=root/'diagnostic';dst=pathlib.Path('/tmp/nxrt-float-moe-accumulator/docs/benchmarks/float-moe-accumulator-workspace');out=dst/'diagnostic';out.mkdir(exist_ok=True)
runs=json.loads((src/'runs.json').read_text());summary=json.loads((src/'summary.json').read_text());assert len(runs)==72 and len(summary)==4
allruns=json.loads((root/'runs.json').read_text())+runs;assert len(allruns)==144
for c in summary:
 rows=[r for r in allruns if r['shape']==c['shape']];assert len(rows)==27 and len({r['output_sha256'] for r in rows})==1 and len({tuple(r['argmax']) for r in rows})==1
for name in ['protocol.json','runs.json','summary.json']:shutil.copy2(src/name,out/name)
shutil.copy2(root/'diagnostic.py',out/'driver.py');shutil.copy2(__file__,dst/'archive-diagnostic.py')
p=dst/'README.md';s=p.read_text();section_idx=s.index('## Correctness and validation')
lines=['## Fixed supplementary diagnostic','','The original serial split-SwiGLU control failed its 5% gate, and parallel cells had noticeable null drift. One prescribed diagnostic kept the source and binaries unchanged, doubled repetitions and ran all six arm permutations across the four affected shapes (72 further processes, six per arm). Hashes were verified before running. The original 72 processes remain above and in runs.json; diagnostic results do not replace them. Complete outputs and argmaxes match across all 144 formal processes. No further retries were prescribed.','','| rows / H / I / threads / variant | CPU after vs base | null vs base | wall after vs base | control after / null | duration / control gate |','|---|---:|---:|---:|---:|---|']
for c in summary:
 m,h,i,t,g,v=c['shape'];a=c['arms'];lines.append(f"| {m} / {h} / {i} / {t} / {v} | {(c['after_over_baseline']-1)*100:+.2f}% | {(c['null_over_baseline']-1)*100:+.2f}% | {(a['after']['wall_us_median']/a['baseline']['wall_us_median']-1)*100:+.2f}% | {c['control_after_over_baseline']:.4f} / {c['control_null_over_baseline']:.4f} | {c['duration_pass']} / {c['control_5pct_pass']} |")
lines+=['','Read timing differences alongside the null/control and per-process ranges in diagnostic/summary.json. Zero warm workspace issuer calls is a tested resource property; it does not establish exactly zero overall inference overhead.','']
s=s[:section_idx]+'\n'.join(lines)+'\n'+s[section_idx:]
s=s.replace('No measured failing timing cell was rerun to obtain a better result.','The fixed supplementary diagnostic is reported separately above; neither study was repeated until a favourable result appeared.')
p.write_text(s)
print(json.dumps([{'shape':c['shape'],'after_pct':100*(c['after_over_baseline']-1),'null_pct':100*(c['null_over_baseline']-1),'control_pass':c['control_5pct_pass'],'duration_pass':c['duration_pass']} for c in summary],indent=2))
