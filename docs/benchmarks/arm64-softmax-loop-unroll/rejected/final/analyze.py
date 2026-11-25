import csv,gzip,hashlib,io,itertools,json,math,pathlib,statistics,sys
folder=pathlib.Path(sys.argv[1] if len(sys.argv)>1 else '/tmp/nxrt-neon-unroll-final')
SHAPES=[(1,4),(1,7),(1,8),(1,16),(1,257),(32,1024),(128,4099),(16,65539)]
FIELDS=['sweep','width','serial','rows','cols','iters','rep','arm','cpu_ns','wall_ns','token']
def analyze(raw,metadata,validations):
 assert metadata['schema']==2 and metadata['repetitions']==15 and metadata['sweeps']==3
 assert metadata['arms']=={'0':'original_same_arithmetic','1':'unrolled_native','2':'unrolled_native_aa','3':'mlas','4':'original_outlined'}
 assert hashlib.sha256(raw).hexdigest()==metadata['raw_sha256']
 receipts=metadata['receipts'];assert len(receipts)==9
 assert {(r['sweep'],r['width']) for r in receipts}==set(itertools.product(range(3),[1,4,8]))
 for r in receipts:
  assert r['exit']==0 and r['cpu_s']>0 and r['wall_s']>0 and r['maxrss_bytes']>0
  assert r['fpcr_end_equality_assertion_passed'] is True and not(r['fpcr_start']&((1<<1)|(3<<22)|(1<<24)|(1<<25)))
 axes={(s,w,serial,n,d) for s,w,serial in itertools.product(range(3),[1,4,8],['true','false']) for n,d in SHAPES}
 seen=set();hashes={}
 for row in validations:
  sweep,width,serial,n,d,arm,digest,worst,workers=row
  sweep,width,n,d,arm,digest,workers=map(int,[sweep,width,n,d,arm,digest,workers]);axis=(sweep,width,serial,n,d);key=(*axis,arm)
  assert axis in axes and arm in range(5) and key not in seen
  assert digest>0 and math.isfinite(float(worst)) and 0<=float(worst)<=1e-6
  parallel=serial=='false' and width>1 and n>=2 and n*d>=16384
  assert workers>=0 and (workers>0)==parallel
  assert hashes.setdefault((n,d,arm),digest)==digest
  seen.add(key)
 assert seen=={(*axis,a) for axis in axes for a in range(5)}
 for n,d in SHAPES:assert hashes[n,d,0]==hashes[n,d,1]==hashes[n,d,2]==hashes[n,d,4]
 reader=csv.DictReader(io.StringIO(raw.decode()));assert reader.fieldnames==FIELDS
 seen=set();groups={};tokens={}
 for row in reader:
  assert None not in row and None not in row.values()
  row={k:v if k=='serial' else int(v) for k,v in row.items()}
  axis=tuple(row[k] for k in FIELDS[:5]);key=(*axis,row['rep'],row['arm'])
  assert axis in axes and row['arm'] in range(5) and row['rep'] in range(15) and key not in seen
  assert row['iters']==min(8192,max(1,1048576//(row['rows']*row['cols'])))
  assert row['cpu_ns']>0 and row['wall_ns']>0 and row['token']>0
  tokenkey=(row['rows'],row['cols'],row['arm'])
  assert tokens.setdefault(tokenkey,row['token'])==row['token']
  groups.setdefault(axis,{a:[] for a in range(5)})[row['arm']].append((row['cpu_ns']/row['iters'],row['wall_ns']/row['iters']))
  seen.add(key)
 assert seen=={(*axis,r,a) for axis in axes for r in range(15) for a in range(5)}
 for n,d in SHAPES:assert tokens[n,d,0]==tokens[n,d,1]==tokens[n,d,2]==tokens[n,d,4]
 cells=[]
 for axis,arms in sorted(groups.items()):
  cpu={str(a):statistics.median(x[0] for x in xs) for a,xs in arms.items()}
  wall={str(a):statistics.median(x[1] for x in xs) for a,xs in arms.items()}
  aa=cpu['1']/cpu['2'];waa=wall['1']/wall['2']
  cells.append(dict(zip(FIELDS[:5],axis))|{'cpu_ns':cpu,'wall_ns':wall,'cpu_aa':aa,'wall_aa':waa,'qualified':0.95<=aa<=1.05,'original_over_candidate_cpu':cpu['0']/cpu['1'],'mlas_over_candidate_cpu':cpu['3']/cpu['1'],'original_over_outline_cpu':cpu['0']/cpu['4'],'outline_over_candidate_cpu':cpu['4']/cpu['1']})
 return {'samples':len(seen),'validations':len(validations),'cells':cells}
raw=gzip.decompress((folder/'samples.csv.gz').read_bytes());metadata=json.loads((folder/'metadata.json').read_text());validations=json.loads((folder/'validations.json').read_text())
result=analyze(raw,metadata,validations)
(folder/'results.json').write_text(json.dumps(result,indent=2))
print('samples',result['samples'],'validations',result['validations'],'qualified',sum(c['qualified'] for c in result['cells']),len(result['cells']))
for width,serial in itertools.product([1,4,8],['true','false']):
 for n,d in SHAPES:
  cells=[c for c in result['cells'] if (c['width'],c['serial'],c['rows'],c['cols'])==(width,serial,n,d)]
  ratios=[c['original_over_candidate_cpu'] for c in cells];aa=[c['cpu_aa'] for c in cells]
  print(width,serial,n,d,'speedup',round(statistics.median(ratios),4),[round(x,4) for x in ratios],'AA',[round(x,4) for x in aa])
