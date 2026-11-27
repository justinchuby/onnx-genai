import gzip,hashlib,json,pathlib,subprocess,sys
out=pathlib.Path(sys.argv[1])
binary=out/'measured.binary'
tool='/opt/homebrew/opt/llvm/bin/llvm-nm';dis='/opt/homebrew/opt/llvm/bin/llvm-objdump'
text=subprocess.check_output([tool,'-n','--defined-only',str(binary)],text=True)
symbols=[]
for line in text.splitlines():
 words=line.split()
 if len(words)>=3 and words[1] in ['T','t']:
  try: symbols.append((int(words[0],16),words[2]))
  except ValueError:pass
records=[]
for address,name in symbols:
 if ('softmax30softmax_rows_serial_out_native' not in name and 'softmax_neon11softmax_row' not in name and not ('unroll_tests' in name and ('8original' in name or '13original_rows' in name or '21original_row_outlined' in name))):continue
 stop=min(a for a,n in symbols if a>address)
 cmd=[dis,'-d',f'--start-address={address}',f'--stop-address={stop}',str(binary)]
 raw=subprocess.check_output(cmd);stem=('original-rows' if '13original_rows' in name else 'original-row-outlined' if 'original_row_outlined' in name else 'original-wrapper' if 'original' in name else 'native-row' if 'softmax_neon11softmax_row' in name else 'native')
 if any(r['artifact']==stem+'.disasm.gz' for r in records): stem+=f'-{address:x}'
 (out/(stem+'.disasm.gz')).write_bytes(gzip.compress(raw,mtime=0))
 records.append({'symbol':name,'start':address,'stop':stop,'bytes':stop-address,'command':cmd,'raw_sha256':hashlib.sha256(raw).hexdigest(),'artifact':stem+'.disasm.gz'})
assert any('original' in r['symbol'] for r in records) and any('softmax_rows_serial_out_native' in r['symbol'] for r in records),records
assert len({r['artifact'] for r in records})==len(records)
for r in records: assert hashlib.sha256(gzip.decompress((out/r['artifact']).read_bytes())).hexdigest()==r['raw_sha256']
(out/'codegen.json').write_text(json.dumps({'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'llvm':subprocess.check_output([dis,'--version'],text=True).splitlines()[0],'records':records},indent=2))
print([(r['artifact'],r['bytes']) for r in records])
