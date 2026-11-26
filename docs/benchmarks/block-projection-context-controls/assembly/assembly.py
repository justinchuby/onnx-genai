from pathlib import Path
import subprocess,json,re,hashlib
out=Path(__file__).parent;src=Path('/tmp/nxrt-block-context-study');objdump=subprocess.check_output(['xcrun','-f','llvm-objdump'],text=True).strip();records={}
for label in ['baseline','candidate']:
 binary=src/(label+'.binary');symbols=subprocess.check_output(['nm','-n',str(binary)],text=True)
 needles={'decode':'decode_mxfp4_block','project':'DecodedProjectionPlan7projectKj4_','workspace':'BlockQuantizedMoEKernel35workspace_requirement_for_execution'}
 record={}
 for key,needle in needles.items():
  lines=[l for l in symbols.splitlines() if needle in l and l.split()[1].lower()=='t']
  if key=='project':lines=[l for l in lines if l.split()[-1].startswith('__RINvMs_')]
  if key=='decode':lines=[l for l in lines if l.split()[-1].endswith(needle)]
  if not lines:record[key]={'present':False};continue
  name=lines[0].split()[-1];r=subprocess.run([objdump,'--disassemble-symbols='+name,'--no-show-raw-insn',str(binary)],capture_output=True,text=True);assert r.returncode==0,r.stderr
  (out/(label+'-'+key+'.asm')).write_text(r.stdout)
  instructions=[l.strip().split('\t',1)[-1] for l in r.stdout.splitlines() if re.match(r'^\s*[0-9a-f]+:',l)]
  record[key]={'present':True,'instructions':len(instructions),'symbol':name,'sha256':hashlib.sha256(r.stdout.encode()).hexdigest()}
 records[label]=record
(out/'assembly.json').write_text(json.dumps(records,indent=2));print(json.dumps({label:{k:{kk:vv for kk,vv in v.items() if kk!='symbol'} for k,v in r.items()} for label,r in records.items()},indent=2))
