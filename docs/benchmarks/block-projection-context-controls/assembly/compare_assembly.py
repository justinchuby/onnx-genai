from pathlib import Path
import re,json
p=Path(__file__).parent;results={}
for kind in ['decode','project']:
 decoded={}
 for label in ['baseline','candidate']:
  s=(p/(label+'-'+kind+'.asm')).read_text();lines=[]
  header=re.search(r'^([0-9a-f]+) <',s,re.M);start=int(header[1],16)
  for l in s.splitlines():
   if not re.match(r'^\s*[0-9a-f]+:',l):continue
   address,inst=l.split(':',1);inst=inst.strip()
   # Internal branch offsets are relative; retain external symbol identity.
   def addr(m):
    value=int(m[1],16)
    return ('@'+hex(value-start)) if start<=value<start+0x10000 else '<absolute>'
   inst=re.sub(r'0x([0-9a-f]+)(?= <)',addr,inst)
   inst=re.sub(r'<([^>]+)>',lambda m:'<'+re.sub(r'0x[0-9a-f]+','OFFSET',m[1])+'>',inst)
   lines.append(inst)
  decoded[label]=lines
 differences=[{'instruction':i,'baseline':a,'candidate':b} for i,(a,b) in enumerate(zip(decoded['baseline'],decoded['candidate'])) if a!=b]
 results[kind]={'baseline_instructions':len(decoded['baseline']),'candidate_instructions':len(decoded['candidate']),'differences':differences}
(p/'assembly-comparison.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
