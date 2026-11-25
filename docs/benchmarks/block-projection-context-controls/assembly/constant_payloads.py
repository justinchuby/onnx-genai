from pathlib import Path
import struct,re,json,hashlib
out=Path(__file__).parent;src=Path('/tmp/nxrt-block-context-study');records={}
for label in ['baseline','candidate']:
 s=(out/(label+'-decode.asm')).read_text();page=int(re.search(r'adrp\s+x13, 0x([0-9a-f]+)',s)[1],16);low=int(re.search(r'add\s+x13, x13, #0x([0-9a-f]+)',s)[1],16);address=page+low
 data=(src/(label+'.binary')).read_bytes();assert struct.unpack_from('<I',data)[0]==0xfeedfacf
 count=struct.unpack_from('<I',data,16)[0];pos=32;payload=None
 for _ in range(count):
  cmd,size=struct.unpack_from('<II',data,pos)
  if cmd==0x19:
   vmaddr,vmsize,fileoff,filesize=struct.unpack_from('<QQQQ',data,pos+24)
   if vmaddr<=address and address+64<=vmaddr+filesize:payload=data[fileoff+address-vmaddr:fileoff+address-vmaddr+64]
  pos+=size
 assert payload is not None
 records[label]={'address':hex(address),'alignment_mod_128':address%128,'sha256':hashlib.sha256(payload).hexdigest(),'f32':list(struct.unpack('<16f',payload))}
assert records['baseline']['sha256']==records['candidate']['sha256'];(out/'constant-payloads.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
