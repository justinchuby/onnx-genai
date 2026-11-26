import pathlib,subprocess,json,hashlib,struct
root=pathlib.Path(__file__).resolve().parent
cases=json.loads((root/'manifest.json').read_text())['cases']
receipt=[]
for case in cases:
 name=case['name'];out=root/(name+'.webm')
 command=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-i',str(root/(name+'.ogg')),'-c:a','copy',str(out)]
 subprocess.run(command,check=True)
 pcm=subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-c:a','libopus','-i',str(out),'-f','f32le','-c:a','pcm_f32le','-'],check=True,stdout=subprocess.PIPE).stdout
 reference=(root/(name+'.reference.f32')).read_bytes()
 values=lambda data:struct.unpack('<'+'f'*(len(data)//4),data)
 a,b=values(pcm),values(reference)
 assert len(a)==len(b),(name,len(a),len(b))
 error=max(abs(x-y) for x,y in zip(a,b));assert error<=1/32768,(name,error)
 receipt.append(dict(name=name,webm_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),reference_sha256=hashlib.sha256(reference).hexdigest(),full_pcm_samples=len(a),max_error=error,command=command))
(root/'webm-manifest.json').write_text(json.dumps(dict(generator='actual FFmpeg stream copy of unchanged checked-in Opus packets; independent libopus PCM oracle',toolchain=subprocess.run(['ffmpeg','-version'],check=True,stdout=subprocess.PIPE,text=True).stdout.splitlines()[0],cases=receipt),indent=2)+'\n')
print(json.dumps(receipt,indent=2))
