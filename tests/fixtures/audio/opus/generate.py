import array,collections,json,math,os,pathlib,shutil,struct,subprocess,wave
root=pathlib.Path(__file__).resolve().parent;work=root
# Fixed before any observed result; never widen to hide decoder differences.
TOLERANCE=1/32768

def run(args,**kw):
 return subprocess.run(args,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,**kw)
def ogg(path):
 data=path.read_bytes();offset=0;partial=bytearray();packets=[];granule=None
 while offset<len(data):
  assert data[offset:offset+4]==b'OggS'
  n=data[offset+26];segments=data[offset+27:offset+27+n];pos=offset+27+n
  gp=struct.unpack_from('<Q',data,offset+6)[0]
  if gp!=(1<<64)-1:granule=gp
  for size in segments:
   partial.extend(data[pos:pos+size]);pos+=size
   if size<255:packets.append(bytes(partial));partial.clear()
  offset=pos
 assert not partial and len(packets)>2 and packets[0][:8]==b'OpusHead' and packets[1][:8]==b'OpusTags'
 header=packets[0];pre_skip=struct.unpack_from('<H',header,10)[0];channels=header[9];family=header[18]
 counts=collections.Counter(('silk' if (p[0]>>3)<12 else 'hybrid' if (p[0]>>3)<16 else 'celt') for p in packets[2:] if p)
 return pre_skip,granule,channels,family,dict(counts),len(packets)-2

def wave_source(channels):
 path=work/f'source-{channels}.wav'
 samples=11520
 with wave.open(str(path),'wb') as wav:
  wav.setparams((channels,2,48000,samples,'NONE','not compressed'))
  values=[]
  for n in range(samples):
   t=n/48000
   for c in range(channels):
    # Distinct channel mixtures plus evolving envelope: reproducible, non-silent.
    env=.55+.3*math.sin(2*math.pi*3*t)
    f=180+170*c
    v=env*(.20*math.sin(2*math.pi*f*t)+.08*math.sin(2*math.pi*(2*f+37)*t)+.025*math.sin(2*math.pi*(3200+233*c)*t))
    values.append(round(v*32767))
  wav.writeframes(struct.pack('<'+'h'*len(values),*values))
 return path

def floats(path):
 values=array.array('f');values.frombytes(path.read_bytes());return values

# Generate all fixtures first and prove their actual encoded modes.
fixtures=[]
for channels in [1,2]:
 source=wave_source(channels)
 configs=[('silk','voip','12k' if channels==1 else '20k','8000'),('hybrid','voip','24k' if channels==1 else '40k','12000'),('celt','lowdelay','64k' if channels==1 else '96k','20000')]
 for intended,application,bitrate,cutoff in configs:
  path=work/f'{intended}-{channels}.ogg'
  run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-i',str(source),'-c:a','libopus','-application',application,'-b:a',bitrate,'-cutoff',cutoff,'-frame_duration','20','-vbr','off',str(path)])
  info=ogg(path);fixtures.append((path,info,intended))
for channels in [3,6]:
 source=wave_source(channels);path=work/f'mapping1-{channels}.ogg'
 run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-channel_layout','3.0' if channels==3 else '5.1','-i',str(source),'-c:a','libopus','-application','lowdelay','-b:a','192k','-mapping_family','1','-frame_duration','20','-vbr','off',str(path)])
 fixtures.append((path,ogg(path),'celt'))
for path,info,intended in fixtures:print(json.dumps({'fixture':path.name,'preskip':info[0],'end_granule':info[1],'channels':info[2],'mapping':info[3],'toc_modes':info[4],'intended':intended}),flush=True)
if any(info[4].get(intended,0)==0 for path,info,intended in fixtures):
 raise RuntimeError('fixture coverage incomplete: intended TOC mode absent; adjust generation, not numerical oracle')
# Archive independent libopus presented PCM before invoking candidate.
for path,info,intended in fixtures:
 run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-c:a','libopus','-i',str(path),'-c:a','pcm_f32le','-f','f32le',str(path.with_suffix('.reference.f32'))])

# References preserve canonical full-channel PCM, so downmix cannot hide routing bugs.
# The checked-in manifest records one measured toolchain/fixture receipt; refreshed
# binaries can differ in container serials/tags across encoder versions.
for path in work.glob('source-*.wav'):
 path.unlink()

# Same Opus packets in other containers prove explicit capability rejection.
for suffix in ['webm','mp4']:
 run(['ffmpeg','-nostdin','-v','error','-y','-i',str(work/'celt-2.ogg'),'-c:a','copy',str(work/f'unsupported-opus.{suffix}')])
