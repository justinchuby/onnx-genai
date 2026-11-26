from pathlib import Path
import json,hashlib,gzip,math,statistics,itertools,sys,struct
ROOT=Path('/tmp/nxrt-conv-gap-study')
LABELS={'production':('default','native'),'production-aa':('default','native'),'generic':('linked','native'),'generic-aa':('linked','native'),'reference':('linked','reference'),'reference-aa':('linked','reference')}
PHASES=['reuse','new'];KINDS=['setup','work','double']
CASES=[([1,8,16,16],[16,8,3,3],1,1,1,1,False),([1,8,16,16],[16,8,3,3],1,2,1,2,False),([2,16,12,12],[24,8,3,3],2,1,1,1,False),([1,16,16,16],[16,1,3,3],16,1,1,1,False),([1,256,4,4],[128,256,1,1],1,1,1,0,False),([1,8,1,128],[16,8,1,5],1,1,2,2,True)]
sha=lambda b:hashlib.sha256(b).hexdigest()
def values(n,seed):
 state=seed;out=[]
 for _ in range(n):
  state=(state*6364136223846793005+1442695040888963407)&((1<<64)-1)
  v=struct.unpack('<f',struct.pack('<f',((state>>32)/((1<<32)-1)-0.5)))[0]*0.5;out.append(v)
 return out

def oracle(case):
 xs,ws,g,d,s,p,rank3=CASES[case];b,ic,ih,iw=xs;oc,ig,kh,kw=ws;oh=1 if rank3 else (ih+2*p-((kh-1)*d+1))//s+1;ow=(iw+2*p-((kw-1)*d+1))//s+1
 x=values(math.prod(xs),71);w=values(math.prod(ws),101);bias=values(oc,37);out=[]
 for bb in range(b):
  for cc in range(oc):
   gg=cc//(oc//g)
   for yy in range(oh):
    for zz in range(ow):
     total=bias[cc]
     for i in range(ig):
      for ky in range(kh):
       py=0 if rank3 else yy*s+ky*d-p
       if not 0<=py<ih:continue
       for kx in range(kw):
        px=zz*s+kx*d-p
        if 0<=px<iw:total+=x[((bb*ic+gg*ig+i)*ih+py)*iw+px]*w[((cc*ig+i)*kh+ky)*kw+kx]
     out.append(total)
 return out
