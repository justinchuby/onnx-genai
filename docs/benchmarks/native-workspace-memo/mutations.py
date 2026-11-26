#!/usr/bin/env python3
"""Challenge the verifier with checksum-repaired semantic corruptions."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import analyze
import build
import verify


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('results',type=Path);args=parser.parse_args()
    verify.verify(args.results)
    outcomes=[]
    for kind in ['dispatch_count','complete_output','derived_foreign_cpu']:
        with tempfile.TemporaryDirectory(prefix='nxrt-memo-mutant-') as temporary:
            root=Path(temporary)/'results';shutil.copytree(args.results,root)
            records=analyze.load(root/'processes.json');r=next(r for r in records if r['exit']==0)
            if kind=='dispatch_count':
                r['sample']['windows'][0]['prebind_delta']+=1
            elif kind=='complete_output':
                s=r['sample'];s['output_bits'][0]^=1;h=0xcbf29ce484222325
                for bits in s['output_bits']:h=((h^bits)*0x100000001b3)&0xffffffffffffffff
                s['digest']=f'{h:016x}'
                for window in s['windows']:window['digest']=s['digest']
            else:
                r['host_busy_cores']+=0.25
                r['foreign_process_average_cores_raw']=r['host_busy_cores']-r['complete_process_cpu_s']/r['complete_process_wall_s']
                r['foreign_process_average_cores']=max(0,r['foreign_process_average_cores_raw'])
            stdout=root/f'{r["name"]}.stdout'
            stdout.write_text(json.dumps(r['sample'],sort_keys=True)+'\n')
            r['stdout_sha256']=hashlib.sha256(stdout.read_bytes()).hexdigest()
            build.write(root/f'{r["name"]}.json',{k:v for k,v in r.items() if k!='rep'})
            build.write(root/'processes.json',records)
            build.write(root/'summary.json',analyze.summarize(root))
            rejected=False
            try:
                verify.verify(root)
            except AssertionError as error:
                rejected=True;message=str(error)
            assert rejected,f'{kind}: verifier accepted semantic corruption after checksum repair'
            outcomes.append({'mutation':kind,'checksums_and_summary_repaired':True,'rejected':True,'message':message})
    print(json.dumps(outcomes,indent=2,sort_keys=True))

if __name__=='__main__':
    main()
