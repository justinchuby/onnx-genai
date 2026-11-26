#!/usr/bin/env python3
"""Execute one frozen native Session comparison cohort, retaining all outcomes."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import build

HERE, ROOT, LOCK = build.HERE, build.ROOT, build.LOCK

def oracle(case):
    import struct
    kind, nodes, rows, _ = case
    expected = []
    for row in range(rows):
        for column in range(32):
            x = 127 if column == 0 else ((row*17+column*13)%31)-15
            value = max(x,0) if kind=='relu' else x if column==0 else x+nodes
            expected.append(struct.unpack('<I',struct.pack('<f',float(value)))[0])
    h = 0xcbf29ce484222325
    for bits in expected:
        h = ((h^bits)*0x100000001b3)&0xffffffffffffffff
    return expected, f'{h:016x}'

def main():
    assert sys.platform=='darwin', 'whole-process sampler requires Darwin'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('build_receipt',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    receipt=json.loads(args.build_receipt.read_text())
    protocol=json.loads((HERE/'protocol.json').read_text())
    assert receipt['protocol']==protocol and receipt['candidate_restored']
    assert receipt['probe_sources']=={p:build.digest(HERE/p) for p in receipt['probe_sources']}
    for label,b in receipt['builds'].items():
        assert build.digest(b['binary'])==b['binary_sha256']
    out=args.output.resolve()
    assert not out.exists(), 'never overwrite or repeat a completed cohort'
    path=HERE.parent/'packed-activation-study/run.py'
    spec=importlib.util.spec_from_file_location('frozen_mach_sampler',path)
    sampler=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sampler)
    host=sampler.HostCPU()
    scripts=['protocol.json','run.py','analyze.py','verify.py','../packed-activation-study/run.py']
    hashes={p:build.digest(HERE/p) for p in scripts}
    env={k:v for k,v in os.environ.items() if not k.startswith(('NXRT_','ONNX_GENAI_','RAYON_','VECLIB_','MLAS_','OMP_'))}
    env.update(RAYON_NUM_THREADS='1',ONNX_GENAI_CPU_DECODE_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    with LOCK.open('r+') as lock:
        print(json.dumps({'status':'queued','pid':os.getpid(),'inode':os.fstat(lock.fileno()).st_ino}),flush=True)
        fcntl.flock(lock,fcntl.LOCK_EX)
        inode=os.fstat(lock.fileno()).st_ino
        assert inode==LOCK.stat().st_ino
        env.update(NXRT_STUDY_LOCK_FD=str(lock.fileno()),NXRT_STUDY_LOCK_INODE=str(inode))
        out.mkdir(parents=True)
        build_root=args.build_receipt.resolve().parent
        shutil.copytree(build_root/'sources',out/'build/sources')
        for name in ['build.json','baseline-cargo.jsonl','baseline-cargo.stderr','candidate-cargo.jsonl','candidate-cargo.stderr']:
            shutil.copy2(build_root/name,out/'build'/name)
        manifest={'protocol':protocol,'build_receipt':receipt,'build_receipt_sha256':build.digest(args.build_receipt),'harness_sources':hashes,'source_commit':build.command('git','rev-parse','HEAD'),'machine':build.command('sysctl','-n','machdep.cpu.brand_string'),'os':build.command('sw_vers'),'logical_cpus':host.cores,'environment':{k:env[k] for k in ['RAYON_NUM_THREADS','ONNX_GENAI_CPU_DECODE_THREADS','VECLIB_MAXIMUM_THREADS']},'lock':{'path':str(LOCK),'inode':inode,'inherited_by_children':True}}
        build.write(out/'manifest.json',manifest)
        def execute(label,case,calls,name):
            b=receipt['builds']['candidate' if label=='candidate' else 'baseline']
            cmd=[b['binary'],*map(str,case),str(calls)]
            with (out/f'{name}.stdout').open('w') as stdout, (out/f'{name}.stderr').open('w') as stderr:
                before=host.read(); start=time.monotonic()
                process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,pass_fds=(lock.fileno(),))
                pid,status,usage=os.wait4(process.pid,0)
                end=time.monotonic(); after=host.read()
                assert pid==process.pid
                process.returncode=os.waitstatus_to_exitcode(status)
            wall=end-start
            cpu=usage.ru_utime+usage.ru_stime
            stdout_text=(out/f'{name}.stdout').read_text()
            sample=None
            if process.returncode==0:
                sample=json.loads(stdout_text)
                bits,digest=oracle(case)
                assert sample['output_bits']==bits and sample['digest']==digest, f'{name}: independent full-output oracle rejected sample'
            busy=host.active(before,after)
            record={'name':name,'label':label,'case':case,'calls':calls,'command':cmd,'exit':process.returncode,'sample':sample,'complete_process_cpu_s':cpu,'complete_process_wall_s':wall,'absolute_process_peak_rss_bytes':usage.ru_maxrss,'cpu_user_s':usage.ru_utime,'cpu_sys_s':usage.ru_stime,'monotonic_start_s':start,'monotonic_end_s':end,'host_before':before,'host_after':after,'host_busy_cores':busy,'foreign_process_average_cores_raw':busy-cpu/wall,'foreign_process_average_cores':max(0,busy-cpu/wall),'lock_inode_before':inode,'lock_inode_after':LOCK.stat().st_ino,'stdout_sha256':build.digest(out/f'{name}.stdout'),'stderr_sha256':build.digest(out/f'{name}.stderr')}
            build.write(out/f'{name}.json',record)
            print(json.dumps({'status':'process_complete','name':name,'exit':record['exit'],'foreign_cores':record['foreign_process_average_cores']}),flush=True)
            return record
        pilots=[]; calls={}
        for index,case in enumerate(protocol['cases']):
            estimates=[]
            for label in ['baseline','candidate']:
                result=execute(label,case,0,f'cal-{index}-{label}')
                pilots.append(result)
                assert result['exit']==0, 'calibration failed; inspect retained output, do not start a replacement cohort'
                estimates.append(result['sample']['calibration_calls'])
            calls[index]=max(estimates)
        build.write(out/'calibrations.json',pilots)
        schedule=[]
        rng=random.Random(protocol['schedule_seed'])
        for rep in range(protocol['process_repetitions_per_label']):
            cases=list(enumerate(protocol['cases']));rng.shuffle(cases)
            for index,case in cases:
                labels=protocol['labels']
                offset=(rep+index)%len(labels)
                for label in labels[offset:]+labels[:offset]:
                    schedule.append({'name':f'run-{len(schedule):03}','rep':rep,'label':label,'case':case,'calls':calls[index]})
        build.write(out/'schedule.json',schedule)
        results=[]
        for planned in schedule:
            record=execute(planned['label'],planned['case'],planned['calls'],planned['name'])
            record['rep']=planned['rep'];results.append(record)
            build.write(out/'processes.json',results)
        assert hashes=={p:build.digest(HERE/p) for p in hashes}, 'harness changed during measurement'
        import analyze
        summary=analyze.summarize(out)
        build.write(out/'summary.json',summary)
        import verify
        verify.verify(out)
        print(json.dumps({'status':'complete','accepted_cells':sum(c['accepted'] for c in summary['cells']),'total_cells':len(summary['cells'])}),flush=True)

if __name__=='__main__':
    main()
