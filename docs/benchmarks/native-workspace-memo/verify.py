#!/usr/bin/env python3
"""Verify source receipts, independent outputs, raw arithmetic and exact schedule."""
import argparse
import gzip
import hashlib
import json
import math
import random
from pathlib import Path
import statistics
import analyze
import build
import run

def close(a,b):
    assert math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9),(a,b)

def sample_checks(s,case,calls,inode):
    assert s['case']==case and s['reps']==calls and s['threads']==1 and s['width']==32
    expected,digest=run.oracle(case)
    assert s['output_bits']==expected and s['digest']==digest, 'independent full-output oracle'
    expected_bytes=0 if case[0]=='relu' else ((case[2]*34+3)//4)*4 + case[2]*32*4 + (case[2]*32*4 if case[2]>1 else 0)
    assert s['funded_bytes']==expected_bytes, 'independent complete workspace geometry'
    assert len(s['allocation_identities'])==int(case[0]=='q8')
    assert s['host_authority_limit_bytes']==64<<20 and s['zero_byte_counter_control']
    assert s['issuer_delta']=={'claims':0,'returns':0}
    assert s['warm_issuer_counts']==s['issuer_counts_before_settlement']
    assert s['settled'] and s['funded_after_settlement']==s['manager_used_after_settlement']==0
    assert s['inherited_lock']['inode']==inode and not s['inherited_lock']['owned_by_probe']
    assert s['cold_cpu_s']>0 and s['cold_wall_s']>0 and s['warmup_calls']>0
    windows=s['windows'] if calls else s['calibration_observations']
    assert s['validation_runs_total']==2+len(windows)
    assert 'untimed' in s['validation_scope']
    if calls:
        assert s['calibration_calls'] is None and len(windows)==18
        expected_order=[]
        for index in range(9):
            expected_order += [(kind,index) for kind in (['single','double'] if index%2==0 else ['double','single'])]
        assert [(w['kind'],w['index']) for w in windows]==expected_order
    else:
        assert not s['windows'] and windows
        assert [w['outer_calls'] for w in windows]==[1<<i for i in range(len(windows))]
        assert windows[-1]['cpu_s']>=0.05 and all(w['cpu_s']<0.05 for w in windows[:-1])
        assert s['calibration_calls']==math.ceil(windows[-1]['outer_calls']*.15/windows[-1]['cpu_s'])
    for w in windows:
        outer=calls or w['outer_calls'];factor=2 if w['kind']=='double' else 1
        assert w['outer_calls']==outer and w['actual_calls']==outer*factor
        assert w['prebind_delta']==case[1]*w['actual_calls'], 'actual prebound dispatch cardinality'
        assert w['digest']==digest and w['issuer_delta']=={'claims':0,'returns':0} and w['untimed_validation_calls']==1
        assert all(math.isfinite(w[k]) and w[k]>0 for k in ['cpu_s','wall_s'])
    assert s['whole_before_settlement_cpu_s']>=sum(w['cpu_s'] for w in windows)

def verify(root):
    m=analyze.load(root/'manifest.json');p=m['protocol'];b=m['build_receipt']
    assert p==analyze.load(build.HERE/'protocol.json') and b['protocol']==p and b['candidate_restored']
    assert b==analyze.load(root/'build/build.json')
    assert m['harness_sources']=={path:build.digest(build.HERE/path) for path in m['harness_sources']}
    assert b['probe_sources']=={path:build.digest(build.HERE/path) for path in b['probe_sources']}
    for label,arm in b['builds'].items():
        assert arm['revision']==p[label] and arm['probe_artifact']['profile']['opt_level']=='3'
        assert not arm['probe_artifact']['profile']['test'] and 'mlas' not in arm['cpu_artifact']['features']
        for path,h in arm['runtime_sources'].items():
            assert hashlib.sha256(gzip.decompress((root/'build/sources'/label/(path+'.gz')).read_bytes())).hexdigest()==h
        if Path(arm['binary']).exists():
            assert build.digest(arm['binary'])==arm['binary_sha256']
    assert b['builds']['baseline']['binary_sha256']!=b['builds']['candidate']['binary_sha256']
    schedule=analyze.load(root/'schedule.json');records=analyze.load(root/'processes.json');pilots=analyze.load(root/'calibrations.json')
    assert len(pilots)==len(p['cases'])*2
    for pilot in pilots:
        assert pilot['exit']==0 and pilot['calls']==0
    estimates={tuple(case):max(x['sample']['calibration_calls'] for x in pilots if x['case']==case) for case in p['cases']}
    assert len(records)==len(schedule)==54
    expected_schedule=[];rng=random.Random(p['schedule_seed'])
    for rep in range(p['process_repetitions_per_label']):
        ordered=list(enumerate(p['cases']));rng.shuffle(ordered)
        for index,case in ordered:
            labels=p['labels'];offset=(rep+index)%len(labels)
            for label in labels[offset:]+labels[:offset]:
                expected_schedule.append({'name':f'run-{len(expected_schedule):03}','rep':rep,'label':label,'case':case,'calls':estimates[tuple(case)]})
    assert schedule==expected_schedule, 'frozen randomized Latin label schedule'
    groups={}
    for planned,r in zip(schedule,records):
        assert all(r[k]==planned[k] for k in ['name','rep','label','case','calls'])
        assert r['calls']==estimates[tuple(r['case'])]
        groups.setdefault(tuple(r['case']),[]).append((r['rep'],r['label']))
    for case in p['cases']:
        assert sorted(groups[tuple(case)])==sorted((rep,label) for rep in range(3) for label in p['labels'])
    for r in pilots+records:
        assert r['lock_inode_before']==r['lock_inode_after']==m['lock']['inode']
        assert r['stdout_sha256']==build.digest(root/f'{r["name"]}.stdout')
        assert r['stderr_sha256']==build.digest(root/f'{r["name"]}.stderr')
        assert r==analyze.load(root/f'{r["name"]}.json') or (r.get('rep') is not None and {k:v for k,v in r.items() if k!='rep'}==analyze.load(root/f'{r["name"]}.json'))
        arm='candidate' if r['label']=='candidate' else 'baseline'
        assert r['command']==[b['builds'][arm]['binary'],*map(str,r['case']),str(r['calls'])]
        close(r['complete_process_cpu_s'],r['cpu_user_s']+r['cpu_sys_s'])
        close(r['complete_process_wall_s'],r['monotonic_end_s']-r['monotonic_start_s'])
        delta=[(after-before)&0xffffffff for before,after in zip(r['host_before'],r['host_after'])]
        busy=m['logical_cpus']*(delta[0]+delta[1]+delta[3])/sum(delta)
        close(r['host_busy_cores'],busy)
        raw=busy-r['complete_process_cpu_s']/r['complete_process_wall_s']
        close(r['foreign_process_average_cores_raw'],raw)
        close(r['foreign_process_average_cores'],max(0,raw))
        if r['exit']==0:
            assert json.loads((root/f'{r["name"]}.stdout').read_text())==r['sample']
            sample_checks(r['sample'],r['case'],r['calls'],m['lock']['inode'])
    expected=analyze.summarize(root)
    assert expected==analyze.load(root/'summary.json'), 'summary must exactly reproduce frozen gates'
    for cell in expected['cells']:
        assert cell['accepted']==(not cell['failed_gates'])
        if not cell['accepted']:
            assert cell['ratios'] is None and cell['resolved_direction'] is None
    return expected

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('results',type=Path);args=parser.parse_args()
    result=verify(args.results)
    print(json.dumps({'verified':True,'cells':len(result['cells']),'accepted':sum(c['accepted'] for c in result['cells'])}))

if __name__=='__main__':
    main()
