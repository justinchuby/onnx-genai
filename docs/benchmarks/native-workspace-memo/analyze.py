#!/usr/bin/env python3
"""Apply prespecified gates without publishing ratios for rejected cells."""
import argparse
import json
import math
from pathlib import Path
import statistics


def load(path):
    return json.loads(Path(path).read_text())

def median(xs):
    return statistics.median(xs)

def inside(value,bounds):
    return bounds[0] <= value <= bounds[1]

def summarize(root):
    manifest=load(root/'manifest.json');p=manifest['protocol'];g=p['gates']
    records=load(root/'processes.json');schedule=load(root/'schedule.json')
    assert len(records)==len(schedule)==len(p['cases'])*len(p['labels'])*p['process_repetitions_per_label']
    cells=[]
    for case in p['cases']:
        selected=[r for r in records if r['case']==case]
        failures=[];values={label:{'cpu':[],'wall':[]} for label in p['labels']}
        for r in selected:
            name=r['name']
            if r['exit']!=0:
                failures.append(f'{name}: process exit {r["exit"]}');continue
            s=r['sample'];windows=s['windows']
            assert len(windows)==18
            if r['foreign_process_average_cores']>g['foreign_process_average_cores_max']:
                failures.append(f'{name}: foreign process-average CPU')
            measured=sum(w['cpu_s'] for w in windows)
            setup=(r['complete_process_cpu_s']-measured)/r['complete_process_cpu_s']
            assert setup>=-0.001, 'measured CPU cannot exceed complete child CPU'
            if setup>g['setup_fraction_max']:
                failures.append(f'{name}: whole-process setup fraction')
            for w in windows:
                assert all(math.isfinite(w[k]) and w[k]>0 for k in ['cpu_s','wall_s'])
                if w['cpu_s']<g['minimum_window_cpu_s']:
                    failures.append(f'{name}/{w["kind"]}/{w["index"]}: short CPU window')
                if not inside(w['cpu_s']/w['wall_s'],g['cpu_per_wall']):
                    failures.append(f'{name}/{w["kind"]}/{w["index"]}: CPU/wall')
            for clock,field in [('cpu','cpu_s'),('wall','wall_s')]:
                single=[w[field] for w in windows if w['kind']=='single']
                double=[w[field] for w in windows if w['kind']=='double']
                assert len(single)==len(double)==9
                if not inside(median(double)/median(single),g['double_over_single']):
                    failures.append(f'{name}/{clock}: double-call control')
                if not inside(median(single[::2])/median(single[1::2]),g['within_process_aa']):
                    failures.append(f'{name}/{clock}: within-process A/A')
                values[r['label']][clock].append(median(single)/r['calls'])
        for label in p['labels']:
            for clock in ['cpu','wall']:
                xs=values[label][clock]
                if len(xs)!=p['process_repetitions_per_label']:
                    failures.append(f'{label}/{clock}: incomplete process cohort')
                elif max(xs)/min(xs)>g['process_median_spread_max']:
                    failures.append(f'{label}/{clock}: process spread')
        for clock in ['cpu','wall']:
            a=values['baseline'][clock];n=values['null'][clock]
            if a and n and not inside(median(n)/median(a),g['baseline_null_ratio']):
                failures.append(f'{clock}: identical-binary null drift')
        accepted=not failures
        cell={'case':case,'accepted':accepted,'failed_gates':failures,'process_medians_seconds_per_call':values,'ratios':None,'resolved_direction':None}
        if accepted:
            cell['ratios']={clock:median(values['candidate'][clock])/median(values['baseline'][clock]) for clock in ['cpu','wall']}
            cell['ranges_seconds_per_call']={label:{clock:[min(xs),max(xs)] for clock,xs in clocks.items()} for label,clocks in values.items()}
            directions=[]
            for clock in ['cpu','wall']:
                a=values['baseline'][clock];b=values['candidate'][clock]
                directions.append('lower' if max(b)<min(a) else 'higher' if min(b)>max(a) else 'overlapping')
            cell['resolved_direction']=directions[0] if directions[0]==directions[1] and directions[0]!='overlapping' else 'unresolved'
        cells.append(cell)
    return {'schema_version':1,'cells':cells,'timed_public_calls':sum(w['actual_calls'] for r in records if r['sample'] for w in r['sample']['windows']),'timed_node_dispatches':sum(w['prebind_delta'] for r in records if r['sample'] for w in r['sample']['windows']),'scope':p['scope'],'no_cuda_execution':True}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('results',type=Path);args=parser.parse_args()
    print(json.dumps(summarize(args.results),indent=2,sort_keys=True))

if __name__=='__main__':
    main()
