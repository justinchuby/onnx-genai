#!/usr/bin/env python3
"""Recompute frozen control gates; rejected cells never publish a speed ratio."""
import argparse
import json
import math
from pathlib import Path
import statistics


def load(path):
    return json.loads(path.read_text())


def summarize(root):
    protocol, schedule, records = [load(root / p) for p in ['protocol.json', 'schedule.json', 'processes.json']]
    cases = protocol['cases']
    assert protocol['repetitions'] == 4 and len(cases) == 16
    assert len(records) == len(schedule['processes']) == len(cases) * 4
    gates = protocol['gates']
    result = []
    calibrations = {tuple(r['case']): r['calibration'][0] for r in load(root / 'calibrations.json')}
    assert len(calibrations) == len(cases)
    for i, (record, planned) in enumerate(zip(records, schedule['processes'])):
        assert record['label'] == f'run-{i:03}'
        assert all(record[key] == planned[key] for key in ['case', 'rep', 'calls', 'reverse'])
        assert record['calls'] == calibrations[tuple(record['case'])]['calls']
    for case in cases:
        selected = [r for r in records if r['case'] == case]
        assert sorted(r['rep'] for r in selected) == list(range(4))
        failures, wall_ratios, cpu_ratios = [], [], []
        for record in selected:
            prefix = record['label']
            assert record['exit'] == 0
            assert record['lock_inode_before'] == record['lock_inode_after'] == protocol['lock']['inode']
            sequence = [w['label'] for w in record['windows']]
            paired = ['block', 'row', 'row', 'block'] if record['reverse'] else ['row', 'block', 'block', 'row']
            assert sequence == paired + ['aa1', 'aa2', 'baa1', 'baa2', 'setup', 'double', 'bdouble']
            by_label = {}
            for window in record['windows']:
                label = window['label']
                by_label.setdefault(label, []).append(window)
                assert window['calls'] == record['calls']
                work = 0 if label == 'setup' else 2 if label in ['double', 'bdouble'] else 1
                assert window['projections'] == record['calls'] * work
                assert window['digest'] == calibrations[tuple(case)]['digest']
                assert math.isfinite(window['wall_s']) and window['wall_s'] > 0
                assert math.isfinite(window['cpu_s']) and window['cpu_s'] >= 0
                if work:
                    efficiency = window['cpu_s'] / window['wall_s']
                    if not gates['cpu_per_wall'][0] <= efficiency <= gates['cpu_per_wall'][1]:
                        failures.append(f'{prefix}/{label}: process CPU/wall outside gate')
            if record['foreign_process_average_cores'] > gates['foreign_process_average_cores_max']:
                failures.append(f'{prefix}: foreign process-average CPU outside gate')
            base = {arm: {clock: statistics.median(w[clock] for w in by_label[arm]) for clock in ['wall_s', 'cpu_s']} for arm in ['row', 'block']}
            for arm, first, second, double in [('row', 'aa1', 'aa2', 'double'), ('block', 'baa1', 'baa2', 'bdouble')]:
                for clock in ['wall_s', 'cpu_s']:
                    aa = by_label[second][0][clock] / by_label[first][0][clock]
                    doubled = by_label[double][0][clock] / base[arm][clock]
                    if not gates['aa_ratio'][0] <= aa <= gates['aa_ratio'][1]:
                        failures.append(f'{prefix}/{arm}/{clock}: A/A outside gate')
                    if not gates['double_ratio'][0] <= doubled <= gates['double_ratio'][1]:
                        failures.append(f'{prefix}/{arm}/{clock}: doubled work outside gate')
                setup = by_label['setup'][0]['wall_s'] / base[arm]['wall_s']
                if setup > gates['setup_fraction_max']:
                    failures.append(f'{prefix}/{arm}: checked-binding setup fraction outside gate')
            wall_ratios.append(base['block']['wall_s'] / base['row']['wall_s'])
            cpu_ratios.append(base['block']['cpu_s'] / base['row']['cpu_s'])
        def stats(values):
            return {'median': statistics.median(values), 'min': min(values), 'max': max(values), 'n': len(values)}
        result.append({'format': case[0], 'rows': case[1], 'K': case[2], 'N': protocol['n'],
                       'classification': 'diagnostic' if failures else 'qualified under frozen coarse controls',
                       'gate_failures': sorted(failures),
                       'candidate_over_row_wall': None if failures else stats(wall_ratios),
                       'candidate_over_row_cpu': None if failures else stats(cpu_ratios)})
    return {'scope': protocol['scope'], 'production_changed': False, 'processes': len(records),
            'windows': sum(len(r['windows']) for r in records), 'cells': result,
            'qualified_cells': sum(not r['gate_failures'] for r in result),
            'decision': 'Keep production unchanged. This is a private-projection loop experiment; operator/model and multiworker effects are not established.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.results), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
