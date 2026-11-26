#!/usr/bin/env python3
"""Apply the frozen gates to every cell; retain invalid cells as diagnostic."""
import json
from pathlib import Path
import statistics
import sys


def median(values):
    return statistics.median(values)


def analyze(directory):
    protocol = json.loads((directory / 'protocol.json').read_text())
    records = json.loads((directory / 'processes.json').read_text())
    assert (directory / 'complete.json').exists(), 'protocol did not finish; no performance conclusion'
    by_label = {record['label']: record for record in records}
    gates = protocol['gates']
    results = []
    for fmt, rows, k, n in protocol['cases']:
        name = f'{fmt}-{rows}'
        selected = [r for r in records if r.get('case') == [fmt, rows, k, n]
                    and not r['label'].startswith('calibrate-')]
        failures = []
        epochs = sorted({r.get('session', 'initial') for r in selected})
        if len(epochs) != 1:
            failures.append({'gate': 'single uninterrupted lock epoch', 'epochs': epochs})
        foreign = [r['label'] for r in selected if r['foreign_process_average_cores'] > gates['foreign_process_average_cores_max']]
        if foreign:
            failures.append({'gate': 'foreign process-average CPU', 'processes': foreign})
        low, high = gates['window_cpu_per_wall']
        inefficient = [(r['label'], index, w['cpu_per_wall'])
                       for r in selected for index, w in enumerate(r['result']['windows'])
                       if not w['setup_only'] and not low <= w['cpu_per_wall'] <= high]
        if inefficient:
            failures.append({'gate': 'window CPU/wall', 'windows': inefficient})
        samples = {0: [], 2: []}
        ratios, aa_deltas = [], []
        for rep in range(protocol['repetitions']):
            record = by_label[f'paired-{name}-{rep}']
            windows = record['result']['windows']
            nulls = {arm: [w['wall_s'] / w['calls'] for w in windows
                           if w['arm'] == arm and w['label'] in ['aa', 'aa_reverse']]
                     for arm in [0, 1]}
            aa_deltas.append(abs(median(nulls[0]) / median(nulls[1]) - 1) * 100)
            paired = {}
            for arm in [0, 2]:
                values = [w['wall_s'] / w['calls'] for w in windows
                          if w['arm'] == arm and w['label'] in ['ab', 'ba']]
                samples[arm].extend(values)
                paired[arm] = median(values)
            ratios.append(paired[0] / paired[2])
        if max(aa_deltas) > gates['aa_median_delta_pct_max']:
            failures.append({'gate': 'paired A/A', 'delta_pct_per_rep': aa_deltas})
        ranges = {arm: (max(values) - min(values)) / median(values) * 100
                  for arm, values in samples.items()}
        if max(ranges.values()) > gates['per_arm_range_pct_max']:
            failures.append({'gate': 'paired arm range', 'range_pct': ranges})
        controls = {}
        for arm in ['float32', 'q8_0']:
            null, fraction, double = [], [], []
            for rep in range(protocol['repetitions']):
                def cpu(control):
                    return by_label[f'process-{name}-{arm}-{control}-{rep}']['cpu_s']
                work, copy = cpu('work'), cpu('work-copy')
                setup, setup_double = cpu('setup'), cpu('setup-double')
                null.append(abs(work / copy - 1) * 100)
                fraction.append(setup / work)
                assert work > setup, (name, arm, rep, work, setup)
                double.append((cpu('work-double') - setup_double) / (work - setup))
            controls[arm] = {'aa_delta_pct': null, 'setup_fraction': fraction,
                             'double_marginal_cpu_ratio': double}
            if median(null) > gates['aa_process_delta_pct_max']:
                failures.append({'gate': 'process A/A', 'arm': arm, 'delta_pct': null})
            if max(fraction) > gates['setup_parent_cpu_fraction_max']:
                failures.append({'gate': 'setup CPU fraction', 'arm': arm, 'fractions': fraction})
            low, high = gates['double_marginal_cpu_ratio']
            if not all(low <= value <= high for value in double):
                failures.append({'gate': 'double-call CPU linearity', 'arm': arm, 'ratios': double})
        first = by_label[f'paired-{name}-0']['result']
        results.append({
            'format': fmt, 'rows': rows, 'k': k, 'n': n,
            'status': 'screened' if not failures else 'diagnostic', 'failures': failures,
            'calls_per_window': protocol['calls'][name], 'paired_repetitions': 3,
            'paired_aa_delta_pct': aa_deltas, 'paired_range_pct': ranges,
            'float32_us': {'median': median(samples[0]) * 1e6,
                           'min': min(samples[0]) * 1e6, 'max': max(samples[0]) * 1e6},
            'q8_us': {'median': median(samples[2]) * 1e6,
                      'min': min(samples[2]) * 1e6, 'max': max(samples[2]) * 1e6},
            'float32_over_q8_paired_ratios': ratios,
            'median_ratio': median(ratios), 'process_controls': controls,
            'declared_workspace_bytes': first['requirements'],
            'retained_physical_workspace_bytes': first['physical_bytes'],
            'warm_issuer_callbacks': {'claims': 0, 'returns': 0},
            'output_contract': 'Every output checked bitwise against independent integer coordinate oracle; exact-quantization fixture only',
        })
    # Keep all raw samples in processes.json, but never promote failed cells'
    # descriptive timings to a headline comparison in the canonical summary.
    for cell in results:
        if cell['status'] != 'screened':
            for key in ['float32_us', 'q8_us', 'float32_over_q8_paired_ratios', 'median_ratio']:
                cell.pop(key)
            cell['timing_comparison'] = None
    result = {'cells': results, 'interpretation': protocol['interpretation'],
              'policy': 'Screened cells pass the stated process-average/AA/range controls, without proving window-level absence of contention; diagnostic cells remain recorded and excluded from timing conclusions.'}
    (directory / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    assert len(sys.argv) == 2, 'usage: analyze.py results-directory'
    analyze(Path(sys.argv[1]))
