#!/usr/bin/env python3
"""Verify the full exported cohort independently of timing qualification."""
import hashlib
import json
from pathlib import Path
import struct
import sys


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def oracle_digest(fmt, rows, k, n):
    value = 0xcbf29ce484222325
    for row in range(rows):
        for feature in range(n):
            total = feature % 7
            for index in range(k):
                activation = 127 if index % 32 == 0 else (row * 17 + index * 13) % 255 - 127
                block, lane = divmod(index, 32)
                code = feature * 7 + block * 3 + lane * 11 + lane // 16 * 5
                weight = code % (16 if fmt == 'q4' else 255) - (8 if fmt == 'q4' else 127)
                total += activation * weight
            bits, = struct.unpack('<I', struct.pack('<f', total / 256))
            value = ((value ^ bits) * 0x100000001b3) & ((1 << 64) - 1)
    return f'{value:016x}'


def verify(directory):
    load = lambda name: json.loads((directory / name).read_text())
    protocol, records, complete = load('protocol.json'), load('processes.json'), load('complete.json')
    require(complete['all_launched_processes_succeeded'], 'cohort completion flag rejected')
    require(protocol['repetitions'] == 3, 'expected the frozen three repetitions')
    expected = {}
    for fmt, rows, k, n in protocol['cases']:
        name = f'{fmt}-{rows}'
        expected[f'partial-{name}'] = ([fmt, rows, 1003, n], 'check', 2)
        calls = protocol['calls'][name]
        for rep in range(3):
            expected[f'paired-{name}-{rep}'] = ([fmt, rows, k, n], 'paired', calls)
            for arm in ['float32', 'q8_0']:
                for control in ['work', 'work-copy', 'setup', 'work-double', 'setup-double']:
                    mode = 'setup-' + arm if control.startswith('setup') else arm
                    expected[f'process-{name}-{arm}-{control}-{rep}'] = (
                        [fmt, rows, k, n], mode, calls * (2 if control.endswith('double') else 1))
    require(len(records) == complete['processes'] == len(expected) == 136, 'missing or extra process in the complete 136-request cohort')
    require(len({r['label'] for r in records}) == len(records), 'duplicate process label would hide a repeated sample')
    require({r['label'] for r in records} == set(expected), 'process identities differ from the frozen workload matrix')
    require(len({r.get('session', 'initial') for r in records}) == 1, 'cohort crosses a lock interruption')
    digests = {}
    windows = calls_total = 0
    for record in records:
        label = record['label']
        case, mode, calls = expected[label]
        require(record['case'] == case and record['mode'] == mode and record['calls'] == calls,
                f'{label}: request differs from frozen geometry/mode/calls')
        require(record['exit_code'] == 0, f'{label}: child did not execute successfully')
        result = record['result']
        fmt, rows, k, n = case
        require([result[key] for key in ['format', 'rows', 'k', 'n']] == case, f'{label}: result geometry changed')
        require(result['settled'] and result['zero_byte_counter_control'] and result['dense_expansion_positive_control'],
                f'{label}: missing resource settlement or positive observation controls')
        require(min(result['warm_issuer_counts']) >= 1, f'{label}: issuer positive control was not observed')
        req = result['requirements']
        q8_bytes = ((rows * ((k + 31) // 32) * 34 + 3) // 4) * 4
        if rows > 1 and n > 1:
            q8_bytes += rows * n * 4
        require(len(req) == 3 and req[0] == req[1] and req[2] == q8_bytes,
                f'{label}: projection scratch does not match the two arithmetic contracts')
        require(result['peak_workspace_bytes'] == result['physical_bytes'] == max(req),
                f'{label}: retained physical lane differs from the unified whole-operation peak')
        require(bool(result['allocation_identity']), f'{label}: missing real managed allocation identity')
        if mode == 'paired':
            sequence = [(group, arm) for group, arms in protocol['paired_sequences'].items() for arm in arms]
        elif mode == 'check':
            sequence = [('check', 0), ('check', 2)]
        else:
            sequence = [('setup' if mode.startswith('setup') else 'work', 2 if 'q8_0' in mode else 0)]
        require(result['window_count'] == len(result['windows']) == len(sequence), f'{label}: missing window')
        key = tuple(case)
        if key not in digests:
            digests[key] = oracle_digest(*case)
        for index, (window, (group, arm)) in enumerate(zip(result['windows'], sequence)):
            context = f'{label} window {index}'
            require(window['label'] == group and window['arm'] == arm and window['calls'] == calls,
                    f'{context}: window identity/order/call count changed')
            require(window['setup_only'] == mode.startswith('setup'), f'{context}: setup control executed the wrong path')
            require(window['claims'] == window['returns'] == 0, f'{context}: warm issuer callback observed')
            require(window['output_elements_checked'] == rows * n and window['output_digest'] == digests[key],
                    f'{context}: complete output check disagrees with independent Python integer oracle')
            windows += 1
            calls_total += calls
    expected_calls = sum(count * (16 if mode == 'paired' else 2 if mode == 'check' else 1) for _, mode, count in expected.values())
    require(windows == 320 and calls_total == expected_calls, 'window or invocation cardinality changed')
    provenance = load('provenance.json')
    here = Path(__file__).resolve().parent
    for path, key in [('run.py', 'runner_sha256'), ('probe/src/main.rs', 'probe_sha256'),
                      ('probe/Cargo.toml', 'manifest_sha256'), ('probe/Cargo.lock', 'lockfile_sha256')]:
        require(hashlib.sha256((here / path).read_bytes()).hexdigest() == provenance[key], f'{path}: differs from measured source')
    summary = load('summary.json')
    require(len(summary['cells']) == 4, 'summary lost an experimental cell')
    for cell in summary['cells']:
        if cell['status'] != 'screened':
            require(cell['failures'] and cell['timing_comparison'] is None and 'median_ratio' not in cell,
                    'diagnostic cell exposed a timing comparison')
    return {'processes': len(records), 'windows': windows, 'calls': calls_total,
            'independent_oracle_cases': len(digests), 'all_resource_and_output_checks_passed': True}


if __name__ == '__main__':
    try:
        require(len(sys.argv) == 2, 'usage: verify.py exported-results-directory')
        print(json.dumps(verify(Path(sys.argv[1])), indent=2))
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit(f'Packed activation cohort verification rejected: {error}. Inspect the frozen protocol and original process records; preserve rejected evidence instead of editing observations to pass.')
