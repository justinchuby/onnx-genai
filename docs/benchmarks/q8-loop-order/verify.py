#!/usr/bin/env python3
"""Independently audit raw receipts, fixture digests, and summary reproducibility."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def fixture_digest(q, rows, k, n):
    # Independent integer reconstruction of each logical product; do not call
    # Rust, parse its dot results, or reuse its activation converter.
    codes = 16 if q == 'q4' else 9
    offset = 8 if q == 'q4' else 4
    result = 0xcbf29ce484222325
    for row in range(rows):
        for feature in range(n):
            total = 0.
            for block in range((k + 31) // 32):
                integer = 0
                for lane in range(min(32, k - block * 32)):
                    x = 0 if (row + block + 1) % 5 == 0 else 127 if lane == 0 else (row * 17 + block * 11 + lane * 13) % 255 - 127
                    w = (feature * 7 + block * 11 + lane * 3) % codes - offset
                    integer += x * w
                total += integer * (2. ** ((row + block) % 3 - 2)) * [0., .125, .25, -.5][(feature + block) % 4]
            total += feature % 7 / 16. - .1875
            bits = struct.unpack('<I', struct.pack('<f', total))[0]
            result = ((result ^ bits) * 0x100000001b3) & ((1 << 64) - 1)
    return f'{result:016x}'


def verify(root):
    protocol, schedule, complete = [load(root / p) for p in ['protocol.json', 'schedule.json', 'complete.json']]
    assert schedule['protocol_sha256'] == digest(root / 'protocol.json')
    assert schedule['calibrations_sha256'] == digest(root / 'calibrations.json')
    assert complete['schedule_sha256'] == digest(root / 'schedule.json')
    assert complete['processes_sha256'] == digest(root / 'processes.json')
    assert complete['binary_sha256'] == protocol['binary_sha256']
    build = protocol['build_receipt']
    build_bytes = (json.dumps(build, indent=2, sort_keys=True) + '\n').encode()
    assert hashlib.sha256(build_bytes).hexdigest() == protocol['build_receipt_sha256']
    assert build['binary_sha256'] == protocol['binary_sha256']
    assert all(protocol['sources'][p] == d for p, d in build['sources'].items())
    assert build['controlled_environment']['CARGO_PROFILE_RELEASE_CODEGEN_UNITS'] == '1'
    assert 'full' in build['cargo_artifact']['features'] and 'mlas' not in build['cargo_artifact']['features']
    assert protocol['production_parent_before_study_sha256'] == 'ec9f1d6806e83099c673160ab71f76b2a987fbdc914d99d45961f11a37a67f67'
    assert protocol['pool_width'] == 1 and protocol['n'] == 256
    assert protocol['gates'] == {'foreign_process_average_cores_max': .5, 'cpu_per_wall': [.85, 1.15], 'aa_ratio': [.95, 1.05], 'double_ratio': [1.8, 2.2], 'setup_fraction_max': .1}
    assert protocol['cases'] == [[q, r, k] for q in ['q4', 'q8'] for r in [1, 3, 8, 32] for k in [32, 1024]]
    records, calibrations = [load(root / p) for p in ['processes.json', 'calibrations.json']]
    expected = {tuple(case): fixture_digest(*case, protocol['n']) for case in protocol['cases']}
    assert len(records) == complete['processes'] == 64 and len(calibrations) == 16, 'incomplete process cardinality; require all 64 confirmatory processes and 16 calibrations'
    for record in calibrations + records:
        prefix = record['label']
        stdout = root / (prefix + '.stdout')
        stderr = root / (prefix + '.stderr')
        assert digest(stdout) == record['stdout_sha256'] and digest(stderr) == record['stderr_sha256']
        text = stdout.read_text()
        assert 'kernels::packed_projection::loop_order_study::paired_loop_order_process' in text
        assert '1 passed; 0 failed; 0 ignored' in text and record['exit'] == 0
        marker = 'NXRT_LOOP_WINDOW ' if record['calls'] else 'NXRT_LOOP_CALIBRATION '
        raw = [json.loads(line.split(marker, 1)[1]) for line in text.splitlines() if marker in line]
        parsed = record['windows'] if record['calls'] else record['calibration']
        assert raw == parsed, f'{prefix}: raw windows differ from parsed record'
        for value in raw:
            assert value['digest'] == expected[tuple(record['case'])], f'{prefix}: independent oracle digest mismatch'
        deltas = [(b - a) & 0xffffffff for a, b in zip(record['host_ticks_before'], record['host_ticks_after'])]
        assert sum(deltas) > 0
        active = protocol['logical_cpus'] * (deltas[0] + deltas[1] + deltas[3]) / sum(deltas)
        foreign = max(0., active - record['process_cpu_s'] / record['process_wall_s'])
        assert abs(active - record['host_process_average_active_cores']) < 1e-12
        assert abs(foreign - record['foreign_process_average_cores']) < 1e-12
        assert record['lock_inode_before'] == record['lock_inode_after'] == protocol['lock']['inode']
    module_path = Path(__file__).with_name('analyze.py')
    spec = importlib.util.spec_from_file_location('loop_analyze', module_path)
    analyze = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyze)
    summary = analyze.summarize(root)
    assert json.dumps(summary, indent=2, sort_keys=True) + '\n' == (root / 'summary.json').read_text(), 'summary must reproduce byte-for-byte'
    assert summary['production_changed'] is False
    return {'processes': len(records), 'calibrations': len(calibrations), 'windows': sum(len(r['windows']) for r in records),
            'independent_oracle_fixtures': len(expected), 'qualified_cells': summary['qualified_cells'],
            'summary_reproduced': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.results), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
