#!/usr/bin/env python3
"""Falsify acceptance with forged summaries, incomplete runs and fake digests."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from verify import verify


def load(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    original = verify(args.results)
    outcomes = []
    with tempfile.TemporaryDirectory(prefix='nxrt-loop-verifier-') as temp:
        for mutation in ['fake-summary', 'missing-process', 'fake-fixture']:
            root = Path(temp) / mutation
            shutil.copytree(args.results, root)
            if mutation == 'fake-summary':
                summary = load(root / 'summary.json')
                summary['production_changed'] = True
                write(root / 'summary.json', summary)
            elif mutation == 'missing-process':
                records = load(root / 'processes.json')[:-1]
                write(root / 'processes.json', records)
                complete = load(root / 'complete.json')
                complete['processes'] = len(records)
                complete['processes_sha256'] = digest(root / 'processes.json')
                write(root / 'complete.json', complete)
            else:
                calibrations = load(root / 'calibrations.json')
                records = load(root / 'processes.json')
                case = calibrations[0]['case']
                old = calibrations[0]['calibration'][0]['digest']
                forged = '0' * 16 if old != '0' * 16 else 'f' * 16
                for record in calibrations + records:
                    if record['case'] != case:
                        continue
                    key = 'windows' if record['calls'] else 'calibration'
                    for value in record[key]:
                        value['digest'] = forged
                    stdout = root / (record['label'] + '.stdout')
                    stdout.write_text(stdout.read_text().replace(old, forged))
                    record['stdout_sha256'] = digest(stdout)
                write(root / 'calibrations.json', calibrations)
                write(root / 'processes.json', records)
                schedule = load(root / 'schedule.json')
                schedule['calibrations_sha256'] = digest(root / 'calibrations.json')
                write(root / 'schedule.json', schedule)
                complete = load(root / 'complete.json')
                complete['schedule_sha256'] = digest(root / 'schedule.json')
                complete['processes_sha256'] = digest(root / 'processes.json')
                write(root / 'complete.json', complete)
            try:
                verify(root)
            except AssertionError as error:
                expected = {'fake-summary': 'summary must reproduce', 'missing-process': 'process cardinality',
                            'fake-fixture': 'independent oracle digest mismatch'}[mutation]
                assert expected in str(error), f'{mutation}: rejected for an unrelated reason: {error}'
                outcomes.append({'mutation': mutation, 'rejected': True, 'reason': str(error)})
            else:
                raise AssertionError(f'{mutation}: corrupted result was accepted')
    print(json.dumps({'valid': original, 'negative_controls': outcomes}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
