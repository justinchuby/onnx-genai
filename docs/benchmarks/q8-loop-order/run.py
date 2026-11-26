#!/usr/bin/env python3
"""Run a frozen Darwin Q8 projection experiment under the existing shared flock."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LOCK = Path('/tmp/onnx-genai-hostlock-darwin.flock')
BASELINE_RUNTIME_SHA256 = 'ec9f1d6806e83099c673160ab71f76b2a987fbdc914d99d45961f11a37a67f67'
SUBJECT = 'kernels::packed_projection::loop_order_study::paired_loop_order_process'
CHECK = 'kernels::packed_projection::loop_order_study::loop_orders_preserve_oracle_bits_dirty_reuse_and_scale_errors'
SOURCES = [
    'crates/onnx-runtime-ep-cpu/src/kernels/packed_projection.rs',
    'crates/onnx-runtime-ep-cpu/src/kernels/packed_projection/loop_order_study.rs',
    'docs/benchmarks/packed-activation-study/run.py',
    'docs/benchmarks/q8-loop-order/run.py',
    'docs/benchmarks/q8-loop-order/build.py',
    'docs/benchmarks/q8-loop-order/analyze.py',
    'docs/benchmarks/q8-loop-order/verify.py',
    'Cargo.lock', 'Cargo.toml', 'crates/onnx-runtime-ep-cpu/Cargo.toml',
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_parent_digest():
    source = (ROOT / SOURCES[0]).read_bytes()
    suffix = b'\n#[cfg(test)]\nmod loop_order_study;\n'
    assert source.endswith(suffix), 'study entry must remain behind cfg(test); restore the frozen source'
    actual = hashlib.sha256(source[:-len(suffix)]).hexdigest()
    assert actual == BASELINE_RUNTIME_SHA256, 'production projection changed; use the frozen study revision'
    return actual


def write(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def windows(stdout, marker):
    return [json.loads(line.split(marker, 1)[1]) for line in stdout.splitlines() if marker in line]


def main():
    assert sys.platform == 'darwin', 'this sampler requires Darwin; correctness tests remain portable'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--build-receipt', type=Path, required=True)
    args = parser.parse_args()
    binary, out = args.binary.resolve(), args.output.resolve()
    assert binary.is_file(), 'build the release test executable first'
    runtime_parent_digest()
    build = json.loads(args.build_receipt.read_text())
    assert build['binary_sha256'] == digest(binary) and Path(build['binary']).resolve() == binary, 'binary does not match its build receipt'
    assert build['sources'] == {p: digest(ROOT / p) for p in build['sources']}, 'sources differ from the frozen build'

    assert not out.exists(), 'choose a new results directory; completed measurements must not be rerun'
    # Reuse the already audited uncached Mach sampler, not a second implementation.
    prior = HERE.parent / 'packed-activation-study/run.py'
    spec = importlib.util.spec_from_file_location('prior_sampler', prior)
    sampler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sampler)
    host = sampler.HostCPU()
    cases = [(q, r, k) for q in ['q4', 'q8'] for r in [1, 3, 8, 32] for k in [32, 1024]]
    seed = 20261126
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('NXRT_', 'ONNX_GENAI_', 'RAYON_', 'VECLIB_', 'MLAS_'))}
    env.update(RAYON_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    with LOCK.open('r+') as lock:
        print(json.dumps({'status': 'queued', 'pid': os.getpid(), 'inode': os.fstat(lock.fileno()).st_ino}), flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        inode = os.fstat(lock.fileno()).st_ino
        assert inode == LOCK.stat().st_ino, 'shared flock was replaced; do not measure'
        print(json.dumps({'status': 'acquired', 'pid': os.getpid(), 'inode': inode}), flush=True)
        out.mkdir(parents=True)
        protocol = {
            'version': 1, 'base': command('git', 'rev-parse', 'HEAD'),
            'production_parent_before_study_sha256': runtime_parent_digest(),
            'build_receipt': build, 'build_receipt_sha256': digest(args.build_receipt),
            'binary_sha256': digest(binary), 'binary': str(binary),
            'sources': {p: digest(ROOT / p) for p in SOURCES},
            'rustc': command('rustc', '-Vv'), 'machine': command('sysctl', '-n', 'machdep.cpu.brand_string'),
            'logical_cpus': host.cores, 'os': command('sw_vers'),
            'pool_width': 1, 'n': 256, 'cases': cases, 'repetitions': 4,
            'seed': seed, 'target_window_s': 0.2,
            'sampler': 'uncached host_processor_info over each complete child process; complete child CPU subtracted; no affinity or per-window foreign-CPU bound',
            'gates': {'foreign_process_average_cores_max': 0.5, 'cpu_per_wall': [0.85, 1.15],
                      'aa_ratio': [0.95, 1.05], 'double_ratio': [1.8, 2.2], 'setup_fraction_max': 0.1},
            'lock': {'path': str(LOCK), 'inode': inode, 'parent_pid': os.getpid(), 'inherited_by_children': True},
            'environment': {'RAYON_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1'},
            'scope': 'full private projection plus fresh checked test lane binding; excludes operator publication, routing, governed admission and model inference',
        }
        write(out / 'protocol.json', protocol)
        listing = subprocess.check_output([str(binary), '--list'], env=env, text=True, pass_fds=(lock.fileno(),))
        (out / 'test-list.txt').write_text(listing)
        for name in [CHECK, SUBJECT]:
            assert listing.splitlines().count(name + ': test') == 1, f'test selector drift: {name}'
        check = subprocess.run([str(binary), CHECK, '--exact', '--nocapture', '--test-threads=1'],
                               env=env, text=True, capture_output=True, pass_fds=(lock.fileno(),))
        (out / 'correctness.stdout').write_text(check.stdout)
        (out / 'correctness.stderr').write_text(check.stderr)
        assert check.returncode == 0 and f'test {CHECK} ... ok' in check.stdout and '1 passed; 0 failed; 0 ignored' in check.stdout, 'named correctness test must run and pass before timing'

        # A bounded schema smoke catches a stale executable/control mismatch
        # before calibration or confirmatory measurements. Its times are unused.
        pre_env = env | {'NXRT_LOOP_FORMAT': 'q4', 'NXRT_LOOP_ROWS': '1', 'NXRT_LOOP_K': '32',
                         'NXRT_LOOP_CALLS': '1', 'NXRT_LOOP_REVERSE': '0'}
        pre = subprocess.run([str(binary), SUBJECT, '--exact', '--ignored', '--nocapture', '--test-threads=1'],
                             env=pre_env, text=True, capture_output=True, pass_fds=(lock.fileno(),))
        (out / 'schema.stdout').write_text(pre.stdout)
        (out / 'schema.stderr').write_text(pre.stderr)
        schema = windows(pre.stdout, 'NXRT_LOOP_WINDOW ')
        assert pre.returncode == 0 and [w['label'] for w in schema] == ['row', 'block', 'block', 'row', 'aa1', 'aa2', 'baa1', 'baa2', 'setup', 'double', 'bdouble'], 'stale research executable or control schema; rebuild before timing'

        def launch(label, case, calls, reverse):
            q, rows, k = case
            child_env = env | {'NXRT_LOOP_FORMAT': q, 'NXRT_LOOP_ROWS': str(rows), 'NXRT_LOOP_K': str(k),
                               'NXRT_LOOP_CALLS': str(calls), 'NXRT_LOOP_REVERSE': str(int(reverse))}
            assert digest(binary) == protocol['binary_sha256'], 'measured binary changed'
            assert os.fstat(lock.fileno()).st_ino == LOCK.stat().st_ino == inode
            usage = resource.getrusage(resource.RUSAGE_CHILDREN)
            before_cpu = usage.ru_utime + usage.ru_stime
            before = host.read()
            start = time.perf_counter()
            with (out / (label + '.stdout')).open('w') as stdout, (out / (label + '.stderr')).open('w') as stderr:
                child = subprocess.run([str(binary), SUBJECT, '--exact', '--ignored', '--nocapture', '--test-threads=1'],
                                       env=child_env, stdout=stdout, stderr=stderr, pass_fds=(lock.fileno(),))
            wall = time.perf_counter() - start
            after = host.read()
            usage = resource.getrusage(resource.RUSAGE_CHILDREN)
            cpu = usage.ru_utime + usage.ru_stime - before_cpu
            assert os.fstat(lock.fileno()).st_ino == LOCK.stat().st_ino == inode
            stdout = (out / (label + '.stdout')).read_text()
            assert child.returncode == 0 and SUBJECT in stdout and '1 passed; 0 failed; 0 ignored' in stdout, f'{label}: research test failed or did not execute'
            active = host.active(before, after)
            record = {'label': label, 'case': case, 'calls': calls, 'reverse': reverse, 'exit': child.returncode,
                      'process_wall_s': wall, 'process_cpu_s': cpu, 'host_ticks_before': before, 'host_ticks_after': after,
                      'host_process_average_active_cores': active, 'foreign_process_average_cores': max(0., active - cpu / wall),
                      'lock_inode_before': inode, 'lock_inode_after': LOCK.stat().st_ino,
                      'stdout_sha256': digest(out / (label + '.stdout')), 'stderr_sha256': digest(out / (label + '.stderr'))}
            if calls:
                record['windows'] = windows(stdout, 'NXRT_LOOP_WINDOW ')
                assert len(record['windows']) == 11, f'{label}: incomplete window sequence'
            else:
                record['calibration'] = windows(stdout, 'NXRT_LOOP_CALIBRATION ')
                assert len(record['calibration']) == 1
            print(json.dumps({'completed': label}), flush=True)
            return record

        calibrations = [launch(f'cal-{i:02}', case, 0, False) for i, case in enumerate(cases)]
        write(out / 'calibrations.json', calibrations)
        schedule = [{'case': case, 'rep': rep, 'calls': calibrations[i]['calibration'][0]['calls'], 'reverse': bool(rep % 2)}
                    for i, case in enumerate(cases) for rep in range(4)]
        random.Random(seed).shuffle(schedule)
        frozen = {'protocol_sha256': digest(out / 'protocol.json'), 'calibrations_sha256': digest(out / 'calibrations.json'), 'processes': schedule}
        write(out / 'schedule.json', frozen)
        records = []
        for i, item in enumerate(schedule):
            record = launch(f'run-{i:03}', item['case'], item['calls'], item['reverse'])
            record['rep'] = item['rep']
            records.append(record)
            write(out / 'processes.json', records)
        assert {p: digest(ROOT / p) for p in SOURCES} == protocol['sources'], 'source changed during study'
        write(out / 'complete.json', {'processes': len(records), 'schedule_sha256': digest(out / 'schedule.json'),
                                     'processes_sha256': digest(out / 'processes.json'), 'binary_sha256': digest(binary)})
        print(json.dumps({'complete': len(records), 'output': str(out)}), flush=True)


if __name__ == '__main__':
    main()
