#!/usr/bin/env python3
"""Darwin study runner. Requires the inherited, already held shared flock FD 3.

Never runs measurements concurrently. All process outputs, admissions, failures,
and controls survive in the chosen results directory. A failed control makes a
cell diagnostic; it is not rerun until favorable.
"""
import argparse
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LOCK = Path('/tmp/onnx-genai-hostlock-darwin.flock')
CASES = [('q4', 1, 1024, 256), ('q4', 8, 1024, 256),
         ('q8', 1, 1024, 256), ('q8', 8, 1024, 256)]
CONTROL_ENV = {
    'RAYON_NUM_THREADS': '1', 'ONNX_GENAI_CPU_DECODE_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1', 'ONNX_GENAI_MLAS_THREADPOOL_THREADS': '1',
    'ONNX_GENAI_HOST_ALLOC_CACHE_FLOOR_BYTES': '1048576',
    'ONNX_GENAI_CPU_DENSE_WEIGHT_CACHE_BYTES': '2097152',
}
ENV = {key: value for key, value in os.environ.items()
       if not key.startswith(('NXRT_', 'ONNX_GENAI_', 'RAYON_', 'VECLIB_'))}
ENV.update(CONTROL_ENV)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


class HostCPU:
    """Uncached per-processor Mach ticks, aggregated without CPU affinity.

    SDK processor_info.h: PROCESSOR_CPU_LOAD_INFO=2 and four u32 states.
    host_statistics snapshots were observed to repeat under its rate limiting;
    host_processor_info instead asks each processor for its counters. Release
    each returned out-of-line array after copying its values.
    """
    def __init__(self):
        self.lib = ctypes.CDLL('/usr/lib/libSystem.B.dylib')
        self.lib.mach_host_self.restype = ctypes.c_uint
        self.lib.host_processor_info.argtypes = [
            ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint)), ctypes.POINTER(ctypes.c_uint)]
        self.lib.mach_vm_deallocate.argtypes = [ctypes.c_uint, ctypes.c_uint64, ctypes.c_uint64]
        self.port = self.lib.mach_host_self()
        self.task = ctypes.c_uint.in_dll(self.lib, 'mach_task_self_').value
        self.cores = int(command('sysctl', '-n', 'hw.logicalcpu'))

    def read(self):
        processors, count = ctypes.c_uint(), ctypes.c_uint()
        info = ctypes.POINTER(ctypes.c_uint)()
        status = self.lib.host_processor_info(
            self.port, 2, ctypes.byref(processors), ctypes.byref(info), ctypes.byref(count))
        assert status == 0 and processors.value == self.cores and count.value == self.cores * 4, (status, processors.value, count.value)
        try:
            return [sum(info[cpu * 4 + state] for cpu in range(self.cores)) for state in range(4)]
        finally:
            status = self.lib.mach_vm_deallocate(
                self.task, ctypes.cast(info, ctypes.c_void_p).value, count.value * 4)
            assert status == 0, 'release the Mach counter array'

    def active(self, before, after):
        delta = [(b - a) & 0xffffffff for a, b in zip(before, after)]
        total = sum(delta)
        assert total > 0, 'CPU tick window was too short to observe'
        return self.cores * (delta[0] + delta[1] + delta[3]) / total


def main():
    assert sys.platform == 'darwin', 'this sampler is Darwin-specific'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=Path)
    parser.add_argument('output', type=Path)
    choices = parser.add_mutually_exclusive_group()
    choices.add_argument('--calls-from', type=Path, help='reuse frozen call counts for the identical binary and cases')
    choices.add_argument('--resume', action='store_true', help='continue an interrupted protocol; never repeat completed measurements')
    args = parser.parse_args()
    binary, output = args.binary.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / 'protocol.json'
    if args.resume:
        assert protocol_path.exists() and not (output / 'complete.json').exists(), 'resume requires an incomplete frozen protocol'
        frozen_resume = json.loads(protocol_path.read_text())
        prior_records_hash = digest(output / 'processes.json')
        records = json.loads((output / 'processes.json').read_text())
    else:
        assert not protocol_path.exists(), 'use a fresh directory or explicitly resume the existing protocol'
        frozen_resume, prior_records_hash, records = None, None, []
    lock_stat = os.fstat(3)
    assert (lock_stat.st_dev, lock_stat.st_ino) == (LOCK.stat().st_dev, LOCK.stat().st_ino)
    # An inherited descriptor shares the outer wrapper's open-file lock. Keep it
    # held for calibration, freeze, all paired/null/process controls and checks.
    fcntl.flock(3, fcntl.LOCK_EX | fcntl.LOCK_NB)
    host = HostCPU()
    sampler_check = []
    for _ in range(5):
        before = host.read()
        time.sleep(0.2)
        after = host.read()
        sampler_check.append({'before': before, 'after': after, 'active_cores': host.active(before, after)})
    session_tag = f'resume-{len(list(output.glob("resume-*-provenance.json"))) + 1}' if args.resume else 'initial'
    (output / (session_tag + '-sampler-check.json')).write_text(json.dumps(sampler_check, indent=2) + '\n')
    binary_sha = digest(binary)
    provenance = {
        'runtime_commit': command('git', 'rev-parse', 'HEAD'),
        'runtime_changes': command('git', 'diff', '--name-only', 'HEAD', '--', 'crates'),
        'binary': str(binary), 'binary_sha256': binary_sha,
        'probe_sha256': digest(HERE / 'probe/src/main.rs'),
        'runner_sha256': digest(__file__),
        'manifest_sha256': digest(HERE / 'probe/Cargo.toml'),
        'lockfile_sha256': digest(HERE / 'probe/Cargo.lock'),
        'rustc': command('rustc', '-Vv'), 'cargo': command('cargo', '-V'),
        'os': command('sw_vers'), 'uname': command('uname', '-a'),
        'hardware': command('sysctl', 'hw.model', 'machdep.cpu.brand_string',
                            'hw.physicalcpu', 'hw.logicalcpu', 'hw.memsize'),
        'lock_path': str(LOCK), 'lock_inode': lock_stat.st_ino,
        'lock_wrapper_pid': os.getppid(), 'runner_pid': os.getpid(),
        'environment': CONTROL_ENV,
        'sampler': 'host_processor_info per-CPU ticks aggregated over whole process including untimed idle tail; no affinity or window-level contention bound; complete child CPU subtracted',
    }
    assert provenance['runtime_changes'] in ['', 'crates/onnx-runtime-cpuinfo/vendor/cpuinfo'], provenance
    if args.resume:
        assert frozen_resume['binary_sha256'] == binary_sha, 'resume binary changed'
        provenance['prior_records_sha256'] = prior_records_hash
    provenance_name = session_tag + '-provenance.json' if args.resume else 'provenance.json'
    (output / provenance_name).write_text(json.dumps(provenance, indent=2) + '\n')
    completed = {record['label']: record for record in records
                 if record.get('exit_code') == 0 and 'result' in record}


    def save():
        (output / 'processes.json').write_text(json.dumps(records, indent=2) + '\n')

    def invoke(case, mode, calls, label):
        if label in completed:
            previous = completed[label]
            assert previous['case'] == list(case) and previous['mode'] == mode and previous['calls'] == calls, 'resume changed an already completed request'
            return previous
        assert digest(binary) == binary_sha, 'executable changed during protocol'
        admissions = []
        for _ in range(120):
            before = host.read()
            time.sleep(0.2)
            after = host.read()
            activity = host.active(before, after)
            admissions.append({'ticks_before': before, 'ticks_after': after, 'active_cores': activity})
            if activity <= 0.5:
                break
            time.sleep(0.3)
        else:
            records.append({'label': label, 'admission_failed': True, 'admissions': admissions, 'session': session_tag})
            save()
            raise RuntimeError('host never met the pre-run <=0.5 active-core gate; no measurement started')
        cmd = [str(binary), *map(str, case), mode, str(calls)]
        stdout, stderr = output / (label + '.json'), output / (label + '.stderr')
        processes = []
        for line in command('ps', '-axo', 'pid,pcpu,comm').splitlines()[1:]:
            fields = line.split(maxsplit=2)
            if len(fields) == 3:
                processes.append(fields[:2] + [Path(fields[2]).name])
        (output / (label + '.ps')).write_text(json.dumps(processes) + '\n')
        ticks_before = host.read()
        start = time.perf_counter()
        with stdout.open('w') as out, stderr.open('w') as err:
            child = subprocess.Popen(cmd, cwd=ROOT, env=ENV, stdout=out, stderr=err)
            _, status, usage = os.wait4(child.pid, 0)
            child.returncode = os.waitstatus_to_exitcode(status)
        wall = time.perf_counter() - start
        ticks_after = host.read()
        cpu = usage.ru_utime + usage.ru_stime
        active = host.active(ticks_before, ticks_after)
        foreign = max(0., active - cpu / wall)
        record = {'label': label, 'session': session_tag, 'case': case, 'mode': mode, 'calls': calls,
                  'command': cmd, 'exit_code': child.returncode, 'cpu_s': cpu,
                  'wall_s': wall, 'cpu_per_wall': cpu / wall,
                  'host_process_average_active_cores': active, 'foreign_process_average_cores': foreign,
                  'ticks_before': ticks_before, 'ticks_after': ticks_after,
                  'admissions': admissions, 'stdout': stdout.name, 'stderr': stderr.name}
        records.append(record)
        save()
        assert child.returncode == 0, (label, stderr.read_text()[-6000:])
        data = json.loads(stdout.read_text())
        assert data['settled'] and data['zero_byte_counter_control'] and data['dense_expansion_positive_control']
        assert all(w['claims'] == w['returns'] == 0 for w in data['windows'])
        record['result'] = data
        save()
        print(json.dumps({'completed': label, 'foreign_process_average_cores': foreign,
                          'window_count': data['window_count']}), flush=True)
        return record

    # Correctness prechecks are separate from calibration and timed conclusions.
    for fmt, rows, _, n in CASES:
        invoke((fmt, rows, 1003, n), 'check', 2, f'partial-{fmt}-{rows}')
    calls = {}
    calibration_source = None
    if args.resume:
        calls = frozen_resume['calls']
        calibration_source = frozen_resume['calibration_source']
    elif args.calls_from is not None:
        prior = args.calls_from.resolve()
        frozen = json.loads(prior.read_text())
        assert frozen['binary_sha256'] == binary_sha and frozen['cases'] == [list(case) for case in CASES], 'frozen counts belong to a different binary or geometry'
        calls = frozen['calls']
        assert all(type(calls[f'{fmt}-{rows}']) is int and calls[f'{fmt}-{rows}'] > 0 for fmt, rows, _, _ in CASES)
        calibration_source = {'path': str(prior), 'sha256': digest(prior)}
    else:
        for fmt, rows, k, n in CASES:
            rec = invoke((fmt, rows, k, n), 'calibrate', 8, f'calibrate-{fmt}-{rows}')
            calls[f'{fmt}-{rows}'] = rec['result']['calibrated_calls']
    protocol = {
        'calls': calls, 'repetitions': 3, 'cases': CASES, 'calibration_source': calibration_source,
        'paired_sequences': {'aa': [0, 1, 1, 0], 'ab': [0, 2, 2, 0],
                             'ba': [2, 0, 0, 2], 'aa_reverse': [1, 0, 0, 1]},
        'arm_names': ['float32-A', 'float32-A-copy', 'q8_0'],
        'admission_policy': {'max_attempts': 120, 'sample_s': 0.2, 'retry_sleep_s': 0.3, 'max_active_cores': 0.5},
        'parent_controls': ['work', 'work-copy', 'work-double', 'setup', 'setup-double'],
        'gates': {'foreign_process_average_cores_max': 0.5, 'window_cpu_per_wall': [0.85, 1.15],
                  'aa_median_delta_pct_max': 3, 'aa_process_delta_pct_max': 5,
                  'per_arm_range_pct_max': 15, 'setup_parent_cpu_fraction_max': 0.1,
                  'double_marginal_cpu_ratio': [1.7, 2.3]},
        'binary_sha256': binary_sha,
        'interpretation': 'Cache-hot synthetic CPU projection; one thread; no model/GPU/quality extrapolation',
    }
    if args.resume:
        assert json.loads(json.dumps(protocol)) == frozen_resume, 'resume changed the frozen protocol'
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2) + '\n')
    # Freeze before measuring. No adaptive repetitions or favorable re-runs.
    for rep in range(3):
        rotated = CASES[rep:] + CASES[:rep]
        for case in rotated:
            fmt, rows, _, _ = case
            count = calls[f'{fmt}-{rows}']
            invoke(case, 'paired', count, f'paired-{fmt}-{rows}-{rep}')
            for arm in (['float32', 'q8_0'] if rep % 2 == 0 else ['q8_0', 'float32']):
                for control, multiplier in [('work', 1), ('work-copy', 1), ('setup', 1),
                                            ('work-double', 2), ('setup-double', 2)]:
                    mode = 'setup-' + arm if control.startswith('setup') else arm
                    invoke(case, mode, count * multiplier,
                           f'process-{fmt}-{rows}-{arm}-{control}-{rep}')
    assert digest(binary) == binary_sha
    (output / 'complete.json').write_text(json.dumps({'processes': len(records), 'all_launched_processes_succeeded': True}, indent=2) + '\n')


if __name__ == '__main__':
    main()
