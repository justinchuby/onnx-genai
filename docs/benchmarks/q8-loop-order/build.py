#!/usr/bin/env python3
"""Build the native release study and bind its executable to source receipts."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import run as study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', type=Path)
    args = parser.parse_args()
    study.runtime_parent_digest()
    target = args.target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    source_paths = [p for p in study.SOURCES if p.endswith('.rs') or p in ['Cargo.toml', 'Cargo.lock', 'crates/onnx-runtime-ep-cpu/Cargo.toml']]
    sources = {p: study.digest(study.ROOT / p) for p in source_paths}
    controlled = {'CARGO_TARGET_DIR': str(target), 'CARGO_INCREMENTAL': '0', 'CARGO_BUILD_JOBS': '1',
                  'CARGO_PROFILE_RELEASE_DEBUG': '0', 'CARGO_PROFILE_RELEASE_CODEGEN_UNITS': '1'}
    env = {k: v for k, v in os.environ.items() if not k.startswith(('CARGO_PROFILE_', 'NXRT_', 'ONNX_GENAI_')) and k not in ['RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS']}
    env.update(controlled)
    command = ['cargo', 'test', '--locked', '--release', '-p', 'onnx-runtime-ep-cpu', '--lib',
               'packed_projection::loop_order_study', '--no-run', '--message-format=json-render-diagnostics']
    with study.LOCK.open('r+') as lock:
        print(f'Queued release build on shared flock inode {os.fstat(lock.fileno()).st_ino}', file=sys.stderr, flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        inode = os.fstat(lock.fileno()).st_ino
        assert inode == study.LOCK.stat().st_ino
        with (target / 'loop-study-cargo.jsonl').open('w') as stdout, (target / 'loop-study-cargo.stderr').open('w') as stderr:
            result = subprocess.run(command, cwd=study.ROOT, env=env, stdout=stdout, stderr=stderr, pass_fds=(lock.fileno(),))
        assert result.returncode == 0, f'build failed; inspect {target / "loop-study-cargo.stderr"}'
        assert sources == {p: study.digest(study.ROOT / p) for p in source_paths}, 'source changed during build; freeze it before rebuilding'
        artifacts = [json.loads(line) for line in (target / 'loop-study-cargo.jsonl').read_text().splitlines()]
        artifacts = [a for a in artifacts if a.get('reason') == 'compiler-artifact' and a.get('executable') and a.get('target', {}).get('name') == 'onnx_runtime_ep_cpu']
        assert len(artifacts) == 1
        artifact = artifacts[0]
        assert artifact['profile']['test'] and artifact['profile']['opt_level'] == '3'
        assert 'full' in artifact['features'] and 'mlas' not in artifact['features']
        binary = Path(artifact['executable'])
        receipt = {'binary': str(binary), 'binary_sha256': study.digest(binary), 'sources': sources,
                   'command': command, 'controlled_environment': controlled, 'cargo_artifact': artifact,
                   'base': study.command('git', 'rev-parse', 'HEAD'), 'rustc': study.command('rustc', '-Vv'),
                   'lock': {'path': str(study.LOCK), 'inode': inode}}
        study.write(target / 'loop-study-build.json', receipt)
        print(binary)


if __name__ == '__main__':
    main()
