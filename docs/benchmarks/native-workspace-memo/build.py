#!/usr/bin/env python3
"""Build two optimized runtime projections at the identical source/target paths."""
import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tomllib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LOCK = Path('/tmp/onnx-genai-hostlock-darwin.flock')
PROTOCOL = json.loads((HERE / 'protocol.json').read_text())
RUNTIME = ['bindings.rs', 'capture.rs', 'dispatch.rs', 'host_strings.rs', 'kernel_cache.rs']
RUNTIME = [f'crates/onnx-runtime-session/src/executor/{p}' for p in RUNTIME]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()

def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    target, out = args.target.resolve(), args.output.resolve()
    assert not out.exists(), 'use a fresh build receipt directory'
    changed = command('git', 'diff', '--name-only', PROTOCOL['candidate'], '--', 'crates').splitlines()
    assert changed == ['crates/onnx-runtime-cpuinfo/vendor/cpuinfo'] or not changed, 'restore pinned candidate runtime source before building'
    production = command('git', 'diff', '--name-only', PROTOCOL['baseline'], PROTOCOL['candidate'], '--', 'crates').splitlines()
    excluded = [f'crates/onnx-runtime-session/src/executor/{p}' for p in ['tests.rs','workspace_context_tests.rs','workspace_loan_tests.rs','workspace_metadata_tests.rs']]
    assert set(production) == set(RUNTIME + excluded), production
    root_packages=tomllib.loads((ROOT/'Cargo.lock').read_text())['package']
    probe_packages=tomllib.loads((HERE/'probe/Cargo.lock').read_text())['package']
    root_versions={(p['name'],p['version'],p.get('source')) for p in root_packages}
    assert all((p['name'],p['version'],p['source']) in root_versions for p in probe_packages if p.get('source')), 'probe dependency versions must match the pinned runtime lock'
    original = {p: (ROOT / p).read_bytes() for p in RUNTIME}
    inputs = ['probe/Cargo.toml', 'probe/Cargo.lock', 'probe/src/main.rs']
    sources = {p: digest(HERE / p) for p in inputs}
    env = {k:v for k,v in os.environ.items() if not k.startswith(('CARGO_PROFILE_', 'NXRT_', 'ONNX_', 'RAYON_', 'VECLIB_', 'MLAS_', 'OMP_')) and k not in ['RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS']}
    env.update(CARGO_TARGET_DIR=str(target), CARGO_INCREMENTAL='0', CARGO_BUILD_JOBS='1', CARGO_PROFILE_RELEASE_DEBUG='0')
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit('terminated; restoring candidate source')))
    with LOCK.open('r+') as lock:
        print(json.dumps({'status':'queued','pid':os.getpid(),'inode':os.fstat(lock.fileno()).st_ino}), flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        inode = os.fstat(lock.fileno()).st_ino
        assert inode == LOCK.stat().st_ino
        out.mkdir(parents=True)
        for p,data in original.items():
            backup = out / 'candidate-backup' / p
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_bytes(data)
        receipt = {'probe_sources':sources,'protocol':PROTOCOL,'source_projection_paths':RUNTIME,'probe_commit':command('git','rev-parse','HEAD'),'excluded_cfg_test_paths':excluded,'unrelated_dirty_gitlink':changed,'rustc':command('rustc','-Vv'),'environment':{k:env[k] for k in ['CARGO_TARGET_DIR','CARGO_INCREMENTAL','CARGO_BUILD_JOBS','CARGO_PROFILE_RELEASE_DEBUG']},'root_cargo_lock_sha256':digest(ROOT/'Cargo.lock'),'root_cargo_toml_sha256':digest(ROOT/'Cargo.toml'),'build_script_sha256':digest(HERE/'build.py'),'lock':{'path':str(LOCK),'inode':inode},'builds':{}}
        write(out/'build.json',receipt)
        try:
            for label in ['baseline','candidate']:
                revision = PROTOCOL[label]
                for p in RUNTIME:
                    data = subprocess.check_output(['git','show',f'{revision}:{p}'],cwd=ROOT)
                    (ROOT/p).write_bytes(data)
                    archive = out/'sources'/label/(p+'.gz')
                    archive.parent.mkdir(parents=True,exist_ok=True)
                    archive.write_bytes(gzip.compress(data,mtime=0))
                hashes = {p:digest(ROOT/p) for p in RUNTIME}
                cmd = ['cargo','build','--offline','--locked','--release','--manifest-path',str(HERE/'probe/Cargo.toml'),'--message-format=json-render-diagnostics']
                print(json.dumps({'status':'building','label':label,'revision':revision}),flush=True)
                with (out/f'{label}-cargo.jsonl').open('w') as stdout, (out/f'{label}-cargo.stderr').open('w') as stderr:
                    subprocess.run(cmd,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,pass_fds=(lock.fileno(),),check=True)
                assert hashes == {p:digest(ROOT/p) for p in RUNTIME}
                artifacts = [json.loads(line) for line in (out/f'{label}-cargo.jsonl').read_text().splitlines()]
                cpu = [a for a in artifacts if a.get('reason')=='compiler-artifact' and a.get('target',{}).get('name')=='onnx_runtime_ep_cpu']
                assert len(cpu)==1 and 'full' in cpu[0]['features'] and 'mlas' not in cpu[0]['features']
                binaries = [a for a in artifacts if a.get('reason')=='compiler-artifact' and a.get('executable') and a.get('target',{}).get('name')=='native-workspace-memo-study']
                assert len(binaries)==1 and binaries[0]['profile']['opt_level']=='3' and not binaries[0]['profile']['test']
                binary = out/label
                shutil.copy2(binaries[0]['executable'],binary)
                dependencies = command('otool','-L',str(binary))
                symbols = command('nm',str(binary))
                assert ' _main' in symbols
                assert not any('Mlas' in line and (' T ' in line or ' U ' in line) for line in symbols.splitlines())
                receipt['builds'][label]={'revision':revision,'binary':str(binary),'binary_sha256':digest(binary),'runtime_sources':hashes,'command':cmd,'cpu_artifact':cpu[0],'probe_artifact':binaries[0],'dependencies':dependencies,'native_mlas_symbol_audit':True}
                write(out/'build.json',receipt)
        finally:
            for p,data in original.items():
                (ROOT/p).write_bytes(data)
            assert all((ROOT/p).read_bytes()==data for p,data in original.items())
            receipt['candidate_restored']=True
            write(out/'build.json',receipt)
        assert sources == {p:digest(HERE/p) for p in sources}, 'probe changed while building; freeze and rebuild both arms'
        assert receipt['builds']['baseline']['binary_sha256'] != receipt['builds']['candidate']['binary_sha256']
        print(json.dumps({'status':'complete','receipt':str(out/'build.json')}),flush=True)

if __name__=='__main__':
    main()
