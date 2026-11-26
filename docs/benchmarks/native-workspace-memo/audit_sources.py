#!/usr/bin/env python3
"""Bind archived measured sources to immutable runtime revisions and receipts."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import build


def sha(data):
    return hashlib.sha256(data).hexdigest()

def git_blob(revision,path):
    return subprocess.check_output(['git','show',f'{revision}:{path}'],cwd=build.ROOT)

def audit(root):
    inventory=json.loads((root/'checksums.json').read_text())
    assert set(inventory)=={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name!='checksums.json'}
    for path,digest in inventory.items():
        assert sha((root/path).read_bytes())==digest,path
    m=json.loads((root/'manifest.json').read_text());b=m['build_receipt'];p=m['protocol']
    expected={}
    for path,digest in b['probe_sources'].items():
        relative=(build.HERE/path).relative_to(build.ROOT).as_posix();expected[relative]=digest
    for path,digest in m['harness_sources'].items():
        relative=(build.HERE/path).resolve().relative_to(build.ROOT.resolve()).as_posix();expected[relative]=digest
    expected['docs/benchmarks/native-workspace-memo/build.py']=b['build_script_sha256']
    actual=json.loads((root/'source-snapshot.json').read_text());assert actual==expected
    for path,digest in expected.items():
        assert sha(gzip.decompress((root/'source-snapshot'/(path+'.gz')).read_bytes()))==digest,path
    runtime_files=0
    for label,arm in b['builds'].items():
        assert arm['revision']==p[label]
        for path,digest in arm['runtime_sources'].items():
            archived=gzip.decompress((root/'build/sources'/label/(path+'.gz')).read_bytes())
            assert archived==git_blob(p[label],path),f'{label}: source does not match pinned runtime commit: {path}'
            assert sha(archived)==digest
            runtime_files+=1
    assert sha(git_blob(p['candidate'],'Cargo.lock'))==b['root_cargo_lock_sha256']
    assert sha(git_blob(p['candidate'],'Cargo.toml'))==b['root_cargo_toml_sha256']
    return {'verified':True,'runtime_file_projections':runtime_files,'measured_probe_controller_files':len(expected),'runtime_commit_pins':[p['baseline'],p['candidate']]}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('results',type=Path);args=parser.parse_args();print(json.dumps(audit(args.results),sort_keys=True))
