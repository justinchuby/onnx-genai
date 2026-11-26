#!/usr/bin/env python3
"""Archive flat synthetic study receipts deterministically, retaining every raw output."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def bundle(source):
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode='wb', filename='', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w', format=tarfile.USTAR_FORMAT) as tar:
            for path in sorted(source.iterdir()):
                assert path.is_file() and not path.is_symlink(), f'expected flat regular study file: {path}'
                data = path.read_bytes()
                info = tarfile.TarInfo(path.name)
                info.size = len(data)
                info.mode = 0o644
                info.mtime = 0
                tar.addfile(info, io.BytesIO(data))
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    assert args.source.resolve() != args.destination.resolve() and args.source.resolve() not in args.destination.resolve().parents, 'destination must be outside the flat source results directory'
    args.destination.mkdir(parents=True, exist_ok=True)
    first = bundle(args.source)
    assert first == bundle(args.source), 'archive must reproduce byte-for-byte'
    (args.destination / 'raw-results.tar.gz').write_bytes(first)
    files = {p.name: {'sha256': sha(p.read_bytes()), 'bytes': p.stat().st_size} for p in sorted(args.source.iterdir())}
    for name in ['protocol.json', 'summary.json', 'verification.json', 'negative-controls.json']:
        shutil.copyfile(args.source / name, args.destination / name)
    manifest = {'archive_sha256': sha(first), 'archive_bytes': len(first), 'flat_regular_files': files,
                'copied_views': ['protocol.json', 'summary.json', 'verification.json', 'negative-controls.json']}
    (args.destination / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'archive_bytes': len(first), 'raw_files': len(files), 'archive_sha256': sha(first)}))


if __name__ == '__main__':
    main()
