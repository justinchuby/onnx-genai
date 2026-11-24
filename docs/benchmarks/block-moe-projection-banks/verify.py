"""Verify retained data; optionally bind it to a checked-out source root."""
from pathlib import Path
import base64
import gzip
import hashlib
import json
import sys

folder = Path(__file__).resolve().parent
sha = lambda data: hashlib.sha256(data).hexdigest()
manifest = json.loads((folder / 'manifest.json').read_text())
actual = {str(p.relative_to(folder)) for p in folder.rglob('*')
          if p.is_file() and p.name != 'manifest.json' and '__pycache__' not in p.parts}
assert actual == set(manifest)
for name, digest in manifest.items():
    assert sha((folder / name).read_bytes()) == digest, name
for phase in ['initial', 'inline']:
    target = folder / phase
    metadata = json.loads((target / 'metadata.json').read_text())
    raw = gzip.decompress((target / 'receipts.json.gz').read_bytes())
    assert sha(raw) == metadata['receipts_sha256']
    rows = json.loads(raw)
    outputs = json.loads(gzip.decompress((target / 'full-outputs.json.gz').read_bytes()))
    logs = json.loads(gzip.decompress((target / 'process-logs.json.gz').read_bytes()))
    assert len(rows) == len(outputs) == 144 and len(logs) == 288
    for row in rows:
        name = f"r{row['rep']}-c{row['case']}-{row['kind']}-{row['arm']}"
        values = base64.b64decode(outputs[name + '.f32'])
        assert sha(values) == row['output_sha256']
        assert len(values) == row['rows'] * row['hidden'] * 4
        receipt = json.loads(logs[name + '.stdout'].strip().splitlines()[-1])
        assert all(row[key] == value for key, value in receipt.items())
    for path, digest in metadata['source_sha256'].items():
        assert sha(gzip.decompress((target / (Path(path).name + '.gz')).read_bytes())) == digest
    assert sha((target / 'run.py').read_bytes()) == metadata['runner_sha256']
metadata = json.loads((folder / 'inline/metadata.json').read_text())
validation = json.loads((folder / 'validation/validation.json').read_text())
assert validation['source_sha256'] == metadata['source_sha256']
assert len(validation['results']) == 8
for result in validation['results']:
    assert result['exit'] == 0
    log = folder / 'validation' / (result['name'] + '.log.gz')
    assert sha(gzip.decompress(log.read_bytes())) == result['log_sha256']
if len(sys.argv) > 1:
    source = Path(sys.argv[1])
    for path, digest in metadata['source_sha256'].items():
        assert sha((source / path).read_bytes()) == digest, path
print(f'Verified {len(manifest)} payloads, 288 full outputs, both frozen matrices and 8 final checks.')
