"""Verify sealed prefetch protocol checks; optionally check the current source."""
from pathlib import Path
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
for path, digest in manifest.items():
    assert sha((folder / path).read_bytes()) == digest, path
source = gzip.decompress((folder / 'final.rs.gz').read_bytes())
validation = json.loads((folder / 'validation/validation.json').read_text())
source_path = 'crates/onnx-runtime-ep-cpu/src/kernels/qmoe.rs'
assert validation['source_sha256'][source_path] == sha(source)
assert len(validation['results']) == 6
for result in validation['results']:
    assert result['exit'] == 0
    log = gzip.decompress((folder / ('validation/' + result['name'] + '.log.gz')).read_bytes())
    assert sha(log) == result['log_sha256']
    if 'counts' in result:
        assert result['counts'] and all(int(p) > 0 and int(f) == 0 for p, f, _ in result['counts'])
    if result['name'] in ('qmoe', 'qmoe-mlas'):
        assert result['counts'] == [['68', '0', '1']]
        for test in ('prefetch_group_driver_preserves_sparse_ids_and_two_live_payload_bound',
                     'prefetch_group_driver_propagates_callback_failures_and_retires_pending_weights',
                     'sparse_prefetch_expert_keys_match_serial_across_capacity_modes'):
            assert ('test kernels::qmoe::tests::' + test + ' ... ok').encode() in log
controls = json.loads((folder / 'controls/controls.json').read_text())
assert [r['name'] for r in controls] == ['wrong-id', 'spawn-failure']
for result in controls:
    mutant = gzip.decompress((folder / ('controls/' + result['name'] + '-qmoe.rs.gz')).read_bytes())
    assert sha(mutant) == result['mutated_source_sha256']
    log = gzip.decompress((folder / ('controls/' + result['name'] + '.log.gz')).read_bytes())
    assert b'error[E' not in log
    if result['name'] == 'wrong-id':
        assert result['exit'] != 0
        assert b'0 passed; 1 failed' in log
        assert b'loaded weights must match the group expert ID' in log
        assert mutant == source.replace(b'let leased = load(expert);', b'let _ = expert; let leased = load(0);')
    else:
        assert result['exit'] == 0 and b'1 passed; 0 failed' in log
        assert b'could not start a host worker:' in log
        assert b'os error' in log
        assert b'downcast_ref::<std::io::Error>()' in mutant
        changed = source.replace(b'std::thread::Builder::new()\n            .spawn_scoped',
                                 b'std::thread::Builder::new().stack_size(1usize << 62)\n            .spawn_scoped')
        assert mutant.startswith(changed)
        assert mutant[len(changed):].startswith(b'\n#[cfg(test)]\nmod prefetch_spawn_failure_probe')
restored = json.loads((folder / 'controls/restoration.json').read_text())
assert restored['restored_sha256'] == sha(source)
if len(sys.argv) > 1:
    assert (Path(sys.argv[1]) / source_path).read_bytes() == source
print(f'Verified {len(manifest)} sealed payloads, nonempty suites, wrong-ID control and typed OS failure probe.')
