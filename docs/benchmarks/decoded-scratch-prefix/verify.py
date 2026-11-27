"""Verify live-prefix initialization tests and the real QMoE mutation control."""
from pathlib import Path
import gzip
import hashlib
import json
import sys

folder = Path(__file__).resolve().parent
sha = lambda b: hashlib.sha256(b).hexdigest()
read = lambda p: gzip.decompress((folder / p).read_bytes())
manifest = json.loads((folder / 'manifest.json').read_text())
actual = {str(p.relative_to(folder)) for p in folder.rglob('*')
          if p.is_file() and p.name != 'manifest.json' and '__pycache__' not in p.parts}
assert actual == set(manifest)
for p, h in manifest.items():
    assert sha((folder / p).read_bytes()) == h, p
records = []
expected = {'projection': ['8', '0', '0'], 'block-matmul': ['39', '0', '1'],
            'block-moe': ['31', '0', '0'], 'qmoe': ['68', '0', '1'],
            'layout': ['17', '0', '0'], 'oracle': ['1', '0', '0']}
for mode, count in [('native', 8), ('mlas', 7)]:
    record = json.loads((folder / (mode + '/validation.json')).read_text())
    records.append(record)
    assert len(record['results']) == count
    for result in record['results']:
        assert result['exit'] == 0
        log = read(mode + '/' + result['name'] + '.log.gz')
        assert sha(log) == result['log_sha256']
        if result['name'] in expected:
            assert result['counts'] == [expected[result['name']]]
        if result['name'] == 'list':
            for name in ['scratch_initialization_extends_only_the_live_prefix_and_refusal_is_transactional',
                         'projection_prefix_tracks_actual_group_rows_and_survives_decoder_error']:
                assert ('kernels::decoded_projection::tests::' + name + ': test').encode() in log
assert records[0]['source_sha256'] == records[1]['source_sha256']
for p, h in records[0]['source_sha256'].items():
    assert sha(read('source/' + p + '.gz')) == h, p
source_path = 'crates/onnx-runtime-ep-cpu/src/kernels/decoded_projection.rs'
source = read('source/' + source_path + '.gz')
assert b'fn initialized(' not in source
negative = json.loads((folder / 'negative/receipt.json').read_text())
assert negative['selected_test_failure'] and negative['exit'] != 0
original = read('negative/original.rs.gz')
mutant = read('negative/mutant.rs.gz')
log = read('negative/negative.log.gz')
assert original == source
assert sha(original) == negative['original_sha256']
assert sha(mutant) == negative['mutant_sha256'] and sha(log) == negative['log_sha256']
a = b'Ok(scratch.initialized_prefix(elements)?.split_at_mut(columns))'
b = b'let values = scratch.initialized_prefix(available)?; Ok(values[..elements].split_at_mut(columns))'
assert original.count(a) == 1 and original.replace(a, b) == mutant
assert b'0 passed; 1 failed' in log and b'error[E' not in log
assert b'only actual grouped projections may initialize decoder prefixes' in log
assert json.loads((folder / 'negative/restoration.json').read_text())['restored_sha256'] == sha(source)
if len(sys.argv) > 1:
    root = Path(sys.argv[1])
    for p, h in records[0]['source_sha256'].items():
        assert sha((root / p).read_bytes()) == h, p
print(f'Verified {len(manifest)} sealed payloads, native/MLAS suites, actual QMoE mutation failure and source restoration.')
