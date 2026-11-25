"""Verify Conv group-driver evidence; optionally compare against a checkout."""
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
source = read('final.rs.gz')
base = read('baseline.rs.gz')
validation = json.loads((folder / 'validation/validation.json').read_text())
path = 'crates/onnx-runtime-ep-cpu/src/kernels/conv_ref.rs'
test_path = 'crates/onnx-runtime-ep-cpu/tests/conv_im2col_allocation.rs'
assert sha(source) == validation['source_sha256'][path]
assert len(validation['results']) == 6
for result in validation['results']:
    assert result['exit'] == 0
    log = read('validation/' + result['name'] + '.log.gz')
    assert sha(log) == result['log_sha256']
    if 'counts' in result:
        assert result['counts'] and all(int(p) > 0 and int(f) == 0 for p, f, _ in result['counts'])
    if result['name'] == 'conv':
        assert result['counts'] == [['46', '0', '1']]
    if result['name'].startswith('allocation'):
        assert result['counts'] == [['1', '0', '0']]
        for group, pointwise, size in [(1, 'false', 1200), (2, 'false', 280), (2, 'true', 280)]:
            for batch in [0, 1, 2, 17]:
                matches = int(batch != 0 and pointwise == 'false')
                assert (f'group={group} pointwise={pointwise} batch={batch} col_bytes={size} col_allocations={matches} ').encode() in log
controls = json.loads((folder / 'controls/controls.json').read_text())
assert [r['name'] for r in controls] == ['historical-allocation', 'eager-col', 'same-image', 'same-image-generic']
optional = b'''    let mut col = if is_1x1 {
        Vec::new()
    } else {
        vec![0.0f32; k * n]
    };'''
assert source.count(optional) == 1
for r in controls:
    body = read('controls/' + r['name'] + '.rs.gz')
    log = read('controls/' + r['name'] + '.log.gz')
    assert sha(body) == r['source_sha256'] and sha(log) == r['log_sha256']
    assert b'error[E' not in log
    if r['name'] in ('historical-allocation', 'eager-col'):
        assert r['exit'] != 0 and b'0 passed; 1 failed' in log
        if r['name'] == 'historical-allocation':
            assert body == base
            assert b'group=1 pointwise=false batch=2 total=3' in log
        else:
            assert body == source.replace(optional, b'    let mut col = vec![0.0f32; k * n];')
            assert b'group=2 pointwise=true batch=1 total=2' in log
    else:
        assert r['exit'] == 0 and b'1 passed; 0 failed' in log
        assert b'historical/new comparison executed 540 geometries' in log
        assert body.startswith(source + b'\n#[cfg(test)]\nmod historical_conv_driver_probe')
        old1 = base[base.index(b'fn im2col_gemm_execute('):base.index(b'fn im2col(')]
        start = base.index(b'fn grouped_im2col_gemm_execute(')
        old2 = base[start:base.index('\n// ═══'.encode(), start)]
        assert (old1 + old2).replace(b'super::matmul::', b'crate::kernels::matmul::') in body
assert read('controls/same-image.rs.gz') == read('controls/same-image-generic.rs.gz')
assert json.loads((folder / 'controls/restoration.json').read_text())['restored_sha256'] == sha(source)
initial = json.loads((folder / 'initial/validation.json').read_text())
assert initial['source_sha256'][path] == sha(source)
assert initial['source_sha256'][test_path] == sha(read('initial/allocation-test.rs.gz'))
assert b'native execution still owns staging/geometry allocations' in read('initial/allocation.log.gz')
if len(sys.argv) > 1:
    root = Path(sys.argv[1])
    for p, digest in validation['source_sha256'].items():
        assert sha((root / p).read_bytes()) == digest, p
print(f'Verified {len(manifest)} payloads,540 scalar cases per backend, executed allocation controls and old/new source restoration.')
