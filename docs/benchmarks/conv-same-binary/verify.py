"""Verify independent Conv factories, executed falsifiers and unchanged default source."""
from pathlib import Path
import gzip
import hashlib
import json
import sys

folder = Path(__file__).resolve().parent
sha = lambda b: hashlib.sha256(b).hexdigest()
read = lambda p: gzip.decompress((folder / p).read_bytes())
manifest = json.loads((folder / 'manifest.json').read_text())
actual = {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()
          and p.name != 'manifest.json' and '__pycache__' not in p.parts}
assert actual == set(manifest)
for path, digest in manifest.items():
    assert sha((folder / path).read_bytes()) == digest, path
validation = json.loads((folder / 'validation/validation.json').read_text())
assert len(validation['results']) == 8
for result in validation['results']:
    assert result['exit'] == 0
    log = read('validation/' + result['name'] + '.log.gz')
    assert sha(log) == result['log_sha256']
    if result['name'] in ('default-conv', 'native-linked'):
        assert result['counts'] == [['46', '0', '1']]
    elif result['name'] in ('default-registry', 'allocation', 'pair'):
        assert result['counts'] == [['1', '0', '0']]
    if result['name'] == 'pair':
        assert result['extra_env'] == {'NXRT_CPU_GEMM_BACKEND': 'mlas'}
        assert b'compared 36 native/MLAS cases with two executions per factory; native empty batch also executed' in log
    if result['name'] == 'pair-list':
        assert b'linked::linked_conv_factories_are_independent_and_match_f64_on_real_execute: test' in log
sources = {p: read('source/' + p + '.gz') for p in validation['source_sha256']}
for p, body in sources.items():
    assert sha(body) == validation['source_sha256'][p], p
controls = json.loads((folder / 'negative/controls.json').read_text())
assert [r['name'] for r in controls] == ['alias-reference', 'ambient-mlas']
for result in controls:
    original = sources[result['source_path']]
    mutant = read('negative/' + result['name'] + '.rs.gz')
    log = read('negative/' + result['name'] + '.log.gz')
    assert sha(original) == result['original_sha256'] and sha(mutant) == result['mutant_sha256']
    assert sha(log) == result['log_sha256']
    assert result['expected_failure'] and result['exit'] != 0
    assert result['extra_env'] == {'NXRT_CPU_GEMM_BACKEND': 'mlas'}
    assert b'0 passed; 1 failed' in log and b'error[E' not in log
    if result['name'] == 'alias-reference':
        a = b'native_generic_gemm: Box::new(crate::kernels::conv_native::ConvFactory),'
        b = b'native_generic_gemm: Box::new(crate::kernels::conv::ConvFactory),'
        assert b'native and reference must be independent concrete kernels' in log
    else:
        a = b'let backend = crate::backend::CpuBackend::Generic;'
        b = b'let backend = crate::backend::CpuBackend::auto_detect();'
        assert b'native Conv must not route GEMM to MLAS' in log
    assert original.count(a) == 1 and original.replace(a, b) == mutant
assert json.loads((folder / 'negative/restoration.json').read_text()) == validation['source_sha256']
p = 'crates/onnx-runtime-ep-cpu/src/kernels/conv_ref.rs'
s = sources[p]
a = s.index(b'    // This source is the production Conv without MLAS.')
b = s.index(b'    let mut output =', a)
base = read('baseline-conv.rs.gz')
assert s[:a] + b'    let backend = crate::backend::CpuBackend::auto_detect();\n' + s[b:] == base
receipt = json.loads((folder / 'default-reconstruction.json').read_text())
assert receipt['default_conv_reconstructs_base_byte_for_byte']
assert receipt['base_conv_sha256'] == sha(base) and receipt['final_conv_sha256'] == sha(s)
if len(sys.argv) > 1:
    root = Path(sys.argv[1])
    for p, digest in validation['source_sha256'].items():
        assert sha((root / p).read_bytes()) == digest, p
print(f'Verified {len(manifest)} payloads, actual paired executes, distinct-kernel/backend controls and default Conv reconstruction.')
