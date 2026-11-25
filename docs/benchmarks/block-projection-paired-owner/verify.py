"""Verify archived diagnostic evidence without running benchmarks."""
from pathlib import Path
import gzip, hashlib, importlib.util, json, math, struct, sys
root = Path(__file__).resolve().parent
sha = lambda b: hashlib.sha256(b).hexdigest()
manifest = json.loads((root / 'manifest.json').read_text())
actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name != 'manifest.json' and '__pycache__' not in p.parts}
assert actual == set(manifest)
for p, h in manifest.items():
    assert sha((root / p).read_bytes()) == h, p
meta = json.loads((root / 'metadata.json').read_text())
raw = gzip.decompress((root / 'receipts.json.gz').read_bytes())
assert sha(raw) == meta['receipts_sha256']
assert sha((root / 'probe/src/main.rs').read_bytes()) == meta['probe_sha256']
assert sha((root / 'probe/Cargo.lock').read_bytes()) == meta['probe_lock_sha256']
assert sha((root / 'run.py').read_bytes()) == meta['runner_sha256']
assert meta['binary_sha256'] == meta['binary_after_sha256'] and len(meta['single_decode_symbol']) == 1
spec = importlib.util.spec_from_file_location('reader', root / 'analyze.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
rows = json.loads(raw)
assert m.analyze(rows) == json.loads((root / 'results.json').read_text())
old = json.loads(gzip.decompress((root.parent / 'block-projection-execution-context/study/receipts.json.gz').read_bytes()))
expected = {r['output_sha256'] for r in old if r['case'] == 9}
for r in rows:
    name = 'r' + str(r['repetition'])
    buf = gzip.decompress((root / (name + '.f32.gz')).read_bytes())
    assert len(buf) == 8192 and sha(buf) == r['output_sha256'] and {sha(buf)} == expected
    values = struct.unpack('<2048f', buf)
    assert all(math.isfinite(v) for v in values) and any(v != 0 for v in values)
    receipt = json.loads(gzip.decompress((root / (name + '.stdout.gz')).read_bytes()).strip().splitlines()[-1])
    assert all(r[k] == v for k, v in receipt.items())
    assert not gzip.decompress((root / (name + '.stderr.gz')).read_bytes())
controls = json.loads((root / 'reader-controls.json').read_text())
assert controls['reader_sha256'] == sha((root / 'analyze.py').read_bytes())
assert controls['synthetic_qualified'] and controls['zero_empty_only']
assert len(controls['controls']) == 11 and all(r['rejected'] for r in controls['controls'])
assert len(controls['quality_controls']) == 2 and all(r['failed_aa_preserved'] for r in controls['quality_controls'])
seal = json.loads((root / 'source-seal.json').read_text())
validated = json.loads((root.parent / 'block-projection-context-controls/cache-validation/validation.json').read_text())
assert seal['validated_production_sources'] == validated['source_sha256']
assert all(r['exit'] == 0 for r in validated['results'])
assert all(seal['validated_production_sources'][p] == h for p,h in meta['source_sha256'].items())
if len(sys.argv) > 1:
    for p,h in seal['validated_production_sources'].items():
        assert sha((Path(sys.argv[1]) / p).read_bytes()) == h, p
print(f'Verified {len(manifest)} sealed payloads, 72 windows, three complete outputs, and production source seal.')
