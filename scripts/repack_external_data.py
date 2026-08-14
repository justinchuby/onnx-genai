"""Repack an ONNX model's external data so only referenced bytes are stored.

Streams byte ranges from the old blob to the new one and rewrites offsets in
the (small) model proto. Never loads tensor data into memory, so it works on
blobs far larger than RAM.

Why this exists
---------------
`onnx.external_data_helper.save_external_data` opens the target with ``"r+b"``
and immediately seeks to the end, so it **appends**. An exporter that saves
twice into the same directory — an fp16 pass followed by a quantization pass, a
retry, a re-run over an existing output — silently doubles the blob and leaves
the first generation referenced by nothing. The model still loads and still
produces correct output, which is why it goes unnoticed.

`models/qwen14b-zp` was affected: 16.652 GB on disk, of which a contiguous
8.323 GB prefix (50.02%) was unreferenced. That cost 2x disk, 2x download, and
2x pinned host RAM under the memory-mapped weight path, which registers the
whole mapping (`host_registered_bytes = 16,652,453,888` for 8.33 GB of live
weights). It also produced a wrong number: sizing weight budgets from file size
rather than referenced extents made the model look 2.00x larger than it is
(#853, fixed in #856). Reported upstream as onnxruntime/mobius#488.

Verify the result, do not assume it: a repacked model must produce byte-identical
token IDs to the original on the same prompt and binary.

Usage: python scripts/repack_external_data.py SRC_DIR DST_DIR
"""

import shutil
import sys
from pathlib import Path

import onnx

ALIGN = 4096
COPY_CHUNK = 32 << 20

if len(sys.argv) != 3:
    sys.exit(f"usage: python {sys.argv[0]} SRC_DIR DST_DIR")

src_dir = Path(sys.argv[1])
dst_dir = Path(sys.argv[2])
dst_dir.mkdir(parents=True, exist_ok=True)

model = onnx.load(str(src_dir / "model.onnx"), load_external_data=False)

spans = []
for init in model.graph.initializer:
    if init.data_location != onnx.TensorProto.EXTERNAL:
        continue
    entry = {kv.key: kv.value for kv in init.external_data}
    spans.append(
        (int(entry.get("offset", 0)), int(entry.get("length", 0)), entry["location"], init)
    )

locations = {loc for _, _, loc, _ in spans}
if len(locations) != 1:
    sys.exit(f"expected exactly one external blob, found {locations}")
location = locations.pop()

src_blob = src_dir / location
dst_blob = dst_dir / location
src_size = src_blob.stat().st_size

# Preserve the original relative ordering.
spans.sort(key=lambda s: s[0])

written = 0
with open(src_blob, "rb") as fin, open(dst_blob, "wb") as fout:
    for offset, length, _, init in spans:
        pad = (-written) % ALIGN
        if pad:
            fout.write(b"\0" * pad)
            written += pad
        new_offset = written
        fin.seek(offset)
        remaining = length
        while remaining:
            chunk = fin.read(min(COPY_CHUNK, remaining))
            if not chunk:
                sys.exit(f"unexpected EOF reading {init.name}")
            fout.write(chunk)
            remaining -= len(chunk)
        written += length

        del init.external_data[:]
        for key, value in (
            ("location", location),
            ("offset", str(new_offset)),
            ("length", str(length)),
        ):
            entry = init.external_data.add()
            entry.key = key
            entry.value = value

with open(dst_dir / "model.onnx", "wb") as fout:
    fout.write(model.SerializeToString())

for name in (
    "config.json",
    "genai_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "inference_metadata.yaml",
):
    candidate = src_dir / name
    if candidate.is_file():
        shutil.copy2(candidate, dst_dir / name)

print(f"source blob : {src_size:,} bytes ({src_size / 1e9:.3f} GB)")
print(f"repacked    : {written:,} bytes ({written / 1e9:.3f} GB)")
print(f"reduction   : {100 * (1 - written / src_size):.2f}%")
print(f"tensors     : {len(spans)}")
