# onnx-std Python bindings

`onnx_std` is the Python extension for the `onnx-std` ONNX serialization
library. It provides an opaque `Model` handle plus binary protobuf, readable
text, protobuf JSON, and protobuf TextFormat round-trips.

The distribution name is `onnx-std`; the import name is `onnx_std`, so it does
not collide with the `onnx` package.

## Install

From this directory, install directly with pip:

```bash
python -m pip install .
```

For development, create/activate a virtual environment and use maturin:

```bash
python -m pip install maturin
maturin develop
```

The extension uses PyO3's stable ABI with a minimum CPython version of 3.10.

## Quickstart

```python
from pathlib import Path

import onnx_std

model = onnx_std.load_model("model.onnx")  # str, os.PathLike, or bytes

# onnx-std readable text DSL
text = onnx_std.to_text(model)
model_from_text = onnx_std.from_text(text)

# Canonical ONNX protobuf JSON
json_document = onnx_std.to_json(model)
model_from_json = onnx_std.from_json(json_document)

# Protobuf TextFormat (.onnxtxt/.pbtxt)
textproto = onnx_std.to_textproto(model)
model_from_textproto = onnx_std.from_textproto(textproto)

onnx_std.save_model(model_from_textproto, Path("roundtrip.onnx"))
```

The readable text DSL describes graph structure but intentionally replaces
initializer payloads with typed placeholders. Use JSON, TextProto, or binary
protobuf when tensor bytes must be preserved.

Raw protobuf bytes are accepted directly:

```python
model = onnx_std.load_model(Path("model.onnx").read_bytes())
```

Pass a filesystem path, rather than bytes, for models with external tensor
data so relative external-data paths resolve beside the model.
