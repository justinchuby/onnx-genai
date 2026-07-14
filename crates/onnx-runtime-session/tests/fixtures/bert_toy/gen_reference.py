#!/usr/bin/env python3
"""Generate deterministic ORT reference fixtures for the bert_toy conformance test.

Loads ``model.onnx`` via onnxruntime's CPUExecutionProvider, feeds a set of
hardcoded (deterministic) inputs, runs inference, and writes both the inputs
and the reference outputs to committed fixture files:

  * ``<name>.bin``  -- raw little-endian element bytes for each input/output.
  * ``manifest.json`` -- names, shapes, dtypes, and the literal input values.

The Rust conformance test reads these ``.bin`` files directly, so no Python /
onnxruntime is required at test time. Re-run this script only to regenerate the
reference (e.g. after intentionally changing the inputs):

    python3 gen_reference.py

Inputs (batch=1, seq=8):
  input_ids      = [7, 3, 19, 2, 25, 11, 0, 14]   (toy vocab, small ids)
  token_type_ids = [0, 0, 0, 0, 0, 0, 0, 0]
  input_mask     = [1, 1, 1, 1, 1, 1, 1, 1]
"""

import json
import pathlib

import numpy as np
import onnxruntime as ort

HERE = pathlib.Path(__file__).resolve().parent
MODEL = HERE / "model.onnx"

BATCH = 1
SEQ = 8

# Deterministic, hardcoded inputs. int64 as required by the model. ids are kept
# small (< toy vocab) so the embedding gather is in-bounds.
INPUTS = {
    "input_ids": np.array([[7, 3, 19, 2, 25, 11, 0, 14]], dtype=np.int64),
    "token_type_ids": np.zeros((BATCH, SEQ), dtype=np.int64),
    "input_mask": np.ones((BATCH, SEQ), dtype=np.int64),
}

# ONNX TensorProto elem_type -> (numpy dtype, string tag used in manifest).
DTYPE_TAG = {
    np.dtype("int64"): "int64",
    np.dtype("float32"): "float32",
}


def write_array(name: str, arr: np.ndarray) -> dict:
    arr = np.ascontiguousarray(arr)
    path = HERE / f"{name}.bin"
    path.write_bytes(arr.tobytes(order="C"))
    return {
        "name": name,
        "file": f"{name}.bin",
        "dtype": DTYPE_TAG[arr.dtype],
        "shape": list(arr.shape),
    }


def main() -> None:
    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])

    output_names = [o.name for o in sess.get_outputs()]
    outputs = sess.run(output_names, {k: v for k, v in INPUTS.items()})

    manifest = {
        "model": "model.onnx",
        "provider": "CPUExecutionProvider",
        "onnxruntime_version": ort.__version__,
        "batch": BATCH,
        "seq": SEQ,
        "inputs": [],
        "outputs": [],
        "input_values": {k: v.tolist() for k, v in INPUTS.items()},
    }

    for name, arr in INPUTS.items():
        manifest["inputs"].append(write_array(name, arr))

    for name, arr in zip(output_names, outputs):
        entry = write_array(name, arr.astype(np.float32, copy=False))
        manifest["outputs"].append(entry)
        print(f"{name}: shape={arr.shape} dtype={arr.dtype} "
              f"min={arr.min():.6f} max={arr.max():.6f} mean={arr.mean():.6f}")

    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote manifest with {len(manifest['inputs'])} inputs, "
          f"{len(manifest['outputs'])} outputs")


if __name__ == "__main__":
    main()
