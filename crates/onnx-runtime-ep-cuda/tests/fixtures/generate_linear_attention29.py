"""Freeze ONNX's deterministic LinearAttention-29 backend vectors.

Requires NumPy and ONNX. Run from any directory; writes the sibling fixture.
The pinned upstream source is Apache-2.0 (ONNX Project Contributors).
"""

from pathlib import Path
from urllib.request import urlopen

import numpy as np

PIN = "622adf94d4e720142950aa8178f7357d52e63db0"
BASE = f"https://raw.githubusercontent.com/onnx/onnx/{PIN}/onnx"


def source(path):
    with urlopen(f"{BASE}/{path}") as response:
        return response.read().decode()


def generate():
    ref = {}
    exec(source("reference/ops/op_linear_attention.py"), ref)
    lines = source("backend/test/case/node/linear_attention.py")
    start = lines.index("from onnx.reference.ops.op_linear_attention")
    lines = lines[:start] + lines[lines.index("\n_OPSET", start):]
    lines = lines.replace("from onnx.backend.test.case.base import Base\n", "")
    lines = lines.replace("from onnx.backend.test.case.node import expect\n", "")
    fixture = bytearray()

    def expect(node, inputs, outputs, name, opset_imports):
        assert opset_imports[0].version == 29
        attrs = {a.name: a for a in node.attribute}
        name = name.removeprefix("test_linear_attention_").encode()
        rule = attrs["update_rule"].s if "update_rule" in attrs else b"gated_delta"
        fixture.extend(bytes([len(name), len(rule), len(outputs)]))
        fixture.extend(name + rule)
        fixture.extend(attrs[a].i for a in ("q_num_heads", "kv_num_heads", "v_num_heads"))
        packed_inputs = iter(inputs)
        for array in [*(next(packed_inputs) if name else None for name in node.input), *outputs]:
            if array is None:
                fixture.append(0)
                continue
            width = array.dtype.itemsize
            assert width in (2, 4)
            fixture.extend(bytes([array.ndim, width, *array.shape]))
            fixture.extend(array.astype(np.dtype(f"<f{width}"), copy=False).tobytes())

    ns = dict(_RefLinearAttention=ref["LinearAttention"], Base=object, expect=expect)
    exec(lines, ns)
    ns["LinearAttention"].export_state_trajectory()
    ns["LinearAttention"].export_compact_value_heads()
    Path(__file__).with_name("linear_attention29.bin").write_bytes(fixture)


if __name__ == "__main__":
    generate()
