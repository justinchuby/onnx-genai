"""Generate the tiny external-data QMoE model with the ONNX IR API."""

from pathlib import Path

import numpy as np
from onnxscript import ir


EXPERTS = 4
HIDDEN = 16
INTER = 16
BITS = 4
BLOCK_SIZE = 16
PACK_SIZE = 8 // BITS


def value(name: str, dtype: ir.DataType, shape: list[int], data=None) -> ir.Value:
    return ir.Value(
        name=name,
        type=ir.TensorType(dtype),
        shape=ir.Shape(shape),
        const_value=None if data is None else ir.tensor(data, name=name),
    )


def quantize(out_features: int, in_features: int) -> tuple[np.ndarray, np.ndarray]:
    blocks = in_features // BLOCK_SIZE
    packed_in = in_features // PACK_SIZE
    packed = np.zeros((EXPERTS, out_features, packed_in), dtype=np.uint8)
    scales = np.zeros((EXPERTS, out_features, blocks), dtype=np.float32)
    zero_point = 1 << (BITS - 1)
    for expert in range(EXPERTS):
        for row in range(out_features):
            for block in range(blocks):
                scale = 0.25 + 0.125 * ((expert + row + block) % 3)
                scales[expert, row, block] = scale
                for offset in range(BLOCK_SIZE):
                    depth = block * BLOCK_SIZE + offset
                    centered = (expert * 3 + row * 5 + depth * 7) % 7 - 3
                    quantized = (centered + zero_point) & 0xF
                    packed[expert, row, depth // PACK_SIZE] |= np.uint8(
                        quantized << ((depth % PACK_SIZE) * BITS)
                    )
    return packed, scales


def main() -> None:
    output_dir = Path(__file__).parent
    fc1_packed_data, fc1_scales_data = quantize(INTER, HIDDEN)
    fc2_packed_data, fc2_scales_data = quantize(HIDDEN, INTER)

    x = value("X", ir.DataType.FLOAT, [4, HIDDEN])
    router = value("router", ir.DataType.FLOAT, [4, EXPERTS])
    fc1_packed = value(
        "fc1_packed", ir.DataType.UINT8, list(fc1_packed_data.shape), fc1_packed_data
    )
    fc1_scales = value(
        "fc1_scales", ir.DataType.FLOAT, list(fc1_scales_data.shape), fc1_scales_data
    )
    fc2_packed = value(
        "fc2_packed", ir.DataType.UINT8, list(fc2_packed_data.shape), fc2_packed_data
    )
    fc2_scales = value(
        "fc2_scales", ir.DataType.FLOAT, list(fc2_scales_data.shape), fc2_scales_data
    )
    y = value("Y", ir.DataType.FLOAT, [4, HIDDEN])
    node = ir.Node(
        "com.microsoft",
        "QMoE",
        [x, router, fc1_packed, fc1_scales, None, fc2_packed, fc2_scales],
        [
            ir.AttrInt64("expert_weight_bits", BITS),
            ir.AttrInt64("block_size", BLOCK_SIZE),
            ir.AttrInt64("k", 1),
            ir.AttrString("activation_type", "identity"),
            ir.AttrInt64("normalize_routing_weights", 0),
        ],
        outputs=[y],
        version=1,
    )
    graph = ir.Graph(
        [x, router],
        [y],
        nodes=[node],
        initializers=[fc1_packed, fc1_scales, fc2_packed, fc2_scales],
        opset_imports={"": 17, "com.microsoft": 1},
        name="qmoe_weight_offload",
    )
    proto = ir.to_proto(ir.Model(graph, ir_version=10))

    payloads = {
        "fc1_packed": fc1_packed_data.tobytes(),
        "fc1_scales": fc1_scales_data.tobytes(),
        "fc2_packed": fc2_packed_data.tobytes(),
        "fc2_scales": fc2_scales_data.tobytes(),
    }
    # Deliberately dtype-aligned but not DeviceBuffer's 64-byte alignment.
    external = bytearray(4)
    for initializer in proto.graph.initializer:
        payload = payloads[initializer.name]
        offset = len(external)
        external.extend(payload)
        initializer.ClearField("raw_data")
        initializer.ClearField("float_data")
        initializer.ClearField("int32_data")
        initializer.data_location = 1
        del initializer.external_data[:]
        for key, val in (
            ("location", "weights.bin"),
            ("offset", str(offset)),
            ("length", str(len(payload))),
        ):
            entry = initializer.external_data.add()
            entry.key = key
            entry.value = val

    (output_dir / "model.onnx").write_bytes(proto.SerializeToString())
    (output_dir / "weights.bin").write_bytes(external)


if __name__ == "__main__":
    main()
