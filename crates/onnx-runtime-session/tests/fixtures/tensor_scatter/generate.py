"""Generate the TensorScatter fixture and ONNX Runtime CPU reference tensors."""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from google.protobuf import text_format
from onnx import TensorProto, helper


FIXTURE_DIRECTORY = Path(__file__).resolve().parent
CACHE_SHAPE = [2, 5, 2, 2]
UPDATES_SHAPE = [2, 2, 2, 2]


def build_model() -> onnx.ModelProto:
    graph = helper.make_graph(
        [
            helper.make_node(
                "TensorScatter",
                ["cache", "updates", "write_indices"],
                ["updated_cache"],
                axis=1,
            )
        ],
        "tensor_scatter_parity",
        [
            helper.make_tensor_value_info("cache", TensorProto.FLOAT, CACHE_SHAPE),
            helper.make_tensor_value_info("updates", TensorProto.FLOAT, UPDATES_SHAPE),
            helper.make_tensor_value_info(
                "write_indices", TensorProto.INT64, [CACHE_SHAPE[0]]
            ),
        ],
        [
            helper.make_tensor_value_info(
                "updated_cache", TensorProto.FLOAT, CACHE_SHAPE
            )
        ],
    )
    return helper.make_model(
        graph,
        producer_name="onnx-genai TensorScatter parity fixture",
        opset_imports=[helper.make_opsetid("", 24)],
        ir_version=11,
    )


def main() -> None:
    model = build_model()
    onnx.checker.check_model(model)
    (FIXTURE_DIRECTORY / "model.onnx.textproto").write_text(
        text_format.MessageToString(model), encoding="utf-8"
    )

    cache = np.arange(np.prod(CACHE_SHAPE), dtype=np.float32).reshape(CACHE_SHAPE)
    updates = (
        np.arange(np.prod(UPDATES_SHAPE), dtype=np.float32).reshape(UPDATES_SHAPE)
        + np.float32(1000.0)
    )
    write_indices = np.asarray([1, 3], dtype=np.int64)
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    updated_cache = session.run(
        None,
        {
            "cache": cache,
            "updates": updates,
            "write_indices": write_indices,
        },
    )[0]

    expected = cache.copy()
    expected[0, 1:3, :, :] = updates[0]
    expected[1, 3:5, :, :] = updates[1]
    np.testing.assert_array_equal(updated_cache, expected)

    cache.tofile(FIXTURE_DIRECTORY / "cache.f32.bin")
    updates.tofile(FIXTURE_DIRECTORY / "updates.f32.bin")
    write_indices.tofile(FIXTURE_DIRECTORY / "write_indices.i64.bin")
    updated_cache.tofile(FIXTURE_DIRECTORY / "updated_cache.ort.f32.bin")
    print("TensorScatter ORT parity fixture: 40/40 float32 elements exact")


if __name__ == "__main__":
    main()
