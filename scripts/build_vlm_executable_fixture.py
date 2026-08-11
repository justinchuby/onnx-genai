"""Generate the tiny executable ONNX graphs for the `vlm-executable` compat fixture.

The server's sidecar-free VLM compatibility test
(`sidecar_free_compatibility_package_builds_server_pipeline_and_preprocesses_image`)
loads `crates/onnx-genai-genai-config/tests/fixtures/vlm-executable/` as a real
split-VLM package. `genai_config.json` names three graphs whose interfaces this
script materializes as the smallest possible, rank-consistent identity graphs:

* vision.onnx    pixel_values[num_patches,768] f32, image_grid_thw[img,3] i64
                 -> image_features[num_patches,768] f32   (flat [num_image_tokens, hidden])
* embedding.onnx input_ids[batch,seq] i64, image_features[num_image_tokens,768] f32
                 -> inputs_embeds[batch,seq,1] f32        (raises token ids to embeds)
* text.onnx      inputs_embeds[batch,seq,1] f32, attention_mask[batch,total] i64,
                 position_ids[3,batch,seq] i64, past_key_values.0.{key,value}[batch,2,past,64] f32
                 -> logits[batch,seq,1] f32, present.0.{key,value} f32

These graphs only need to load (build the server pipeline) and let image
preprocessing run; the test never executes decode, so the bodies are trivial.
"""

from pathlib import Path

import onnx
from onnx import TensorProto, helper

FIXTURE = (
    Path(__file__).resolve().parent.parent
    / "crates/onnx-genai-genai-config/tests/fixtures/vlm-executable"
)

OPSET = 18
IR_VERSION = 10


def vi(name, elem_type, shape):
    return helper.make_tensor_value_info(name, elem_type, shape)


def save(model, path):
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    print(f"wrote {path} ({path.stat().st_size} bytes)")


def build_vision():
    pixel_values = vi("pixel_values", TensorProto.FLOAT, ["num_patches", 768])
    image_grid_thw = vi("image_grid_thw", TensorProto.INT64, ["image_count", 3])
    image_features = vi("image_features", TensorProto.FLOAT, ["num_patches", 768])
    node = helper.make_node("Identity", ["pixel_values"], ["image_features"])
    graph = helper.make_graph(
        [node], "vision", [pixel_values, image_grid_thw], [image_features]
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])


def build_embedding():
    input_ids = vi("input_ids", TensorProto.INT64, ["batch", "sequence"])
    image_features = vi(
        "image_features", TensorProto.FLOAT, ["num_image_tokens", 768]
    )
    inputs_embeds = vi("inputs_embeds", TensorProto.FLOAT, ["batch", "sequence", 1])
    axes = helper.make_tensor("unsqueeze_axes", TensorProto.INT64, [1], [2])
    cast = helper.make_node("Cast", ["input_ids"], ["input_ids_f32"], to=TensorProto.FLOAT)
    unsqueeze = helper.make_node(
        "Unsqueeze", ["input_ids_f32", "unsqueeze_axes"], ["inputs_embeds"]
    )
    graph = helper.make_graph(
        [cast, unsqueeze],
        "embedding",
        [input_ids, image_features],
        [inputs_embeds],
        initializer=[axes],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])


def build_text():
    inputs_embeds = vi("inputs_embeds", TensorProto.FLOAT, ["batch", "sequence", 1])
    attention_mask = vi("attention_mask", TensorProto.INT64, ["batch", "total_sequence"])
    position_ids = vi("position_ids", TensorProto.INT64, [3, "batch", "sequence"])
    past_key = vi(
        "past_key_values.0.key", TensorProto.FLOAT, ["batch", 2, "past_sequence", 64]
    )
    past_value = vi(
        "past_key_values.0.value", TensorProto.FLOAT, ["batch", 2, "past_sequence", 64]
    )
    logits = vi("logits", TensorProto.FLOAT, ["batch", "sequence", 1])
    present_key = vi(
        "present.0.key", TensorProto.FLOAT, ["batch", 2, "past_sequence", 64]
    )
    present_value = vi(
        "present.0.value", TensorProto.FLOAT, ["batch", 2, "past_sequence", 64]
    )
    nodes = [
        helper.make_node("Identity", ["inputs_embeds"], ["logits"]),
        helper.make_node("Identity", ["past_key_values.0.key"], ["present.0.key"]),
        helper.make_node("Identity", ["past_key_values.0.value"], ["present.0.value"]),
    ]
    graph = helper.make_graph(
        nodes,
        "text",
        [inputs_embeds, attention_mask, position_ids, past_key, past_value],
        [logits, present_key, present_value],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])


def main():
    save(build_vision(), FIXTURE / "vision.onnx")
    save(build_embedding(), FIXTURE / "embedding.onnx")
    save(build_text(), FIXTURE / "text.onnx")


if __name__ == "__main__":
    main()
