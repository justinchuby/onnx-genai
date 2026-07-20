#!/usr/bin/env python3
"""Build the tiny deterministic **nested-autoregressive** (multi-decoder TTS)
pipeline fixture — the Qwen3-TTS-style shape the flat composite contract could
not express (DESIGN.md §20.3).

It is a *dual, hierarchically-nested* autoregressive loop:

  * `talker` (outer AR decoder): one outer step == one audio frame. A KV-cached
    decoder whose logits are `-(vocab_index - position)^2` (so the argmax first
    code group == position), and which also emits a per-frame
    `last_hidden_state[1, seq, HIDDEN]` equal to the position broadcast across
    the hidden dim. Starting from a 1-token prompt, frame `f` has position `f`,
    so `seed_f == f`.
  * `code_predictor` (inner AR decoder): for each talker frame it runs an inner
    AR loop of `num_code_groups` steps over the residual codebooks. Its logits
    are `-(vocab_index - (mean(inputs_embeds) + 1))^2`, so
    `code == mean(inputs_embeds) + 1`. It also emits `code_embeds`, the emitted
    code broadcast across the hidden dim, which the engine threads back into
    `inputs_embeds` for the next inner step (inner step 0 is seeded by the
    talker's `last_hidden_state`). Hence:
        - inner step 0: mean == seed_f == f            -> code = f + 1
        - inner step g: mean == prev code == f + g     -> code = f + g + 1
    so `code[f][g] == f + g + 1`.
  * `vocoder` (`final_only` single_pass): `codes[1, F, G] (int64) -> audio[1,
    F*G]` with `audio == 2 * flatten(codes)`.

With `num_code_groups = 4` and `max_frames = 3` the assembled codes are
`[[1,2,3,4],[2,3,4,5],[3,4,5,6]]` and the waveform is
`2 * [1,2,3,4,2,3,4,5,3,4,5,6] = [2,4,6,8,4,6,8,10,6,8,10,12]`.

The generated codes are published by the engine into the shared pool as the
synthetic tensor `talker.output_codes` of shape `[1, F, G]` (int64), routed to
the vocoder by the dataflow edge `talker.output_codes -> vocoder.codes`. The
per-frame hidden binding is the edge `talker.last_hidden_state ->
code_predictor.inputs_embeds` (inner step 0 seed).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnxscript import ir

VOCAB = 16
HEAD_DIM = 4
HIDDEN = 8
NUM_CODE_GROUPS = 4
MAX_FRAMES = 3


def tensor_value(name: str, dtype: ir.DataType, shape: list[int | str]) -> ir.Value:
    return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))


def initializer(name: str, array: np.ndarray) -> ir.Value:
    return ir.Value(name=name, const_value=ir.Tensor(array, name=name))


def node(
    op_type: str,
    inputs: list[ir.Value],
    output: str,
    *,
    attributes: tuple[ir.Attr, ...] | list[ir.Attr] = (),
) -> ir.Node:
    return ir.Node(
        "",
        op_type,
        inputs,
        attributes,
        outputs=[ir.Value(name=output)],
    )


def constant(name: str, array: np.ndarray) -> ir.Node:
    return node(
        "Constant",
        [],
        name,
        attributes=[ir.AttrTensor("value", ir.Tensor(array, name=f"{name}_value"))],
    )


def save_model(model: ir.Model, path: Path) -> None:
    ir.save(model, path)
    onnx.checker.check_model(path)


def build_talker(path: Path) -> None:
    """Outer AR decoder: logits argmax == position; also emits last_hidden_state.

    logits[b, s, v] = -(v - position[b, s])^2  =>  argmax_v == position[b, s].
    last_hidden_state[b, s, :] = position[b, s] (broadcast across HIDDEN). The KV
    outputs mirror the Whisper contract; their values are inert.
    """
    decoder_input_ids = tensor_value(
        "decoder_input_ids", ir.DataType.INT64, ["batch", "sequence_len"]
    )
    position_ids = tensor_value(
        "position_ids", ir.DataType.INT64, ["batch", "sequence_len"]
    )
    past_key = tensor_value(
        "past_key_values.0.key",
        ir.DataType.FLOAT,
        [1, 1, "past_sequence_len", HEAD_DIM],
    )
    past_value = tensor_value(
        "past_key_values.0.value",
        ir.DataType.FLOAT,
        [1, 1, "past_sequence_len", HEAD_DIM],
    )

    # --- KV cache (Whisper-style; values are inert placeholders) ---
    input_shape = node("Shape", [decoder_input_ids], "input_shape")
    batch_index = constant("batch_index", np.array(0, dtype=np.int64))
    sequence_index = constant("sequence_index", np.array(1, dtype=np.int64))
    axes_zero = constant("axes_zero", np.array([0], dtype=np.int64))
    batch = node(
        "Gather",
        [input_shape.outputs[0], batch_index.outputs[0]],
        "batch",
        attributes=[ir.AttrInt64("axis", 0)],
    )
    sequence = node(
        "Gather",
        [input_shape.outputs[0], sequence_index.outputs[0]],
        "sequence",
        attributes=[ir.AttrInt64("axis", 0)],
    )
    batch_vec = node("Unsqueeze", [batch.outputs[0], axes_zero.outputs[0]], "batch_vec")
    sequence_vec = node(
        "Unsqueeze", [sequence.outputs[0], axes_zero.outputs[0]], "sequence_vec"
    )
    one = constant("one", np.array([1], dtype=np.int64))
    head = constant("head", np.array([HEAD_DIM], dtype=np.int64))
    hidden = constant("hidden", np.array([HIDDEN], dtype=np.int64))
    cache_shape = node(
        "Concat",
        [batch_vec.outputs[0], one.outputs[0], sequence_vec.outputs[0], head.outputs[0]],
        "cache_shape",
        attributes=[ir.AttrInt64("axis", 0)],
    )
    ids_float = node(
        "Cast",
        [decoder_input_ids],
        "ids_float",
        attributes=[ir.AttrInt64("to", int(ir.DataType.FLOAT))],
    )
    cache_axes = constant("cache_axes", np.array([1, 3], dtype=np.int64))
    ids_cache = node(
        "Unsqueeze", [ids_float.outputs[0], cache_axes.outputs[0]], "ids_cache"
    )
    current_key = node(
        "Expand", [ids_cache.outputs[0], cache_shape.outputs[0]], "current_key"
    )
    value_offset = initializer("value_offset", np.array(0.5, dtype=np.float32))
    current_value = node("Add", [current_key.outputs[0], value_offset], "current_value")
    present_key = node(
        "Concat",
        [past_key, current_key.outputs[0]],
        "present.0.key",
        attributes=[ir.AttrInt64("axis", 2)],
    )
    present_value = node(
        "Concat",
        [past_value, current_value.outputs[0]],
        "present.0.value",
        attributes=[ir.AttrInt64("axis", 2)],
    )

    # --- Position-indexed logits: argmax_v (-(v - position)^2) == position ---
    position_float = node(
        "Cast",
        [position_ids],
        "position_float",
        attributes=[ir.AttrInt64("to", int(ir.DataType.FLOAT))],
    )
    logits_axis = constant("logits_axis", np.array([2], dtype=np.int64))
    position_col = node(
        "Unsqueeze", [position_float.outputs[0], logits_axis.outputs[0]], "position_col"
    )
    vocab_index = initializer(
        "vocab_index",
        np.arange(VOCAB, dtype=np.float32).reshape(1, 1, VOCAB),
    )
    diff = node("Sub", [vocab_index, position_col.outputs[0]], "diff")
    diff_sq = node("Mul", [diff.outputs[0], diff.outputs[0]], "diff_sq")
    logits = node("Neg", [diff_sq.outputs[0]], "logits")
    logits.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    logits.outputs[0].shape = ir.Shape(["batch", "sequence_len", VOCAB])

    # --- Per-frame hidden state: position broadcast across HIDDEN ---
    hidden_shape = node(
        "Concat",
        [batch_vec.outputs[0], sequence_vec.outputs[0], hidden.outputs[0]],
        "hidden_shape",
        attributes=[ir.AttrInt64("axis", 0)],
    )
    last_hidden_state = node(
        "Expand", [position_col.outputs[0], hidden_shape.outputs[0]], "last_hidden_state"
    )
    last_hidden_state.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    last_hidden_state.outputs[0].shape = ir.Shape(["batch", "sequence_len", HIDDEN])

    for output in (present_key.outputs[0], present_value.outputs[0]):
        output.type = ir.TensorType(ir.DataType.FLOAT)
        output.shape = ir.Shape([1, 1, "total_sequence_len", HEAD_DIM])

    nodes = [
        input_shape,
        batch_index,
        sequence_index,
        axes_zero,
        batch,
        sequence,
        batch_vec,
        sequence_vec,
        one,
        head,
        hidden,
        cache_shape,
        ids_float,
        cache_axes,
        ids_cache,
        current_key,
        current_value,
        present_key,
        present_value,
        position_float,
        logits_axis,
        position_col,
        diff,
        diff_sq,
        logits,
        hidden_shape,
        last_hidden_state,
    ]
    graph = ir.Graph(
        [decoder_input_ids, position_ids, past_key, past_value],
        [
            logits.outputs[0],
            last_hidden_state.outputs[0],
            present_key.outputs[0],
            present_value.outputs[0],
        ],
        nodes=nodes,
        initializers=[value_offset, vocab_index],
        opset_imports={"": 13},
        name="tiny_tts_nested_talker",
    )
    save_model(
        ir.Model(graph, ir_version=8, producer_name="onnx-genai tiny-tts-nested fixture"),
        path,
    )


def build_code_predictor(path: Path) -> None:
    """Inner AR decoder: code == mean(inputs_embeds) + 1; emits code_embeds.

    logits[b, s, v] = -(v - (mean(inputs_embeds) + 1))^2
        => argmax_v == round(mean(inputs_embeds) + 1).
    code_embeds[b, s, :] = (mean(inputs_embeds) + 1) broadcast across HIDDEN, so
    threading it back as the next step's inputs_embeds advances the residual by
    one each inner step. The KV outputs are inert (Whisper-style contract).
    """
    inputs_embeds = tensor_value(
        "inputs_embeds", ir.DataType.FLOAT, ["batch", "sequence_len", HIDDEN]
    )
    past_key = tensor_value(
        "past_key_values.0.key",
        ir.DataType.FLOAT,
        [1, 1, "past_sequence_len", HEAD_DIM],
    )
    past_value = tensor_value(
        "past_key_values.0.value",
        ir.DataType.FLOAT,
        [1, 1, "past_sequence_len", HEAD_DIM],
    )

    # code_val[b, s, 1] = mean_over_hidden(inputs_embeds) + 1
    mean = node(
        "ReduceMean",
        [inputs_embeds],
        "embeds_mean",
        attributes=[ir.AttrInt64s("axes", [2]), ir.AttrInt64("keepdims", 1)],
    )
    one_f = initializer("one_f", np.array(1.0, dtype=np.float32))
    code_val = node("Add", [mean.outputs[0], one_f], "code_val")  # [b, s, 1]

    # logits[b, s, v] = -(v - code_val)^2  =>  argmax == round(code_val)
    vocab_index = initializer(
        "vocab_index",
        np.arange(VOCAB, dtype=np.float32).reshape(1, 1, VOCAB),
    )
    diff = node("Sub", [vocab_index, code_val.outputs[0]], "diff")
    diff_sq = node("Mul", [diff.outputs[0], diff.outputs[0]], "diff_sq")
    logits = node("Neg", [diff_sq.outputs[0]], "logits")
    logits.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    logits.outputs[0].shape = ir.Shape(["batch", "sequence_len", VOCAB])

    # code_embeds[b, s, :] = code_val broadcast across HIDDEN (the emitted code's
    # embedding), threaded back into inputs_embeds on the next inner step.
    emb_shape = node("Shape", [inputs_embeds], "emb_shape")
    batch_index = constant("batch_index", np.array(0, dtype=np.int64))
    sequence_index = constant("sequence_index", np.array(1, dtype=np.int64))
    axes_zero = constant("axes_zero", np.array([0], dtype=np.int64))
    batch = node(
        "Gather",
        [emb_shape.outputs[0], batch_index.outputs[0]],
        "batch",
        attributes=[ir.AttrInt64("axis", 0)],
    )
    sequence = node(
        "Gather",
        [emb_shape.outputs[0], sequence_index.outputs[0]],
        "sequence",
        attributes=[ir.AttrInt64("axis", 0)],
    )
    batch_vec = node("Unsqueeze", [batch.outputs[0], axes_zero.outputs[0]], "batch_vec")
    sequence_vec = node(
        "Unsqueeze", [sequence.outputs[0], axes_zero.outputs[0]], "sequence_vec"
    )
    one = constant("one", np.array([1], dtype=np.int64))
    head = constant("head", np.array([HEAD_DIM], dtype=np.int64))
    hidden = constant("hidden", np.array([HIDDEN], dtype=np.int64))
    hidden_shape = node(
        "Concat",
        [batch_vec.outputs[0], sequence_vec.outputs[0], hidden.outputs[0]],
        "hidden_shape",
        attributes=[ir.AttrInt64("axis", 0)],
    )
    code_embeds = node(
        "Expand", [code_val.outputs[0], hidden_shape.outputs[0]], "code_embeds"
    )
    code_embeds.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    code_embeds.outputs[0].shape = ir.Shape(["batch", "sequence_len", HIDDEN])

    # Inert KV cache (Whisper-style contract) derived from code_val.
    cache_shape = node(
        "Concat",
        [batch_vec.outputs[0], one.outputs[0], sequence_vec.outputs[0], head.outputs[0]],
        "cache_shape",
        attributes=[ir.AttrInt64("axis", 0)],
    )
    cache_axis = constant("cache_axis", np.array([1], dtype=np.int64))
    code_val_cache = node(
        "Unsqueeze", [code_val.outputs[0], cache_axis.outputs[0]], "code_val_cache"
    )  # [b, 1, s, 1]
    current_key = node(
        "Expand", [code_val_cache.outputs[0], cache_shape.outputs[0]], "current_key"
    )
    value_offset = initializer("value_offset", np.array(0.5, dtype=np.float32))
    current_value = node("Add", [current_key.outputs[0], value_offset], "current_value")
    present_key = node(
        "Concat",
        [past_key, current_key.outputs[0]],
        "present.0.key",
        attributes=[ir.AttrInt64("axis", 2)],
    )
    present_value = node(
        "Concat",
        [past_value, current_value.outputs[0]],
        "present.0.value",
        attributes=[ir.AttrInt64("axis", 2)],
    )
    for output in (present_key.outputs[0], present_value.outputs[0]):
        output.type = ir.TensorType(ir.DataType.FLOAT)
        output.shape = ir.Shape([1, 1, "total_sequence_len", HEAD_DIM])

    nodes = [
        mean,
        code_val,
        diff,
        diff_sq,
        logits,
        emb_shape,
        batch_index,
        sequence_index,
        axes_zero,
        batch,
        sequence,
        batch_vec,
        sequence_vec,
        one,
        head,
        hidden,
        hidden_shape,
        code_embeds,
        cache_shape,
        cache_axis,
        code_val_cache,
        current_key,
        current_value,
        present_key,
        present_value,
    ]
    graph = ir.Graph(
        [inputs_embeds, past_key, past_value],
        [
            logits.outputs[0],
            code_embeds.outputs[0],
            present_key.outputs[0],
            present_value.outputs[0],
        ],
        nodes=nodes,
        initializers=[one_f, vocab_index, value_offset],
        opset_imports={"": 13},
        name="tiny_tts_nested_code_predictor",
    )
    save_model(
        ir.Model(graph, ir_version=8, producer_name="onnx-genai tiny-tts-nested fixture"),
        path,
    )


def build_vocoder(path: Path) -> None:
    """codes[1, F, G] (int64) -> audio[1, F*G] with audio == 2 * flatten(codes)."""
    codes = tensor_value("codes", ir.DataType.INT64, [1, "frames", "groups"])
    codes_float = node(
        "Cast",
        [codes],
        "codes_float",
        attributes=[ir.AttrInt64("to", int(ir.DataType.FLOAT))],
    )
    gain = initializer("gain", np.array(2.0, dtype=np.float32))
    scaled = node("Mul", [codes_float.outputs[0], gain], "scaled")
    flat_shape = constant("flat_shape", np.array([1, -1], dtype=np.int64))
    audio = node("Reshape", [scaled.outputs[0], flat_shape.outputs[0]], "audio")
    audio.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    audio.outputs[0].shape = ir.Shape([1, "num_samples"])

    graph = ir.Graph(
        [codes],
        [audio.outputs[0]],
        nodes=[codes_float, scaled, flat_shape, audio],
        initializers=[gain],
        opset_imports={"": 13},
        name="tiny_tts_nested_vocoder",
    )
    save_model(
        ir.Model(graph, ir_version=8, producer_name="onnx-genai tiny-tts-nested fixture"),
        path,
    )


def write_tokenizer(path: Path) -> None:
    vocab = {"[UNK]": 0, "[EOS]": 1}
    for i in range(2, VOCAB):
        vocab[f"c{i}"] = i
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": 0,
                "content": "[UNK]",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
            {
                "id": 1,
                "content": "[EOS]",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"},
    }
    path.write_text(json.dumps(tokenizer, indent=2) + "\n")


def write_metadata(path: Path) -> None:
    path.write_text(
        f"""# Tiny nested-autoregressive (multi-decoder TTS) pipeline fixture.
# Built by scripts/build_tiny_tts_nested.py; exercises the nested_autoregressive
# contract (DESIGN.md §20.3): an OUTER talker AR loop where each frame drives an
# INNER code_predictor AR loop of num_code_groups steps, seeded by the talker's
# last_hidden_state. Assembled codes are published as `talker.output_codes`
# [1, frames, num_code_groups] and vocoded into a waveform.
pipeline:
  models:
    talker:
      filename: talker.onnx
      type: decoder
      tokenizer: tokenizer.json
    code_predictor:
      filename: code_predictor.onnx
      type: decoder
    vocoder:
      filename: vocoder.onnx
      type: vocoder
  dataflow:
    # Per-frame hidden binding: inner step 0 seed.
    - from: talker.last_hidden_state
      to: code_predictor.inputs_embeds
      dtype: fp32
      device_transfer: false
    # Assembled per-frame codes -> vocoder.
    - from: talker.output_codes
      to: vocoder.codes
      dtype: int64
      device_transfer: false
  strategy:
    kind: composite
    stages:
      - name: generate_codes
        strategy:
          kind: nested_autoregressive
          outer: talker
          inner: code_predictor
          num_code_groups: {NUM_CODE_GROUPS}
          max_tokens: {MAX_FRAMES}
        run_on: every_step
      - name: vocode
        strategy:
          kind: single_pass
          model: vocoder
        run_on: final_only
  phases:
    talker:
      run_on: every_step
    code_predictor:
      run_on: every_step
    vocoder:
      run_on: final_only
"""
    )


def _expected_codes() -> np.ndarray:
    codes = np.zeros((MAX_FRAMES, NUM_CODE_GROUPS), dtype=np.int64)
    for f in range(MAX_FRAMES):
        for g in range(NUM_CODE_GROUPS):
            codes[f, g] = f + g + 1
    return codes


def validate_with_ort(output_dir: Path) -> None:
    import onnxruntime as ort

    talker = ort.InferenceSession(
        str(output_dir / "talker.onnx"), providers=["CPUExecutionProvider"]
    )
    code_predictor = ort.InferenceSession(
        str(output_dir / "code_predictor.onnx"), providers=["CPUExecutionProvider"]
    )
    vocoder = ort.InferenceSession(
        str(output_dir / "vocoder.onnx"), providers=["CPUExecutionProvider"]
    )

    talker_past_key = np.zeros((1, 1, 0, HEAD_DIM), dtype=np.float32)
    talker_past_value = np.zeros((1, 1, 0, HEAD_DIM), dtype=np.float32)
    input_ids = np.array([[8]], dtype=np.int64)  # prompt token (id irrelevant)
    codes = np.zeros((MAX_FRAMES, NUM_CODE_GROUPS), dtype=np.int64)

    for f in range(MAX_FRAMES):
        position_ids = np.array([[f]], dtype=np.int64)
        logits, last_hidden_state, talker_past_key, talker_past_value = talker.run(
            None,
            {
                "decoder_input_ids": input_ids,
                "position_ids": position_ids,
                "past_key_values.0.key": talker_past_key,
                "past_key_values.0.value": talker_past_value,
            },
        )
        outer_token = int(logits[0, -1].argmax())
        input_ids = np.array([[outer_token]], dtype=np.int64)

        # Inner loop seeded by the talker's last hidden position.
        inner_embeds = last_hidden_state[:, -1:, :].astype(np.float32)  # [1, 1, HIDDEN]
        inner_past_key = np.zeros((1, 1, 0, HEAD_DIM), dtype=np.float32)
        inner_past_value = np.zeros((1, 1, 0, HEAD_DIM), dtype=np.float32)
        for g in range(NUM_CODE_GROUPS):
            (
                inner_logits,
                code_embeds,
                inner_past_key,
                inner_past_value,
            ) = code_predictor.run(
                None,
                {
                    "inputs_embeds": inner_embeds,
                    "past_key_values.0.key": inner_past_key,
                    "past_key_values.0.value": inner_past_value,
                },
            )
            codes[f, g] = int(inner_logits[0, -1].argmax())
            inner_embeds = code_embeds[:, -1:, :].astype(np.float32)

    expected_codes = _expected_codes()
    assert np.array_equal(codes, expected_codes), (codes, expected_codes)

    audio = vocoder.run(None, {"codes": codes.reshape(1, MAX_FRAMES, NUM_CODE_GROUPS)})[0]
    expected_audio = (2 * expected_codes.reshape(1, -1)).astype(np.float32)
    assert audio.shape == expected_audio.shape, (audio.shape, expected_audio.shape)
    assert np.allclose(audio, expected_audio), (audio, expected_audio)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/tiny-tts-nested"),
    )
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    build_talker(args.output / "talker.onnx")
    build_code_predictor(args.output / "code_predictor.onnx")
    build_vocoder(args.output / "vocoder.onnx")
    write_tokenizer(args.output / "tokenizer.json")
    write_metadata(args.output / "inference_metadata.yaml")
    if not args.no_validate:
        validate_with_ort(args.output)
    total_size = sum(path.stat().st_size for path in args.output.iterdir())
    print(f"Wrote {args.output} ({total_size} bytes)")


if __name__ == "__main__":
    main()
