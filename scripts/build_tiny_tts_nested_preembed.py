#!/usr/bin/env python3
"""Build the tiny deterministic **pre-embedder-driven nested-autoregressive**
(multi-decoder TTS) pipeline fixture — the real Qwen3-TTS talker shape where the
outer decoder is driven by `inputs_embeds` (materialized from the previous
frame's codes), NOT by `input_ids` (DESIGN.md §20.3).

It is the same *dual, hierarchically-nested* AR loop as `tiny-tts-nested`, but
the outer `talker` consumes `inputs_embeds` and a new `talker_step_embedder`
(the pre-embedder) materializes those embeddings each frame:

  * `talker_step_embedder` (pre-embedder): `frame_codes[1, G] (int64)` +
    `text_embed[1, 1, HIDDEN] (float)` -> `inputs_embeds[1, 1, HIDDEN]`. It sums
    a tiny per-code embedding table (`codec_embed`) over the group axis and adds
    `text_embed`, mirroring the real
    `codec_sum = codec_embed(code_0) + Σ_i cp_codec_weights[i][codes[i+1]]`
    construction. The table is the identity map (row `c` == `c` broadcast across
    HIDDEN), so `inputs_embeds == (Σ_i frame_codes[i]) + text_embed`. The engine
    feeds `text_embed` zeros for now, so `inputs_embeds == Σ_i frame_codes[i]`.
  * `talker` (outer AR decoder): `inputs_embeds`-driven. `logits == -(v - S)^2`
    (argmax first code group `outer_code_0 == round(S)`) and
    `last_hidden_state == S` broadcast across HIDDEN, where
    `S == mean(inputs_embeds) == Σ_i frame_codes[i]`.
  * `code_predictor` (inner AR decoder): `code == mean(inputs_embeds) + 1`,
    seeded at inner step 0 by the talker `last_hidden_state == S`, threading
    `code_embeds` back into `inputs_embeds`, so `code[f][g] == S_f + g + 1`.
  * `vocoder` (`final_only` single_pass): `codes[1, F, G] -> audio[1, F*G]` with
    `audio == 2 * flatten(codes)`.

The engine assembles the next frame's `frame_codes` from the PREVIOUS frame's
code tuple `[outer_code_0, inner_code_1, ..., inner_code_{G-1}]` (group 0 is the
talker's own code; groups 1..G-1 are the code predictor's residuals), matching
the real Qwen3-TTS layout. Frame 0 uses a zero `frame_codes` seed (real
prompt-embeds prefill is a documented follow-up).

With `NUM_CODE_GROUPS = 4` and `MAX_FRAMES = 3` the recurrence is:
  frame 0: frame_codes=[0,0,0,0] -> S=0  -> inner codes [1,2,3,4]
  frame 1: frame_codes=[0,2,3,4] -> S=9  -> inner codes [10,11,12,13]
  frame 2: frame_codes=[9,11,12,13] -> S=45 -> inner codes [46,47,48,49]
so the pool codes `talker.output_codes` are
`[[1,2,3,4],[10,11,12,13],[46,47,48,49]]` — clearly distinct from the
`input_ids`-driven `tiny-tts-nested` fixture (`[[1,2,3,4],[2,3,4,5],[3,4,5,6]]`),
which proves the pre-embedder path is exercised.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnxscript import ir

VOCAB = 64
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
    ir.save(model, path, format="textproto")
    onnx.checker.check_model(ir.to_proto(model))


def build_step_embedder(path: Path) -> None:
    """Pre-embedder: frame_codes[1, G] (+ text_embed[1, 1, HIDDEN]) -> inputs_embeds.

    inputs_embeds[b, 0, :] = Σ_i codec_embed[frame_codes[b, i]] + text_embed[b, 0, :].
    The `codec_embed` table is the identity map (row c == c broadcast across
    HIDDEN), so the codec sum equals `Σ_i frame_codes[b, i]` (a materialized
    stand-in for the real `codec_embed(code_0) + Σ_i cp_codec_weights[i]`).
    """
    frame_codes = tensor_value(
        "frame_codes", ir.DataType.INT64, ["batch", NUM_CODE_GROUPS]
    )
    text_embed = tensor_value(
        "text_embed", ir.DataType.FLOAT, ["batch", 1, HIDDEN]
    )

    # codec_embed[c, :] = c (broadcast across HIDDEN): identity embedding table.
    table = np.repeat(
        np.arange(VOCAB, dtype=np.float32).reshape(VOCAB, 1), HIDDEN, axis=1
    )
    codec_embed = initializer("codec_embed", table)

    # Gather per-code embeddings -> [b, G, HIDDEN]; sum over the group axis.
    gathered = node(
        "Gather",
        [codec_embed, frame_codes],
        "gathered",
        attributes=[ir.AttrInt64("axis", 0)],
    )
    sum_axes = constant("sum_axes", np.array([1], dtype=np.int64))
    codec_sum = node(
        "ReduceSum",
        [gathered.outputs[0], sum_axes.outputs[0]],
        "codec_sum",
        attributes=[ir.AttrInt64("keepdims", 1)],
    )  # [b, 1, HIDDEN]
    inputs_embeds = node("Add", [codec_sum.outputs[0], text_embed], "inputs_embeds")
    inputs_embeds.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    inputs_embeds.outputs[0].shape = ir.Shape(["batch", 1, HIDDEN])

    graph = ir.Graph(
        [frame_codes, text_embed],
        [inputs_embeds.outputs[0]],
        nodes=[gathered, sum_axes, codec_sum, inputs_embeds],
        initializers=[codec_embed],
        opset_imports={"": 13},
        name="tiny_tts_nested_preembed_step_embedder",
    )
    save_model(
        ir.Model(
            graph,
            ir_version=8,
            producer_name="onnx-genai tiny-tts-nested-preembed fixture",
        ),
        path,
    )


def build_talker(path: Path) -> None:
    """Outer AR decoder driven by inputs_embeds (NOT input_ids).

    logits[b, s, v] = -(v - S)^2  =>  argmax_v == round(S), where
    S == mean_over_hidden(inputs_embeds). last_hidden_state[b, s, :] = S broadcast
    across HIDDEN (the inner-loop seed). The KV outputs are inert (Whisper-style).
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

    # S[b, s, 1] = mean_over_hidden(inputs_embeds)
    mean = node(
        "ReduceMean",
        [inputs_embeds],
        "embeds_mean",
        attributes=[ir.AttrInt64s("axes", [2]), ir.AttrInt64("keepdims", 1)],
    )

    # logits[b, s, v] = -(v - S)^2  =>  argmax == round(S)
    vocab_index = initializer(
        "vocab_index",
        np.arange(VOCAB, dtype=np.float32).reshape(1, 1, VOCAB),
    )
    diff = node("Sub", [vocab_index, mean.outputs[0]], "diff")
    diff_sq = node("Mul", [diff.outputs[0], diff.outputs[0]], "diff_sq")
    logits = node("Neg", [diff_sq.outputs[0]], "logits")
    logits.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    logits.outputs[0].shape = ir.Shape(["batch", "sequence_len", VOCAB])

    # last_hidden_state[b, s, :] = S broadcast across HIDDEN (inner-loop seed).
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
    last_hidden_state = node(
        "Expand", [mean.outputs[0], hidden_shape.outputs[0]], "last_hidden_state"
    )
    last_hidden_state.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    last_hidden_state.outputs[0].shape = ir.Shape(["batch", "sequence_len", HIDDEN])

    # Inert KV cache (Whisper-style contract) derived from S.
    cache_shape = node(
        "Concat",
        [batch_vec.outputs[0], one.outputs[0], sequence_vec.outputs[0], head.outputs[0]],
        "cache_shape",
        attributes=[ir.AttrInt64("axis", 0)],
    )
    cache_axis = constant("cache_axis", np.array([1], dtype=np.int64))
    mean_cache = node(
        "Unsqueeze", [mean.outputs[0], cache_axis.outputs[0]], "mean_cache"
    )  # [b, 1, s, 1]
    current_key = node(
        "Expand", [mean_cache.outputs[0], cache_shape.outputs[0]], "current_key"
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
        last_hidden_state,
        cache_shape,
        cache_axis,
        mean_cache,
        current_key,
        current_value,
        present_key,
        present_value,
    ]
    graph = ir.Graph(
        [inputs_embeds, past_key, past_value],
        [
            logits.outputs[0],
            last_hidden_state.outputs[0],
            present_key.outputs[0],
            present_value.outputs[0],
        ],
        nodes=nodes,
        initializers=[vocab_index, value_offset],
        opset_imports={"": 13},
        name="tiny_tts_nested_preembed_talker",
    )
    save_model(
        ir.Model(
            graph,
            ir_version=8,
            producer_name="onnx-genai tiny-tts-nested-preembed fixture",
        ),
        path,
    )


def build_code_predictor(path: Path) -> None:
    """Inner AR decoder: code == mean(inputs_embeds) + 1; emits code_embeds.

    Identical to the `tiny-tts-nested` code predictor: threading code_embeds back
    as the next step's inputs_embeds advances the residual by one each inner step.
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

    mean = node(
        "ReduceMean",
        [inputs_embeds],
        "embeds_mean",
        attributes=[ir.AttrInt64s("axes", [2]), ir.AttrInt64("keepdims", 1)],
    )
    one_f = initializer("one_f", np.array(1.0, dtype=np.float32))
    code_val = node("Add", [mean.outputs[0], one_f], "code_val")  # [b, s, 1]

    vocab_index = initializer(
        "vocab_index",
        np.arange(VOCAB, dtype=np.float32).reshape(1, 1, VOCAB),
    )
    diff = node("Sub", [vocab_index, code_val.outputs[0]], "diff")
    diff_sq = node("Mul", [diff.outputs[0], diff.outputs[0]], "diff_sq")
    logits = node("Neg", [diff_sq.outputs[0]], "logits")
    logits.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    logits.outputs[0].shape = ir.Shape(["batch", "sequence_len", VOCAB])

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
        name="tiny_tts_nested_preembed_code_predictor",
    )
    save_model(
        ir.Model(
            graph,
            ir_version=8,
            producer_name="onnx-genai tiny-tts-nested-preembed fixture",
        ),
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
        name="tiny_tts_nested_preembed_vocoder",
    )
    save_model(
        ir.Model(
            graph,
            ir_version=8,
            producer_name="onnx-genai tiny-tts-nested-preembed fixture",
        ),
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
        f"""# Tiny PRE-EMBEDDER-DRIVEN nested-autoregressive (multi-decoder TTS) fixture.
# Built by scripts/build_tiny_tts_nested_preembed.py; exercises the OPTIONAL
# `pre_embedder` extension of the nested_autoregressive contract (DESIGN.md
# §20.3). The outer `talker` is driven by `inputs_embeds` materialized each frame
# from the PREVIOUS frame's codes by `talker_step_embedder` (a codec-sum
# pre-embedder: frame_codes[+text_embed] -> inputs_embeds), NOT by input_ids.
# The engine feeds text_embed zeros for now (real trailing-text/prefill embeds
# are a documented follow-up). The inner code_predictor loop is unchanged.
pipeline:
  models:
    talker:
      filename: talker.onnx.textproto
      type: decoder
      tokenizer: tokenizer.json
    talker_step_embedder:
      filename: talker_step_embedder.onnx.textproto
      type: embedding
    code_predictor:
      filename: code_predictor.onnx.textproto
      type: decoder
    vocoder:
      filename: vocoder.onnx.textproto
      type: vocoder
  dataflow:
    # Per-step pre-embedder feed: the talker's inputs_embeds each frame.
    - from: talker_step_embedder.inputs_embeds
      to: talker.inputs_embeds
      dtype: fp32
      device_transfer: false
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
          pre_embedder: talker_step_embedder
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
    talker_step_embedder:
      run_on: on_demand
    code_predictor:
      run_on: every_step
    vocoder:
      run_on: final_only
"""
    )


def _expected_pool_codes() -> np.ndarray:
    """Simulate the engine's pre-embedder-driven loop to get the pool codes."""
    codes = np.zeros((MAX_FRAMES, NUM_CODE_GROUPS), dtype=np.int64)
    prev_tuple: list[int] | None = None
    for f in range(MAX_FRAMES):
        frame_codes = prev_tuple if prev_tuple is not None else [0] * NUM_CODE_GROUPS
        s = int(sum(frame_codes))  # inputs_embeds == S; talker mean == S
        outer_code_0 = s  # argmax(-(v - S)^2) == round(S)
        seed = s
        for g in range(NUM_CODE_GROUPS):
            codes[f, g] = seed + g + 1
        prev_tuple = [outer_code_0] + [int(codes[f, g]) for g in range(1, NUM_CODE_GROUPS)]
    return codes


def _textproto_bytes(path) -> bytes:
    """Load a .onnx.textproto fixture and return serialized binary ModelProto bytes."""
    return ir.to_proto(ir.load(str(path), format="textproto")).SerializeToString()


def validate_with_ort(output_dir: Path) -> None:
    import onnxruntime as ort

    step_embedder = ort.InferenceSession(
        _textproto_bytes(output_dir / "talker_step_embedder.onnx.textproto"),
        providers=["CPUExecutionProvider"],
    )
    talker = ort.InferenceSession(
        _textproto_bytes(output_dir / "talker.onnx.textproto"),
        providers=["CPUExecutionProvider"],
    )
    code_predictor = ort.InferenceSession(
        _textproto_bytes(output_dir / "code_predictor.onnx.textproto"),
        providers=["CPUExecutionProvider"],
    )
    vocoder = ort.InferenceSession(
        _textproto_bytes(output_dir / "vocoder.onnx.textproto"),
        providers=["CPUExecutionProvider"],
    )

    talker_past_key = np.zeros((1, 1, 0, HEAD_DIM), dtype=np.float32)
    talker_past_value = np.zeros((1, 1, 0, HEAD_DIM), dtype=np.float32)
    codes = np.zeros((MAX_FRAMES, NUM_CODE_GROUPS), dtype=np.int64)
    prev_tuple: np.ndarray | None = None

    for f in range(MAX_FRAMES):
        frame_codes = (
            prev_tuple
            if prev_tuple is not None
            else np.zeros((1, NUM_CODE_GROUPS), dtype=np.int64)
        )
        text_embed = np.zeros((1, 1, HIDDEN), dtype=np.float32)
        (inputs_embeds,) = step_embedder.run(
            None, {"frame_codes": frame_codes, "text_embed": text_embed}
        )

        logits, last_hidden_state, talker_past_key, talker_past_value = talker.run(
            None,
            {
                "inputs_embeds": inputs_embeds.astype(np.float32),
                "past_key_values.0.key": talker_past_key,
                "past_key_values.0.value": talker_past_value,
            },
        )
        outer_code_0 = int(logits[0, -1].argmax())

        inner_embeds = last_hidden_state[:, -1:, :].astype(np.float32)
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

        prev_tuple = np.array(
            [[outer_code_0, *[int(codes[f, g]) for g in range(1, NUM_CODE_GROUPS)]]],
            dtype=np.int64,
        )

    expected_codes = _expected_pool_codes()
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
        default=Path("tests/fixtures/tiny-tts-nested-preembed"),
    )
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    build_step_embedder(args.output / "talker_step_embedder.onnx.textproto")
    build_talker(args.output / "talker.onnx.textproto")
    build_code_predictor(args.output / "code_predictor.onnx.textproto")
    build_vocoder(args.output / "vocoder.onnx.textproto")
    write_tokenizer(args.output / "tokenizer.json")
    write_metadata(args.output / "inference_metadata.yaml")
    if not args.no_validate:
        validate_with_ort(args.output)
    total_size = sum(path.stat().st_size for path in args.output.iterdir())
    print(f"Wrote {args.output} ({total_size} bytes)")


if __name__ == "__main__":
    main()
