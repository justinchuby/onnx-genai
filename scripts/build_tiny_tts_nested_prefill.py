#!/usr/bin/env python3
"""Build the tiny deterministic **prefill + trailing-text pre-embedder-driven
nested-autoregressive** (multi-decoder TTS) pipeline fixture — the real Qwen3-TTS
talker shape where the outer decoder is (1) PREFILLED on frame 0 with a real
multi-position embedding sequence and (2) conditioned on one trailing-text
embedding per subsequent frame (DESIGN.md §20.3).

It extends `tiny-tts-nested-preembed` with a `talker_prefill_embedder` component
that materializes both the frame-0 prefill sequence and the per-frame
trailing-text vectors from the tokenized prompt (`text_ids`):

  * `talker_prefill_embedder` (prefill embedder, `prompt_only`):
    `text_ids[1, L] (int64)` ->
      - `prefill_embeds[1, P, HIDDEN] (float)`: the talker's frame-0
        multi-position PREFILL sequence. Every position holds `TS == Σ text_ids`
        (broadcast across HIDDEN), so the talker's frame-0 mean `S_0 == TS`.
      - `trailing_text_embeds[1, T, HIDDEN] (float)`: one vector per outer frame
        `k >= 1`, fed as the pre-embedder's `text_embed`. Row `k` holds `k + 1`
        (broadcast across HIDDEN), so frame 1 adds `t_0 == 1`.
  * `talker_step_embedder` (pre-embedder, `on_demand`): unchanged from the
    preembed fixture — `frame_codes[1, G] + text_embed[1, 1, HIDDEN] ->
    inputs_embeds[1, 1, HIDDEN] == Σ_i frame_codes[i] + text_embed`.
  * `talker` (outer AR, inputs_embeds-driven): `argmax logits == round(S)` and
    `last_hidden_state == S` broadcast, where `S == mean_over_hidden(inputs_embeds)`.
  * `code_predictor` (inner AR): `code == mean(inputs_embeds) + 1`, seeded by the
    talker hidden state, so `code[f][g] == S_f + g + 1`.
  * `vocoder` (`final_only`): `codes[1, F, G] -> audio == 2 * flatten(codes)`.

The engine feeds `prefill_embeds` DIRECTLY to the talker on frame 0 (advancing
the KV past by `P`, NOT running the pre-embedder), and on frames `k >= 1` runs
the pre-embedder with `text_embed = trailing_text_embeds[:, k-1, :]`.

With `NUM_CODE_GROUPS = 4`, `MAX_FRAMES = 2`, `P = 2`, `T = 2`, prompt `[8]`
(`TS == 8`) the recurrence is:
  frame 0 (PREFILL): S_0 = 8  -> inner codes [9,10,11,12], code_0 = 8
    -> tuple [8,10,11,12] (Σ = 41)
  frame 1: frame_codes=[8,10,11,12], text_embed=t_0=1 -> S_1 = 42
    -> inner codes [43,44,45,46]
so the pool codes `talker.output_codes` are `[[9,10,11,12],[43,44,45,46]]` —
clearly distinct from the ZERO-SEED (no-prefill) path (`[[1,2,3,4],[10,11,12,13]]`),
proving the prefill + trailing-text path is exercised.
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
MAX_FRAMES = 2
PREFILL_LEN = 2
TRAILING_LEN = 2


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


def build_prefill_embedder(path: Path) -> None:
    """Prefill embedder: text_ids[1, L] (int64) -> prefill_embeds + trailing.

    prefill_embeds[b, p, :] = TS  (TS == Σ_l text_ids[b, l]), for all p in [0, P).
    trailing_text_embeds[b, k, :] = k + 1  (constant per row), for all k in [0, T).
    Both are broadcast across HIDDEN so the talker / pre-embedder means are clean.
    """
    text_ids = tensor_value("text_ids", ir.DataType.INT64, ["batch", "text_len"])

    # TS[b, 1] = Σ_l text_ids[b, l]  (as float).
    text_f = node(
        "Cast",
        [text_ids],
        "text_f",
        attributes=[ir.AttrInt64("to", int(ir.DataType.FLOAT))],
    )
    sum_axis = constant("sum_axis", np.array([1], dtype=np.int64))
    ts = node(
        "ReduceSum",
        [text_f.outputs[0], sum_axis.outputs[0]],
        "ts",
        attributes=[ir.AttrInt64("keepdims", 1)],
    )  # [b, 1]

    # batch dim from text_ids shape (keeps the graph batch-generic).
    ids_shape = node("Shape", [text_ids], "ids_shape")
    zero_index = constant("zero_index", np.array([0], dtype=np.int64))
    batch_vec = node(
        "Gather",
        [ids_shape.outputs[0], zero_index.outputs[0]],
        "batch_vec",
        attributes=[ir.AttrInt64("axis", 0)],
    )  # [1] == [batch]

    # prefill_embeds[b, p, :] = TS. Reshape TS -> [b, 1, 1], expand to [b, P, HIDDEN].
    axes_2 = constant("axes_2", np.array([2], dtype=np.int64))
    ts_3d = node("Unsqueeze", [ts.outputs[0], axes_2.outputs[0]], "ts_3d")  # [b,1,1]
    prefill_p = constant("prefill_p", np.array([PREFILL_LEN], dtype=np.int64))
    hidden_dim = constant("hidden_dim", np.array([HIDDEN], dtype=np.int64))
    prefill_shape = node(
        "Concat",
        [batch_vec.outputs[0], prefill_p.outputs[0], hidden_dim.outputs[0]],
        "prefill_shape",
        attributes=[ir.AttrInt64("axis", 0)],
    )
    prefill_embeds = node(
        "Expand", [ts_3d.outputs[0], prefill_shape.outputs[0]], "prefill_embeds"
    )
    prefill_embeds.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    prefill_embeds.outputs[0].shape = ir.Shape(["batch", PREFILL_LEN, HIDDEN])

    # trailing_text_embeds[b, k, :] = k + 1. Constant [1, T, HIDDEN] broadcast to
    # [b, T, HIDDEN].
    trailing_table = np.repeat(
        (np.arange(TRAILING_LEN, dtype=np.float32) + 1.0).reshape(1, TRAILING_LEN, 1),
        HIDDEN,
        axis=2,
    )
    trailing_const = initializer("trailing_const", trailing_table)
    trailing_t = constant("trailing_t", np.array([TRAILING_LEN], dtype=np.int64))
    trailing_shape = node(
        "Concat",
        [batch_vec.outputs[0], trailing_t.outputs[0], hidden_dim.outputs[0]],
        "trailing_shape",
        attributes=[ir.AttrInt64("axis", 0)],
    )
    trailing_text_embeds = node(
        "Expand", [trailing_const, trailing_shape.outputs[0]], "trailing_text_embeds"
    )
    trailing_text_embeds.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    trailing_text_embeds.outputs[0].shape = ir.Shape(["batch", TRAILING_LEN, HIDDEN])

    nodes = [
        text_f,
        sum_axis,
        ts,
        ids_shape,
        zero_index,
        batch_vec,
        axes_2,
        ts_3d,
        prefill_p,
        hidden_dim,
        prefill_shape,
        prefill_embeds,
        trailing_t,
        trailing_shape,
        trailing_text_embeds,
    ]
    graph = ir.Graph(
        [text_ids],
        [prefill_embeds.outputs[0], trailing_text_embeds.outputs[0]],
        nodes=nodes,
        initializers=[trailing_const],
        opset_imports={"": 13},
        name="tiny_tts_nested_prefill_prefill_embedder",
    )
    save_model(
        ir.Model(
            graph,
            ir_version=8,
            producer_name="onnx-genai tiny-tts-nested-prefill fixture",
        ),
        path,
    )


def build_step_embedder(path: Path) -> None:
    """Pre-embedder: frame_codes[1, G] (+ text_embed[1, 1, HIDDEN]) -> inputs_embeds.

    inputs_embeds[b, 0, :] = Σ_i codec_embed[frame_codes[b, i]] + text_embed[b, 0, :].
    The `codec_embed` table is the identity map (row c == c broadcast across
    HIDDEN), so the codec sum equals `Σ_i frame_codes[b, i]`.
    """
    frame_codes = tensor_value(
        "frame_codes", ir.DataType.INT64, ["batch", NUM_CODE_GROUPS]
    )
    text_embed = tensor_value("text_embed", ir.DataType.FLOAT, ["batch", 1, HIDDEN])

    table = np.repeat(
        np.arange(VOCAB, dtype=np.float32).reshape(VOCAB, 1), HIDDEN, axis=1
    )
    codec_embed = initializer("codec_embed", table)

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
        name="tiny_tts_nested_prefill_step_embedder",
    )
    save_model(
        ir.Model(
            graph,
            ir_version=8,
            producer_name="onnx-genai tiny-tts-nested-prefill fixture",
        ),
        path,
    )


def build_talker(path: Path) -> None:
    """Outer AR decoder driven by inputs_embeds (NOT input_ids).

    logits[b, s, v] = -(v - S)^2 => argmax_v == round(S), S == mean_over_hidden.
    last_hidden_state[b, s, :] = S broadcast (inner-loop seed). Inert KV cache.
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

    vocab_index = initializer(
        "vocab_index",
        np.arange(VOCAB, dtype=np.float32).reshape(1, 1, VOCAB),
    )
    diff = node("Sub", [vocab_index, mean.outputs[0]], "diff")
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
    last_hidden_state = node(
        "Expand", [mean.outputs[0], hidden_shape.outputs[0]], "last_hidden_state"
    )
    last_hidden_state.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    last_hidden_state.outputs[0].shape = ir.Shape(["batch", "sequence_len", HIDDEN])

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
        name="tiny_tts_nested_prefill_talker",
    )
    save_model(
        ir.Model(
            graph,
            ir_version=8,
            producer_name="onnx-genai tiny-tts-nested-prefill fixture",
        ),
        path,
    )


def build_code_predictor(path: Path) -> None:
    """Inner AR decoder: code == mean(inputs_embeds) + 1; emits code_embeds."""
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
        name="tiny_tts_nested_prefill_code_predictor",
    )
    save_model(
        ir.Model(
            graph,
            ir_version=8,
            producer_name="onnx-genai tiny-tts-nested-prefill fixture",
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
        name="tiny_tts_nested_prefill_vocoder",
    )
    save_model(
        ir.Model(
            graph,
            ir_version=8,
            producer_name="onnx-genai tiny-tts-nested-prefill fixture",
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
        f"""# Tiny PREFILL + TRAILING-TEXT pre-embedder-driven nested-autoregressive
# (multi-decoder TTS) fixture. Built by scripts/build_tiny_tts_nested_prefill.py;
# exercises the OPTIONAL `prefill_embedder` extension of the nested_autoregressive
# contract (DESIGN.md §20.3). A `talker_prefill_embedder` (prompt_only) maps the
# tokenized prompt `text_ids` -> `prefill_embeds` (the talker's frame-0
# multi-position PREFILL sequence, fed DIRECTLY on frame 0) + `trailing_text_embeds`
# (one vector per outer frame k>=1, fed as the pre-embedder's `text_embed`). The
# per-frame `talker_step_embedder` pre-embedder and the inner code_predictor loop
# are unchanged from tiny-tts-nested-preembed.
pipeline:
  models:
    talker:
      filename: talker.onnx.textproto
      type: decoder
      tokenizer: tokenizer.json
    talker_prefill_embedder:
      filename: talker_prefill_embedder.onnx.textproto
      type: embedding
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
    # Per-step pre-embedder feed: the talker's inputs_embeds each frame (k>=1).
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
          prefill_embedder: talker_prefill_embedder
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
    talker_prefill_embedder:
      run_on: prompt_only
    talker_step_embedder:
      run_on: on_demand
    code_predictor:
      run_on: every_step
    vocoder:
      run_on: final_only
"""
    )


def _expected_pool_codes(prompt_ids: list[int]) -> np.ndarray:
    """Simulate the engine's prefill + trailing-text loop to get the pool codes."""
    ts = int(sum(prompt_ids))
    trailing = [k + 1 for k in range(TRAILING_LEN)]
    codes = np.zeros((MAX_FRAMES, NUM_CODE_GROUPS), dtype=np.int64)
    prev_tuple: list[int] | None = None
    for f in range(MAX_FRAMES):
        if f == 0:
            s = ts  # frame-0 prefill: talker mean == TS
        else:
            frame_codes = prev_tuple  # type: ignore[assignment]
            t = trailing[f - 1] if (f - 1) < TRAILING_LEN else 0
            s = int(sum(frame_codes)) + t
        outer_code_0 = s  # argmax(-(v - S)^2) == round(S)
        for g in range(NUM_CODE_GROUPS):
            codes[f, g] = s + g + 1
        prev_tuple = [outer_code_0] + [int(codes[f, g]) for g in range(1, NUM_CODE_GROUPS)]
    return codes


def _textproto_bytes(path) -> bytes:
    return ir.to_proto(ir.load(str(path), format="textproto")).SerializeToString()


def validate_with_ort(output_dir: Path) -> None:
    import onnxruntime as ort

    prompt_ids = [8]

    prefill_embedder = ort.InferenceSession(
        _textproto_bytes(output_dir / "talker_prefill_embedder.onnx.textproto"),
        providers=["CPUExecutionProvider"],
    )
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

    text_ids = np.array([prompt_ids], dtype=np.int64)
    prefill_embeds, trailing_text_embeds = prefill_embedder.run(None, {"text_ids": text_ids})
    assert prefill_embeds.shape == (1, PREFILL_LEN, HIDDEN), prefill_embeds.shape
    assert trailing_text_embeds.shape == (1, TRAILING_LEN, HIDDEN), trailing_text_embeds.shape

    talker_past_key = np.zeros((1, 1, 0, HEAD_DIM), dtype=np.float32)
    talker_past_value = np.zeros((1, 1, 0, HEAD_DIM), dtype=np.float32)
    codes = np.zeros((MAX_FRAMES, NUM_CODE_GROUPS), dtype=np.int64)
    prev_tuple: np.ndarray | None = None

    for f in range(MAX_FRAMES):
        if f == 0:
            inputs_embeds = prefill_embeds.astype(np.float32)
        else:
            frame_codes = prev_tuple
            idx = f - 1
            text_embed = (
                trailing_text_embeds[:, idx : idx + 1, :].astype(np.float32)
                if idx < TRAILING_LEN
                else np.zeros((1, 1, HIDDEN), dtype=np.float32)
            )
            (inputs_embeds,) = step_embedder.run(
                None, {"frame_codes": frame_codes, "text_embed": text_embed}
            )
            inputs_embeds = inputs_embeds.astype(np.float32)

        logits, last_hidden_state, talker_past_key, talker_past_value = talker.run(
            None,
            {
                "inputs_embeds": inputs_embeds,
                "past_key_values.0.key": talker_past_key,
                "past_key_values.0.value": talker_past_value,
            },
        )
        # Frame 0 advances the KV past by PREFILL_LEN (multi-position prefill).
        if f == 0:
            assert talker_past_key.shape[2] == PREFILL_LEN, talker_past_key.shape
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

    expected_codes = _expected_pool_codes(prompt_ids)
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
        default=Path("tests/fixtures/tiny-tts-nested-prefill"),
    )
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    build_prefill_embedder(args.output / "talker_prefill_embedder.onnx.textproto")
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
