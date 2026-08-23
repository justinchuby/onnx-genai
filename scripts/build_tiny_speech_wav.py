#!/usr/bin/env python3
"""Build the tiny self-contained text-to-speech WAV workflow fixture.

This fixture lets onnx-genai own its *own* raw ``/v1/audio/speech`` and buffered
PCM16 WAV audio conformance without depending on any producer-supplied package,
model-specific ``speech_processor.json``, or Rust copied from another repository.

The package is a minimal canonical ``pipeline.workflow`` document that a runtime
consumes generically through its declared metadata fields:

  * ``request.prompt_tokens`` (role ``prompt_tokens``) is bound from the request
    prompt token rows, exactly like every other workflow package.
  * a single ONNX ``vocoder`` component maps the prompt tokens to a planar
    ``[batch, 2, samples]`` float32 waveform,
  * the ``audio`` output declares a ``pre_adapter`` buffered PCM16 WAV media
    contract (``sample_rate_hz`` 24000, ``source_sample_rate_hz`` 48000, two
    channels), so the API boundary resamples and PCM16-encodes the waveform, and
  * a ``speech_text_assembly`` adapter component (ABI
    ``onnx-genai.text-assembly``) carries a *generic* literal/field program with
    no model-family identifiers, so the server can assemble a speech prompt.

Contract exercised by the runtime tests:

  vocoder: prompt_tokens[batch, seq] (int64)
        -> audio[batch, 2, 64]   (float32, finite, |sample| <= 0.5)

  audio[0, 0, i] = 0.5 * sin(2*pi * (i/64 + 0.001*mean(tokens)))
  audio[0, 1, i] = 0.4 * sin(2*pi * (i/64 + 0.5 + 0.001*mean(tokens)))

The waveform values only have to be deterministic and finite; the fixture proves
the *contract* (prompt tokens in, buffered PCM16 WAV out), not audio quality.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnxscript import ir

SAMPLES = 64
CHANNELS = 2
TARGET_SAMPLE_RATE = 24000
SOURCE_SAMPLE_RATE = 48000
TWO_PI = float(2.0 * np.pi)


def tensor_value(name: str, dtype: ir.DataType, shape: list[int | str]) -> ir.Value:
    return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))


def node(
    op_type: str,
    inputs: list[ir.Value],
    output: str,
    *,
    attributes: tuple[ir.Attr, ...] | list[ir.Attr] = (),
) -> ir.Node:
    return ir.Node("", op_type, inputs, attributes, outputs=[ir.Value(name=output)])


def constant(name: str, array: np.ndarray) -> ir.Node:
    return node(
        "Constant",
        [],
        name,
        attributes=[ir.AttrTensor("value", ir.Tensor(array, name=f"{name}_value"))],
    )


def save_model(model: ir.Model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ir.save(model, path, format="textproto")
    onnx.checker.check_model(ir.to_proto(model))


def build_vocoder(path: Path) -> None:
    """prompt_tokens[batch, seq] (int64) -> audio[batch, 2, SAMPLES] (float32)."""
    _build_vocoder(path, dual_output=False)


def build_dual_output_vocoder(path: Path) -> None:
    """prompt_tokens -> audio[batch, 2, SAMPLES] and waveform[batch, 1, SAMPLES].

    The stereo ``audio`` and the mono ``waveform`` (channel 0 alone) are two
    distinct declared outputs. Fixtures use them to exercise resolution when a
    workflow declares more than one ``role: audio`` output.
    """
    _build_vocoder(path, dual_output=True)


def _build_vocoder(path: Path, *, dual_output: bool) -> None:
    prompt_tokens = tensor_value("prompt_tokens", ir.DataType.INT64, ["batch", "sequence"])

    ramp = constant(
        "ramp",
        (np.arange(SAMPLES, dtype=np.float32) / SAMPLES).reshape(1, 1, SAMPLES),
    )
    mean_scale = constant("mean_scale", np.array(0.001, dtype=np.float32))
    two_pi = constant("two_pi", np.array(TWO_PI, dtype=np.float32))
    channel_shift = constant("channel_shift", np.array(0.5, dtype=np.float32))
    amplitude0 = constant("amplitude0", np.array(0.5, dtype=np.float32))
    amplitude1 = constant("amplitude1", np.array(0.4, dtype=np.float32))
    unsqueeze_axes = constant("unsqueeze_axes", np.array([2], dtype=np.int64))
    # ReduceMean moved `axes` from an attribute to an input in opset 18, so the
    # reduced axis is supplied as an int64 tensor input (opset-24 schema).
    reducemean_axes = constant("reducemean_axes", np.array([1], dtype=np.int64))

    # Reduce the prompt tokens to a per-row scalar so the emitted waveform is a
    # deterministic, bounded function of the request.
    tokens_f = node(
        "Cast",
        [prompt_tokens],
        "tokens_f",
        attributes=[ir.AttrInt64("to", ir.DataType.FLOAT.value)],
    )
    mean = node(
        "ReduceMean",
        [tokens_f.outputs[0], reducemean_axes.outputs[0]],
        "mean",
        attributes=[ir.AttrInt64("keepdims", 1)],
    )
    mean_scaled = node("Mul", [mean.outputs[0], mean_scale.outputs[0]], "mean_scaled")
    # [batch, 1] -> [batch, 1, 1] so it broadcasts across the sample axis.
    mean_col = node(
        "Unsqueeze", [mean_scaled.outputs[0], unsqueeze_axes.outputs[0]], "mean_col"
    )

    # Channel 0: 0.5 * sin(2*pi * (ramp + mean)).
    phase0 = node("Add", [ramp.outputs[0], mean_col.outputs[0]], "phase0")
    angle0 = node("Mul", [phase0.outputs[0], two_pi.outputs[0]], "angle0")
    sin0 = node("Sin", [angle0.outputs[0]], "sin0")
    channel0 = node("Mul", [sin0.outputs[0], amplitude0.outputs[0]], "channel0")

    # Channel 1: 0.4 * sin(2*pi * (ramp + 0.5 + mean)) so the channels differ.
    phase1 = node("Add", [phase0.outputs[0], channel_shift.outputs[0]], "phase1")
    angle1 = node("Mul", [phase1.outputs[0], two_pi.outputs[0]], "angle1")
    sin1 = node("Sin", [angle1.outputs[0]], "sin1")
    channel1 = node("Mul", [sin1.outputs[0], amplitude1.outputs[0]], "channel1")

    audio = node(
        "Concat",
        [channel0.outputs[0], channel1.outputs[0]],
        "audio",
        attributes=[ir.AttrInt64("axis", 1)],
    )
    audio.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    audio.outputs[0].shape = ir.Shape(["batch", CHANNELS, SAMPLES])

    nodes = [
        ramp,
        mean_scale,
        two_pi,
        channel_shift,
        amplitude0,
        amplitude1,
        unsqueeze_axes,
        reducemean_axes,
        tokens_f,
        mean,
        mean_scaled,
        mean_col,
        phase0,
        angle0,
        sin0,
        channel0,
        phase1,
        angle1,
        sin1,
        channel1,
        audio,
    ]
    outputs = [audio.outputs[0]]

    if dual_output:
        # Expose channel 0 alone as a second, mono [batch, 1, SAMPLES] output so
        # a workflow can declare two distinct role: audio outputs.
        waveform = node("Identity", [channel0.outputs[0]], "waveform")
        waveform.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
        waveform.outputs[0].shape = ir.Shape(["batch", 1, SAMPLES])
        nodes.append(waveform)
        outputs.append(waveform.outputs[0])

    graph = ir.Graph(
        [prompt_tokens],
        outputs,
        nodes=nodes,
        opset_imports={"": 24},
        name="tiny_speech_wav_vocoder",
    )
    save_model(
        ir.Model(graph, ir_version=11, producer_name="onnx-genai tiny-speech-wav fixture"),
        path,
    )


# A generic text-assembly program. It contains only literal punctuation and the
# two request fields with model-agnostic text transforms - no model-family
# tokens, tags, or guidance rows. Without `guidance_rows`, `token_rows` returns a
# single row, so the vocoder receives a batch-of-one prompt.
SPEECH_PROCESSOR = {
    "max_input_tokens": 256,
    "max_output_units": 64,
    "state_advance_units": 0,
    "segments": [
        {
            "field": "instructions",
            "transforms": [
                {"kind": "strip_markdown"},
                {"kind": "collapse_newlines"},
            ],
        },
        {"literal": " "},
        {
            "field": "input",
            "transforms": [
                {"kind": "collapse_newlines"},
            ],
        },
    ],
}

# A tiny, generic WordLevel tokenizer. It resolves unknown words to `<unk>` so an
# arbitrary assembled prompt always tokenizes. This is not a model-specific
# vocabulary; it is the shared tiny tokenizer used across onnx-genai fixtures.
TOKENIZER = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [],
    "normalizer": None,
    "pre_tokenizer": {"type": "Whitespace"},
    "post_processor": {
        "type": "TemplateProcessing",
        "single": [
            {"SpecialToken": {"id": "<bos>", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": "<eos>", "type_id": 0}},
        ],
        "pair": [
            {"SpecialToken": {"id": "<bos>", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": "<eos>", "type_id": 0}},
            {"Sequence": {"id": "B", "type_id": 1}},
            {"SpecialToken": {"id": "<eos>", "type_id": 1}},
        ],
        "special_tokens": {
            "<bos>": {"id": "<bos>", "ids": [2], "tokens": ["<bos>"]},
            "<eos>": {"id": "<eos>", "ids": [3], "tokens": ["<eos>"]},
        },
    },
    "decoder": None,
    "model": {
        "type": "WordLevel",
        "vocab": {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3,
            "hello": 4,
            "world": 5,
            "the": 6,
            "quick": 7,
            "brown": 8,
            "fox": 9,
            "jumps": 10,
            "over": 11,
            "lazy": 12,
            "dog": 13,
            ".": 14,
            ",": 15,
        },
        "unk_token": "<unk>",
    },
}

METADATA = f"""\
schema_version: v1
pipeline:
  workflow:
    manifest:
      adapter_abis:
        onnx-genai.text-assembly: '1'
      capabilities:
      - workflow_ssa
      - linear_effects
      - typed_emit
    inputs:
      request.prompt_tokens:
        contract:
          dtype: int64
          rank: 2
          shape:
          - batch
          - sequence_length
          batch_layout:
            kind: request_aligned
            axis: 0
        role:
          kind: runtime
          version: '1.0'
          role: prompt_tokens
        source:
          kind: request
        required: true
    outputs:
      audio:
        contract:
          dtype: float32
          rank: 3
          shape:
          - batch
          - {CHANNELS}
          - samples
          batch_layout:
            kind: request_aligned
            axis: 0
        role: audio
        stage: pre_adapter
        media:
          container: wav
          encoding: pcm_s16_le
          sample_rate_hz: {TARGET_SAMPLE_RATE}
          source_sample_rate_hz: {SOURCE_SAMPLE_RATE}
          channels: {CHANNELS}
          delivery: buffered
    components:
      vocoder:
        implementation:
          kind: onnx
          artifact: vocoder/model.onnx.textproto
      speech_text_assembly:
        implementation:
          kind: adapter
          abi: onnx-genai.text-assembly
          version: '1'
          artifact: speech_processor.json
        contract:
          id: onnx-genai.text-assembly
          version: '1'
    steps:
    - kind: invoke
      component: vocoder
      inputs:
        prompt_tokens: request.prompt_tokens
      outputs:
        audio: speech.audio
    - kind: emit
      value: speech.audio
      output: audio
      mode: replace
package:
  tokenizer:
    algorithm: bpe
    vocab_size: 32000
    byte_level: true
    special_tokens:
      bos:
        id: 2
        content: <bos>
      eos:
        id: 3
        content: <eos>
"""


_MANIFEST_AND_INPUTS = """\
schema_version: v1
pipeline:
  workflow:
    manifest:
      adapter_abis:
        onnx-genai.text-assembly: '1'
      capabilities:
      - workflow_ssa
      - linear_effects
      - typed_emit
    inputs:
      request.prompt_tokens:
        contract:
          dtype: int64
          rank: 2
          shape:
          - batch
          - sequence_length
          batch_layout:
            kind: request_aligned
            axis: 0
        role:
          kind: runtime
          version: '1.0'
          role: prompt_tokens
        source:
          kind: request
        required: true
"""

_PACKAGE_FOOTER = """\
package:
  tokenizer:
    algorithm: bpe
    vocab_size: 32000
    byte_level: true
    special_tokens:
      bos:
        id: 2
        content: <bos>
      eos:
        id: 3
        content: <eos>
"""

# Media contract fragments (indented for the `media:` block of an output).
_COMPAT_STEREO_MEDIA = (
    "          container: wav\n"
    "          encoding: pcm_s16_le\n"
    "          sample_rate_hz: 24000\n"
    "          source_sample_rate_hz: 24000\n"
    "          channels: 2\n"
    "          delivery: buffered\n"
)
_COMPAT_MONO_MEDIA = (
    "          container: wav\n"
    "          encoding: pcm_s16_le\n"
    "          sample_rate_hz: 16000\n"
    "          source_sample_rate_hz: 16000\n"
    "          channels: 1\n"
    "          delivery: buffered\n"
)
# Incompatible: a raw float32 PCM stream, not a buffered PCM16 WAV.
_INCOMPAT_STEREO_MEDIA = (
    "          container: raw\n"
    "          encoding: pcm_f32_le\n"
    "          sample_rate_hz: 24000\n"
    "          channels: 2\n"
    "          delivery: buffered\n"
)

_VOCODER_COMPONENT = (
    "      vocoder:\n"
    "        implementation:\n"
    "          kind: onnx\n"
    "          artifact: vocoder/model.onnx.textproto\n"
)


def _audio_output_block(name: str, channels: int, media: str) -> str:
    return (
        f"      {name}:\n"
        "        contract:\n"
        "          dtype: float32\n"
        "          rank: 3\n"
        "          shape:\n"
        "          - batch\n"
        f"          - {channels}\n"
        "          - samples\n"
        "          batch_layout:\n"
        "            kind: request_aligned\n"
        "            axis: 0\n"
        "        role: audio\n"
        "        stage: pre_adapter\n"
        "        media:\n"
    ) + media


def _adapter_component(name: str) -> str:
    return (
        f"      {name}:\n"
        "        implementation:\n"
        "          kind: adapter\n"
        "          abi: onnx-genai.text-assembly\n"
        "          version: '1'\n"
        "          artifact: speech_processor.json\n"
        "        contract:\n"
        "          id: onnx-genai.text-assembly\n"
        "          version: '1'\n"
    )


def _invoke_step(*, dual: bool) -> str:
    outputs = "        audio: speech.audio\n"
    if dual:
        outputs += "        waveform: speech.waveform\n"
    return (
        "    - kind: invoke\n"
        "      component: vocoder\n"
        "      inputs:\n"
        "        prompt_tokens: request.prompt_tokens\n"
        "      outputs:\n"
    ) + outputs


def _emit_step(value: str, output: str) -> str:
    return f"    - kind: emit\n      value: {value}\n      output: {output}\n      mode: replace\n"


def _document(outputs_block: str, components_block: str, steps_block: str) -> str:
    return (
        _MANIFEST_AND_INPUTS
        + "    outputs:\n"
        + outputs_block
        + "    components:\n"
        + components_block
        + "    steps:\n"
        + steps_block
        + _PACKAGE_FOOTER
    )


# Two role: audio outputs where the map-first output (`audio`) is an incompatible
# raw float stream and only the second (`waveform`) is a compatible buffered
# PCM16 WAV. Resolution must bind and encode exactly `waveform`.
MIXED_METADATA = _document(
    _audio_output_block("audio", CHANNELS, _INCOMPAT_STEREO_MEDIA)
    + _audio_output_block("waveform", 1, _COMPAT_MONO_MEDIA),
    _VOCODER_COMPONENT + _adapter_component("speech_text_assembly"),
    _invoke_step(dual=True)
    + _emit_step("speech.audio", "audio")
    + _emit_step("speech.waveform", "waveform"),
)

# Two compatible role: audio outputs. Resolution is ambiguous and must fail
# closed at load time.
TWO_AUDIO_METADATA = _document(
    _audio_output_block("audio", CHANNELS, _COMPAT_STEREO_MEDIA)
    + _audio_output_block("waveform", 1, _COMPAT_MONO_MEDIA),
    _VOCODER_COMPONENT + _adapter_component("speech_text_assembly"),
    _invoke_step(dual=True)
    + _emit_step("speech.audio", "audio")
    + _emit_step("speech.waveform", "waveform"),
)

# One compatible audio output but two text-assembly components. Resolution of the
# processor is ambiguous and must fail closed at load time.
TWO_ADAPTERS_METADATA = _document(
    _audio_output_block("audio", CHANNELS, _COMPAT_STEREO_MEDIA),
    _VOCODER_COMPONENT
    + _adapter_component("speech_text_assembly")
    + _adapter_component("speech_text_assembly_alt"),
    _invoke_step(dual=False) + _emit_step("speech.audio", "audio"),
)


def _write_package(output: Path, metadata: str, *, dual_output: bool) -> None:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    if dual_output:
        build_dual_output_vocoder(output / "vocoder" / "model.onnx.textproto")
    else:
        build_vocoder(output / "vocoder" / "model.onnx.textproto")
    (output / "inference_metadata.yaml").write_text(metadata)
    (output / "speech_processor.json").write_text(
        json.dumps(SPEECH_PROCESSOR, indent=2) + "\n"
    )
    (output / "tokenizer.json").write_text(json.dumps(TOKENIZER, indent=2) + "\n")
    print(f"wrote fixture to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/onnx_genai_workflows/speech_wav"),
        help="destination package directory for the base fixture",
    )
    args = parser.parse_args()
    base: Path = args.output
    parent = base.parent

    _write_package(base, METADATA, dual_output=False)
    # Multi-candidate variants that exercise fail-closed admission/execution
    # binding. They share the base fixture's generic processor and tokenizer.
    _write_package(parent / "speech_wav_mixed_audio", MIXED_METADATA, dual_output=True)
    _write_package(parent / "speech_wav_two_audio", TWO_AUDIO_METADATA, dual_output=True)
    _write_package(
        parent / "speech_wav_two_adapters", TWO_ADAPTERS_METADATA, dual_output=False
    )


if __name__ == "__main__":
    main()
