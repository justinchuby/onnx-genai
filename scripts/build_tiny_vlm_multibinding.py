#!/usr/bin/env python3
"""Build the tiny deterministic VLM **multi-binding** pipeline fixture.

This fixture stresses the generic ``every_step`` component executor (VLM WP3)
beyond the single-output ``inputs_embeds`` fusion the ``tiny-gemma4-vlm`` fixture
covers. It proves two properties a one-output special case cannot express:

  1. **Every declared output is refreshed each step.** The ``embedding``
     component emits TWO sequence-dependent tensors — ``inputs_embeds`` AND a
     second tensor ``aux`` — and BOTH are routed into the decoder. A correct
     generic executor re-runs the whole component on every step (over the full
     prompt at prefill, over the single running token at decode), so both
     tensors track the running token. If the second tensor were left stale
     (frozen at its prompt value, as the old special case would), the generated
     token ids differ — see ``stale_aux`` below.

  2. **No token / embeds exclusivity.** The ``decoder`` consumes the RAW
     ``input_ids`` token stream *and* the routed ``inputs_embeds`` (+ ``aux``)
     in the same forward pass. The pipeline executor routes the per-step
     embeddings while the decode step feeds the running token to ``input_ids``;
     nothing forbids a decoder from taking both. If ``input_ids`` were ignored
     the ids differ — see ``ids_ignored`` below.

Everything is architecture-neutral DATA: the ``embedding``'s running-token port
is declared via ``io.token_input``; the two outputs route to the decoder purely
through ``dataflow`` edges. The engine never inspects tensor names.

Closed form (H = 8 hidden, V = 8 vocab, E = identity):

  embedding: input_ids[1,s] -> inputs_embeds[1,s,8] = E[input_ids]
                            -> aux[1,s,8]           = A[input_ids]

  decoder:   combined = inputs_embeds + aux + G[input_ids]
             logits[1,s,8] = combined @ W + tie_bias
             W[:, v] = E[(v - 1) mod 8]   (embedding-similarity "next token" head)

  With E = identity, A[t] = 0.6*e_{(t+3)%8} and G[t] = 0.6*e_{(t+3)%8} share the
  slot (t+3), so combined has 1.2 there vs 1.0 at the inputs_embeds self-slot t;
  the next token is (t+4) mod 8. Freezing aux OR dropping the raw-token gate drops
  that slot to 0.6 < 1.0, so the argmax falls back to (t+1) mod 8 — a different id
  stream. Both refreshed tensors are therefore jointly necessary.

The decoder additionally emits a contract-only growing KV cache built from
``combined`` (mirrors the tiny-gemma4 decoder), so the autoregressive loop
exercises real prefill + KV-append decode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnxscript import ir

HIDDEN = 8
VOCAB = 8

# Fixed token embedding table E[token, hidden] (V x H) = identity, so
# inputs_embeds[t] is the one-hot e_t and dot products against the head are the
# clean per-slot component values (no cross-term muddying).
EMBEDDING_TABLE = np.eye(VOCAB, HIDDEN, dtype=np.float32)

# Second sequence-dependent table A[token, hidden]: 0.6 * e_{(t+3) mod 8}. It
# shares its "winning" slot (t+3) with the raw-token gate below, so the two are
# JOINTLY decisive: 0.6 + 0.6 = 1.2 beats the inputs_embeds self-slot (1.0),
# steering the next token to t+4, while either one alone (0.6) loses to 1.0.
AUX_TABLE = (
    0.6 * np.stack([EMBEDDING_TABLE[(t + 3) % VOCAB] for t in range(VOCAB)])
).astype(np.float32)

# Raw-token contribution table G[token, hidden] = 0.6 * e_{(t+3) mod 8}, consumed
# by the decoder from the RAW input_ids input. It lands in the same slot as aux,
# so dropping it (ids ignored) collapses that slot to 0.6 < 1.0 and changes the
# generated ids — proving the decoder truly reads input_ids alongside embeds.
GATE_TABLE = (
    0.6 * np.stack([EMBEDDING_TABLE[(t + 3) % VOCAB] for t in range(VOCAB)])
).astype(np.float32)

# lm_head W[hidden, vocab] with W[:, v] = E[(v - 1) mod 8]. With E = identity this
# is a permutation, so logits[v] = combined[(v - 1) mod 8] and the argmax token is
# (argmax_slot + 1) mod 8.
LM_HEAD = np.stack([EMBEDDING_TABLE[(v - 1) % VOCAB] for v in range(VOCAB)], axis=1).astype(
    np.float32
)

# Deterministic tie-breaker (identical in ORT and the Rust engine).
TIE_BIAS = (np.arange(VOCAB, dtype=np.float32) * 1e-3).reshape(1, 1, VOCAB)


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
    return ir.Node("", op_type, inputs, attributes, outputs=[ir.Value(name=output)])


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


def build_embedding(path: Path) -> None:
    """input_ids[1,s] -> inputs_embeds[1,s,H]=E[ids], aux[1,s,H]=A[ids]."""
    input_ids = tensor_value("input_ids", ir.DataType.INT64, ["batch", "sequence"])

    e_table = initializer("embedding_table", EMBEDDING_TABLE)
    a_table = initializer("aux_table", AUX_TABLE)

    inputs_embeds = node(
        "Gather", [e_table, input_ids], "inputs_embeds", attributes=[ir.AttrInt64("axis", 0)]
    )
    inputs_embeds.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    inputs_embeds.outputs[0].shape = ir.Shape([1, "sequence", HIDDEN])

    aux = node("Gather", [a_table, input_ids], "aux", attributes=[ir.AttrInt64("axis", 0)])
    aux.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    aux.outputs[0].shape = ir.Shape([1, "sequence", HIDDEN])

    graph = ir.Graph(
        [input_ids],
        [inputs_embeds.outputs[0], aux.outputs[0]],
        nodes=[inputs_embeds, aux],
        initializers=[e_table, a_table],
        opset_imports={"": 13},
        name="tiny_vlm_multibinding_embedding",
    )
    save_model(
        ir.Model(graph, ir_version=8, producer_name="onnx-genai tiny-vlm-multibinding fixture"),
        path,
    )


def build_decoder(path: Path) -> None:
    """input_ids + inputs_embeds + aux (+ KV) -> logits + present KV.

    combined = inputs_embeds + aux + G[input_ids]; logits = combined @ W + bias.
    """
    input_ids = tensor_value("input_ids", ir.DataType.INT64, ["batch", "sequence"])
    inputs_embeds = tensor_value("inputs_embeds", ir.DataType.FLOAT, [1, "sequence", HIDDEN])
    aux = tensor_value("aux", ir.DataType.FLOAT, [1, "sequence", HIDDEN])
    past_key = tensor_value(
        "past_key_values.0.key", ir.DataType.FLOAT, [1, 1, "past_sequence", HIDDEN]
    )
    past_value = tensor_value(
        "past_key_values.0.value", ir.DataType.FLOAT, [1, 1, "past_sequence", HIDDEN]
    )

    g_table = initializer("gate_table", GATE_TABLE)
    gate = node("Gather", [g_table, input_ids], "gate_embed", attributes=[ir.AttrInt64("axis", 0)])
    sum1 = node("Add", [inputs_embeds, aux], "embeds_plus_aux")
    combined = node("Add", [sum1.outputs[0], gate.outputs[0]], "combined")

    lm_head = initializer("lm_head", LM_HEAD)
    matmul = node("MatMul", [combined.outputs[0], lm_head], "logits_base")
    tie_bias = initializer("tie_bias", TIE_BIAS)
    logits = node("Add", [matmul.outputs[0], tie_bias], "logits")
    logits.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    logits.outputs[0].shape = ir.Shape([1, "sequence", VOCAB])

    # KV contract: current key/value [1,1,s,4] appended to the past cache.
    kv_axis = constant("kv_axis", np.array([1], dtype=np.int64))
    current_key = node("Unsqueeze", [combined.outputs[0], kv_axis.outputs[0]], "current_key")
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
    for out in (present_key.outputs[0], present_value.outputs[0]):
        out.type = ir.TensorType(ir.DataType.FLOAT)
        out.shape = ir.Shape([1, 1, "total_sequence", HIDDEN])

    graph = ir.Graph(
        [input_ids, inputs_embeds, aux, past_key, past_value],
        [logits.outputs[0], present_key.outputs[0], present_value.outputs[0]],
        nodes=[
            gate,
            sum1,
            combined,
            matmul,
            logits,
            kv_axis,
            current_key,
            current_value,
            present_key,
            present_value,
        ],
        initializers=[g_table, lm_head, tie_bias, value_offset],
        opset_imports={"": 13},
        name="tiny_vlm_multibinding_decoder",
    )
    save_model(
        ir.Model(graph, ir_version=8, producer_name="onnx-genai tiny-vlm-multibinding fixture"),
        path,
    )


def write_tokenizer(path: Path) -> None:
    vocab = {f"tok{i}": i for i in range(VOCAB)}
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": 0,
                "content": "tok0",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "tok0"},
    }
    path.write_text(json.dumps(tokenizer, indent=2) + "\n")


METADATA = """\
# Tiny VLM multi-binding pipeline fixture (VLM WP3).
# Built by scripts/build_tiny_vlm_multibinding.py. The every_step `embedding`
# component emits TWO sequence-dependent tensors (inputs_embeds AND aux), both
# routed into a decoder that ALSO consumes the raw input_ids token stream. The
# generic executor refreshes every declared output each step; the running-token
# port is declared explicitly via io.token_input (no tensor-name heuristics).
pipeline:
  models:
    embedding:
      filename: embedding.onnx.textproto
      type: encoder
      io:
        token_input: input_ids
    decoder:
      filename: decoder.onnx.textproto
      type: decoder
      tokenizer: tokenizer.json
  dataflow:
    - from: embedding.inputs_embeds
      to: decoder.inputs_embeds
      dtype: fp32
      device_transfer: false
    - from: embedding.aux
      to: decoder.aux
      dtype: fp32
      device_transfer: false
  strategy:
    kind: autoregressive
    decoder: decoder
    max_tokens: 4
  phases:
    embedding:
      run_on: every_step
    decoder:
      run_on: every_step
"""


def _embed(ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    return EMBEDDING_TABLE[ids], AUX_TABLE[ids]


def _logits(embeds: np.ndarray, aux: np.ndarray, ids: list[int]) -> np.ndarray:
    combined = embeds + aux + GATE_TABLE[ids]
    return combined.reshape(1, len(ids), HIDDEN) @ LM_HEAD + TIE_BIAS


def fresh_both(prompt: list[int], max_new_tokens: int) -> list[int]:
    """Reference: BOTH tensors refreshed each step (the correct generic path)."""
    embeds, aux = _embed(prompt)
    logits = _logits(embeds, aux, prompt)
    generated = [int(logits[0, -1].argmax())]
    for _ in range(1, max_new_tokens):
        cur = [generated[-1]]
        e, a = _embed(cur)
        generated.append(int(_logits(e, a, cur)[0, -1].argmax()))
    return generated


def stale_aux(prompt: list[int], max_new_tokens: int) -> list[int]:
    """Counter-reference: aux FROZEN at the prompt's last position (as a stale
    one-output special case would leave it), inputs_embeds still refreshed."""
    embeds, aux = _embed(prompt)
    frozen_aux = AUX_TABLE[prompt[-1]].reshape(1, HIDDEN)
    logits = _logits(embeds, aux, prompt)
    generated = [int(logits[0, -1].argmax())]
    for _ in range(1, max_new_tokens):
        cur = [generated[-1]]
        e, _a = _embed(cur)
        generated.append(int(_logits(e, frozen_aux, cur)[0, -1].argmax()))
    return generated


def ids_ignored(prompt: list[int], max_new_tokens: int) -> list[int]:
    """Counter-reference: the decoder's raw input_ids contribution (G) dropped,
    proving the decoder truly consumes input_ids alongside inputs_embeds."""

    def logits_no_ids(embeds, aux, ids):
        combined = embeds + aux
        return combined.reshape(1, len(ids), HIDDEN) @ LM_HEAD + TIE_BIAS

    embeds, aux = _embed(prompt)
    logits = logits_no_ids(embeds, aux, prompt)
    generated = [int(logits[0, -1].argmax())]
    for _ in range(1, max_new_tokens):
        cur = [generated[-1]]
        e, a = _embed(cur)
        generated.append(int(logits_no_ids(e, a, cur)[0, -1].argmax()))
    return generated


def _textproto_bytes(path) -> bytes:
    return ir.to_proto(ir.load(str(path), format="textproto")).SerializeToString()


def validate_with_ort(output_dir: Path, prompt: list[int], max_new_tokens: int) -> list[int]:
    import onnxruntime as ort

    embedding = ort.InferenceSession(
        _textproto_bytes(output_dir / "embedding.onnx.textproto"),
        providers=["CPUExecutionProvider"],
    )
    decoder = ort.InferenceSession(
        _textproto_bytes(output_dir / "decoder.onnx.textproto"),
        providers=["CPUExecutionProvider"],
    )

    def embed(ids: list[int]):
        return embedding.run(None, {"input_ids": np.array([ids], dtype=np.int64)})

    def decode(ids, embeds, aux, past_k, past_v):
        return decoder.run(
            None,
            {
                "input_ids": np.array([ids], dtype=np.int64),
                "inputs_embeds": embeds.astype(np.float32),
                "aux": aux.astype(np.float32),
                "past_key_values.0.key": past_k,
                "past_key_values.0.value": past_v,
            },
        )

    past_k = np.zeros((1, 1, 0, HIDDEN), dtype=np.float32)
    past_v = np.zeros((1, 1, 0, HIDDEN), dtype=np.float32)
    embeds, aux = embed(prompt)
    logits, past_k, past_v = decode(prompt, embeds, aux, past_k, past_v)
    generated = [int(logits[0, -1].argmax())]
    for _ in range(1, max_new_tokens):
        cur = [generated[-1]]
        embeds, aux = embed(cur)
        logits, past_k, past_v = decode(cur, embeds, aux, past_k, past_v)
        generated.append(int(logits[0, -1].argmax()))
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "out_dir",
        nargs="?",
        default=str(
            Path(__file__).resolve().parent.parent / "tests/fixtures/tiny-vlm-multibinding"
        ),
        help="output fixture directory",
    )
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    build_embedding(out_dir / "embedding.onnx.textproto")
    build_decoder(out_dir / "decoder.onnx.textproto")
    write_tokenizer(out_dir / "tokenizer.json")
    (out_dir / "inference_metadata.yaml").write_text(METADATA)

    prompt = [1, 4]
    max_new_tokens = 4
    expected = fresh_both(prompt, max_new_tokens)
    stale = stale_aux(prompt, max_new_tokens)
    no_ids = ids_ignored(prompt, max_new_tokens)

    assert expected != stale, (
        f"fresh_both {expected} must differ from stale_aux {stale}: the second tensor "
        "would not be exercised"
    )
    assert expected != no_ids, (
        f"fresh_both {expected} must differ from ids_ignored {no_ids}: the raw input_ids "
        "would not be exercised"
    )

    if not args.no_validate:
        ort_tokens = validate_with_ort(out_dir, prompt, max_new_tokens)
        assert ort_tokens == expected, f"ORT {ort_tokens} != closed form {expected}"

    (out_dir / "expected.json").write_text(
        json.dumps(
            {
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "fresh_both": expected,
                "stale_aux": stale,
                "ids_ignored": no_ids,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"prompt={prompt}")
    print(f"  fresh_both (engine must match): {expected}")
    print(f"  stale_aux  (counter-ref):       {stale}")
    print(f"  ids_ignored(counter-ref):       {no_ids}")
    total = sum(p.stat().st_size for p in out_dir.iterdir())
    print(f"wrote tiny-vlm-multibinding fixture to {out_dir} ({total} bytes)")


if __name__ == "__main__":
    main()
