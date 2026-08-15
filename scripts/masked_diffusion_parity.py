#!/usr/bin/env python3
"""Parity: onnx-genai `masked_diffusion` vs the reference LLaDA `generate`.

Language (discrete) diffusion has no external reference library the way image
diffusion has diffusers, so we validate onnx-genai's `masked_diffusion` scheduler
directly against the sampling algorithm from ML-GSAI/LLaDA's ``generate.py``
(single block, ``cfg_scale=0``, ``temperature=0``, ``remasking="low_confidence"``).

Both sides are driven by the *same* deterministic ONNX "language model" whose
logits are a fixed function of the current token sequence (token embedding +
previous-token coupling + position bias), so each denoise step produces
identical logits. Because the confidence-ranked commit order is fully
determined, onnx-genai must reproduce the reference token sequence exactly.

Usage (conda `onnx` env, after `cargo build --release -p onnx-genai --bin run_diffusion`):
    conda run -n onnx python scripts/masked_diffusion_parity.py
"""

from __future__ import annotations

import glob
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "target" / "release" / "run_diffusion"

SEQUENCE_LENGTH = 8
VOCAB = 32
MASK_TOKEN = 0
STEPS = 8
PROMPT = [5, 11]  # fixed (non-mask) prompt tokens
SEED = 1234
NEGATIVE_INFINITY = -1.0e9


def _tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    current_embedding = rng.standard_normal((VOCAB, VOCAB)).astype(np.float32)
    previous_embedding = (rng.standard_normal((VOCAB, VOCAB)) * 0.5).astype(np.float32)
    position_bias = rng.standard_normal((1, SEQUENCE_LENGTH, VOCAB)).astype(np.float32)
    # The mask token is never a valid prediction: force its logit column to -inf
    # (LLaDA's mask id lies outside the emitted vocabulary).
    position_bias[:, :, MASK_TOKEN] = NEGATIVE_INFINITY
    return current_embedding, previous_embedding, position_bias


CURRENT_EMBEDDING, PREVIOUS_EMBEDDING, POSITION_BIAS = _tables()


def model_logits(input_ids: np.ndarray) -> np.ndarray:
    """Deterministic logits [1, S, V] as a fixed function of input_ids [1, S]."""
    ids = input_ids[0]
    previous = np.concatenate([[MASK_TOKEN], ids[:-1]])
    logits = CURRENT_EMBEDDING[ids] + PREVIOUS_EMBEDDING[previous] + POSITION_BIAS[0]
    return logits[None].astype(np.float32)


def build_onnx_model(path: Path) -> None:
    """Build the coupled deterministic LM as an ONNX graph writing `logits`."""

    def const(name: str, array: np.ndarray) -> onnx.NodeProto:
        return helper.make_node(
            "Constant", [], [name], value=numpy_helper.from_array(array, name)
        )

    nodes = [
        const("current_embedding", CURRENT_EMBEDDING),
        const("previous_embedding", PREVIOUS_EMBEDDING),
        const("position_bias", POSITION_BIAS),
        const("mask_front", np.array([[MASK_TOKEN]], dtype=np.int64)),
        const("slice_starts", np.array([0], dtype=np.int64)),
        const("slice_ends", np.array([SEQUENCE_LENGTH - 1], dtype=np.int64)),
        const("slice_axes", np.array([1], dtype=np.int64)),
        # Current-token embedding: [1, S, V].
        helper.make_node("Gather", ["current_embedding", "input_ids"], ["current"], axis=0),
        # Previous-token ids: shift right by one, filling the front with MASK.
        helper.make_node(
            "Slice",
            ["input_ids", "slice_starts", "slice_ends", "slice_axes"],
            ["sliced"],
        ),
        helper.make_node("Concat", ["mask_front", "sliced"], ["previous_ids"], axis=1),
        helper.make_node("Gather", ["previous_embedding", "previous_ids"], ["previous"], axis=0),
        helper.make_node("Add", ["current", "previous"], ["coupled"]),
        helper.make_node("Add", ["coupled", "position_bias"], ["logits"]),
    ]

    graph = helper.make_graph(
        nodes,
        "coupled_masked_diffusion_lm",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, SEQUENCE_LENGTH])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, SEQUENCE_LENGTH, VOCAB])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def get_num_transfer_tokens(mask_count: int, steps: int) -> np.ndarray:
    """LLaDA's even split of the masked count across `steps` steps."""
    base = mask_count // steps
    remainder = mask_count % steps
    counts = np.full(steps, base, dtype=np.int64)
    counts[:remainder] += 1
    return counts


def reference_generate(
    prompt: list[int],
    gen_length: int,
    steps: int,
    block_length: int,
    cfg_scale: float = 0.0,
) -> np.ndarray:
    """Faithful LLaDA `generate` core: temperature=0, low_confidence.

    Supports semi-autoregressive block decoding (`block_length < gen_length`) and
    unsupervised classifier-free guidance (`cfg_scale > 0`: the unconditional pass
    re-masks the prompt, and `logits = un + (cfg_scale + 1) * (cond - un)`).
    """
    prompt_length = len(prompt)
    x = np.full((1, prompt_length + gen_length), MASK_TOKEN, dtype=np.int64)
    x[0, :prompt_length] = prompt
    prompt_index = x != MASK_TOKEN  # marks the (fixed) prompt positions

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    for block in range(num_blocks):
        block_start = prompt_length + block * block_length
        block_end = prompt_length + (block + 1) * block_length
        block_mask = x[0, block_start:block_end] == MASK_TOKEN
        num_transfer_tokens = get_num_transfer_tokens(int(block_mask.sum()), steps_per_block)
        for i in range(steps_per_block):
            mask_index = x == MASK_TOKEN
            if cfg_scale > 0:
                unconditional = x.copy()
                unconditional[prompt_index] = MASK_TOKEN
                conditional_logits = model_logits(x)
                unconditional_logits = model_logits(unconditional)
                logits = unconditional_logits + (cfg_scale + 1) * (
                    conditional_logits - unconditional_logits
                )
            else:
                logits = model_logits(x)  # temperature 0 => argmax over clean logits
            predicted = logits.argmax(axis=-1)  # [1, S]
            shifted = logits - logits.max(axis=-1, keepdims=True)
            probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
            chosen_probability = np.take_along_axis(
                probabilities, predicted[..., None], axis=-1
            )[..., 0]

            # Only positions inside the current block are eligible this step.
            chosen_probability[:, block_end:] = -np.inf
            predicted = np.where(mask_index, predicted, x)
            confidence = np.where(mask_index, chosen_probability, -np.inf)

            k = int(num_transfer_tokens[i])
            if k > 0:
                select = np.argsort(-confidence[0], kind="stable")[:k]
                x[0, select] = predicted[0, select]
    return x[0]


def write_metadata(directory: Path, block_length: int | None, cfg_scale: float = 0.0) -> None:
    lines = [
        "pipeline:",
        "  models:",
        "    denoiser:",
        "      filename: lm.onnx",
        "      type: denoiser",
        "  dataflow:",
        "    - from: denoiser.logits",
        "      to: denoiser.input_ids",
        "  strategy:",
        "    kind: iterative",
        "    denoiser: denoiser",
        f"    num_steps: {STEPS}",
    ]
    if cfg_scale > 0:
        # The runtime combines uncond + scale*(cond - uncond); LLaDA's effective
        # multiplier is cfg_scale + 1, so that is the guidance_scale we declare.
        lines.append(f"    guidance_scale: {cfg_scale + 1}")
    lines += [
        "    scheduler_config:",
        "      kind: masked_diffusion",
        f"      mask_token_id: {MASK_TOKEN}",
    ]
    if block_length is not None:
        lines.append(f"      block_length: {block_length}")
    (directory / "inference_metadata.yaml").write_text("\n".join(lines) + "\n")


def run_onnx_genai(directory: Path, seed_tokens: np.ndarray) -> np.ndarray:
    seed_path = directory / "seed.i64"
    out_path = directory / "out.i64"
    seed_tokens.astype("<i8").tofile(seed_path)

    hits = sorted(glob.glob(str(REPO / "target/*/build/onnx-genai-ort-sys-*/out/ort-prebuilt/lib")))
    if not hits:
        raise SystemExit("could not locate prebuilt ORT lib dir")
    import os

    env = dict(os.environ)
    env["DYLD_LIBRARY_PATH"] = hits[-1] + ":" + env.get("DYLD_LIBRARY_PATH", "")
    subprocess.run(
        [
            str(RUNNER),
            str(directory),
            "denoiser.input_ids",
            str(out_path),
            f"denoiser.input_ids:i64:1,{SEQUENCE_LENGTH}:{seed_path}",
        ],
        env=env,
        check=True,
    )
    return np.fromfile(out_path, dtype="<i8")


def main() -> int:
    if not RUNNER.exists():
        raise SystemExit(
            "run_diffusion not built; run: cargo build --release -p onnx-genai --bin run_diffusion"
        )

    # Sanity: div_ceil(remaining, remaining_steps) reproduces get_num_transfer_tokens.
    for mask_count in range(0, 40):
        for steps in range(1, 12):
            reference_counts = get_num_transfer_tokens(mask_count, steps)
            remaining = mask_count
            greedy = []
            for step in range(steps):
                commit = -(-remaining // (steps - step))  # ceil
                greedy.append(commit)
                remaining -= commit
            assert list(reference_counts) == greedy, (mask_count, steps)
    print("num_transfer schedule: div_ceil == LLaDA get_num_transfer_tokens ✓")

    gen_length = SEQUENCE_LENGTH - len(PROMPT)
    seed = np.full(SEQUENCE_LENGTH, MASK_TOKEN, dtype=np.int64)
    seed[: len(PROMPT)] = PROMPT

    # (label, block_length metadata, effective block length, cfg_scale).
    cases = [
        ("single block", None, gen_length, 0.0),
        ("semi-autoregressive (block_length=3)", 3, 3, 0.0),
        ("classifier-free guidance (cfg_scale=1.5)", None, gen_length, 1.5),
    ]
    for label, block_metadata, block_length, cfg_scale in cases:
        if gen_length % block_length != 0 or STEPS % (gen_length // block_length) != 0:
            raise SystemExit(f"bad test config for {label}")
        expected = reference_generate(PROMPT, gen_length, STEPS, block_length, cfg_scale)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            build_onnx_model(directory / "lm.onnx")
            write_metadata(directory, block_metadata, cfg_scale)
            actual = run_onnx_genai(directory, seed)
        print(f"[{label}]")
        print(f"  reference (LLaDA): {expected.tolist()}")
        print(f"  onnx-genai:        {actual.tolist()}")
        assert np.array_equal(actual, expected), f"masked_diffusion diverged from LLaDA ({label})"
        assert MASK_TOKEN not in actual[len(PROMPT):], f"mask tokens remain ({label})"
    print("PARITY OK: masked_diffusion == LLaDA generate (temp=0; single, semi-AR, CFG)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
