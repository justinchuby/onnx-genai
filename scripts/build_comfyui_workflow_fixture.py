#!/usr/bin/env python3
"""Generate the executable ComfyUI-import conformance package.

    python scripts/build_comfyui_workflow_fixture.py

The package under `tests/fixtures/comfyui_workflows/txt2img_sd15/` is the
end-to-end proof that a ComfyUI workflow, once imported, executes on the generic
workflow runtime with no ComfyUI code in the loop:

    workflow.json  --(onnx-genai-comfyui-config)-->  inference_metadata.yaml
                                                            |
                                            generic workflow engine executes it

`inference_metadata.yaml` is NOT written by this script. It is regenerated from
`workflow.json` by the converter (`cargo run -p onnx-genai-comfyui-config --bin
comfyui_to_metadata -- --textproto --out ... workflow.json`), and a test asserts
the checked-in document still matches. That is what makes the golden file a
regression test on the converter rather than a snapshot of this script.

What this script writes is the *package the emitted metadata references*: the
component ONNX graphs, in the exact ABI the converter emits. Every graph is
tiny, deterministic, and meaningful enough that a mis-wired port changes the
output:

* `latent_noise` consumes `seed`, `offset`, and `row_shape`, so swapping any of
  them changes the drawn latent.
* `denoiser` mixes the sample, the timestep, and the conditioning, so routing
  the unconditional embedding into the conditional pass is observable.
* `guidance_combine` is a genuine `uncond + scale * (cond - uncond)`, so a
  guidance scale that never reaches it changes the image.
* `solver_step` is the k-diffusion Euler update against the emitted schedule,
  so an off-by-one step index changes the trajectory.

Artifacts are written as ONNX protobuf **TextFormat** (`*.onnx.textproto`), not
binary, so the checked-in package is reviewable in a diff. The runtime accepts
either encoding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from google.protobuf import text_format
from onnx import TensorProto, helper, numpy_helper

OPSET = 24
IR_VERSION = 11

# Tiny but structured: 4 latent channels on a 4x4 grid, an 8-wide text encoder,
# a 16-token vocabulary, and a 4-step schedule.
CHANNELS = 4
SIZE = 4
HIDDEN = 8
VOCAB = 16
STEPS = 4
ROW_ELEMENTS = CHANNELS * SIZE * SIZE


def _vi(name: str, elem_type: int, shape: list) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, elem_type, shape)


def _const(name: str, array: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(array, name)


def _model(graph: onnx.GraphProto) -> onnx.ModelProto:
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", OPSET)],
        producer_name="onnx-genai-comfyui-fixture",
    )
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model, full_check=True)
    return model


def _save(model: onnx.ModelProto, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text_format.MessageToString(model))
    print(f"wrote {path} ({path.stat().st_size} bytes)")


def _sigmas() -> np.ndarray:
    """Karras-spaced sigmas, descending to an exact zero at the final step."""
    rho, sigma_min, sigma_max = 7.0, 0.03, 4.0
    ramp = np.linspace(0.0, 1.0, STEPS, dtype=np.float64)
    inv_rho = 1.0 / rho
    sigmas = (
        sigma_max**inv_rho + ramp * (sigma_min**inv_rho - sigma_max**inv_rho)
    ) ** rho
    return np.append(sigmas, 0.0).astype(np.float32)


# ── model components ────────────────────────────────────────────────────────


def build_text_encoder() -> onnx.ModelProto:
    """`input_ids[batch, seq] -> encoder_hidden_states[batch, seq, HIDDEN]`."""
    rng = np.random.default_rng(20260821)
    table = rng.standard_normal((VOCAB, HIDDEN)).astype(np.float32) * 0.25
    graph = helper.make_graph(
        [helper.make_node("Gather", ["embedding", "input_ids"], ["encoder_hidden_states"], axis=0)],
        "text_encoder",
        [_vi("input_ids", TensorProto.INT64, ["batch", "sequence"])],
        [_vi("encoder_hidden_states", TensorProto.FLOAT, ["batch", "sequence", HIDDEN])],
        initializer=[_const("embedding", table)],
    )
    return _model(graph)


def build_denoiser() -> onnx.ModelProto:
    """Mix the sample, the timestep, and the conditioning into a prediction.

    Every input reaches the output, so a mis-routed conditioning branch or a
    dropped timestep is visible in the produced image rather than silently
    ignored.
    """
    nodes = [
        # Conditioning summary: mean over sequence and hidden -> [batch, 1, 1, 1].
        helper.make_node("ReduceMean", ["encoder_hidden_states", "axes_12"], ["cond_mean"], keepdims=0),
        helper.make_node("Reshape", ["cond_mean", "row_shape_4d"], ["cond_row"]),
        # Timestep -> [batch, 1, 1, 1].
        helper.make_node("Reshape", ["timestep", "row_shape_4d"], ["step_row"]),
        helper.make_node("Mul", ["step_row", "step_gain"], ["step_term"]),
        helper.make_node("Mul", ["sample", "sample_gain"], ["sample_term"]),
        helper.make_node("Add", ["sample_term", "cond_row"], ["partial"]),
        helper.make_node("Add", ["partial", "step_term"], ["noise_pred"]),
    ]
    graph = helper.make_graph(
        nodes,
        "denoiser",
        [
            _vi("sample", TensorProto.FLOAT, ["batch", CHANNELS, SIZE, SIZE]),
            _vi("timestep", TensorProto.FLOAT, ["batch"]),
            _vi("encoder_hidden_states", TensorProto.FLOAT, ["batch", "sequence", HIDDEN]),
        ],
        [_vi("noise_pred", TensorProto.FLOAT, ["batch", CHANNELS, SIZE, SIZE])],
        initializer=[
            _const("axes_12", np.array([1, 2], dtype=np.int64)),
            _const("row_shape_4d", np.array([-1, 1, 1, 1], dtype=np.int64)),
            _const("step_gain", np.array([0.05], dtype=np.float32)),
            _const("sample_gain", np.array([0.6], dtype=np.float32)),
        ],
    )
    return _model(graph)


def build_vae_decoder() -> onnx.ModelProto:
    """`latent[batch, 4, H, W] -> image[batch, 3, H, W]` in `[-1, 1]`."""
    nodes = [
        helper.make_node("Slice", ["latent", "start_0", "end_3", "axis_1"], ["rgb"]),
        helper.make_node("Mul", ["rgb", "decode_gain"], ["scaled"]),
        helper.make_node("Tanh", ["scaled"], ["image"]),
    ]
    graph = helper.make_graph(
        nodes,
        "vae_decoder",
        [_vi("latent", TensorProto.FLOAT, ["batch", CHANNELS, SIZE, SIZE])],
        [_vi("image", TensorProto.FLOAT, ["batch", 3, SIZE, SIZE])],
        initializer=[
            _const("start_0", np.array([0], dtype=np.int64)),
            _const("end_3", np.array([3], dtype=np.int64)),
            _const("axis_1", np.array([1], dtype=np.int64)),
            _const("decode_gain", np.array([0.4], dtype=np.float32)),
        ],
    )
    return _model(graph)


# ── policy components ───────────────────────────────────────────────────────


def build_diffusion_schedule() -> onnx.ModelProto:
    """The sigma schedule the solver steps along: `STEPS + 1` entries."""
    graph = helper.make_graph(
        [helper.make_node("Identity", ["sigmas"], ["schedule"])],
        "diffusion_schedule",
        [],
        [_vi("schedule", TensorProto.FLOAT, [STEPS + 1])],
        initializer=[_const("sigmas", _sigmas())],
    )
    return _model(graph)


def build_diffusion_timesteps() -> onnx.ModelProto:
    """The per-step timestep the denoiser is conditioned on.

    k-diffusion conditions an epsilon model on the sigma itself, so the fixture
    does the same rather than inventing a second scale that would only make the
    tiny trajectory saturate.
    """
    timesteps = _sigmas()[:STEPS].astype(np.float32)
    graph = helper.make_graph(
        [helper.make_node("Identity", ["timesteps"], ["schedule"])],
        "diffusion_timesteps",
        [],
        [_vi("schedule", TensorProto.FLOAT, [STEPS])],
        initializer=[_const("timesteps", timesteps)],
    )
    return _model(graph)


def build_schedule_lookup() -> onnx.ModelProto:
    """`schedule[L], step[batch] -> timestep[batch]`."""
    graph = helper.make_graph(
        [helper.make_node("Gather", ["schedule", "step"], ["timestep"], axis=0)],
        "schedule_lookup",
        [
            _vi("schedule", TensorProto.FLOAT, ["schedule_length"]),
            _vi("step", TensorProto.INT64, ["batch"]),
        ],
        [_vi("timestep", TensorProto.FLOAT, ["batch"])],
    )
    return _model(graph)


def build_model_input() -> onnx.ModelProto:
    """k-diffusion `scale_model_input`: `sample / sqrt(sigma^2 + 1)`."""
    nodes = [
        helper.make_node("Gather", ["schedule", "step"], ["sigma"], axis=0),
        helper.make_node("Reshape", ["sigma", "row_shape_4d"], ["sigma_row"]),
        helper.make_node("Mul", ["sigma_row", "sigma_row"], ["sigma_sq"]),
        helper.make_node("Add", ["sigma_sq", "one"], ["denominator_sq"]),
        helper.make_node("Sqrt", ["denominator_sq"], ["denominator"]),
        helper.make_node("Div", ["sample", "denominator"], ["model_input"]),
    ]
    graph = helper.make_graph(
        nodes,
        "model_input",
        [
            _vi("sample", TensorProto.FLOAT, ["batch", CHANNELS, SIZE, SIZE]),
            _vi("step", TensorProto.INT64, ["batch"]),
            _vi("schedule", TensorProto.FLOAT, ["schedule_length"]),
        ],
        [_vi("model_input", TensorProto.FLOAT, ["batch", CHANNELS, SIZE, SIZE])],
        initializer=[
            _const("row_shape_4d", np.array([-1, 1, 1, 1], dtype=np.int64)),
            _const("one", np.array([1.0], dtype=np.float32)),
        ],
    )
    return _model(graph)


def build_solver_step() -> onnx.ModelProto:
    """Euler: `next = sample + (sigma[step + 1] - sigma[step]) * estimate`."""
    nodes = [
        helper.make_node("Gather", ["schedule", "step"], ["sigma"], axis=0),
        helper.make_node("Add", ["step", "one_i64"], ["next_step"]),
        helper.make_node("Gather", ["schedule", "next_step"], ["sigma_next"], axis=0),
        helper.make_node("Sub", ["sigma_next", "sigma"], ["delta"]),
        helper.make_node("Reshape", ["delta", "row_shape_4d"], ["delta_row"]),
        helper.make_node("Mul", ["estimate", "delta_row"], ["increment"]),
        helper.make_node("Add", ["sample", "increment"], ["next_state"]),
    ]
    graph = helper.make_graph(
        nodes,
        "solver_step",
        [
            _vi("sample", TensorProto.FLOAT, ["batch", CHANNELS, SIZE, SIZE]),
            _vi("estimate", TensorProto.FLOAT, ["batch", CHANNELS, SIZE, SIZE]),
            _vi("step", TensorProto.INT64, ["batch"]),
            _vi("schedule", TensorProto.FLOAT, ["schedule_length"]),
        ],
        [_vi("next_state", TensorProto.FLOAT, ["batch", CHANNELS, SIZE, SIZE])],
        initializer=[
            _const("one_i64", np.array([1], dtype=np.int64)),
            _const("row_shape_4d", np.array([-1, 1, 1, 1], dtype=np.int64)),
        ],
    )
    return _model(graph)


def build_guidance_combine() -> onnx.ModelProto:
    """`uncond + scale * (cond - uncond)`, the definition of CFG."""
    nodes = [
        helper.make_node("Sub", ["conditional", "unconditional"], ["difference"]),
        helper.make_node("Reshape", ["scale", "row_shape_4d"], ["scale_row"]),
        helper.make_node("Mul", ["difference", "scale_row"], ["guided"]),
        helper.make_node("Add", ["unconditional", "guided"], ["estimate"]),
    ]
    graph = helper.make_graph(
        nodes,
        "guidance_combine",
        [
            _vi("unconditional", TensorProto.FLOAT, ["batch", CHANNELS, SIZE, SIZE]),
            _vi("conditional", TensorProto.FLOAT, ["batch", CHANNELS, SIZE, SIZE]),
            _vi("scale", TensorProto.FLOAT, ["batch"]),
        ],
        [_vi("estimate", TensorProto.FLOAT, ["batch", CHANNELS, SIZE, SIZE])],
        initializer=[_const("row_shape_4d", np.array([-1, 1, 1, 1], dtype=np.int64))],
    )
    return _model(graph)


def build_continue_predicate() -> onnx.ModelProto:
    """`continue[1] = not all(done)`."""
    nodes = [
        helper.make_node("Cast", ["done"], ["done_i32"], to=TensorProto.INT32),
        helper.make_node("ReduceMin", ["done_i32", "axis_0"], ["all_done_i32"], keepdims=1),
        helper.make_node("Cast", ["all_done_i32"], ["all_done"], to=TensorProto.BOOL),
        helper.make_node("Not", ["all_done"], ["continue"]),
    ]
    graph = helper.make_graph(
        nodes,
        "continue_predicate",
        [_vi("done", TensorProto.BOOL, ["batch"])],
        [_vi("continue", TensorProto.BOOL, [1])],
        initializer=[_const("axis_0", np.array([0], dtype=np.int64))],
    )
    return _model(graph)


def build_latent_row_shape() -> onnx.ModelProto:
    """The per-row latent shape the RNG draws into."""
    graph = helper.make_graph(
        [helper.make_node("Identity", ["row_shape"], ["shape"])],
        "latent_row_shape",
        [],
        [_vi("shape", TensorProto.INT64, [3])],
        initializer=[_const("row_shape", np.array([CHANNELS, SIZE, SIZE], dtype=np.int64))],
    )
    return _model(graph)


def build_latent_noise() -> onnx.ModelProto:
    """Counter-based RNG: a deterministic function of `(seed, offset, index)`.

    The draw depends on every input, so a workflow that forgets to route the
    seed, reuses a counter, or reshapes with the wrong row shape produces a
    different latent instead of quietly working.
    """
    nodes = [
        # key[batch, 1] = seed * 1000003 + offset * 9176 + 1
        helper.make_node("Mul", ["seed", "seed_stride"], ["seed_term"]),
        helper.make_node("Mul", ["offset", "offset_stride"], ["offset_term"]),
        helper.make_node("Add", ["seed_term", "offset_term"], ["key_flat"]),
        helper.make_node("Reshape", ["key_flat", "column_shape"], ["key"]),
        # counter[batch, ROW_ELEMENTS] = key + element index, hashed with an
        # integer Lehmer step. The hash is deliberately integer-only: a
        # `frac(sin(x) * 43758)` hash amplifies a one-ulp difference between two
        # `sin` implementations by four orders of magnitude, which would make
        # this fixture disagree with any independent reference for no reason
        # that has anything to do with the workflow.
        helper.make_node("Add", ["key", "indices"], ["counter"]),
        helper.make_node("Mod", ["counter", "modulus"], ["base"]),
        helper.make_node("Mul", ["base", "multiplier"], ["stepped"]),
        helper.make_node("Add", ["stepped", "increment"], ["biased"]),
        helper.make_node("Mod", ["biased", "modulus"], ["hashed"]),
        helper.make_node("Cast", ["hashed"], ["hashed_f"], to=TensorProto.FLOAT),
        helper.make_node("Div", ["hashed_f", "modulus_f"], ["unit"]),
        helper.make_node("Mul", ["unit", "two"], ["doubled"]),
        helper.make_node("Sub", ["doubled", "one"], ["flat_noise"]),
        # Reshape to [batch] ++ row_shape, so the row shape input is load-bearing.
        helper.make_node("Shape", ["seed"], ["batch_shape"]),
        helper.make_node("Concat", ["batch_shape", "row_shape"], ["noise_shape"], axis=0),
        helper.make_node("Reshape", ["flat_noise", "noise_shape"], ["noise"]),
        helper.make_node("Add", ["offset", "one_i64"], ["next_offset"]),
    ]
    graph = helper.make_graph(
        nodes,
        "latent_noise",
        [
            _vi("seed", TensorProto.INT64, ["batch"]),
            _vi("offset", TensorProto.INT64, ["batch"]),
            _vi("row_shape", TensorProto.INT64, ["row_rank"]),
        ],
        [
            _vi("noise", TensorProto.FLOAT, ["batch", CHANNELS, SIZE, SIZE]),
            _vi("next_offset", TensorProto.INT64, ["batch"]),
        ],
        initializer=[
            _const("seed_stride", np.array([1000003], dtype=np.int64)),
            _const("offset_stride", np.array([9176], dtype=np.int64)),
            _const("modulus", np.array([2147483647], dtype=np.int64)),
            _const("multiplier", np.array([1103515245], dtype=np.int64)),
            _const("increment", np.array([12345], dtype=np.int64)),
            _const("modulus_f", np.array([2147483647.0], dtype=np.float32)),
            _const("column_shape", np.array([-1, 1], dtype=np.int64)),
            _const("indices", np.arange(ROW_ELEMENTS, dtype=np.int64).reshape(1, ROW_ELEMENTS)),
            _const("two", np.array([2.0], dtype=np.float32)),
            _const("one", np.array([1.0], dtype=np.float32)),
            _const("one_i64", np.array([1], dtype=np.int64)),
        ],
    )
    return _model(graph)


# ── independent reference ───────────────────────────────────────────────────


def _embedding_table() -> np.ndarray:
    rng = np.random.default_rng(20260821)
    return rng.standard_normal((VOCAB, HIDDEN)).astype(np.float32) * 0.25


def _draw_noise(seed: int, offset: int, rows: int) -> np.ndarray:
    """The same counter-based draw `latent_noise` performs, in numpy.

    Integer-only up to the final division, so this is bit-comparable with the
    ONNX graph rather than merely close to it.
    """
    modulus = 2147483647
    key = np.array([[seed * 1000003 + offset * 9176]] * rows, dtype=np.int64)
    counter = key + np.arange(ROW_ELEMENTS, dtype=np.int64).reshape(1, ROW_ELEMENTS)
    base = np.mod(counter, modulus)
    hashed = np.mod(base * 1103515245 + 12345, modulus)
    unit = hashed.astype(np.float32) / np.float32(float(modulus))
    flat = unit * np.float32(2.0) - np.float32(1.0)
    return flat.reshape(rows, CHANNELS, SIZE, SIZE)


def reference_run(
    prompt_tokens: list[int],
    negative_tokens: list[int],
    seed: int,
    guidance_scale: float,
    steps: int,
) -> dict:
    """Simulate the converted workflow independently of ONNX Runtime.

    This is a second implementation of what the *emitted metadata* says should
    happen: draw a seeded latent, encode both prompts, run guidance before the
    solver, index the schedule by the loop iteration, and decode at the end. It
    shares no code with the ONNX graphs, so agreement between the two is real
    evidence that the converted workflow is wired the way the importer claims.

    The seeded draw stays in float32 because it is a bit-exact hash whose input
    is large: evaluating the same sine in double precision is a different
    function, not a more accurate one. Everything after it is computed in
    float64, so the reference is the higher-precision answer and the runtime's
    float32 execution is expected to agree to roughly 1e-3, not exactly.
    """
    table = _embedding_table().astype(np.float64)
    sigmas = _sigmas().astype(np.float64)
    timesteps = sigmas[:STEPS]

    def encode(tokens: list[int]) -> np.ndarray:
        return table[np.array(tokens, dtype=np.int64)][None, ...]

    conditional = encode(prompt_tokens).mean(axis=(1, 2)).reshape(1, 1, 1, 1)
    unconditional = encode(negative_tokens).mean(axis=(1, 2)).reshape(1, 1, 1, 1)

    latent = _draw_noise(seed, 0, 1).astype(np.float64)
    for step in range(steps):
        sigma = sigmas[step]
        timestep = timesteps[step]
        model_input = latent / np.sqrt(sigma * sigma + 1.0)

        def denoise(cond: np.ndarray) -> np.ndarray:
            return model_input * 0.6 + cond + timestep * 0.05

        estimate_uncond = denoise(unconditional)
        estimate_cond = denoise(conditional)
        estimate = estimate_uncond + guidance_scale * (estimate_cond - estimate_uncond)
        latent = latent + (sigmas[step + 1] - sigma) * estimate

    image = np.tanh(latent[:, :3] * 0.4)
    return {
        "prompt_tokens": prompt_tokens,
        "negative_tokens": negative_tokens,
        "seed": seed,
        "guidance_scale": guidance_scale,
        "steps": steps,
        "tolerance": 1e-3,
        "latent_shape": list(latent.shape),
        "latent": [float(value) for value in latent.reshape(-1)],
        "image_shape": list(image.shape),
        "image": [float(value) for value in image.reshape(-1)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "tests/fixtures/comfyui_workflows/txt2img_sd15",
    )
    args = parser.parse_args()
    out = args.out
    _save(build_text_encoder(), out / "text_encoder/model.onnx.textproto")
    _save(build_denoiser(), out / "denoiser/model.onnx.textproto")
    _save(build_vae_decoder(), out / "vae_decoder/model.onnx.textproto")
    _save(build_diffusion_schedule(), out / "policies/diffusion_schedule.onnx.textproto")
    _save(build_diffusion_timesteps(), out / "policies/diffusion_timesteps.onnx.textproto")
    _save(build_schedule_lookup(), out / "policies/schedule_lookup.onnx.textproto")
    _save(build_model_input(), out / "policies/model_input.onnx.textproto")
    _save(build_solver_step(), out / "policies/solver_step.onnx.textproto")
    _save(build_guidance_combine(), out / "policies/guidance_combine.onnx.textproto")
    _save(build_continue_predicate(), out / "policies/continue_predicate.onnx.textproto")
    _save(build_latent_row_shape(), out / "policies/latent_row_shape.onnx.textproto")
    _save(build_latent_noise(), out / "policies/latent_noise.onnx.textproto")

    reference = reference_run(
        prompt_tokens=[3, 7, 11, 2],
        negative_tokens=[0, 1, 0, 1],
        seed=20260821,
        guidance_scale=7.5,
        steps=STEPS,
    )
    path = out / "reference.json"
    path.write_text(json.dumps(reference, indent=2) + "\n")
    print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
