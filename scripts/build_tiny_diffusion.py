#!/usr/bin/env python3
"""Build the tiny deterministic diffusion (iterative) pipeline fixture.

This hand-constructs the minimal ONNX pair needed to exercise the iterative
pipeline seam (`PipelineEngine::run_pipeline` over a `kind: iterative` strategy):

  * ``denoiser.onnx``  — one denoise step: ``denoised = (sample + cond) * 0.5``.
      * ``sample`` is loop-carried (a ``denoiser.denoised -> denoiser.sample``
        self-edge feeds each step's output into the next step's input).
      * ``cond`` is constant conditioning re-supplied every step.
  * ``vae.onnx``       — a ``final_only`` single-pass decoder: ``image = latent * 2 + 1``,
      fed the final denoiser output via ``denoiser.denoised -> vae.latent``.

The transforms are affine and deterministic so the Rust test can assert the
exact tensor after N steps in closed form.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnx_ir as ir


def tensor_value(name: str, dtype: ir.DataType, shape: list[int | str]) -> ir.Value:
    return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))


def initializer(name: str, array: np.ndarray) -> ir.Value:
    return ir.Value(name=name, const_value=ir.Tensor(array, name=name))


def node(op_type: str, inputs: list[ir.Value], output: str, *, name: str = "") -> ir.Node:
    return ir.Node("", op_type, inputs, (), outputs=[ir.Value(name=output)], name=name)


def save_model(model: ir.Model, path: Path) -> None:
    ir.save(model, path)
    onnx.checker.check_model(path)


def build_denoiser(path: Path) -> None:
    # denoised = (sample + cond) * 0.5
    sample = tensor_value("sample", ir.DataType.FLOAT, [1, 4])
    cond = tensor_value("cond", ir.DataType.FLOAT, [1, 4])
    half = initializer("half", np.array([0.5], dtype=np.float32))

    add = node("Add", [sample, cond], "summed")
    mul = node("Mul", [add.outputs[0], half], "denoised")
    mul.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    mul.outputs[0].shape = ir.Shape([1, 4])

    graph = ir.Graph(
        [sample, cond],
        [mul.outputs[0]],
        nodes=[add, mul],
        initializers=[half],
        opset_imports={"": 13},
        name="tiny_diffusion_denoiser",
    )
    save_model(ir.Model(graph, ir_version=8, producer_name="onnx-genai tiny-diffusion"), path)


def build_denoiser_multi(path: Path) -> None:
    # Two independent loop-carried states + one constant conditioning input:
    #   x_next = (x + cond) * 0.5     (loop-carried x)
    #   y_next = (y + x)    * 0.5     (loop-carried y, coupled to this step's x)
    x = tensor_value("x", ir.DataType.FLOAT, [1, 4])
    y = tensor_value("y", ir.DataType.FLOAT, [1, 4])
    cond = tensor_value("cond", ir.DataType.FLOAT, [1, 4])
    half = initializer("half_m", np.array([0.5], dtype=np.float32))

    x_sum = node("Add", [x, cond], "x_sum")
    x_next = node("Mul", [x_sum.outputs[0], half], "x_next")
    x_next.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    x_next.outputs[0].shape = ir.Shape([1, 4])

    y_sum = node("Add", [y, x], "y_sum")
    y_next = node("Mul", [y_sum.outputs[0], half], "y_next")
    y_next.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    y_next.outputs[0].shape = ir.Shape([1, 4])

    graph = ir.Graph(
        [x, y, cond],
        [x_next.outputs[0], y_next.outputs[0]],
        nodes=[x_sum, x_next, y_sum, y_next],
        initializers=[half],
        opset_imports={"": 13},
        name="tiny_diffusion_denoiser_multi",
    )
    save_model(ir.Model(graph, ir_version=8, producer_name="onnx-genai tiny-diffusion"), path)


def build_denoiser_step(path: Path) -> None:
    # Step-aware denoiser: consumes a per-step timestep scalar `t`.
    #   denoised = sample + t   (t broadcasts over [1,4])
    # With sample_0 = 0 and loop edge denoised->sample, the result after N steps
    # is the running sum of the per-step timesteps.
    sample = tensor_value("sample", ir.DataType.FLOAT, [1, 4])
    t = tensor_value("t", ir.DataType.FLOAT, [1])
    add = node("Add", [sample, t], "denoised")
    add.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    add.outputs[0].shape = ir.Shape([1, 4])

    graph = ir.Graph(
        [sample, t],
        [add.outputs[0]],
        nodes=[add],
        initializers=[],
        opset_imports={"": 13},
        name="tiny_diffusion_denoiser_step",
    )
    save_model(ir.Model(graph, ir_version=8, producer_name="onnx-genai tiny-diffusion"), path)


def build_vae(path: Path) -> None:
    latent = tensor_value("latent", ir.DataType.FLOAT, [1, 4])
    two = initializer("two", np.array([2.0], dtype=np.float32))
    one = initializer("one", np.array([1.0], dtype=np.float32))

    mul = node("Mul", [latent, two], "scaled")
    add = node("Add", [mul.outputs[0], one], "image")
    add.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    add.outputs[0].shape = ir.Shape([1, 4])

    graph = ir.Graph(
        [latent],
        [add.outputs[0]],
        nodes=[mul, add],
        initializers=[two, one],
        opset_imports={"": 13},
        name="tiny_diffusion_vae",
    )
    save_model(ir.Model(graph, ir_version=8, producer_name="onnx-genai tiny-diffusion"), path)


METADATA = """\
pipeline:
  models:
    denoiser:
      filename: denoiser.onnx
      type: denoiser
    vae:
      filename: vae.onnx
      type: vae
  dataflow:
    - from: denoiser.denoised
      to: denoiser.sample
      dtype: fp32
    - from: denoiser.denoised
      to: vae.latent
      dtype: fp32
  strategy:
    kind: iterative
    denoiser: denoiser
    num_steps: 3
  phases:
    vae:
      run_on: final_only
"""


def write_metadata(path: Path) -> None:
    path.write_text(METADATA)


def validate_with_ort(output_dir: Path) -> None:
    import onnxruntime as ort

    denoiser = ort.InferenceSession(
        str(output_dir / "denoiser.onnx"), providers=["CPUExecutionProvider"]
    )
    vae = ort.InferenceSession(str(output_dir / "vae.onnx"), providers=["CPUExecutionProvider"])

    sample = np.zeros((1, 4), dtype=np.float32)
    cond = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    for _ in range(3):
        sample = denoiser.run(None, {"sample": sample, "cond": cond})[0]
    # Closed form: s_{k} = (s_{k-1} + c) / 2, s_0 = 0  ->  s_3 = c * (7/8)
    expected = cond * (7.0 / 8.0)
    assert np.allclose(sample, expected), (sample, expected)
    image = vae.run(None, {"latent": sample})[0]
    assert np.allclose(image, sample * 2.0 + 1.0), image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/tiny-diffusion"),
        help="Output pipeline directory (default: tests/fixtures/tiny-diffusion)",
    )
    parser.add_argument(
        "--no-validate", action="store_true", help="Skip ONNX Runtime smoke validation"
    )
    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    build_denoiser(output_dir / "denoiser.onnx")
    build_denoiser_multi(output_dir / "denoiser_multi.onnx")
    build_denoiser_step(output_dir / "denoiser_step.onnx")
    build_vae(output_dir / "vae.onnx")
    write_metadata(output_dir / "inference_metadata.yaml")
    if not args.no_validate:
        validate_with_ort(output_dir)

    total_size = sum(p.stat().st_size for p in output_dir.iterdir() if p.is_file())
    print(f"Wrote {output_dir} ({total_size} bytes)")


if __name__ == "__main__":
    main()
