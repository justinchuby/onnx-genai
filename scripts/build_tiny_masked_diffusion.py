#!/usr/bin/env python3
"""Build a tiny synthetic masked-diffusion language-model fixture.

The "LM" takes ``input_ids`` [1, 4] int64 and emits fixed ``logits`` [1, 4, 6]
whose per-position argmax is the target sequence ``[2, 3, 4, 5]`` with *strictly
decreasing* confidence (position 0 highest). Driven by onnx-genai's
``masked_diffusion`` scheduler (mask_token_id=1, num_steps=4), the loop unmasks
one highest-confidence position per step, so the all-mask seed [1,1,1,1] refines
to [2,3,4,5] — a deterministic proof of the discrete language-diffusion loop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnx_ir as ir

TARGET = [2, 3, 4, 5]
VOCAB = 6
MASK_TOKEN = 1

METADATA = """\
pipeline:
  models:
    denoiser:
      filename: lm.onnx.textproto
      type: denoiser
  dataflow:
    - from: denoiser.logits
      to: denoiser.input_ids
  strategy:
    kind: iterative
    denoiser: denoiser
    num_steps: 4
    scheduler_config:
      kind: masked_diffusion
      mask_token_id: 1
"""


def build(output_dir: Path) -> None:
    # Fixed logits: argmax at TARGET[s], confidence = 10 - 2*s (decreasing).
    logits = np.zeros((1, len(TARGET), VOCAB), dtype=np.float32)
    for s, tok in enumerate(TARGET):
        logits[0, s, tok] = 10.0 - 2.0 * s

    input_ids = ir.Value(
        name="input_ids", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([1, 4])
    )
    const = ir.Node(
        "",
        "Constant",
        [],
        [ir.AttrTensor("value", ir.Tensor(logits, name="logits_value"))],
        outputs=[ir.Value(name="logits")],
        name="fixed_logits",
    )
    const.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
    const.outputs[0].shape = ir.Shape([1, 4, VOCAB])

    graph = ir.Graph(
        [input_ids],
        [const.outputs[0]],
        nodes=[const],
        opset_imports={"": 13},
        name="tiny_masked_diffusion_lm",
    )
    model = ir.Model(graph, ir_version=8, producer_name="onnx-genai tiny-masked-diffusion")

    output_dir.mkdir(parents=True, exist_ok=True)
    ir.save(model, output_dir / "lm.onnx.textproto", format="textproto")
    onnx.checker.check_model(ir.to_proto(model))
    (output_dir / "inference_metadata.yaml").write_text(METADATA)
    print(f"Wrote {output_dir} (target {TARGET}, mask {MASK_TOKEN})")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("tests/fixtures/tiny-masked-diffusion"))
    build(ap.parse_args().output)
