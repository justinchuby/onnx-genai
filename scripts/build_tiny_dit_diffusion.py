#!/usr/bin/env python3
"""Build a tiny but REAL DiT diffusion denoiser fixture using Mobius.

Unlike the hand-written `build_tiny_diffusion.py` toy denoisers, this exports an
actual diffusion-transformer architecture (patch embed + AdaLN + self/cross
attention + feed-forward) via `mobius.models.dit.DiTTransformer2DModel`, with
the standard denoiser I/O contract:

    sample [B, C, H, W] + timestep [B] (int64) + encoder_hidden_states [B, S, D]
        -> noise_pred [B, C, H, W]

Weights are seeded random (small), so the fixture is deterministic and runnable
in ONNX Runtime; outputs are not meaningful — this validates that onnx-genai's
iterative/DDIM pipeline drives a *real* denoiser architecture (rank-4 latents +
int64 timestep + a DDIM scheduler), not a toy Add/Mul graph.

Requires the `onnx` conda env (torch + diffusers + mobius):
    conda run -n onnx python scripts/build_tiny_dit_diffusion.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx_ir as ir

METADATA = """\
pipeline:
  models:
    denoiser:
      filename: denoiser.onnx
      type: denoiser
  dataflow:
    - from: denoiser.noise_pred
      to: denoiser.sample
  strategy:
    kind: iterative
    denoiser: denoiser
    num_steps: 3
    timestep_input: timestep
    scheduler_config:
      kind: ddim
      num_train_timesteps: 1000
      beta_start: 0.00085
      beta_end: 0.012
"""


def build(output_dir: Path) -> None:
    from mobius import build_from_module
    from mobius.models.dit import DiTConfig, DiTTransformer2DModel

    cfg = DiTConfig(
        in_channels=4,
        out_channels=4,
        patch_size=2,
        hidden_size=32,
        num_layers=1,
        num_attention_heads=2,
        cross_attention_dim=16,
        caption_channels=16,
        sample_size=8,
    )
    pkg = build_from_module(DiTTransformer2DModel(cfg), cfg, "denoising")
    model = pkg["model"]

    rng = np.random.default_rng(0)
    for name, init in model.graph.initializers.items():
        if init.const_value is None:
            shape = tuple(int(d) for d in init.shape)
            arr = (rng.standard_normal(shape) * 0.05).astype(np.float32)
            init.const_value = ir.Tensor(arr, name=name)

    output_dir.mkdir(parents=True, exist_ok=True)
    ir.save(model, output_dir / "denoiser.onnx")
    (output_dir / "inference_metadata.yaml").write_text(METADATA)

    # Smoke check with ONNX Runtime.
    import onnxruntime as ort

    sess = ort.InferenceSession(
        str(output_dir / "denoiser.onnx"), providers=["CPUExecutionProvider"]
    )
    out = sess.run(
        None,
        {
            "sample": rng.standard_normal((1, 4, 8, 8)).astype(np.float32),
            "timestep": np.array([10], dtype=np.int64),
            "encoder_hidden_states": rng.standard_normal((1, 4, 16)).astype(np.float32),
        },
    )
    assert out[0].shape == (1, 4, 8, 8), out[0].shape
    size = (output_dir / "denoiser.onnx").stat().st_size
    print(f"Wrote {output_dir} (denoiser.onnx {size} bytes); noise_pred {out[0].shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/tiny-dit-diffusion"),
    )
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
