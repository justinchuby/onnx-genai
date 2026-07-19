#!/usr/bin/env python3
"""Verify the iterative diffusion pipeline handles batch_size > 1.

Exports a small SD UNet (dynamic batch), builds a minimal denoise-loop metadata,
and runs it with a batch of 2 latents. Confirms onnx-genai produces a batch-2
output — i.e. the loop-carried state, timestep injection, and scheduler are all
batch-generic (a common ComfyUI need; the tool has only exercised batch=1).

Run (conda `onnx`, after `cargo build --release -p onnx-genai --bin run_diffusion`):
    conda run -n onnx python scripts/batch_check.py
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / "target" / "batch-check"
RUNNER = REPO / "target" / "release" / "run_diffusion"
MODEL = "OFA-Sys/small-stable-diffusion-v0"
STEPS = 4
SIZE = 64
BATCH = 2


def ort_lib_dir() -> str:
    hits = sorted(glob.glob(str(REPO / "target/*/build/onnx-genai-ort-sys-*/out/ort-prebuilt/lib")))
    if not hits:
        raise SystemExit("could not locate prebuilt ORT lib dir")
    return hits[-1]


def main() -> int:
    if not RUNNER.exists():
        print(f"missing {RUNNER}", file=sys.stderr)
        return 1
    from diffusers import DDIMScheduler
    from mobius.integrations.onnx_genai.checkpoint_export import export_checkpoint

    WORK.mkdir(parents=True, exist_ok=True)
    pdir = WORK / "pipeline"
    print("exporting small-SD denoiser (dynamic batch) ...", flush=True)
    ex = export_checkpoint(MODEL, str(pdir), height=SIZE, width=SIZE, components=("denoiser",))
    ch, sz = ex.in_channels, SIZE // 8

    sched = DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
                          beta_schedule="scaled_linear", prediction_type="epsilon",
                          set_alpha_to_one=True, steps_offset=0, clip_sample=False)
    sched.set_timesteps(STEPS)
    ts = "".join(f"      - {float(t)}\n" for t in sched.timesteps)
    (pdir / "inference_metadata.yaml").write_text(
        "pipeline:\n  models:\n    denoiser:\n      filename: denoiser.onnx\n      type: denoiser\n"
        "  dataflow:\n    - from: denoiser.noise_pred\n      to: denoiser.sample\n"
        "  strategy:\n    kind: iterative\n    denoiser: denoiser\n"
        f"    num_steps: {STEPS}\n    timestep_input: timestep\n    timesteps:\n" + ts +
        "    scheduler_config:\n      kind: ddim\n      num_train_timesteps: 1000\n"
        "      beta_start: 0.00085\n      beta_end: 0.012\n      beta_schedule: scaled_linear\n"
    )

    g = torch.Generator().manual_seed(0)
    sample = torch.randn(BATCH, ch, sz, sz, generator=g).numpy().astype("<f4")
    cond = torch.randn(BATCH, 77, 768, generator=g).numpy().astype("<f4")
    sample.tofile(pdir / "sample.f32")
    cond.tofile(pdir / "cond.f32")

    env = dict(os.environ)
    env["DYLD_LIBRARY_PATH"] = ort_lib_dir() + ":" + env.get("DYLD_LIBRARY_PATH", "")
    out = pdir / "out.f32"
    print(f"running batch={BATCH} denoise loop through onnx-genai ...", flush=True)
    subprocess.run(
        [str(RUNNER), str(pdir), "denoiser.sample", str(out),
         f"denoiser.sample:{BATCH},{ch},{sz},{sz}:{pdir / 'sample.f32'}",
         f"denoiser.encoder_hidden_states:{BATCH},77,768:{pdir / 'cond.f32'}"],
        env=env, check=True,
    )
    arr = np.fromfile(out, dtype="<f4")
    expected = BATCH * ch * sz * sz
    print(f"\noutput elems={arr.size} expected={expected} (batch {BATCH})")
    assert arr.size == expected, f"batch output size {arr.size} != {expected}"
    assert np.isfinite(arr).all(), "non-finite output"
    b0, b1 = arr.reshape(BATCH, -1)
    assert not np.allclose(b0, b1), "the two batch items are identical (batch not independent)"
    print("OK: batch>1 denoise loop runs, produces independent per-item latents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
