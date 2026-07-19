#!/usr/bin/env python3
"""Validate inpainting (9-channel UNet) via combined export vs diffusers.

An inpainting UNet takes 9 input channels (latent[4] + mask[1] + masked_latent[4])
and predicts the 4-channel noise. Like SDXL/ControlNet it fits the pipeline with NO
runtime change: the fused denoiser takes the 4-ch loop-carried `sample` plus two
constant inputs `mask`/`masked_latent`, concatenating them for the UNet.

No tiny inpainting model exists, so we construct one by expanding the small-SD
UNet's conv_in 4->9 (the standard way inpainting UNets are made), with the extra 5
channels randomized so mask/masked_latent have effect. Then export via
checkpoint_export and check the fused ONNX matches the torch UNet and that
mask/masked_latent change the output.

Run (conda `onnx`):  conda run -n onnx python scripts/inpaint_e2e.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / "target" / "inpaint-e2e"
MODEL = "OFA-Sys/small-stable-diffusion-v0"
SIZE = 64


def main() -> int:
    from diffusers import StableDiffusionPipeline

    from mobius.integrations.onnx_genai.checkpoint_export import export_checkpoint

    WORK.mkdir(parents=True, exist_ok=True)
    inpaint_dir = WORK / "inpaint-model"

    print("constructing a 9-channel inpainting UNet (conv_in 4->9) ...", flush=True)
    pipe = StableDiffusionPipeline.from_pretrained(MODEL, safety_checker=None)
    unet = pipe.unet
    old = unet.conv_in
    new = torch.nn.Conv2d(9, old.out_channels, old.kernel_size, old.stride, old.padding)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :4] = old.weight
        torch.nn.init.normal_(new.weight[:, 4:], std=0.05)  # extra 5 channels -> effect
        new.bias.copy_(old.bias)
    unet.conv_in = new
    unet.register_to_config(in_channels=9)
    pipe.save_pretrained(str(inpaint_dir))

    print("exporting inpainting denoiser via checkpoint_export ...", flush=True)
    exported = export_checkpoint(str(inpaint_dir), str(WORK / "fused"), height=SIZE, width=SIZE,
                                 components=("denoiser",))
    assert exported.inpaint, "checkpoint_export did not detect the inpainting model"
    assert exported.in_channels == 4, f"latent channels should be 4, got {exported.in_channels}"

    unet.eval()
    sz = SIZE // 8
    torch.manual_seed(0)
    sample = torch.randn(1, 4, sz, sz)
    mask = torch.rand(1, 1, sz, sz)
    masked = torch.randn(1, 4, sz, sz)
    ehs = torch.randn(1, 77, unet.config.cross_attention_dim)
    t = 500
    with torch.no_grad():
        ref = unet(torch.cat([sample, mask, masked], 1), t, encoder_hidden_states=ehs).sample.numpy()
        base = unet(torch.cat([sample, torch.zeros_like(mask), torch.zeros_like(masked)], 1),
                    t, encoder_hidden_states=ehs).sample.numpy()

    import onnxruntime as ort
    sess = ort.InferenceSession(str(WORK / "fused" / "denoiser.onnx"), providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    feed = dict(zip(names, [sample.numpy().astype("<f4"), np.array([t], dtype=np.int64),
                            ehs.numpy().astype("<f4"), mask.numpy().astype("<f4"),
                            masked.numpy().astype("<f4")]))
    og = sess.run(None, feed)[0]

    match = np.abs(og - ref)
    effect = np.abs(ref - base)
    print("\n=== inpainting (9-channel) combined export ===")
    print(f"  fused ONNX vs diffusers 9ch UNet: max|diff|={match.max():.3e} mean={match.mean():.3e}")
    print(f"  mask/masked_latent effect        : max|diff|={effect.max():.3e}")
    assert match.max() < 1e-2, f"mismatch: {match.max()}"
    assert effect.max() > 1e-3, f"mask/masked_latent had no effect: {effect.max()}"
    print("\nOK: inpainting export matches diffusers AND mask/masked_latent take effect")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
