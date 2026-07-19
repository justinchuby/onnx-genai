#!/usr/bin/env python3
"""Validate ControlNet via combined ControlNet+UNet export.

Like SDXL, ControlNet fits onnx-genai's declarative pipeline with NO runtime
change: the denoiser is a fused ControlNet+UNet taking an extra constant
`controlnet_cond` image input (the ControlNet produces residuals injected into the
UNet). This creates a dim-matched ControlNet (ControlNetModel.from_unet, with its
zero-init output projections randomized so it has effect), exports the fused
denoiser via checkpoint_export, and checks it (a) MATCHES a diffusers reference
(UNet + ControlNet residuals) and (b) DIFFERS from the no-ControlNet UNet.

Run (conda `onnx`):  conda run -n onnx python scripts/controlnet_e2e.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / "target" / "controlnet-e2e"
MODEL = "OFA-Sys/small-stable-diffusion-v0"
SIZE = 64


def main() -> int:
    from diffusers import ControlNetModel, UNet2DConditionModel

    from mobius.integrations.onnx_genai.checkpoint_export import export_checkpoint

    WORK.mkdir(parents=True, exist_ok=True)
    cn_dir = WORK / "controlnet"

    unet = UNet2DConditionModel.from_pretrained(MODEL, subfolder="unet").eval()
    print("creating dim-matched ControlNet (from_unet) ...", flush=True)
    cn = ControlNetModel.from_unet(unet)
    # from_unet zero-inits the output projections (untrained ControlNet = no
    # effect); randomize them so the ControlNet measurably changes the output.
    with torch.no_grad():
        for m in list(cn.controlnet_down_blocks) + [cn.controlnet_mid_block]:
            torch.nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, std=0.02)
    cn.save_pretrained(str(cn_dir))
    cn.eval()

    print("exporting fused ControlNet+UNet + base denoisers ...", flush=True)
    fused = export_checkpoint(
        MODEL, str(WORK / "fused"), height=SIZE, width=SIZE, components=("denoiser",),
        controlnet=str(cn_dir),
    )
    base = export_checkpoint(
        MODEL, str(WORK / "base"), height=SIZE, width=SIZE, components=("denoiser",),
    )

    ch, sz = unet.config.in_channels, SIZE // 8
    torch.manual_seed(0)
    sample = torch.randn(1, ch, sz, sz)
    cond = torch.randn(1, 77, unet.config.cross_attention_dim)
    ctl_img = torch.rand(1, 3, SIZE, SIZE)
    timestep = 500

    # diffusers reference: UNet with ControlNet residuals.
    with torch.no_grad():
        down, mid = cn(sample, timestep, encoder_hidden_states=cond,
                       controlnet_cond=ctl_img, return_dict=False)
        ref = unet(sample, timestep, encoder_hidden_states=cond,
                   down_block_additional_residuals=down,
                   mid_block_additional_residual=mid).sample.numpy()
        base_out = unet(sample, timestep, encoder_hidden_states=cond).sample.numpy()

    import onnxruntime as ort

    def run(path, feed):
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        names = [i.name for i in sess.get_inputs()]
        return sess.run(None, dict(zip(names, feed)))[0]

    og_fused = run(
        WORK / "fused" / fused.denoiser_filename,
        [sample.numpy().astype("<f4"), np.array([timestep], dtype=np.int64),
         cond.numpy().astype("<f4"), ctl_img.numpy().astype("<f4")],
    )

    match = np.abs(og_fused - ref)
    ctl_effect = np.abs(ref - base_out)
    print("\n=== ControlNet combined export ===")
    print(f"  fused ONNX vs diffusers(unet+controlnet): max|diff|={match.max():.3e} mean={match.mean():.3e}")
    print(f"  controlnet effect (ref vs base UNet)     : max|diff|={ctl_effect.max():.3e}")
    assert match.max() < 1e-2, f"fused export does not match diffusers: {match.max()}"
    assert ctl_effect.max() > 1e-3, f"ControlNet had no effect: {ctl_effect.max()}"
    print("\nOK: fused ControlNet+UNet export matches diffusers AND the ControlNet has effect")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
