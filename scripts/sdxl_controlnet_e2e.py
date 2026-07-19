#!/usr/bin/env python3
"""Validate SDXL + ControlNet combined export vs diffusers.

Composes the SDXL dual-encoder and ControlNet residual paths: the fused denoiser
takes sample/timestep/encoder_hidden_states/text_embeds/time_ids/controlnet_cond.
Uses a dim-matched ControlNet (from_unet on the SDXL UNet, output projections
randomized). Checks the fused ONNX matches diffusers (sdxl unet + controlnet
residuals) and differs from the no-controlnet SDXL UNet.

Run (conda `onnx`):  conda run -n onnx python scripts/sdxl_controlnet_e2e.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / "target" / "sdxl-cn-e2e"
MODEL = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"
SIZE = 64


def main() -> int:
    from diffusers import ControlNetModel, StableDiffusionXLPipeline

    from mobius.integrations.onnx_genai.checkpoint_export import export_checkpoint

    WORK.mkdir(parents=True, exist_ok=True)
    cn_dir = WORK / "controlnet"
    pipe = StableDiffusionXLPipeline.from_pretrained(MODEL)
    unet, te1, te2 = pipe.unet.eval(), pipe.text_encoder.eval(), pipe.text_encoder_2.eval()
    tok1, tok2 = pipe.tokenizer, pipe.tokenizer_2

    print("creating dim-matched SDXL ControlNet ...", flush=True)
    cn = ControlNetModel.from_unet(unet)
    with torch.no_grad():
        for m in list(cn.controlnet_down_blocks) + [cn.controlnet_mid_block]:
            torch.nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, std=0.02)
    cn.save_pretrained(str(cn_dir))
    cn.eval()

    print("exporting fused SDXL ControlNet denoiser ...", flush=True)
    export_checkpoint(MODEL, str(WORK / "fused"), height=SIZE, width=SIZE,
                      components=("denoiser",), controlnet=str(cn_dir))

    ch, sz = unet.config.in_channels, SIZE // 8

    def enc(prompt):
        i1 = tok1(prompt, padding="max_length", max_length=tok1.model_max_length, truncation=True, return_tensors="pt").input_ids
        i2 = tok2(prompt, padding="max_length", max_length=tok2.model_max_length, truncation=True, return_tensors="pt").input_ids
        with torch.no_grad():
            o1, o2 = te1(i1, output_hidden_states=True), te2(i2, output_hidden_states=True)
        return torch.cat([o1.hidden_states[-2], o2.hidden_states[-2]], dim=-1), o2[0]

    ehs, pooled = enc("a cat")
    torch.manual_seed(0)
    sample = torch.randn(1, ch, sz, sz)
    time_ids = torch.tensor([[SIZE, SIZE, 0, 0, SIZE, SIZE]], dtype=torch.float32)
    ctl_img = torch.rand(1, 3, SIZE, SIZE)
    t = 500
    added = {"text_embeds": pooled, "time_ids": time_ids}
    with torch.no_grad():
        down, mid = cn(sample, t, encoder_hidden_states=ehs, added_cond_kwargs=added,
                       controlnet_cond=ctl_img, return_dict=False)
        ref = unet(sample, t, encoder_hidden_states=ehs, added_cond_kwargs=added,
                   down_block_additional_residuals=down, mid_block_additional_residual=mid).sample.numpy()
        base = unet(sample, t, encoder_hidden_states=ehs, added_cond_kwargs=added).sample.numpy()

    import onnxruntime as ort
    sess = ort.InferenceSession(str(WORK / "fused" / "denoiser.onnx"), providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    feed = dict(zip(names, [sample.numpy().astype("<f4"), np.array([t], dtype=np.int64),
                            ehs.numpy().astype("<f4"), pooled.numpy().astype("<f4"),
                            time_ids.numpy().astype("<f4"), ctl_img.numpy().astype("<f4")]))
    og = sess.run(None, feed)[0]

    match = np.abs(og - ref)
    effect = np.abs(ref - base)
    print("\n=== SDXL + ControlNet combined export ===")
    print(f"  fused ONNX vs diffusers(sdxl+controlnet): max|diff|={match.max():.3e} mean={match.mean():.3e}")
    print(f"  controlnet effect (ref vs base)         : max|diff|={effect.max():.3e}")
    assert match.max() < 1e-2, f"mismatch: {match.max()}"
    assert effect.max() > 1e-3, f"no effect: {effect.max()}"
    print("\nOK: fused SDXL ControlNet export matches diffusers AND the ControlNet has effect")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
