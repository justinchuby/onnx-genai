#!/usr/bin/env python3
"""Full end-to-end image diffusion INSIDE onnx-genai and compare with diffusers.

Unlike diffusion_image.py (which ran only the UNet in onnx-genai and did text
encoding + VAE decode in Python), this exports all three neural components and
runs the whole pipeline through onnx-genai:

    text_encoder (prompt_only)  input_ids -> last_hidden_state
        -> denoiser (iterative, DDIM+CFG)  sample/timestep/encoder_hidden_states
        -> vae (final_only, scale baked in)  latent -> image

onnx-genai therefore produces the final RGB image tensor directly. The
unconditional CFG embedding is supplied externally (empty-prompt), computed by
the same text encoder. We compare onnx-genai's image against diffusers.

Run in the `onnx` conda env after building the runner:
    cargo build -p onnx-genai --bin run_diffusion
    conda run -n onnx python scripts/diffusion_e2e.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / "target" / "diffusion-e2e"
RUNNER = REPO / "target" / "debug" / "run_diffusion"


def save_png(img_chw_m11: np.ndarray, path: Path) -> None:
    from PIL import Image

    img = (img_chw_m11 / 2 + 0.5).clip(0, 1)
    Image.fromarray((img.transpose(1, 2, 0) * 255).round().astype(np.uint8)).save(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="OFA-Sys/small-stable-diffusion-v0")
    ap.add_argument("--prompt", default="a photograph of an astronaut riding a horse")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    if not RUNNER.exists():
        print(f"missing {RUNNER}; build it first", file=sys.stderr)
        return 1
    WORK.mkdir(parents=True, exist_ok=True)
    pdir = WORK / "sd-pipeline"
    pdir.mkdir(parents=True, exist_ok=True)

    unet = UNet2DConditionModel.from_pretrained(args.model, subfolder="unet").eval()
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae").eval()
    text_encoder = CLIPTextModel.from_pretrained(args.model, subfolder="text_encoder").eval()
    tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    sf = getattr(vae.config, "scaling_factor", 0.18215)

    scheduler = DDIMScheduler(
        num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear", prediction_type="epsilon",
        set_alpha_to_one=True, steps_offset=0, clip_sample=False,
    )
    scheduler.set_timesteps(args.steps)
    timesteps = [int(t) for t in scheduler.timesteps]

    def tokenize(text: str) -> torch.Tensor:
        return tokenizer(
            text, padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids

    ids_cond = tokenize(args.prompt)
    ids_uncond = tokenize("")
    with torch.no_grad():
        emb_cond = text_encoder(ids_cond)[0]
        emb_uncond = text_encoder(ids_uncond)[0]

    ch = unet.config.in_channels
    sz = args.size // 8
    g = torch.Generator().manual_seed(args.seed)
    latent0 = torch.randn(1, ch, sz, sz, generator=g) * scheduler.init_noise_sigma

    # diffusers reference (full pipeline).
    lat = latent0.clone()
    with torch.no_grad():
        for t in scheduler.timesteps:
            nc = unet(lat, t, encoder_hidden_states=emb_cond).sample
            nu = unet(lat, t, encoder_hidden_states=emb_uncond).sample
            lat = scheduler.step(nu + args.guidance * (nc - nu), t, lat).prev_sample
        img_ref = vae.decode(lat / sf).sample[0].numpy()

    # Export the three components.
    class UNetWrap(torch.nn.Module):
        def __init__(self, u):
            super().__init__()
            self.u = u

        def forward(self, sample, timestep, encoder_hidden_states):
            return self.u(sample, timestep, encoder_hidden_states=encoder_hidden_states).sample

    class TextWrap(torch.nn.Module):
        def __init__(self, t):
            super().__init__()
            self.t = t

        def forward(self, input_ids):
            return self.t(input_ids)[0]

    class VaeWrap(torch.nn.Module):
        # Bake the diffusers latent-scaling into the VAE so the pipeline can route
        # the raw final latent straight into it.
        def __init__(self, v, scale):
            super().__init__()
            self.v = v
            self.scale = scale

        def forward(self, latent):
            return self.v.decode(latent / self.scale).sample

    print("exporting text_encoder / unet / vae ...", flush=True)
    torch.onnx.export(
        TextWrap(text_encoder), (ids_cond,), str(pdir / "text_encoder.onnx"),
        input_names=["input_ids"], output_names=["last_hidden_state"],
        opset_version=17, dynamo=False,
    )
    torch.onnx.export(
        UNetWrap(unet), (latent0, torch.tensor([timesteps[0]], dtype=torch.long), emb_cond),
        str(pdir / "unet.onnx"),
        input_names=["sample", "timestep", "encoder_hidden_states"],
        output_names=["noise_pred"], opset_version=17, dynamo=False,
    )
    torch.onnx.export(
        VaeWrap(vae, sf), (latent0,), str(pdir / "vae.onnx"),
        input_names=["latent"], output_names=["image"], opset_version=17, dynamo=False,
    )

    # Pipeline metadata emitted by the Mobius integration (proves Mobius can
    # declare the full runnable composite, not just the denoiser).
    from mobius.integrations.onnx_genai import (
        SchedulerConfig,
        build_diffusion_pipeline_metadata,
    )
    import yaml as _yaml

    meta = build_diffusion_pipeline_metadata(
        num_inference_steps=args.steps,
        denoiser_filename="unet.onnx",
        vae_filename="vae.onnx",
        vae_latent_input="latent",
        text_encoder_filename="text_encoder.onnx",
        guidance_scale=args.guidance,
        timesteps=[float(t) for t in timesteps],
        scheduler=SchedulerConfig(
            kind="ddim", num_train_timesteps=1000, beta_start=0.00085,
            beta_end=0.012, beta_schedule="scaled_linear", prediction_type="epsilon",
        ),
    )
    (pdir / "inference_metadata.yaml").write_text(_yaml.safe_dump(meta, sort_keys=False))

    # External inputs: token ids (prompt), initial latent, and the uncond embedding.
    ids_cond.numpy().astype("<i8").tofile(pdir / "ids.i64")
    latent0.numpy().astype("<f4").tofile(pdir / "sample.f32")
    emb_uncond.detach().numpy().astype("<f4").tofile(pdir / "uncond.f32")
    s, d = emb_cond.shape[1], emb_cond.shape[2]
    seq = ids_cond.shape[1]

    print("running onnx-genai full pipeline (text_encoder -> denoiser -> vae) ...", flush=True)
    out_path = pdir / "image.f32"
    subprocess.run(
        [
            str(RUNNER), str(pdir), "vae.image", str(out_path),
            f"text_encoder.input_ids:i64:1,{seq}:{pdir / 'ids.i64'}",
            f"denoiser.sample:1,{ch},{sz},{sz}:{pdir / 'sample.f32'}",
            f"denoiser.encoder_hidden_states.uncond:1,{s},{d}:{pdir / 'uncond.f32'}",
        ],
        check=True,
    )
    # VAE output shape [1,3,H,W].
    hw = args.size
    og = np.fromfile(out_path, dtype="<f4")
    og_img = og.reshape(1, 3, hw, hw)[0]

    diff = np.abs(og_img - img_ref)
    print("\n=== onnx-genai full-pipeline image vs diffusers ===")
    print(f"  max|diff|  = {diff.max():.3e}")
    print(f"  mean|diff| = {diff.mean():.3e}  (pixel range ~[-1,1])")

    save_png(img_ref, WORK / "diffusers.png")
    save_png(og_img, WORK / "onnx_genai.png")
    from PIL import Image

    a, b = Image.open(WORK / "diffusers.png"), Image.open(WORK / "onnx_genai.png")
    combo = Image.new("RGB", (a.width + b.width + 8, max(a.height, b.height)), (255, 255, 255))
    combo.paste(a, (0, 0))
    combo.paste(b, (a.width + 8, 0))
    combo.save(WORK / "compare.png")
    print(f"\nprompt: {args.prompt!r}\nsaved: {WORK/'compare.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
