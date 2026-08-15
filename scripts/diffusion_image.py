#!/usr/bin/env python3
"""Generate a REAL image with a trained Stable Diffusion model using
classifier-free guidance, two ways, and compare:

  (A) the diffusers DDIM + CFG reference loop
  (B) onnx-genai's iterative pipeline (DDIM + CFG) driving the same exported UNet

Both use the empty-prompt unconditional embedding (real CFG), the same seed /
prompt / timesteps / guidance, so onnx-genai should reproduce diffusers.

Run in the `onnx` conda env after building the runner:
    cargo build -p onnx-genai --bin run_diffusion
    conda run -n onnx python scripts/diffusion_image.py \
        --model OFA-Sys/small-stable-diffusion-v0 --steps 20 --guidance 7.5 \
        --size 256 --prompt "a photograph of an astronaut riding a horse"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / "target" / "diffusion-image"
RUNNER = REPO / "target" / "debug" / "run_diffusion"


def save_png(img_chw_m11: np.ndarray, path: Path) -> None:
    from PIL import Image

    img = (img_chw_m11 / 2 + 0.5).clip(0, 1)
    img = (img.transpose(1, 2, 0) * 255).round().astype(np.uint8)
    Image.fromarray(img).save(path)


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
        print(f"missing {RUNNER}; run: cargo build -p onnx-genai --bin run_diffusion", file=sys.stderr)
        return 1
    WORK.mkdir(parents=True, exist_ok=True)

    unet = UNet2DConditionModel.from_pretrained(args.model, subfolder="unet").eval()
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae").eval()
    text_encoder = CLIPTextModel.from_pretrained(args.model, subfolder="text_encoder").eval()
    tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer")

    scheduler = DDIMScheduler(
        num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear", prediction_type="epsilon",
        set_alpha_to_one=True, steps_offset=0, clip_sample=False,
    )
    scheduler.set_timesteps(args.steps)
    timesteps = [int(t) for t in scheduler.timesteps]

    def embed(text: str) -> torch.Tensor:
        ids = tokenizer(
            text, padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids
        with torch.no_grad():
            return text_encoder(ids)[0]

    cond = embed(args.prompt)
    uncond = embed("")

    latent_ch = unet.config.in_channels
    latent_sz = args.size // 8
    g = torch.Generator().manual_seed(args.seed)
    latent0 = torch.randn(1, latent_ch, latent_sz, latent_sz, generator=g)
    # SD scales the initial noise by the scheduler's init_noise_sigma (1.0 for DDIM).
    latent0 = latent0 * scheduler.init_noise_sigma

    # (A) diffusers reference: DDIM + CFG.
    lat = latent0.clone()
    with torch.no_grad():
        for t in scheduler.timesteps:
            nc = unet(lat, t, encoder_hidden_states=cond).sample
            nu = unet(lat, t, encoder_hidden_states=uncond).sample
            noise = nu + args.guidance * (nc - nu)
            lat = scheduler.step(noise, t, lat).prev_sample
    ref_latent = lat.numpy()

    # Export the UNet.
    class Wrap(torch.nn.Module):
        def __init__(self, u):
            super().__init__()
            self.u = u

        def forward(self, sample, timestep, encoder_hidden_states):
            return self.u(sample, timestep, encoder_hidden_states=encoder_hidden_states).sample

    pdir = WORK / "sd-pipeline"
    pdir.mkdir(parents=True, exist_ok=True)
    print("exporting UNet to ONNX ...", flush=True)
    torch.onnx.export(
        Wrap(unet),
        (latent0, torch.tensor([timesteps[0]], dtype=torch.long), cond),
        str(pdir / "unet.onnx"),
        input_names=["sample", "timestep", "encoder_hidden_states"],
        output_names=["noise_pred"],
        opset_version=17,
        dynamo=False,
    )

    ts_yaml = "".join(f"      - {t}.0\n" for t in timesteps)
    (pdir / "inference_metadata.yaml").write_text(
        "pipeline:\n  models:\n    denoiser:\n      filename: unet.onnx\n      type: denoiser\n"
        "  dataflow:\n    - from: denoiser.noise_pred\n      to: denoiser.sample\n"
        "  strategy:\n    kind: iterative\n    denoiser: denoiser\n"
        f"    num_steps: {args.steps}\n    timestep_input: timestep\n"
        f"    guidance_scale: {args.guidance}\n"
        "    cfg_conditioning_input: encoder_hidden_states\n"
        "    timesteps:\n" + ts_yaml +
        "    scheduler_config:\n      kind: ddim\n      num_train_timesteps: 1000\n"
        "      beta_start: 0.00085\n      beta_end: 0.012\n      beta_schedule: scaled_linear\n"
        "      prediction_type: epsilon\n"
    )

    latent0.numpy().astype("<f4").tofile(pdir / "sample.f32")
    cond.detach().numpy().astype("<f4").tofile(pdir / "cond.f32")
    uncond.detach().numpy().astype("<f4").tofile(pdir / "uncond.f32")
    s, d = cond.shape[1], cond.shape[2]

    print("running onnx-genai DDIM+CFG pipeline ...", flush=True)
    out_path = pdir / "og_latent.f32"
    subprocess.run(
        [
            str(RUNNER), str(pdir), "denoiser.sample", str(out_path),
            f"denoiser.sample:1,{latent_ch},{latent_sz},{latent_sz}:{pdir / 'sample.f32'}",
            f"denoiser.encoder_hidden_states:1,{s},{d}:{pdir / 'cond.f32'}",
            f"denoiser.encoder_hidden_states.uncond:1,{s},{d}:{pdir / 'uncond.f32'}",
        ],
        check=True,
    )
    og_latent = np.fromfile(out_path, dtype="<f4").reshape(1, latent_ch, latent_sz, latent_sz)

    diff = np.abs(og_latent - ref_latent)
    print(f"\n=== latent parity (onnx-genai vs diffusers, DDIM+CFG) ===")
    print(f"  max|diff|  = {diff.max():.3e}")
    print(f"  mean|diff| = {diff.mean():.3e}")

    sf = getattr(vae.config, "scaling_factor", 0.18215)
    with torch.no_grad():
        img_ref = vae.decode(torch.from_numpy(ref_latent) / sf).sample[0].numpy()
        img_og = vae.decode(torch.from_numpy(og_latent) / sf).sample[0].numpy()
    save_png(img_ref, WORK / "diffusers.png")
    save_png(img_og, WORK / "onnx_genai.png")
    from PIL import Image

    a, b = Image.open(WORK / "diffusers.png"), Image.open(WORK / "onnx_genai.png")
    combo = Image.new("RGB", (a.width + b.width + 8, max(a.height, b.height)), (255, 255, 255))
    combo.paste(a, (0, 0))
    combo.paste(b, (a.width + 8, 0))
    combo.save(WORK / "compare.png")
    print(f"\nprompt: {args.prompt!r}")
    print(f"saved: {WORK/'diffusers.png'}\n       {WORK/'onnx_genai.png'}\n       {WORK/'compare.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
