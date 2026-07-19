#!/usr/bin/env python3
"""Generate an image with a real (tiny) Stable Diffusion model two ways and
compare: (A) the diffusers reference denoise loop, and (B) onnx-genai's DDIM
iterative pipeline driving the SAME exported UNet.

Both use an identical DDIM configuration (scaled_linear betas, epsilon,
set_alpha_to_one, no CFG, same seed/prompt/timesteps), so a correct onnx-genai
scheduler reproduces diffusers' final latent. We then decode both final latents
with the same VAE and save side-by-side PNGs plus a difference report.

Run in the `onnx` conda env (torch + diffusers + transformers + onnxruntime),
after building the runner:  cargo build -p onnx-genai --bin run_diffusion

    conda run -n onnx python scripts/diffusion_compare.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

MODEL = "hf-internal-testing/tiny-stable-diffusion-torch"
N_STEPS = 10
SEED = 0
PROMPT = "a photo of a cat"

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / "target" / "diffusion-compare"
RUNNER = REPO / "target" / "debug" / "run_diffusion"


def save_png(img_chw_m11: np.ndarray, path: Path) -> None:
    from PIL import Image

    img = (img_chw_m11 / 2 + 0.5).clip(0, 1)
    img = (img.transpose(1, 2, 0) * 255).round().astype(np.uint8)
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    Image.fromarray(img).save(path)


def main() -> int:
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    if not RUNNER.exists():
        print(f"missing {RUNNER}; run: cargo build -p onnx-genai --bin run_diffusion", file=sys.stderr)
        return 1
    WORK.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)

    unet = UNet2DConditionModel.from_pretrained(MODEL, subfolder="unet").eval()
    vae = AutoencoderKL.from_pretrained(MODEL, subfolder="vae").eval()
    text_encoder = CLIPTextModel.from_pretrained(MODEL, subfolder="text_encoder").eval()
    tokenizer = CLIPTokenizer.from_pretrained(MODEL, subfolder="tokenizer")

    # DDIM configured to exactly match onnx-genai's implementation.
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
        set_alpha_to_one=True,
        steps_offset=0,
        clip_sample=False,
    )
    scheduler.set_timesteps(N_STEPS)
    timesteps = [int(t) for t in scheduler.timesteps]
    print("timesteps:", timesteps)

    # Conditioning (no classifier-free guidance).
    ids = tokenizer(
        PROMPT, padding="max_length", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    ).input_ids
    with torch.no_grad():
        emb = text_encoder(ids)[0]  # [1, 77, cross_attention_dim]

    ch = unet.config.in_channels
    sz = unet.config.sample_size
    g = torch.Generator().manual_seed(SEED)
    latent0 = torch.randn(1, ch, sz, sz, generator=g)

    # (A) diffusers reference denoise loop.
    lat = latent0.clone()
    with torch.no_grad():
        for t in scheduler.timesteps:
            noise = unet(lat, t, encoder_hidden_states=emb).sample
            lat = scheduler.step(noise, t, lat).prev_sample
    ref_latent = lat.numpy()

    # Export the UNet to ONNX (sample, int64 timestep[1], encoder_hidden_states -> noise_pred).
    class Wrap(torch.nn.Module):
        def __init__(self, u):
            super().__init__()
            self.u = u

        def forward(self, sample, timestep, encoder_hidden_states):
            return self.u(sample, timestep, encoder_hidden_states=encoder_hidden_states).sample

    pdir = WORK / "sd-pipeline"
    pdir.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        Wrap(unet),
        (latent0, torch.tensor([timesteps[0]], dtype=torch.long), emb),
        str(pdir / "unet.onnx"),
        input_names=["sample", "timestep", "encoder_hidden_states"],
        output_names=["noise_pred"],
        opset_version=17,
        dynamo=False,
    )

    # onnx-genai pipeline metadata: iterative DDIM with the exact diffusers timesteps.
    ts_yaml = "".join(f"      - {t}.0\n" for t in timesteps)
    (pdir / "inference_metadata.yaml").write_text(
        "pipeline:\n"
        "  models:\n"
        "    denoiser:\n"
        "      filename: unet.onnx\n"
        "      type: denoiser\n"
        "  dataflow:\n"
        "    - from: denoiser.noise_pred\n"
        "      to: denoiser.sample\n"
        "  strategy:\n"
        "    kind: iterative\n"
        "    denoiser: denoiser\n"
        f"    num_steps: {N_STEPS}\n"
        "    timestep_input: timestep\n"
        "    timesteps:\n" + ts_yaml +
        "    scheduler_config:\n"
        "      kind: ddim\n"
        "      num_train_timesteps: 1000\n"
        "      beta_start: 0.00085\n"
        "      beta_end: 0.012\n"
        "      beta_schedule: scaled_linear\n"
        "      prediction_type: epsilon\n"
    )

    # Write raw-f32 inputs for the runner.
    latent0.numpy().astype("<f4").tofile(pdir / "sample.f32")
    emb.detach().numpy().astype("<f4").tofile(pdir / "ehs.f32")

    # (B) Run onnx-genai's DDIM iterative pipeline.
    out_path = pdir / "onnxgenai_latent.f32"
    subprocess.run(
        [
            str(RUNNER), str(pdir), "denoiser.sample", str(out_path),
            f"denoiser.sample:1,{ch},{sz},{sz}:{pdir / 'sample.f32'}",
            f"denoiser.encoder_hidden_states:1,{emb.shape[1]},{emb.shape[2]}:{pdir / 'ehs.f32'}",
        ],
        check=True,
    )
    og_latent = np.fromfile(out_path, dtype="<f4").reshape(1, ch, sz, sz)

    # Compare final latents.
    diff = np.abs(og_latent - ref_latent)
    denom = np.abs(ref_latent).mean() + 1e-8
    print("\n=== latent comparison (onnx-genai vs diffusers) ===")
    print(f"  max|diff|   = {diff.max():.3e}")
    print(f"  mean|diff|  = {diff.mean():.3e}")
    print(f"  rel mean    = {diff.mean() / denom:.3e}")

    # Decode both latents with the same VAE and save images.
    sf = getattr(vae.config, "scaling_factor", 0.18215)
    with torch.no_grad():
        img_ref = vae.decode(torch.from_numpy(ref_latent) / sf).sample[0].numpy()
        img_og = vae.decode(torch.from_numpy(og_latent) / sf).sample[0].numpy()
    save_png(img_ref, WORK / "diffusers.png")
    save_png(img_og, WORK / "onnx_genai.png")
    # Side-by-side.
    from PIL import Image

    a = Image.open(WORK / "diffusers.png")
    b = Image.open(WORK / "onnx_genai.png")
    combo = Image.new("RGB", (a.width + b.width + 8, max(a.height, b.height)), (255, 255, 255))
    combo.paste(a, (0, 0))
    combo.paste(b, (a.width + 8, 0))
    combo.save(WORK / "compare.png")
    print(f"\nsaved: {WORK/'diffusers.png'}\n       {WORK/'onnx_genai.png'}\n       {WORK/'compare.png'}")
    img_diff = np.abs(img_og - img_ref)
    print(f"image max|diff| = {img_diff.max():.3e}, mean|diff| = {img_diff.mean():.3e} (pixel range ~[-1,1])")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
