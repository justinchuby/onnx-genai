#!/usr/bin/env python3
"""End-to-end validation of onnx-genai's Euler Ancestral (stochastic) sampler.

Euler-a injects fresh Gaussian noise each step, so it can only match diffusers
when both consume the *same* noise sequence. We draw the per-step noise with a
seeded generator, feed it to onnx-genai as `denoiser.sample.noise`
([num_steps, C, H, W]), and drive the diffusers reference with a generator of the
same seed (which draws the identical sequence, one randn per step).

Run (conda `onnx`, after `cargo build --release -p onnx-genai --bin run_diffusion`):
    conda run -n onnx python scripts/euler_a_e2e.py
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
WORK = REPO / "target" / "euler-a-e2e"
RUNNER = REPO / "target" / "release" / "run_diffusion"
MODEL = "OFA-Sys/small-stable-diffusion-v0"
PROMPT = "a photograph of an astronaut riding a horse"
NEG = "blurry"
STEPS = 12
CFG = 7.5
SIZE = 256


def ort_lib_dir() -> str:
    hits = sorted(glob.glob(str(REPO / "target/*/build/onnx-genai-ort-sys-*/out/ort-prebuilt/lib")))
    if not hits:
        raise SystemExit("could not locate prebuilt ORT lib dir")
    return hits[-1]


def main() -> int:
    if not RUNNER.exists():
        print(f"missing {RUNNER}", file=sys.stderr)
        return 1
    WORK.mkdir(parents=True, exist_ok=True)
    pdir = WORK / "pipeline"

    workflow = {
        "3": {"class_type": "KSampler", "inputs": {
            "seed": 0, "steps": STEPS, "cfg": CFG, "sampler_name": "euler_ancestral", "scheduler": "normal",
            "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd.safetensors"}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": SIZE, "height": SIZE}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["4", 1]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
    }

    from mobius.integrations.onnx_genai import convert_comfyui_workflow

    print("converting euler_ancestral workflow -> onnx-genai pipeline ...", flush=True)
    convert_comfyui_workflow(workflow, MODEL, str(pdir))

    from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    unet = UNet2DConditionModel.from_pretrained(MODEL, subfolder="unet").eval()
    vae = AutoencoderKL.from_pretrained(MODEL, subfolder="vae").eval()
    text_encoder = CLIPTextModel.from_pretrained(MODEL, subfolder="text_encoder").eval()
    tokenizer = CLIPTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
    sf = float(getattr(vae.config, "scaling_factor", 0.18215))

    sched = EulerAncestralDiscreteScheduler(
        num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear", timestep_spacing="linspace", prediction_type="epsilon",
    )
    sched.set_timesteps(STEPS)

    def tok(t: str) -> torch.Tensor:
        return tokenizer(t, padding="max_length", max_length=tokenizer.model_max_length,
                         truncation=True, return_tensors="pt").input_ids

    ids_pos, ids_neg = tok(PROMPT), tok(NEG)
    with torch.no_grad():
        emb_pos = text_encoder(ids_pos)[0]
        emb_neg = text_encoder(ids_neg)[0]

    ch, sz = unet.config.in_channels, SIZE // 8
    latent0 = torch.randn(1, ch, sz, sz, generator=torch.Generator().manual_seed(0)) * sched.init_noise_sigma

    # Per-step noise sequence, drawn with a seeded generator (one randn per step).
    ngen = torch.Generator().manual_seed(1)
    noise_seq = torch.stack([torch.randn(1, ch, sz, sz, generator=ngen) for _ in range(STEPS)])

    # diffusers reference: a generator of the same seed draws the identical sequence.
    lat = latent0.clone()
    ref_gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for t in sched.timesteps:
            si = sched.scale_model_input(lat, t)
            nc = unet(si, t, encoder_hidden_states=emb_pos).sample
            nu = unet(si, t, encoder_hidden_states=emb_neg).sample
            lat = sched.step(nu + CFG * (nc - nu), t, lat, generator=ref_gen).prev_sample
        img_ref = vae.decode(lat / sf).sample[0].numpy()

    ids_pos.numpy().astype("<i8").tofile(pdir / "ids.i64")
    latent0.numpy().astype("<f4").tofile(pdir / "sample.f32")
    emb_neg.detach().numpy().astype("<f4").tofile(pdir / "uncond.f32")
    noise_seq.numpy().astype("<f4").tofile(pdir / "noise.f32")
    seq = ids_pos.shape[1]
    s, d = emb_pos.shape[1], emb_pos.shape[2]

    env = dict(os.environ)
    env["DYLD_LIBRARY_PATH"] = ort_lib_dir() + ":" + env.get("DYLD_LIBRARY_PATH", "")
    out_path = pdir / "image.f32"
    print("running euler_ancestral through onnx-genai (matched noise) ...", flush=True)
    subprocess.run(
        [
            str(RUNNER), str(pdir), "vae.image", str(out_path),
            f"text_encoder.input_ids:i64:1,{seq}:{pdir / 'ids.i64'}",
            f"denoiser.sample:1,{ch},{sz},{sz}:{pdir / 'sample.f32'}",
            f"denoiser.encoder_hidden_states.uncond:1,{s},{d}:{pdir / 'uncond.f32'}",
            f"denoiser.sample.noise:{STEPS},{ch},{sz},{sz}:{pdir / 'noise.f32'}",
        ],
        env=env, check=True,
    )
    og_img = np.fromfile(out_path, dtype="<f4").reshape(1, 3, SIZE, SIZE)[0]

    diff = np.abs(og_img - img_ref)
    print("\n=== onnx-genai Euler Ancestral (matched noise) vs diffusers ===")
    print(f"  max|diff|  = {diff.max():.3e}")
    print(f"  mean|diff| = {diff.mean():.3e}  (pixel range ~[-1,1])")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
