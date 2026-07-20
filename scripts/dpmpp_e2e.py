#!/usr/bin/env python3
"""End-to-end validation of onnx-genai's DPM-Solver++ (2M) on a real SD UNet.

Converts a ComfyUI workflow whose sampler is `dpmpp_2m` into a runnable
onnx-genai pipeline (via mobius), runs it, and compares the rendered image to a
diffusers `DPMSolverMultistepScheduler` reference.

Run in the `onnx` conda env after building the release runner:
    conda run -n onnx python scripts/dpmpp_e2e.py
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
WORK = REPO / "target" / "dpmpp-e2e"
RUNNER = REPO / "target" / "release" / "run_diffusion"
MODEL = "OFA-Sys/small-stable-diffusion-v0"
PROMPT = "a photograph of an astronaut riding a horse"
STEPS = 12
CFG = 7.5
SIZE = 256
KARRAS = os.environ.get("ONNX_GENAI_KARRAS", "0") == "1"
SPACING = "karras" if KARRAS else "normal"

WORKFLOW = {
    "3": {"class_type": "KSampler", "inputs": {
        "seed": 0, "steps": STEPS, "cfg": CFG, "sampler_name": "dpmpp_2m", "scheduler": SPACING,
        "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "small-sd.safetensors"}},
    "5": {"class_type": "EmptyLatentImage", "inputs": {"width": SIZE, "height": SIZE, "batch_size": 1}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["4", 1]}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["4", 1]}},
    "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
    "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0]}},
}


def ort_lib_dir() -> str:
    hits = sorted(glob.glob(str(REPO / "target/*/build/onnx-genai-ort-sys-*/out/ort-prebuilt/lib")))
    if not hits:
        raise SystemExit("could not locate prebuilt ORT lib dir")
    return hits[-1]


def save_png(img_chw_m11: np.ndarray, path: Path) -> None:
    from PIL import Image

    img = (img_chw_m11 / 2 + 0.5).clip(0, 1)
    Image.fromarray((img.transpose(1, 2, 0) * 255).round().astype(np.uint8)).save(path)


def main() -> int:
    if not RUNNER.exists():
        print(f"missing {RUNNER}; build: cargo build --release -p onnx-genai --bin run_diffusion", file=sys.stderr)
        return 1
    WORK.mkdir(parents=True, exist_ok=True)
    pdir = WORK / "pipeline"

    from mobius.integrations.onnx_genai import convert_comfyui_workflow

    print("converting ComfyUI (dpmpp_2m) workflow -> onnx-genai pipeline ...", flush=True)
    convert_comfyui_workflow(WORKFLOW, MODEL, str(pdir))

    from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    unet = UNet2DConditionModel.from_pretrained(MODEL, subfolder="unet").eval()
    vae = AutoencoderKL.from_pretrained(MODEL, subfolder="vae").eval()
    text_encoder = CLIPTextModel.from_pretrained(MODEL, subfolder="text_encoder").eval()
    tokenizer = CLIPTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
    sf = float(getattr(vae.config, "scaling_factor", 0.18215))

    sched = DPMSolverMultistepScheduler(
        num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
        algorithm_type="dpmsolver++", solver_order=2, solver_type="midpoint",
        use_karras_sigmas=KARRAS, timestep_spacing="linspace", final_sigmas_type="zero",
        prediction_type="epsilon", lower_order_final=True,
    )
    sched.set_timesteps(STEPS)

    def tok(text: str) -> torch.Tensor:
        return tokenizer(text, padding="max_length", max_length=tokenizer.model_max_length,
                         truncation=True, return_tensors="pt").input_ids

    ids_cond, ids_uncond = tok(PROMPT), tok("")
    with torch.no_grad():
        emb_cond = text_encoder(ids_cond)[0]
        emb_uncond = text_encoder(ids_uncond)[0]

    ch, sz = unet.config.in_channels, SIZE // 8
    g = torch.Generator().manual_seed(0)
    latent0 = torch.randn(1, ch, sz, sz, generator=g) * sched.init_noise_sigma

    lat = latent0.clone()
    with torch.no_grad():
        for t in sched.timesteps:
            scaled = sched.scale_model_input(lat, t)
            nc = unet(scaled, t, encoder_hidden_states=emb_cond).sample
            nu = unet(scaled, t, encoder_hidden_states=emb_uncond).sample
            lat = sched.step(nu + CFG * (nc - nu), t, lat).prev_sample
        img_ref = vae.decode(lat / sf).sample[0].numpy()

    ids_cond.numpy().astype("<i8").tofile(pdir / "ids.i64")
    latent0.numpy().astype("<f4").tofile(pdir / "sample.f32")
    emb_uncond.detach().numpy().astype("<f4").tofile(pdir / "uncond.f32")
    seq = ids_cond.shape[1]
    s, d = emb_cond.shape[1], emb_cond.shape[2]

    env = dict(os.environ)
    env["DYLD_LIBRARY_PATH"] = ort_lib_dir() + ":" + env.get("DYLD_LIBRARY_PATH", "")
    out_path = pdir / "image.f32"
    print("running converted dpmpp_2m pipeline through onnx-genai ...", flush=True)
    subprocess.run(
        [
            str(RUNNER), str(pdir), "vae.image", str(out_path),
            f"text_encoder.input_ids:i64:1,{seq}:{pdir / 'ids.i64'}",
            f"denoiser.sample:1,{ch},{sz},{sz}:{pdir / 'sample.f32'}",
            f"denoiser.encoder_hidden_states.uncond:1,{s},{d}:{pdir / 'uncond.f32'}",
        ],
        env=env, check=True,
    )
    og_img = np.fromfile(out_path, dtype="<f4").reshape(1, 3, SIZE, SIZE)[0]

    diff = np.abs(og_img - img_ref)
    label = "DPM++2M Karras" if KARRAS else "DPM++2M"
    print(f"\n=== onnx-genai {label} (from ComfyUI) vs diffusers DPMSolverMultistep ===")
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
    print(f"\nsaved: {WORK / 'compare.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
