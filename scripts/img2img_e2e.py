#!/usr/bin/env python3
"""End-to-end img2img validation: onnx-genai partial denoise loop vs diffusers.

A ComfyUI KSampler `denoise` < 1.0 is img2img: encode a source image to a latent,
noise it to an intermediate step, and run only the tail of the denoise loop. This
converts such a workflow (start_step in the metadata), feeds onnx-genai the same
noised latent diffusers uses, runs the partial loop, and compares.

Uses DPM++ 2M (deterministic) so the comparison is exact.

Run (conda `onnx`, after `cargo build --release -p onnx-genai --bin run_diffusion`):
    conda run -n onnx python scripts/img2img_e2e.py
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / "target" / "img2img-e2e"
RUNNER = REPO / "target" / "release" / "run_diffusion"
MODEL = "OFA-Sys/small-stable-diffusion-v0"
PROMPT = "a photograph of an astronaut riding a horse"
NEG = "blurry"
STEPS = 12
CFG = 7.5
SIZE = 256
DENOISE = 0.6


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
            "seed": 0, "steps": STEPS, "cfg": CFG, "sampler_name": "dpmpp_2m", "scheduler": "normal",
            "denoise": DENOISE, "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0],
            "latent_image": ["5", 0]}},
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd.safetensors"}},
        # a VAEEncode/LoadImage would appear here in a real img2img graph; the
        # latent shape still comes through and the driver supplies the encoded image.
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": SIZE, "height": SIZE}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["4", 1]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
    }

    from mobius.integrations.onnx_genai import convert_comfyui_workflow

    print("converting img2img workflow -> onnx-genai pipeline ...", flush=True)
    result = convert_comfyui_workflow(workflow, MODEL, str(pdir))
    wf = result.workflow
    start_step = wf.start_step
    print(f"  denoise={DENOISE} -> start_step={start_step}/{STEPS}")
    assert start_step > 0, "expected a partial loop"

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
        use_karras_sigmas=False, timestep_spacing="linspace", final_sigmas_type="zero",
        prediction_type="epsilon", lower_order_final=True,
    )
    sched.set_timesteps(STEPS)

    def tok(t: str) -> torch.Tensor:
        return tokenizer(t, padding="max_length", max_length=tokenizer.model_max_length,
                         truncation=True, return_tensors="pt").input_ids

    ids_pos, ids_neg = tok(PROMPT), tok(NEG)
    with torch.no_grad():
        emb_pos = text_encoder(ids_pos)[0]
        emb_neg = text_encoder(ids_neg)[0]

    # Deterministic "source image" -> encode -> noise to the start step.
    ch, sz = unet.config.in_channels, SIZE // 8
    g = torch.Generator().manual_seed(1)
    src = torch.rand(1, 3, SIZE, SIZE, generator=g) * 2 - 1  # [-1,1]
    with torch.no_grad():
        encoded = vae.encode(src).latent_dist.mean * sf
    noise = torch.randn(1, ch, sz, sz, generator=torch.Generator().manual_seed(0))
    sched.set_begin_index(start_step)
    t_start = sched.timesteps[start_step:start_step + 1]
    latent_start = sched.add_noise(encoded, noise, t_start)

    # diffusers img2img reference: run the tail of the loop.
    lat = latent_start.clone()
    with torch.no_grad():
        for t in sched.timesteps[start_step:]:
            si = sched.scale_model_input(lat, t)
            nc = unet(si, t, encoder_hidden_states=emb_pos).sample
            nu = unet(si, t, encoder_hidden_states=emb_neg).sample
            lat = sched.step(nu + CFG * (nc - nu), t, lat).prev_sample
        img_ref = vae.decode(lat / sf).sample[0].numpy()

    # onnx-genai: feed the SAME noised latent as the seed and run the partial loop.
    ids_pos.numpy().astype("<i8").tofile(pdir / "ids.i64")
    latent_start.detach().numpy().astype("<f4").tofile(pdir / "sample.f32")
    emb_neg.detach().numpy().astype("<f4").tofile(pdir / "uncond.f32")
    seq = ids_pos.shape[1]
    s, d = emb_pos.shape[1], emb_pos.shape[2]

    env = dict(os.environ)
    env["DYLD_LIBRARY_PATH"] = ort_lib_dir() + ":" + env.get("DYLD_LIBRARY_PATH", "")
    out_path = pdir / "image.f32"
    print("running img2img partial loop through onnx-genai ...", flush=True)
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
    print("\n=== onnx-genai img2img (start_step partial loop) vs diffusers ===")
    print(f"  max|diff|  = {diff.max():.3e}")
    print(f"  mean|diff| = {diff.mean():.3e}  (pixel range ~[-1,1])")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
