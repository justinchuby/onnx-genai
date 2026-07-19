#!/usr/bin/env python3
"""Run a ComfyUI workflow through onnx-genai and save the image.

The end-to-end tool: convert a ComfyUI API-format workflow + checkpoint into a
runnable onnx-genai pipeline (via mobius), prepare the runtime inputs — including
the **negative-prompt** unconditional embedding (real ComfyUI workflows carry a
negative prompt, not an empty one) — run the pipeline, and save the PNG.

Unlike the *_e2e.py validators, this is a usable driver: it renders from any
supported ComfyUI txt2img workflow. Pass --compare to also diff against a
diffusers reference (with the same negative prompt).

Usage (conda `onnx` env, after `cargo build --release -p onnx-genai --bin run_diffusion`):
    conda run -n onnx python scripts/run_comfyui.py \
        --workflow workflow.json --checkpoint OFA-Sys/small-stable-diffusion-v0 \
        --output out.png [--compare]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "target" / "release" / "run_diffusion"


def ort_lib_dir() -> str:
    hits = sorted(glob.glob(str(REPO / "target/*/build/onnx-genai-ort-sys-*/out/ort-prebuilt/lib")))
    if not hits:
        raise SystemExit("could not locate prebuilt ORT lib dir")
    return hits[-1]


def save_png(img_chw_m11: np.ndarray, path: Path) -> None:
    from PIL import Image

    img = (img_chw_m11 / 2 + 0.5).clip(0, 1)
    Image.fromarray((img.transpose(1, 2, 0) * 255).round().astype(np.uint8)).save(path)


def _diffusers_init_noise_sigma(kind: str, sc: dict, steps: int) -> float:
    """init_noise_sigma for the scheduler (euler scales the seed; ddim/dpm do not)."""
    if kind == "ddim" or kind == "dpmpp_2m":
        return 1.0
    from diffusers import EulerDiscreteScheduler

    sched = EulerDiscreteScheduler(
        num_train_timesteps=sc.get("num_train_timesteps", 1000),
        beta_start=sc.get("beta_start", 0.00085), beta_end=sc.get("beta_end", 0.012),
        beta_schedule=sc.get("beta_schedule", "scaled_linear"),
        use_karras_sigmas=sc.get("use_karras_sigmas", False),
        timestep_spacing="linspace", interpolation_type="linear",
    )
    sched.set_timesteps(steps)
    return float(sched.init_noise_sigma)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workflow", required=True, help="ComfyUI API-format workflow JSON.")
    ap.add_argument("--checkpoint", required=True, help=".safetensors/.ckpt, diffusers dir, or HF id.")
    ap.add_argument("--output", "-o", default="comfyui_out.png", help="Output PNG path.")
    ap.add_argument("--workdir", default=str(REPO / "target" / "run-comfyui"))
    ap.add_argument("--compare", action="store_true", help="Also diff against a diffusers reference.")
    args = ap.parse_args()

    if not RUNNER.exists():
        print(f"missing {RUNNER}; build: cargo build --release -p onnx-genai --bin run_diffusion", file=sys.stderr)
        return 1

    import onnxruntime as ort
    from transformers import CLIPTokenizer

    from mobius.integrations.onnx_genai import convert_comfyui_workflow

    with open(args.workflow, encoding="utf-8") as handle:
        workflow = json.load(handle)

    pdir = Path(args.workdir) / "pipeline"
    print("converting ComfyUI workflow -> onnx-genai pipeline ...", flush=True)
    result = convert_comfyui_workflow(workflow, args.checkpoint, str(pdir))
    wf = result.workflow
    meta = json.loads(json.dumps(__import__("yaml").safe_load(open(result.metadata_path))))
    sc = meta["pipeline"]["strategy"]["scheduler_config"]
    prompt = wf.prompt or ""
    negative = wf.negative_prompt or ""
    print(f"  prompt={prompt!r}  negative={negative!r}  {wf.steps} steps, cfg {wf.cfg}, "
          f"{wf.sampler_name} ({wf.scheduler_kind}{'/karras' if wf.scheduler_spacing == 'karras' else ''})")

    tokenizer = CLIPTokenizer.from_pretrained(args.checkpoint, subfolder="tokenizer")

    def tok(text: str) -> np.ndarray:
        return tokenizer(text, padding="max_length", max_length=tokenizer.model_max_length,
                         truncation=True, return_tensors="np").input_ids.astype(np.int64)

    ids_pos = tok(prompt)
    ids_neg = tok(negative)

    # The pipeline runs the text encoder on the positive prompt; compute the
    # NEGATIVE-prompt unconditional embedding here from the same exported encoder.
    te = ort.InferenceSession(str(pdir / "text_encoder.onnx"), providers=["CPUExecutionProvider"])
    te_in = te.get_inputs()[0].name
    emb_neg = te.run(None, {te_in: ids_neg})[0].astype(np.float32)

    ch = wf.metadata["pipeline"]["models"]["denoiser"]
    latent_ch = json.load(open(result.run_params_path))["latent_channels"]
    sz = wf.height // 8
    init_sigma = _diffusers_init_noise_sigma(wf.scheduler_kind, sc, wf.steps)
    rng = np.random.default_rng(wf.seed)
    latent0 = (rng.standard_normal((1, latent_ch, sz, sz)).astype(np.float32)) * init_sigma

    ids_pos.tofile(pdir / "ids.i64")
    latent0.tofile(pdir / "sample.f32")
    emb_neg.tofile(pdir / "uncond.f32")
    seq = ids_pos.shape[1]
    s, d = emb_neg.shape[1], emb_neg.shape[2]

    env = dict(os.environ)
    env["DYLD_LIBRARY_PATH"] = ort_lib_dir() + ":" + env.get("DYLD_LIBRARY_PATH", "")
    out_path = pdir / "image.f32"
    print("rendering through onnx-genai ...", flush=True)
    subprocess.run(
        [
            str(RUNNER), str(pdir), "vae.image", str(out_path),
            f"text_encoder.input_ids:i64:1,{seq}:{pdir / 'ids.i64'}",
            f"denoiser.sample:1,{latent_ch},{sz},{sz}:{pdir / 'sample.f32'}",
            f"denoiser.encoder_hidden_states.uncond:1,{s},{d}:{pdir / 'uncond.f32'}",
        ],
        env=env, check=True,
    )
    img = np.fromfile(out_path, dtype="<f4").reshape(1, 3, wf.height, wf.width)[0]
    save_png(img, Path(args.output))
    print(f"saved: {args.output}")

    if args.compare:
        _compare_diffusers(args.checkpoint, wf, sc, latent0, ids_pos, ids_neg, img)
    return 0


def _compare_diffusers(checkpoint, wf, sc, latent0, ids_pos, ids_neg, og_img):
    import torch
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextModel

    unet = UNet2DConditionModel.from_pretrained(checkpoint, subfolder="unet").eval()
    vae = AutoencoderKL.from_pretrained(checkpoint, subfolder="vae").eval()
    text_encoder = CLIPTextModel.from_pretrained(checkpoint, subfolder="text_encoder").eval()
    sf = float(getattr(vae.config, "scaling_factor", 0.18215))
    kind = wf.scheduler_kind
    if kind == "euler":
        from diffusers import EulerDiscreteScheduler as S
        extra = {"timestep_spacing": "linspace", "interpolation_type": "linear",
                 "use_karras_sigmas": sc.get("use_karras_sigmas", False)}
    elif kind == "dpmpp_2m":
        from diffusers import DPMSolverMultistepScheduler as S
        extra = {"algorithm_type": "dpmsolver++", "solver_order": 2, "solver_type": "midpoint",
                 "use_karras_sigmas": sc.get("use_karras_sigmas", False),
                 "timestep_spacing": "linspace", "final_sigmas_type": "zero"}
    else:
        from diffusers import DDIMScheduler as S
        extra = {"set_alpha_to_one": True, "steps_offset": 0, "clip_sample": False}
    sched = S(num_train_timesteps=sc["num_train_timesteps"], beta_start=sc["beta_start"],
              beta_end=sc["beta_end"], beta_schedule=sc["beta_schedule"], prediction_type="epsilon", **extra)
    sched.set_timesteps(wf.steps)
    with torch.no_grad():
        ec = text_encoder(torch.from_numpy(ids_pos))[0]
        eu = text_encoder(torch.from_numpy(ids_neg))[0]
        lat = torch.from_numpy(latent0.copy())
        for t in sched.timesteps:
            si = sched.scale_model_input(lat, t)
            nc = unet(si, t, encoder_hidden_states=ec).sample
            nu = unet(si, t, encoder_hidden_states=eu).sample
            lat = sched.step(nu + wf.cfg * (nc - nu), t, lat).prev_sample
        ref = vae.decode(lat / sf).sample[0].numpy()
    diff = np.abs(og_img - ref)
    print(f"  [compare] onnx-genai vs diffusers (same negative prompt): "
          f"max|diff|={diff.max():.3e} mean|diff|={diff.mean():.3e}")


if __name__ == "__main__":
    raise SystemExit(main())
