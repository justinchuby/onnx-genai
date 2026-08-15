#!/usr/bin/env python3
"""Benchmark the diffusion denoise loop: diffusers (torch) vs onnx-genai.

Exports a real Stable Diffusion UNet to ONNX once, then times an identical
DDIM+CFG denoise loop (2 UNet passes/step) four ways:

  (A) diffusers UNet on torch CPU
  (B) diffusers UNet on torch MPS (Apple GPU), if available
  (C) onnx-genai run_diffusion on the CPU execution provider
  (D) onnx-genai run_diffusion on the MLX execution provider (Apple GPU)

onnx-genai timing comes from the runner's own "[timing] run=" line (model load
excluded). For (D) set the plugin path via --mlx-lib (or ONNX_GENAI_METAL_EP_LIB).

Run in the `onnx` conda env after building the release runner:
    cargo build --release -p onnx-genai --bin run_diffusion
    conda run -n onnx python scripts/diffusion_bench.py --steps 20 --size 256
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / "target" / "diffusion-bench"
RUNNER = REPO / "target" / "release" / "run_diffusion"
DEFAULT_MLX = (
    "/Users/justinc/Documents/GitHub/onnxruntime-mlx/python/src/"
    "onnxruntime_ep_mlx/libonnxruntime_mlx_ep.dylib"
)


def ort_lib_dir() -> str:
    import glob

    hits = sorted(
        glob.glob(
            str(REPO / "target" / "*" / "build" / "onnx-genai-ort-sys-*" / "out" / "ort-prebuilt" / "lib")
        )
    )
    if not hits:
        raise SystemExit("could not locate prebuilt ORT lib dir")
    return hits[-1]


def time_torch_loop(unet, latent0, cond, uncond, timesteps, scheduler, guidance, device, iters):
    unet = unet.to(device)
    lat0 = latent0.to(device)
    cond_d = cond.to(device)
    uncond_d = uncond.to(device)
    # warm up (JIT/graph/caches)
    with torch.no_grad():
        t0 = timesteps[0]
        _ = unet(lat0, t0, encoder_hidden_states=cond_d).sample
        if device == "mps":
            torch.mps.synchronize()
    best = float("inf")
    for _ in range(iters):
        lat = lat0.clone()
        start = time.perf_counter()
        with torch.no_grad():
            for t in scheduler.timesteps:
                nc = unet(lat, t, encoder_hidden_states=cond_d).sample
                nu = unet(lat, t, encoder_hidden_states=uncond_d).sample
                noise = nu + guidance * (nc - nu)
                lat = scheduler.step(noise, t, lat).prev_sample
            if device == "mps":
                torch.mps.synchronize()
        best = min(best, time.perf_counter() - start)
    unet.to("cpu")
    return best * 1e3


def time_onnx_genai(pdir, shapes, ep_env, iters):
    latent_ch, latent_sz, s, d = shapes
    env = dict(os.environ)
    env["DYLD_LIBRARY_PATH"] = ort_lib_dir() + ":" + env.get("DYLD_LIBRARY_PATH", "")
    env.update(ep_env)
    cmd = [
        str(RUNNER), str(pdir), "denoiser.sample", str(pdir / "og_latent.f32"),
        f"denoiser.sample:1,{latent_ch},{latent_sz},{latent_sz}:{pdir / 'sample.f32'}",
        f"denoiser.encoder_hidden_states:1,{s},{d}:{pdir / 'cond.f32'}",
        f"denoiser.encoder_hidden_states.uncond:1,{s},{d}:{pdir / 'uncond.f32'}",
    ]
    best = float("inf")
    last_err = ""
    for _ in range(iters):
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            last_err = proc.stderr
            continue
        m = re.search(r"\[timing\] load=([\d.]+)ms run=([\d.]+)ms", proc.stderr)
        if m:
            best = min(best, float(m.group(2)))
    if best == float("inf"):
        return None, last_err
    return best, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="OFA-Sys/small-stable-diffusion-v0")
    ap.add_argument("--prompt", default="a photograph of an astronaut riding a horse")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--mlx-lib", default=os.environ.get("ONNX_GENAI_METAL_EP_LIB", DEFAULT_MLX))
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()

    from diffusers import DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    if not RUNNER.exists():
        print(f"missing {RUNNER}; run: cargo build --release -p onnx-genai --bin run_diffusion", file=sys.stderr)
        return 1
    WORK.mkdir(parents=True, exist_ok=True)
    pdir = WORK / "sd-pipeline"
    pdir.mkdir(parents=True, exist_ok=True)

    unet = UNet2DConditionModel.from_pretrained(args.model, subfolder="unet").eval()
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
    latent0 = torch.randn(1, latent_ch, latent_sz, latent_sz, generator=g) * scheduler.init_noise_sigma
    s, d = cond.shape[1], cond.shape[2]
    shapes = (latent_ch, latent_sz, s, d)

    if not args.skip_export:
        class Wrap(torch.nn.Module):
            def __init__(self, u):
                super().__init__()
                self.u = u

            def forward(self, sample, timestep, encoder_hidden_states):
                return self.u(sample, timestep, encoder_hidden_states=encoder_hidden_states).sample

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

    passes = 2 * args.steps
    print(f"\nmodel={args.model}  size={args.size}px (latent {latent_ch}x{latent_sz}x{latent_sz})")
    print(f"steps={args.steps}  guidance={args.guidance}  UNet passes/loop={passes}  iters={args.iters}")
    print(f"MLX plugin: {args.mlx_lib}\n")

    results = {}

    # (A) diffusers CPU
    results["diffusers torch-CPU"] = time_torch_loop(
        unet, latent0, cond, uncond, timesteps, scheduler, args.guidance, "cpu", args.iters
    )
    # (B) diffusers MPS
    if torch.backends.mps.is_available():
        try:
            results["diffusers torch-MPS"] = time_torch_loop(
                unet, latent0, cond, uncond, timesteps, scheduler, args.guidance, "mps", args.iters
            )
        except Exception as e:  # noqa: BLE001
            print(f"  (MPS failed: {e})")
    else:
        print("  (torch MPS unavailable)")

    # (C) onnx-genai CPU EP
    t, err = time_onnx_genai(pdir, shapes, {"ONNX_GENAI_EP": "cpu"}, args.iters)
    if t is not None:
        results["onnx-genai CPU-EP"] = t
    else:
        print(f"  (onnx-genai CPU EP failed)\n{err[-800:]}")

    # (D) onnx-genai MLX EP
    mlx_env = {"ONNX_GENAI_EP": "metal", "ONNX_GENAI_METAL_EP_LIB": args.mlx_lib}
    t, err = time_onnx_genai(pdir, shapes, mlx_env, args.iters)
    if t is not None:
        results["onnx-genai MLX-EP"] = t
    else:
        print(f"  (onnx-genai MLX EP failed)\n{err[-1500:]}")

    print("\n=== denoise loop wall time (best of {}) ===".format(args.iters))
    baseline = results.get("diffusers torch-MPS") or results.get("diffusers torch-CPU")
    width = max(len(k) for k in results)
    for name, ms in sorted(results.items(), key=lambda kv: kv[1]):
        per = ms / passes
        rel = f"{baseline / ms:.2f}x vs diffusers" if baseline else ""
        print(f"  {name:<{width}}  {ms:8.1f} ms total   {per:6.1f} ms/pass   {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
