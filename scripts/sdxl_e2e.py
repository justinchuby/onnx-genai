#!/usr/bin/env python3
"""End-to-end SDXL validation: onnx-genai vs diffusers on a real dual-conditioning UNet.

SDXL differs from SD 1.x: two text encoders (concatenated penultimate hidden states
-> encoder_hidden_states) plus a pooled text_embeds and a time_ids conditioning
vector fed to the UNet. This exercises onnx-genai's multi-input classifier-free
guidance (the uncond pass overrides BOTH encoder_hidden_states and text_embeds,
sharing time_ids) end to end.

Exports the three components, drives them through onnx-genai, and compares to a
diffusers reference denoise loop with the same components. Uses tiny random SDXL
weights (parity is what matters, not the image).

Run (conda `onnx`, after `cargo build --release -p onnx-genai --bin run_diffusion`):
    conda run -n onnx python scripts/sdxl_e2e.py
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
WORK = REPO / "target" / "sdxl-e2e"
RUNNER = REPO / "target" / "release" / "run_diffusion"
MODEL = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"
PROMPT = "a photograph of an astronaut riding a horse"
NEG = "blurry, low quality"
STEPS = 10
CFG = 7.5
SIZE = 64  # tiny SDXL sample_size is 32 -> 256px; use 64px (latent 8) for speed


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
    pdir.mkdir(parents=True, exist_ok=True)

    from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline.from_pretrained(MODEL)
    unet, vae = pipe.unet.eval(), pipe.vae.eval()
    te1, te2 = pipe.text_encoder.eval(), pipe.text_encoder_2.eval()
    tok1, tok2 = pipe.tokenizer, pipe.tokenizer_2
    sf = float(getattr(vae.config, "scaling_factor", 0.13025))
    ch, sz = unet.config.in_channels, SIZE // 8

    sched = DPMSolverMultistepScheduler(
        num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
        algorithm_type="dpmsolver++", solver_order=2, solver_type="midpoint",
        use_karras_sigmas=False, timestep_spacing="linspace", final_sigmas_type="zero",
        prediction_type="epsilon", lower_order_final=True,
    )
    sched.set_timesteps(STEPS)
    timesteps = [float(t) for t in sched.timesteps]

    def encode(prompt: str):
        ids1 = tok1(prompt, padding="max_length", max_length=tok1.model_max_length,
                    truncation=True, return_tensors="pt").input_ids
        ids2 = tok2(prompt, padding="max_length", max_length=tok2.model_max_length,
                    truncation=True, return_tensors="pt").input_ids
        with torch.no_grad():
            o1 = te1(ids1, output_hidden_states=True)
            o2 = te2(ids2, output_hidden_states=True)
        h = torch.cat([o1.hidden_states[-2], o2.hidden_states[-2]], dim=-1)  # [B,S,64]
        pooled = o2[0]  # [B,32]
        return ids1, ids2, h, pooled

    ids1_p, ids2_p, enc_pos, pooled_pos = encode(PROMPT)
    _, _, enc_neg, pooled_neg = encode(NEG)
    time_ids = torch.tensor([[SIZE, SIZE, 0, 0, SIZE, SIZE]], dtype=torch.float32)

    latent0 = torch.randn(1, ch, sz, sz, generator=torch.Generator().manual_seed(0)) * sched.init_noise_sigma

    # --- diffusers reference denoise loop ---
    lat = latent0.clone()
    with torch.no_grad():
        for t in sched.timesteps:
            si = sched.scale_model_input(lat, t)
            add_p = {"text_embeds": pooled_pos, "time_ids": time_ids}
            add_n = {"text_embeds": pooled_neg, "time_ids": time_ids}
            nc = unet(si, t, encoder_hidden_states=enc_pos, added_cond_kwargs=add_p).sample
            nu = unet(si, t, encoder_hidden_states=enc_neg, added_cond_kwargs=add_n).sample
            lat = sched.step(nu + CFG * (nc - nu), t, lat).prev_sample
        img_ref = vae.decode(lat / sf).sample[0].numpy()

    # --- export the three ONNX components ---
    class TextWrap(torch.nn.Module):
        def __init__(self, a, b):
            super().__init__()
            self.a, self.b = a, b

        def forward(self, input_ids, input_ids_2):
            o1 = self.a(input_ids, output_hidden_states=True)
            o2 = self.b(input_ids_2, output_hidden_states=True)
            return torch.cat([o1.hidden_states[-2], o2.hidden_states[-2]], dim=-1), o2[0]

    class UNetWrap(torch.nn.Module):
        def __init__(self, u):
            super().__init__()
            self.u = u

        def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
            added = {"text_embeds": text_embeds, "time_ids": time_ids}
            return self.u(sample, timestep, encoder_hidden_states=encoder_hidden_states,
                          added_cond_kwargs=added).sample

    class VaeWrap(torch.nn.Module):
        def __init__(self, v, scale):
            super().__init__()
            self.v, self.scale = v, scale

        def forward(self, latent):
            return self.v.decode(latent / self.scale).sample

    print("exporting SDXL text_encoder / denoiser / vae ...", flush=True)
    torch.onnx.export(
        TextWrap(te1, te2), (ids1_p, ids2_p), str(pdir / "text_encoder.onnx"),
        input_names=["input_ids", "input_ids_2"], output_names=["encoder_hidden_states", "text_embeds"],
        dynamic_axes={"input_ids": {0: "b", 1: "s"}, "input_ids_2": {0: "b", 1: "s"}},
        opset_version=17, dynamo=False,
    )
    torch.onnx.export(
        UNetWrap(unet),
        (latent0, torch.tensor([int(timesteps[0])], dtype=torch.long), enc_pos, pooled_pos, time_ids),
        str(pdir / "denoiser.onnx"),
        input_names=["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"],
        output_names=["noise_pred"],
        dynamic_axes={"sample": {0: "b", 2: "h", 3: "w"}, "encoder_hidden_states": {0: "b", 1: "s"}},
        opset_version=17, dynamo=False,
    )
    torch.onnx.export(
        VaeWrap(vae, sf), (latent0,), str(pdir / "vae.onnx"),
        input_names=["latent"], output_names=["image"],
        dynamic_axes={"latent": {0: "b", 2: "h", 3: "w"}}, opset_version=17, dynamo=False,
    )

    # --- SDXL pipeline metadata ---
    ts_yaml = "".join(f"      - {t}\n" for t in timesteps)
    (pdir / "inference_metadata.yaml").write_text(
        "pipeline:\n  models:\n"
        "    text_encoder:\n      filename: text_encoder.onnx\n      type: encoder\n"
        "    denoiser:\n      filename: denoiser.onnx\n      type: denoiser\n"
        "    vae:\n      filename: vae.onnx\n      type: vae\n"
        "  dataflow:\n"
        "    - from: text_encoder.encoder_hidden_states\n      to: denoiser.encoder_hidden_states\n"
        "    - from: text_encoder.text_embeds\n      to: denoiser.text_embeds\n"
        "    - from: denoiser.noise_pred\n      to: denoiser.sample\n"
        "    - from: denoiser.sample\n      to: vae.latent\n"
        "  strategy:\n    kind: iterative\n    denoiser: denoiser\n"
        f"    num_steps: {STEPS}\n    timestep_input: timestep\n"
        f"    guidance_scale: {CFG}\n    cfg_conditioning_input: encoder_hidden_states\n"
        "    timesteps:\n" + ts_yaml +
        "    scheduler_config:\n      kind: dpmpp_2m\n      num_train_timesteps: 1000\n"
        "      beta_start: 0.00085\n      beta_end: 0.012\n      beta_schedule: scaled_linear\n"
        "  phases:\n    text_encoder:\n      run_on: prompt_only\n    vae:\n      run_on: final_only\n"
    )

    # --- runtime inputs ---
    ids1_p.numpy().astype("<i8").tofile(pdir / "ids1.i64")
    ids2_p.numpy().astype("<i8").tofile(pdir / "ids2.i64")
    latent0.numpy().astype("<f4").tofile(pdir / "sample.f32")
    time_ids.numpy().astype("<f4").tofile(pdir / "time_ids.f32")
    enc_neg.detach().numpy().astype("<f4").tofile(pdir / "ehs_uncond.f32")
    pooled_neg.detach().numpy().astype("<f4").tofile(pdir / "pooled_uncond.f32")
    seq1, seq2 = ids1_p.shape[1], ids2_p.shape[1]
    es, ed = enc_pos.shape[1], enc_pos.shape[2]
    pd = pooled_pos.shape[1]

    env = dict(os.environ)
    env["DYLD_LIBRARY_PATH"] = ort_lib_dir() + ":" + env.get("DYLD_LIBRARY_PATH", "")
    out_path = pdir / "image.f32"
    print("running SDXL pipeline through onnx-genai ...", flush=True)
    subprocess.run(
        [
            str(RUNNER), str(pdir), "vae.image", str(out_path),
            f"text_encoder.input_ids:i64:1,{seq1}:{pdir / 'ids1.i64'}",
            f"text_encoder.input_ids_2:i64:1,{seq2}:{pdir / 'ids2.i64'}",
            f"denoiser.sample:1,{ch},{sz},{sz}:{pdir / 'sample.f32'}",
            f"denoiser.time_ids:1,6:{pdir / 'time_ids.f32'}",
            f"denoiser.encoder_hidden_states.uncond:1,{es},{ed}:{pdir / 'ehs_uncond.f32'}",
            f"denoiser.text_embeds.uncond:1,{pd}:{pdir / 'pooled_uncond.f32'}",
        ],
        env=env, check=True,
    )
    og = np.fromfile(out_path, dtype="<f4").reshape(img_ref.shape)

    diff = np.abs(og - img_ref)
    print("\n=== onnx-genai SDXL (multi-input CFG + time_ids) vs diffusers ===")
    print(f"  max|diff|  = {diff.max():.3e}")
    print(f"  mean|diff| = {diff.mean():.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
