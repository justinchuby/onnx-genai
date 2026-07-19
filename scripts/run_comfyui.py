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
    ap.add_argument("--control-image", help="Control image for a ControlNet workflow (resized to WxH).")
    ap.add_argument("--source-image", help="Source image for an inpainting workflow.")
    ap.add_argument("--mask-image", help="Mask (white=inpaint region) for an inpainting workflow.")
    ap.add_argument("--lora", action="append", metavar="NAME=PATH", help="Resolve a ComfyUI LoRA name to a path (fused).")
    ap.add_argument("--controlnet", action="append", metavar="NAME=PATH", help="Resolve a ComfyUI ControlNet name to a diffusers dir/file (fused).")
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
    def _pairs(entries):
        out = {}
        for e in entries or []:
            n, _, p = e.partition("=")
            if p:
                out[n] = p
        return out or None

    result = convert_comfyui_workflow(
        workflow, args.checkpoint, str(pdir),
        lora_paths=_pairs(getattr(args, "lora", None)),
        controlnet_paths=_pairs(getattr(args, "controlnet", None)),
    )
    wf = result.workflow
    meta = json.loads(json.dumps(__import__("yaml").safe_load(open(result.metadata_path))))
    sc = meta["pipeline"]["strategy"]["scheduler_config"]
    prompt = wf.prompt or ""
    negative = wf.negative_prompt or ""
    print(f"  prompt={prompt!r}  negative={negative!r}  {wf.steps} steps, cfg {wf.cfg}, "
          f"{wf.sampler_name} ({wf.scheduler_kind}{'/karras' if wf.scheduler_spacing == 'karras' else ''})")

    run_params = json.load(open(result.run_params_path))
    sdxl = bool(run_params.get("sdxl", False))
    latent_ch = run_params["latent_channels"]
    sz = wf.height // 8
    bs = int(run_params.get("batch_size", 1))
    # Batched generation is wired for the plain SD txt2img path; the variant
    # branches (SDXL/ControlNet/inpaint) below feed batch-1 constant tensors, so
    # fall back to a single image there.
    plain_sd = not sdxl and not run_params.get("controlnet") and not run_params.get("inpaint")
    if bs > 1 and not plain_sd:
        print(f"  (batch_size={bs} not yet driven for this variant; rendering 1 image)")
        bs = 1

    tokenizer = CLIPTokenizer.from_pretrained(args.checkpoint, subfolder="tokenizer")

    def tok(text: str, tk) -> np.ndarray:
        return tk(text, padding="max_length", max_length=tk.model_max_length,
                  truncation=True, return_tensors="np").input_ids.astype("<i8")

    ids_pos = tok(prompt, tokenizer)
    ids_neg = tok(negative, tokenizer)

    te = ort.InferenceSession(str(pdir / "text_encoder.onnx"), providers=["CPUExecutionProvider"])
    init_sigma = _diffusers_init_noise_sigma(wf.scheduler_kind, sc, wf.steps)
    rng = np.random.default_rng(wf.seed)
    latent0 = (rng.standard_normal((bs, latent_ch, sz, sz)).astype("<f4")) * init_sigma
    latent0.tofile(pdir / "sample.f32")
    # Repeat the (single) prompt across the batch.
    ids_pos_b = np.repeat(ids_pos, bs, axis=0) if bs > 1 else ids_pos
    ids_pos_b.astype("<i8").tofile(pdir / "ids.i64")

    inputs = [str(RUNNER), str(pdir), "vae.image", str(pdir / "image.f32")]

    if sdxl:
        # SDXL: two tokenizers + two conditioning inputs + time_ids. The pipeline
        # runs the dual encoder on the positive prompts; the negative-prompt uncond
        # (encoder_hidden_states + pooled text_embeds) is computed here.
        tok2 = CLIPTokenizer.from_pretrained(args.checkpoint, subfolder="tokenizer_2")
        ids2_pos = tok(prompt, tok2)
        ids2_neg = tok(negative, tok2)
        te_ins = [i.name for i in te.get_inputs()]
        ehs_neg, pooled_neg = te.run(None, {te_ins[0]: ids_neg, te_ins[1]: ids2_neg})
        ehs_neg = ehs_neg.astype("<f4")
        pooled_neg = pooled_neg.astype("<f4")
        time_ids = np.array([[wf.height, wf.width, 0, 0, wf.height, wf.width]], dtype="<f4")
        ids2_pos.tofile(pdir / "ids2.i64")
        ehs_neg.tofile(pdir / "ehs_uncond.f32")
        pooled_neg.tofile(pdir / "pooled_uncond.f32")
        time_ids.tofile(pdir / "time_ids.f32")
        inputs += [
            f"text_encoder.input_ids:i64:1,{ids_pos.shape[1]}:{pdir / 'ids.i64'}",
            f"text_encoder.input_ids_2:i64:1,{ids2_pos.shape[1]}:{pdir / 'ids2.i64'}",
            f"denoiser.sample:1,{latent_ch},{sz},{sz}:{pdir / 'sample.f32'}",
            f"denoiser.time_ids:1,6:{pdir / 'time_ids.f32'}",
            f"denoiser.encoder_hidden_states.uncond:1,{ehs_neg.shape[1]},{ehs_neg.shape[2]}:{pdir / 'ehs_uncond.f32'}",
            f"denoiser.text_embeds.uncond:1,{pooled_neg.shape[1]}:{pdir / 'pooled_uncond.f32'}",
        ]
    else:
        emb_neg = te.run(None, {te.get_inputs()[0].name: ids_neg})[0].astype("<f4")
        emb_neg_b = np.repeat(emb_neg, bs, axis=0) if bs > 1 else emb_neg
        emb_neg_b.tofile(pdir / "uncond.f32")
        inputs += [
            f"text_encoder.input_ids:i64:{bs},{ids_pos.shape[1]}:{pdir / 'ids.i64'}",
            f"denoiser.sample:{bs},{latent_ch},{sz},{sz}:{pdir / 'sample.f32'}",
            f"denoiser.encoder_hidden_states.uncond:{bs},{emb_neg.shape[1]},{emb_neg.shape[2]}:{pdir / 'uncond.f32'}",
        ]

    controlnet = bool(run_params.get("controlnet", False))
    inpaint = bool(run_params.get("inpaint", False))
    if inpaint:
        from PIL import Image

        ve = ort.InferenceSession(str(pdir / "vae_encoder.onnx"), providers=["CPUExecutionProvider"])
        if getattr(args, "source_image", None) and getattr(args, "mask_image", None):
            src = Image.open(args.source_image).convert("RGB").resize((wf.width, wf.height))
            src_arr = (np.asarray(src, dtype=np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[None]
            m = Image.open(args.mask_image).convert("L").resize((wf.width, wf.height))
            m_pix = (np.asarray(m, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)  # 1 = inpaint
        else:
            print("  (no --source-image/--mask-image; using zeros + full mask)")
            src_arr = np.zeros((1, 3, wf.height, wf.width), dtype=np.float32)
            m_pix = np.ones((wf.height, wf.width), dtype=np.float32)
        # masked image = source with the inpaint region zeroed; encode to a latent.
        masked_src = (src_arr * (1.0 - m_pix)[None, None]).astype("<f4")
        masked_latent = ve.run(None, {ve.get_inputs()[0].name: masked_src})[0].astype("<f4")
        # mask downsampled to latent resolution.
        mask_lat = np.asarray(
            Image.fromarray((m_pix * 255).astype("uint8")).resize((sz, sz), Image.NEAREST),
            dtype=np.float32,
        )[None, None] / 255.0
        mask_lat.astype("<f4").tofile(pdir / "mask.f32")
        masked_latent.tofile(pdir / "masked_latent.f32")
        inputs += [
            f"denoiser.mask:1,1,{sz},{sz}:{pdir / 'mask.f32'}",
            f"denoiser.masked_latent:1,{latent_ch},{sz},{sz}:{pdir / 'masked_latent.f32'}",
        ]
    if controlnet:
        cond_ch = int(run_params.get("conditioning_channels", 3))
        from PIL import Image

        if getattr(args, "control_image", None):
            ci = Image.open(args.control_image).convert("RGB").resize((wf.width, wf.height))
            arr = (np.asarray(ci, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]
        else:
            print("  (no --control-image; using a zero control map)")
            arr = np.zeros((1, cond_ch, wf.height, wf.width), dtype=np.float32)
        arr.astype("<f4").tofile(pdir / "control.f32")
        inputs.append(
            f"denoiser.controlnet_cond:1,{cond_ch},{wf.height},{wf.width}:{pdir / 'control.f32'}"
        )

    env = dict(os.environ)
    env["DYLD_LIBRARY_PATH"] = ort_lib_dir() + ":" + env.get("DYLD_LIBRARY_PATH", "")
    tag = "SDXL" if sdxl else "SD"
    if controlnet:
        tag += "+ControlNet"
    if inpaint:
        tag += "+Inpaint"
    print(f"rendering through onnx-genai ({tag}) ...", flush=True)
    subprocess.run(inputs, env=env, check=True)
    flat = np.fromfile(pdir / "image.f32", dtype="<f4")
    hw = int(round((flat.size / (3 * bs)) ** 0.5))
    batch = flat.reshape(bs, 3, hw, hw)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    img = batch[0]
    if bs == 1:
        save_png(img, Path(args.output))
        print(f"saved: {args.output}")
    else:
        out = Path(args.output)
        stem, suffix = out.stem, (out.suffix or ".png")
        for i in range(bs):
            p = out.with_name(f"{stem}_{i}{suffix}")
            save_png(batch[i], p)
            print(f"saved: {p}")

    if args.compare and not sdxl:
        if bs == 1:
            _compare_diffusers(args.checkpoint, wf, sc, latent0, ids_pos, ids_neg, img)
        else:
            print("  (--compare not supported for batch_size>1)")
    elif args.compare and sdxl:
        print("  (--compare not supported for SDXL here; see scripts/sdxl_e2e.py)")
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
