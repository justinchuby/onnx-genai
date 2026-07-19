#!/usr/bin/env python3
"""Export a full Stable Diffusion pipeline into a self-contained onnx-genai
package the demo (and `run_comfyui`) can render end to end -- no Mobius required.

This is a convenience exporter for the diffusion demo. It exports the three
components with the exact ports the onnx-genai iterative pipeline expects:

    text_encoder.onnx : input_ids [1,77] i64      -> last_hidden_state
    denoiser.onnx     : sample, timestep,          -> noise_pred   (UNet)
                        encoder_hidden_states
    vae.onnx          : latent [1,4,H/8,W/8]       -> image        (1/scaling baked in)

plus `inference_metadata.yaml`, `run.json`, a CLIP `tokenizer.json`, and a
ComfyUI API-format `workflow.json`. Point the demo at the output directory:

    ONNX_GENAI_SD_PACKAGE=<output-dir> npm run dev

Spatial dimensions are exported as dynamic axes, so you can change the render
resolution just by editing `workflow.json` (width/height) without re-exporting.

Usage:
    python export_sd_package.py --output ./sd15-package \
        --model stable-diffusion-v1-5/stable-diffusion-v1-5 --size 384 --steps 25

Requires: torch, diffusers, transformers (pip install torch diffusers transformers).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def export_component(module, example_inputs, path, input_names, output_names,
                     dynamic_axes=None):
    torch.onnx.export(
        module,
        example_inputs,
        str(path),
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        dynamo=False,
        dynamic_axes=dynamic_axes,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True,
                        help="output package directory")
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5",
                        help="diffusers model id or local directory")
    parser.add_argument("--prompt",
                        default="a photograph of an astronaut riding a horse, highly detailed")
    parser.add_argument("--negative-prompt",
                        default="blurry, low quality, distorted, deformed")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--size", type=int, default=384,
                        help="render resolution; 384 fits an 8GB GPU in fp32, 512 needs more VRAM")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizerFast

    package_dir = arguments.output
    package_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {arguments.model} ...", flush=True)
    unet = UNet2DConditionModel.from_pretrained(arguments.model, subfolder="unet").eval()
    vae = AutoencoderKL.from_pretrained(arguments.model, subfolder="vae").eval()
    text_encoder = CLIPTextModel.from_pretrained(arguments.model, subfolder="text_encoder").eval()
    tokenizer = CLIPTokenizerFast.from_pretrained(arguments.model, subfolder="tokenizer")

    scaling_factor = float(getattr(vae.config, "scaling_factor", 0.18215))
    latent_channels = unet.config.in_channels
    latent_size = arguments.size // 8
    context_length = tokenizer.model_max_length
    hidden_dim = text_encoder.config.hidden_size

    scheduler = DDIMScheduler(
        num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear", prediction_type="epsilon",
        set_alpha_to_one=True, steps_offset=0, clip_sample=False,
    )
    scheduler.set_timesteps(arguments.steps)
    timesteps = [int(t) for t in scheduler.timesteps]

    example_ids = torch.zeros(1, context_length, dtype=torch.long)
    example_latent = torch.randn(1, latent_channels, latent_size, latent_size)
    example_hidden = torch.randn(1, context_length, hidden_dim)

    class TextEncoderWrapper(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, input_ids):
            return self.encoder(input_ids)[0]

    class UnetWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, sample, timestep, encoder_hidden_states):
            return self.model(sample, timestep,
                              encoder_hidden_states=encoder_hidden_states).sample

    class VaeDecoderWrapper(torch.nn.Module):
        def __init__(self, autoencoder, factor):
            super().__init__()
            self.autoencoder = autoencoder
            self.factor = factor

        def forward(self, latent):
            return self.autoencoder.decode(latent / self.factor).sample

    print("exporting text_encoder ...", flush=True)
    export_component(TextEncoderWrapper(text_encoder), (example_ids,),
                     package_dir / "text_encoder.onnx",
                     ["input_ids"], ["last_hidden_state"])

    print("exporting denoiser (UNet) ...", flush=True)
    export_component(
        UnetWrapper(unet),
        (example_latent, torch.tensor([timesteps[0]], dtype=torch.long), example_hidden),
        package_dir / "denoiser.onnx",
        ["sample", "timestep", "encoder_hidden_states"], ["noise_pred"],
        dynamic_axes={"sample": {2: "height", 3: "width"},
                      "noise_pred": {2: "height", 3: "width"}},
    )

    print("exporting vae decoder ...", flush=True)
    export_component(
        VaeDecoderWrapper(vae, scaling_factor), (example_latent,),
        package_dir / "vae.onnx", ["latent"], ["image"],
        dynamic_axes={"latent": {2: "latent_height", 3: "latent_width"},
                      "image": {2: "image_height", 3: "image_width"}},
    )

    print("writing tokenizer.json ...", flush=True)
    tokenizer.backend_tokenizer.save(str(package_dir / "tokenizer.json"))

    timesteps_yaml = "".join(f"      - {t}.0\n" for t in timesteps)
    (package_dir / "inference_metadata.yaml").write_text(
        "pipeline:\n"
        "  models:\n"
        "    text_encoder:\n      filename: text_encoder.onnx\n      type: text_encoder\n"
        "    denoiser:\n      filename: denoiser.onnx\n      type: denoiser\n"
        "    vae:\n      filename: vae.onnx\n      type: vae\n"
        "  dataflow:\n"
        "    - from: text_encoder.last_hidden_state\n"
        "      to: denoiser.encoder_hidden_states\n      dtype: fp32\n"
        "    - from: denoiser.noise_pred\n      to: denoiser.sample\n      dtype: fp32\n"
        "    - from: denoiser.sample\n      to: vae.latent\n      dtype: fp32\n"
        "  strategy:\n"
        "    kind: iterative\n    denoiser: denoiser\n"
        f"    num_steps: {arguments.steps}\n    timestep_input: timestep\n"
        f"    guidance_scale: {arguments.guidance}\n"
        "    cfg_conditioning_input: encoder_hidden_states\n"
        "    timesteps:\n" + timesteps_yaml +
        "    scheduler_config:\n      kind: ddim\n      num_train_timesteps: 1000\n"
        "      beta_start: 0.00085\n      beta_end: 0.012\n      beta_schedule: scaled_linear\n"
        "      prediction_type: epsilon\n"
        "  phases:\n"
        "    text_encoder:\n      run_on: prompt_only\n"
        "    vae:\n      run_on: final_only\n"
    )

    (package_dir / "run.json").write_text(json.dumps({
        "latent_channels": latent_channels,
        "height": arguments.size,
        "width": arguments.size,
        "scaling_factor": scaling_factor,
    }, indent=2))

    workflow = {
        "3": {"class_type": "KSampler", "inputs": {
            "seed": arguments.seed, "steps": arguments.steps, "cfg": arguments.guidance,
            "sampler_name": "ddim", "scheduler": "normal", "denoise": 1.0,
            "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0],
            "latent_image": ["5", 0]}},
        "4": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "sd15.safetensors"}},
        "5": {"class_type": "EmptyLatentImage",
              "inputs": {"width": arguments.size, "height": arguments.size, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"text": arguments.prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode",
              "inputs": {"text": arguments.negative_prompt, "clip": ["4", 1]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0]}},
    }
    (package_dir / "workflow.json").write_text(json.dumps(workflow, indent=2))

    print(f"\nPackage written to {package_dir}")
    print(f"  model={arguments.model} size={arguments.size}px "
          f"latent={latent_channels}x{latent_size}x{latent_size} "
          f"context={context_length}x{hidden_dim} steps={arguments.steps} "
          f"guidance={arguments.guidance} scaling_factor={scaling_factor}")
    print(f"\nRun the demo with:\n  ONNX_GENAI_SD_PACKAGE={package_dir} npm run dev")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
