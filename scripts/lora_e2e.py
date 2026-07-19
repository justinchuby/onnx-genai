#!/usr/bin/env python3
"""Validate LoRA fuse-before-export: onnx-genai fused UNet vs diffusers-fused.

A ComfyUI LoraLoader is handled by *fusing* the LoRA into the base model before
ONNX export (no runtime LoRA support needed). This creates a tiny random LoRA with
peft, exports the base UNet fused with it via checkpoint_export, and checks the
exported denoiser (a) MATCHES a diffusers pipeline with the same LoRA fused, and
(b) DIFFERS from the no-LoRA export (so the LoRA actually took effect).

Run (conda `onnx`):  conda run -n onnx python scripts/lora_e2e.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / "target" / "lora-e2e"
MODEL = "OFA-Sys/small-stable-diffusion-v0"


def make_tiny_lora(save_path: Path) -> None:
    from diffusers import UNet2DConditionModel
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import save_file

    unet = UNet2DConditionModel.from_pretrained(MODEL, subfolder="unet")
    unet.add_adapter(LoraConfig(
        r=4, lora_alpha=4, init_lora_weights=False,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    ))
    # init_lora_weights=False gives random (non-zero) A and B, so the LoRA has effect.
    sd = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    sd = {f"unet.{k}": v for k, v in sd.items()}
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(save_path))


def unet_forward_onnx(path: Path, sample, timestep, cond) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    feed = {
        names[0]: sample.numpy().astype("<f4"),
        names[1]: np.array([int(timestep)], dtype=np.int64),
        names[2]: cond.numpy().astype("<f4"),
    }
    return sess.run(None, feed)[0]


def main() -> int:
    from mobius.integrations.onnx_genai.checkpoint_export import export_checkpoint

    WORK.mkdir(parents=True, exist_ok=True)
    lora = WORK / "tiny.safetensors"
    print("creating tiny LoRA ...", flush=True)
    make_tiny_lora(lora)

    print("exporting base (no LoRA) and fused (LoRA) denoisers ...", flush=True)
    base = export_checkpoint(MODEL, str(WORK / "base"), height=64, width=64, components=("denoiser",))
    fused = export_checkpoint(
        MODEL, str(WORK / "fused"), height=64, width=64, components=("denoiser",),
        loras=[(str(lora), 1.0)],
    )

    # diffusers reference: a fresh pipe with the LoRA fused.
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(MODEL, safety_checker=None)
    pipe.load_lora_weights(str(lora))
    pipe.fuse_lora(lora_scale=1.0)
    ref_unet = pipe.unet.eval()

    torch.manual_seed(0)
    sample = torch.randn(1, 4, 8, 8)
    cond = torch.randn(1, 77, 768)
    timestep = 500
    with torch.no_grad():
        ref = ref_unet(sample, timestep, encoder_hidden_states=cond).sample.numpy()

    og_fused = unet_forward_onnx(WORK / "fused" / fused.denoiser_filename, sample, timestep, cond)
    og_base = unet_forward_onnx(WORK / "base" / base.denoiser_filename, sample, timestep, cond)

    match = np.abs(og_fused - ref)
    lora_effect = np.abs(og_fused - og_base)
    print("\n=== LoRA fuse-before-export ===")
    print(f"  fused ONNX vs diffusers-fused : max|diff|={match.max():.3e} mean={match.mean():.3e}")
    print(f"  fused vs base (LoRA effect)   : max|diff|={lora_effect.max():.3e} mean={lora_effect.mean():.3e}")
    assert match.max() < 1e-2, f"fused export does not match diffusers: {match.max()}"
    assert lora_effect.max() > 1e-3, f"LoRA had no effect (fuse was a no-op): {lora_effect.max()}"
    print("\nOK: fused export matches diffusers AND differs from base (LoRA applied)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
