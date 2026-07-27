### 2026-07-27: Route native ComfyUI adapter inputs as denoiser inputs
**By:** Dallas
**What:** `run_comfyui` converts ControlNet hints to batched RGB CHW `[0,1]` tensors and supplies ControlNet scales and LoRA strengths through package-defined denoiser inputs. Multiple ControlNets use adapter-named input suffixes.
**Why:** The fused ControlNet and baked LoRA graphs expose these values as ordinary denoiser inputs, so the generic pipeline request path needs no engine special-case. `DIFFUSION.md` specifies the single-ControlNet `controlnet_cond` port and named `lora_gate.{name}` ports, but does not define multi-ControlNet port naming or explicitly document the scale port; adapter-name suffixes preserve deterministic graph-to-input routing.
