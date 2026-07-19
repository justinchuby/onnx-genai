# Diffusion & ComfyUI Support

onnx-genai runs **diffusion pipelines** (Stable Diffusion–style text-to-image, img2img, and
discrete language diffusion) through the same **standard-driven, declarative** mechanism it uses
for text generation: an `inference_metadata` document describes the components and dataflow, and
the engine drives the loop. There is **no per-model Python dispatch** — a Mobius-built diffusion
package, or a translated **ComfyUI** workflow, is directly runnable.

This document is the reference for that capability. The generation (LLM) side is covered by
[DESIGN.md](./DESIGN.md) and [PIPELINE.md](./PIPELINE.md).

---

## 1. Architecture

A diffusion model is a **composite iterative pipeline**:

```
text_encoder (prompt_only)   input_ids -> last_hidden_state
    -> denoiser (iterative)  sample / timestep / encoder_hidden_states -> noise_pred
    -> vae (final_only)      latent -> image
```

- **`kind: iterative`** strategy. The denoiser is re-invoked once per step. Its output
  (`noise_pred`) is fed back to its `sample` input via a **loop-carried self-edge** in `dataflow`;
  a scheduler combines the two into the next latent.
- **Prompt-phase** components (`run_on: prompt_only`, e.g. the text encoder) run once before the
  loop; **final-phase** components (`run_on: final_only`, e.g. the VAE) run once after.
- **Per-step timestep injection** feeds `timesteps[step]` into the denoiser's `timestep_input`
  (honoring its port dtype — int64 or float32, see §4).
- **Classifier-free guidance** (`guidance_scale` + `cfg_conditioning_input`): a conditional and an
  unconditional denoiser pass per step, combined as `uncond + scale·(cond − uncond)`. The
  unconditional pass uses a caller-supplied embedding on `{denoiser}.{port}.uncond` (the negative
  or empty prompt), so a real negative prompt is honored (§6).

The core is `crates/onnx-genai-engine/src/pipeline.rs` (`PipelineEngine::run_pipeline`,
`run_iterative`). The metadata schema is `crates/onnx-genai-metadata/src/schema.rs`
(`PipelineStrategy`, `SchedulerSpec`), published as `schema/inference_metadata.schema.json`.

---

## 2. Supported samplers / schedulers

All are validated **against diffusers** (deterministic ones to floating-point precision; the
stochastic one by feeding a matched noise sequence). Numbers are the end-to-end image difference
on `OFA-Sys/small-stable-diffusion-v0` unless noted.

| onnx-genai `kind`   | diffusers scheduler                | ComfyUI `sampler_name` | validation (max\|Δ\| / mean) | script |
|---------------------|------------------------------------|------------------------|------------------------------|--------|
| `ddim`              | `DDIMScheduler`                    | `ddim`                 | 6.9e-4 latent / img ~1.8e-3  | `diffusion_image.py`, `diffusion_e2e.py` |
| `euler`             | `EulerDiscreteScheduler`           | `euler`                | 7.9e-3 / 2.5e-5              | `comfyui_e2e.py`, `euler_parity.py` |
| `dpmpp_2m`          | `DPMSolverMultistepScheduler`      | `dpmpp_2m`             | 1.8e-3 / 1.7e-5             | `dpmpp_e2e.py`, `dpmpp_parity.py` |
| `dpmpp_2m` + Karras | `…(use_karras_sigmas=True)`        | `dpmpp_2m` / `karras`  | 5.9e-3 / 7.4e-5             | `dpmpp_e2e.py` (`ONNX_GENAI_KARRAS=1`), `karras_parity.py` |
| `euler_ancestral`   | `EulerAncestralDiscreteScheduler`  | `euler_ancestral`      | 4.7e-2 / 1.3e-3 (stochastic) | `euler_a_e2e.py` |
| `masked_diffusion`  | — (discrete language diffusion)    | —                      | synthetic fixture           | §7 |

**Sigma spacing.** `use_karras_sigmas` (Karras rho=7) or `use_exponential_sigmas` on `SchedulerSpec` replace the default linspace schedule for `euler`/`dpmpp_2m`; `DPM++ 2M Karras` is the most popular real-world combo. Both match diffusers (`scripts/karras_parity.py`).

**Timestep dtype.** Euler and any Karras schedule produce *fractional* timesteps, so the exported
denoiser must take a **float32** timestep (int64 would truncate and corrupt the time embedding).
DDIM and linspace DPM++ timesteps are integers (int64). The converter selects this automatically.

**Scheduler conventions** (why they differ):

- **DDIM** — no input scaling; `init_noise_sigma = 1.0`; integer timesteps.
- **Euler / Euler-a** — scale the model input by `1/√(σ²+1)`; seed scaled by `init_noise_sigma =
  σ[0]`; fractional timesteps.
- **DPM++ 2M** — no input scaling; `init_noise_sigma = 1.0`; multistep (keeps the previous data
  prediction, reset each loop); integer timesteps.
- **Euler Ancestral** — like Euler but injects `noise·σ_up` each step (stochastic, §5).

### 2.1 Extensible schedulers

`Scheduler` is a **public trait** (`onnx_genai_engine`); users register custom kinds with a
`SchedulerRegistry` and load via `Engine::from_pipeline_dir_with_schedulers`. The built-ins are
just implementations. Trait surface:

```rust
pub trait Scheduler: Send + Sync + Debug {
    fn step(&self, step, num_steps, sample: &Value, model_output: &Value) -> Result<Value>;
    fn reset(&self) {}                       // clear per-loop state (multistep)
    fn needs_noise(&self) -> bool { false }  // stochastic samplers opt in
    fn step_with_noise(&self, …, noise: Option<&Value>) -> Result<Value> { self.step(…) }
    fn scale_input(&self, step, num_steps, sample) -> Result<Option<Value>> { Ok(None) }
}
```

---

## 3. ComfyUI workflow support

A ComfyUI **API-format** workflow (Save → "Save (API Format)") is a flat node graph. The core
text-to-image graph is KSampler-centric and maps directly onto the composite pipeline. The
translator (`mobius.integrations.onnx_genai.comfyui`) walks the graph links to recover the full
run:

| ComfyUI                    | onnx-genai                                   |
|----------------------------|----------------------------------------------|
| `KSampler.steps`           | `num_steps`                                  |
| `KSampler.cfg`             | `guidance_scale` (1.0 disables CFG)          |
| `KSampler.sampler_name`    | scheduler `kind`                             |
| `KSampler.scheduler`       | `use_karras_sigmas` when `karras`            |
| `KSampler.denoise` (<1.0)  | `start_step` (img2img, §6)                   |
| `KSampler.seed`            | seed                                         |
| `CLIPTextEncode` (+/−)     | text encoder + CFG cond/uncond               |
| `EmptyLatentImage`         | latent width/height                          |
| `CheckpointLoaderSimple`   | checkpoint (traced through LoraLoader, etc.) |
| `VAEDecode`                | vae (final phase)                            |

Unsupported samplers or non-txt2img graphs raise a clear error rather than silently running the
wrong dynamics.

### 3.1 safetensors → ONNX

ComfyUI references a checkpoint by filename; onnx-genai runs ONNX. `checkpoint_export.py` bridges
this via diffusers: `from_single_file` for ComfyUI's original-SD single `.safetensors`, or
`from_pretrained` for a diffusers directory / HF id. It exports three components with the exact
ports the pipeline expects:

```
text_encoder.onnx : input_ids            -> last_hidden_state
denoiser.onnx     : sample, timestep,      -> noise_pred
                    encoder_hidden_states
vae.onnx          : latent               -> image   (1/scaling_factor baked in)
```

### 3.2 One-command conversion & rendering

```bash
# Convert a workflow + checkpoint into a runnable onnx-genai pipeline directory
# (this also emits a self-contained fast-format `tokenizer.json` into `out/`):
mobius convert-comfyui workflow.json --checkpoint <.safetensors | dir | HF-id> -o out/

# Render an image end-to-end with the native runner (honors the negative prompt).
# The runner consumes an already-exported ONNX package — no re-export or Python:
cargo build --release -p onnx-genai --bin run_comfyui
run_comfyui --workflow workflow.json --pipeline-dir out/ --output image.png
```

The native `run_comfyui` binary parses the workflow with `onnx-genai-comfyui-config`, tokenizes the
positive and negative prompts natively with the Hugging Face `tokenizers` crate (loading the
package's `tokenizer.json`), draws the seed latent — pre-scaled by the scheduler's
`init_noise_sigma`, queried from the engine via `PipelineEngine::diffusion_init_noise_sigma()` — plus
the per-step noise for ancestral samplers, runs the pipeline, and saves the PNG(s). For `batch_size >
1` it writes numbered files (`image_0.png`, `image_1.png`, …). It requires the model to already be
ONNX (`denoiser.onnx` / `text_encoder.onnx` / `vae.onnx` in the package).

`convert_comfyui_workflow` reconciles the ComfyUI sampler (kind/steps/cfg) with the checkpoint's
own noise schedule (betas / `num_train_timesteps`, which the ComfyUI JSON never carries), computes
the exact diffusers timesteps, and writes `inference_metadata.yaml` + `run.json`.

---

## 4. img2img (partial denoise)

A `KSampler.denoise` < 1.0 is img2img: encode a source image to a latent, noise it to an
intermediate step, and run only the tail of the loop. The `start_step` field on the iterative
strategy runs `start_step..num_steps` from the noised encoded-image seed, with
`start_step = num_steps − round(num_steps·denoise)` (matching diffusers `get_timesteps`).
`scripts/img2img_e2e.py` validates it against diffusers img2img (max|Δ| ~1.0e-2).

---

## 5. Stochastic (ancestral) sampling

Ancestral samplers inject fresh Gaussian noise each step, so they only match a reference when both
consume the *same* noise. onnx-genai exposes a reusable **per-step noise primitive**: a scheduler
returns `needs_noise() == true` and the loop slices noise from an external tensor
`{denoiser}.{port}.noise` shaped `[num_steps, *sample_shape]`. A driver draws the sequence with a
seeded generator and feeds it; the reference uses a generator of the same seed. See
`scripts/euler_a_e2e.py`.

---

## 6. Negative prompts

Real ComfyUI workflows carry a negative prompt (not empty). The CFG unconditional pass uses the
embedding supplied on `{denoiser}.encoder_hidden_states.uncond`; the native `run_comfyui` binary
computes it from the **negative** prompt via the exported text encoder (onnxruntime), so the negative
prompt is honored end-to-end.

---

## 7. Language diffusion

`masked_diffusion` drives **discrete** (text) diffusion: the loop-carried tensor is an int64 token
sequence, the denoiser emits `[B, S, V]` logits, and each step commits the highest-confidence
still-masked positions (argmax), unmasking progressively so all masked positions are filled by the
final step. Configured with `mask_token_id`.

---

## 8. Validation scripts (`scripts/`)

Numeric parity (formula vs diffusers): `euler_parity.py`, `dpmpp_parity.py`, `karras_parity.py`.
End-to-end image (full pipeline through onnx-genai vs diffusers): `diffusion_image.py`,
`diffusion_e2e.py`, `comfyui_e2e.py`, `dpmpp_e2e.py` (`ONNX_GENAI_KARRAS=1` for Karras),
`img2img_e2e.py`, `euler_a_e2e.py`. Run in the conda `onnx` env after
`cargo build --release -p onnx-genai --bin run_diffusion`.

> **Note:** each e2e run exports an SD checkpoint to ~319 external-data files (~2–3 GB) under
> `target/`. Clean `target/*-e2e` dirs to reclaim disk.

---

## 9. Limitations & roadmap

- **Samplers:** euler, euler_ancestral, ddim, dpmpp_2m (+Karras). Not yet: other DPM++ variants,
  exponential/beta sigma spacings, other SDE/ancestral samplers.
- **Models:** SD 1.x-style **and SDXL** run through the same declarative pipeline. **SDXL is
  validated end-to-end** (`scripts/sdxl_e2e.py`) — the two text encoders' penultimate hidden states
  are concatenated into `encoder_hidden_states`, the pooled `text_embeds` and the `time_ids` vector
  are routed as extra denoiser conditioning, and **multi-input CFG** guides both
  `encoder_hidden_states` and `text_embeds` while sharing `time_ids`. This needed **no
  SDXL-specific runtime code** — the composite pipeline (multiple dataflow edges + external
  constants + multi-input CFG) handles it. Matches diffusers to max|Δ|~4e-2 on tiny-random SDXL
  weights (the exact multi-input CFG path is unit-tested to 1e-5). The Mobius **export side is
  done** — `checkpoint_export` auto-detects SDXL and emits the dual-encoder + 5-input-UNet pipeline,
  and `mobius convert-comfyui` routes both conditioning edges automatically. The native `run_comfyui`
  binary currently drives the plain **SD txt2img** path (+ batched generation); rendering the SDXL /
  ControlNet / inpaint variants through the one-command runner is not yet ported to native Rust.
- **ControlNet** ✅ handled by a **combined ControlNet+UNet export** (like SDXL, no runtime change):
  the denoiser is a fused ControlNet+UNet taking an extra constant `controlnet_cond` image input
  (the ControlNet produces down/mid residuals injected into the UNet). The translator collects
  `ControlNetApply` (name + strength), `checkpoint_export(controlnet=...)` fuses it, and
  `mobius convert-comfyui --controlnet NAME=PATH` resolves it; `controlnet_cond` is an external
  denoiser input (like SDXL `time_ids`) shared across the CFG cond/uncond passes. Validated
  (`scripts/controlnet_e2e.py`): a fused export matches diffusers to 5.8e-6 and differs from base by
  0.45 (ControlNet takes effect). *Remaining:* native `run_comfyui` rendering of ControlNet / LoRA /
  SDXL ControlNet / inpaint workflows (conversion via `mobius convert-comfyui` already supports them).
- **img2img** is supported; inpainting (mask) is not.
