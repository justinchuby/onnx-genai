# Importing a ComfyUI workflow

ComfyUI is a node-graph UI for diffusion. Its *"Save (API Format)"* export is a
flat JSON map:

```json
{"3": {"class_type": "KSampler", "inputs": {"steps": 20, "model": ["4", 0]}}}
```

where a value of the form `[src_id, slot]` links to another node's output.

`onnx-genai-comfyui-config` reads that document and lowers it into the canonical
`pipeline.workflow` inference metadata this runtime executes. This page is the
whole design: what the importer produces, what it refuses, and why the direction
only goes one way.

## ComfyUI is an import source, exactly like `genai_config.json`

The runtime has one source of execution truth: `pipeline.workflow`. A ComfyUI
document is *input to a conversion*, never an input to execution.

* Conversion happens once, ahead of time or at package-build time.
* The emitted `inference_metadata.yaml` is an ordinary workflow package. Nothing
  distinguishes it from one Mobius exported natively.
* No runtime code path reads the ComfyUI document, dispatches on a `class_type`,
  or asks "was this imported?". There is no ComfyUI execution shim.
* There is **no export direction**. A reverse synthesizer would have to
  approximate facts the canonical contract states precisely — port contracts,
  batch layouts, solver bindings — and the approximation would be indis­tinguish­able
  from truth once written. `onnx-genai-genai-config` declines the same thing for
  the same reason.

## What the importer emits

The canonical IR, with the same components and contracts the checked Mobius
diffusion packages carry (`tests/fixtures/onnx_genai_workflows/diffusion`,
`.../diffusion_guided`):

```
loop
├── setup
│   ├── diffusion_schedule   ()                        -> schedule            [sigmas, steps+1]
│   ├── diffusion_timesteps  ()                        -> schedule            [timesteps, steps]
│   ├── latent_row_shape     ()                        -> shape
│   ├── latent_noise         (seed, offset, row_shape) -> noise, next_offset  onnx-genai.counter-rng@1
│   ├── text_encoder         (input_ids)               -> encoder_hidden_states
│   ├── text_encoder         (negative_input_ids)      -> encoder_hidden_states     (guided only)
│   └── continue_predicate   (done)                    -> continue
├── body
│   ├── step_offset          (iteration, offset)       -> step                      (partial denoise only)
│   ├── schedule_lookup      (schedule, step)          -> timestep
│   ├── model_input          (sample, step, schedule)  -> model_input
│   ├── controlnet           (sample, timestep, encoder_hidden_states, hint, conditioning_scale)
│   │                                                  -> control                   onnx-genai.controlnet-residual@1
│   ├── denoiser             (sample, timestep, encoder_hidden_states[, text_embeds, time_ids][, control])
│   │                                                  -> noise_pred                (twice when guided)
│   ├── guidance_combine     (unconditional, conditional, scale) -> estimate        onnx-genai.guidance-combine@1
│   ├── solver_step          (sample, estimate, step, schedule[, history][, noise])
│   │                                                  -> next_state[, next_history] onnx-genai.solver-step@1
│   ├── masked_blend         (current, reference, noise, mask, step, schedule)
│   │                                                  -> blended                    onnx-genai.masked-blend@1
│   └── continue_predicate   (done)                    -> continue
├── vae_decoder              (latent)                  -> image
└── emit image, emit latent
```

Every one of those is a generic workflow component. There is no diffusion-shaped
step kind, no ComfyUI-shaped contract, and no new schema type: the importer added
nothing to `schema/inference_metadata.schema.json`.

### The run parameters become typed inputs, not constants

| ComfyUI | canonical |
| --- | --- |
| `KSampler.seed` | `request.seed` (runtime `seed` role), default from the graph |
| `KSampler.steps` | `request.max_iterations` (runtime role), default from the graph |
| `KSampler.cfg` | `request.guidance_scale`, default from the graph |
| `KSampler.sampler_name` | `solver_step.contract.parameters.solver` |
| `KSampler.scheduler` | `solver_step.contract.parameters.spacing` |
| `KSampler.denoise` | `package.start_step` plus the reduced `max_iterations` |
| `ControlNetApply.strength` | `request.control_strength` |
| `LoraLoader.lora_name` | `adapters.artifacts` selection through `request.adapter_*` |

A value the graph fixed becomes a *default*, not a folded constant, so a caller
can still override it. That is the difference between importing the workflow and
freezing it.

### Denoise strength

`start_step = steps - round_ties_even(steps * denoise)`, matching diffusers
`get_timesteps` and ComfyUI's own slicing. The emitted loop then runs
`end_step - start_step` iterations, and a `step_offset` component turns the loop
induction value into the schedule index, so step 0 of the loop is step
`start_step` of the schedule. `KSamplerAdvanced` supplies `start_at_step` and
`end_at_step` directly.

### Inpainting

The mask is a typed latent-space input, and it gates the solver's output on
*every* step through `onnx-genai.masked-blend@1`, which renoises the encoded
original to the step's sigma before blending. A mask applied once at the end
would be a different computation, so the importer does not emit one.

## Fail-closed conversion

Conversion walks backwards from the workflow's single image sink and must
understand every node that can reach it. The rules:

1. **Exactly one image sink.** Two sinks are ambiguous topology, not a hint.
2. **Every node on the output path must be recognized.** An unknown class is an
   error naming the node id, the class, and the remedy.
3. **Nodes that cannot reach the sink are reported and ignored.** That is the
   only sound reason to skip a node, and the report lists each one.
4. **Nothing that changes the image is dropped quietly.** A dangling link, a
   sampler with no canonical solver, a sigma spacing with no canonical schedule,
   a chained or step-windowed ControlNet, merged/region/step-scoped conditioning,
   a truncated CLIP, a patched denoiser (FreeU, RescaleCFG), a preprocessed hint
   image, a latent resample, and a LoRA the package does not declare are all
   errors.

Every refusal names the node and how to fix it:

```
ComfyUI node 3 (KSampler) cannot be represented: sampler "uni_pc_bh2" has no
canonical solver contract. How to fix: select one of: ddim, dpmpp_2m, euler,
euler_ancestral. A solver the runtime cannot reproduce would change every step
of the trajectory, so it is refused rather than approximated
```

### Coverage

| Capability | Status |
| --- | --- |
| txt2img SD 1.x / SD 2.x | supported |
| txt2img SDXL (`CLIPTextEncodeSDXL`) | supported: two encoders, `text_embeds`, `time_ids` |
| CFG | supported: two encoder passes plus `onnx-genai.guidance-combine@1` |
| Euler, Euler ancestral, DDIM, DPM++ 2M | supported |
| `normal`/`simple`, `ddim_uniform`, `karras`, `exponential`, `beta` | supported |
| img2img (`VAEEncode` + `denoise`) | supported |
| Inpainting (`VAEEncodeForInpaint`, `SetLatentNoiseMask`, `InpaintModelConditioning`) | supported |
| Single ControlNet (`ControlNetApply`, `ControlNetApplyAdvanced`) | supported |
| Multiple / step-windowed ControlNet | **refused** — no residual-merge contract exists |
| LoRA | supported *with* the package's own `adapters` contract; refused without it |
| Flow matching (`ModelSamplingSD3`/`Flux`/`AuraFlow`, `EmptySD3LatentImage`) | supported structurally |
| Qwen-Image-Edit, Flux guidance embedding, custom samplers/guiders | **refused** with a named diagnostic |
| Unknown / custom nodes on the output path | **refused** |

#### Why ControlNet chains are refused

The canonical denoiser ABI accepts one `control` residual input. Two ControlNets
produce two residual tensors, and merging them needs a contract this schema does
not define. Emitting the first and dropping the second would produce a package
that runs and is wrong, which is the failure mode this importer exists to avoid.

#### Why LoRA needs the package's adapter contract

A ComfyUI graph names a *file*. Canonical adapter metadata needs the artifact
identity, its base-model fingerprint, and the exact ONNX initializer each factor
binds to. None of that is in the workflow document. So the importer routes the
workflow's *selection* — which adapters, in what order, at what strengths —
through the `adapters` contract the package already declares, and refuses when
one is not supplied:

```bash
comfyui_to_metadata --adapters package/inference_metadata.yaml workflow.json
```

#### Why Qwen-Image-Edit is refused rather than approximated

Flow matching itself is representable: `ModelSamplingSD3`/`Flux`/`AuraFlow` set
`prediction: flow_velocity` on the solver contract, and `EmptySD3LatentImage`
selects the 16-channel latent. What is *not* representable is Qwen-Image editing,
which conditions the transformer on a vision-language encoder and a reference
image at once — a multi-encoder package shape, not the single text-conditioning
ABI the importer emits. That gets a named diagnostic rather than a generic
"unknown node".

## The package the emitted metadata expects

The importer names components and artifacts; it never creates ONNX graphs. The
default layout is the one Mobius's diffusion exporter writes:

```
package/
├── inference_metadata.yaml     <- written by the importer
├── text_encoder/model.onnx
├── text_encoder_2/model.onnx   (SDXL)
├── denoiser/model.onnx
├── vae_encoder/model.onnx      (img2img, inpainting)
├── vae_decoder/model.onnx
├── controlnet/model.onnx       (ControlNet)
└── policies/
    ├── diffusion_schedule.onnx
    ├── diffusion_timesteps.onnx
    ├── schedule_lookup.onnx
    ├── model_input.onnx
    ├── solver_step.onnx
    ├── continue_predicate.onnx
    ├── latent_row_shape.onnx
    ├── latent_noise.onnx
    ├── guidance_combine.onnx   (CFG)
    ├── history_initializer.onnx (multistep solvers)
    ├── add_noise.onnx          (img2img, inpainting)
    ├── step_offset.onnx        (partial denoise)
    └── masked_blend.onnx       (inpainting)
```

`ComponentLayout` overrides any of those paths, and `--textproto` selects
`*.onnx.textproto` artifacts. Checked-in fixtures use TextFormat so a package
under review is readable in a diff; the runtime accepts either encoding.

## Tools

```bash
# Convert only.
cargo run -p onnx-genai-comfyui-config --bin comfyui_to_metadata -- \
    --out package/inference_metadata.yaml workflow.json

# Convert, then execute on the generic workflow engine.
cargo run -p onnx-genai --bin run_comfyui -- \
    --package package --output image.ppm workflow.json
```

`run_comfyui` is a thin wrapper: convert, write, `Engine::from_pipeline_dir`,
run, save. It contains no diffusion logic, because every step of the loop is
executed by the generic workflow runtime from the emitted metadata.

## Tests

| What | Where |
| --- | --- |
| Graph walking, lowering, every fail-closed refusal | `crates/onnx-genai-comfyui-config/src/tests.rs` |
| Golden document, determinism, artifact resolution | `crates/onnx-genai-comfyui-config/tests/golden_workflow.rs` |
| End-to-end execution against an independent reference | `crates/onnx-genai-engine/tests/comfyui_workflow_e2e.rs` |
| Executable fixture package | `tests/fixtures/comfyui_workflows/txt2img_sd15/` |

Two properties are worth calling out:

* **Normalized semantic identity.** Renumbering every node and reordering the
  document changes nothing about what the workflow computes, and the converted
  metadata's `semantic_identity` is unchanged. Changing a run parameter changes
  it. Node ids are an import detail and never reach the canonical identity.
* **Deterministic regeneration.** The golden `inference_metadata.yaml` is
  regenerated by the converter and compared byte for byte, so it is a regression
  test rather than a snapshot.

Regenerate the fixture package with:

```bash
python scripts/build_comfyui_workflow_fixture.py
cargo run -p onnx-genai-comfyui-config --bin comfyui_to_metadata -- --textproto \
    --out tests/fixtures/comfyui_workflows/txt2img_sd15/inference_metadata.yaml \
    tests/fixtures/comfyui_workflows/txt2img_sd15/workflow.json
```
