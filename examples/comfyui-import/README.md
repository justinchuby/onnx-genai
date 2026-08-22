# ComfyUI import: a converted workflow, executed

This is the smallest complete demonstration that a ComfyUI workflow becomes an
ordinary onnx-genai workflow package.

```
workflow.json  ──(onnx-genai-comfyui-config)──▶  inference_metadata.yaml
                                                          │
                                        generic workflow engine executes it
                                                          │
                                                        image
```

## The fixture

[`tests/fixtures/comfyui_workflows/txt2img_sd15/`](../../tests/fixtures/comfyui_workflows/txt2img_sd15)
holds all four pieces:

| File | What it is |
| --- | --- |
| `workflow.json` | A ComfyUI API-format text-to-image graph: `EmptyLatentImage → KSampler → VAEDecode → SaveImage`, two `CLIPTextEncode` branches, `cfg 7.5`, `euler`, `karras`, `seed 20260821`, 4 steps |
| `inference_metadata.yaml` | What the importer lowered it into. Regenerated and compared byte for byte by a test |
| `*/model.onnx.textproto`, `policies/*.onnx.textproto` | Tiny ONNX components in the ABI the emitted metadata references, checked in as protobuf TextFormat so they are readable in a diff |
| `reference.json` | An independent double-precision simulation of what the emitted metadata says should happen |

Regenerate the components and the reference with:

```bash
python scripts/build_comfyui_workflow_fixture.py
```

Regenerate the metadata with:

```bash
cargo run -p onnx-genai-comfyui-config --bin comfyui_to_metadata -- --textproto \
  --out tests/fixtures/comfyui_workflows/txt2img_sd15/inference_metadata.yaml \
  tests/fixtures/comfyui_workflows/txt2img_sd15/workflow.json
```

## Convert and run

```bash
cargo run -p onnx-genai --bin run_comfyui -- \
  --package tests/fixtures/comfyui_workflows/txt2img_sd15 \
  --textproto --overwrite \
  --prompt-tokens 3,7,11,2 --negative-tokens 0,1,0,1 \
  --output fox.ppm \
  tests/fixtures/comfyui_workflows/txt2img_sd15/workflow.json
```

```text
converted .../workflow.json -> .../inference_metadata.yaml (4 steps, solver=euler, spacing=karras, guidance=7.5)
executed 4 steps in 0.001s (6718.14 steps/s); image shape [1, 3, 4, 4]
wrote fox.ppm
```

![The 4x4 image the converted workflow produced, upscaled](converted-run.png)

The fixture components are deliberately tiny, so the measured 6.1k–7.0k steps/s
across five release-mode runs is the workflow engine's per-step loop overhead,
not model throughput. The first three pixels are `(208, 213, 196)`,
`(198, 203, 208)`, `(209, 213, 197)`; the E2E test compares every value against
`reference.json` rather than against an eyeballed picture.

`run_comfyui` converts, writes, loads, and runs. It has no diffusion logic of its
own: the schedule, the two guided denoiser passes, the guidance combine, the
Euler solver step, and the VAE decode are all executed by the generic workflow
engine from `inference_metadata.yaml`.

## What the conversion produced

The ComfyUI run parameters become typed workflow inputs with the graph's values
as defaults, so a caller can still override them:

```yaml
request.seed:            { role: {kind: runtime, role: seed},            default: 20260821 }
request.max_iterations:  { role: {kind: runtime, role: max_iterations},  default: 4 }
request.guidance_scale:  { source: {kind: application, name: guidance_scale}, default: 7.5 }
```

The sampler choice reaches the solver contract rather than a runtime branch:

```yaml
solver_step:
  contract:
    id: onnx-genai.solver-step
    version: '1'
    parameters: { solver: euler, spacing: karras, prediction: epsilon }
```

Classifier-free guidance is two text-encoder invocations and one
`onnx-genai.guidance-combine` component — the same shape the natively exported
`tests/fixtures/onnx_genai_workflows/diffusion_guided` package uses.

Nothing ComfyUI-specific survives the conversion: the emitted document contains
no `class_type`, no `KSampler`, and no node ids. A test asserts exactly that.

## Refusals are the interesting part

Conversion is fail-closed. Anything that would change the produced image but has
no canonical representation is an error naming the node and the remedy:

```text
$ comfyui_to_metadata multi_controlnet.json
failed to convert multi_controlnet.json: ComfyUI workflow uses 2 chained ControlNets,
which the canonical workflow contract does not carry: each ControlNet contributes its
own residual tensor, and the canonical denoiser ABI accepts exactly one `control`
input. ... How to fix: import a workflow with a single ControlNetApply, or extend the
canonical component vocabulary with a residual-merge contract first
```

See [docs/genai/COMFYUI_IMPORT.md](../../docs/genai/COMFYUI_IMPORT.md) for the
full coverage table and the reasoning behind each refusal.
