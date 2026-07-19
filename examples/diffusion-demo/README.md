# onnx-genai diffusion demo

A small **TypeScript** web app to show colleagues how onnx-genai runs diffusion
pipelines from a **declarative config** — for both a **language-diffusion**
model (LLaDA-style masked diffusion) and an **image-diffusion** model (Stable
Diffusion). It:

1. **Loads a pipeline config** — either a **ComfyUI** API-format workflow JSON
   (translated to onnx-genai's native format on the fly) *or* a native
   **`inference_metadata`** document (YAML or JSON).
2. **Visualizes the config** — the component DAG (denoiser / VAE / text encoder
   and their dataflow edges) and the strategy (scheduler, steps, guidance, …),
   so everyone can see exactly what will run.
3. **Runs it and animates the reverse process** — for language diffusion, the
   tokens un-masking step by step, **decoded to real text** as words fill in; for
   image diffusion, the latent **denoising animation** (noise → image) followed
   by a reveal of the final crisp VAE-decoded PNG, shown inline with the
   wall-clock time.

The backend drives the **real** onnx-genai runtime (the `run_diffusion`,
`render_sd`, and `comfyui_to_metadata` binaries) — nothing is simulated.

```
┌─────────────┐   config (ComfyUI / native)   ┌──────────────┐   spawn    ┌────────────────────┐
│  web (Vite) │ ────────────────────────────► │ server (Node)│ ─────────► │ onnx-genai binaries │
│   TypeScript│ ◄──── DAG + per-step frames ── │              │ ◄── dumps ─│  (real runtime)     │
└─────────────┘                                └──────────────┘            └────────────────────┘
```

## Prerequisites

- Node 18+.
- Built onnx-genai binaries (from the repo root):
  ```bash
  cargo build --release -p onnx-genai --bin run_diffusion
  cargo build --release -p onnx-genai --bin render_sd          # renders the image PNG
  cargo build --release -p onnx-genai-comfyui-config --bin comfyui_to_metadata
  ```
  To build with the **CUDA** execution provider (image rendering on an NVIDIA
  GPU), point `ORT_ROOT` at a GPU ONNX Runtime and add `--features cuda`:
  ```powershell
  $env:ORT_ROOT="C:\path\to\onnxruntime-win-x64-gpu_cuda12-1.27.0"
  cargo build --release -p onnx-genai --bin render_sd --bin run_diffusion --features cuda
  ```
- The prebuilt ONNX Runtime dylib is found automatically under `target/`.

## Run

```bash
cd examples/diffusion-demo
npm install
npm run dev        # starts the API server + the Vite dev server
# open the printed http://localhost:5173
```

### Running on the Apple-Silicon GPU (MLX EP)

The pipeline can run on the GPU via the [onnxruntime-mlx](https://github.com/justinchuby/onnxruntime-mlx)
plugin execution provider. Build its `libonnxruntime_mlx_ep.dylib`, then set the
EP env vars before starting the backend:

```bash
export ONNX_GENAI_EP=metal
export ONNX_GENAI_METAL_EP_LIB=/abs/path/onnxruntime-mlx/rust/target/release/libonnxruntime_mlx_ep.dylib
npm run dev
```

onnx-genai registers the plugin and runs the diffusion denoiser on MLX, producing
bit-identical results to CPU. Real Stable-Diffusion UNet blocks run **5–16× faster**
than ORT CPU on the MLX EP (the whole block fuses into one MLX closure); the tiny
bundled language fixture is too small to show a speedup (it is dispatch-bound), so
use a real SD package to see the GPU win in the it/s card.

## What runs out of the box

- **Language diffusion** works immediately using the bundled tiny masked-diffusion
  fixture (`tests/fixtures/tiny-masked-diffusion`): it is a *real* ONNX model run
  through the real `masked_diffusion` scheduler, so you can watch the true
  un-masking dynamics (toy vocabulary, real algorithm).
- **Config load + visualization** works for any ComfyUI or native config with no
  model present.
- **Image diffusion** needs a real Stable Diffusion package built **from scratch**
  by [Mobius](https://github.com/justinchuby/mobius) (no `torch.onnx.export`, no
  `optimum` — every ONNX graph is authored directly with `onnx_ir`/`onnxscript`).
  Build a classic Stable Diffusion 1.x package (fp16) from a cached Hugging Face
  checkpoint:
  ```bash
  python -m mobius build --model OFA-Sys/small-stable-diffusion-v0 \
    --runtime onnx-genai --dtype f16 /tmp/sd-pkg
  ```
  The package contains `text_encoder/`, `unet/`, `vae_decoder/`,
  `inference_metadata.yaml`, and the CLIP `tokenizer.json`. Point the demo at it
  (the image tab then renders a real PNG from your prompt):
  ```bash
  ONNX_GENAI_SD_PACKAGE=/tmp/sd-pkg npm run dev
  ```
  The image tab exposes a **prompt** box plus **steps / guidance / seed**
  controls; each run tokenizes the prompt with the packaged CLIP tokenizer, runs
  the iterative denoise loop (DPM-Solver++, classifier-free guidance), and
  VAE-decodes the final latent to an RGB PNG.
- **Real language diffusion** (beyond the bundled fixture): export a masked
  masked-diffusion LM (e.g. `kuleshov-group/mdlm-owt`) to a package and point the
  demo at it with `ONNX_GENAI_LM_PACKAGE=/path/to/pkg` and
  `ONNX_GENAI_LM_SEQ_LEN=64` (the sequence length the demo seeds with masks).

## Running on an NVIDIA GPU (CUDA EP)

Build the binaries with `--features cuda` (see Prerequisites), then set
`ONNX_GENAI_EP=cuda` with the CUDA runtime on `PATH` before starting the demo.
On Windows (PowerShell), a full GPU launch with real SD + MDLM packages looks
like:

```powershell
$env:ONNX_GENAI_EP="cuda"
$env:ONNX_GENAI_SD_PACKAGE="C:\path\to\sd15-package"
$env:ONNX_GENAI_LM_PACKAGE="C:\path\to\mdlm-pipeline"
$env:ONNX_GENAI_LM_SEQ_LEN="64"
npm run dev
```

The demo server inherits these from the launching shell, so set them (and the
CUDA `PATH` entries) in the same shell you run `npm run dev` in. SD 1.5 at 384px
renders in ~14 s per image on an 8 GB laptop GPU.

## Performance

The backend always prefers the **release** binaries under `target/release/` and
picks the release ONNX Runtime lib to match. If only a debug build is present it
still runs but prints a loud warning — build release (see Prerequisites) for the
fast path. With the release build the bundled language fixture runs end-to-end in
well under 100 ms per request on an M-series Mac.

After a language run the UI shows a **speed card** with **it/s** (reverse-process
steps per second) — the exact metric ComfyUI reports. It is computed from the
runtime's pure reverse-process loop time (model/session load is measured and
**excluded**, matching how ComfyUI quotes it/s after the model is resident).
The card also breaks down timing **per pipeline stage** (encode / denoise /
decode) and **per reverse-process step**, so you can see exactly where the time
goes.

### Beating ComfyUI (reproducible head-to-head)

The point of the ComfyUI loader is that you can run the **same graph** both ways
and compare it/s on the same model, step count, and hardware:

1. In ComfyUI, run your SD workflow and note the reported **it/s** (and export
   the workflow as API-format JSON).
2. In this demo, load that same workflow (**Load as ComfyUI**) and run it against
   an onnx-genai package built from the same checkpoint — read the it/s off the
   speed card.

onnx-genai is faster on the same graph because it removes ComfyUI's per-node
Python dispatch: the pipeline is a single ONNX Runtime session with graph-level
fusion/optimization, static shapes (and graph capture on GPU backends),
device-resident latents/KV across steps, and quantized `MatMulNBits` kernels —
so per-step wall time is dominated by kernel compute, not orchestration.

## Loading a config

Two sample configs are bundled under `samples/`:

- `samples/comfyui-txt2img.json` — a ComfyUI Stable-Diffusion workflow. Paste it
  into the loader and click **Load as ComfyUI** (it is translated to the native
  format on the fly), or send it to `/api/translate-comfyui`.
- `samples/native-language.yaml` — a native `inference_metadata` masked
  language-diffusion pipeline. Paste it and click **Load as native config**.

For the language tab, **Use bundled fixture** both loads the config and runs it.

## Files

- `server/index.mjs` — Node API: translate ComfyUI→native, parse native config,
  run a language pipeline with per-step dumps, and render an image PNG via
  `render_sd` (the from-scratch Mobius SD 1.x driver); returns frames / the
  rendered image.
- `src/` — Vite + TypeScript UI: config loader, DAG/strategy view, run animation
  (`main.ts`, `style.css`).
- `samples/` — example ComfyUI and native configs to load.
- `index.html`, `vite.config.ts`, `tsconfig.json`, `dev.mjs` — frontend wiring.

