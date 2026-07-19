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
   tokens un-masking step by step; for image diffusion, the latent denoising to
   an image.

The backend drives the **real** onnx-genai runtime (the `run_diffusion` and
`comfyui_to_metadata` binaries) — nothing is simulated.

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
  cargo build --release -p onnx-genai-comfyui-config --bin comfyui_to_metadata
  ```
- The prebuilt ONNX Runtime dylib is found automatically under `target/`.

## Run

```bash
cd examples/diffusion-demo
npm install
npm run dev        # starts the API server + the Vite dev server
# open the printed http://localhost:5173
```

## What runs out of the box

- **Language diffusion** works immediately using the bundled tiny masked-diffusion
  fixture (`tests/fixtures/tiny-masked-diffusion`): it is a *real* ONNX model run
  through the real `masked_diffusion` scheduler, so you can watch the true
  un-masking dynamics (toy vocabulary, real algorithm).
- **Config load + visualization** works for any ComfyUI or native config with no
  model present.
- **Image diffusion** needs a real Stable Diffusion package built from scratch by
  Mobius. Point the demo at it with `ONNX_GENAI_SD_PACKAGE=/path/to/pkg`:
  ```bash
  # in a Mobius checkout (conda `onnx` env):
  python -c "from mobius import build_diffusers_pipeline; from mobius.integrations.onnx_genai import write_onnx_genai_config; \
             pkg = build_diffusers_pipeline('OFA-Sys/small-stable-diffusion-v0'); \
             pkg.save('/tmp/sd-pkg'); write_onnx_genai_config(pkg, '/tmp/sd-pkg', source='OFA-Sys/small-stable-diffusion-v0')"
  ONNX_GENAI_SD_PACKAGE=/tmp/sd-pkg npm run dev
  ```

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
  run a pipeline with per-step dumps, return frames.
- `src/` — Vite + TypeScript UI: config loader, DAG/strategy view, run animation
  (`main.ts`, `style.css`).
- `samples/` — example ComfyUI and native configs to load.
- `index.html`, `vite.config.ts`, `tsconfig.json`, `dev.mjs` — frontend wiring.

