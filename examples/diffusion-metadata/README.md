# Diffusion pipeline metadata example

`stable-diffusion-v1-5-txt2img.inference_metadata.yaml` describes a full
**non-autoregressive** text-to-image latent-diffusion pipeline with the
`inference_metadata` pipeline contract. It is the hardest generality case for the
schema: there is no token loop. Instead:

```
text_encoder (CLIP)  --once-->  encoder_hidden_states
      │
      ▼
denoiser (UNet)  --N scheduler steps-->  loop-carried latent
      │   denoiser.out_sample --(Euler+Karras scheduler)--> denoiser.sample
      ▼
vae_decoder (AutoencoderKL.decode)  --once-->  RGB image
```

## What it demonstrates

- `strategy.kind: iterative` — a scheduler-driven denoise loop, not autoregression.
- `scheduler_config` — sampler (`euler`), noise schedule (`scaled_linear`, β
  0.00085→0.012, 1000 train steps), `prediction_type: epsilon`, Karras sigmas.
- Classifier-free guidance — `guidance_scale` + `cfg_conditioning_input`.
- Loop-carried latent — the `denoiser.out_sample -> denoiser.sample` self-edge.
- Phase gating — encoder `prompt_only`, denoiser `every_step`, VAE `final_only`.

## Provenance

Component / port / filename naming follows the diffusers ONNX export of
`runwayml/stable-diffusion-v1-5` (`optimum-cli export onnx` /
`ORTStableDiffusionPipeline`). The YAML is a hand-authored DESCRIPTION for the
metadata contract, not a model export — no weights are needed to validate it. The
engine that executes this shape end-to-end is covered by
`crates/onnx-genai-engine/tests/iterative_pipeline_e2e.rs` (tiny synthetic
fixtures) and driven for real by `crates/onnx-genai/src/bin/render_sd.rs`.

## Validation

The same document is committed as
`crates/onnx-genai-metadata/tests/fixtures/pipeline_diffusion_txt2img.yaml` and
validated by `pipeline_validation_accepts_full_diffusion_txt2img_contract` in
`crates/onnx-genai-metadata/tests/metadata_fixtures.rs`:

```
cargo test -p onnx-genai-metadata --test metadata_fixtures \
  pipeline_validation_accepts_full_diffusion_txt2img_contract
```

## Known contract gaps

See `.squad/decisions/inbox/deckard-diffusion-metadata.md` for the original
fit/gap analysis. The VAE **latent scaling factor** (0.18215) and **latent
geometry** ([4,64,64], ÷8) are still supplied out-of-band by `render_sd`.
Continuous diffusion schedulers now execute `epsilon`, `v_prediction`, and
`sample`/`x0`; `flow_matching` consumes velocity/vector-field output directly.
